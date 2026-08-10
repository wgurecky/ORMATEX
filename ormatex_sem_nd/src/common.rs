//! Shared finite-element contexts and geometry assembly infrastructure.

use faer::sparse::{SparseColMat, Triplet};
use ndelement::{
    ciarlet::LagrangeElementFamily,
    traits::{ElementFamily, FiniteElement},
    types::ReferenceCellType,
};
use ndfunctionspace::{traits::FunctionSpace, FunctionSpaceImpl};
use ndmesh::traits::{Entity, Geometry, GeometryMap, Mesh, Point, Topology};
use quadraturerules::{single_integral_quadrature, Domain, QuadratureRule};
use rlst::{rlst_dynamic_array, DynArray};

use crate::kernels::kernel_common::BoundaryIntegrator;
use crate::material::{CellMeta, FacetMeta, MaterialContext};

/// Cached per-cell-type quadrature, basis tabulation, and geometry data.
pub struct CellData {
    pub wts: Vec<f64>,
    pub npts: usize,
    pub ndofs: usize,
    pub table: DynArray<f64, 4>,
    pub reference_values: Vec<f64>,
    pub nodal_quadrature: Vec<usize>,
    pub jinv_cache: DynArray<f64, 4>,
    pub jdets_cache: Vec<f64>,
    pub physical_points_cache: Vec<f64>,
    #[allow(dead_code)]
    pub pts: DynArray<f64, 2>,
}

/// Per-cell assembly context with physical basis values and gradients.
pub struct LocalCtx<'a> {
    pub time: f64,
    pub cell: CellMeta,
    pub tdim: usize,
    pub gdim: usize,
    pub ncomp: usize,
    pub npts: usize,
    pub ndofs: usize,
    pub wts: &'a [f64],
    pub jdets: &'a [f64],
    pub points: &'a [f64],
    pub values: &'a [f64],
    pub grads: &'a [f64],
}

impl<'a> LocalCtx<'a> {
    pub fn point(&self, q: usize) -> &'a [f64] {
        &self.points[q * self.gdim..(q + 1) * self.gdim]
    }

    pub fn material_context<'b>(
        &'b self,
        state: Option<&'b CellState<'b>>,
        q: usize,
    ) -> MaterialContext<'b> {
        MaterialContext {
            time: self.time,
            point: self.point(q),
            cell: self.cell,
            state,
            q,
        }
    }

    pub fn test(&'a self, i: usize, comp: usize) -> ShapeFn<'a> {
        ShapeFn {
            npts: self.npts,
            ncomp: self.ncomp,
            gdim: self.gdim,
            values: self.values,
            grads: self.grads,
            i,
            comp,
        }
    }

    pub fn trial(&'a self, i: usize, comp: usize) -> ShapeFn<'a> {
        self.test(i, comp)
    }
}

/// Per-dof view over basis values and gradients.
pub struct ShapeFn<'a> {
    pub npts: usize,
    pub ncomp: usize,
    pub gdim: usize,
    pub values: &'a [f64],
    pub grads: &'a [f64],
    pub i: usize,
    pub comp: usize,
}

impl<'a> ShapeFn<'a> {
    pub fn v(&self, q: usize) -> f64 {
        self.values[(self.i * self.ncomp + self.comp) * self.npts + q]
    }

    pub fn grad(&self, q: usize, gd: usize) -> f64 {
        self.grads[((self.i * self.ncomp + self.comp) * self.gdim + gd) * self.npts + q]
    }

    pub fn v_slice(&self) -> &'a [f64] {
        let start = (self.i * self.ncomp + self.comp) * self.npts;
        &self.values[start..start + self.npts]
    }

    pub fn grad_slice(&self, gd: usize) -> &'a [f64] {
        let start = ((self.i * self.ncomp + self.comp) * self.gdim + gd) * self.npts;
        &self.grads[start..start + self.npts]
    }
}

/// Interpolated scalar PDE fields at one cell's quadrature points.
pub struct CellState<'a> {
    pub nfields: usize,
    pub npts: usize,
    pub gdim: usize,
    pub values: &'a [f64],
    pub grads: &'a [f64],
}

impl<'a> CellState<'a> {
    pub fn value(&self, field: usize, q: usize) -> f64 {
        self.values[field * self.npts + q]
    }

    pub fn grad(&self, field: usize, q: usize, gd: usize) -> f64 {
        self.grads[(field * self.gdim + gd) * self.npts + q]
    }
}

/// Reduced boundary RHS and matrix contributions.
pub struct BoundaryContributions {
    pub rhs: Vec<f64>,
    pub mat: SparseColMat<usize, f64>,
}

/// Geometry used to select a natural boundary kernel.
#[derive(Clone, Copy, Debug)]
pub struct BoundaryFacet {
    pub index: usize,
    pub midpoint: [f64; 2],
    pub physical_region: Option<crate::material::PhysicalRegion>,
}

/// Assemble selected natural-boundary kernels on straight quadrilateral facets.
pub(crate) fn assemble_quad_boundaries<'a, M, D, F>(
    mesh: &M,
    family: &LagrangeElementFamily<f64>,
    p: usize,
    reduced_size: usize,
    facet_metadata: &[Option<crate::material::PhysicalRegion>],
    time: f64,
    target_dof: D,
    mut select_kernel: F,
) -> BoundaryContributions
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>,
    D: Fn(usize) -> Option<usize>,
    F: FnMut(BoundaryFacet) -> Option<&'a dyn BoundaryIntegrator>,
{
    assert!(p >= 1, "boundary quadrature requires p >= 1");
    assert_eq!(mesh.topology_dim(), 2, "boundary assembly is 2D only");
    assert_eq!(mesh.geometry_dim(), 2, "boundary assembly is 2D-in-2D only");
    let space = FunctionSpaceImpl::new(mesh, family);
    let (qpts, wts) = single_integral_quadrature(
        QuadratureRule::GaussLobattoLegendre,
        Domain::Interval,
        p - 1,
    )
    .unwrap();
    let npts = wts.len();
    let xs: Vec<f64> = (0..npts).map(|q| qpts[2 * q + 1]).collect();
    let mut rhs = Vec::new();
    let mut nfields = 1;
    let mut triplets = Vec::new();
    let mut coord = [0.0; 2];

    for facet in mesh.entity_iter(ReferenceCellType::Interval) {
        let facet_index = facet.local_index();
        let topology = facet.topology();
        let mut cells = topology.connected_entity_iter(ReferenceCellType::Quadrilateral);
        let Some(cell_index) = cells.next() else {
            continue;
        };
        if cells.next().is_some() {
            continue;
        }

        let mut midpoint = [0.0; 2];
        let mut endpoints = [[0.0; 2]; 2];
        let mut count = 0;
        for point in facet.geometry().points() {
            point.coords(&mut coord);
            if count < 2 {
                endpoints[count] = coord;
            }
            midpoint[0] += coord[0];
            midpoint[1] += coord[1];
            count += 1;
        }
        assert_eq!(count, 2, "quadrilateral boundary facets must be intervals");
        midpoint[0] /= 2.0;
        midpoint[1] /= 2.0;
        let Some(kernel) = select_kernel(BoundaryFacet {
            index: facet_index,
            midpoint,
            physical_region: facet_metadata.get(facet_index).copied().flatten(),
        }) else {
            continue;
        };
        if rhs.is_empty() {
            nfields = kernel.nfields();
            assert!(
                nfields > 0,
                "boundary integrator must contain at least one field"
            );
            rhs.resize(nfields * reduced_size, 0.0);
        } else {
            assert_eq!(
                kernel.nfields(),
                nfields,
                "all selected boundary integrators must have the same field count"
            );
        }

        let cell = mesh
            .entity(ReferenceCellType::Quadrilateral, cell_index)
            .expect("boundary facet owner must be a quadrilateral");
        let local_facet = cell
            .topology()
            .sub_entity_iter(ReferenceCellType::Interval)
            .position(|index| index == facet_index)
            .expect("boundary facet missing from owning quadrilateral");

        let mut pts = rlst_dynamic_array!(f64, [2, npts]);
        for q in 0..npts {
            let (xi, eta) = match local_facet {
                0 => (xs[q], 0.0),
                1 => (0.0, xs[q]),
                2 => (1.0, xs[q]),
                3 => (xs[q], 1.0),
                _ => panic!("invalid quadrilateral local interval index {local_facet}"),
            };
            *pts.get_mut([0, q]).unwrap() = xi;
            *pts.get_mut([1, q]).unwrap() = eta;
        }
        let element = family.element(ReferenceCellType::Quadrilateral);
        let mut table = DynArray::<f64, 4>::from_shape(element.tabulate_array_shape(0, npts));
        element.tabulate(&pts, 0, &mut table);
        let cell_dofs = space
            .entity_closure_dofs(ReferenceCellType::Quadrilateral, cell_index)
            .unwrap();
        let facet_dofs = space
            .entity_closure_dofs(ReferenceCellType::Interval, facet_index)
            .unwrap();
        let nfacet = facet_dofs.len();
        let mut values = vec![0.0; nfacet * npts];
        for (i, &facet_dof) in facet_dofs.iter().enumerate() {
            let cell_i = cell_dofs
                .iter()
                .position(|&d| d == facet_dof)
                .expect("facet dof missing from owning quadrilateral");
            for q in 0..npts {
                values[i * npts + q] = *table.get([0, q, cell_i, 0]).unwrap();
            }
        }

        let length = ((endpoints[1][0] - endpoints[0][0]).powi(2)
            + (endpoints[1][1] - endpoints[0][1]).powi(2))
        .sqrt();
        assert!(length > 0.0, "boundary facet has zero length");
        let mut cell_center = [0.0; 2];
        let mut cell_point_count = 0;
        for point in cell.geometry().points() {
            point.coords(&mut coord);
            cell_center[0] += coord[0];
            cell_center[1] += coord[1];
            cell_point_count += 1;
        }
        assert!(
            cell_point_count >= 4,
            "quadrilateral owner has too few geometry points"
        );
        cell_center[0] /= cell_point_count as f64;
        cell_center[1] /= cell_point_count as f64;
        let mut normal = [
            -(endpoints[1][1] - endpoints[0][1]) / length,
            (endpoints[1][0] - endpoints[0][0]) / length,
        ];
        if normal[0] * (midpoint[0] - cell_center[0]) + normal[1] * (midpoint[1] - cell_center[1])
            < 0.0
        {
            normal[0] = -normal[0];
            normal[1] = -normal[1];
        }

        let jfacet_det = vec![length; npts];
        let gmap = mesh.geometry_map(ReferenceCellType::Quadrilateral, 1, &pts);
        let mut physical_points = rlst_dynamic_array!(f64, [2, npts]);
        gmap.physical_points(cell_index, &mut physical_points);
        let mut points = vec![0.0; npts * 2];
        for q in 0..npts {
            points[2 * q] = *physical_points.get([0, q]).unwrap();
            points[2 * q + 1] = *physical_points.get([1, q]).unwrap();
        }
        let ctx = FacetCtx {
            time,
            facet: FacetMeta {
                local_index: facet_index,
                physical_region: facet_metadata.get(facet_index).copied().flatten(),
            },
            tdim: 1,
            gdim: 2,
            ncomp: 1,
            npts,
            ndofs: nfacet,
            wts: &wts,
            jfacet_det: &jfacet_det,
            points: &points,
            normal: &normal,
            values: &values,
            grads: &[],
        };
        let local_size = nfields * nfacet;
        let mut local_rhs = vec![0.0; local_size];
        let mut local_mat = vec![0.0; local_size * local_size];
        kernel.assemble_facet_rhs(&ctx, &mut local_rhs);
        kernel.assemble_facet_mat(&ctx, &mut local_mat);
        for equation in 0..nfields {
            for (local_i, &full_i) in facet_dofs.iter().enumerate() {
                let Some(reduced_i) = target_dof(full_i) else {
                    continue;
                };
                rhs[equation * reduced_size + reduced_i] += local_rhs[equation * nfacet + local_i];
                for unknown in 0..nfields {
                    for (local_j, &full_j) in facet_dofs.iter().enumerate() {
                        if let Some(reduced_j) = target_dof(full_j) {
                            let row = equation * nfacet + local_i;
                            let col = unknown * nfacet + local_j;
                            let value = local_mat[row * local_size + col];
                            if value != 0.0 {
                                triplets.push(Triplet::new(
                                    equation * reduced_size + reduced_i,
                                    unknown * reduced_size + reduced_j,
                                    value,
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    if rhs.is_empty() {
        rhs.resize(reduced_size, 0.0);
    }
    let system_size = nfields * reduced_size;
    BoundaryContributions {
        rhs,
        mat: SparseColMat::try_new_from_triplets(system_size, system_size, &triplets).unwrap(),
    }
}

/// Per-facet boundary-assembly context.
pub struct FacetCtx<'a> {
    pub time: f64,
    pub facet: FacetMeta,
    pub tdim: usize,
    pub gdim: usize,
    pub ncomp: usize,
    pub npts: usize,
    pub ndofs: usize,
    pub wts: &'a [f64],
    pub jfacet_det: &'a [f64],
    pub points: &'a [f64],
    pub normal: &'a [f64],
    pub values: &'a [f64],
    pub grads: &'a [f64],
}

impl<'a> FacetCtx<'a> {
    pub fn point(&self, q: usize) -> &'a [f64] {
        &self.points[q * self.gdim..(q + 1) * self.gdim]
    }

    pub fn test(&'a self, i: usize, comp: usize) -> ShapeFn<'a> {
        ShapeFn {
            npts: self.npts,
            ncomp: self.ncomp,
            gdim: self.gdim,
            values: self.values,
            grads: self.grads,
            i,
            comp,
        }
    }

    pub fn trial(&'a self, i: usize, comp: usize) -> ShapeFn<'a> {
        self.test(i, comp)
    }
}
