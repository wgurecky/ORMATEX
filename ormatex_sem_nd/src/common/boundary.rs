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
use crate::material::{MeshMetadata, PhysicalRegion};

use super::contexts::FacetCtx;

/// Reduced boundary RHS and matrix contributions.
pub struct BoundaryContributions {
    /// Field-major reduced right-hand-side contributions.
    pub rhs: Vec<f64>,
    /// Field-major reduced sparse matrix contributions.
    pub mat: SparseColMat<usize, f64>,
}

/// Geometry used to select a natural-boundary kernel on a quadrilateral mesh.
#[derive(Clone, Copy, Debug)]
pub struct BoundaryFacet {
    /// Mesh-local interval facet index.
    pub index: usize,
    /// Physical midpoint of the facet.
    pub midpoint: [f64; 2],
    /// Optional physical region metadata for the facet.
    pub physical_region: Option<PhysicalRegion>,
}

/// Add a local matrix's prescribed-DOF contribution to a reduced RHS.
pub(crate) fn add_dirichlet_rhs_correction(
    rhs: &mut [f64],
    local: &[f64],
    field_reduced_dofs: &[&[Option<usize>]],
    field_prescribed_values: &[&[Option<f64>]],
    field_count: usize,
    field_offsets: &[usize],
) {
    let local_dof_count = field_reduced_dofs[0].len();
    let local_size = field_count * local_dof_count;
    for equation in 0..field_count {
        for (test_dof, &reduced_test_dof) in field_reduced_dofs[equation].iter().enumerate() {
            let Some(reduced_test_dof) = reduced_test_dof else {
                continue;
            };
            let row = equation * local_dof_count + test_dof;
            for unknown in 0..field_count {
                for (trial_dof, &value) in field_prescribed_values[unknown].iter().enumerate() {
                    let Some(value) = value else {
                        continue;
                    };
                    let col = unknown * local_dof_count + trial_dof;
                    rhs[field_offsets[equation] + reduced_test_dof] -=
                        local[row * local_size + col] * value;
                }
            }
        }
    }
}

/// Assemble selected natural-boundary kernels on straight quadrilateral
/// facets.
///
/// The selector receives each mesh boundary facet's index, physical midpoint,
/// and optional physical region. Returning `None` skips that facet; selected
/// facets are integrated with Gauss-Lobatto-Legendre quadrature of order
/// `polynomial_degree - 1`, using an outward unit normal. `target_dof` maps
/// full facet degrees of freedom to reduced indices, so eliminated DOFs are
/// omitted during scattering. All selected kernels must report the same field
/// count.
///
/// Only two-dimensional meshes embedded in two dimensions are supported, and
/// only facets with exactly one connected quadrilateral cell are assembled.
/// The returned RHS and matrix use field-major reduced indexing.
///
/// # Panics
///
/// Panics if `polynomial_degree` is zero, the mesh is not 2D-in-2D, a selected
/// kernel has no fields, selected kernels disagree about their field count, or
/// the mesh has malformed quadrilateral boundary geometry.
pub(crate) fn assemble_quad_boundaries<'a, M, D, S, F>(
    mesh: &M,
    family: &LagrangeElementFamily<f64>,
    polynomial_degree: usize,
    metadata: &MeshMetadata,
    time: f64,
    target_dof: D,
    field_reduced_size: S,
    mut select_kernel: F,
) -> BoundaryContributions
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>,
    D: Fn(usize, usize) -> Option<usize>,
    S: Fn(usize) -> usize,
    F: FnMut(BoundaryFacet) -> Option<&'a dyn BoundaryIntegrator>,
{
    assert!(
        polynomial_degree >= 1,
        "boundary quadrature requires p >= 1"
    );
    assert_eq!(mesh.topology_dim(), 2, "boundary assembly is 2D only");
    assert_eq!(mesh.geometry_dim(), 2, "boundary assembly is 2D-in-2D only");
    let space = FunctionSpaceImpl::new(mesh, family);
    let (qpts, wts) = single_integral_quadrature(
        QuadratureRule::GaussLobattoLegendre,
        Domain::Interval,
        polynomial_degree - 1,
    )
    .unwrap();
    let npts = wts.len();
    let xs: Vec<f64> = (0..npts).map(|q| qpts[2 * q + 1]).collect();
    let mut rhs = Vec::new();
    let mut nfields = 1;
    let mut field_offsets = Vec::new();
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
            physical_region: metadata.facet(facet_index).physical_region,
        }) else {
            continue;
        };
        if rhs.is_empty() {
            nfields = kernel.nfields();
            assert!(
                nfields > 0,
                "boundary integrator must contain at least one field"
            );
            field_offsets = (0..nfields)
                .scan(0, |offset, field| {
                    let current = *offset;
                    *offset += field_reduced_size(field);
                    Some(current)
                })
                .collect();
            let system_size =
                field_offsets.last().copied().unwrap_or(0) + field_reduced_size(nfields - 1);
            rhs.resize(system_size, 0.0);
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
            facet: metadata.facet(facet_index),
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
                let Some(reduced_i) = target_dof(equation, full_i) else {
                    continue;
                };
                rhs[field_offsets[equation] + reduced_i] += local_rhs[equation * nfacet + local_i];
                for unknown in 0..nfields {
                    for (local_j, &full_j) in facet_dofs.iter().enumerate() {
                        if let Some(reduced_j) = target_dof(unknown, full_j) {
                            let row = equation * nfacet + local_i;
                            let col = unknown * nfacet + local_j;
                            let value = local_mat[row * local_size + col];
                            if value != 0.0 {
                                triplets.push(Triplet::new(
                                    field_offsets[equation] + reduced_i,
                                    field_offsets[unknown] + reduced_j,
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
        rhs.resize(field_reduced_size(0), 0.0);
    }
    let system_size = if field_offsets.is_empty() {
        field_reduced_size(0)
    } else {
        field_offsets.last().copied().unwrap_or(0) + field_reduced_size(nfields - 1)
    };
    BoundaryContributions {
        rhs,
        mat: SparseColMat::try_new_from_triplets(system_size, system_size, &triplets).unwrap(),
    }
}
