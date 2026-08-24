use faer::prelude::{Mat, MatRef};
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

use crate::fields::FieldRegistry;
use crate::kernels::kernel_common::{
    BoundaryIntegrator, StateBoundaryIntegrator, StateBoundaryTerms,
};
use crate::material::{MeshMetadata, PhysicalRegion};

use super::contexts::{CellState, FacetCtx};

/// Reduced boundary RHS and matrix contributions.
pub struct BoundaryContributions {
    /// Field-major reduced right-hand-side contributions.
    pub rhs: Vec<f64>,
    /// Field-major reduced sparse matrix contributions.
    pub mat: SparseColMat<usize, f64>,
}

/// Nonlinear state-dependent boundary residual and Jacobian contributions.
pub struct StateBoundaryContributions {
    pub residual: Vec<f64>,
    pub jacobian: SparseColMat<usize, f64>,
}

pub(crate) fn assemble_quad_state_boundary_terms<M>(
    mesh: &M,
    family: &LagrangeElementFamily<f64>,
    polynomial_degree: usize,
    metadata: &MeshMetadata,
    fields: &FieldRegistry,
    time: f64,
    state: MatRef<'_, f64>,
    field_reduced_dofs: &[Vec<Vec<Option<usize>>>],
    field_prescribed_values: &[Vec<Vec<Option<f64>>>],
    field_offsets: &[usize],
    terms: &StateBoundaryTerms,
) -> StateBoundaryContributions
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>,
{
    let (residual, jacobian) = assemble_quad_state_boundary_impl(
        mesh,
        family,
        polynomial_degree,
        metadata,
        fields,
        time,
        state,
        field_reduced_dofs,
        field_prescribed_values,
        field_offsets,
        |facet| terms.kernel_for(facet.index),
        true,
        true,
    );
    StateBoundaryContributions {
        residual: residual.unwrap(),
        jacobian: jacobian.unwrap(),
    }
}

pub(crate) fn assemble_quad_state_boundary_residual<M>(
    mesh: &M,
    family: &LagrangeElementFamily<f64>,
    polynomial_degree: usize,
    metadata: &MeshMetadata,
    fields: &FieldRegistry,
    time: f64,
    state: MatRef<'_, f64>,
    field_reduced_dofs: &[Vec<Vec<Option<usize>>>],
    field_prescribed_values: &[Vec<Vec<Option<f64>>>],
    field_offsets: &[usize],
    terms: &StateBoundaryTerms,
) -> Vec<f64>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>,
{
    assemble_quad_state_boundary_impl(
        mesh,
        family,
        polynomial_degree,
        metadata,
        fields,
        time,
        state,
        field_reduced_dofs,
        field_prescribed_values,
        field_offsets,
        |facet| terms.kernel_for(facet.index),
        true,
        false,
    )
    .0
    .unwrap()
}

pub(crate) fn assemble_quad_state_boundary_jacobian<M>(
    mesh: &M,
    family: &LagrangeElementFamily<f64>,
    polynomial_degree: usize,
    metadata: &MeshMetadata,
    fields: &FieldRegistry,
    time: f64,
    state: MatRef<'_, f64>,
    field_reduced_dofs: &[Vec<Vec<Option<usize>>>],
    field_prescribed_values: &[Vec<Vec<Option<f64>>>],
    field_offsets: &[usize],
    terms: &StateBoundaryTerms,
) -> SparseColMat<usize, f64>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>,
{
    assemble_quad_state_boundary_impl(
        mesh,
        family,
        polynomial_degree,
        metadata,
        fields,
        time,
        state,
        field_reduced_dofs,
        field_prescribed_values,
        field_offsets,
        |facet| terms.kernel_for(facet.index),
        false,
        true,
    )
    .1
    .unwrap()
}

pub(crate) fn apply_quad_state_boundary_terms<M>(
    mesh: &M,
    family: &LagrangeElementFamily<f64>,
    polynomial_degree: usize,
    metadata: &MeshMetadata,
    fields: &FieldRegistry,
    time: f64,
    state: MatRef<'_, f64>,
    direction: MatRef<'_, f64>,
    field_reduced_dofs: &[Vec<Vec<Option<usize>>>],
    field_prescribed_values: &[Vec<Vec<Option<f64>>>],
    field_offsets: &[usize],
    terms: &StateBoundaryTerms,
) -> Mat<f64>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>,
{
    assert!(
        polynomial_degree >= 1,
        "boundary quadrature requires p >= 1"
    );
    assert_eq!(mesh.topology_dim(), 2, "boundary assembly is 2D only");
    assert_eq!(mesh.geometry_dim(), 2, "boundary assembly is 2D-in-2D only");
    let nfields = fields.len();
    let system_size = *field_offsets.last().unwrap();
    assert_eq!(field_offsets.len(), nfields + 1);
    assert_eq!(state.nrows(), system_size, "state size mismatch");
    assert_eq!(state.ncols(), 1, "boundary state requires one column");
    assert_eq!(
        direction.nrows(),
        system_size,
        "boundary direction size mismatch"
    );
    assert!(
        field_reduced_dofs.len() == 1 || field_reduced_dofs.len() == nfields,
        "field DOF maps must be shared or field-specific"
    );
    let space = FunctionSpaceImpl::new(mesh, family);
    let (qpts, wts) = single_integral_quadrature(
        QuadratureRule::GaussLobattoLegendre,
        Domain::Interval,
        polynomial_degree - 1,
    )
    .unwrap();
    let npts = wts.len();
    let xs: Vec<f64> = (0..npts).map(|q| qpts[2 * q + 1]).collect();
    let mut out = Mat::<f64>::zeros(system_size, direction.ncols());
    let mut coord = [0.0; 2];

    for facet in mesh.entity_iter(ReferenceCellType::Interval) {
        let facet_index = facet.local_index();
        let Some(kernel) = terms.kernel_for(facet_index) else {
            continue;
        };
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
        let mut table = DynArray::<f64, 4>::from_shape(element.tabulate_array_shape(1, npts));
        element.tabulate(&pts, 1, &mut table);
        let cell_dofs = space
            .entity_closure_dofs(ReferenceCellType::Quadrilateral, cell_index)
            .unwrap();
        let facet_dofs = space
            .entity_closure_dofs(ReferenceCellType::Interval, facet_index)
            .unwrap();
        let nfacet = facet_dofs.len();
        let facet_cell_indices: Vec<usize> = facet_dofs
            .iter()
            .map(|&dof| {
                cell_dofs
                    .iter()
                    .position(|&cell_dof| cell_dof == dof)
                    .expect("facet dof missing from owning quadrilateral")
            })
            .collect();
        let mut values = vec![0.0; nfacet * npts];
        for (i, &cell_i) in facet_cell_indices.iter().enumerate() {
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
        let maps = if field_reduced_dofs.len() == 1 {
            (0..nfields)
                .map(|_| field_reduced_dofs[0][cell_index].as_slice())
                .collect::<Vec<_>>()
        } else {
            (0..nfields)
                .map(|field| field_reduced_dofs[field][cell_index].as_slice())
                .collect::<Vec<_>>()
        };
        let prescribed = if field_prescribed_values.len() == 1 {
            (0..nfields)
                .map(|_| field_prescribed_values[0][cell_index].as_slice())
                .collect::<Vec<_>>()
        } else {
            (0..nfields)
                .map(|field| field_prescribed_values[field][cell_index].as_slice())
                .collect::<Vec<_>>()
        };
        let mut state_values = vec![0.0; nfields * npts];
        for field in 0..nfields {
            for (facet_i, &cell_i) in facet_cell_indices.iter().enumerate() {
                let coefficient = maps[field][cell_i]
                    .map_or(prescribed[field][cell_i].unwrap_or(0.0), |reduced| {
                        state[(field_offsets[field] + reduced, 0)]
                    });
                for q in 0..npts {
                    state_values[field * npts + q] += coefficient * values[facet_i * npts + q];
                }
            }
        }
        let mut state_grads = vec![0.0; nfields * 2 * npts];
        for field in 0..nfields {
            for &cell_i in &facet_cell_indices {
                let coefficient = maps[field][cell_i]
                    .map_or(prescribed[field][cell_i].unwrap_or(0.0), |reduced| {
                        state[(field_offsets[field] + reduced, 0)]
                    });
                for q in 0..npts {
                    for direction in 0..2 {
                        state_grads[(field * 2 + direction) * npts + q] +=
                            coefficient * *table.get([1 + direction, q, cell_i, 0]).unwrap();
                    }
                }
            }
        }
        let facet_state = CellState {
            nfields,
            npts,
            gdim: 2,
            values: &state_values,
            grads: &state_grads,
        };
        let local_size = nfields * nfacet;
        let mut local_direction = vec![0.0; local_size];
        let mut local_action = vec![0.0; local_size];
        for column in 0..direction.ncols() {
            for field in 0..nfields {
                for (facet_i, &cell_i) in facet_cell_indices.iter().enumerate() {
                    local_direction[field * nfacet + facet_i] = maps[field][cell_i]
                        .map_or(0.0, |reduced| {
                            direction[(field_offsets[field] + reduced, column)]
                        });
                }
            }
            kernel.apply_local_jacobian(&ctx, &facet_state, &local_direction, &mut local_action);
            for field in 0..nfields {
                for (facet_i, &cell_i) in facet_cell_indices.iter().enumerate() {
                    let Some(reduced) = maps[field][cell_i] else {
                        continue;
                    };
                    out[(field_offsets[field] + reduced, column)] +=
                        local_action[field * nfacet + facet_i];
                }
            }
        }
    }
    out
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
    fields: &FieldRegistry,
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
    let nfields = fields.len();
    let mut selected = false;
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
        if !selected {
            assert_eq!(
                kernel.nfields(),
                nfields,
                "boundary integrator field count does not match the SEM problem"
            );
            if let Some(names) = kernel.field_names() {
                assert_eq!(
                    names.as_slice(),
                    fields.names(),
                    "boundary integrator field names/order do not match the SEM problem"
                );
            }
            selected = true;
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
            if let Some(names) = kernel.field_names() {
                assert_eq!(
                    names.as_slice(),
                    fields.names(),
                    "boundary integrator field names/order do not match the SEM problem"
                );
            }
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
        let mut table = DynArray::<f64, 4>::from_shape(element.tabulate_array_shape(1, npts));
        element.tabulate(&pts, 1, &mut table);
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

    let system_size = if field_offsets.is_empty() {
        (0..nfields).map(|field| field_reduced_size(field)).sum()
    } else {
        field_offsets.last().copied().unwrap_or(0) + field_reduced_size(nfields - 1)
    };
    if rhs.is_empty() {
        rhs.resize(system_size, 0.0);
    }
    BoundaryContributions {
        rhs,
        mat: SparseColMat::try_new_from_triplets(system_size, system_size, &triplets).unwrap(),
    }
}

fn assemble_quad_state_boundary_impl<'a, M, F>(
    mesh: &M,
    family: &LagrangeElementFamily<f64>,
    polynomial_degree: usize,
    metadata: &MeshMetadata,
    fields: &FieldRegistry,
    time: f64,
    state: MatRef<'_, f64>,
    field_reduced_dofs: &[Vec<Vec<Option<usize>>>],
    field_prescribed_values: &[Vec<Vec<Option<f64>>>],
    field_offsets: &[usize],
    mut select_kernel: F,
    include_residual: bool,
    include_jacobian: bool,
) -> (Option<Vec<f64>>, Option<SparseColMat<usize, f64>>)
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>,
    F: FnMut(BoundaryFacet) -> Option<&'a dyn StateBoundaryIntegrator>,
{
    assert!(
        polynomial_degree >= 1,
        "boundary quadrature requires p >= 1"
    );
    assert_eq!(mesh.topology_dim(), 2, "boundary assembly is 2D only");
    assert_eq!(mesh.geometry_dim(), 2, "boundary assembly is 2D-in-2D only");
    let nfields = fields.len();
    assert_eq!(field_offsets.len(), nfields + 1);
    assert_eq!(state.nrows(), *field_offsets.last().unwrap());
    assert_eq!(state.ncols(), 1);
    assert!(
        field_reduced_dofs.len() == 1 || field_reduced_dofs.len() == nfields,
        "field DOF maps must be shared or field-specific"
    );
    let space = FunctionSpaceImpl::new(mesh, family);
    let (qpts, wts) = single_integral_quadrature(
        QuadratureRule::GaussLobattoLegendre,
        Domain::Interval,
        polynomial_degree - 1,
    )
    .unwrap();
    let npts = wts.len();
    let xs: Vec<f64> = (0..npts).map(|q| qpts[2 * q + 1]).collect();
    let system_size = *field_offsets.last().unwrap();
    let mut residual = include_residual.then(|| vec![0.0; system_size]);
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
        assert_eq!(
            kernel.nfields(),
            nfields,
            "state boundary field count does not match the SEM problem"
        );
        if let Some(names) = kernel.field_names() {
            assert_eq!(
                names.as_slice(),
                fields.names(),
                "state boundary field names/order do not match the SEM problem"
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
        let mut table = DynArray::<f64, 4>::from_shape(element.tabulate_array_shape(1, npts));
        element.tabulate(&pts, 1, &mut table);
        let cell_dofs = space
            .entity_closure_dofs(ReferenceCellType::Quadrilateral, cell_index)
            .unwrap();
        let facet_dofs = space
            .entity_closure_dofs(ReferenceCellType::Interval, facet_index)
            .unwrap();
        let nfacet = facet_dofs.len();
        let mut values = vec![0.0; nfacet * npts];
        let mut facet_cell_indices = Vec::with_capacity(nfacet);
        for (i, &facet_dof) in facet_dofs.iter().enumerate() {
            let cell_i = cell_dofs
                .iter()
                .position(|&d| d == facet_dof)
                .expect("facet dof missing from owning quadrilateral");
            facet_cell_indices.push(cell_i);
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

        let maps = if field_reduced_dofs.len() == 1 {
            (0..nfields)
                .map(|_| field_reduced_dofs[0][cell_index].as_slice())
                .collect::<Vec<_>>()
        } else {
            (0..nfields)
                .map(|field| field_reduced_dofs[field][cell_index].as_slice())
                .collect::<Vec<_>>()
        };
        let prescribed = if field_prescribed_values.len() == 1 {
            (0..nfields)
                .map(|_| field_prescribed_values[0][cell_index].as_slice())
                .collect::<Vec<_>>()
        } else {
            (0..nfields)
                .map(|field| field_prescribed_values[field][cell_index].as_slice())
                .collect::<Vec<_>>()
        };
        let mut state_values = vec![0.0; nfields * npts];
        for field in 0..nfields {
            for (facet_i, &cell_i) in facet_cell_indices.iter().enumerate() {
                let coefficient = maps[field][cell_i]
                    .map_or(prescribed[field][cell_i].unwrap_or(0.0), |reduced| {
                        state[(field_offsets[field] + reduced, 0)]
                    });
                for q in 0..npts {
                    state_values[field * npts + q] += coefficient * values[facet_i * npts + q];
                }
            }
        }
        let mut state_grads = vec![0.0; nfields * 2 * npts];
        for field in 0..nfields {
            for &cell_i in &facet_cell_indices {
                let coefficient = maps[field][cell_i]
                    .map_or(prescribed[field][cell_i].unwrap_or(0.0), |reduced| {
                        state[(field_offsets[field] + reduced, 0)]
                    });
                for q in 0..npts {
                    for direction in 0..2 {
                        state_grads[(field * 2 + direction) * npts + q] +=
                            coefficient * *table.get([1 + direction, q, cell_i, 0]).unwrap();
                    }
                }
            }
        }
        let facet_state = CellState {
            nfields,
            npts,
            gdim: 2,
            values: &state_values,
            grads: &state_grads,
        };

        let local_size = nfields * nfacet;
        let mut local_residual = include_residual.then(|| vec![0.0; local_size]);
        let mut local_jacobian = include_jacobian.then(|| vec![0.0; local_size * local_size]);
        if let Some(local_residual) = local_residual.as_mut() {
            kernel.assemble_local_residual(&ctx, &facet_state, local_residual);
        }
        if let Some(local_jacobian) = local_jacobian.as_mut() {
            kernel.assemble_local_jacobian(&ctx, &facet_state, local_jacobian);
        }
        for equation in 0..nfields {
            for (local_i, &full_i) in facet_dofs.iter().enumerate() {
                let cell_i = cell_i_for_dof(&cell_dofs, full_i);
                let target = if field_reduced_dofs.len() == 1 {
                    field_reduced_dofs[0][cell_index][cell_i]
                } else {
                    field_reduced_dofs[equation][cell_index][cell_i]
                };
                let Some(reduced_i) = target else {
                    continue;
                };
                if let Some(residual) = residual.as_mut() {
                    residual[field_offsets[equation] + reduced_i] +=
                        local_residual.as_ref().unwrap()[equation * nfacet + local_i];
                }
                if let Some(local_jacobian) = local_jacobian.as_ref() {
                    for unknown in 0..nfields {
                        for (local_j, &full_j) in facet_dofs.iter().enumerate() {
                            let cell_j = cell_i_for_dof(&cell_dofs, full_j);
                            let target = if field_reduced_dofs.len() == 1 {
                                field_reduced_dofs[0][cell_index][cell_j]
                            } else {
                                field_reduced_dofs[unknown][cell_index][cell_j]
                            };
                            let Some(reduced_j) = target else {
                                continue;
                            };
                            let row = equation * nfacet + local_i;
                            let col = unknown * nfacet + local_j;
                            let value = local_jacobian[row * local_size + col];
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
    let jacobian = include_jacobian
        .then(|| SparseColMat::try_new_from_triplets(system_size, system_size, &triplets).unwrap());
    (residual, jacobian)
}

fn cell_i_for_dof(cell_dofs: &[usize], full_dof: usize) -> usize {
    cell_dofs
        .iter()
        .position(|&dof| dof == full_dof)
        .expect("boundary dof missing from owning cell")
}
