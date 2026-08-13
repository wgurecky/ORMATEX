//! Shared finite-element contexts and geometry assembly infrastructure.

use faer::prelude::MatRef;
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
use crate::material::{CellMeta, FacetMeta, MaterialContext, MeshMetadata};

// ponytail: fixed batches reuse scratch without creating a task per cell; tune only after profiling.
pub(crate) const CELL_BATCH_SIZE: usize = 32;

/// Cached per-cell-type quadrature, basis tabulation, and geometry data.
pub(crate) struct CellData {
    pub(crate) wts: Vec<f64>,
    pub(crate) npts: usize,
    pub(crate) ndofs: usize,
    pub(crate) table: DynArray<f64, 4>,
    pub(crate) reference_values: Vec<f64>,
    pub(crate) nodal_quadrature: Vec<usize>,
    pub(crate) jinv_cache: DynArray<f64, 4>,
    pub(crate) jdets_cache: Vec<f64>,
    pub(crate) physical_points_cache: Vec<f64>,
}

/// Maps full function-space DOFs to the reduced system.
pub(crate) struct ReducedDofMap {
    /// Full DOF -> reduced DOF. `Some(i)` retains a DOF at reduced index `i`;
    /// `None` eliminates it. Multiple full DOFs may share one reduced index
    /// when periodic DOFs are identified.
    dof_lut: Vec<Option<usize>>,
    prescribed_values: Vec<Option<f64>>,
    n_reduced: usize,
}

impl ReducedDofMap {
    pub(crate) fn identity(n: usize) -> Self {
        Self {
            dof_lut: (0..n).map(Some).collect(),
            prescribed_values: vec![None; n],
            n_reduced: n,
        }
    }

    pub(crate) fn from_representatives(representatives: Vec<usize>) -> Self {
        let n = representatives.len();
        let mut representatives_in_order = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for &representative in &representatives {
            assert!(representative < n, "DOF representative out of range");
            if seen.insert(representative) {
                representatives_in_order.push(representative);
            }
        }
        let mut reduced_index = vec![usize::MAX; n];
        for (reduced, representative) in representatives_in_order.iter().enumerate() {
            reduced_index[*representative] = reduced;
        }
        let dof_lut = representatives
            .into_iter()
            .map(|representative| Some(reduced_index[representative]))
            .collect();
        Self::new(dof_lut, vec![None; n], representatives_in_order.len())
    }

    pub(crate) fn from_dirichlet_values<I>(n: usize, values: I) -> Self
    where
        I: IntoIterator<Item = (usize, f64)>,
    {
        let mut prescribed_values: Vec<Option<f64>> = vec![None; n];
        for (dof, value) in values {
            assert!(dof < n, "Dirichlet DOF out of range");
            assert!(value.is_finite(), "Dirichlet value must be finite");
            if let Some(previous) = prescribed_values[dof] {
                assert!(
                    (previous - value).abs() <= 1e-12 * previous.abs().max(value.abs()).max(1.0),
                    "conflicting Dirichlet values for DOF {dof}"
                );
            } else {
                prescribed_values[dof] = Some(value);
            }
        }
        let mut reduced = 0;
        let dof_lut = prescribed_values
            .iter()
            .map(|value| {
                if value.is_some() {
                    None
                } else {
                    let index = Some(reduced);
                    reduced += 1;
                    index
                }
            })
            .collect();
        Self::new(dof_lut, prescribed_values, reduced)
    }

    fn new(
        dof_lut: Vec<Option<usize>>,
        prescribed_values: Vec<Option<f64>>,
        n_reduced: usize,
    ) -> Self {
        assert_eq!(
            dof_lut.len(),
            prescribed_values.len(),
            "DOF map and prescribed-value lengths must match"
        );
        let mut present = vec![false; n_reduced];
        for (full, &reduced) in dof_lut.iter().enumerate() {
            match reduced {
                Some(reduced) => {
                    assert!(reduced < n_reduced, "reduced DOF out of range");
                    assert!(
                        prescribed_values[full].is_none(),
                        "retained DOF cannot have a prescribed value"
                    );
                    present[reduced] = true;
                }
                None => assert!(
                    prescribed_values[full].is_some(),
                    "eliminated DOF must have a prescribed value"
                ),
            }
        }
        assert!(
            present.into_iter().all(|present| present),
            "reduced DOF indices must be contiguous"
        );
        Self {
            dof_lut,
            prescribed_values,
            n_reduced,
        }
    }

    pub(crate) fn target(&self, full: usize) -> Option<usize> {
        self.dof_lut[full]
    }

    pub(crate) fn full_size(&self) -> usize {
        self.dof_lut.len()
    }

    pub(crate) fn reduced_size(&self) -> usize {
        self.n_reduced
    }

    pub(crate) fn map_cells(
        &self,
        cell_dofs: &[Vec<usize>],
    ) -> (Vec<Vec<Option<usize>>>, Vec<Vec<Option<f64>>>) {
        let reduced = cell_dofs
            .iter()
            .map(|dofs| dofs.iter().map(|&dof| self.target(dof)).collect())
            .collect();
        let prescribed = cell_dofs
            .iter()
            .map(|dofs| {
                dofs.iter()
                    .map(|&dof| self.prescribed_values[dof])
                    .collect()
            })
            .collect();
        (reduced, prescribed)
    }
}

/// Build a [`LocalCtx`] from cached data for one cell.
///
/// `cell_index` selects the cell metadata, Jacobian determinants, and physical
/// quadrature points. `grads` must contain physical basis gradients in
/// `[local_dof, geometric_direction, quadrature_point]` order for the first
/// `ndofs` local basis functions. The returned context borrows both the cached
/// data and `grads`.
pub(crate) fn cell_ctx<'a>(
    cell_data: &'a CellData,
    metadata: &'a MeshMetadata,
    tdim: usize,
    gdim: usize,
    time: f64,
    cell_index: usize,
    ndofs: usize,
    grads: &'a [f64],
) -> LocalCtx<'a> {
    let npts = cell_data.npts;
    LocalCtx {
        time,
        cell: metadata.cell(cell_index),
        tdim,
        gdim,
        ncomp: 1,
        npts,
        ndofs,
        wts: &cell_data.wts,
        jdets: &cell_data.jdets_cache[cell_index * npts..(cell_index + 1) * npts],
        points: &cell_data.physical_points_cache
            [cell_index * npts * gdim..(cell_index + 1) * npts * gdim],
        values: &cell_data.reference_values,
        grads: &grads[..ndofs * gdim * npts],
    }
}

/// Interpolate a reduced global state and its physical gradients to cell
/// quadrature points, including prescribed values for eliminated DOFs.
///
/// `reduced_dofs` maps each local basis function to a reduced global degree of
/// freedom. An entry of `None` uses the matching `prescribed_values` entry.
/// The state is field-major with row
/// `field * n_reduced + reduced_dof` and must have one column. The output
/// buffers use `values[field * npts + q]` and
/// `field_grads[(field * gdim + gd) * npts + q]` layouts; the returned
/// [`CellState`] borrows the initialized prefixes of those buffers.
pub(crate) fn interpolate_cell_state<'a>(
    cell_data: &CellData,
    gdim: usize,
    n_reduced: usize,
    nfields: usize,
    reduced_dofs: &[Option<usize>],
    prescribed_values: &[Option<f64>],
    state: MatRef<'_, f64>,
    basis_grads: &[f64],
    values: &'a mut [f64],
    field_grads: &'a mut [f64],
) -> CellState<'a> {
    let npts = cell_data.npts;
    assert_eq!(
        reduced_dofs.len(),
        prescribed_values.len(),
        "cell DOF maps must have matching lengths"
    );
    values[..nfields * npts].fill(0.0);
    field_grads[..nfields * gdim * npts].fill(0.0);
    for field in 0..nfields {
        for (local_i, &reduced) in reduced_dofs.iter().enumerate() {
            let coefficient = reduced.map_or(prescribed_values[local_i].unwrap_or(0.0), |i| {
                state[(field * n_reduced + i, 0)]
            });
            for q in 0..npts {
                values[field * npts + q] +=
                    coefficient * cell_data.reference_values[local_i * npts + q];
                for gd in 0..gdim {
                    field_grads[(field * gdim + gd) * npts + q] +=
                        coefficient * basis_grads[(local_i * gdim + gd) * npts + q];
                }
            }
        }
    }
    CellState {
        nfields,
        npts,
        gdim,
        values: &values[..nfields * npts],
        grads: &field_grads[..nfields * gdim * npts],
    }
}

/// Add the RHS correction caused by prescribed local DOF values.
///
/// For a local matrix `A` and prescribed values `u_b`, this appends
/// `-A_fb * u_b` to the reduced free-DOF RHS. The prescribed values are used
/// for every scalar field; scalar problems are the primary use case.
pub(crate) fn add_dirichlet_rhs_correction(
    rhs: &mut [f64],
    local: &[f64],
    reduced_dofs: &[Option<usize>],
    prescribed_values: &[Option<f64>],
    nfields: usize,
    n_reduced: usize,
) {
    let ndofs = reduced_dofs.len();
    let local_size = nfields * ndofs;
    assert_eq!(
        prescribed_values.len(),
        ndofs,
        "cell DOF map length mismatch"
    );
    for equation in 0..nfields {
        for (test_i, &reduced_i) in reduced_dofs.iter().enumerate() {
            let Some(reduced_i) = reduced_i else {
                continue;
            };
            let row = equation * ndofs + test_i;
            for unknown in 0..nfields {
                for (trial_i, &value) in prescribed_values.iter().enumerate() {
                    let Some(value) = value else {
                        continue;
                    };
                    let col = unknown * ndofs + trial_i;
                    rhs[equation * n_reduced + reduced_i] -= local[row * local_size + col] * value;
                }
            }
        }
    }
}

/// Add a field-major local vector into a reduced global vector.
///
/// `local[field * ndofs + local_dof]` is accumulated into
/// `out[field * n_reduced + reduced_dof]`. Entries whose reduced degree of
/// freedom is `None` are omitted. This is an additive scatter, so shared or
/// periodically identified degrees of freedom are combined by repeated
/// additions.
pub(crate) fn scatter_local_vector(
    out: &mut [f64],
    local: &[f64],
    reduced_dofs: &[Option<usize>],
    nfields: usize,
    n_reduced: usize,
) {
    let ndofs = reduced_dofs.len();
    for field in 0..nfields {
        for (local_i, &reduced_i) in reduced_dofs.iter().enumerate() {
            if let Some(reduced_i) = reduced_i {
                out[field * n_reduced + reduced_i] += local[field * ndofs + local_i];
            }
        }
    }
}

/// Append selected entries of a field-major local matrix as global triplets.
///
/// The local matrix is row-major with field-major rows and columns: a row is
/// `equation * ndofs + test_dof`, and a column is
/// `unknown * ndofs + trial_dof`. Reduced degrees of freedom mapped to
/// `None` are omitted, while `keep` decides whether each remaining value is
/// appended. Existing triplets are preserved and may contain duplicate global
/// coordinates from neighboring cells.
pub(crate) fn push_local_matrix_triplets(
    triplets: &mut Vec<Triplet<usize, usize, f64>>,
    local: &[f64],
    reduced_dofs: &[Option<usize>],
    nfields: usize,
    n_reduced: usize,
    keep: impl Fn(f64) -> bool,
) {
    let ndofs = reduced_dofs.len();
    let local_size = nfields * ndofs;
    for equation in 0..nfields {
        for (ti, &reduced_i) in reduced_dofs.iter().enumerate() {
            let Some(reduced_i) = reduced_i else {
                continue;
            };
            for unknown in 0..nfields {
                for (si, &reduced_j) in reduced_dofs.iter().enumerate() {
                    let Some(reduced_j) = reduced_j else {
                        continue;
                    };
                    let value = local[(equation * ndofs + ti) * local_size + unknown * ndofs + si];
                    if keep(value) {
                        triplets.push(Triplet::new(
                            equation * n_reduced + reduced_i,
                            unknown * n_reduced + reduced_j,
                            value,
                        ));
                    }
                }
            }
        }
    }
}

/// Assemble a block-diagonal lumped GLL mass matrix for multiple scalar fields.
///
/// Each local nodal basis function contributes its quadrature weight multiplied
/// by the cell Jacobian determinant at that node. Eliminated degrees of freedom
/// are skipped, and contributions from cells sharing a reduced degree of
/// freedom are combined in the resulting matrix. The matrix has size
/// `nfields * n_reduced` and has one identical scalar diagonal block per field.
///
/// # Panics
///
/// Panics if `nfields` is zero or if the cached cell and DOF data are
/// inconsistent.
pub(crate) fn assemble_lumped_mass(
    cell_data: &CellData,
    cell_reduced_dofs: &[Vec<Option<usize>>],
    n_reduced: usize,
    nfields: usize,
) -> SparseColMat<usize, f64> {
    assert!(nfields > 0, "mass requires at least one field");
    let mut triplets = Vec::with_capacity(cell_reduced_dofs.len() * cell_data.ndofs * nfields);
    for field in 0..nfields {
        for (cell, reduced_dofs) in cell_reduced_dofs.iter().enumerate() {
            for (local_dof, &reduced) in reduced_dofs.iter().enumerate() {
                if let Some(reduced) = reduced {
                    let q = cell_data.nodal_quadrature[local_dof];
                    let index = field * n_reduced + reduced;
                    triplets.push(Triplet::new(
                        index,
                        index,
                        cell_data.wts[q] * cell_data.jdets_cache[cell * cell_data.npts + q],
                    ));
                }
            }
        }
    }
    let system_size = nfields * n_reduced;
    SparseColMat::try_new_from_triplets(system_size, system_size, &triplets).unwrap()
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
    /// Return the physical coordinates of quadrature point `q`.
    ///
    /// The returned slice has length `gdim` and borrows the context's
    /// point-storage buffer.
    pub fn point(&self, q: usize) -> &'a [f64] {
        &self.points[q * self.gdim..(q + 1) * self.gdim]
    }

    /// Build the material-evaluation context for quadrature point `q`.
    ///
    /// `state` is `None` for state-independent coefficients and `Some` when a
    /// coefficient needs the interpolated [`CellState`]. The returned context
    /// includes this cell's physical point, metadata, time, and quadrature
    /// index.
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

    /// Return the `i`th test basis function and component as a [`ShapeFn`]
    /// view.
    ///
    /// The view borrows this context's basis values and physical gradients.
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

    /// Return the `i`th trial basis function and component as a [`ShapeFn`]
    /// view.
    ///
    /// Test and trial functions use the same basis data here; the separate
    /// methods make the role of the basis function explicit in bilinear forms.
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
    /// Return this basis function's value at quadrature point `q`.
    pub fn v(&self, q: usize) -> f64 {
        self.values[(self.i * self.ncomp + self.comp) * self.npts + q]
    }

    /// Return this basis function's physical gradient component `gd` at
    /// quadrature point `q`.
    pub fn grad(&self, q: usize, gd: usize) -> f64 {
        self.grads[((self.i * self.ncomp + self.comp) * self.gdim + gd) * self.npts + q]
    }

    /// Return all quadrature-point values for this basis function and
    /// component.
    ///
    /// The returned slice is ordered by quadrature point and has length
    /// `npts`.
    pub fn v_slice(&self) -> &'a [f64] {
        let start = (self.i * self.ncomp + self.comp) * self.npts;
        &self.values[start..start + self.npts]
    }

    /// Return all quadrature-point gradient values in geometric direction
    /// `gd`.
    ///
    /// The returned slice is ordered by quadrature point and has length
    /// `npts`.
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
    /// Return the interpolated value of field `field` at quadrature point `q`.
    pub fn value(&self, field: usize, q: usize) -> f64 {
        self.values[field * self.npts + q]
    }

    /// Return the physical gradient component `gd` of field `field` at
    /// quadrature point `q`.
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

/// Assemble selected natural-boundary kernels on straight quadrilateral
/// facets.
///
/// The selector receives each mesh boundary facet's index, physical midpoint,
/// and optional physical region. Returning `None` skips that facet; selected
/// facets are integrated with Gauss-Lobatto-Legendre quadrature of order
/// `p - 1`, using an outward unit normal. `target_dof` maps full facet degrees
/// of freedom to reduced indices, so eliminated DOFs are omitted during
/// scattering. All selected kernels must report the same field count.
///
/// Only two-dimensional meshes embedded in two dimensions are supported, and
/// only facets with exactly one connected quadrilateral cell are assembled.
/// The returned RHS and matrix use field-major reduced indexing and have size
/// `nfields * reduced_size`.
///
/// # Panics
///
/// Panics if `p` is zero, the mesh is not 2D-in-2D, a selected kernel has no
/// fields, selected kernels disagree about their field count, or the mesh has
/// malformed quadrilateral boundary geometry.
pub(crate) fn assemble_quad_boundaries<'a, M, D, F>(
    mesh: &M,
    family: &LagrangeElementFamily<f64>,
    p: usize,
    reduced_size: usize,
    metadata: &MeshMetadata,
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
    /// Return the physical coordinates of facet quadrature point `q`.
    ///
    /// The returned slice has length `gdim` and borrows the context's
    /// point-storage buffer.
    pub fn point(&self, q: usize) -> &'a [f64] {
        &self.points[q * self.gdim..(q + 1) * self.gdim]
    }

    /// Return the `i`th facet test basis function and component as a
    /// [`ShapeFn`] view.
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

    /// Return the `i`th facet trial basis function and component as a
    /// [`ShapeFn`] view.
    ///
    /// Test and trial functions use the same facet basis data here.
    pub fn trial(&'a self, i: usize, comp: usize) -> ShapeFn<'a> {
        self.test(i, comp)
    }
}
