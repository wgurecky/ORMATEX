//! Shared finite-element evaluation contexts and interpolated state views.
//!
//! # Lane-packed layouts (SIMD-over-element, `LANES` = 8)
//!
//! Volume tensor kernels evaluate `LANES` cells at once from lane-packed
//! buffers. For `nfields` fields, `npts` quadrature points, and geometric
//! dimension `gdim`, the layouts are:
//!
//! ```text
//! values[(field * npts + q) * LANES + lane]
//! grads[((field * gdim + d) * npts + q) * LANES + lane]
//! ```
//!
//! Facet (boundary) lane states reuse the same packing with `npts` set to the
//! facet quadrature count and `gdim = 2`. Facet grads are present only when the
//! boundary kernel reports `tensor_requires_gradients() == true`; otherwise
//! the grads slice may be empty and kernels must not read it.
//!
//! [`StateView`] is the per-lane scalar view over either layout: `stride = 1`
//! and `lane = 0` for a [`CellState`], `stride = LANES` for lane `lane` of a
//! [`LaneState`]. Coefficient evaluation ([`MaterialContext`]) takes
//! `Option<StateView>` so volume, boundary, and material code share one code
//! path.
//!
//! # Lane-packed contract
//!
//! Volume `tensor_residual` / `tensor_jacobian_action` evaluate `LANES` cells
//! at once: lanes are independent (lane `l` depends only on lane `l` of the
//! state and direction plus `ctxs[l]`, with no `mul_add` reassociation),
//! 1D kernels must write `f1y = 0.0` (and read only `gdim = 1` grads), and
//! outputs are overwritten. The same contract applies to the boundary kernels
//! in `StateTensorBoundaryIntegrator`.
use crate::material::MaterialContext;
use crate::regions::{CellMeta, FacetMeta};

/// Per-cell context passed to volume kernels.
///
/// Array fields use point-major storage for physical coordinates and
/// field-major storage for basis values and gradients. The context describes
/// the first `ndofs` local basis functions at `npts` quadrature points.
pub struct LocalCtx<'a> {
    /// Evaluation time for the current assembly operation.
    pub time: f64,
    /// Mesh metadata for the current cell.
    pub cell: CellMeta,
    /// Topological dimension of the reference cell.
    pub tdim: usize,
    /// Geometric dimension of the physical embedding space.
    pub gdim: usize,
    /// Number of components in each basis function.
    pub ncomp: usize,
    /// Number of quadrature points.
    pub npts: usize,
    /// Number of local basis functions represented by this context.
    pub ndofs: usize,
    /// Reference-cell quadrature weights, indexed by quadrature point.
    pub wts: &'a [f64],
    /// Cell Jacobian determinants, indexed by quadrature point.
    pub jdets: &'a [f64],
    /// Physical quadrature coordinates in `[quadrature_point, geometric_direction]` order.
    pub points: &'a [f64],
    /// Basis values in `[(basis, component), quadrature_point]` order.
    pub values: &'a [f64],
    /// Physical basis gradients in
    /// `[(basis, component, geometric_direction), quadrature_point]` order.
    pub grads: &'a [f64],
}

/// Pointwise context for a tensor-product operator.
///
/// Unlike [`LocalCtx`], this context deliberately contains no basis-pair
/// accessors. It is used between tensor-product evaluation and integration,
/// where kernels operate on complete quadrature-point fields. It supports the
/// 1D interval and 2D quadrilateral evaluators; the geometric dimension is
/// inferred from the cached point and Jacobian slice lengths.
pub struct TensorCtx<'a> {
    /// Evaluation time for the current assembly operation.
    pub time: f64,
    /// Mesh metadata for the current cell.
    pub cell: CellMeta,
    /// Number of one-dimensional GLL nodes.
    pub n1d: usize,
    /// Number of quadrature points (`n1d` in 1D, `n1d * n1d` in 2D).
    pub npts: usize,
    /// Reference-cell quadrature weights, indexed by quadrature point.
    pub wts: &'a [f64],
    /// Cell Jacobian determinants, indexed by quadrature point.
    pub jdets: &'a [f64],
    /// Weighted physical cell measure, indexed by quadrature point.
    pub wdet: &'a [f64],
    /// Physical quadrature coordinates in `[quadrature_point, direction]` order.
    pub points: &'a [f64],
    /// One-dimensional GLL differentiation matrix in row-major order.
    pub differentiation: &'a [f64],
    /// Quadrature-node to finite-element local-basis permutation.
    pub q_to_local: &'a [usize],
    /// Physical inverse Jacobians in `[q, reference_direction, physical_direction]` order.
    pub jinv: &'a [f64],
    /// Square root of the physical cell measure.
    pub cell_size: f64,
}

impl<'a> TensorCtx<'a> {
    /// Return the geometric dimension represented by this context.
    #[inline(always)]
    pub fn geometric_dimension(&self) -> usize {
        self.points.len() / self.npts
    }

    /// Return the physical coordinates of `quadrature_index`.
    #[inline(always)]
    pub fn point(&self, quadrature_index: usize) -> &'a [f64] {
        let gdim = self.geometric_dimension();
        &self.points[quadrature_index * gdim..quadrature_index * gdim + gdim]
    }

    /// Build the material-evaluation context at `quadrature_index`.
    #[inline(always)]
    pub fn material_context<'b>(
        &'b self,
        state: Option<&'b CellState<'b>>,
        quadrature_index: usize,
    ) -> MaterialContext<'b> {
        MaterialContext {
            time: self.time,
            point: self.point(quadrature_index),
            cell: self.cell,
            state: state.map(|s| s.view()),
            q: quadrature_index,
        }
    }

    /// Build the material-evaluation context for one lane of a lane-packed state.
    ///
    /// `self` must be that lane's [`TensorCtx`]; the returned context borrows
    /// the lane's scalar [`StateView`] (`stride = LANES`) at `quadrature_index`.
    #[inline(always)]
    pub fn lane_material_context<'b>(
        &'b self,
        state: Option<&'b LaneState<'b>>,
        lane: usize,
        quadrature_index: usize,
    ) -> MaterialContext<'b> {
        debug_assert!(lane < LANES, "lane index out of range");
        MaterialContext {
            time: self.time,
            point: self.point(quadrature_index),
            cell: self.cell,
            state: state.map(|s| s.lane(lane)),
            q: quadrature_index,
        }
    }
}

impl<'a> LocalCtx<'a> {
    /// Bridge a weak 1D context to the pointwise flux interface.
    ///
    /// [`FluxKernel1D`](crate::kernels::common::FluxKernel1D) callbacks may
    /// only read `time`, `cell`, `point(q)` and `npts` from the context; this
    /// bridge preserves exactly those fields from `self` and fills the
    /// sum-factorized geometry (weights, differentiation, inverse Jacobians)
    /// with empty slices. Flux implementations must not read the latter.
    ///
    /// # Returns
    /// [`TensorCtx`] borrowing `self`'s time/cell/point/weight slices.
    #[inline(always)]
    pub(crate) fn flux_tensor_ctx(&self) -> TensorCtx<'a> {
        TensorCtx {
            time: self.time,
            cell: self.cell,
            n1d: self.ndofs,
            npts: self.npts,
            wts: self.wts,
            jdets: self.jdets,
            wdet: &[],
            points: self.points,
            differentiation: &[],
            q_to_local: &[],
            jinv: &[],
            cell_size: 0.0,
        }
    }

    /// Return the physical coordinates of `quadrature_index`.
    ///
    /// The returned slice has length `gdim` and borrows the context's point
    /// storage.
    pub fn point(&self, quadrature_index: usize) -> &'a [f64] {
        &self.points[quadrature_index * self.gdim..(quadrature_index + 1) * self.gdim]
    }

    /// Build the material-evaluation context at `quadrature_index`.
    ///
    /// `state` is `None` for state-independent coefficients and `Some` when a
    /// coefficient needs the interpolated [`CellState`].
    pub fn material_context<'b>(
        &'b self,
        state: Option<&'b CellState<'b>>,
        quadrature_index: usize,
    ) -> MaterialContext<'b> {
        MaterialContext {
            time: self.time,
            point: self.point(quadrature_index),
            cell: self.cell,
            state: state.map(|s| s.view()),
            q: quadrature_index,
        }
    }

    /// Return the local test basis function at `basis_index` and `component`.
    ///
    /// A test function is the basis function used for the equation row in a
    /// weak form. The returned view borrows this context's basis data.
    pub fn test(&'a self, basis_index: usize, component: usize) -> ShapeFn<'a> {
        ShapeFn {
            npts: self.npts,
            ncomp: self.ncomp,
            gdim: self.gdim,
            values: self.values,
            grads: self.grads,
            i: basis_index,
            comp: component,
        }
    }

    /// Return the local trial basis function at `basis_index` and `component`.
    ///
    /// Test and trial functions use the same basis data here; separate methods
    /// make their roles explicit in bilinear forms.
    pub fn trial(&'a self, basis_index: usize, component: usize) -> ShapeFn<'a> {
        self.test(basis_index, component)
    }
}

/// A per-facet context passed to natural-boundary kernels.
///
/// The basis-value and gradient layouts match [`LocalCtx`]. `normal` stores
/// the outward unit normal at the facet.
pub struct FacetCtx<'a> {
    /// Evaluation time for the current assembly operation.
    pub time: f64,
    /// Mesh metadata for the current facet.
    pub facet: FacetMeta,
    /// Topological dimension of the facet reference cell.
    pub tdim: usize,
    /// Geometric dimension of the physical embedding space.
    pub gdim: usize,
    /// Number of components in each basis function.
    pub ncomp: usize,
    /// Number of quadrature points.
    pub npts: usize,
    /// Number of local facet basis functions.
    pub ndofs: usize,
    /// Facet reference-cell quadrature weights.
    pub wts: &'a [f64],
    /// Facet Jacobian determinants, indexed by quadrature point.
    pub jfacet_det: &'a [f64],
    /// Physical quadrature coordinates in `[quadrature_point, geometric_direction]` order.
    pub points: &'a [f64],
    /// Outward unit normal vector in physical geometric directions.
    pub normal: &'a [f64],
    /// Facet basis values in `[(basis, component), quadrature_point]` order.
    pub values: &'a [f64],
    /// Physical facet basis gradients in
    /// `[(basis, component, geometric_direction), quadrature_point]` order.
    pub grads: &'a [f64],
}

/// Pointwise context for a tensor-product boundary evaluator.
///
/// It intentionally omits basis-pair data. Tensor-capable state boundary
/// kernels return one pointwise flux/action, which the SEM layer contracts
/// against the one-dimensional trace basis.
pub struct TensorFacetCtx<'a> {
    /// Evaluation time for the current assembly operation.
    pub time: f64,
    /// Mesh metadata for the current facet.
    pub facet: FacetMeta,
    /// Number of facet quadrature points.
    pub npts: usize,
    /// Facet reference-cell quadrature weights, indexed by quadrature point.
    pub wts: &'a [f64],
    /// Facet Jacobian determinants, indexed by quadrature point.
    pub jfacet_det: &'a [f64],
    /// Physical quadrature coordinates in `[quadrature_point, direction]` order.
    pub points: &'a [f64],
    /// Outward unit normal vector in physical geometric directions.
    pub normal: &'a [f64],
}

impl<'a> TensorFacetCtx<'a> {
    #[inline(always)]
    pub fn point(&self, quadrature_index: usize) -> &'a [f64] {
        &self.points[quadrature_index * 2..(quadrature_index + 1) * 2]
    }
}

impl<'a> FacetCtx<'a> {
    /// Return the physical coordinates of `quadrature_index`.
    ///
    /// The returned slice has length `gdim` and borrows the context's point
    /// storage.
    pub fn point(&self, quadrature_index: usize) -> &'a [f64] {
        &self.points[quadrature_index * self.gdim..(quadrature_index + 1) * self.gdim]
    }

    /// Return the facet test basis function at `basis_index` and `component`.
    pub fn test(&'a self, basis_index: usize, component: usize) -> ShapeFn<'a> {
        ShapeFn {
            npts: self.npts,
            ncomp: self.ncomp,
            gdim: self.gdim,
            values: self.values,
            grads: self.grads,
            i: basis_index,
            comp: component,
        }
    }

    /// Return the facet trial basis function at `basis_index` and `component`.
    ///
    /// Test and trial functions use the same facet basis data here.
    pub fn trial(&'a self, basis_index: usize, component: usize) -> ShapeFn<'a> {
        self.test(basis_index, component)
    }
}

/// A view over one basis function's values and physical gradients.
pub struct ShapeFn<'a> {
    /// Number of quadrature points in the view.
    pub npts: usize,
    /// Number of components per basis function.
    pub ncomp: usize,
    /// Geometric dimension of the physical space.
    pub gdim: usize,
    /// Basis values in `[(basis, component), quadrature_point]` order.
    pub values: &'a [f64],
    /// Physical gradients in
    /// `[(basis, component, geometric_direction), quadrature_point]` order.
    pub grads: &'a [f64],
    /// Local basis-function index. Retained as `i` for struct-literal compatibility.
    pub i: usize,
    /// Component index. Retained as `comp` for struct-literal compatibility.
    pub comp: usize,
}

impl<'a> ShapeFn<'a> {
    /// Return this basis function's value at `quadrature_index`.
    pub fn v(&self, quadrature_index: usize) -> f64 {
        self.values[(self.i * self.ncomp + self.comp) * self.npts + quadrature_index]
    }

    /// Return the physical gradient in `geometric_direction` at
    /// `quadrature_index`.
    ///
    /// `geometric_direction` is a zero-based physical coordinate direction,
    /// such as `0` for x and `1` for y.
    pub fn grad(&self, quadrature_index: usize, geometric_direction: usize) -> f64 {
        self.grads[((self.i * self.ncomp + self.comp) * self.gdim + geometric_direction)
            * self.npts
            + quadrature_index]
    }

    /// Return all quadrature-point values for this basis function and
    /// component, ordered by quadrature point.
    pub fn v_slice(&self) -> &'a [f64] {
        let start = (self.i * self.ncomp + self.comp) * self.npts;
        &self.values[start..start + self.npts]
    }

    /// Return all gradient values in `geometric_direction`, ordered by
    /// quadrature point.
    pub fn grad_slice(&self, geometric_direction: usize) -> &'a [f64] {
        let start =
            ((self.i * self.ncomp + self.comp) * self.gdim + geometric_direction) * self.npts;
        &self.grads[start..start + self.npts]
    }
}

/// Interpolated PDE fields at one cell's quadrature points.
pub struct CellState<'a> {
    /// Number of scalar fields represented by this state.
    pub nfields: usize,
    /// Number of quadrature points.
    pub npts: usize,
    /// Geometric dimension of the physical space.
    pub gdim: usize,
    /// Field values in `[field_index, quadrature_point]` order.
    pub values: &'a [f64],
    /// Physical field gradients in
    /// `[(field_index, geometric_direction), quadrature_point]` order.
    pub grads: &'a [f64],
    /// Optional local-field to backing-field map. An empty map means identity.
    pub field_indices: &'a [usize],
}

impl<'a> CellState<'a> {
    /// Return `field_index`'s interpolated value at `quadrature_index`.
    #[inline(always)]
    pub fn value(&self, field_index: usize, quadrature_index: usize) -> f64 {
        let field = self
            .field_indices
            .get(field_index)
            .copied()
            .unwrap_or(field_index);
        self.values[field * self.npts + quadrature_index]
    }

    /// Return `field_index`'s physical gradient in `geometric_direction` at
    /// `quadrature_index`.
    #[inline(always)]
    pub fn grad(
        &self,
        field_index: usize,
        quadrature_index: usize,
        geometric_direction: usize,
    ) -> f64 {
        let field = self
            .field_indices
            .get(field_index)
            .copied()
            .unwrap_or(field_index);
        self.grads[(field * self.gdim + geometric_direction) * self.npts + quadrature_index]
    }

    /// Return the scalar [`StateView`] over this state (`stride` 1, `lane` 0).
    ///
    /// Only `Copy` fields and backing slices are read, so the view does not
    /// borrow the `CellState` wrapper itself and works on temporaries.
    ///
    /// # Returns
    /// Copyable view borrowing the same backing slices.
    #[inline(always)]
    pub fn view(&self) -> StateView<'a> {
        StateView {
            nfields: self.nfields,
            npts: self.npts,
            gdim: self.gdim,
            values: self.values,
            grads: self.grads,
            field_indices: self.field_indices,
            stride: 1,
            lane: 0,
        }
    }
}

/// Number of SIMD-over-element lanes for lane-packed tensor evaluation.
pub const LANES: usize = 8;

/// Fixed-size lane vector holding one scalar per SIMD-over-element lane.
pub type Lanes = [f64; LANES];

const _: () = assert!(LANES == super::cell::SIMD_CELL_WIDTH);

/// Lane-packed interpolated fields for [`LANES`] cells at all quadrature points.
///
/// `values` pack `LANES` cells as `values[(field * npts + q) * LANES + lane]`
/// and `grads` as `grads[((field * gdim + d) * npts + q) * LANES + lane]`,
/// matching the SIMD-over-element tensor assembler buffers. `field_indices`
/// carries the same remap semantics as [`CellState`]: an empty slice means
/// identity, otherwise `field_indices[local]` is the backing field.
#[derive(Clone, Copy)]
pub struct LaneState<'a> {
    /// Number of scalar fields represented by this state.
    pub nfields: usize,
    /// Number of quadrature points.
    pub npts: usize,
    /// Geometric dimension of the physical space.
    pub gdim: usize,
    /// Lane-packed field values in `[(field * npts + q) * LANES + lane]` order.
    pub values: &'a [f64],
    /// Lane-packed physical gradients in
    /// `[((field * gdim + d) * npts + q) * LANES + lane]` order.
    pub grads: &'a [f64],
    /// Optional local-field to backing-field map. An empty map means identity.
    pub field_indices: &'a [usize],
}

impl<'a> LaneState<'a> {
    /// Return the lane vector of `field_index`'s value at `quadrature_index`.
    ///
    /// # Arguments
    /// * `field_index` - local field index remapped through `field_indices`.
    /// * `quadrature_index` - quadrature-point index in `0..npts`.
    ///
    /// # Returns
    /// Borrowed `&Lanes` with one entry per lane; a plain pointer offset.
    #[inline(always)]
    pub fn value(&self, field_index: usize, quadrature_index: usize) -> &Lanes {
        let field = self
            .field_indices
            .get(field_index)
            .copied()
            .unwrap_or(field_index);
        let base = (field * self.npts + quadrature_index) * LANES;
        <&[f64; LANES]>::try_from(&self.values[base..base + LANES]).unwrap()
    }

    /// Return the lane vector of `field_index`'s gradient in `geometric_direction`.
    ///
    /// # Arguments
    /// * `field_index` - local field index remapped through `field_indices`.
    /// * `quadrature_index` - quadrature-point index in `0..npts`.
    /// * `geometric_direction` - physical coordinate direction.
    ///
    /// # Returns
    /// Borrowed `&Lanes` with one entry per lane; a plain pointer offset.
    #[inline(always)]
    pub fn grad(
        &self,
        field_index: usize,
        quadrature_index: usize,
        geometric_direction: usize,
    ) -> &Lanes {
        let field = self
            .field_indices
            .get(field_index)
            .copied()
            .unwrap_or(field_index);
        let base =
            ((field * self.gdim + geometric_direction) * self.npts + quadrature_index) * LANES;
        <&[f64; LANES]>::try_from(&self.grads[base..base + LANES]).unwrap()
    }

    /// Return the scalar [`StateView`] for one lane (`stride` [`LANES`]).
    ///
    /// # Arguments
    /// * `lane` - lane index in `0..LANES`.
    ///
    /// # Returns
    /// Copyable per-lane view borrowing the same backing slices.
    #[inline(always)]
    pub fn lane(&self, lane: usize) -> StateView<'a> {
        debug_assert!(lane < LANES, "lane index out of range");
        StateView {
            nfields: self.nfields,
            npts: self.npts,
            gdim: self.gdim,
            values: self.values,
            grads: self.grads,
            field_indices: self.field_indices,
            stride: LANES,
            lane,
        }
    }
}

/// Scalar per-lane view over interpolated PDE fields at quadrature points.
///
/// `StateView` unifies scalar [`CellState`] access (`stride` 1, `lane` 0) and
/// one lane of a lane-packed [`LaneState`] (`stride` [`LANES`]) behind the
/// same [`value`](Self::value)/[`grad`](Self::grad) interface with the same
/// `field_indices` remap semantics: an empty slice means identity, otherwise
/// `field_indices[local]` is the backing field. Coefficient evaluation
/// ([`MaterialContext`]) carries `Option<StateView>` so scalar assembly and
/// lane kernels share one code path.
///
/// Indexing is `values[(f * npts + q) * stride + lane]` and
/// `grads[((f * gdim + d) * npts + q) * stride + lane]`. For facet lane states
/// `npts` is the facet quadrature count and `gdim` is 2; when the boundary
/// kernel does not require gradients the grads slice may be empty and must
/// not be read.
#[derive(Clone, Copy)]
pub struct StateView<'a> {
    /// Number of scalar fields represented by this view.
    pub nfields: usize,
    /// Number of quadrature points.
    pub npts: usize,
    /// Geometric dimension of the physical space.
    pub gdim: usize,
    /// Backing field values with strided lane layout (see struct docs).
    pub values: &'a [f64],
    /// Backing physical gradients with strided lane layout (see struct docs).
    pub grads: &'a [f64],
    /// Optional local-field to backing-field map. An empty map means identity.
    pub field_indices: &'a [usize],
    /// Lane stride: 1 for a scalar [`CellState`], [`LANES`] for a [`LaneState`] lane.
    pub stride: usize,
    /// Lane index: always 0 for a scalar [`CellState`].
    pub lane: usize,
}

impl<'a> StateView<'a> {
    /// Return `field_index`'s interpolated value at `quadrature_index` for this view's lane.
    ///
    /// # Arguments
    /// * `field_index` - local field index remapped through `field_indices`.
    /// * `quadrature_index` - quadrature-point index in `0..npts`.
    ///
    /// # Returns
    /// Scalar value `values[(f * npts + q) * stride + lane]`.
    #[inline(always)]
    pub fn value(&self, field_index: usize, quadrature_index: usize) -> f64 {
        let field = self
            .field_indices
            .get(field_index)
            .copied()
            .unwrap_or(field_index);
        self.values[(field * self.npts + quadrature_index) * self.stride + self.lane]
    }

    /// Return `field_index`'s physical gradient in `geometric_direction` at
    /// `quadrature_index` for this view's lane.
    ///
    /// # Arguments
    /// * `field_index` - local field index remapped through `field_indices`.
    /// * `quadrature_index` - quadrature-point index in `0..npts`.
    /// * `geometric_direction` - physical coordinate direction.
    ///
    /// # Returns
    /// Scalar gradient `grads[((f * gdim + d) * npts + q) * stride + lane]`.
    #[inline(always)]
    pub fn grad(
        &self,
        field_index: usize,
        quadrature_index: usize,
        geometric_direction: usize,
    ) -> f64 {
        let field = self
            .field_indices
            .get(field_index)
            .copied()
            .unwrap_or(field_index);
        self.grads[((field * self.gdim + geometric_direction) * self.npts + quadrature_index)
            * self.stride
            + self.lane]
    }
}

impl<'a> From<&'a CellState<'a>> for StateView<'a> {
    /// Build the scalar view (`stride` 1, `lane` 0) borrowing the same slices.
    ///
    /// # Arguments
    /// * `state` - scalar cell state to view.
    ///
    /// # Returns
    /// Copyable [`StateView`] over `state`.
    #[inline(always)]
    fn from(state: &'a CellState<'a>) -> Self {
        StateView {
            nfields: state.nfields,
            npts: state.npts,
            gdim: state.gdim,
            values: state.values,
            grads: state.grads,
            field_indices: state.field_indices,
            stride: 1,
            lane: 0,
        }
    }
}
