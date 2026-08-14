use crate::material::{CellMeta, FacetMeta, MaterialContext};

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

impl<'a> LocalCtx<'a> {
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
            state,
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
}

impl<'a> CellState<'a> {
    /// Return `field_index`'s interpolated value at `quadrature_index`.
    pub fn value(&self, field_index: usize, quadrature_index: usize) -> f64 {
        self.values[field_index * self.npts + quadrature_index]
    }

    /// Return `field_index`'s physical gradient in `geometric_direction` at
    /// `quadrature_index`.
    pub fn grad(
        &self,
        field_index: usize,
        quadrature_index: usize,
        geometric_direction: usize,
    ) -> f64 {
        self.grads[(field_index * self.gdim + geometric_direction) * self.npts + quadrature_index]
    }
}
