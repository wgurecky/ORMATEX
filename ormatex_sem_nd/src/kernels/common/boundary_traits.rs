//! State-dependent and state-independent boundary traits.
use crate::common::{CellState, FacetCtx, TensorFacetCtx};

/// Boundary integrator trait for Neumann and Robin forms.
pub trait BoundaryIntegrator {
    /// Number of scalar equation/unknown fields in this boundary form.
    ///
    /// # Returns
    /// Field count; also the block count of the facet matrix/RHS.
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the boundary form.
    ///
    /// # Returns
    /// `None` for anonymous forms, otherwise one name per field.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    /// Bare facet integrand for one equation and test dof.
    ///
    /// Returns the pointwise integrand *without* facet weight or facet Jacobian;
    /// assembly forms `sum_q wts[q] * jfacet_det[q] * integrand_rhs(...)`.
    ///
    /// # Arguments
    /// * `ctx` - facet context with trace basis values/gradients, normals, and points.
    /// * `equation` - equation index in `0..nfields()`.
    /// * `q` - facet quadrature-point index in `0..ctx.npts`.
    /// * `test_i` - facet test basis index in `0..ctx.ndofs`.
    ///
    /// # Returns
    /// Scalar boundary integrand at `q` for test function `test_i`.
    fn integrand_rhs(&self, ctx: &FacetCtx, equation: usize, q: usize, test_i: usize) -> f64;

    /// Bare facet integrand for one equation/unknown block and basis pair.
    ///
    /// Default is `0.0` (pure Neumann/RHS-only form). Assembly multiplies by
    /// `wts[q] * jfacet_det[q]` and sums over `q`.
    ///
    /// # Arguments
    /// * `ctx` - facet context with trace basis data.
    /// * `equation` - equation index in `0..nfields()`.
    /// * `unknown` - unknown index in `0..nfields()`.
    /// * `q` - facet quadrature-point index in `0..ctx.npts`.
    /// * `test_i` - facet test basis index in `0..ctx.ndofs`.
    /// * `trial_i` - facet trial basis index in `0..ctx.ndofs`.
    ///
    /// # Returns
    /// Scalar boundary matrix integrand at `q`.
    fn integrand_mat(
        &self,
        _ctx: &FacetCtx,
        _equation: usize,
        _unknown: usize,
        _q: usize,
        _test_i: usize,
        _trial_i: usize,
    ) -> f64 {
        0.0
    }

    /// Assemble a field-major facet RHS vector.
    ///
    /// # Arguments
    /// * `ctx` - facet context with `ndofs` trace basis functions.
    /// * `rhs` - vector of length `nf*ndofs` with layout
    ///   `rhs[equation*n + test_i]`. Overwritten.
    fn assemble_facet_rhs(&self, ctx: &FacetCtx, rhs: &mut [f64]) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        assert!(
            nf > 0,
            "boundary integrator must contain at least one field"
        );
        assert_eq!(rhs.len(), nf * n, "local boundary RHS size mismatch");
        for equation in 0..nf {
            for ti in 0..n {
                let mut acc = 0.0;
                for q in 0..ctx.npts {
                    acc +=
                        ctx.wts[q] * ctx.jfacet_det[q] * self.integrand_rhs(ctx, equation, q, ti);
                }
                rhs[equation * n + ti] = acc;
            }
        }
    }

    /// Assemble a field-major facet matrix.
    ///
    /// # Arguments
    /// * `ctx` - facet context with `ndofs` trace basis functions.
    /// * `mat` - matrix of length `(nf*n)^2`, row-major with
    ///   `row = equation*n + test_i`, `col = unknown*n + trial_i`. Overwritten.
    fn assemble_facet_mat(&self, ctx: &FacetCtx, mat: &mut [f64]) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        let local_size = nf * n;
        assert_eq!(
            mat.len(),
            local_size * local_size,
            "local boundary matrix size mismatch"
        );
        for equation in 0..nf {
            for unknown in 0..nf {
                for ti in 0..n {
                    for si in 0..n {
                        let mut acc = 0.0;
                        for q in 0..ctx.npts {
                            acc += ctx.wts[q]
                                * ctx.jfacet_det[q]
                                * self.integrand_mat(ctx, equation, unknown, q, ti, si);
                        }
                        let row = equation * n + ti;
                        let col = unknown * n + si;
                        mat[row * local_size + col] = acc;
                    }
                }
            }
        }
    }
}

/// State-aware nonlinear boundary form.
///
/// Unlike [`BoundaryIntegrator`], this form receives the interpolated state at
/// facet quadrature points and can therefore represent nonlinear conditions
/// such as backflow-stabilizing outflow traction.
pub trait StateBoundaryIntegrator: Send + Sync {
    /// Number of scalar equation/unknown fields in this boundary form.
    ///
    /// # Returns
    /// Field count used when input/output counts are not overridden.
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the form.
    ///
    /// # Returns
    /// `None` for anonymous forms, otherwise one name per field.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    /// Number of compact state fields consumed by this term.
    ///
    /// # Returns
    /// Input field count; must match `state.nfields` at assembly.
    fn input_nfields(&self) -> usize {
        self.nfields()
    }

    /// Number of compact residual fields produced by this term.
    ///
    /// # Returns
    /// Output equation count; sets the row-block count of the facet residual.
    fn output_nfields(&self) -> usize {
        self.nfields()
    }

    /// Optional ordered names for compact state fields.
    ///
    /// # Returns
    /// `None` for positional fields, otherwise one name per input field.
    fn input_field_names(&self) -> Option<Vec<String>> {
        self.field_names()
    }

    /// Optional ordered names for compact residual fields.
    ///
    /// # Returns
    /// `None` for positional fields, otherwise one name per output field.
    fn output_field_names(&self) -> Option<Vec<String>> {
        self.field_names()
    }

    /// Bare facet residual integrand for one equation and test dof.
    ///
    /// Returns the pointwise integrand *without* `wts[q]*jfacet_det[q]`;
    /// assembly forms `out[equation*n + test_i] = sum_q wts[q]*jfacet_det[q]*integrand`.
    ///
    /// # Arguments
    /// * `ctx` - facet context with trace basis, normals, and points.
    /// * `state` - interpolated facet state with `input_nfields()` fields.
    /// * `equation` - equation index in `0..output_nfields()`.
    /// * `q` - facet quadrature-point index in `0..ctx.npts`.
    /// * `test_i` - facet test basis index in `0..ctx.ndofs`.
    ///
    /// # Returns
    /// Scalar weak boundary integrand at `q`.
    fn residual_integrand(
        &self,
        ctx: &FacetCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64;

    /// Bare facet Jacobian integrand for one equation/unknown block and basis pair.
    ///
    /// Gateaux derivative of [`residual_integrand`](Self::residual_integrand)
    /// in the `trial_i` direction of `unknown`.
    ///
    /// # Arguments
    /// * `ctx` - facet context with trace basis data.
    /// * `state` - linearization point with `input_nfields()` fields.
    /// * `equation` - equation index in `0..output_nfields()`.
    /// * `unknown` - unknown index in `0..input_nfields()`.
    /// * `q` - facet quadrature-point index in `0..ctx.npts`.
    /// * `test_i` - facet test basis index in `0..ctx.ndofs`.
    /// * `trial_i` - facet trial basis index in `0..ctx.ndofs`.
    ///
    /// # Returns
    /// Scalar facet Jacobian integrand at `q`.
    fn jacobian_integrand(
        &self,
        ctx: &FacetCtx,
        state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64;

    /// Apply the facet Jacobian to a coefficient-space direction.
    ///
    /// Computes `out = J(state) * direction` without forming `J`.
    ///
    /// # Arguments
    /// * `ctx` - facet context with `ndofs` trace basis functions.
    /// * `state` - linearization point with `input_nfields()` fields.
    /// * `direction` - trial coefficients of length `input_nfields()*ndofs`
    ///   with layout `direction[unknown*n + trial_i]`.
    /// * `out` - action of length `output_nfields()*ndofs` with layout
    ///   `out[equation*n + test_i]`. Overwritten.
    fn apply_local_jacobian(
        &self,
        ctx: &FacetCtx,
        state: &CellState,
        direction: &[f64],
        out: &mut [f64],
    ) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        let local_size = nf * n;
        assert_eq!(
            direction.len(),
            local_size,
            "boundary direction size mismatch"
        );
        assert_eq!(out.len(), local_size, "boundary action size mismatch");
        for equation in 0..nf {
            for ti in 0..n {
                let mut acc = 0.0;
                for unknown in 0..nf {
                    for si in 0..n {
                        for q in 0..ctx.npts {
                            acc += ctx.wts[q]
                                * ctx.jfacet_det[q]
                                * self.jacobian_integrand(ctx, state, equation, unknown, q, ti, si)
                                * direction[unknown * n + si];
                        }
                    }
                }
                out[equation * n + ti] = acc;
            }
        }
    }

    /// Assemble a field-major facet residual vector.
    ///
    /// # Arguments
    /// * `ctx` - facet context with `ndofs` trace basis functions.
    /// * `state` - interpolated facet state with `input_nfields()` fields.
    /// * `out` - vector of length `output_nfields()*ndofs` with layout
    ///   `out[equation*n + test_i]`. Overwritten.
    fn assemble_local_residual(&self, ctx: &FacetCtx, state: &CellState, out: &mut [f64]) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        assert!(
            nf > 0,
            "state boundary form must contain at least one field"
        );
        assert_eq!(state.nfields, nf, "boundary/state field count mismatch");
        assert_eq!(out.len(), nf * n, "local boundary residual size mismatch");
        for equation in 0..nf {
            for ti in 0..n {
                let mut acc = 0.0;
                for q in 0..ctx.npts {
                    acc += ctx.wts[q]
                        * ctx.jfacet_det[q]
                        * self.residual_integrand(ctx, state, equation, q, ti);
                }
                out[equation * n + ti] = acc;
            }
        }
    }

    /// Assemble a field-major facet Jacobian matrix.
    ///
    /// # Arguments
    /// * `ctx` - facet context with `ndofs` trace basis functions.
    /// * `state` - linearization point with `input_nfields()` fields.
    /// * `out` - matrix of length `(output_nfields()*n) * (input_nfields()*n)`,
    ///   row-major with `row = equation*n + test_i`,
    ///   `col = unknown*n + trial_i`. Overwritten.
    fn assemble_local_jacobian(&self, ctx: &FacetCtx, state: &CellState, out: &mut [f64]) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        let local_size = nf * n;
        assert_eq!(state.nfields, nf, "boundary/state field count mismatch");
        assert_eq!(
            out.len(),
            local_size * local_size,
            "local boundary Jacobian size mismatch"
        );
        for equation in 0..nf {
            for unknown in 0..nf {
                for ti in 0..n {
                    for si in 0..n {
                        let mut acc = 0.0;
                        for q in 0..ctx.npts {
                            acc += ctx.wts[q]
                                * ctx.jfacet_det[q]
                                * self.jacobian_integrand(ctx, state, equation, unknown, q, ti, si);
                        }
                        let row = equation * n + ti;
                        let col = unknown * n + si;
                        out[row * local_size + col] = acc;
                    }
                }
            }
        }
    }
}

/// Statically dispatched tensor-product state boundary kernel.
///
/// # Type parameters
/// * `GDIM` - geometric dimension of the physical space: `1` for the 1D
///   interval, `2` for the 2D quadrilateral. Must match the volume kernel's
///   `GDIM` and the facet context it is assembled with.
///
/// `GDIM` is part of the type so tensor boundary assembly can select this
/// interface without probing the weak [`StateBoundaryIntegrator`] API.
pub trait StateTensorBoundaryIntegrator<const GDIM: usize>: Send + Sync {
    /// Number of scalar equation/unknown fields in this boundary form.
    ///
    /// # Returns
    /// Field count used when input/output counts are not overridden.
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the form.
    ///
    /// # Returns
    /// `None` for anonymous forms, otherwise one name per field.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    /// Number of compact state fields consumed by this term.
    ///
    /// # Returns
    /// Input field count; must match `state.nfields` at assembly.
    fn input_nfields(&self) -> usize {
        self.nfields()
    }

    /// Number of compact residual fields produced by this term.
    ///
    /// # Returns
    /// Output equation count; the assembler scatters one trace per equation.
    fn output_nfields(&self) -> usize {
        self.nfields()
    }

    /// Optional ordered names for compact state fields.
    ///
    /// # Returns
    /// `None` for positional fields, otherwise one name per input field.
    fn input_field_names(&self) -> Option<Vec<String>> {
        self.field_names()
    }

    /// Optional ordered names for compact residual fields.
    ///
    /// # Returns
    /// `None` for positional fields, otherwise one name per output field.
    fn output_field_names(&self) -> Option<Vec<String>> {
        self.field_names()
    }

    /// Whether evaluating this boundary condition needs state gradients.
    ///
    /// # Returns
    /// `true` if [`tensor_residual`](Self::tensor_residual) or
    /// [`tensor_jacobian_action`](Self::tensor_jacobian_action) reads
    /// `state.grad(..)`; the assembler skips gradient interpolation otherwise.
    fn tensor_requires_gradients(&self) -> bool {
        false
    }

    /// Return the pointwise trace residual for one equation.
    ///
    /// Returns the scalar boundary flux `g` such that the weak facet integrand
    /// for a trace test function `v` is `g * v(q)`. The assembler forms
    /// `sum_q wts[q] * jfacet_det[q] * g * phi_trace[q]`; the kernel must NOT
    /// include weights, facet determinants, or basis values. This is the facet
    /// analogue of the volume `f0` slot (there is no `f1` flux slot because the
    /// trace has no volume gradient to contract).
    ///
    /// # Arguments
    /// * `ctx` - tensor facet context (time, facet metadata, points in
    ///   `ctx.points`, outward unit normal in `ctx.normal`).
    /// * `state` - interpolated facet state; read with `state.value(field, q)`
    ///   and, if `tensor_requires_gradients()` is true, `state.grad(field, q, d)`.
    /// * `equation` - output equation index in `0..output_nfields()`.
    /// * `q` - facet quadrature-point index in `0..ctx.npts`.
    ///
    /// # Returns
    /// Scalar trace flux `g` at `q` (e.g. a prescribed traction component,
    /// `-p*n[i]`, or a nonlinear outflow flux).
    fn tensor_residual(
        &self,
        ctx: &TensorFacetCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> f64;

    /// Return the pointwise trace Jacobian action for one equation.
    ///
    /// Gateaux derivative `d/dε tensor_residual(state + ε*direction)|ε=0` at
    /// `q`, with the same scalar-trace meaning: the weak facet Jacobian action
    /// for test `v` is `dg * v(q)`.
    ///
    /// # Arguments
    /// * `ctx` - tensor facet context (same use as in `tensor_residual`).
    /// * `state` - linearization point with `input_nfields()` fields.
    /// * `direction` - Gateaux direction with `input_nfields()` fields; read
    ///   with `direction.value(_, q)` / `direction.grad(_, q, _)`.
    /// * `equation` - output equation index in `0..output_nfields()`.
    /// * `q` - facet quadrature-point index in `0..ctx.npts`.
    ///
    /// # Returns
    /// Scalar linearized trace flux `dg` at `q`.
    fn tensor_jacobian_action(
        &self,
        ctx: &TensorFacetCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> f64;
}
