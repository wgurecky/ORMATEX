//! State-dependent and state-independent boundary traits.
use crate::common::{CellState, FacetCtx, TensorFacetCtx};

/// Boundary integrator trait for Neumann and Robin forms.
pub trait BoundaryIntegrator {
    /// Number of scalar equation/unknown fields in this boundary form.
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the boundary form.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    fn integrand_rhs(&self, ctx: &FacetCtx, equation: usize, q: usize, test_i: usize) -> f64;

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
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the form.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    fn input_nfields(&self) -> usize {
        self.nfields()
    }

    fn output_nfields(&self) -> usize {
        self.nfields()
    }

    fn input_field_names(&self) -> Option<Vec<String>> {
        self.field_names()
    }

    fn output_field_names(&self) -> Option<Vec<String>> {
        self.field_names()
    }

    fn residual_integrand(
        &self,
        ctx: &FacetCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64;

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
/// `GDIM` is part of the type so tensor boundary assembly can select this
/// interface without probing the weak [`StateBoundaryIntegrator`] API.
pub trait StateTensorBoundaryIntegrator<const GDIM: usize>: Send + Sync {
    /// Number of scalar equation/unknown fields in this boundary form.
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the form.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    fn input_nfields(&self) -> usize {
        self.nfields()
    }

    fn output_nfields(&self) -> usize {
        self.nfields()
    }

    fn input_field_names(&self) -> Option<Vec<String>> {
        self.field_names()
    }

    fn output_field_names(&self) -> Option<Vec<String>> {
        self.field_names()
    }

    /// Whether evaluating this boundary condition needs state gradients.
    fn tensor_requires_gradients(&self) -> bool {
        false
    }

    /// Return the pointwise trace residual for one equation.
    fn tensor_residual(
        &self,
        ctx: &TensorFacetCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> f64;

    /// Return the pointwise trace Jacobian action for one equation.
    fn tensor_jacobian_action(
        &self,
        ctx: &TensorFacetCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> f64;
}
