//! Core volume kernel traits.
//!
//! `BilinearForm` / `LinearForm` describe state-independent forms;
//! `ResidualKernel` / `TensorResidualKernel` describe state-dependent
//! residuals for weak and sum-factorized assembly. Boundary traits live
//! in [`boundary_traits`](super::boundary_traits).
use crate::common::{CellState, LocalCtx, TensorCtx};


/// Per-cell bilinear-form kernel.
pub trait BilinearForm {
    /// Number of scalar equation/unknown fields in this form.
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the form.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    /// Bare integrand for one equation/unknown block and test/trial pair.
    fn integrand(
        &self,
        ctx: &LocalCtx,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64;

    /// Whether this form implements tensor-product bilinear evaluation in its
    /// supported geometric dimension.
    fn supports_tensor_bilinear(&self) -> bool {
        false
    }

    /// Whether this form implements tensor-product bilinear evaluation on an
    /// interval. This is separate from the 2D capability because a kernel may
    /// support only one geometric dimension.
    fn supports_tensor_bilinear_1d(&self) -> bool {
        false
    }

    /// Return the pointwise action of one trial basis function.
    ///
    /// The returned values are `(f0, f1_x, f1_y)` for
    /// `f0 * test + f1 . grad(test)`. In 1D, only `f0` and `f1_x` are used.
    /// `trial_value` and `trial_grad` are the value and physical gradient of
    /// the trial basis at `q`.
    fn tensor_bilinear(
        &self,
        _ctx: &TensorCtx<'_>,
        _equation: usize,
        _unknown: usize,
        _q: usize,
        _trial_value: f64,
        _trial_grad: [f64; 2],
    ) -> [f64; 3] {
        unreachable!("bilinear form does not support tensor-product evaluation")
    }

    /// Assemble a field-major local block matrix.
    fn assemble_local(&self, ctx: &LocalCtx, out: &mut [f64]) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        let local_size = nf * n;
        assert!(nf > 0, "bilinear form must contain at least one field");
        assert_eq!(
            out.len(),
            local_size * local_size,
            "local matrix size mismatch"
        );
        for equation in 0..nf {
            for unknown in 0..nf {
                for ti in 0..n {
                    for si in 0..n {
                        let mut acc = 0.0;
                        for q in 0..ctx.npts {
                            acc += ctx.wts[q]
                                * ctx.jdets[q]
                                * self.integrand(ctx, equation, unknown, q, ti, si);
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

/// Per-cell linear-form kernel (RHS contribution).
pub trait LinearForm {
    /// Number of scalar equation fields in this form.
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the form.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    /// Bare integrand for one equation and test dof.
    fn integrand(&self, ctx: &LocalCtx, equation: usize, q: usize, test_i: usize) -> f64;

    /// Assemble a field-major local RHS.
    fn assemble_local_rhs(&self, ctx: &LocalCtx, out: &mut [f64]) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        assert!(nf > 0, "linear form must contain at least one field");
        assert_eq!(out.len(), nf * n, "local RHS size mismatch");
        for equation in 0..nf {
            for ti in 0..n {
                let mut acc = 0.0;
                for q in 0..ctx.npts {
                    acc += ctx.wts[q] * ctx.jdets[q] * self.integrand(ctx, equation, q, ti);
                }
                out[equation * n + ti] = acc;
            }
        }
    }
}

/// State-aware cell kernel for residual and Jacobian assembly.
pub trait ResidualKernel {
    /// Number of scalar PDE fields/equations in this kernel.
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the kernel.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    /// Number of compact state fields consumed by this term.
    fn input_nfields(&self) -> usize {
        self.nfields()
    }

    /// Number of compact residual fields produced by this term.
    fn output_nfields(&self) -> usize {
        self.nfields()
    }

    /// Optional ordered names for compact state fields.
    fn input_field_names(&self) -> Option<Vec<String>> {
        self.field_names()
    }

    /// Optional ordered names for compact residual fields.
    fn output_field_names(&self) -> Option<Vec<String>> {
        self.field_names()
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64;

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64;

    fn assemble_local_residual(&self, ctx: &LocalCtx, state: &CellState, out: &mut [f64]) {
        let ni = self.input_nfields();
        let no = self.output_nfields();
        let n = ctx.ndofs;
        assert_eq!(state.nfields, ni, "kernel/state field count mismatch");
        assert_eq!(out.len(), no * n, "local residual size mismatch");
        for equation in 0..no {
            for ti in 0..n {
                let mut acc = 0.0;
                for q in 0..ctx.npts {
                    acc += ctx.wts[q]
                        * ctx.jdets[q]
                        * self.residual_integrand(ctx, state, equation, q, ti);
                }
                out[equation * n + ti] = acc;
            }
        }
    }

    fn assemble_local_jacobian(&self, ctx: &LocalCtx, state: &CellState, out: &mut [f64]) {
        let ni = self.input_nfields();
        let no = self.output_nfields();
        let n = ctx.ndofs;
        let row_size = no * n;
        let col_size = ni * n;
        assert_eq!(state.nfields, ni, "kernel/state field count mismatch");
        assert_eq!(
            out.len(),
            row_size * col_size,
            "local Jacobian size mismatch"
        );
        for equation in 0..no {
            for unknown in 0..ni {
                for ti in 0..n {
                    for si in 0..n {
                        let mut acc = 0.0;
                        for q in 0..ctx.npts {
                            acc += ctx.wts[q]
                                * ctx.jdets[q]
                                * self.jacobian_integrand(ctx, state, equation, unknown, q, ti, si);
                        }
                        let row = equation * n + ti;
                        let col = unknown * n + si;
                        out[row * col_size + col] = acc;
                    }
                }
            }
        }
    }

    fn apply_local_jacobian(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        direction: &[f64],
        out: &mut [f64],
    ) {
        let ni = self.input_nfields();
        let no = self.output_nfields();
        let n = ctx.ndofs;
        let input_size = ni * n;
        let output_size = no * n;
        assert_eq!(state.nfields, ni, "kernel/state field count mismatch");
        assert_eq!(direction.len(), input_size, "local direction size mismatch");
        assert_eq!(
            out.len(),
            output_size,
            "local Jacobian action size mismatch"
        );
        for equation in 0..no {
            for ti in 0..n {
                let mut acc = 0.0;
                for unknown in 0..ni {
                    for si in 0..n {
                        for q in 0..ctx.npts {
                            acc += ctx.wts[q]
                                * ctx.jdets[q]
                                * self.jacobian_integrand(ctx, state, equation, unknown, q, ti, si)
                                * direction[unknown * n + si];
                        }
                    }
                }
                out[equation * n + ti] = acc;
            }
        }
    }
}

/// Statically dispatched tensor-product residual kernel.
///
/// `GDIM` is part of the type, so tensor assembly never needs a capability
/// query to select its volume path.
pub trait TensorResidualKernel<const GDIM: usize>: Send + Sync {
    fn nfields(&self) -> usize {
        1
    }
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
    /// Whether this kernel contributes to `equation`.
    ///
    /// Kernels that own only an equation subset (e.g. momentum-only terms of
    /// a coupled system) override this so fused sums can skip zero blocks
    /// without per-point calls. The default keeps existing kernels exact.
    fn owns_equation(&self, _equation: usize) -> bool {
        true
    }
    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3];
    fn tensor_jacobian_action(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3];

    /// Batched pointwise residual across element lanes (SIMD-over-element hook).
    ///
    /// Default loops over lanes calling the scalar path (correct fallback).
    /// Built-in kernels may override with lane-vectorized physics.
    /// `ctxs/states` have one entry per lane; outputs are per-lane scalars.
    fn tensor_residual_batch(
        &self,
        ctxs: &[TensorCtx<'_>],
        states: &[CellState<'_>],
        equation: usize,
        q: usize,
        f0: &mut [f64],
        f1x: &mut [f64],
        f1y: &mut [f64],
    ) {
        debug_assert_eq!(ctxs.len(), states.len());
        debug_assert_eq!(f0.len(), ctxs.len());
        for (i, (ctx, state)) in ctxs.iter().zip(states.iter()).enumerate() {
            let [a, b, c] = self.tensor_residual(ctx, state, equation, q);
            f0[i] = a;
            f1x[i] = b;
            f1y[i] = c;
        }
    }

    /// Batched pointwise Jacobian action across element lanes.
    fn tensor_jacobian_action_batch(
        &self,
        ctxs: &[TensorCtx<'_>],
        states: &[CellState<'_>],
        directions: &[CellState<'_>],
        equation: usize,
        q: usize,
        f0: &mut [f64],
        f1x: &mut [f64],
        f1y: &mut [f64],
    ) {
        debug_assert_eq!(ctxs.len(), states.len());
        debug_assert_eq!(ctxs.len(), directions.len());
        for (i, ((ctx, state), dir)) in
            ctxs.iter().zip(states.iter()).zip(directions.iter()).enumerate()
        {
            let [a, b, c] = self.tensor_jacobian_action(ctx, state, dir, equation, q);
            f0[i] = a;
            f1x[i] = b;
            f1y[i] = c;
        }
    }
}
pub trait FluxKernel1D {
    fn nfields(&self) -> usize;

    /// Optional ordered names for fields whose meaning is part of the flux.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    fn flux(&self, ctx: &LocalCtx, state: &CellState, equation: usize, q: usize) -> f64;

    fn flux_jacobian(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
    ) -> f64;
}
