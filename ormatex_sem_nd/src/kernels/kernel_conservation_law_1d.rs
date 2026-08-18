use crate::common::{CellState, LocalCtx};

use super::kernel_common::{FluxKernel1D, ResidualKernel};

/// Generic Galerkin residual for `U_t + dF(U)/dx = 0`.
pub struct KernelConservationLaw1D<F> {
    pub flux: F,
}

impl<F> KernelConservationLaw1D<F> {
    pub fn new(flux: F) -> Self {
        Self { flux }
    }
}

impl<F: FluxKernel1D> ResidualKernel for KernelConservationLaw1D<F> {
    fn nfields(&self) -> usize {
        self.flux.nfields()
    }

    fn field_names(&self) -> Option<Vec<String>> {
        self.flux.field_names()
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        assert_eq!(ctx.gdim, 1, "KernelConservationLaw1D requires gdim == 1");
        -self.flux.flux(ctx, state, equation, q) * ctx.test(test_i, 0).grad(q, 0)
    }

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        assert_eq!(ctx.gdim, 1, "KernelConservationLaw1D requires gdim == 1");
        -self.flux.flux_jacobian(ctx, state, equation, unknown, q)
            * ctx.trial(trial_i, 0).v(q)
            * ctx.test(test_i, 0).grad(q, 0)
    }
}
