use crate::common::{CellState, LocalCtx, TensorCtx};

use super::kernel_common::{FluxKernel1D, ResidualKernel, TensorResidualKernel};

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

/// Tensor-product 1D conservation-law kernel.
pub struct TensorKernelConservationLaw1D<F>(pub KernelConservationLaw1D<F>);

impl<F> TensorKernelConservationLaw1D<F> {
    pub fn new(flux: F) -> Self {
        Self(KernelConservationLaw1D::new(flux))
    }
}

impl<F: FluxKernel1D + Send + Sync> TensorResidualKernel<1> for TensorKernelConservationLaw1D<F> {
    fn nfields(&self) -> usize {
        self.0.flux.nfields()
    }

    fn field_names(&self) -> Option<Vec<String>> {
        self.0.flux.field_names()
    }

    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let local_ctx = ctx.local_flux_context();
        [0.0, -self.0.flux.flux(&local_ctx, state, equation, q), 0.0]
    }

    fn tensor_jacobian_action(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let local_ctx = ctx.local_flux_context();
        let flux_action = (0..self.0.flux.nfields())
            .map(|unknown| {
                self.0
                    .flux
                    .flux_jacobian(&local_ctx, state, equation, unknown, q)
                    * direction.value(unknown, q)
            })
            .sum::<f64>();
        [0.0, -flux_action, 0.0]
    }
}
