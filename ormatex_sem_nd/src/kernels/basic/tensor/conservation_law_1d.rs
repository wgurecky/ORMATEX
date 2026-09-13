use crate::common::{CellState, TensorCtx};

use crate::kernels::common::{FluxKernel1D, TensorResidualKernel};

use crate::kernels::basic::weak::conservation_law_1d::KernelConservationLaw1D;

/// Tensor generic Galerkin flux for `U_t + dF(U)/dx = 0`.
///
/// Mathematics: the triple is `(0, -F(U), 0)` evaluated through the facet
/// `local_flux_context()`; the action contracts `dF/dU` with the direction
/// values. Generic over `FluxKernel1D`. Weak counterpart:
/// [`KernelConservationLaw1D`].
/// Tensor-product 1D conservation-law kernel (sum-factorized counterpart).
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
