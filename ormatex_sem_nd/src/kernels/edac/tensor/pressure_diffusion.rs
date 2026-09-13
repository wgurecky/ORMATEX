//! Tensor EDAC pressure diffusion `div(k grad(p))` for `p`.
//!
//! Mathematics: for equation 2 the triple is `(0, k d_0 p, k d_1 p)` with
//! `k` the pressure diffusivity (artificial sound speed times the
//! Smagorinsky filter width); the action is linear in the direction
//! gradient. Owns equation 2 only. Weak counterpart:
//! [`KernelEdacPressureDiffusion2D`](crate::kernels::edac::weak::pressure_diffusion::KernelEdacPressureDiffusion2D).

use crate::common::{CellState, TensorCtx};

use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config::{fluid_field_names, EdacNavierStokes2DConfig};

/// Tensor pressure-diffusion kernel (sum-factorized counterpart, owns equation 2).
pub struct TensorKernelEdacPressureDiffusion2D {
    pub config: EdacNavierStokes2DConfig,
}

impl TensorKernelEdacPressureDiffusion2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl TensorResidualKernel<2> for TensorKernelEdacPressureDiffusion2D {
    fn nfields(&self) -> usize {
        3
    }
    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation == 2
    }
    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation != 2 {
            [0.0; 3]
        } else {
            let k = self.config.pressure_diffusivity_tensor(ctx);
            [0.0, k * state.grad(2, q, 0), k * state.grad(2, q, 1)]
        }
    }
    fn tensor_jacobian_action(
        &self,
        ctx: &TensorCtx<'_>,
        _: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation != 2 {
            [0.0; 3]
        } else {
            let k = self.config.pressure_diffusivity_tensor(ctx);
            [
                0.0,
                k * direction.grad(2, q, 0),
                k * direction.grad(2, q, 1),
            ]
        }
    }
}
