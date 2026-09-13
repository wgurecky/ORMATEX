//! Tensor EDAC pressure diffusion `d(k dp/dx)/dx` for 1D `p`.
//!
//! Mathematics: for equation 1 the triple is `(0, k dp/dx, 0)` with `k` the
//! pressure diffusivity (artificial sound speed times the cell length); the
//! action is linear in the direction gradient. Owns equation 1 only. Weak
//! counterpart:
//! [`KernelEdacPressureDiffusion1D`](crate::kernels::edac::weak::pressure_diffusion_1d::KernelEdacPressureDiffusion1D).

use crate::common::{CellState, TensorCtx};

use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config_1d::{fluid_field_names_1d, EdacNavierStokes1DConfig};

/// Tensor pressure-diffusion kernel (sum-factorized counterpart, owns equation 1).
pub struct TensorKernelEdacPressureDiffusion1D {
    pub config: EdacNavierStokes1DConfig,
}

impl TensorKernelEdacPressureDiffusion1D {
    pub fn new(config: EdacNavierStokes1DConfig) -> Self {
        Self { config }
    }
}

impl TensorResidualKernel<1> for TensorKernelEdacPressureDiffusion1D {
    fn nfields(&self) -> usize {
        2
    }
    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names_1d()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation == 1
    }
    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation != 1 {
            [0.0; 3]
        } else {
            let k = self.config.pressure_diffusivity_tensor(ctx);
            [0.0, k * state.grad(1, q, 0), 0.0]
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
        if equation != 1 {
            [0.0; 3]
        } else {
            let k = self.config.pressure_diffusivity_tensor(ctx);
            [0.0, k * direction.grad(1, q, 0), 0.0]
        }
    }
}
