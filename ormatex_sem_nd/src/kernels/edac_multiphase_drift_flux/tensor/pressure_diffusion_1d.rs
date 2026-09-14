//! Mixture EDAC pressure diffusion `d(k dp/dx)/dx` (1D).
//!
//! Mathematics: `(0, k dp/dx, 0)` for equation 1 with `k` from the cell
//! length; owns equation 1 only.
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config_1d::{
    drift_field_names_1d, DriftFlux1DConfig,
};

/// Tensor mixture pressure-diffusion (owns equation 1).
pub struct TensorDriftPressureDiffusion1D {
    pub config: DriftFlux1DConfig,
}
impl TensorDriftPressureDiffusion1D {
    pub fn new(config: DriftFlux1DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<1> for TensorDriftPressureDiffusion1D {
    fn nfields(&self) -> usize {
        3
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names_1d()
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
            return [0.0; 3];
        }
        let k = self.config.pressure_diffusivity_tensor(ctx);
        [0.0, k * state.grad(1, q, 0), 0.0]
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
            return [0.0; 3];
        }
        let k = self.config.pressure_diffusivity_tensor(ctx);
        [0.0, k * direction.grad(1, q, 0), 0.0]
    }
}
