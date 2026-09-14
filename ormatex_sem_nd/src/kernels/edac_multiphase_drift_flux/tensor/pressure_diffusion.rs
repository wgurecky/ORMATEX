//! Mixture EDAC pressure diffusion `div(k grad(p))` (2D).
//!
//! Mathematics: `(0, k d_0 p, k d_1 p)` for equation 2 with `k` the pressure
//! diffusivity (artificial sound speed times the Smagorinsky filter width),
//! identical to the base kernel but on the four-field state.
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config::{drift_field_names, DriftFlux2DConfig};

/// Tensor mixture pressure-diffusion (owns equation 2).
pub struct TensorDriftPressureDiffusion2D {
    pub config: DriftFlux2DConfig,
}
impl TensorDriftPressureDiffusion2D {
    pub fn new(config: DriftFlux2DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<2> for TensorDriftPressureDiffusion2D {
    fn nfields(&self) -> usize {
        4
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names()
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
            return [0.0; 3];
        }
        let k = self.config.pressure_diffusivity_tensor(ctx);
        [0.0, k * state.grad(2, q, 0), k * state.grad(2, q, 1)]
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
            return [0.0; 3];
        }
        let k = self.config.pressure_diffusivity_tensor(ctx);
        [
            0.0,
            k * direction.grad(2, q, 0),
            k * direction.grad(2, q, 1),
        ]
    }
}
