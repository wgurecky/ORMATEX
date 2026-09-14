//! Buoyant gravity body-force tensor kernel (2D).
//!
//! Mathematics: rectangular-equivalent source `(rho_m - rho_l)/rho_m * g_i`
//! for momentum equations 0-1. The buoyant form vanishes at `alpha = 0` so
//! the low-void limit recovers the gravity-free base EDAC solution; the
//! action carries the exact `drho_m/dalpha` linearization. Owns equations 0-1.
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config::{
    drift_field_names, DriftFlux2DConfig, ALPHA_2D,
};

/// Tensor buoyant gravity source (owns equations 0-1).
pub struct TensorDriftGravity2D {
    pub config: DriftFlux2DConfig,
}
impl TensorDriftGravity2D {
    pub fn new(config: DriftFlux2DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<2> for TensorDriftGravity2D {
    fn nfields(&self) -> usize {
        4
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation < 2
    }
    fn tensor_residual(
        &self,
        _: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation >= 2 {
            return [0.0; 3];
        }
        let rho = self.config.mixture_density(state.value(ALPHA_2D, q));
        [
            (rho - self.config.rho_l) / rho * self.config.gravity[equation],
            0.0,
            0.0,
        ]
    }
    fn tensor_jacobian_action(
        &self,
        _: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation >= 2 {
            return [0.0; 3];
        }
        let rho = self.config.mixture_density(state.value(ALPHA_2D, q));
        let drho = self.config.mixture_density_derivative();
        // d/dalpha [(rho - rho_l)/rho] = drho * rho_l / rho^2.
        let factor = drho * self.config.rho_l / (rho * rho) * self.config.gravity[equation];
        [factor * direction.value(ALPHA_2D, q), 0.0, 0.0]
    }
}
