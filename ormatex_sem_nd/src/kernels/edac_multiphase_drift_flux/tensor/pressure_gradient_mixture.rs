//! Mixture-density pressure-gradient tensor kernel (2D).
//!
//! Mathematics: `d_i p / rho_m(alpha)` for momentum equations 0-1; the action
//! adds `-d_i p / rho_m^2 drho_m/dalpha dalpha`. At `alpha = 0` this reduces
//! to the base `grad(p)/rho_l` term.
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config::{
    drift_field_names, DriftFlux2DConfig, ALPHA_2D,
};

/// Tensor mixture pressure-gradient (owns equations 0-1).
pub struct TensorDriftPressureGradient2D {
    pub config: DriftFlux2DConfig,
}
impl TensorDriftPressureGradient2D {
    pub fn new(config: DriftFlux2DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<2> for TensorDriftPressureGradient2D {
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
        [state.grad(2, q, equation) / rho, 0.0, 0.0]
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
        let grad_p = state.grad(2, q, equation);
        [
            direction.grad(2, q, equation) / rho
                - grad_p * self.config.mixture_density_derivative() * direction.value(ALPHA_2D, q)
                    / (rho * rho),
            0.0,
            0.0,
        ]
    }
}
