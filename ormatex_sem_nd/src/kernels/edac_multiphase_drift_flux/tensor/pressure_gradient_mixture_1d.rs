//! Mixture-density pressure-gradient tensor kernel (1D).
//!
//! Mathematics: `dp/dx / rho_m(alpha)` for equation 0 with the
//! `-dp/dx / rho_m^2 drho_m/dalpha dalpha` action term.
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config_1d::{
    drift_field_names_1d, DriftFlux1DConfig, ALPHA_1D,
};

/// Tensor mixture pressure-gradient (owns equation 0).
pub struct TensorDriftPressureGradient1D {
    pub config: DriftFlux1DConfig,
}
impl TensorDriftPressureGradient1D {
    pub fn new(config: DriftFlux1DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<1> for TensorDriftPressureGradient1D {
    fn nfields(&self) -> usize {
        3
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names_1d()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation == 0
    }
    fn tensor_residual(
        &self,
        _: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation != 0 {
            return [0.0; 3];
        }
        let rho = self.config.mixture_density(state.value(ALPHA_1D, q));
        [state.grad(1, q, 0) / rho, 0.0, 0.0]
    }
    fn tensor_jacobian_action(
        &self,
        _: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation != 0 {
            return [0.0; 3];
        }
        let rho = self.config.mixture_density(state.value(ALPHA_1D, q));
        let grad_p = state.grad(1, q, 0);
        [
            direction.grad(1, q, 0) / rho
                - grad_p * self.config.mixture_density_derivative() * direction.value(ALPHA_1D, q)
                    / (rho * rho),
            0.0,
            0.0,
        ]
    }
}
