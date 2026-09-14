//! Mixture-density pressure-divergence tensor kernel (1D).
//!
//! Mathematics: `rho_m(alpha) c0^2 du/dx` for equation 1 with the
//! `drho_m/dalpha` action term; owns equation 1 only.
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config_1d::{
    drift_field_names_1d, DriftFlux1DConfig, ALPHA_1D,
};

/// Tensor mixture pressure-divergence (owns equation 1).
pub struct TensorDriftPressureDivergence1D {
    pub config: DriftFlux1DConfig,
}
impl TensorDriftPressureDivergence1D {
    pub fn new(config: DriftFlux1DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<1> for TensorDriftPressureDivergence1D {
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
        _: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation != 1 {
            return [0.0; 3];
        }
        let rho = self.config.mixture_density(state.value(ALPHA_1D, q));
        [
            rho * self.config.c0 * self.config.c0 * state.grad(0, q, 0),
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
        if equation != 1 {
            return [0.0; 3];
        }
        let c02 = self.config.c0 * self.config.c0;
        let rho = self.config.mixture_density(state.value(ALPHA_1D, q));
        [
            c02 * (self.config.mixture_density_derivative()
                * direction.value(ALPHA_1D, q)
                * state.grad(0, q, 0)
                + rho * direction.grad(0, q, 0)),
            0.0,
            0.0,
        ]
    }
}
