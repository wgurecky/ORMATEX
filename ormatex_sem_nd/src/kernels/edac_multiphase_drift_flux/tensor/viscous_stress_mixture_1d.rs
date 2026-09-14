//! Mixture viscous stress `d(tau_m)/dx` for the 1D momentum equation.
//!
//! Mathematics: `(0, tau_m, 0)` with laminar `tau_m = 2 nu_m(alpha) du/dx`;
//! the action adds the `dnu_m/dalpha` product. Owns equation 0 only.
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config_1d::{
    drift_field_names_1d, DriftFlux1DConfig,
};

/// Tensor mixture viscous-stress (owns equation 0).
pub struct TensorDriftViscousStress1D {
    pub config: DriftFlux1DConfig,
}
impl TensorDriftViscousStress1D {
    pub fn new(config: DriftFlux1DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<1> for TensorDriftViscousStress1D {
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
        [0.0, self.config.stress_tensor(state, q), 0.0]
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
        [
            0.0,
            self.config
                .stress_tensor_directional_derivative(state, direction, q),
            0.0,
        ]
    }
}
