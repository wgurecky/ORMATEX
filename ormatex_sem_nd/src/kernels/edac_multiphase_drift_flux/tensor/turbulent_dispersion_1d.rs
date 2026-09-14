//! Void-fraction turbulent-dispersion tensor kernel (1D).
//!
//! Mathematics: `d(D_td dalpha/dx)/dx` for equation 2 with the simplest
//! constant coefficient `D_td` (same Fickian form as the 2D kernel; see
//! [`super::super::closures`]). Besides spreading the dispersed bubbles it
//! regularizes the split-form void outflow, for which the 1D tensor path
//! provides no facet closure. Owns equation 2 only.
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config_1d::{
    drift_field_names_1d, DriftFlux1DConfig, ALPHA_1D,
};

/// Tensor void turbulent dispersion (owns equation 2).
pub struct TensorDriftTurbulentDispersion1D {
    pub config: DriftFlux1DConfig,
    pub dispersion: f64,
}
impl TensorDriftTurbulentDispersion1D {
    pub fn new(config: DriftFlux1DConfig, dispersion: f64) -> Self {
        assert!(
            dispersion.is_finite() && dispersion >= 0.0,
            "dispersion coefficient must be finite and nonnegative"
        );
        Self { config, dispersion }
    }
}
impl TensorResidualKernel<1> for TensorDriftTurbulentDispersion1D {
    fn nfields(&self) -> usize {
        3
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names_1d()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation == ALPHA_1D
    }
    fn tensor_residual(
        &self,
        _: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation != ALPHA_1D {
            return [0.0; 3];
        }
        [0.0, self.dispersion * state.grad(ALPHA_1D, q, 0), 0.0]
    }
    fn tensor_jacobian_action(
        &self,
        _: &TensorCtx<'_>,
        _: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation != ALPHA_1D {
            return [0.0; 3];
        }
        [0.0, self.dispersion * direction.grad(ALPHA_1D, q, 0), 0.0]
    }
}
