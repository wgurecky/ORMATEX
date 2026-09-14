//! Conservative Ishii-Zuber drift-flux tensor kernel (2D).
//!
//! Mathematics: relative gas flux `div(F(alpha) e)` for equation 3 with
//! `F = a*V_gj(a)` the hindered drift flux and `e` the rise unit vector
//! (opposite gravity; see [`super::super::closures`]). Conservative triple
//! `(0, -F e_0, -F e_1)` with the exact `dF/da` action (the flux is
//! non-monotone past `a = 4/11`, so the exact sign matters). Owns equation 3.
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config::{
    drift_field_names, DriftFlux2DConfig, ALPHA_2D,
};

/// Tensor conservative drift flux (owns equation 3).
pub struct TensorDriftFlux2D {
    pub config: DriftFlux2DConfig,
}
impl TensorDriftFlux2D {
    pub fn new(config: DriftFlux2DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<2> for TensorDriftFlux2D {
    fn nfields(&self) -> usize {
        4
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation == ALPHA_2D
    }
    fn tensor_residual(
        &self,
        _: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation != ALPHA_2D {
            return [0.0; 3];
        }
        let a = state.value(ALPHA_2D, q);
        let f = self.config.drift_flux(a);
        let e = self.config.rise_direction();
        [0.0, -f * e[0], -f * e[1]]
    }
    fn tensor_jacobian_action(
        &self,
        _: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation != ALPHA_2D {
            return [0.0; 3];
        }
        let a = state.value(ALPHA_2D, q);
        let df = self.config.drift_flux_derivative(a);
        let e = self.config.rise_direction();
        let da = direction.value(ALPHA_2D, q);
        [0.0, -df * da * e[0], -df * da * e[1]]
    }
}
