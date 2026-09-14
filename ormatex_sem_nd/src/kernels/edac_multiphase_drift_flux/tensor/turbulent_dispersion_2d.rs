//! Void-fraction turbulent-dispersion tensor kernel (2D).
//!
//! Mathematics: Fickian flux `div(D_td grad(alpha))` for equation 3 with the
//! simplest constant coefficient `D_td` (Burns et al. 2004 Favre-averaged
//! drag / Lopez de Bertodano turbulent-dispersion flux in its
//! constant-coefficient limit; see [`super::super::closures`]). Spreads the
//! dispersed bubbles; linear action. Owns equation 3 only.
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config::{
    drift_field_names, DriftFlux2DConfig, ALPHA_2D,
};

/// Tensor void turbulent dispersion (owns equation 3).
pub struct TensorDriftTurbulentDispersion2D {
    pub config: DriftFlux2DConfig,
    pub dispersion: f64,
}
impl TensorDriftTurbulentDispersion2D {
    pub fn new(config: DriftFlux2DConfig, dispersion: f64) -> Self {
        assert!(
            dispersion.is_finite() && dispersion >= 0.0,
            "dispersion coefficient must be finite and nonnegative"
        );
        Self { config, dispersion }
    }
}
impl TensorResidualKernel<2> for TensorDriftTurbulentDispersion2D {
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
        [
            0.0,
            self.dispersion * state.grad(ALPHA_2D, q, 0),
            self.dispersion * state.grad(ALPHA_2D, q, 1),
        ]
    }
    fn tensor_jacobian_action(
        &self,
        _: &TensorCtx<'_>,
        _: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation != ALPHA_2D {
            return [0.0; 3];
        }
        [
            0.0,
            self.dispersion * direction.grad(ALPHA_2D, q, 0),
            self.dispersion * direction.grad(ALPHA_2D, q, 1),
        ]
    }
}
