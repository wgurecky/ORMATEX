//! Split-form void-fraction advection tensor kernel (1D).
//!
//! Mathematics: `1/2 C0 (u dalpha/dx, -u alpha, 0)` for equation 2; owns
//! equation 2 only.
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config_1d::{
    drift_field_names_1d, DriftFlux1DConfig, ALPHA_1D,
};

/// Tensor split void advection (owns equation 2).
pub struct TensorDriftVoidAdvectionSplit1D {
    pub config: DriftFlux1DConfig,
}
impl TensorDriftVoidAdvectionSplit1D {
    pub fn new(config: DriftFlux1DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<1> for TensorDriftVoidAdvectionSplit1D {
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
        let c0 = self.config.distribution_parameter();
        let u = state.value(0, q);
        let a = state.value(ALPHA_1D, q);
        [
            0.5 * c0 * u * state.grad(ALPHA_1D, q, 0),
            -0.5 * c0 * u * a,
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
        if equation != ALPHA_1D {
            return [0.0; 3];
        }
        let c0 = self.config.distribution_parameter();
        let u = state.value(0, q);
        let du = direction.value(0, q);
        let a = state.value(ALPHA_1D, q);
        let da = direction.value(ALPHA_1D, q);
        [
            0.5 * c0 * (du * state.grad(ALPHA_1D, q, 0) + u * direction.grad(ALPHA_1D, q, 0)),
            -0.5 * c0 * (du * a + u * da),
            0.0,
        ]
    }
}
