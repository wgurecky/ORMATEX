//! Split-form mixture pressure-advection tensor kernel (1D).
//!
//! Mathematics: `1/2 (u dp/dx, -u p, 0)` for equation 1; owns equation 1 only.
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config_1d::{
    drift_field_names_1d, DriftFlux1DConfig,
};

/// Tensor split mixture pressure-advection (owns equation 1).
pub struct TensorDriftPressureAdvectionSplit1D {
    pub config: DriftFlux1DConfig,
}
impl TensorDriftPressureAdvectionSplit1D {
    pub fn new(config: DriftFlux1DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<1> for TensorDriftPressureAdvectionSplit1D {
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
        let u = state.value(0, q);
        let p = state.value(1, q);
        [0.5 * u * state.grad(1, q, 0), -0.5 * u * p, 0.0]
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
        let u = state.value(0, q);
        let du = direction.value(0, q);
        let p = state.value(1, q);
        let dp = direction.value(1, q);
        [
            0.5 * (du * state.grad(1, q, 0) + u * direction.grad(1, q, 0)),
            -0.5 * (du * p + u * dp),
            0.0,
        ]
    }
}
