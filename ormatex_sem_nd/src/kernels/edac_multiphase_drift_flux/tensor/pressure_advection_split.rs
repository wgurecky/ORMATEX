//! Split-form mixture pressure-advection tensor kernel (2D).
//!
//! Mathematics: `1/2 (u_j d_j p, -u_j p, ...)` for equation 2 on the mixture
//! velocity; owns equation 2 only. Needs the drift split-flux boundary.
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config::{
    drift_field_names, velocity, DriftFlux2DConfig,
};

/// Tensor split mixture pressure-advection (owns equation 2).
pub struct TensorDriftPressureAdvectionSplit2D {
    pub config: DriftFlux2DConfig,
}
impl TensorDriftPressureAdvectionSplit2D {
    pub fn new(config: DriftFlux2DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<2> for TensorDriftPressureAdvectionSplit2D {
    fn nfields(&self) -> usize {
        4
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation == 2
    }
    fn tensor_residual(
        &self,
        _: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation != 2 {
            return [0.0; 3];
        }
        let u = velocity(state, q);
        let p = state.value(2, q);
        [
            0.5 * (u[0] * state.grad(2, q, 0) + u[1] * state.grad(2, q, 1)),
            -0.5 * u[0] * p,
            -0.5 * u[1] * p,
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
        if equation != 2 {
            return [0.0; 3];
        }
        let u = velocity(state, q);
        let du = [direction.value(0, q), direction.value(1, q)];
        let p = state.value(2, q);
        let dp = direction.value(2, q);
        [
            0.5 * (du[0] * state.grad(2, q, 0)
                + du[1] * state.grad(2, q, 1)
                + u[0] * direction.grad(2, q, 0)
                + u[1] * direction.grad(2, q, 1)),
            -0.5 * (du[0] * p + u[0] * dp),
            -0.5 * (du[1] * p + u[1] * dp),
        ]
    }
}
