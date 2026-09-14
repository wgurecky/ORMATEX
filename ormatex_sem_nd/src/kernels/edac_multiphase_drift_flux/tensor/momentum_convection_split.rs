//! Split-form mixture-momentum convection tensor kernel (2D).
//!
//! Mathematics: identical split form to the base EDAC kernel,
//! `1/2 (u_j d_j u_i, -u_j u_i, ...)` for equations 0-1 on the mixture
//! velocity; mixture inertia beyond the velocity form enters through the
//! mixture pressure-gradient/divergence/viscous/gravity terms. Needs the
//! drift split-flux boundary.
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config::{
    drift_field_names, velocity, DriftFlux2DConfig,
};

/// Tensor split mixture-momentum convection (owns equations 0-1).
pub struct TensorDriftMomentumConvectionSplit2D {
    pub config: DriftFlux2DConfig,
}
impl TensorDriftMomentumConvectionSplit2D {
    pub fn new(config: DriftFlux2DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<2> for TensorDriftMomentumConvectionSplit2D {
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
        let u = velocity(state, q);
        [
            0.5 * (u[0] * state.grad(equation, q, 0) + u[1] * state.grad(equation, q, 1)),
            -0.5 * u[0] * state.value(equation, q),
            -0.5 * u[1] * state.value(equation, q),
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
        if equation >= 2 {
            return [0.0; 3];
        }
        let u = velocity(state, q);
        let du = [direction.value(0, q), direction.value(1, q)];
        [
            0.5 * (du[0] * state.grad(equation, q, 0)
                + du[1] * state.grad(equation, q, 1)
                + u[0] * direction.grad(equation, q, 0)
                + u[1] * direction.grad(equation, q, 1)),
            -0.5 * (du[0] * state.value(equation, q) + u[0] * direction.value(equation, q)),
            -0.5 * (du[1] * state.value(equation, q) + u[1] * direction.value(equation, q)),
        ]
    }
}
