//! Mixture-density pressure-divergence tensor kernel (2D).
//!
//! Mathematics: modified EDAC continuity `rho_m(alpha) c0^2 div(u_m)` for
//! equation 2; the action adds the `drho_m/dalpha` product. At `alpha = 0`
//! this reduces to the base `rho_l c0^2 div(u)` term.
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config::{
    drift_field_names, DriftFlux2DConfig, ALPHA_2D,
};

/// Tensor mixture pressure-divergence (owns equation 2).
pub struct TensorDriftPressureDivergence2D {
    pub config: DriftFlux2DConfig,
}
impl TensorDriftPressureDivergence2D {
    pub fn new(config: DriftFlux2DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<2> for TensorDriftPressureDivergence2D {
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
        let rho = self.config.mixture_density(state.value(ALPHA_2D, q));
        [
            rho * self.config.c0 * self.config.c0 * (state.grad(0, q, 0) + state.grad(1, q, 1)),
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
        if equation != 2 {
            return [0.0; 3];
        }
        let c02 = self.config.c0 * self.config.c0;
        let rho = self.config.mixture_density(state.value(ALPHA_2D, q));
        let div_u = state.grad(0, q, 0) + state.grad(1, q, 1);
        let div_du = direction.grad(0, q, 0) + direction.grad(1, q, 1);
        [
            c02 * (self.config.mixture_density_derivative() * direction.value(ALPHA_2D, q) * div_u
                + rho * div_du),
            0.0,
            0.0,
        ]
    }
}
