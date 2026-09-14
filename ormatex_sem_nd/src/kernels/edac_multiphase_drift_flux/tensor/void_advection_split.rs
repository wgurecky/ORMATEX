//! Split-form void-fraction advection tensor kernel (2D).
//!
//! Mathematics: gas continuity advected by the mixture velocity,
//! `1/2 C0 (u_j d_j alpha, -u_j alpha, ...)` for equation 3 with the
//! distribution parameter `C0` (Ishii 1975 `u_g = C0 j + V_gj`; here
//! `j ~= u_m`). The relative drift flux lives in the companion
//! drift-flux kernel. Needs the drift split-flux boundary.
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config::{
    drift_field_names, velocity, DriftFlux2DConfig, ALPHA_2D,
};

/// Tensor split void advection (owns equation 3).
pub struct TensorDriftVoidAdvectionSplit2D {
    pub config: DriftFlux2DConfig,
}
impl TensorDriftVoidAdvectionSplit2D {
    pub fn new(config: DriftFlux2DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<2> for TensorDriftVoidAdvectionSplit2D {
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
        let c0 = self.config.distribution_parameter();
        let u = velocity(state, q);
        let a = state.value(ALPHA_2D, q);
        [
            0.5 * c0 * (u[0] * state.grad(ALPHA_2D, q, 0) + u[1] * state.grad(ALPHA_2D, q, 1)),
            -0.5 * c0 * u[0] * a,
            -0.5 * c0 * u[1] * a,
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
        if equation != ALPHA_2D {
            return [0.0; 3];
        }
        let c0 = self.config.distribution_parameter();
        let u = velocity(state, q);
        let du = [direction.value(0, q), direction.value(1, q)];
        let a = state.value(ALPHA_2D, q);
        let da = direction.value(ALPHA_2D, q);
        [
            0.5 * c0
                * (du[0] * state.grad(ALPHA_2D, q, 0)
                    + du[1] * state.grad(ALPHA_2D, q, 1)
                    + u[0] * direction.grad(ALPHA_2D, q, 0)
                    + u[1] * direction.grad(ALPHA_2D, q, 1)),
            -0.5 * c0 * (du[0] * a + u[0] * da),
            -0.5 * c0 * (du[1] * a + u[1] * da),
        ]
    }
}
