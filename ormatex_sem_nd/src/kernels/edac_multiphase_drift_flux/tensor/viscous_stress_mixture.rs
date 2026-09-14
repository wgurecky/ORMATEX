//! Mixture viscous stress `div(tau_m)` for the 2D momentum equations.
//!
//! Mathematics: `(0, tau_i0, tau_i1)` with
//! `tau_ij = 2 (nu_m(alpha) + nu_t) S_ij`, where `nu_m = mu_m/rho_m` is the
//! linearly averaged mixture kinematic viscosity and `nu_t` is the
//! Smagorinsky-Lilly eddy viscosity evaluated on the mixture velocity (same
//! model as the base EDAC kernels, per request). The action linearizes both
//! the `nu_m(alpha)` dependence and the eddy viscosity. Owns equations 0-1.
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config::{drift_field_names, DriftFlux2DConfig};

/// Tensor mixture viscous-stress (owns equations 0-1).
pub struct TensorDriftViscousStress2D {
    pub config: DriftFlux2DConfig,
}
impl TensorDriftViscousStress2D {
    pub fn new(config: DriftFlux2DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<2> for TensorDriftViscousStress2D {
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
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation >= 2 {
            return [0.0; 3];
        }
        let row = self.config.stress_tensor_row(ctx, state, q, equation);
        [0.0, row[0], row[1]]
    }
    fn tensor_jacobian_action(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation >= 2 {
            return [0.0; 3];
        }
        let row = self
            .config
            .stress_tensor_row_directional_derivative(ctx, state, direction, q, equation);
        [0.0, row[0], row[1]]
    }
}
