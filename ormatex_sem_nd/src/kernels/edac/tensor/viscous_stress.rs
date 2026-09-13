//! Tensor viscous stress `div(tau)` for the `u`/`v` equations.
//!
//! Mathematics: for momentum equation `i < 2` the triple is
//! `(0, tau_i0, tau_i1)` with `tau_ij = 2 (nu + nu_t) S_ij`; the action uses
//! the directional-derivative stress row (eddy-viscosity linearization
//! included). Both row components share one viscosity evaluation via
//! `EdacNavierStokes2DConfig::stress_tensor_row`. Owns equations 0–1 only.
//! Weak counterpart:
//! [`KernelEdacViscousStress2D`](crate::kernels::edac::weak::viscous_stress::KernelEdacViscousStress2D).

use crate::common::{CellState, TensorCtx};

use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config::{fluid_field_names, EdacNavierStokes2DConfig};

/// Tensor viscous-stress kernel (sum-factorized counterpart, owns equations 0-1).
pub struct TensorKernelEdacViscousStress2D {
    pub config: EdacNavierStokes2DConfig,
}

impl TensorKernelEdacViscousStress2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl TensorResidualKernel<2> for TensorKernelEdacViscousStress2D {
    fn nfields(&self) -> usize {
        3
    }
    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names()
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
            [0.0; 3]
        } else {
            let row = self.config.stress_tensor_row(ctx, state, q, equation);
            [0.0, row[0], row[1]]
        }
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
            [0.0; 3]
        } else {
            let row = self
                .config
                .stress_tensor_row_directional_derivative(ctx, state, direction, q, equation);
            [0.0, row[0], row[1]]
        }
    }
}
