//! Tensor pressure gradient `grad(p) / rho` for the `u`/`v` equations.
//!
//! Mathematics: for momentum equation `i < 2` the triple is
//! `(d_i p / rho, 0, 0)` with the linear action `(d_i dp / rho, 0, 0)`.
//! Owns equations 0–1 only. Weak counterpart:
//! [`KernelEdacPressureGradient2D`](crate::kernels::edac::weak::pressure_gradient::KernelEdacPressureGradient2D).

use crate::common::{CellState, TensorCtx};

use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config::{fluid_field_names, EdacNavierStokes2DConfig};

/// Tensor pressure-gradient kernel (sum-factorized counterpart, owns equations 0-1).
pub struct TensorKernelEdacPressureGradient2D {
    pub config: EdacNavierStokes2DConfig,
}

impl TensorKernelEdacPressureGradient2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl TensorResidualKernel<2> for TensorKernelEdacPressureGradient2D {
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
        _: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation >= 2 {
            [0.0; 3]
        } else {
            [state.grad(2, q, equation) / self.config.rho, 0.0, 0.0]
        }
    }
    fn tensor_jacobian_action(
        &self,
        _: &TensorCtx<'_>,
        _: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation >= 2 {
            [0.0; 3]
        } else {
            [direction.grad(2, q, equation) / self.config.rho, 0.0, 0.0]
        }
    }
}
