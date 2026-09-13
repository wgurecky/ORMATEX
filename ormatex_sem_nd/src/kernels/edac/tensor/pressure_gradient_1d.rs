//! Tensor pressure gradient `dp/dx / rho` for the 1D `u` equation.
//!
//! Mathematics: for equation 0 the triple is `(dp/dx / rho, 0, 0)` with the
//! linear action. Owns equation 0 only. Weak counterpart:
//! [`KernelEdacPressureGradient1D`](crate::kernels::edac::weak::pressure_gradient_1d::KernelEdacPressureGradient1D).

use crate::common::{CellState, TensorCtx};

use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config_1d::{fluid_field_names_1d, EdacNavierStokes1DConfig};

/// Tensor pressure-gradient kernel (sum-factorized counterpart, owns equation 0).
pub struct TensorKernelEdacPressureGradient1D {
    pub config: EdacNavierStokes1DConfig,
}

impl TensorKernelEdacPressureGradient1D {
    pub fn new(config: EdacNavierStokes1DConfig) -> Self {
        Self { config }
    }
}

impl TensorResidualKernel<1> for TensorKernelEdacPressureGradient1D {
    fn nfields(&self) -> usize {
        2
    }
    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names_1d()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation == 0
    }
    fn tensor_residual(
        &self,
        _: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation != 0 {
            [0.0; 3]
        } else {
            [state.grad(1, q, 0) / self.config.rho, 0.0, 0.0]
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
        if equation != 0 {
            [0.0; 3]
        } else {
            [direction.grad(1, q, 0) / self.config.rho, 0.0, 0.0]
        }
    }
}
