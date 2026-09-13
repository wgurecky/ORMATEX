//! Tensor artificial-compressibility divergence `rho c0^2 du/dx` for 1D `p`.
//!
//! Mathematics: for equation 1 the triple is `(rho c0^2 du/dx, 0, 0)` with
//! the linear action on the direction divergence. Owns equation 1 only.
//! Weak counterpart:
//! [`KernelEdacPressureDivergence1D`](crate::kernels::edac::weak::pressure_divergence_1d::KernelEdacPressureDivergence1D).

use crate::common::{CellState, TensorCtx};

use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config_1d::{fluid_field_names_1d, EdacNavierStokes1DConfig};

/// Tensor pressure-divergence kernel (sum-factorized counterpart, owns equation 1).
pub struct TensorKernelEdacPressureDivergence1D {
    pub config: EdacNavierStokes1DConfig,
}

impl TensorKernelEdacPressureDivergence1D {
    pub fn new(config: EdacNavierStokes1DConfig) -> Self {
        Self { config }
    }
}

impl TensorResidualKernel<1> for TensorKernelEdacPressureDivergence1D {
    fn nfields(&self) -> usize {
        2
    }
    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names_1d()
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
            [0.0; 3]
        } else {
            [
                self.config.rho * self.config.c0 * self.config.c0 * state.grad(0, q, 0),
                0.0,
                0.0,
            ]
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
        if equation != 1 {
            [0.0; 3]
        } else {
            [
                self.config.rho * self.config.c0 * self.config.c0 * direction.grad(0, q, 0),
                0.0,
                0.0,
            ]
        }
    }
}
