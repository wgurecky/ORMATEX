//! Tensor artificial-compressibility divergence `rho c0^2 div(u)` for `p`.
//!
//! Mathematics: for equation 2 the triple is `(rho c0^2 div(u), 0, 0)` with
//! the linear action on the direction divergence. Owns equation 2 only.
//! Weak counterpart:
//! [`KernelEdacPressureDivergence2D`](crate::kernels::edac::weak::pressure_divergence::KernelEdacPressureDivergence2D).

use crate::common::{CellState, TensorCtx};

use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config::{fluid_field_names, EdacNavierStokes2DConfig};

/// Tensor pressure-divergence kernel (sum-factorized counterpart, owns equation 2).
pub struct TensorKernelEdacPressureDivergence2D {
    pub config: EdacNavierStokes2DConfig,
}

impl TensorKernelEdacPressureDivergence2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl TensorResidualKernel<2> for TensorKernelEdacPressureDivergence2D {
    fn nfields(&self) -> usize {
        3
    }
    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names()
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
            [0.0; 3]
        } else {
            [
                self.config.rho
                    * self.config.c0
                    * self.config.c0
                    * (state.grad(0, q, 0) + state.grad(1, q, 1)),
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
        if equation != 2 {
            [0.0; 3]
        } else {
            [
                self.config.rho
                    * self.config.c0
                    * self.config.c0
                    * (direction.grad(0, q, 0) + direction.grad(1, q, 1)),
                0.0,
                0.0,
            ]
        }
    }
}
