//! Tensor momentum convection `(u . grad) u` for the `u`/`v` equations.
//!
//! Mathematics: for momentum equation `i < 2` the `(f0, f1x, f1y)` triple is
//! `(u_j d_j u_i, 0, 0)`; the Jacobian action additionally carries
//! `(du_j d_j u_i + u_j d_j du_i, 0, 0)`. Owns equations 0–1 only, so fused
//! sums skip the pressure block. Weak counterpart:
//! [`KernelEdacMomentumConvection2D`](crate::kernels::edac::weak::momentum_convection::KernelEdacMomentumConvection2D).

use crate::common::{CellState, TensorCtx};

use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config::{fluid_field_names, velocity, EdacNavierStokes2DConfig};

/// Tensor momentum-convection kernel (sum-factorized counterpart, owns equations 0-1).
pub struct TensorKernelEdacMomentumConvection2D {
    pub config: EdacNavierStokes2DConfig,
}

impl TensorKernelEdacMomentumConvection2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl TensorResidualKernel<2> for TensorKernelEdacMomentumConvection2D {
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
            let u = velocity(state, q);
            [
                u[0] * state.grad(equation, q, 0) + u[1] * state.grad(equation, q, 1),
                0.0,
                0.0,
            ]
        }
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
            [0.0; 3]
        } else {
            let u = velocity(state, q);
            [
                direction.value(0, q) * state.grad(equation, q, 0)
                    + direction.value(1, q) * state.grad(equation, q, 1)
                    + u[0] * direction.grad(equation, q, 0)
                    + u[1] * direction.grad(equation, q, 1),
                0.0,
                0.0,
            ]
        }
    }
}
