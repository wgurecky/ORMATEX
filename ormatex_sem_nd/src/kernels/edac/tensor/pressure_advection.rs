//! Tensor pressure advection `(u . grad) p` for the pressure equation.
//!
//! Mathematics: for equation 2 the triple is `(u_j d_j p, 0, 0)`; the action
//! adds `(du_j d_j p + u_j d_j dp, 0, 0)`. Owns equation 2 only. Weak
//! counterpart:
//! [`KernelEdacPressureAdvection2D`](crate::kernels::edac::weak::pressure_advection::KernelEdacPressureAdvection2D).

use crate::common::{CellState, TensorCtx};

use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config::{fluid_field_names, velocity, EdacNavierStokes2DConfig};

/// Tensor pressure-advection kernel (sum-factorized counterpart, owns equation 2).
pub struct TensorKernelEdacPressureAdvection2D {
    pub config: EdacNavierStokes2DConfig,
}

impl TensorKernelEdacPressureAdvection2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl TensorResidualKernel<2> for TensorKernelEdacPressureAdvection2D {
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
            let u = velocity(state, q);
            [
                u[0] * state.grad(2, q, 0) + u[1] * state.grad(2, q, 1),
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
        if equation != 2 {
            [0.0; 3]
        } else {
            let u = velocity(state, q);
            [
                direction.value(0, q) * state.grad(2, q, 0)
                    + direction.value(1, q) * state.grad(2, q, 1)
                    + u[0] * direction.grad(2, q, 0)
                    + u[1] * direction.grad(2, q, 1),
                0.0,
                0.0,
            ]
        }
    }
}
