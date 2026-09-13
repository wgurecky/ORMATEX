//! Split-form momentum convection tensor kernel.
//!
//! Mathematics: half advective plus half conservative-flux triples,
//! `1/2 (u_j d_j u_i, -u_j u_i, ...)` per flux slot with the matching
//! directional action; owns equations 0-1. Needs the split-flux boundary.
//! Weak counterpart:
//! [`KernelEdacMomentumConvectionSplit2D`](crate::kernels::edac::weak::momentum_convection_split::KernelEdacMomentumConvectionSplit2D).
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config::{fluid_field_names, velocity, EdacNavierStokes2DConfig};

/// Tensor split momentum-convection kernel (owns equations 0-1; needs split-flux boundary).
pub struct TensorKernelEdacMomentumConvectionSplit2D {
    pub config: EdacNavierStokes2DConfig,
}
impl TensorKernelEdacMomentumConvectionSplit2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<2> for TensorKernelEdacMomentumConvectionSplit2D {
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
            return [0.0; 3];
        }
        let u = velocity(state, q);
        [
            0.5 * (u[0] * state.grad(equation, q, 0) + u[1] * state.grad(equation, q, 1)),
            -0.5 * u[0] * state.value(equation, q),
            -0.5 * u[1] * state.value(equation, q),
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
        if equation >= 2 {
            return [0.0; 3];
        }
        let u = velocity(state, q);
        let du = [direction.value(0, q), direction.value(1, q)];
        [
            0.5 * (du[0] * state.grad(equation, q, 0)
                + du[1] * state.grad(equation, q, 1)
                + u[0] * direction.grad(equation, q, 0)
                + u[1] * direction.grad(equation, q, 1)),
            -0.5 * (du[0] * state.value(equation, q) + u[0] * direction.value(equation, q)),
            -0.5 * (du[1] * state.value(equation, q) + u[1] * direction.value(equation, q)),
        ]
    }
}
