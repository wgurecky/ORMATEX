//! Split-form pressure advection tensor kernel.
//!
//! Mathematics: half advective plus half conservative-flux triples for the
//! pressure equation with the matching action; owns equation 2 only. Needs
//! the split-flux boundary. Weak counterpart:
//! [`KernelEdacPressureAdvectionSplit2D`](crate::kernels::edac::weak::pressure_advection_split::KernelEdacPressureAdvectionSplit2D).
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config::{fluid_field_names, velocity, EdacNavierStokes2DConfig};

/// Tensor split pressure-advection kernel (owns equation 2; needs split-flux boundary).
pub struct TensorKernelEdacPressureAdvectionSplit2D {
    pub config: EdacNavierStokes2DConfig,
}
impl TensorKernelEdacPressureAdvectionSplit2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<2> for TensorKernelEdacPressureAdvectionSplit2D {
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
            return [0.0; 3];
        }
        let u = velocity(state, q);
        let p = state.value(2, q);
        [
            0.5 * (u[0] * state.grad(2, q, 0) + u[1] * state.grad(2, q, 1)),
            -0.5 * u[0] * p,
            -0.5 * u[1] * p,
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
        if equation != 2 {
            return [0.0; 3];
        }
        let u = velocity(state, q);
        let du = [direction.value(0, q), direction.value(1, q)];
        let p = state.value(2, q);
        let dp = direction.value(2, q);
        [
            0.5 * (du[0] * state.grad(2, q, 0)
                + du[1] * state.grad(2, q, 1)
                + u[0] * direction.grad(2, q, 0)
                + u[1] * direction.grad(2, q, 1)),
            -0.5 * (du[0] * p + u[0] * dp),
            -0.5 * (du[1] * p + u[1] * dp),
        ]
    }
}
