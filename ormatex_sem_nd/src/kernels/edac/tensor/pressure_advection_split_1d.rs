//! Split-form pressure advection tensor kernel (1D).
//!
//! Mathematics: half advective plus half conservative-flux triple,
//! `1/2 (u dp/dx, -u p, 0)` with the matching action; owns equation 1.
//! Weak counterpart:
//! [`KernelEdacPressureAdvectionSplit1D`](crate::kernels::edac::weak::pressure_advection_split_1d::KernelEdacPressureAdvectionSplit1D).
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config_1d::{fluid_field_names_1d, EdacNavierStokes1DConfig};

/// Tensor split pressure-advection kernel (owns equation 1).
pub struct TensorKernelEdacPressureAdvectionSplit1D {
    pub config: EdacNavierStokes1DConfig,
}
impl TensorKernelEdacPressureAdvectionSplit1D {
    pub fn new(config: EdacNavierStokes1DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<1> for TensorKernelEdacPressureAdvectionSplit1D {
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
            return [0.0; 3];
        }
        let u = state.value(0, q);
        let p = state.value(1, q);
        [0.5 * u * state.grad(1, q, 0), -0.5 * u * p, 0.0]
    }
    fn tensor_jacobian_action(
        &self,
        _: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation != 1 {
            return [0.0; 3];
        }
        let u = state.value(0, q);
        let du = direction.value(0, q);
        let p = state.value(1, q);
        let dp = direction.value(1, q);
        [
            0.5 * (du * state.grad(1, q, 0) + u * direction.grad(1, q, 0)),
            -0.5 * (du * p + u * dp),
            0.0,
        ]
    }
}
