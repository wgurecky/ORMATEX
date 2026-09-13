//! Split-form momentum convection tensor kernel (1D).
//!
//! Mathematics: half advective plus half conservative-flux triple,
//! `1/2 (u du/dx, -u u, 0)` with the matching directional action; owns
//! equation 0. Weak counterpart:
//! [`KernelEdacMomentumConvectionSplit1D`](crate::kernels::edac::weak::momentum_convection_split_1d::KernelEdacMomentumConvectionSplit1D).
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config_1d::{fluid_field_names_1d, EdacNavierStokes1DConfig};

/// Tensor split momentum-convection kernel (owns equation 0).
pub struct TensorKernelEdacMomentumConvectionSplit1D {
    pub config: EdacNavierStokes1DConfig,
}
impl TensorKernelEdacMomentumConvectionSplit1D {
    pub fn new(config: EdacNavierStokes1DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<1> for TensorKernelEdacMomentumConvectionSplit1D {
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
            return [0.0; 3];
        }
        let u = state.value(0, q);
        [0.5 * u * state.grad(0, q, 0), -0.5 * u * u, 0.0]
    }
    fn tensor_jacobian_action(
        &self,
        _: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation != 0 {
            return [0.0; 3];
        }
        let u = state.value(0, q);
        let du = direction.value(0, q);
        [
            0.5 * (du * state.grad(0, q, 0) + u * direction.grad(0, q, 0)),
            -0.5 * (du * u + u * du),
            0.0,
        ]
    }
}
