//! Tensor viscous stress `d(tau)/dx` for the 1D `u` equation.
//!
//! Mathematics: for equation 0 the triple is `(0, tau, 0)` with
//! `tau = 2 nu du/dx`; the action is linear in the direction gradient.
//! Owns equation 0 only. Weak counterpart:
//! [`KernelEdacViscousStress1D`](crate::kernels::edac::weak::viscous_stress_1d::KernelEdacViscousStress1D).

use crate::common::{CellState, TensorCtx};

use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config_1d::{fluid_field_names_1d, EdacNavierStokes1DConfig};

/// Tensor viscous-stress kernel (sum-factorized counterpart, owns equation 0).
pub struct TensorKernelEdacViscousStress1D {
    pub config: EdacNavierStokes1DConfig,
}

impl TensorKernelEdacViscousStress1D {
    pub fn new(config: EdacNavierStokes1DConfig) -> Self {
        Self { config }
    }
}

impl TensorResidualKernel<1> for TensorKernelEdacViscousStress1D {
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
            [0.0, self.config.stress_tensor(state, q), 0.0]
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
            [
                0.0,
                self.config
                    .stress_tensor_directional_derivative(direction, q),
                0.0,
            ]
        }
    }
}
