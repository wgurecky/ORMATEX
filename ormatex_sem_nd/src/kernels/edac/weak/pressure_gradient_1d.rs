//! Pressure gradient `dp/dx / rho` for the 1D `u` equation.
//!
//! Mathematics: weak form `(dp/dx / rho) v` with Jacobian coupling the
//! momentum row to the pressure unknown; owns equation 0. Tensor mirror:
//! [`TensorKernelEdacPressureGradient1D`](crate::kernels::edac::tensor::pressure_gradient_1d::TensorKernelEdacPressureGradient1D).
use crate::common::{CellState, LocalCtx};
use crate::kernels::common::ResidualKernel;
use crate::kernels::edac::config_1d::{
    check_weak_cell_1d, fluid_field_names_1d, EdacNavierStokes1DConfig,
};

/// Pressure gradient `dp/dx/rho` for `u` (owns equation 0).
pub struct KernelEdacPressureGradient1D {
    pub config: EdacNavierStokes1DConfig,
}

impl KernelEdacPressureGradient1D {
    pub fn new(config: EdacNavierStokes1DConfig) -> Self {
        Self { config }
    }
}

impl ResidualKernel for KernelEdacPressureGradient1D {
    fn nfields(&self) -> usize {
        2
    }

    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names_1d()
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        check_weak_cell_1d(ctx, state);
        if equation != 0 {
            return 0.0;
        }
        state.grad(1, q, 0) * ctx.test(test_i, 0).v(q) / self.config.rho
    }

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        check_weak_cell_1d(ctx, state);
        assert!(unknown < 2, "fluid unknown field out of range");
        if equation != 0 || unknown != 1 {
            return 0.0;
        }
        ctx.trial(trial_i, 0).grad(q, 0) * ctx.test(test_i, 0).v(q) / self.config.rho
    }
}
