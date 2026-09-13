//! Pressure gradient `grad(p) / rho` for the `u`/`v` equations.
//!
//! Mathematics: weak form `(d_i p / rho) v` with Jacobian `(d_i trial_p / rho)
//! coupling momentum rows to the pressure unknown; owns equations 0-1.
//! Tensor mirror:
//! [`TensorKernelEdacPressureGradient2D`](crate::kernels::edac::tensor::pressure_gradient::TensorKernelEdacPressureGradient2D).
use crate::common::{CellState, LocalCtx};
use crate::kernels::common::ResidualKernel;
use crate::kernels::edac::config::{check_weak_cell, fluid_field_names, EdacNavierStokes2DConfig};

/// Pressure gradient `grad(p)/rho` for `u`/`v` (owns equations 0-1).
pub struct KernelEdacPressureGradient2D {
    pub config: EdacNavierStokes2DConfig,
}

impl KernelEdacPressureGradient2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl ResidualKernel for KernelEdacPressureGradient2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names()
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        check_weak_cell(ctx, state);
        match equation {
            0 | 1 => state.grad(2, q, equation) * ctx.test(test_i, 0).v(q) / self.config.rho,
            2 => 0.0,
            _ => unreachable!(),
        }
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
        check_weak_cell(ctx, state);
        assert!(unknown < 3, "fluid unknown field out of range");
        if equation >= 2 || unknown != 2 {
            return 0.0;
        }
        ctx.trial(trial_i, 0).grad(q, equation) * ctx.test(test_i, 0).v(q) / self.config.rho
    }
}
