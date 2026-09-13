//! Artificial-compressibility divergence `rho c0^2 div(u)` for `p`.
//!
//! Mathematics: weak form `rho c0^2 div(u) v` with Jacobian on the velocity
//! unknowns; owns equation 2 only. Tensor mirror:
//! [`TensorKernelEdacPressureDivergence2D`](crate::kernels::edac::tensor::pressure_divergence::TensorKernelEdacPressureDivergence2D).
use crate::common::{CellState, LocalCtx};
use crate::kernels::common::ResidualKernel;
use crate::kernels::edac::config::{EdacNavierStokes2DConfig, check_weak_cell, fluid_field_names};

/// Artificial-compressibility `rho c0^2 div(u)` for `p` (owns equation 2).
pub struct KernelEdacPressureDivergence2D {
    pub config: EdacNavierStokes2DConfig,
}

impl KernelEdacPressureDivergence2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl ResidualKernel for KernelEdacPressureDivergence2D {
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
        if equation != 2 {
            return 0.0;
        }
        let divergence = state.grad(0, q, 0) + state.grad(1, q, 1);
        self.config.rho * self.config.c0 * self.config.c0 * divergence * ctx.test(test_i, 0).v(q)
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
        if equation != 2 || unknown >= 2 {
            return 0.0;
        }
        self.config.rho
            * self.config.c0
            * self.config.c0
            * ctx.trial(trial_i, 0).grad(q, unknown)
            * ctx.test(test_i, 0).v(q)
    }
}

