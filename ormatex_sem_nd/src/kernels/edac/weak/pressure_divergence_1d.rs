//! Artificial-compressibility divergence `rho c0^2 du/dx` for 1D `p`.
//!
//! Mathematics: weak form `rho c0^2 du/dx v` with Jacobian on the velocity
//! unknown; owns equation 1 only. Tensor mirror:
//! [`TensorKernelEdacPressureDivergence1D`](crate::kernels::edac::tensor::pressure_divergence_1d::TensorKernelEdacPressureDivergence1D).
use crate::common::{CellState, LocalCtx};
use crate::kernels::common::ResidualKernel;
use crate::kernels::edac::config_1d::{
    check_weak_cell_1d, fluid_field_names_1d, EdacNavierStokes1DConfig,
};

/// Artificial-compressibility `rho c0^2 du/dx` for `p` (owns equation 1).
pub struct KernelEdacPressureDivergence1D {
    pub config: EdacNavierStokes1DConfig,
}

impl KernelEdacPressureDivergence1D {
    pub fn new(config: EdacNavierStokes1DConfig) -> Self {
        Self { config }
    }
}

impl ResidualKernel for KernelEdacPressureDivergence1D {
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
        if equation != 1 {
            return 0.0;
        }
        self.config.rho
            * self.config.c0
            * self.config.c0
            * state.grad(0, q, 0)
            * ctx.test(test_i, 0).v(q)
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
        if equation != 1 || unknown != 0 {
            return 0.0;
        }
        self.config.rho
            * self.config.c0
            * self.config.c0
            * ctx.trial(trial_i, 0).grad(q, 0)
            * ctx.test(test_i, 0).v(q)
    }
}
