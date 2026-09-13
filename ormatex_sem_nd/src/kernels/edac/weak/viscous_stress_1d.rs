//! Weak viscous stress `d(tau)/dx` for the 1D `u` equation.
//!
//! Mathematics: weak form `tau dv/dx` with `tau = 2 nu du/dx` (laminar);
//! owns equation 0. Tensor mirror:
//! [`TensorKernelEdacViscousStress1D`](crate::kernels::edac::tensor::viscous_stress_1d::TensorKernelEdacViscousStress1D).
use crate::common::{CellState, LocalCtx};
use crate::kernels::common::ResidualKernel;
use crate::kernels::edac::config_1d::{
    check_weak_cell_1d, fluid_field_names_1d, EdacNavierStokes1DConfig,
};

/// Weak viscous stress `d(tau)/dx` for `u` (owns equation 0).
pub struct KernelEdacViscousStress1D {
    pub config: EdacNavierStokes1DConfig,
}

impl KernelEdacViscousStress1D {
    pub fn new(config: EdacNavierStokes1DConfig) -> Self {
        Self { config }
    }
}

impl ResidualKernel for KernelEdacViscousStress1D {
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
        self.config.stress(state, q) * ctx.test(test_i, 0).grad(q, 0)
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
        if equation != 0 {
            return 0.0;
        }
        let trial = ctx.trial(trial_i, 0);
        self.config.stress_jacobian(trial.grad(q, 0), unknown) * ctx.test(test_i, 0).grad(q, 0)
    }
}
