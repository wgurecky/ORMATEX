//! Split-form pressure advection weak kernel (1D).
use crate::common::{CellState, LocalCtx};
use crate::kernels::common::ResidualKernel;
use crate::kernels::edac::config_1d::{
    check_weak_cell_1d, fluid_field_names_1d, EdacNavierStokes1DConfig,
};

/// Split pressure advection for the 1D EDAC pressure equation (owns equation 1).
///
/// Like the momentum term, the conservative half has a matching state-aware
/// boundary flux.
pub struct KernelEdacPressureAdvectionSplit1D {
    pub config: EdacNavierStokes1DConfig,
}

impl KernelEdacPressureAdvectionSplit1D {
    pub fn new(config: EdacNavierStokes1DConfig) -> Self {
        Self { config }
    }
}

impl ResidualKernel for KernelEdacPressureAdvectionSplit1D {
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
        let test = ctx.test(test_i, 0);
        let u = state.value(0, q);
        let p = state.value(1, q);
        let advective = u * state.grad(1, q, 0);
        let conservative = u * p * test.grad(q, 0);
        0.5 * (advective * test.v(q) - conservative)
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
        if equation != 1 {
            return 0.0;
        }
        let test = ctx.test(test_i, 0);
        let trial = ctx.trial(trial_i, 0);
        let u = state.value(0, q);
        let p = state.value(1, q);
        let advective = if unknown == 1 {
            u * trial.grad(q, 0)
        } else {
            trial.v(q) * state.grad(1, q, 0)
        };
        let flux_variation = if unknown == 1 {
            u * trial.v(q)
        } else {
            p * trial.v(q)
        };
        let conservative = flux_variation * test.grad(q, 0);
        0.5 * (advective * test.v(q) - conservative)
    }
}
