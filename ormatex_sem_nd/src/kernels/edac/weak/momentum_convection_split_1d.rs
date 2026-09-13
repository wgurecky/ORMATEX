//! Split-form momentum convection weak kernel (1D).
use crate::common::{CellState, LocalCtx};
use crate::kernels::common::ResidualKernel;
use crate::kernels::edac::config_1d::{
    check_weak_cell_1d, fluid_field_names_1d, EdacNavierStokes1DConfig,
};

/// Split momentum convection for the 1D `u` equation (owns equation 0).
///
/// Weak form `1/2 (u du/dx v - u u dv/dx)`; the conservative half is the weak
/// volume form whose matching boundary flux is supplied by a state-aware
/// facet kernel.
pub struct KernelEdacMomentumConvectionSplit1D {
    pub config: EdacNavierStokes1DConfig,
}

impl KernelEdacMomentumConvectionSplit1D {
    pub fn new(config: EdacNavierStokes1DConfig) -> Self {
        Self { config }
    }
}

impl ResidualKernel for KernelEdacMomentumConvectionSplit1D {
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
        let test = ctx.test(test_i, 0);
        let u = state.value(0, q);
        let advective = u * state.grad(0, q, 0);
        let conservative = u * state.value(0, q) * test.grad(q, 0);
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
        if equation != 0 || unknown != 0 {
            return 0.0;
        }
        let test = ctx.test(test_i, 0);
        let trial = ctx.trial(trial_i, 0);
        let u = state.value(0, q);
        let advective = u * trial.grad(q, 0) + trial.v(q) * state.grad(0, q, 0);
        let conservative = 2.0 * u * trial.v(q) * test.grad(q, 0);
        0.5 * (advective * test.v(q) - conservative)
    }
}
