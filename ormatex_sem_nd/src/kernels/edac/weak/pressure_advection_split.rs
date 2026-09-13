//! Split-form pressure advection weak kernel.
use crate::common::{CellState, LocalCtx};
use crate::kernels::common::ResidualKernel;
use crate::kernels::edac::config::{EdacNavierStokes2DConfig, check_weak_cell, fluid_field_names, velocity};

/// Split pressure advection contribution for the EDAC pressure equation.
///
/// Like the momentum term, the conservative half has a matching state-aware
/// boundary flux.
pub struct KernelEdacPressureAdvectionSplit2D {
    pub config: EdacNavierStokes2DConfig,
}

impl KernelEdacPressureAdvectionSplit2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl ResidualKernel for KernelEdacPressureAdvectionSplit2D {
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
        let test = ctx.test(test_i, 0);
        let velocity = velocity(state, q);
        let pressure = state.value(2, q);
        let advective = velocity
            .iter()
            .enumerate()
            .map(|(j, &u_j)| u_j * state.grad(2, q, j))
            .sum::<f64>();
        let conservative = velocity
            .iter()
            .enumerate()
            .map(|(j, &u_j)| u_j * pressure * test.grad(q, j))
            .sum::<f64>();
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
        check_weak_cell(ctx, state);
        assert!(unknown < 3, "fluid unknown field out of range");
        if equation != 2 {
            return 0.0;
        }
        let test = ctx.test(test_i, 0);
        let trial = ctx.trial(trial_i, 0);
        let velocity = velocity(state, q);
        let pressure = state.value(2, q);
        let advective = if unknown == 2 {
            velocity
                .iter()
                .enumerate()
                .map(|(j, &u_j)| u_j * trial.grad(q, j))
                .sum::<f64>()
        } else if unknown < 2 {
            trial.v(q) * state.grad(2, q, unknown)
        } else {
            0.0
        };
        let conservative = velocity
            .iter()
            .enumerate()
            .map(|(j, &u_j)| {
                let flux_variation = if unknown == 2 {
                    u_j * trial.v(q)
                } else if unknown == j {
                    pressure * trial.v(q)
                } else {
                    0.0
                };
                flux_variation * test.grad(q, j)
            })
            .sum::<f64>();
        0.5 * (advective * test.v(q) - conservative)
    }
}
