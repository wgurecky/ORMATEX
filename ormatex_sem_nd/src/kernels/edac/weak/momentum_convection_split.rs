//! Split-form momentum convection weak kernel.
use crate::common::{CellState, LocalCtx};
use crate::kernels::common::ResidualKernel;
use crate::kernels::edac::config::{EdacNavierStokes2DConfig, check_weak_cell, fluid_field_names, velocity};

/// Split momentum convection contribution for the `u` and `v` equations.
///
/// The conservative half is represented by its weak volume form. Its matching
/// boundary flux is supplied by a state-aware facet kernel.
pub struct KernelEdacMomentumConvectionSplit2D {
    pub config: EdacNavierStokes2DConfig,
}

impl KernelEdacMomentumConvectionSplit2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl ResidualKernel for KernelEdacMomentumConvectionSplit2D {
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
            0 | 1 => {
                let test = ctx.test(test_i, 0);
                let velocity = velocity(state, q);
                let advective = velocity
                    .iter()
                    .enumerate()
                    .map(|(j, &u_j)| u_j * state.grad(equation, q, j))
                    .sum::<f64>();
                let conservative = velocity
                    .iter()
                    .enumerate()
                    .map(|(j, &u_j)| u_j * state.value(equation, q) * test.grad(q, j))
                    .sum::<f64>();
                0.5 * (advective * test.v(q) - conservative)
            }
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
        if equation >= 2 || unknown >= 2 {
            return 0.0;
        }
        let test = ctx.test(test_i, 0);
        let trial = ctx.trial(trial_i, 0);
        let velocity = velocity(state, q);
        let advective = velocity
            .iter()
            .enumerate()
            .map(|(j, &u_j)| {
                let state_variation = if unknown == equation {
                    u_j * trial.grad(q, j)
                } else {
                    0.0
                };
                let velocity_variation = if unknown == j {
                    trial.v(q) * state.grad(equation, q, j)
                } else {
                    0.0
                };
                state_variation + velocity_variation
            })
            .sum::<f64>();
        let conservative = velocity
            .iter()
            .enumerate()
            .map(|(j, &u_j)| {
                let flux_variation = (if unknown == j {
                    velocity[equation] * trial.v(q)
                } else {
                    0.0
                }) + (if unknown == equation {
                    u_j * trial.v(q)
                } else {
                    0.0
                });
                flux_variation * test.grad(q, j)
            })
            .sum::<f64>();
        0.5 * (advective * test.v(q) - conservative)
    }
}
