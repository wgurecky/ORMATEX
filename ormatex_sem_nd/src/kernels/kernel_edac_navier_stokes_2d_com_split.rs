use crate::common::{CellState, LocalCtx};

use super::kernel_common::ResidualKernel;
use super::kernel_edac_navier_stokes_2d_com::EdacNavierStokes2DConfig;

fn field_names() -> Option<Vec<String>> {
    Some(["u", "v", "p"].into_iter().map(str::to_owned).collect())
}

fn check(ctx: &LocalCtx, state: &CellState) {
    assert_eq!(ctx.gdim, 2, "EDAC split terms require gdim == 2");
    assert_eq!(ctx.ncomp, 1, "fluid fields must be scalar fields");
    assert_eq!(state.nfields, 3, "fluid state must contain [u, v, p]");
}

fn velocity(state: &CellState, q: usize) -> [f64; 2] {
    [state.value(0, q), state.value(1, q)]
}

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
        field_names()
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        check(ctx, state);
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
        check(ctx, state);
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
        field_names()
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        check(ctx, state);
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
        check(ctx, state);
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
