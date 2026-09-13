//! Momentum convection `(u . grad) u` for the `u`/`v` equations.
//!
//! Mathematics: weak form `(u_j d_j u_i) v` with the Jacobian carrying
//! `trial d_j u_i + [unknown == i] u_j d_j trial`; owns equations 0-1,
//! pressure row is zero. Tensor mirror:
//! [`TensorKernelEdacMomentumConvection2D`](crate::kernels::edac::tensor::momentum_convection::TensorKernelEdacMomentumConvection2D).
use crate::common::{CellState, LocalCtx};
use crate::kernels::common::ResidualKernel;
use crate::kernels::edac::config::{EdacNavierStokes2DConfig, check_weak_cell, fluid_field_names, velocity};

/// Momentum convection `(u.grad)u` for `u`/`v` (owns equations 0-1).
pub struct KernelEdacMomentumConvection2D {
    pub config: EdacNavierStokes2DConfig,
}

impl KernelEdacMomentumConvection2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl ResidualKernel for KernelEdacMomentumConvection2D {
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
                let velocity = velocity(state, q);
                let convection = velocity
                    .iter()
                    .enumerate()
                    .map(|(j, &u_j)| u_j * state.grad(equation, q, j))
                    .sum::<f64>();
                convection * ctx.test(test_i, 0).v(q)
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
        if equation == 2 {
            return 0.0;
        }
        let test = ctx.test(test_i, 0);
        if unknown >= 2 {
            return 0.0;
        }
        let trial = ctx.trial(trial_i, 0);
        let velocity = velocity(state, q);
        let convection = trial.v(q) * state.grad(equation, q, unknown)
            + if unknown == equation {
                velocity
                    .iter()
                    .enumerate()
                    .map(|(j, &u_j)| u_j * trial.grad(q, j))
                    .sum::<f64>()
            } else {
                0.0
            };
        convection * test.v(q)
    }
}

