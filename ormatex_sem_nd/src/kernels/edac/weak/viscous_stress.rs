//! Weak viscous stress `div(tau)` for the `u`/`v` equations.
//!
//! Mathematics: weak form `tau_ij d_j v` with `tau = 2 (nu + nu_t) S` closed
//! by Smagorinsky; the Jacobian linearizes both strain and eddy viscosity.
//! Owns equations 0-1. Tensor mirror:
//! [`TensorKernelEdacViscousStress2D`](crate::kernels::edac::tensor::viscous_stress::TensorKernelEdacViscousStress2D).
use crate::common::{CellState, LocalCtx};
use crate::kernels::common::ResidualKernel;
use crate::kernels::edac::config::{EdacNavierStokes2DConfig, check_weak_cell, fluid_field_names};

/// Weak viscous stress `div(tau)` for `u`/`v` (owns equations 0-1).
pub struct KernelEdacViscousStress2D {
    pub config: EdacNavierStokes2DConfig,
}

impl KernelEdacViscousStress2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl ResidualKernel for KernelEdacViscousStress2D {
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
            0 | 1 => (0..2)
                .map(|j| {
                    self.config.stress(ctx, state, q, equation, j) * ctx.test(test_i, 0).grad(q, j)
                })
                .sum(),
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
        if equation >= 2 {
            return 0.0;
        }
        let trial = ctx.trial(trial_i, 0);
        (0..2)
            .map(|j| {
                self.config
                    .stress_jacobian(ctx, state, q, equation, j, unknown, &trial)
                    * ctx.test(test_i, 0).grad(q, j)
            })
            .sum()
    }
}

