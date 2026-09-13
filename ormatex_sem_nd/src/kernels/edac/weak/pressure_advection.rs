//! Pressure advection `(u . grad) p` for the pressure equation.
//!
//! Mathematics: weak form `(u_j d_j p) v` with Jacobian on both velocity
//! (`trial d_j p`) and pressure unknowns; owns equation 2 only. Tensor mirror:
//! [`TensorKernelEdacPressureAdvection2D`](crate::kernels::edac::tensor::pressure_advection::TensorKernelEdacPressureAdvection2D).
use crate::common::{CellState, LocalCtx};
use crate::kernels::common::ResidualKernel;
use crate::kernels::edac::config::{
    check_weak_cell, fluid_field_names, velocity, EdacNavierStokes2DConfig,
};

/// Pressure advection `(u.grad)p` for `p` (owns equation 2).
pub struct KernelEdacPressureAdvection2D {
    pub config: EdacNavierStokes2DConfig,
}

impl KernelEdacPressureAdvection2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl ResidualKernel for KernelEdacPressureAdvection2D {
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
        let velocity = velocity(state, q);
        let pressure_advection = velocity
            .iter()
            .enumerate()
            .map(|(j, &u_j)| u_j * state.grad(2, q, j))
            .sum::<f64>();
        pressure_advection * ctx.test(test_i, 0).v(q)
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
        let trial = ctx.trial(trial_i, 0);
        let pressure_advection = if unknown == 2 {
            let velocity = velocity(state, q);
            velocity
                .iter()
                .enumerate()
                .map(|(j, &u_j)| u_j * trial.grad(q, j))
                .sum::<f64>()
        } else {
            trial.v(q) * state.grad(2, q, unknown)
        };
        pressure_advection * ctx.test(test_i, 0).v(q)
    }
}
