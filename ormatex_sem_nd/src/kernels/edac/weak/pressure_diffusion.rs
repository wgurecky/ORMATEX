//! Weak EDAC pressure diffusion `div(k grad(p))` for `p`.
//!
//! Mathematics: weak form `k d_j p d_j v` with pressure diffusivity from the
//! artificial sound speed and filter width; Jacobian on the pressure unknown
//! only; owns equation 2. Tensor mirror:
//! [`TensorKernelEdacPressureDiffusion2D`](crate::kernels::edac::tensor::pressure_diffusion::TensorKernelEdacPressureDiffusion2D).
use crate::common::{CellState, LocalCtx};
use crate::kernels::common::ResidualKernel;
use crate::kernels::edac::config::{EdacNavierStokes2DConfig, check_weak_cell, fluid_field_names};

/// Weak pressure diffusion `div(k grad(p))` for `p` (owns equation 2).
pub struct KernelEdacPressureDiffusion2D {
    pub config: EdacNavierStokes2DConfig,
}

impl KernelEdacPressureDiffusion2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl ResidualKernel for KernelEdacPressureDiffusion2D {
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
        let pressure_gradient = (0..2)
            .map(|j| state.grad(2, q, j) * ctx.test(test_i, 0).grad(q, j))
            .sum::<f64>();
        self.config.pressure_diffusivity(ctx) * pressure_gradient
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
        if equation != 2 || unknown != 2 {
            return 0.0;
        }
        let pressure_gradient = (0..2)
            .map(|j| ctx.trial(trial_i, 0).grad(q, j) * ctx.test(test_i, 0).grad(q, j))
            .sum::<f64>();
        self.config.pressure_diffusivity(ctx) * pressure_gradient
    }
}

