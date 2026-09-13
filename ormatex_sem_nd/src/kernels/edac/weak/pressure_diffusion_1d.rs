//! Weak EDAC pressure diffusion `d(k dp/dx)/dx` for 1D `p`.
//!
//! Mathematics: weak form `k dp/dx dv/dx` with pressure diffusivity from the
//! artificial sound speed and cell length; Jacobian on the pressure unknown
//! only; owns equation 1. Tensor mirror:
//! [`TensorKernelEdacPressureDiffusion1D`](crate::kernels::edac::tensor::pressure_diffusion_1d::TensorKernelEdacPressureDiffusion1D).
use crate::common::{CellState, LocalCtx};
use crate::kernels::common::ResidualKernel;
use crate::kernels::edac::config_1d::{
    check_weak_cell_1d, fluid_field_names_1d, EdacNavierStokes1DConfig,
};

/// Weak pressure diffusion `d(k dp/dx)/dx` for `p` (owns equation 1).
pub struct KernelEdacPressureDiffusion1D {
    pub config: EdacNavierStokes1DConfig,
}

impl KernelEdacPressureDiffusion1D {
    pub fn new(config: EdacNavierStokes1DConfig) -> Self {
        Self { config }
    }
}

impl ResidualKernel for KernelEdacPressureDiffusion1D {
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
        self.config.pressure_diffusivity(ctx) * state.grad(1, q, 0) * ctx.test(test_i, 0).grad(q, 0)
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
        if equation != 1 || unknown != 1 {
            return 0.0;
        }
        self.config.pressure_diffusivity(ctx)
            * ctx.trial(trial_i, 0).grad(q, 0)
            * ctx.test(test_i, 0).grad(q, 0)
    }
}
