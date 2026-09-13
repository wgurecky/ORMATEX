use crate::common::{CellState, LocalCtx};

use crate::kernels::common::ResidualKernel;

/// 1D energy equation with state-coupled velocity advection and diffusion.
///
/// Rectangular coupling (2 inputs `[u, T]`, 1 output `T`): weak form
/// `(u*dx(T))*v + alpha*dx(T)*dx(v)`. Tensor mirror:
/// [`TensorKernelEnergyAdvectionDiffusion1D`](crate::kernels::basic::tensor::energy_advection_diffusion_1d::TensorKernelEnergyAdvectionDiffusion1D).
pub struct KernelEnergyAdvectionDiffusion1D {
    pub thermal_diffusivity: f64,
}

impl KernelEnergyAdvectionDiffusion1D {
    pub fn new(thermal_diffusivity: f64) -> Self {
        assert!(
            thermal_diffusivity.is_finite() && thermal_diffusivity >= 0.0,
            "thermal diffusivity must be finite and nonnegative"
        );
        Self {
            thermal_diffusivity,
        }
    }
}

impl ResidualKernel for KernelEnergyAdvectionDiffusion1D {
    fn nfields(&self) -> usize {
        1
    }

    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    fn input_nfields(&self) -> usize {
        2
    }

    fn output_nfields(&self) -> usize {
        1
    }

    fn input_field_names(&self) -> Option<Vec<String>> {
        Some(["u", "T"].into_iter().map(str::to_owned).collect())
    }

    fn output_field_names(&self) -> Option<Vec<String>> {
        Some(["T"].into_iter().map(str::to_owned).collect())
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        _equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        assert_eq!(ctx.gdim, 1, "energy terms require gdim == 1");
        assert_eq!(state.nfields, 2, "energy state must contain [u, T]");
        let test = ctx.test(test_i, 0);
        let advection = state.value(0, q) * state.grad(1, q, 0);
        let diffusion = self.thermal_diffusivity * state.grad(1, q, 0) * test.grad(q, 0);
        advection * test.v(q) + diffusion
    }

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        _equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        assert_eq!(state.nfields, 2, "energy state must contain [u, T]");
        let test = ctx.test(test_i, 0);
        let trial = ctx.trial(trial_i, 0);
        let advection = if unknown == 0 {
            trial.v(q) * state.grad(1, q, 0)
        } else {
            state.value(0, q) * trial.grad(q, 0)
        };
        let diffusion = if unknown == 1 {
            self.thermal_diffusivity * trial.grad(q, 0) * test.grad(q, 0)
        } else {
            0.0
        };
        advection * test.v(q) + diffusion
    }
}
