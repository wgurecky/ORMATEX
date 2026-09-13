use crate::common::{CellState, LocalCtx};

use crate::kernels::common::ResidualKernel;

/// Energy equation with state-coupled velocity advection and diffusion.
///
/// Rectangular coupling (3 inputs `[u, v, T]`, 1 output `T`): weak form
/// `(u*dx(T) + v*dy(T))*v + alpha*grad(T).grad(v)`. Tensor mirror:
/// [`TensorKernelEnergyAdvectionDiffusion2D`](crate::kernels::basic::tensor::energy_advection_diffusion_2d::TensorKernelEnergyAdvectionDiffusion2D).
pub struct KernelEnergyAdvectionDiffusion2D {
    pub thermal_diffusivity: f64,
}

impl KernelEnergyAdvectionDiffusion2D {
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

impl ResidualKernel for KernelEnergyAdvectionDiffusion2D {
    fn nfields(&self) -> usize {
        1
    }

    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    fn input_nfields(&self) -> usize {
        3
    }

    fn output_nfields(&self) -> usize {
        1
    }

    fn input_field_names(&self) -> Option<Vec<String>> {
        Some(["u", "v", "T"].into_iter().map(str::to_owned).collect())
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
        assert_eq!(ctx.gdim, 2, "energy terms require gdim == 2");
        assert_eq!(state.nfields, 3, "energy state must contain [u, v, T]");
        let test = ctx.test(test_i, 0);
        let advection =
            state.value(0, q) * state.grad(2, q, 0) + state.value(1, q) * state.grad(2, q, 1);
        let diffusion = self.thermal_diffusivity
            * (state.grad(2, q, 0) * test.grad(q, 0) + state.grad(2, q, 1) * test.grad(q, 1));
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
        assert_eq!(state.nfields, 3, "energy state must contain [u, v, T]");
        let test = ctx.test(test_i, 0);
        let trial = ctx.trial(trial_i, 0);
        let advection = if unknown < 2 {
            trial.v(q) * state.grad(2, q, unknown)
        } else {
            state.value(0, q) * trial.grad(q, 0) + state.value(1, q) * trial.grad(q, 1)
        };
        let diffusion = if unknown == 2 {
            self.thermal_diffusivity
                * (trial.grad(q, 0) * test.grad(q, 0) + trial.grad(q, 1) * test.grad(q, 1))
        } else {
            0.0
        };
        advection * test.v(q) + diffusion
    }
}
