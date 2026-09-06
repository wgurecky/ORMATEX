use crate::common::{CellState, LocalCtx, TensorCtx};

use super::kernel_common::{ResidualKernel, TensorResidualKernel};

/// Energy equation with state-coupled velocity advection and diffusion.
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

/// Tensor-product state-coupled energy kernel.
pub struct TensorKernelEnergyAdvectionDiffusion2D(pub KernelEnergyAdvectionDiffusion2D);

impl TensorKernelEnergyAdvectionDiffusion2D {
    pub fn new(thermal_diffusivity: f64) -> Self {
        Self(KernelEnergyAdvectionDiffusion2D::new(thermal_diffusivity))
    }
}

impl TensorResidualKernel<2> for TensorKernelEnergyAdvectionDiffusion2D {
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

    fn tensor_residual(
        &self,
        _ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        _equation: usize,
        q: usize,
    ) -> [f64; 3] {
        assert_eq!(state.nfields, 3, "energy state must contain [u, v, T]");
        [
            state.value(0, q) * state.grad(2, q, 0) + state.value(1, q) * state.grad(2, q, 1),
            self.0.thermal_diffusivity * state.grad(2, q, 0),
            self.0.thermal_diffusivity * state.grad(2, q, 1),
        ]
    }

    fn tensor_jacobian_action(
        &self,
        _ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        _equation: usize,
        q: usize,
    ) -> [f64; 3] {
        assert_eq!(state.nfields, 3, "energy state must contain [u, v, T]");
        assert_eq!(
            direction.nfields, 3,
            "energy direction must contain [u, v, T]"
        );
        [
            direction.value(0, q) * state.grad(2, q, 0)
                + direction.value(1, q) * state.grad(2, q, 1)
                + state.value(0, q) * direction.grad(2, q, 0)
                + state.value(1, q) * direction.grad(2, q, 1),
            self.0.thermal_diffusivity * direction.grad(2, q, 0),
            self.0.thermal_diffusivity * direction.grad(2, q, 1),
        ]
    }
}
