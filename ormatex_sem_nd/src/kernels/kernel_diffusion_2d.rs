use crate::common::{CellState, LocalCtx};
use crate::material::{ConstantCoefficient, MaterialProperty};

use super::kernel_common::{BilinearForm, ResidualKernel};

/// Scalar 2D diffusion kernel for `-div(nu * grad(T))`.
pub struct KernelDiffusion2D {
    pub nu: Box<dyn MaterialProperty<f64>>,
}

impl KernelDiffusion2D {
    pub fn new(nu: f64) -> Self {
        Self::with_coefficient(ConstantCoefficient(nu))
    }

    pub fn with_coefficient<N>(nu: N) -> Self
    where
        N: MaterialProperty<f64> + 'static,
    {
        Self { nu: Box::new(nu) }
    }
}

impl BilinearForm for KernelDiffusion2D {
    fn integrand(
        &self,
        ctx: &LocalCtx,
        _equation: usize,
        _unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        assert_eq!(ctx.ncomp, 1, "KernelDiffusion2D: scalar only (ncomp==1)");
        assert_eq!(ctx.gdim, 2, "KernelDiffusion2D: 2D only");
        let test = ctx.test(test_i, 0);
        let trial = ctx.trial(trial_i, 0);
        let grad_dot = (0..2)
            .map(|d| test.grad(q, d) * trial.grad(q, d))
            .sum::<f64>();
        self.nu.eval(&ctx.material_context(None, q)) * grad_dot
    }
}

impl ResidualKernel for KernelDiffusion2D {
    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        _equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        assert_eq!(ctx.ncomp, 1, "KernelDiffusion2D: scalar only (ncomp==1)");
        assert_eq!(ctx.gdim, 2, "KernelDiffusion2D: 2D only");
        let test = ctx.test(test_i, 0);
        let grad_dot = (0..2)
            .map(|d| state.grad(0, q, d) * test.grad(q, d))
            .sum::<f64>();
        self.nu.eval(&ctx.material_context(Some(state), q)) * grad_dot
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
        let material = ctx.material_context(Some(state), q);
        let test = ctx.test(test_i, 0);
        let trial = ctx.trial(trial_i, 0);
        let grad_dot = (0..2)
            .map(|d| trial.grad(q, d) * test.grad(q, d))
            .sum::<f64>();
        let state_grad_dot = (0..2)
            .map(|d| state.grad(0, q, d) * test.grad(q, d))
            .sum::<f64>();
        self.nu.eval(&material) * grad_dot
            + self.nu.derivative(&material, unknown).unwrap_or(0.0) * trial.v(q) * state_grad_dot
    }
}
