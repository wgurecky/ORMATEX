use crate::common::{CellState, LocalCtx, TensorCtx};
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
    fn supports_tensor_bilinear(&self) -> bool {
        true
    }

    fn tensor_bilinear(
        &self,
        ctx: &TensorCtx<'_>,
        _equation: usize,
        _unknown: usize,
        q: usize,
        _trial_value: f64,
        trial_grad: [f64; 2],
    ) -> [f64; 3] {
        let material = ctx.material_context(None, q);
        let nu = self.nu.eval(&material);
        [0.0, nu * trial_grad[0], nu * trial_grad[1]]
    }

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
    fn supports_tensor_residual(&self) -> bool {
        true
    }

    fn supports_tensor_jacobian(&self) -> bool {
        true
    }

    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        _equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let material = ctx.material_context(Some(state), q);
        let nu = self.nu.eval(&material);
        [0.0, nu * state.grad(0, q, 0), nu * state.grad(0, q, 1)]
    }

    fn tensor_jacobian_action(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        _equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let material = ctx.material_context(Some(state), q);
        let nu = self.nu.eval(&material);
        let dnu = self.nu.derivative(&material, 0).unwrap_or(0.0);
        let du = direction.value(0, q);
        [
            0.0,
            nu * direction.grad(0, q, 0) + dnu * du * state.grad(0, q, 0),
            nu * direction.grad(0, q, 1) + dnu * du * state.grad(0, q, 1),
        ]
    }

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
