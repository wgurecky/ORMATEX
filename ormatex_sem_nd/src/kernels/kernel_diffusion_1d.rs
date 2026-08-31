use crate::common::{CellState, LocalCtx, TensorCtx};
use crate::material::{ConstantCoefficient, MaterialProperty};

use super::kernel_common::{BilinearForm, ResidualKernel};

/// Scalar 1D diffusion kernel for `-d/dx(nu * dT/dx)`.
pub struct KernelDiffusion {
    pub nu: Box<dyn MaterialProperty<f64>>,
}

impl KernelDiffusion {
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

impl BilinearForm for KernelDiffusion {
    fn supports_tensor_bilinear_1d(&self) -> bool {
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
        let nu = self.nu.eval(&ctx.material_context(None, q));
        [0.0, nu * trial_grad[0], 0.0]
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
        assert_eq!(ctx.ncomp, 1, "KernelDiffusion: scalar only (ncomp==1)");
        assert_eq!(ctx.gdim, 1, "KernelDiffusion: 1D only");
        let material = ctx.material_context(None, q);
        self.nu.eval(&material) * ctx.test(test_i, 0).grad(q, 0) * ctx.trial(trial_i, 0).grad(q, 0)
    }
}

impl ResidualKernel for KernelDiffusion {
    fn supports_tensor_residual_1d(&self) -> bool {
        true
    }

    fn supports_tensor_jacobian_1d(&self) -> bool {
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
        [0.0, self.nu.eval(&material) * state.grad(0, q, 0), 0.0]
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
        [
            0.0,
            nu * direction.grad(0, q, 0) + dnu * direction.value(0, q) * state.grad(0, q, 0),
            0.0,
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
        assert_eq!(ctx.ncomp, 1, "KernelDiffusion: scalar only (ncomp==1)");
        assert_eq!(ctx.gdim, 1, "KernelDiffusion: 1D only");
        let material = ctx.material_context(Some(state), q);
        self.nu.eval(&material) * state.grad(0, q, 0) * ctx.test(test_i, 0).grad(q, 0)
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
        let test_grad = ctx.test(test_i, 0).grad(q, 0);
        let trial = ctx.trial(trial_i, 0);
        self.nu.eval(&material) * trial.grad(q, 0) * test_grad
            + self.nu.derivative(&material, unknown).unwrap_or(0.0)
                * trial.v(q)
                * state.grad(0, q, 0)
                * test_grad
    }
}
