use crate::common::{CellState, LocalCtx};
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
