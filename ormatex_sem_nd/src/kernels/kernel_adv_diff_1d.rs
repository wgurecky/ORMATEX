use crate::common::{CellState, LocalCtx};
use crate::material::{ConstantCoefficient, MaterialProperty};

use super::kernel_common::{BilinearForm, ResidualKernel};

/// 1D advection-diffusion kernel.
pub struct KernelAdvDiff {
    pub nu: Box<dyn MaterialProperty<f64>>,
    pub vel: Box<dyn MaterialProperty<f64>>,
}

impl KernelAdvDiff {
    pub fn new(nu: f64, vel: f64) -> Self {
        Self::with_coefficients(ConstantCoefficient(nu), ConstantCoefficient(vel))
    }

    pub fn with_coefficients<N, V>(nu: N, vel: V) -> Self
    where
        N: MaterialProperty<f64> + 'static,
        V: MaterialProperty<f64> + 'static,
    {
        Self {
            nu: Box::new(nu),
            vel: Box::new(vel),
        }
    }
}

impl BilinearForm for KernelAdvDiff {
    fn integrand(
        &self,
        ctx: &LocalCtx,
        _equation: usize,
        _unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        assert_eq!(ctx.ncomp, 1, "KernelAdvDiff: scalar only (ncomp==1)");
        assert_eq!(ctx.gdim, 1, "KernelAdvDiff: 1D only");
        let gv = ctx.test(test_i, 0).grad(q, 0);
        let gu = ctx.trial(trial_i, 0).grad(q, 0);
        let u = ctx.trial(trial_i, 0).v(q);
        let nu = self.nu.eval(&ctx.material_context(None, q));
        let vel = self.vel.eval(&ctx.material_context(None, q));
        nu * gu * gv - vel * u * gv
    }
}

impl ResidualKernel for KernelAdvDiff {
    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        _equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        assert_eq!(ctx.gdim, 1, "KernelAdvDiff: 1D only");
        let gv = ctx.test(test_i, 0).grad(q, 0);
        let material = ctx.material_context(Some(state), q);
        let nu = self.nu.eval(&material);
        let vel = self.vel.eval(&material);
        nu * state.grad(0, q, 0) * gv - vel * state.value(0, q) * gv
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
        let nu = self.nu.eval(&material);
        let vel = self.vel.eval(&material);
        let dnu = self.nu.derivative(&material, unknown).unwrap_or(0.0);
        let dvel = self.vel.derivative(&material, unknown).unwrap_or(0.0);
        let test_grad = ctx.test(test_i, 0).grad(q, 0);
        (nu * ctx.trial(trial_i, 0).grad(q, 0)
            + dnu * ctx.trial(trial_i, 0).v(q) * state.grad(0, q, 0))
            * test_grad
            - (vel + dvel * state.value(0, q)) * ctx.trial(trial_i, 0).v(q) * test_grad
    }
}
