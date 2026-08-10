use crate::common::{CellState, LocalCtx};
use crate::material::{ConstantCoefficient, MaterialProperty};

use super::kernel_common::{BilinearForm, ResidualKernel};

/// 1D advection-diffusion kernel with SUPG stabilization.
pub struct KernelAdvDiffSUPG {
    pub nu: Box<dyn MaterialProperty<f64>>,
    pub vel: Box<dyn MaterialProperty<f64>>,
    pub tau: Box<dyn MaterialProperty<f64>>,
}

impl KernelAdvDiffSUPG {
    pub fn new(nu: f64, vel: f64, tau: f64) -> Self {
        Self::with_coefficients(
            ConstantCoefficient(nu),
            ConstantCoefficient(vel),
            ConstantCoefficient(tau),
        )
    }

    pub fn with_coefficients<N, V, T>(nu: N, vel: V, tau: T) -> Self
    where
        N: MaterialProperty<f64> + 'static,
        V: MaterialProperty<f64> + 'static,
        T: MaterialProperty<f64> + 'static,
    {
        Self {
            nu: Box::new(nu),
            vel: Box::new(vel),
            tau: Box::new(tau),
        }
    }
}

impl BilinearForm for KernelAdvDiffSUPG {
    fn integrand(
        &self,
        ctx: &LocalCtx,
        _equation: usize,
        _unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        assert_eq!(ctx.ncomp, 1, "KernelAdvDiffSUPG: scalar only (ncomp==1)");
        assert_eq!(ctx.gdim, 1, "KernelAdvDiffSUPG: 1D only");
        let material = ctx.material_context(None, q);
        let nu = self.nu.eval(&material);
        let vel = self.vel.eval(&material);
        let tau = self.tau.eval(&material);
        let supg = tau * vel * vel;
        let gv = ctx.test(test_i, 0).grad(q, 0);
        let gu = ctx.trial(trial_i, 0).grad(q, 0);
        let u = ctx.trial(trial_i, 0).v(q);
        (nu + supg) * gu * gv - vel * u * gv
    }
}

impl ResidualKernel for KernelAdvDiffSUPG {
    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        _equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        assert_eq!(ctx.gdim, 1, "KernelAdvDiffSUPG: 1D only");
        let gv = ctx.test(test_i, 0).grad(q, 0);
        let material = ctx.material_context(Some(state), q);
        let nu = self.nu.eval(&material);
        let vel = self.vel.eval(&material);
        let tau = self.tau.eval(&material);
        (nu + tau * vel * vel) * state.grad(0, q, 0) * gv - vel * state.value(0, q) * gv
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
        let tau = self.tau.eval(&material);
        let dnu = self.nu.derivative(&material, unknown).unwrap_or(0.0);
        let dvel = self.vel.derivative(&material, unknown).unwrap_or(0.0);
        let dtau = self.tau.derivative(&material, unknown).unwrap_or(0.0);
        let trial = ctx.trial(trial_i, 0);
        let test_grad = ctx.test(test_i, 0).grad(q, 0);
        let grad_u = state.grad(0, q, 0);
        let value_u = state.value(0, q);
        let dstabilized = dtau * vel * vel + 2.0 * tau * vel * dvel;
        (nu * trial.grad(q, 0)
            + tau * vel * vel * trial.grad(q, 0)
            + dnu * trial.v(q) * grad_u
            + dstabilized * trial.v(q) * grad_u)
            * test_grad
            - (vel + dvel * value_u) * trial.v(q) * test_grad
    }
}
