use crate::common::{CellState, LocalCtx};
use crate::material::{ConstantCoefficient, MaterialProperty};

use super::kernel_common::{BilinearForm, ResidualKernel};

/// 2D advection-diffusion kernel.
pub struct KernelAdvDiff2D {
    pub nu: Box<dyn MaterialProperty<f64>>,
    pub vel: [Box<dyn MaterialProperty<f64>>; 2],
}

impl KernelAdvDiff2D {
    pub fn new(nu: f64, vel: [f64; 2]) -> Self {
        Self::with_coefficients(
            ConstantCoefficient(nu),
            [ConstantCoefficient(vel[0]), ConstantCoefficient(vel[1])],
        )
    }

    pub fn with_coefficients<N, V>(nu: N, vel: [V; 2]) -> Self
    where
        N: MaterialProperty<f64> + 'static,
        V: MaterialProperty<f64> + 'static,
    {
        Self {
            nu: Box::new(nu),
            vel: vel.map(|value| Box::new(value) as Box<dyn MaterialProperty<f64>>),
        }
    }
}

impl BilinearForm for KernelAdvDiff2D {
    fn integrand(
        &self,
        ctx: &LocalCtx,
        _equation: usize,
        _unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        assert_eq!(ctx.ncomp, 1, "KernelAdvDiff2D: scalar only (ncomp==1)");
        assert_eq!(ctx.gdim, 2, "KernelAdvDiff2D: 2D only (gdim==2)");
        let gv = ctx.test(test_i, 0);
        let gu = ctx.trial(trial_i, 0);
        let u = ctx.trial(trial_i, 0).v(q);
        let grad_dot: f64 = (0..2).map(|d| gu.grad(q, d) * gv.grad(q, d)).sum();
        let material = ctx.material_context(None, q);
        let nu = self.nu.eval(&material);
        let advect: f64 = (0..2)
            .map(|d| self.vel[d].eval(&material) * u * gv.grad(q, d))
            .sum();
        nu * grad_dot - advect
    }
}

impl ResidualKernel for KernelAdvDiff2D {
    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        _equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        assert_eq!(ctx.ncomp, 1, "KernelAdvDiff2D: scalar only (ncomp==1)");
        assert_eq!(ctx.gdim, 2, "KernelAdvDiff2D: 2D only (gdim==2)");
        let gv = ctx.test(test_i, 0);
        let material = ctx.material_context(Some(state), q);
        let nu = self.nu.eval(&material);
        let diffusion: f64 = (0..2).map(|d| state.grad(0, q, d) * gv.grad(q, d)).sum();
        let advection: f64 = (0..2)
            .map(|d| self.vel[d].eval(&material) * state.value(0, q) * gv.grad(q, d))
            .sum();
        nu * diffusion - advection
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
        let dnu = self.nu.derivative(&material, unknown).unwrap_or(0.0);
        let test = ctx.test(test_i, 0);
        let trial = ctx.trial(trial_i, 0);
        let diffusion: f64 = (0..2).map(|d| trial.grad(q, d) * test.grad(q, d)).sum();
        let state_diffusion: f64 = (0..2).map(|d| state.grad(0, q, d) * test.grad(q, d)).sum();
        let mut result = nu * diffusion + dnu * trial.v(q) * state_diffusion;
        for d in 0..2 {
            let vel = self.vel[d].eval(&material);
            let dvel = self.vel[d].derivative(&material, unknown).unwrap_or(0.0);
            result -= (vel + dvel * state.value(0, q)) * trial.v(q) * test.grad(q, d);
        }
        result
    }
}
