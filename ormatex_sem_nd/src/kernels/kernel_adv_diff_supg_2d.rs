use crate::common::{CellState, LocalCtx};
use crate::material::{ConstantCoefficient, MaterialProperty};

use super::kernel_common::{BilinearForm, ResidualKernel};

/// 2D advection-diffusion kernel with SUPG stabilization.
///
/// The per-quadrature-point form is
/// `nu * grad(u) . grad(v) - u * vel . grad(v)` plus
/// `tau * (vel . grad(u)) * (vel . grad(v))`.
pub struct KernelAdvDiffSUPG2D {
    pub nu: Box<dyn MaterialProperty<f64>>,
    pub vel: [Box<dyn MaterialProperty<f64>>; 2],
    pub tau: Box<dyn MaterialProperty<f64>>,
}

impl KernelAdvDiffSUPG2D {
    pub fn new(nu: f64, vel: [f64; 2], tau: f64) -> Self {
        Self::with_coefficients(
            ConstantCoefficient(nu),
            [ConstantCoefficient(vel[0]), ConstantCoefficient(vel[1])],
            ConstantCoefficient(tau),
        )
    }

    pub fn with_coefficients<N, V, T>(nu: N, vel: [V; 2], tau: T) -> Self
    where
        N: MaterialProperty<f64> + 'static,
        V: MaterialProperty<f64> + 'static,
        T: MaterialProperty<f64> + 'static,
    {
        Self {
            nu: Box::new(nu),
            vel: vel.map(|value| Box::new(value) as Box<dyn MaterialProperty<f64>>),
            tau: Box::new(tau),
        }
    }
}

impl BilinearForm for KernelAdvDiffSUPG2D {
    fn integrand(
        &self,
        ctx: &LocalCtx,
        _equation: usize,
        _unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        assert_eq!(ctx.ncomp, 1, "KernelAdvDiffSUPG2D: scalar only (ncomp==1)");
        assert_eq!(ctx.gdim, 2, "KernelAdvDiffSUPG2D: 2D only (gdim==2)");
        let test = ctx.test(test_i, 0);
        let trial = ctx.trial(trial_i, 0);
        let grad_dot: f64 = (0..2).map(|d| trial.grad(q, d) * test.grad(q, d)).sum();
        let material = ctx.material_context(None, q);
        let nu = self.nu.eval(&material);
        let tau = self.tau.eval(&material);
        let advect_test: f64 = (0..2)
            .map(|d| self.vel[d].eval(&material) * test.grad(q, d))
            .sum();
        let advect_trial: f64 = (0..2)
            .map(|d| self.vel[d].eval(&material) * trial.grad(q, d))
            .sum();
        nu * grad_dot - trial.v(q) * advect_test + tau * advect_trial * advect_test
    }
}

impl ResidualKernel for KernelAdvDiffSUPG2D {
    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        _equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        assert_eq!(ctx.ncomp, 1, "KernelAdvDiffSUPG2D: scalar only (ncomp==1)");
        assert_eq!(ctx.gdim, 2, "KernelAdvDiffSUPG2D: 2D only (gdim==2)");
        let test = ctx.test(test_i, 0);
        let diffusion: f64 = (0..2).map(|d| state.grad(0, q, d) * test.grad(q, d)).sum();
        let material = ctx.material_context(Some(state), q);
        let nu = self.nu.eval(&material);
        let tau = self.tau.eval(&material);
        let advect_test: f64 = (0..2)
            .map(|d| self.vel[d].eval(&material) * test.grad(q, d))
            .sum();
        let advect_state: f64 = (0..2)
            .map(|d| self.vel[d].eval(&material) * state.grad(0, q, d))
            .sum();
        nu * diffusion - state.value(0, q) * advect_test + tau * advect_state * advect_test
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
        let tau = self.tau.eval(&material);
        let dnu = self.nu.derivative(&material, unknown).unwrap_or(0.0);
        let dtau = self.tau.derivative(&material, unknown).unwrap_or(0.0);
        let test = ctx.test(test_i, 0);
        let trial = ctx.trial(trial_i, 0);
        let mut vel = [0.0; 2];
        let mut dvel = [0.0; 2];
        for d in 0..2 {
            vel[d] = self.vel[d].eval(&material);
            dvel[d] = self.vel[d].derivative(&material, unknown).unwrap_or(0.0);
        }
        let grad_dot: f64 = (0..2).map(|d| trial.grad(q, d) * test.grad(q, d)).sum();
        let state_diffusion: f64 = (0..2).map(|d| state.grad(0, q, d) * test.grad(q, d)).sum();
        let advect_test: f64 = (0..2).map(|d| vel[d] * test.grad(q, d)).sum();
        let d_advect_test: f64 = (0..2).map(|d| dvel[d] * test.grad(q, d)).sum();
        let advect_trial: f64 = (0..2).map(|d| vel[d] * trial.grad(q, d)).sum();
        let d_advect_state: f64 = (0..2).map(|d| dvel[d] * state.grad(0, q, d)).sum();
        let advect_state: f64 = (0..2).map(|d| vel[d] * state.grad(0, q, d)).sum();
        let d_tau = dtau * trial.v(q);
        let d_advect_trial_state = advect_trial + d_advect_state * trial.v(q);
        nu * grad_dot + dnu * trial.v(q) * state_diffusion
            - trial.v(q) * advect_test
            - d_advect_test * trial.v(q) * state.value(0, q)
            + tau * (d_advect_trial_state * advect_test + advect_state * d_advect_test * trial.v(q))
            + d_tau * advect_state * advect_test
    }
}
