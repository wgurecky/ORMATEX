use crate::common::{CellState, LocalCtx, TensorCtx};
use crate::material::{ConstantCoefficient, MaterialProperty};

use super::kernel_common::{BilinearForm, ResidualKernel, TensorResidualKernel};

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
    fn supports_tensor_bilinear(&self) -> bool {
        true
    }

    fn tensor_bilinear(
        &self,
        ctx: &TensorCtx<'_>,
        _equation: usize,
        _unknown: usize,
        q: usize,
        trial_value: f64,
        trial_grad: [f64; 2],
    ) -> [f64; 3] {
        let material = ctx.material_context(None, q);
        let nu = self.nu.eval(&material);
        let tau = self.tau.eval(&material);
        let velocity = [self.vel[0].eval(&material), self.vel[1].eval(&material)];
        let advect_trial = velocity[0] * trial_grad[0] + velocity[1] * trial_grad[1];
        [
            0.0,
            nu * trial_grad[0] + tau * advect_trial * velocity[0] - velocity[0] * trial_value,
            nu * trial_grad[1] + tau * advect_trial * velocity[1] - velocity[1] * trial_value,
        ]
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

/// Tensor-product 2D SUPG advection-diffusion kernel.
pub struct TensorKernelAdvDiffSUPG2D(pub KernelAdvDiffSUPG2D);

impl TensorKernelAdvDiffSUPG2D {
    pub fn new(nu: f64, vel: [f64; 2], tau: f64) -> Self {
        Self(KernelAdvDiffSUPG2D::new(nu, vel, tau))
    }
    pub fn with_coefficients<N, V, T>(nu: N, vel: [V; 2], tau: T) -> Self
    where
        N: MaterialProperty<f64> + 'static,
        V: MaterialProperty<f64> + 'static,
        T: MaterialProperty<f64> + 'static,
    {
        Self(KernelAdvDiffSUPG2D::with_coefficients(nu, vel, tau))
    }
}

impl TensorResidualKernel<2> for TensorKernelAdvDiffSUPG2D {
    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        _equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let m = ctx.material_context(Some(state), q);
        let nu = self.0.nu.eval(&m);
        let tau = self.0.tau.eval(&m);
        let v = [self.0.vel[0].eval(&m), self.0.vel[1].eval(&m)];
        let value = state.value(0, q);
        let advection = v[0] * state.grad(0, q, 0) + v[1] * state.grad(0, q, 1);
        [
            0.0,
            nu * state.grad(0, q, 0) + tau * advection * v[0] - v[0] * value,
            nu * state.grad(0, q, 1) + tau * advection * v[1] - v[1] * value,
        ]
    }
    fn tensor_jacobian_action(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        _equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let m = ctx.material_context(Some(state), q);
        let nu = self.0.nu.eval(&m);
        let tau = self.0.tau.eval(&m);
        let dnu = self.0.nu.derivative(&m, 0).unwrap_or(0.0);
        let dtau = self.0.tau.derivative(&m, 0).unwrap_or(0.0);
        let v = [self.0.vel[0].eval(&m), self.0.vel[1].eval(&m)];
        let dvv = [
            self.0.vel[0].derivative(&m, 0).unwrap_or(0.0),
            self.0.vel[1].derivative(&m, 0).unwrap_or(0.0),
        ];
        let value = state.value(0, q);
        let dv = direction.value(0, q);
        let g = [state.grad(0, q, 0), state.grad(0, q, 1)];
        let dg = [direction.grad(0, q, 0), direction.grad(0, q, 1)];
        let advection = v[0] * g[0] + v[1] * g[1];
        let dadvection = dvv[0] * dv * g[0] + dvv[1] * dv * g[1] + v[0] * dg[0] + v[1] * dg[1];
        [
            0.0,
            nu * dg[0]
                + dnu * dv * g[0]
                + dtau * dv * advection * v[0]
                + tau * dadvection * v[0]
                + tau * advection * dvv[0] * dv
                - (v[0] + dvv[0] * value) * dv,
            nu * dg[1]
                + dnu * dv * g[1]
                + dtau * dv * advection * v[1]
                + tau * dadvection * v[1]
                + tau * advection * dvv[1] * dv
                - (v[1] + dvv[1] * value) * dv,
        ]
    }
}
