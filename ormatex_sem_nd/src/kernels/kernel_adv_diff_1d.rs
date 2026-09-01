use crate::common::{CellState, LocalCtx, TensorCtx};
use crate::material::{ConstantCoefficient, MaterialProperty};

use super::kernel_common::{BilinearForm, ResidualKernel, TensorResidualKernel};

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
    fn supports_tensor_bilinear_1d(&self) -> bool {
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
        [
            0.0,
            self.nu.eval(&material) * trial_grad[0] - self.vel.eval(&material) * trial_value,
            0.0,
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

/// Tensor-product 1D advection-diffusion kernel.
pub struct TensorKernelAdvDiff(pub KernelAdvDiff);

impl TensorKernelAdvDiff {
    pub fn new(nu: f64, vel: f64) -> Self {
        Self(KernelAdvDiff::new(nu, vel))
    }
    pub fn with_coefficients<N, V>(nu: N, vel: V) -> Self
    where
        N: MaterialProperty<f64> + 'static,
        V: MaterialProperty<f64> + 'static,
    {
        Self(KernelAdvDiff::with_coefficients(nu, vel))
    }
}

impl TensorResidualKernel<1> for TensorKernelAdvDiff {
    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        _equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let material = ctx.material_context(Some(state), q);
        [
            0.0,
            self.0.nu.eval(&material) * state.grad(0, q, 0)
                - self.0.vel.eval(&material) * state.value(0, q),
            0.0,
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
        let material = ctx.material_context(Some(state), q);
        let value = state.value(0, q);
        let direction_value = direction.value(0, q);
        let nu = self.0.nu.eval(&material);
        let vel = self.0.vel.eval(&material);
        [
            0.0,
            nu * direction.grad(0, q, 0)
                + self.0.nu.derivative(&material, 0).unwrap_or(0.0)
                    * direction_value
                    * state.grad(0, q, 0)
                - (vel + self.0.vel.derivative(&material, 0).unwrap_or(0.0) * value)
                    * direction_value,
            0.0,
        ]
    }
}
