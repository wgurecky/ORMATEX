use crate::common::{CellState, LocalCtx, TensorCtx};
use crate::material::{ConstantCoefficient, MaterialProperty};

use super::kernel_common::{BilinearForm, ResidualKernel, TensorResidualKernel};

/// Scalar 2D advection kernel for `velocity . grad(u)`.
pub struct KernelAdvection2D {
    pub vel: [Box<dyn MaterialProperty<f64>>; 2],
}

impl KernelAdvection2D {
    pub fn new(vel: [f64; 2]) -> Self {
        Self::with_velocity([ConstantCoefficient(vel[0]), ConstantCoefficient(vel[1])])
    }

    pub fn with_velocity<V>(vel: [V; 2]) -> Self
    where
        V: MaterialProperty<f64> + 'static,
    {
        Self {
            vel: vel.map(|value| Box::new(value) as Box<dyn MaterialProperty<f64>>),
        }
    }
}

impl BilinearForm for KernelAdvection2D {
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
        _trial_grad: [f64; 2],
    ) -> [f64; 3] {
        let material = ctx.material_context(None, q);
        [
            0.0,
            -self.vel[0].eval(&material) * trial_value,
            -self.vel[1].eval(&material) * trial_value,
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
        assert_eq!(ctx.ncomp, 1, "KernelAdvection2D: scalar only (ncomp==1)");
        assert_eq!(ctx.gdim, 2, "KernelAdvection2D: 2D only");
        let test = ctx.test(test_i, 0);
        let trial = ctx.trial(trial_i, 0);
        let material = ctx.material_context(None, q);
        -(0..2)
            .map(|d| self.vel[d].eval(&material) * trial.v(q) * test.grad(q, d))
            .sum::<f64>()
    }
}

impl ResidualKernel for KernelAdvection2D {
    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        _equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        assert_eq!(ctx.ncomp, 1, "KernelAdvection2D: scalar only (ncomp==1)");
        assert_eq!(ctx.gdim, 2, "KernelAdvection2D: 2D only");
        let test = ctx.test(test_i, 0);
        let material = ctx.material_context(Some(state), q);
        -state.value(0, q)
            * (0..2)
                .map(|d| self.vel[d].eval(&material) * test.grad(q, d))
                .sum::<f64>()
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
        assert_eq!(ctx.ncomp, 1, "KernelAdvection2D: scalar only (ncomp==1)");
        assert_eq!(ctx.gdim, 2, "KernelAdvection2D: 2D only");
        let test = ctx.test(test_i, 0);
        let trial = ctx.trial(trial_i, 0);
        let material = ctx.material_context(Some(state), q);
        -((0..2)
            .map(|d| {
                let velocity = self.vel[d].eval(&material);
                let derivative = self.vel[d].derivative(&material, unknown).unwrap_or(0.0);
                (velocity + derivative * state.value(0, q)) * trial.v(q) * test.grad(q, d)
            })
            .sum::<f64>())
    }
}

/// Tensor-product 2D advection kernel.
pub struct TensorKernelAdvection2D(pub KernelAdvection2D);

impl TensorKernelAdvection2D {
    pub fn new(vel: [f64; 2]) -> Self {
        Self(KernelAdvection2D::new(vel))
    }
    pub fn with_velocity<V>(vel: [V; 2]) -> Self
    where
        V: MaterialProperty<f64> + 'static,
    {
        Self(KernelAdvection2D::with_velocity(vel))
    }
}

impl TensorResidualKernel<2> for TensorKernelAdvection2D {
    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        _equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let material = ctx.material_context(Some(state), q);
        let value = state.value(0, q);
        [
            0.0,
            -self.0.vel[0].eval(&material) * value,
            -self.0.vel[1].eval(&material) * value,
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
        let dv = direction.value(0, q);
        [
            0.0,
            -(self.0.vel[0].eval(&material)
                + self.0.vel[0].derivative(&material, 0).unwrap_or(0.0) * value)
                * dv,
            -(self.0.vel[1].eval(&material)
                + self.0.vel[1].derivative(&material, 0).unwrap_or(0.0) * value)
                * dv,
        ]
    }
}
