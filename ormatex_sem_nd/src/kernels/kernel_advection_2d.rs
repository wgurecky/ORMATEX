use crate::common::{CellState, LocalCtx};
use crate::material::{ConstantCoefficient, MaterialProperty};

use super::kernel_common::{BilinearForm, ResidualKernel};

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
