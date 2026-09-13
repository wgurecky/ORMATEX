use crate::common::{CellState, LocalCtx, TensorCtx};
use crate::material::{ConstantCoefficient, MaterialProperty};

use crate::kernels::common::{BilinearForm, ResidualKernel};

/// 2D advection-diffusion kernel.
///
/// Weak form `(nu*grad(u) - vel*u).grad(v)`; the Jacobian adds the `dnu`/`dvel`
/// material-derivative terms. Tensor mirror:
/// [`TensorKernelAdvDiff2D`](crate::kernels::basic::tensor::adv_diff_2d::TensorKernelAdvDiff2D).
pub struct KernelAdvDiff2D {
    pub nu: Box<dyn MaterialProperty<f64>>,
    pub vel: [Box<dyn MaterialProperty<f64>>; 2],
    field_names: Option<Vec<String>>,
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
            field_names: None,
        }
    }

    /// Attach the single solution-field name so this 1->1 term can join a
    /// named `ResidualKernelSet` / `TensorResidualKernelSet`
    /// (e.g. one frozen-velocity transport per species).
    pub fn with_field_name(mut self, name: impl Into<String>) -> Self {
        self.field_names = Some(vec![name.into()]);
        self
    }
}

impl BilinearForm for KernelAdvDiff2D {
    fn field_names(&self) -> Option<Vec<String>> {
        self.field_names.clone()
    }
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
        [
            0.0,
            nu * trial_grad[0] - self.vel[0].eval(&material) * trial_value,
            nu * trial_grad[1] - self.vel[1].eval(&material) * trial_value,
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
    fn field_names(&self) -> Option<Vec<String>> {
        self.field_names.clone()
    }

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
