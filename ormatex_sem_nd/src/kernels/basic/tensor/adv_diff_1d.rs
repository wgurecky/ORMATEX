use crate::common::{CellState, TensorCtx};
use crate::material::MaterialProperty;

use crate::kernels::common::TensorResidualKernel;

use crate::kernels::basic::weak::adv_diff_1d::KernelAdvDiff;

/// Tensor advection-diffusion for a scalar field.
///
/// Mathematics: the triple is `(0, nu*u' - vel*u, 0)`; the action carries the
/// `dnu`/`dvel` material-derivative product-rule terms. Weak counterpart:
/// [`KernelAdvDiff`].
/// Tensor-product 1D advection-diffusion kernel (sum-factorized counterpart).
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
