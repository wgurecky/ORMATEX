use crate::common::{CellState, TensorCtx};
use crate::material::MaterialProperty;

use crate::kernels::common::TensorResidualKernel;

use crate::kernels::basic::weak::adv_diff_supg_1d::KernelAdvDiffSUPG;

/// Tensor SUPG advection-diffusion for a scalar field.
///
/// Mathematics: streamline stabilization folds into the flux slot, giving
/// `(0, (nu + tau*vel^2)*u' - vel*u, 0)`; the action differentiates `nu`,
/// `vel`, and `tau` (full product rule). Weak counterpart:
/// [`KernelAdvDiffSUPG`].
/// Tensor-product 1D SUPG advection-diffusion kernel (sum-factorized counterpart).
pub struct TensorKernelAdvDiffSUPG(pub KernelAdvDiffSUPG);

impl TensorKernelAdvDiffSUPG {
    pub fn new(nu: f64, vel: f64, tau: f64) -> Self {
        Self(KernelAdvDiffSUPG::new(nu, vel, tau))
    }
    pub fn with_coefficients<N, V, T>(nu: N, vel: V, tau: T) -> Self
    where
        N: MaterialProperty<f64> + 'static,
        V: MaterialProperty<f64> + 'static,
        T: MaterialProperty<f64> + 'static,
    {
        Self(KernelAdvDiffSUPG::with_coefficients(nu, vel, tau))
    }
}

impl TensorResidualKernel<1> for TensorKernelAdvDiffSUPG {
    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        _equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let m = ctx.material_context(Some(state), q);
        let nu = self.0.nu.eval(&m);
        let vel = self.0.vel.eval(&m);
        let tau = self.0.tau.eval(&m);
        [
            0.0,
            (nu + tau * vel * vel) * state.grad(0, q, 0) - vel * state.value(0, q),
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
        let m = ctx.material_context(Some(state), q);
        let nu = self.0.nu.eval(&m);
        let vel = self.0.vel.eval(&m);
        let tau = self.0.tau.eval(&m);
        let value = state.value(0, q);
        let dv = direction.value(0, q);
        let gradient = state.grad(0, q, 0);
        let dg = direction.grad(0, q, 0);
        let dnu = self.0.nu.derivative(&m, 0).unwrap_or(0.0);
        let dvel = self.0.vel.derivative(&m, 0).unwrap_or(0.0);
        let dtau = self.0.tau.derivative(&m, 0).unwrap_or(0.0);
        [
            0.0,
            (nu + tau * vel * vel) * dg
                + (dnu + dtau * vel * vel + 2.0 * tau * vel * dvel) * dv * gradient
                - (vel + dvel * value) * dv,
            0.0,
        ]
    }
}
