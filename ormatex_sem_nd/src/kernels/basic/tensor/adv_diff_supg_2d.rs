use crate::common::{CellState, TensorCtx};
use crate::material::MaterialProperty;

use crate::kernels::common::TensorResidualKernel;

use crate::kernels::basic::weak::adv_diff_supg_2d::KernelAdvDiffSUPG2D;

/// Tensor SUPG advection-diffusion for a scalar field.
///
/// Mathematics: with `adv = vel.grad(u)`, the flux slots are
/// `nu*grad(u) + tau*adv*vel - vel*u`; the action differentiates `nu`, `tau`,
/// and `vel` (full product rule). Weak counterpart:
/// [`KernelAdvDiffSUPG2D`].
/// Tensor-product 2D SUPG advection-diffusion kernel (sum-factorized counterpart).
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
