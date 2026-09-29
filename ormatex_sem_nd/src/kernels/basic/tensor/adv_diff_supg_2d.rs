use crate::common::{LaneState, Lanes, TensorCtx, LANES};
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
    /// Lane-packed SUPG residual for all lanes.
    ///
    /// Coefficients are evaluated per lane with the real `q` via
    /// `lane_material_context`, exactly the value the scalar path sees.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed solution.
    /// * `equation` - unused (single output).
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0`/`f1x`/`f1y` - lane output slots. Overwritten.
    #[inline]
    fn tensor_residual(
        &self,
        ctxs: &[TensorCtx<'_>],
        state: &LaneState<'_>,
        _equation: usize,
        q: usize,
        f0: &mut Lanes,
        f1x: &mut Lanes,
        f1y: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        let mut nu = [0.0; LANES];
        let mut tau = [0.0; LANES];
        let mut vx = [0.0; LANES];
        let mut vy = [0.0; LANES];
        for l in 0..LANES {
            let m = ctxs[l].lane_material_context(Some(state), l, q);
            nu[l] = self.0.nu.eval(&m);
            tau[l] = self.0.tau.eval(&m);
            vx[l] = self.0.vel[0].eval(&m);
            vy[l] = self.0.vel[1].eval(&m);
        }
        let v = state.value(0, q);
        let gx = state.grad(0, q, 0);
        let gy = state.grad(0, q, 1);
        for l in 0..LANES {
            let advection = vx[l] * gx[l] + vy[l] * gy[l];
            f0[l] = 0.0;
            f1x[l] = nu[l] * gx[l] + tau[l] * advection * vx[l] - vx[l] * v[l];
            f1y[l] = nu[l] * gy[l] + tau[l] * advection * vy[l] - vy[l] * v[l];
        }
    }

    /// Lane-packed SUPG Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed linearization point.
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - unused (single output).
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0`/`f1x`/`f1y` - lane output slots. Overwritten.
    #[inline]
    fn tensor_jacobian_action(
        &self,
        ctxs: &[TensorCtx<'_>],
        state: &LaneState<'_>,
        direction: &LaneState<'_>,
        _equation: usize,
        q: usize,
        f0: &mut Lanes,
        f1x: &mut Lanes,
        f1y: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        let mut nu = [0.0; LANES];
        let mut tau = [0.0; LANES];
        let mut dnu = [0.0; LANES];
        let mut dtau = [0.0; LANES];
        let mut vx = [0.0; LANES];
        let mut vy = [0.0; LANES];
        let mut dvx = [0.0; LANES];
        let mut dvy = [0.0; LANES];
        for l in 0..LANES {
            let m = ctxs[l].lane_material_context(Some(state), l, q);
            nu[l] = self.0.nu.eval(&m);
            tau[l] = self.0.tau.eval(&m);
            dnu[l] = self.0.nu.derivative(&m, 0).unwrap_or(0.0);
            dtau[l] = self.0.tau.derivative(&m, 0).unwrap_or(0.0);
            vx[l] = self.0.vel[0].eval(&m);
            vy[l] = self.0.vel[1].eval(&m);
            dvx[l] = self.0.vel[0].derivative(&m, 0).unwrap_or(0.0);
            dvy[l] = self.0.vel[1].derivative(&m, 0).unwrap_or(0.0);
        }
        let v = state.value(0, q);
        let gx = state.grad(0, q, 0);
        let gy = state.grad(0, q, 1);
        let dv = direction.value(0, q);
        let dgx = direction.grad(0, q, 0);
        let dgy = direction.grad(0, q, 1);
        for l in 0..LANES {
            let advection = vx[l] * gx[l] + vy[l] * gy[l];
            let dadvection =
                dvx[l] * dv[l] * gx[l] + dvy[l] * dv[l] * gy[l] + vx[l] * dgx[l] + vy[l] * dgy[l];
            f0[l] = 0.0;
            f1x[l] = nu[l] * dgx[l]
                + dnu[l] * dv[l] * gx[l]
                + dtau[l] * dv[l] * advection * vx[l]
                + tau[l] * dadvection * vx[l]
                + tau[l] * advection * dvx[l] * dv[l]
                - (vx[l] + dvx[l] * v[l]) * dv[l];
            f1y[l] = nu[l] * dgy[l]
                + dnu[l] * dv[l] * gy[l]
                + dtau[l] * dv[l] * advection * vy[l]
                + tau[l] * dadvection * vy[l]
                + tau[l] * advection * dvy[l] * dv[l]
                - (vy[l] + dvy[l] * v[l]) * dv[l];
        }
    }
}
