use crate::common::{LaneState, Lanes, TensorCtx, LANES};
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
        let mut vel = [0.0; LANES];
        let mut tau = [0.0; LANES];
        for l in 0..LANES {
            let m = ctxs[l].lane_material_context(Some(state), l, q);
            nu[l] = self.0.nu.eval(&m);
            vel[l] = self.0.vel.eval(&m);
            tau[l] = self.0.tau.eval(&m);
        }
        let g = state.grad(0, q, 0);
        let v = state.value(0, q);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = (nu[l] + tau[l] * vel[l] * vel[l]) * g[l] - vel[l] * v[l];
            f1y[l] = 0.0;
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
        let mut vel = [0.0; LANES];
        let mut tau = [0.0; LANES];
        let mut dnu = [0.0; LANES];
        let mut dvel = [0.0; LANES];
        let mut dtau = [0.0; LANES];
        for l in 0..LANES {
            let m = ctxs[l].lane_material_context(Some(state), l, q);
            nu[l] = self.0.nu.eval(&m);
            vel[l] = self.0.vel.eval(&m);
            tau[l] = self.0.tau.eval(&m);
            dnu[l] = self.0.nu.derivative(&m, 0).unwrap_or(0.0);
            dvel[l] = self.0.vel.derivative(&m, 0).unwrap_or(0.0);
            dtau[l] = self.0.tau.derivative(&m, 0).unwrap_or(0.0);
        }
        let v = state.value(0, q);
        let g = state.grad(0, q, 0);
        let dv = direction.value(0, q);
        let dg = direction.grad(0, q, 0);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = (nu[l] + tau[l] * vel[l] * vel[l]) * dg[l]
                + (dnu[l] + dtau[l] * vel[l] * vel[l] + 2.0 * tau[l] * vel[l] * dvel[l])
                    * dv[l]
                    * g[l]
                - (vel[l] + dvel[l] * v[l]) * dv[l];
            f1y[l] = 0.0;
        }
    }
}
