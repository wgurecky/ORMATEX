use crate::common::{LaneState, Lanes, TensorCtx, LANES};
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
    pub fn with_field_name(self, name: impl Into<String>) -> Self {
        Self(self.0.with_field_name(name))
    }
}

impl TensorResidualKernel<1> for TensorKernelAdvDiff {
    #[inline]
    fn field_names(&self) -> Option<Vec<String>> {
        <KernelAdvDiff as crate::kernels::common::ResidualKernel>::field_names(&self.0)
    }

    /// Lane-packed advection-diffusion residual for all lanes.
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
        for l in 0..LANES {
            let material = ctxs[l].lane_material_context(Some(state), l, q);
            nu[l] = self.0.nu.eval(&material);
            vel[l] = self.0.vel.eval(&material);
        }
        let g = state.grad(0, q, 0);
        let v = state.value(0, q);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = nu[l] * g[l] - vel[l] * v[l];
            f1y[l] = 0.0;
        }
    }

    /// Lane-packed advection-diffusion Jacobian action for all lanes.
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
        let mut dnu = [0.0; LANES];
        let mut dvel = [0.0; LANES];
        for l in 0..LANES {
            let material = ctxs[l].lane_material_context(Some(state), l, q);
            nu[l] = self.0.nu.eval(&material);
            vel[l] = self.0.vel.eval(&material);
            dnu[l] = self.0.nu.derivative(&material, 0).unwrap_or(0.0);
            dvel[l] = self.0.vel.derivative(&material, 0).unwrap_or(0.0);
        }
        let v = state.value(0, q);
        let g = state.grad(0, q, 0);
        let dv = direction.value(0, q);
        let dg = direction.grad(0, q, 0);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = nu[l] * dg[l] + dnu[l] * dv[l] * g[l] - (vel[l] + dvel[l] * v[l]) * dv[l];
            f1y[l] = 0.0;
        }
    }
}
