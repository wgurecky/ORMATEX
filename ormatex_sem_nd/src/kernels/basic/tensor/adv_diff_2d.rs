use crate::common::{LaneState, Lanes, TensorCtx, LANES};
use crate::material::MaterialProperty;

use crate::kernels::common::TensorResidualKernel;

use crate::kernels::basic::weak::adv_diff_2d::KernelAdvDiff2D;

/// Tensor advection-diffusion for a scalar field.
///
/// Mathematics: the triple is `(0, nu*dx(u) - velx*u, nu*dy(u) - vely*u)`;
/// the action carries the `dnu`/`dvel` material-derivative product-rule
/// terms. Weak counterpart: [`KernelAdvDiff2D`].
/// Tensor-product 2D advection-diffusion kernel (sum-factorized counterpart).
pub struct TensorKernelAdvDiff2D(pub KernelAdvDiff2D);

impl TensorKernelAdvDiff2D {
    pub fn new(nu: f64, vel: [f64; 2]) -> Self {
        Self(KernelAdvDiff2D::new(nu, vel))
    }
    pub fn with_coefficients<N, V>(nu: N, vel: [V; 2]) -> Self
    where
        N: MaterialProperty<f64> + 'static,
        V: MaterialProperty<f64> + 'static,
    {
        Self(KernelAdvDiff2D::with_coefficients(nu, vel))
    }
    pub fn with_field_name(self, name: impl Into<String>) -> Self {
        Self(self.0.with_field_name(name))
    }
}

impl TensorResidualKernel<2> for TensorKernelAdvDiff2D {
    #[inline]
    fn field_names(&self) -> Option<Vec<String>> {
        <KernelAdvDiff2D as crate::kernels::common::ResidualKernel>::field_names(&self.0)
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
        let mut vx = [0.0; LANES];
        let mut vy = [0.0; LANES];
        for l in 0..LANES {
            let material = ctxs[l].lane_material_context(Some(state), l, q);
            nu[l] = self.0.nu.eval(&material);
            vx[l] = self.0.vel[0].eval(&material);
            vy[l] = self.0.vel[1].eval(&material);
        }
        let v = state.value(0, q);
        let gx = state.grad(0, q, 0);
        let gy = state.grad(0, q, 1);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = nu[l] * gx[l] - vx[l] * v[l];
            f1y[l] = nu[l] * gy[l] - vy[l] * v[l];
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
        let mut dnu = [0.0; LANES];
        let mut vx = [0.0; LANES];
        let mut vy = [0.0; LANES];
        let mut dvx = [0.0; LANES];
        let mut dvy = [0.0; LANES];
        for l in 0..LANES {
            let material = ctxs[l].lane_material_context(Some(state), l, q);
            nu[l] = self.0.nu.eval(&material);
            dnu[l] = self.0.nu.derivative(&material, 0).unwrap_or(0.0);
            vx[l] = self.0.vel[0].eval(&material);
            vy[l] = self.0.vel[1].eval(&material);
            dvx[l] = self.0.vel[0].derivative(&material, 0).unwrap_or(0.0);
            dvy[l] = self.0.vel[1].derivative(&material, 0).unwrap_or(0.0);
        }
        let v = state.value(0, q);
        let gx = state.grad(0, q, 0);
        let gy = state.grad(0, q, 1);
        let dv = direction.value(0, q);
        let dgx = direction.grad(0, q, 0);
        let dgy = direction.grad(0, q, 1);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = nu[l] * dgx[l] + dnu[l] * dv[l] * gx[l] - (vx[l] + dvx[l] * v[l]) * dv[l];
            f1y[l] = nu[l] * dgy[l] + dnu[l] * dv[l] * gy[l] - (vy[l] + dvy[l] * v[l]) * dv[l];
        }
    }
}
