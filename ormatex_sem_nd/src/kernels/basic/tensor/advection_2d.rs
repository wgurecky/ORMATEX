use crate::common::{LaneState, Lanes, TensorCtx, LANES};
use crate::material::MaterialProperty;

use crate::kernels::common::TensorResidualKernel;

use crate::kernels::basic::weak::advection_2d::KernelAdvection2D;

/// Tensor conservative advection for a scalar field.
///
/// Mathematics: the triple is `(0, -velx*u, -vely*u)`, i.e. the `-(vel u)`
/// flux whose weak form is `-(vel*u).grad(v)`; the action adds the `dvel`
/// material-derivative terms. Weak counterpart:
/// [`KernelAdvection2D`].
/// Tensor-product 2D advection kernel (sum-factorized counterpart).
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
    /// Lane-packed advection residual for all lanes.
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
        let mut vx = [0.0; LANES];
        let mut vy = [0.0; LANES];
        for l in 0..LANES {
            let material = ctxs[l].lane_material_context(Some(state), l, q);
            vx[l] = self.0.vel[0].eval(&material);
            vy[l] = self.0.vel[1].eval(&material);
        }
        let v = state.value(0, q);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = -vx[l] * v[l];
            f1y[l] = -vy[l] * v[l];
        }
    }

    /// Lane-packed advection Jacobian action for all lanes.
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
        let mut vx = [0.0; LANES];
        let mut vy = [0.0; LANES];
        let mut dvx = [0.0; LANES];
        let mut dvy = [0.0; LANES];
        for l in 0..LANES {
            let material = ctxs[l].lane_material_context(Some(state), l, q);
            vx[l] = self.0.vel[0].eval(&material);
            vy[l] = self.0.vel[1].eval(&material);
            dvx[l] = self.0.vel[0].derivative(&material, 0).unwrap_or(0.0);
            dvy[l] = self.0.vel[1].derivative(&material, 0).unwrap_or(0.0);
        }
        let v = state.value(0, q);
        let dv = direction.value(0, q);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = -(vx[l] + dvx[l] * v[l]) * dv[l];
            f1y[l] = -(vy[l] + dvy[l] * v[l]) * dv[l];
        }
    }
}
