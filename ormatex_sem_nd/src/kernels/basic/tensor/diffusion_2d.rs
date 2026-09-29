use crate::common::{LaneState, Lanes, TensorCtx, LANES};
use crate::material::MaterialProperty;

use crate::kernels::common::TensorResidualKernel;

use crate::kernels::basic::weak::diffusion_2d::KernelDiffusion2D;

/// Tensor diffusion `-div(nu grad(u))` for a scalar field.
///
/// Mathematics: the triple is `(0, nu*dx(u), nu*dy(u))`; the action adds the
/// `dnu` material-derivative terms. Serves the `BilinearForm` fast path.
/// Weak counterpart: [`KernelDiffusion2D`].
/// Tensor-product 2D diffusion kernel (sum-factorized counterpart).
pub struct TensorKernelDiffusion2D(pub KernelDiffusion2D);

impl TensorKernelDiffusion2D {
    pub fn new(nu: f64) -> Self {
        Self(KernelDiffusion2D::new(nu))
    }
    pub fn with_coefficient<N>(nu: N) -> Self
    where
        N: MaterialProperty<f64> + 'static,
    {
        Self(KernelDiffusion2D::with_coefficient(nu))
    }
}

impl TensorResidualKernel<2> for TensorKernelDiffusion2D {
    /// Lane-packed diffusion residual for all lanes.
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
        for l in 0..LANES {
            nu[l] = self
                .0
                .nu
                .eval(&ctxs[l].lane_material_context(Some(state), l, q));
        }
        let gx = state.grad(0, q, 0);
        let gy = state.grad(0, q, 1);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = nu[l] * gx[l];
            f1y[l] = nu[l] * gy[l];
        }
    }

    /// Lane-packed diffusion Jacobian action for all lanes.
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
        for l in 0..LANES {
            let material = ctxs[l].lane_material_context(Some(state), l, q);
            nu[l] = self.0.nu.eval(&material);
            dnu[l] = self.0.nu.derivative(&material, 0).unwrap_or(0.0);
        }
        let gx = state.grad(0, q, 0);
        let gy = state.grad(0, q, 1);
        let dgx = direction.grad(0, q, 0);
        let dgy = direction.grad(0, q, 1);
        let du = direction.value(0, q);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = nu[l] * dgx[l] + dnu[l] * du[l] * gx[l];
            f1y[l] = nu[l] * dgy[l] + dnu[l] * du[l] * gy[l];
        }
    }
}
