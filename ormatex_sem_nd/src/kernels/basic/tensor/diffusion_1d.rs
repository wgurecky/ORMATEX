use crate::common::{LaneState, Lanes, TensorCtx, LANES};
use crate::material::MaterialProperty;

use crate::kernels::common::TensorResidualKernel;

use crate::kernels::basic::weak::diffusion_1d::KernelDiffusion;

/// Tensor diffusion `d/dx(nu du/dx)` for a scalar field.
///
/// Mathematics: the `(f0, f1x, f1y)` triple is `(0, nu*u', 0)`; the Jacobian
/// action is `(0, nu*du' + dnu*du*u', 0)` with `nu` and its state derivative
/// from the material context. Weak counterpart: [`KernelDiffusion`].
/// Tensor-product 1D diffusion kernel (sum-factorized counterpart).
pub struct TensorKernelDiffusion(pub KernelDiffusion);

impl TensorKernelDiffusion {
    pub fn new(nu: f64) -> Self {
        Self(KernelDiffusion::new(nu))
    }
    pub fn with_coefficient<N>(nu: N) -> Self
    where
        N: MaterialProperty<f64> + 'static,
    {
        Self(KernelDiffusion::with_coefficient(nu))
    }
}

impl TensorResidualKernel<1> for TensorKernelDiffusion {
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
        let g = state.grad(0, q, 0);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = nu[l] * g[l];
            f1y[l] = 0.0;
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
        let g = state.grad(0, q, 0);
        let dg = direction.grad(0, q, 0);
        let dv = direction.value(0, q);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = nu[l] * dg[l] + dnu[l] * dv[l] * g[l];
            f1y[l] = 0.0;
        }
    }
}
