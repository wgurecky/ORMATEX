//! Tensor EDAC pressure diffusion `d(k dp/dx)/dx` for 1D `p`.
//!
//! Mathematics: for equation 1 the triple is `(0, k dp/dx, 0)` with `k` the
//! pressure diffusivity (artificial sound speed times the cell length); the
//! action is linear in the direction gradient. Owns equation 1 only. Weak
//! counterpart:
//! [`KernelEdacPressureDiffusion1D`](crate::kernels::edac::weak::pressure_diffusion_1d::KernelEdacPressureDiffusion1D).

use crate::common::{LaneState, Lanes, TensorCtx, LANES};

use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config_1d::{fluid_field_names_1d, EdacNavierStokes1DConfig};

/// Tensor pressure-diffusion kernel (sum-factorized counterpart, owns equation 1).
pub struct TensorKernelEdacPressureDiffusion1D {
    pub config: EdacNavierStokes1DConfig,
}

impl TensorKernelEdacPressureDiffusion1D {
    pub fn new(config: EdacNavierStokes1DConfig) -> Self {
        Self { config }
    }
}

impl TensorResidualKernel<1> for TensorKernelEdacPressureDiffusion1D {
    #[inline]
    fn nfields(&self) -> usize {
        2
    }
    #[inline]
    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names_1d()
    }
    #[inline]
    fn owns_equation(&self, equation: usize) -> bool {
        equation == 1
    }

    /// Lane-packed pressure-diffusion residual for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed solution.
    /// * `equation` - pressure equation shared by all lanes.
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0`/`f1x`/`f1y` - lane output slots. Overwritten.
    #[inline]
    fn tensor_residual(
        &self,
        ctxs: &[TensorCtx<'_>],
        state: &LaneState<'_>,
        equation: usize,
        q: usize,
        f0: &mut Lanes,
        f1x: &mut Lanes,
        f1y: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        if equation != 1 {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let k = self.config.pressure_diffusivities_tensor_lanes(ctxs);
        let g = state.grad(1, q, 0);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = k[l] * g[l];
            f1y[l] = 0.0;
        }
    }

    /// Lane-packed pressure-diffusion Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed linearization point (unused).
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - pressure equation shared by all lanes.
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0`/`f1x`/`f1y` - lane output slots. Overwritten.
    #[inline]
    fn tensor_jacobian_action(
        &self,
        ctxs: &[TensorCtx<'_>],
        _state: &LaneState<'_>,
        direction: &LaneState<'_>,
        equation: usize,
        q: usize,
        f0: &mut Lanes,
        f1x: &mut Lanes,
        f1y: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        if equation != 1 {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let k = self.config.pressure_diffusivities_tensor_lanes(ctxs);
        let dg = direction.grad(1, q, 0);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = k[l] * dg[l];
            f1y[l] = 0.0;
        }
    }
}
