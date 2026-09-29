//! Mixture EDAC pressure diffusion `d(k dp/dx)/dx` (1D).
//!
//! Mathematics: `(0, k dp/dx, 0)` for equation 1 with `k` from the cell
//! length; owns equation 1 only.
use crate::common::{LaneState, Lanes, TensorCtx, LANES};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config_1d::{
    drift_field_names_1d, DriftFlux1DConfig,
};

/// Tensor mixture pressure-diffusion (owns equation 1).
pub struct TensorDriftPressureDiffusion1D {
    pub config: DriftFlux1DConfig,
}
impl TensorDriftPressureDiffusion1D {
    pub fn new(config: DriftFlux1DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<1> for TensorDriftPressureDiffusion1D {
    fn nfields(&self) -> usize {
        3
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names_1d()
    }
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
    /// * `f0` - lane value slots. Overwritten.
    /// * `f1x` - lane x-flux slots. Overwritten.
    /// * `f1y` - lane y-flux slots. Overwritten with `0.0` (unused in 1D).
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
    /// * `f0` - lane linearized value slots. Overwritten.
    /// * `f1x` - lane linearized x-flux slots. Overwritten.
    /// * `f1y` - lane linearized y-flux slots. Overwritten with `0.0` (unused in 1D).
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
