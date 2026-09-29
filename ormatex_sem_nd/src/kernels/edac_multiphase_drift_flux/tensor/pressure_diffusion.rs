//! Mixture EDAC pressure diffusion `div(k grad(p))` (2D).
//!
//! Mathematics: `(0, k d_0 p, k d_1 p)` for equation 2 with `k` the pressure
//! diffusivity (artificial sound speed times the Smagorinsky filter width),
//! identical to the base kernel but on the four-field state.
use crate::common::{LaneState, Lanes, TensorCtx, LANES};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config::{drift_field_names, DriftFlux2DConfig};

/// Tensor mixture pressure-diffusion (owns equation 2).
pub struct TensorDriftPressureDiffusion2D {
    pub config: DriftFlux2DConfig,
}
impl TensorDriftPressureDiffusion2D {
    pub fn new(config: DriftFlux2DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<2> for TensorDriftPressureDiffusion2D {
    fn nfields(&self) -> usize {
        4
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation == 2
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
    /// * `f1y` - lane y-flux slots. Overwritten.
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
        if equation != 2 {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let k = self.config.pressure_diffusivities_tensor_lanes(ctxs);
        let g0 = state.grad(2, q, 0);
        let g1 = state.grad(2, q, 1);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = k[l] * g0[l];
            f1y[l] = k[l] * g1[l];
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
    /// * `f1y` - lane linearized y-flux slots. Overwritten.
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
        if equation != 2 {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let k = self.config.pressure_diffusivities_tensor_lanes(ctxs);
        let dg0 = direction.grad(2, q, 0);
        let dg1 = direction.grad(2, q, 1);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = k[l] * dg0[l];
            f1y[l] = k[l] * dg1[l];
        }
    }
}
