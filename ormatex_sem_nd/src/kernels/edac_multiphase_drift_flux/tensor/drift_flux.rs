//! Conservative Ishii-Zuber drift-flux tensor kernel (2D).
//!
//! Mathematics: relative gas flux `div(F(alpha) e)` for equation 3 with the
//! um-referenced hindered drift flux `F = a*V_um(a)` (see
//! [`DriftFlux2DConfig::slip_speed`](super::super::config::DriftFlux2DConfig::slip_speed))
//! and `e` the rise unit vector (opposite gravity; see
//! [`super::super::closures`]). Conservative triple `(0, -F e_0, -F e_1)`
//! with the exact `dF/da` action (the flux is non-monotone, so the exact sign
//! matters). Owns equation 3.
use crate::common::{LaneState, Lanes, TensorCtx, LANES};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config::{
    drift_field_names, DriftFlux2DConfig, ALPHA_2D,
};

/// Tensor conservative drift flux (owns equation 3).
pub struct TensorDriftFlux2D {
    pub config: DriftFlux2DConfig,
}
impl TensorDriftFlux2D {
    pub fn new(config: DriftFlux2DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<2> for TensorDriftFlux2D {
    fn nfields(&self) -> usize {
        4
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation == ALPHA_2D
    }

    /// Lane-packed conservative drift-flux residual for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed solution.
    /// * `equation` - void equation shared by all lanes.
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
        if equation != ALPHA_2D {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let alpha = state.value(ALPHA_2D, q);
        let f = self.config.drift_flux_lanes(alpha);
        let e = self.config.rise_direction();
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = -f[l] * e[0];
            f1y[l] = -f[l] * e[1];
        }
    }

    /// Lane-packed conservative drift-flux Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed linearization point.
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - void equation shared by all lanes.
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0` - lane linearized value slots. Overwritten.
    /// * `f1x` - lane linearized x-flux slots. Overwritten.
    /// * `f1y` - lane linearized y-flux slots. Overwritten.
    #[inline]
    fn tensor_jacobian_action(
        &self,
        ctxs: &[TensorCtx<'_>],
        state: &LaneState<'_>,
        direction: &LaneState<'_>,
        equation: usize,
        q: usize,
        f0: &mut Lanes,
        f1x: &mut Lanes,
        f1y: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        if equation != ALPHA_2D {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let alpha = state.value(ALPHA_2D, q);
        let df = self.config.drift_flux_derivative_lanes(alpha);
        let e = self.config.rise_direction();
        let da = direction.value(ALPHA_2D, q);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = -df[l] * da[l] * e[0];
            f1y[l] = -df[l] * da[l] * e[1];
        }
    }
}
