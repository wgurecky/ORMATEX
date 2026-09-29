//! Buoyant gravity body-force tensor kernel (2D).
//!
//! Mathematics: rectangular-equivalent source `(rho_m - rho_l)/rho_m * g_i`
//! for momentum equations 0-1. The buoyant form vanishes at `alpha = 0` so
//! the low-void limit recovers the gravity-free base EDAC solution; the
//! action carries the exact `drho_m/dalpha` linearization. Owns equations 0-1.
use crate::common::{LaneState, Lanes, TensorCtx, LANES};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config::{
    drift_field_names, DriftFlux2DConfig, ALPHA_2D,
};

/// Tensor buoyant gravity source (owns equations 0-1).
pub struct TensorDriftGravity2D {
    pub config: DriftFlux2DConfig,
}
impl TensorDriftGravity2D {
    pub fn new(config: DriftFlux2DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<2> for TensorDriftGravity2D {
    fn nfields(&self) -> usize {
        4
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation < 2
    }

    /// Lane-packed buoyant-gravity residual for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed solution.
    /// * `equation` - momentum equation shared by all lanes.
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
        if equation >= 2 {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let alpha = state.value(ALPHA_2D, q);
        let g = self.config.gravity[equation];
        for l in 0..LANES {
            let rho = self.config.mixture_density(alpha[l]);
            f0[l] = (rho - self.config.rho_l) / rho * g;
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }

    /// Lane-packed buoyant-gravity Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed linearization point.
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - momentum equation shared by all lanes.
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
        if equation >= 2 {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let drho = self.config.mixture_density_derivative();
        let g = self.config.gravity[equation];
        let alpha = state.value(ALPHA_2D, q);
        let da = direction.value(ALPHA_2D, q);
        for l in 0..LANES {
            let rho = self.config.mixture_density(alpha[l]);
            let factor = drho * self.config.rho_l / (rho * rho) * g;
            f0[l] = factor * da[l];
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }
}
