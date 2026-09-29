//! Mixture-density pressure-divergence tensor kernel (1D).
//!
//! Mathematics: `rho_m(alpha) c0^2 du/dx` for equation 1 with the
//! `drho_m/dalpha` action term; owns equation 1 only.
use crate::common::{LaneState, Lanes, TensorCtx, LANES};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config_1d::{
    drift_field_names_1d, DriftFlux1DConfig, ALPHA_1D,
};

/// Tensor mixture pressure-divergence (owns equation 1).
pub struct TensorDriftPressureDivergence1D {
    pub config: DriftFlux1DConfig,
}
impl TensorDriftPressureDivergence1D {
    pub fn new(config: DriftFlux1DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<1> for TensorDriftPressureDivergence1D {
    fn nfields(&self) -> usize {
        3
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names_1d()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation == 1
    }

    /// Lane-packed mixture pressure-divergence residual for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
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
        let alpha = state.value(ALPHA_1D, q);
        let g = state.grad(0, q, 0);
        for l in 0..LANES {
            let rho = self.config.mixture_density(alpha[l]);
            f0[l] = rho * self.config.c0 * self.config.c0 * g[l];
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }

    /// Lane-packed mixture pressure-divergence Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed linearization point.
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
        state: &LaneState<'_>,
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
        let c02 = self.config.c0 * self.config.c0;
        let drho = self.config.mixture_density_derivative();
        let alpha = state.value(ALPHA_1D, q);
        let da = direction.value(ALPHA_1D, q);
        let g = state.grad(0, q, 0);
        let dg = direction.grad(0, q, 0);
        for l in 0..LANES {
            let rho = self.config.mixture_density(alpha[l]);
            f0[l] = c02 * (drho * da[l] * g[l] + rho * dg[l]);
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }
}
