//! Mixture-density pressure-divergence tensor kernel (2D).
//!
//! Mathematics: modified EDAC continuity `rho_m(alpha) c0^2 div(u_m)` for
//! equation 2; the action adds the `drho_m/dalpha` product. At `alpha = 0`
//! this reduces to the base `rho_l c0^2 div(u)` term.
use crate::common::{LaneState, Lanes, TensorCtx, LANES};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config::{
    drift_field_names, DriftFlux2DConfig, ALPHA_2D,
};

/// Tensor mixture pressure-divergence (owns equation 2).
pub struct TensorDriftPressureDivergence2D {
    pub config: DriftFlux2DConfig,
}
impl TensorDriftPressureDivergence2D {
    pub fn new(config: DriftFlux2DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<2> for TensorDriftPressureDivergence2D {
    fn nfields(&self) -> usize {
        4
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation == 2
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
        let alpha = state.value(ALPHA_2D, q);
        let gx = state.grad(0, q, 0);
        let gy = state.grad(1, q, 1);
        for l in 0..LANES {
            let rho = self.config.mixture_density(alpha[l]);
            f0[l] = rho * self.config.c0 * self.config.c0 * (gx[l] + gy[l]);
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
        if equation != 2 {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let c02 = self.config.c0 * self.config.c0;
        let drho = self.config.mixture_density_derivative();
        let alpha = state.value(ALPHA_2D, q);
        let dalpha = direction.value(ALPHA_2D, q);
        let gx = state.grad(0, q, 0);
        let gy = state.grad(1, q, 1);
        let dgx = direction.grad(0, q, 0);
        let dgy = direction.grad(1, q, 1);
        for l in 0..LANES {
            let rho = self.config.mixture_density(alpha[l]);
            let div_u = gx[l] + gy[l];
            let div_du = dgx[l] + dgy[l];
            f0[l] = c02 * (drho * dalpha[l] * div_u + rho * div_du);
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }
}
