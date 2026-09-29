//! Mixture-density pressure-gradient tensor kernel (1D).
//!
//! Mathematics: `dp/dx / rho_m(alpha)` for equation 0 with the
//! `-dp/dx / rho_m^2 drho_m/dalpha dalpha` action term.
use crate::common::{LaneState, Lanes, TensorCtx, LANES};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config_1d::{
    drift_field_names_1d, DriftFlux1DConfig, ALPHA_1D,
};

/// Tensor mixture pressure-gradient (owns equation 0).
pub struct TensorDriftPressureGradient1D {
    pub config: DriftFlux1DConfig,
}
impl TensorDriftPressureGradient1D {
    pub fn new(config: DriftFlux1DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<1> for TensorDriftPressureGradient1D {
    fn nfields(&self) -> usize {
        3
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names_1d()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation == 0
    }

    /// Lane-packed mixture pressure-gradient residual for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed solution.
    /// * `equation` - momentum equation shared by all lanes.
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
        if equation != 0 {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let alpha = state.value(ALPHA_1D, q);
        let gp = state.grad(1, q, 0);
        for l in 0..LANES {
            let rho = self.config.mixture_density(alpha[l]);
            f0[l] = gp[l] / rho;
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }

    /// Lane-packed mixture pressure-gradient Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed linearization point.
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - momentum equation shared by all lanes.
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
        if equation != 0 {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let drho = self.config.mixture_density_derivative();
        let alpha = state.value(ALPHA_1D, q);
        let gp = state.grad(1, q, 0);
        let dgp = direction.grad(1, q, 0);
        let da = direction.value(ALPHA_1D, q);
        for l in 0..LANES {
            let rho = self.config.mixture_density(alpha[l]);
            f0[l] = dgp[l] / rho - gp[l] * drho * da[l] / (rho * rho);
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }
}
