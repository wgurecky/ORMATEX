//! Split-form void-fraction advection tensor kernel (1D).
//!
//! Mathematics: `1/2 C0 (u dalpha/dx, -u alpha, 0)` for equation 2; owns
//! equation 2 only.
use crate::common::{LaneState, Lanes, TensorCtx, LANES};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config_1d::{
    drift_field_names_1d, DriftFlux1DConfig, ALPHA_1D,
};

/// Tensor split void advection (owns equation 2).
pub struct TensorDriftVoidAdvectionSplit1D {
    pub config: DriftFlux1DConfig,
}
impl TensorDriftVoidAdvectionSplit1D {
    pub fn new(config: DriftFlux1DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<1> for TensorDriftVoidAdvectionSplit1D {
    fn nfields(&self) -> usize {
        3
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names_1d()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation == ALPHA_1D
    }

    /// Lane-packed split void-advection residual for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed solution.
    /// * `equation` - void equation shared by all lanes.
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
        if equation != ALPHA_1D {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let c0 = self.config.distribution_parameter();
        let u = state.value(0, q);
        let g = state.grad(ALPHA_1D, q, 0);
        let a = state.value(ALPHA_1D, q);
        for l in 0..LANES {
            f0[l] = 0.5 * c0 * u[l] * g[l];
            f1x[l] = -0.5 * c0 * u[l] * a[l];
            f1y[l] = 0.0;
        }
    }

    /// Lane-packed split void-advection Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed linearization point.
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - void equation shared by all lanes.
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
        if equation != ALPHA_1D {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let c0 = self.config.distribution_parameter();
        let u = state.value(0, q);
        let du = direction.value(0, q);
        let g = state.grad(ALPHA_1D, q, 0);
        let dg = direction.grad(ALPHA_1D, q, 0);
        let a = state.value(ALPHA_1D, q);
        let da = direction.value(ALPHA_1D, q);
        for l in 0..LANES {
            f0[l] = 0.5 * c0 * (du[l] * g[l] + u[l] * dg[l]);
            f1x[l] = -0.5 * c0 * (du[l] * a[l] + u[l] * da[l]);
            f1y[l] = 0.0;
        }
    }
}
