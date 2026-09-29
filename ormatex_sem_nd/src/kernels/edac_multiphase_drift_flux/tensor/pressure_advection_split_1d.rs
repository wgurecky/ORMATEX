//! Split-form mixture pressure-advection tensor kernel (1D).
//!
//! Mathematics: `1/2 (u dp/dx, -u p, 0)` for equation 1; owns equation 1 only.
use crate::common::{LaneState, Lanes, TensorCtx, LANES};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config_1d::{
    drift_field_names_1d, DriftFlux1DConfig,
};

/// Tensor split mixture pressure-advection (owns equation 1).
pub struct TensorDriftPressureAdvectionSplit1D {
    pub config: DriftFlux1DConfig,
}
impl TensorDriftPressureAdvectionSplit1D {
    pub fn new(config: DriftFlux1DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<1> for TensorDriftPressureAdvectionSplit1D {
    fn nfields(&self) -> usize {
        3
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names_1d()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation == 1
    }

    /// Lane-packed split pressure-advection residual for all lanes.
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
        let u = state.value(0, q);
        let g = state.grad(1, q, 0);
        let p = state.value(1, q);
        for l in 0..LANES {
            f0[l] = 0.5 * u[l] * g[l];
            f1x[l] = -0.5 * u[l] * p[l];
            f1y[l] = 0.0;
        }
    }

    /// Lane-packed split pressure-advection Jacobian action for all lanes.
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
        let u = state.value(0, q);
        let du = direction.value(0, q);
        let g = state.grad(1, q, 0);
        let dg = direction.grad(1, q, 0);
        let p = state.value(1, q);
        let dp = direction.value(1, q);
        for l in 0..LANES {
            f0[l] = 0.5 * (du[l] * g[l] + u[l] * dg[l]);
            f1x[l] = -0.5 * (du[l] * p[l] + u[l] * dp[l]);
            f1y[l] = 0.0;
        }
    }
}
