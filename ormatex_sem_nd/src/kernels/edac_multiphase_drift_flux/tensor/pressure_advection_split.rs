//! Split-form mixture pressure-advection tensor kernel (2D).
//!
//! Mathematics: `1/2 (u_j d_j p, -u_j p, ...)` for equation 2 on the mixture
//! velocity; owns equation 2 only. Needs the drift split-flux boundary.
use crate::common::{LaneState, Lanes, TensorCtx, LANES};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config::{drift_field_names, DriftFlux2DConfig};

/// Tensor split mixture pressure-advection (owns equation 2).
pub struct TensorDriftPressureAdvectionSplit2D {
    pub config: DriftFlux2DConfig,
}
impl TensorDriftPressureAdvectionSplit2D {
    pub fn new(config: DriftFlux2DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<2> for TensorDriftPressureAdvectionSplit2D {
    fn nfields(&self) -> usize {
        4
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation == 2
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
        let u0 = state.value(0, q);
        let u1 = state.value(1, q);
        let g0 = state.grad(2, q, 0);
        let g1 = state.grad(2, q, 1);
        let p = state.value(2, q);
        for l in 0..LANES {
            f0[l] = 0.5 * (u0[l] * g0[l] + u1[l] * g1[l]);
            f1x[l] = -0.5 * u0[l] * p[l];
            f1y[l] = -0.5 * u1[l] * p[l];
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
        let u0 = state.value(0, q);
        let u1 = state.value(1, q);
        let g0 = state.grad(2, q, 0);
        let g1 = state.grad(2, q, 1);
        let p = state.value(2, q);
        let du0 = direction.value(0, q);
        let du1 = direction.value(1, q);
        let dg0 = direction.grad(2, q, 0);
        let dg1 = direction.grad(2, q, 1);
        let dp = direction.value(2, q);
        for l in 0..LANES {
            f0[l] = 0.5 * (du0[l] * g0[l] + du1[l] * g1[l] + u0[l] * dg0[l] + u1[l] * dg1[l]);
            f1x[l] = -0.5 * (du0[l] * p[l] + u0[l] * dp[l]);
            f1y[l] = -0.5 * (du1[l] * p[l] + u1[l] * dp[l]);
        }
    }
}
