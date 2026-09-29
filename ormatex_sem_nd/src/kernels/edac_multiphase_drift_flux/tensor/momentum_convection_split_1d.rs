//! Split-form mixture-momentum convection tensor kernel (1D).
//!
//! Mathematics: `1/2 (u du/dx, -u u, 0)` for equation 0 on the mixture
//! velocity; owns equation 0 only.
use crate::common::{LaneState, Lanes, TensorCtx, LANES};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config_1d::{
    drift_field_names_1d, DriftFlux1DConfig,
};

/// Tensor split mixture-momentum convection (owns equation 0).
pub struct TensorDriftMomentumConvectionSplit1D {
    pub config: DriftFlux1DConfig,
}
impl TensorDriftMomentumConvectionSplit1D {
    pub fn new(config: DriftFlux1DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<1> for TensorDriftMomentumConvectionSplit1D {
    fn nfields(&self) -> usize {
        3
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names_1d()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation == 0
    }

    /// Lane-packed split-form residual for all lanes.
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
        let u = state.value(0, q);
        let g = state.grad(0, q, 0);
        for l in 0..LANES {
            f0[l] = 0.5 * u[l] * g[l];
            f1x[l] = -0.5 * u[l] * u[l];
            f1y[l] = 0.0;
        }
    }

    /// Lane-packed split-form Jacobian action for all lanes.
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
        let u = state.value(0, q);
        let du = direction.value(0, q);
        let g = state.grad(0, q, 0);
        let dg = direction.grad(0, q, 0);
        for l in 0..LANES {
            f0[l] = 0.5 * (du[l] * g[l] + u[l] * dg[l]);
            f1x[l] = -0.5 * (du[l] * u[l] + u[l] * du[l]);
            f1y[l] = 0.0;
        }
    }
}
