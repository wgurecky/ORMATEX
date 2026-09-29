//! Split-form void-fraction advection tensor kernel (2D).
//!
//! Mathematics: gas continuity advected by the mixture velocity,
//! `1/2 C0 (u_j d_j alpha, -u_j alpha, ...)` for equation 3 with the
//! distribution parameter `C0` (Ishii 1975 slip closed on `u_m` with the
//! um-referenced speed; see [`DriftFlux2DConfig::slip_speed`](super::super::config::DriftFlux2DConfig::slip_speed)).
//! The relative drift flux lives in the companion
//! drift-flux kernel. Needs the drift split-flux boundary.
use crate::common::{LaneState, Lanes, TensorCtx, LANES};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config::{
    drift_field_names, DriftFlux2DConfig, ALPHA_2D,
};

/// Tensor split void advection (owns equation 3).
pub struct TensorDriftVoidAdvectionSplit2D {
    pub config: DriftFlux2DConfig,
}
impl TensorDriftVoidAdvectionSplit2D {
    pub fn new(config: DriftFlux2DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<2> for TensorDriftVoidAdvectionSplit2D {
    fn nfields(&self) -> usize {
        4
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation == ALPHA_2D
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
        let c0 = self.config.distribution_parameter();
        let u0 = state.value(0, q);
        let u1 = state.value(1, q);
        let g0 = state.grad(ALPHA_2D, q, 0);
        let g1 = state.grad(ALPHA_2D, q, 1);
        let a = state.value(ALPHA_2D, q);
        for l in 0..LANES {
            f0[l] = 0.5 * c0 * (u0[l] * g0[l] + u1[l] * g1[l]);
            f1x[l] = -0.5 * c0 * u0[l] * a[l];
            f1y[l] = -0.5 * c0 * u1[l] * a[l];
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
        let c0 = self.config.distribution_parameter();
        let u0 = state.value(0, q);
        let u1 = state.value(1, q);
        let g0 = state.grad(ALPHA_2D, q, 0);
        let g1 = state.grad(ALPHA_2D, q, 1);
        let a = state.value(ALPHA_2D, q);
        let du0 = direction.value(0, q);
        let du1 = direction.value(1, q);
        let dg0 = direction.grad(ALPHA_2D, q, 0);
        let dg1 = direction.grad(ALPHA_2D, q, 1);
        let da = direction.value(ALPHA_2D, q);
        for l in 0..LANES {
            f0[l] = 0.5 * c0 * (du0[l] * g0[l] + du1[l] * g1[l] + u0[l] * dg0[l] + u1[l] * dg1[l]);
            f1x[l] = -0.5 * c0 * (du0[l] * a[l] + u0[l] * da[l]);
            f1y[l] = -0.5 * c0 * (du1[l] * a[l] + u1[l] * da[l]);
        }
    }
}
