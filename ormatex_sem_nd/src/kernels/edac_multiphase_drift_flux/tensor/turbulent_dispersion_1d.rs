//! Void-fraction turbulent-dispersion tensor kernel (1D).
//!
//! Mathematics: `d(D_td dalpha/dx)/dx` for equation 2 with the simplest
//! constant coefficient `D_td` (same Fickian form as the 2D kernel; see
//! [`super::super::closures`]). Besides spreading the dispersed bubbles it
//! regularizes the split-form void outflow, for which the 1D tensor path
//! provides no facet closure. Owns equation 2 only.
use crate::common::{LaneState, Lanes, TensorCtx, LANES};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config_1d::{
    drift_field_names_1d, DriftFlux1DConfig, ALPHA_1D,
};

/// Tensor void turbulent dispersion (owns equation 2).
pub struct TensorDriftTurbulentDispersion1D {
    pub config: DriftFlux1DConfig,
    pub dispersion: f64,
}
impl TensorDriftTurbulentDispersion1D {
    pub fn new(config: DriftFlux1DConfig, dispersion: f64) -> Self {
        assert!(
            dispersion.is_finite() && dispersion >= 0.0,
            "dispersion coefficient must be finite and nonnegative"
        );
        Self { config, dispersion }
    }
}
impl TensorResidualKernel<1> for TensorDriftTurbulentDispersion1D {
    fn nfields(&self) -> usize {
        3
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names_1d()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation == ALPHA_1D
    }

    /// Lane-packed turbulent-dispersion residual for all lanes.
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
        let g = state.grad(ALPHA_1D, q, 0);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = self.dispersion * g[l];
            f1y[l] = 0.0;
        }
    }

    /// Lane-packed turbulent-dispersion Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed linearization point (unused).
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
        _state: &LaneState<'_>,
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
        let dg = direction.grad(ALPHA_1D, q, 0);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = self.dispersion * dg[l];
            f1y[l] = 0.0;
        }
    }
}
