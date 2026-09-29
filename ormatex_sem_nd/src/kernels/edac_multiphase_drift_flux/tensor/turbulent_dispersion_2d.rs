//! Void-fraction turbulent-dispersion tensor kernel (2D).
//!
//! Mathematics: Fickian flux `div(D_td grad(alpha))` for equation 3 with the
//! simplest constant coefficient `D_td` (Burns et al. 2004 Favre-averaged
//! drag / Lopez de Bertodano turbulent-dispersion flux in its
//! constant-coefficient limit; see [`super::super::closures`]). Spreads the
//! dispersed bubbles; linear action. Owns equation 3 only.
use crate::common::{LaneState, Lanes, TensorCtx, LANES};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config::{
    drift_field_names, DriftFlux2DConfig, ALPHA_2D,
};

/// Tensor void turbulent dispersion (owns equation 3).
pub struct TensorDriftTurbulentDispersion2D {
    pub config: DriftFlux2DConfig,
    pub dispersion: f64,
}
impl TensorDriftTurbulentDispersion2D {
    pub fn new(config: DriftFlux2DConfig, dispersion: f64) -> Self {
        assert!(
            dispersion.is_finite() && dispersion >= 0.0,
            "dispersion coefficient must be finite and nonnegative"
        );
        Self { config, dispersion }
    }
}
impl TensorResidualKernel<2> for TensorDriftTurbulentDispersion2D {
    fn nfields(&self) -> usize {
        4
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation == ALPHA_2D
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
        let g0 = state.grad(ALPHA_2D, q, 0);
        let g1 = state.grad(ALPHA_2D, q, 1);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = self.dispersion * g0[l];
            f1y[l] = self.dispersion * g1[l];
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
    /// * `f1y` - lane linearized y-flux slots. Overwritten.
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
        if equation != ALPHA_2D {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let dg0 = direction.grad(ALPHA_2D, q, 0);
        let dg1 = direction.grad(ALPHA_2D, q, 1);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = self.dispersion * dg0[l];
            f1y[l] = self.dispersion * dg1[l];
        }
    }
}
