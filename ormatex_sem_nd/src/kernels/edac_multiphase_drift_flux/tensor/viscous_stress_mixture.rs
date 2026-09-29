//! Mixture viscous stress `div(tau_m)` for the 2D momentum equations.
//!
//! Mathematics: `(0, tau_i0, tau_i1)` with
//! `tau_ij = 2 (nu_m(alpha) + nu_t) S_ij`, where `nu_m = mu_m/rho_m` is the
//! linearly averaged mixture kinematic viscosity and `nu_t` is the
//! Smagorinsky-Lilly eddy viscosity evaluated on the mixture velocity (same
//! model as the base EDAC kernels, per request). The action linearizes both
//! the `nu_m(alpha)` dependence and the eddy viscosity. Owns equations 0-1.
use crate::common::{LaneState, Lanes, TensorCtx, LANES};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config::{drift_field_names, DriftFlux2DConfig};

/// Tensor mixture viscous-stress (owns equations 0-1).
pub struct TensorDriftViscousStress2D {
    pub config: DriftFlux2DConfig,
}
impl TensorDriftViscousStress2D {
    pub fn new(config: DriftFlux2DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<2> for TensorDriftViscousStress2D {
    fn nfields(&self) -> usize {
        4
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation < 2
    }

    /// Lane-packed mixture viscous-stress residual for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed solution.
    /// * `equation` - momentum equation shared by all lanes.
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
        if equation >= 2 {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let (row0, row1) = self
            .config
            .stress_tensor_row_lanes(ctxs, state, q, equation);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = row0[l];
            f1y[l] = row1[l];
        }
    }

    /// Lane-packed mixture viscous-stress Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed linearization point.
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - momentum equation shared by all lanes.
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
        if equation >= 2 {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let (row0, row1) = self
            .config
            .stress_tensor_row_directional_derivative_lanes(ctxs, state, direction, q, equation);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = row0[l];
            f1y[l] = row1[l];
        }
    }
}
