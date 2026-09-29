//! Tensor viscous stress `d(tau)/dx` for the 1D `u` equation.
//!
//! Mathematics: for equation 0 the triple is `(0, tau, 0)` with
//! `tau = 2 nu du/dx`; the action is linear in the direction gradient.
//! Owns equation 0 only. Weak counterpart:
//! [`KernelEdacViscousStress1D`](crate::kernels::edac::weak::viscous_stress_1d::KernelEdacViscousStress1D).

use crate::common::{LaneState, Lanes, TensorCtx, LANES};

use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config_1d::{fluid_field_names_1d, EdacNavierStokes1DConfig};

/// Tensor viscous-stress kernel (sum-factorized counterpart, owns equation 0).
pub struct TensorKernelEdacViscousStress1D {
    pub config: EdacNavierStokes1DConfig,
}

impl TensorKernelEdacViscousStress1D {
    pub fn new(config: EdacNavierStokes1DConfig) -> Self {
        Self { config }
    }
}

impl TensorResidualKernel<1> for TensorKernelEdacViscousStress1D {
    #[inline]
    fn nfields(&self) -> usize {
        2
    }
    #[inline]
    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names_1d()
    }
    #[inline]
    fn owns_equation(&self, equation: usize) -> bool {
        equation == 0
    }

    /// Lane-packed viscous-stress residual for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed solution.
    /// * `equation` - momentum equation shared by all lanes.
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0`/`f1x`/`f1y` - lane output slots. Overwritten.
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
        let tau = self.config.stress_tensor_lanes(state, q);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = tau[l];
            f1y[l] = 0.0;
        }
    }

    /// Lane-packed viscous-stress Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed linearization point (unused).
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - momentum equation shared by all lanes.
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0`/`f1x`/`f1y` - lane output slots. Overwritten.
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
        if equation != 0 {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let dtau = self
            .config
            .stress_tensor_directional_derivative_lanes(direction, q);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = dtau[l];
            f1y[l] = 0.0;
        }
    }
}
