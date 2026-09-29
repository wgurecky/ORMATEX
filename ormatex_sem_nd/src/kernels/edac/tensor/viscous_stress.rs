//! Tensor viscous stress `div(tau)` for the `u`/`v` equations.
//!
//! Mathematics: for momentum equation `i < 2` the triple is
//! `(0, tau_i0, tau_i1)` with `tau_ij = 2 (nu + nu_t) S_ij`; the action uses
//! the directional-derivative stress row (eddy-viscosity linearization
//! included). Both row components share one viscosity evaluation via
//! `EdacNavierStokes2DConfig::stress_tensor_row`. Owns equations 0–1 only.
//! Weak counterpart:
//! [`KernelEdacViscousStress2D`](crate::kernels::edac::weak::viscous_stress::KernelEdacViscousStress2D).

use crate::common::{LaneState, Lanes, TensorCtx, LANES};

use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config::{fluid_field_names, EdacNavierStokes2DConfig};

/// Tensor viscous-stress kernel (sum-factorized counterpart, owns equations 0-1).
pub struct TensorKernelEdacViscousStress2D {
    pub config: EdacNavierStokes2DConfig,
}

impl TensorKernelEdacViscousStress2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl TensorResidualKernel<2> for TensorKernelEdacViscousStress2D {
    #[inline]
    fn nfields(&self) -> usize {
        3
    }
    #[inline]
    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names()
    }
    #[inline]
    fn owns_equation(&self, equation: usize) -> bool {
        equation < 2
    }

    /// Lane-packed viscous-stress residual for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
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

    /// Lane-packed viscous-stress Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed linearization point.
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - momentum equation shared by all lanes.
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0`/`f1x`/`f1y` - lane output slots. Overwritten.
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
