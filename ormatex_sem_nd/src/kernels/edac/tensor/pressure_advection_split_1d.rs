//! Split-form pressure advection tensor kernel (1D).
//!
//! Mathematics: half advective plus half conservative-flux triple,
//! `1/2 (u dp/dx, -u p, 0)` with the matching action; owns equation 1.
//! Weak counterpart:
//! [`KernelEdacPressureAdvectionSplit1D`](crate::kernels::edac::weak::pressure_advection_split_1d::KernelEdacPressureAdvectionSplit1D).
use crate::common::{LaneState, Lanes, TensorCtx, LANES};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config_1d::{fluid_field_names_1d, EdacNavierStokes1DConfig};

/// Tensor split pressure-advection kernel (owns equation 1).
pub struct TensorKernelEdacPressureAdvectionSplit1D {
    pub config: EdacNavierStokes1DConfig,
}
impl TensorKernelEdacPressureAdvectionSplit1D {
    pub fn new(config: EdacNavierStokes1DConfig) -> Self {
        Self { config }
    }
}
impl TensorResidualKernel<1> for TensorKernelEdacPressureAdvectionSplit1D {
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
        equation == 1
    }

    /// Lane-packed split-form residual for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed solution.
    /// * `equation` - pressure equation shared by all lanes.
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
        if equation != 1 {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let u = state.value(0, q);
        let p = state.value(1, q);
        let g = state.grad(1, q, 0);
        for l in 0..LANES {
            f0[l] = 0.5 * u[l] * g[l];
            f1x[l] = -0.5 * u[l] * p[l];
            f1y[l] = 0.0;
        }
    }

    /// Lane-packed split-form Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed linearization point.
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - pressure equation shared by all lanes.
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
        if equation != 1 {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let u = state.value(0, q);
        let g = state.grad(1, q, 0);
        let du = direction.value(0, q);
        let p = state.value(1, q);
        let dp = direction.value(1, q);
        let dg = direction.grad(1, q, 0);
        for l in 0..LANES {
            f0[l] = 0.5 * (du[l] * g[l] + u[l] * dg[l]);
            f1x[l] = -0.5 * (du[l] * p[l] + u[l] * dp[l]);
            f1y[l] = 0.0;
        }
    }
}
