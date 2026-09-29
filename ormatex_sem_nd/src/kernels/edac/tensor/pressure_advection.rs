//! Tensor pressure advection `(u . grad) p` for the pressure equation.
//!
//! Mathematics: for equation 2 the triple is `(u_j d_j p, 0, 0)`; the action
//! adds `(du_j d_j p + u_j d_j dp, 0, 0)`. Owns equation 2 only. Weak
//! counterpart:
//! [`KernelEdacPressureAdvection2D`](crate::kernels::edac::weak::pressure_advection::KernelEdacPressureAdvection2D).

use crate::common::{LaneState, Lanes, TensorCtx, LANES};

use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config::{fluid_field_names, EdacNavierStokes2DConfig};

/// Tensor pressure-advection kernel (sum-factorized counterpart, owns equation 2).
pub struct TensorKernelEdacPressureAdvection2D {
    pub config: EdacNavierStokes2DConfig,
}

impl TensorKernelEdacPressureAdvection2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl TensorResidualKernel<2> for TensorKernelEdacPressureAdvection2D {
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
        equation == 2
    }

    /// Lane-packed pressure-advection residual for all lanes.
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
        if equation != 2 {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let u0 = state.value(0, q);
        let u1 = state.value(1, q);
        let gp0 = state.grad(2, q, 0);
        let gp1 = state.grad(2, q, 1);
        for l in 0..LANES {
            f0[l] = u0[l] * gp0[l] + u1[l] * gp1[l];
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }

    /// Lane-packed pressure-advection Jacobian action for all lanes.
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
        if equation != 2 {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let u0 = state.value(0, q);
        let u1 = state.value(1, q);
        let gp0 = state.grad(2, q, 0);
        let gp1 = state.grad(2, q, 1);
        let du0 = direction.value(0, q);
        let du1 = direction.value(1, q);
        let dgp0 = direction.grad(2, q, 0);
        let dgp1 = direction.grad(2, q, 1);
        for l in 0..LANES {
            f0[l] = du0[l] * gp0[l] + du1[l] * gp1[l] + u0[l] * dgp0[l] + u1[l] * dgp1[l];
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }
}
