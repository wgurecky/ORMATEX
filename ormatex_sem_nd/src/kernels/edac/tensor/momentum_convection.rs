//! Tensor momentum convection `(u . grad) u` for the `u`/`v` equations.
//!
//! Mathematics: for momentum equation `i < 2` the `(f0, f1x, f1y)` triple is
//! `(u_j d_j u_i, 0, 0)`; the Jacobian action additionally carries
//! `(du_j d_j u_i + u_j d_j du_i, 0, 0)`. Owns equations 0–1 only, so fused
//! sums skip the pressure block. Weak counterpart:
//! [`KernelEdacMomentumConvection2D`](crate::kernels::edac::weak::momentum_convection::KernelEdacMomentumConvection2D).

use crate::common::{LaneState, Lanes, TensorCtx, LANES};

use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config::{fluid_field_names, EdacNavierStokes2DConfig};

/// Tensor momentum-convection kernel (sum-factorized counterpart, owns equations 0-1).
pub struct TensorKernelEdacMomentumConvection2D {
    pub config: EdacNavierStokes2DConfig,
}

impl TensorKernelEdacMomentumConvection2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl TensorResidualKernel<2> for TensorKernelEdacMomentumConvection2D {
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

    /// Lane-packed residual `(u_j d_j u_i, 0, 0)` for all lanes.
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
        if equation >= 2 {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let u0 = state.value(0, q);
        let u1 = state.value(1, q);
        let g0 = state.grad(equation, q, 0);
        let g1 = state.grad(equation, q, 1);
        for l in 0..LANES {
            f0[l] = u0[l] * g0[l] + u1[l] * g1[l];
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }

    /// Lane-packed Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
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
        let u0 = state.value(0, q);
        let u1 = state.value(1, q);
        let g0 = state.grad(equation, q, 0);
        let g1 = state.grad(equation, q, 1);
        let du0 = direction.value(0, q);
        let du1 = direction.value(1, q);
        let dg0 = direction.grad(equation, q, 0);
        let dg1 = direction.grad(equation, q, 1);
        for l in 0..LANES {
            f0[l] = du0[l] * g0[l] + du1[l] * g1[l] + u0[l] * dg0[l] + u1[l] * dg1[l];
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }
}
