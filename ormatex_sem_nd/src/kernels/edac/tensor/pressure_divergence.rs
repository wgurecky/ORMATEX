//! Tensor artificial-compressibility divergence `rho c0^2 div(u)` for `p`.
//!
//! Mathematics: for equation 2 the triple is `(rho c0^2 div(u), 0, 0)` with
//! the linear action on the direction divergence. Owns equation 2 only.
//! Weak counterpart:
//! [`KernelEdacPressureDivergence2D`](crate::kernels::edac::weak::pressure_divergence::KernelEdacPressureDivergence2D).

use crate::common::{LaneState, Lanes, TensorCtx, LANES};

use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config::{fluid_field_names, EdacNavierStokes2DConfig};

/// Tensor pressure-divergence kernel (sum-factorized counterpart, owns equation 2).
pub struct TensorKernelEdacPressureDivergence2D {
    pub config: EdacNavierStokes2DConfig,
}

impl TensorKernelEdacPressureDivergence2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl TensorResidualKernel<2> for TensorKernelEdacPressureDivergence2D {
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

    /// Lane-packed divergence residual for all lanes.
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
        let g00 = state.grad(0, q, 0);
        let g11 = state.grad(1, q, 1);
        let coeff = self.config.rho * self.config.c0 * self.config.c0;
        for l in 0..LANES {
            f0[l] = coeff * (g00[l] + g11[l]);
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }

    /// Lane-packed divergence Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed linearization point (unused).
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - pressure equation shared by all lanes.
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
        if equation != 2 {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let dg00 = direction.grad(0, q, 0);
        let dg11 = direction.grad(1, q, 1);
        let coeff = self.config.rho * self.config.c0 * self.config.c0;
        for l in 0..LANES {
            f0[l] = coeff * (dg00[l] + dg11[l]);
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }
}
