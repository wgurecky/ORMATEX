//! Tensor artificial-compressibility divergence `rho c0^2 du/dx` for 1D `p`.
//!
//! Mathematics: for equation 1 the triple is `(rho c0^2 du/dx, 0, 0)` with
//! the linear action on the direction divergence. Owns equation 1 only.
//! Weak counterpart:
//! [`KernelEdacPressureDivergence1D`](crate::kernels::edac::weak::pressure_divergence_1d::KernelEdacPressureDivergence1D).

use crate::common::{LaneState, Lanes, TensorCtx, LANES};

use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config_1d::{fluid_field_names_1d, EdacNavierStokes1DConfig};

/// Tensor pressure-divergence kernel (sum-factorized counterpart, owns equation 1).
pub struct TensorKernelEdacPressureDivergence1D {
    pub config: EdacNavierStokes1DConfig,
}

impl TensorKernelEdacPressureDivergence1D {
    pub fn new(config: EdacNavierStokes1DConfig) -> Self {
        Self { config }
    }
}

impl TensorResidualKernel<1> for TensorKernelEdacPressureDivergence1D {
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
        if equation != 1 {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let g = state.grad(0, q, 0);
        for l in 0..LANES {
            f0[l] = self.config.rho * self.config.c0 * self.config.c0 * g[l];
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
        if equation != 1 {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let dg = direction.grad(0, q, 0);
        for l in 0..LANES {
            f0[l] = self.config.rho * self.config.c0 * self.config.c0 * dg[l];
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }
}
