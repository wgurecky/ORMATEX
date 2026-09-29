//! Fused split-form 1D EDAC Navier-Stokes tensor kernel (`[u, p]`).
//!
//! Mathematics: pointwise `(f0, f1x, 0)` triples summing the split advection
//! halves with the conservative pressure-gradient, viscous stress,
//! divergence, and diffusion triples. Owns both equations. Delegates to the
//! part kernels, so it equals their fused sum by construction.
//! Weak counterpart:
//! [`KernelEdacNavierStokesSplit1D`](crate::kernels::edac::weak::navier_stokes_split_1d::KernelEdacNavierStokesSplit1D).

use crate::common::{LaneState, Lanes, TensorCtx, LANES};

use super::momentum_convection_split_1d::TensorKernelEdacMomentumConvectionSplit1D;
use super::pressure_advection_split_1d::TensorKernelEdacPressureAdvectionSplit1D;
use super::pressure_diffusion_1d::TensorKernelEdacPressureDiffusion1D;
use super::pressure_divergence_1d::TensorKernelEdacPressureDivergence1D;
use super::pressure_gradient_1d::TensorKernelEdacPressureGradient1D;
use super::viscous_stress_1d::TensorKernelEdacViscousStress1D;
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config_1d::{fluid_field_names_1d, EdacNavierStokes1DConfig};

/// Fused split-form 1D EDAC Navier-Stokes tensor kernel for `[u, p]`
/// (owns all equations; delegates to the part kernels).
pub struct TensorKernelEdacNavierStokesSplit1D {
    pub config: EdacNavierStokes1DConfig,
}

impl TensorKernelEdacNavierStokesSplit1D {
    pub fn new(config: EdacNavierStokes1DConfig) -> Self {
        Self { config }
    }
}

impl TensorResidualKernel<1> for TensorKernelEdacNavierStokesSplit1D {
    #[inline]
    fn nfields(&self) -> usize {
        2
    }

    #[inline]
    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names_1d()
    }

    /// Lane-packed fused split residual via part-kernel lanes.
    ///
    /// Accumulates the six part triples starting from zero in part order,
    /// matching the scalar `parts.iter().map(..).sum()` order per lane.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed solution.
    /// * `equation` - output equation shared by all lanes.
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
        *f0 = [0.0; LANES];
        *f1x = [0.0; LANES];
        *f1y = [0.0; LANES];
        let mut t0 = [0.0; LANES];
        let mut t1x = [0.0; LANES];
        let mut t1y = [0.0; LANES];
        TensorKernelEdacMomentumConvectionSplit1D::new(self.config)
            .tensor_residual(ctxs, state, equation, q, &mut t0, &mut t1x, &mut t1y);
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
        TensorKernelEdacPressureGradient1D::new(self.config)
            .tensor_residual(ctxs, state, equation, q, &mut t0, &mut t1x, &mut t1y);
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
        TensorKernelEdacViscousStress1D::new(self.config)
            .tensor_residual(ctxs, state, equation, q, &mut t0, &mut t1x, &mut t1y);
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
        TensorKernelEdacPressureDivergence1D::new(self.config)
            .tensor_residual(ctxs, state, equation, q, &mut t0, &mut t1x, &mut t1y);
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
        TensorKernelEdacPressureAdvectionSplit1D::new(self.config)
            .tensor_residual(ctxs, state, equation, q, &mut t0, &mut t1x, &mut t1y);
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
        TensorKernelEdacPressureDiffusion1D::new(self.config)
            .tensor_residual(ctxs, state, equation, q, &mut t0, &mut t1x, &mut t1y);
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
    }

    /// Lane-packed fused split Jacobian action via part-kernel lanes.
    ///
    /// Lane counterpart of the scalar fused action with identical part order.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed linearization point.
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - output equation shared by all lanes.
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
        *f0 = [0.0; LANES];
        *f1x = [0.0; LANES];
        *f1y = [0.0; LANES];
        let mut t0 = [0.0; LANES];
        let mut t1x = [0.0; LANES];
        let mut t1y = [0.0; LANES];
        TensorKernelEdacMomentumConvectionSplit1D::new(self.config).tensor_jacobian_action(
            ctxs, state, direction, equation, q, &mut t0, &mut t1x, &mut t1y,
        );
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
        TensorKernelEdacPressureGradient1D::new(self.config).tensor_jacobian_action(
            ctxs, state, direction, equation, q, &mut t0, &mut t1x, &mut t1y,
        );
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
        TensorKernelEdacViscousStress1D::new(self.config).tensor_jacobian_action(
            ctxs, state, direction, equation, q, &mut t0, &mut t1x, &mut t1y,
        );
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
        TensorKernelEdacPressureDivergence1D::new(self.config).tensor_jacobian_action(
            ctxs, state, direction, equation, q, &mut t0, &mut t1x, &mut t1y,
        );
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
        TensorKernelEdacPressureAdvectionSplit1D::new(self.config).tensor_jacobian_action(
            ctxs, state, direction, equation, q, &mut t0, &mut t1x, &mut t1y,
        );
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
        TensorKernelEdacPressureDiffusion1D::new(self.config).tensor_jacobian_action(
            ctxs, state, direction, equation, q, &mut t0, &mut t1x, &mut t1y,
        );
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
    }
}
