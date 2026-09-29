//! Fused split-form EDAC Navier-Stokes tensor kernel (`[u, v, p]`).
//!
//! Mathematics: pointwise `(f0, f1x, f1y)` triples summing the split advection
//! halves (`1/2` advective + `1/2` conservative flux triples from the
//! `*_split` tensor kernels) with the conservative pressure-gradient, viscous
//! stress, divergence, and diffusion triples. Owns all three equations.
//! Delegates to the part kernels, so it equals their fused sum by
//! construction.
//!
//! Boundary pairing: same rule as the weak fused kernel — a
//! [`SplitBoundaryFlux`](crate::kernels::edac::tensor::split_boundary_flux::TensorKernelEdacSplitBoundaryFlux2D)
//! default with tensor Dong or directional-do-nothing `.with_split_flux(true)`
//! outlets, plus tensor no-slip/slip walls. Weak counterpart:
//! [`KernelEdacNavierStokesSplit2D`](crate::kernels::edac::weak::navier_stokes_split::KernelEdacNavierStokesSplit2D).

use crate::common::{LaneState, Lanes, TensorCtx, LANES};

use super::momentum_convection_split::TensorKernelEdacMomentumConvectionSplit2D;
use super::pressure_advection_split::TensorKernelEdacPressureAdvectionSplit2D;
use super::pressure_diffusion::TensorKernelEdacPressureDiffusion2D;
use super::pressure_divergence::TensorKernelEdacPressureDivergence2D;
use super::pressure_gradient::TensorKernelEdacPressureGradient2D;
use super::viscous_stress::TensorKernelEdacViscousStress2D;
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config::{fluid_field_names, EdacNavierStokes2DConfig};

/// Fused split-form EDAC Navier-Stokes tensor kernel for `[u, v, p]`
/// (owns all equations; delegates to the part kernels).
pub struct TensorKernelEdacNavierStokesSplit2D {
    pub config: EdacNavierStokes2DConfig,
}

impl TensorKernelEdacNavierStokesSplit2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl TensorResidualKernel<2> for TensorKernelEdacNavierStokesSplit2D {
    #[inline]
    fn nfields(&self) -> usize {
        3
    }

    #[inline]
    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names()
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
        TensorKernelEdacMomentumConvectionSplit2D::new(self.config)
            .tensor_residual(ctxs, state, equation, q, &mut t0, &mut t1x, &mut t1y);
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
        TensorKernelEdacPressureGradient2D::new(self.config)
            .tensor_residual(ctxs, state, equation, q, &mut t0, &mut t1x, &mut t1y);
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
        TensorKernelEdacViscousStress2D::new(self.config)
            .tensor_residual(ctxs, state, equation, q, &mut t0, &mut t1x, &mut t1y);
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
        TensorKernelEdacPressureDivergence2D::new(self.config)
            .tensor_residual(ctxs, state, equation, q, &mut t0, &mut t1x, &mut t1y);
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
        TensorKernelEdacPressureAdvectionSplit2D::new(self.config)
            .tensor_residual(ctxs, state, equation, q, &mut t0, &mut t1x, &mut t1y);
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
        TensorKernelEdacPressureDiffusion2D::new(self.config)
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
        TensorKernelEdacMomentumConvectionSplit2D::new(self.config).tensor_jacobian_action(
            ctxs, state, direction, equation, q, &mut t0, &mut t1x, &mut t1y,
        );
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
        TensorKernelEdacPressureGradient2D::new(self.config).tensor_jacobian_action(
            ctxs, state, direction, equation, q, &mut t0, &mut t1x, &mut t1y,
        );
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
        TensorKernelEdacViscousStress2D::new(self.config).tensor_jacobian_action(
            ctxs, state, direction, equation, q, &mut t0, &mut t1x, &mut t1y,
        );
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
        TensorKernelEdacPressureDivergence2D::new(self.config).tensor_jacobian_action(
            ctxs, state, direction, equation, q, &mut t0, &mut t1x, &mut t1y,
        );
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
        TensorKernelEdacPressureAdvectionSplit2D::new(self.config).tensor_jacobian_action(
            ctxs, state, direction, equation, q, &mut t0, &mut t1x, &mut t1y,
        );
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
        TensorKernelEdacPressureDiffusion2D::new(self.config).tensor_jacobian_action(
            ctxs, state, direction, equation, q, &mut t0, &mut t1x, &mut t1y,
        );
        for l in 0..LANES {
            f0[l] += t0[l];
            f1x[l] += t1x[l];
            f1y[l] += t1y[l];
        }
    }
}
