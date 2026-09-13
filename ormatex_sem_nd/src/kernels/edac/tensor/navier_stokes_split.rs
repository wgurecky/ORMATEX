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

use crate::common::{CellState, TensorCtx};

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
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names()
    }

    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let parts = [
            TensorKernelEdacMomentumConvectionSplit2D::new(self.config)
                .tensor_residual(ctx, state, equation, q),
            TensorKernelEdacPressureGradient2D::new(self.config)
                .tensor_residual(ctx, state, equation, q),
            TensorKernelEdacViscousStress2D::new(self.config)
                .tensor_residual(ctx, state, equation, q),
            TensorKernelEdacPressureDivergence2D::new(self.config)
                .tensor_residual(ctx, state, equation, q),
            TensorKernelEdacPressureAdvectionSplit2D::new(self.config)
                .tensor_residual(ctx, state, equation, q),
            TensorKernelEdacPressureDiffusion2D::new(self.config)
                .tensor_residual(ctx, state, equation, q),
        ];
        [
            parts.iter().map(|t| t[0]).sum(),
            parts.iter().map(|t| t[1]).sum(),
            parts.iter().map(|t| t[2]).sum(),
        ]
    }

    fn tensor_jacobian_action(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let parts = [
            TensorKernelEdacMomentumConvectionSplit2D::new(self.config)
                .tensor_jacobian_action(ctx, state, direction, equation, q),
            TensorKernelEdacPressureGradient2D::new(self.config)
                .tensor_jacobian_action(ctx, state, direction, equation, q),
            TensorKernelEdacViscousStress2D::new(self.config)
                .tensor_jacobian_action(ctx, state, direction, equation, q),
            TensorKernelEdacPressureDivergence2D::new(self.config)
                .tensor_jacobian_action(ctx, state, direction, equation, q),
            TensorKernelEdacPressureAdvectionSplit2D::new(self.config)
                .tensor_jacobian_action(ctx, state, direction, equation, q),
            TensorKernelEdacPressureDiffusion2D::new(self.config)
                .tensor_jacobian_action(ctx, state, direction, equation, q),
        ];
        [
            parts.iter().map(|t| t[0]).sum(),
            parts.iter().map(|t| t[1]).sum(),
            parts.iter().map(|t| t[2]).sum(),
        ]
    }
}
