//! Fused split-form 1D EDAC Navier-Stokes tensor kernel (`[u, p]`).
//!
//! Mathematics: pointwise `(f0, f1x, 0)` triples summing the split advection
//! halves with the conservative pressure-gradient, viscous stress,
//! divergence, and diffusion triples. Owns both equations. Delegates to the
//! part kernels, so it equals their fused sum by construction.
//! Weak counterpart:
//! [`KernelEdacNavierStokesSplit1D`](crate::kernels::edac::weak::navier_stokes_split_1d::KernelEdacNavierStokesSplit1D).

use crate::common::{CellState, TensorCtx};

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
    fn nfields(&self) -> usize {
        2
    }

    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names_1d()
    }

    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let parts = [
            TensorKernelEdacMomentumConvectionSplit1D::new(self.config)
                .tensor_residual(ctx, state, equation, q),
            TensorKernelEdacPressureGradient1D::new(self.config)
                .tensor_residual(ctx, state, equation, q),
            TensorKernelEdacViscousStress1D::new(self.config)
                .tensor_residual(ctx, state, equation, q),
            TensorKernelEdacPressureDivergence1D::new(self.config)
                .tensor_residual(ctx, state, equation, q),
            TensorKernelEdacPressureAdvectionSplit1D::new(self.config)
                .tensor_residual(ctx, state, equation, q),
            TensorKernelEdacPressureDiffusion1D::new(self.config)
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
            TensorKernelEdacMomentumConvectionSplit1D::new(self.config)
                .tensor_jacobian_action(ctx, state, direction, equation, q),
            TensorKernelEdacPressureGradient1D::new(self.config)
                .tensor_jacobian_action(ctx, state, direction, equation, q),
            TensorKernelEdacViscousStress1D::new(self.config)
                .tensor_jacobian_action(ctx, state, direction, equation, q),
            TensorKernelEdacPressureDivergence1D::new(self.config)
                .tensor_jacobian_action(ctx, state, direction, equation, q),
            TensorKernelEdacPressureAdvectionSplit1D::new(self.config)
                .tensor_jacobian_action(ctx, state, direction, equation, q),
            TensorKernelEdacPressureDiffusion1D::new(self.config)
                .tensor_jacobian_action(ctx, state, direction, equation, q),
        ];
        [
            parts.iter().map(|t| t[0]).sum(),
            parts.iter().map(|t| t[1]).sum(),
            parts.iter().map(|t| t[2]).sum(),
        ]
    }
}
