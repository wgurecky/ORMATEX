//! Fused split-form 1D EDAC Navier-Stokes kernel (`[u, p]`, weak form).
//!
//! Mathematics: one kernel implementing the split advection formulation used
//! by heated-pipe solvers without assembling six separate terms. Momentum:
//! `1/2 [u du/dx + d(uu)/dx] + dp/dx/rho - d(tau)/dx`; pressure:
//! `rho c0^2 du/dx + 1/2 [u dp/dx + d(up)/dx] - d(k dp/dx)/dx`.
//! Each integrand delegates to the corresponding `*_split_1d` / conservative
//! part kernel, so the fused kernel equals their sum by construction.
//! Tensor counterpart:
//! [`TensorKernelEdacNavierStokesSplit1D`](crate::kernels::edac::tensor::navier_stokes_split_1d::TensorKernelEdacNavierStokesSplit1D).

use crate::common::{CellState, LocalCtx};

use super::momentum_convection_split_1d::KernelEdacMomentumConvectionSplit1D;
use super::pressure_advection_split_1d::KernelEdacPressureAdvectionSplit1D;
use super::pressure_diffusion_1d::KernelEdacPressureDiffusion1D;
use super::pressure_divergence_1d::KernelEdacPressureDivergence1D;
use super::pressure_gradient_1d::KernelEdacPressureGradient1D;
use super::viscous_stress_1d::KernelEdacViscousStress1D;
use crate::kernels::common::ResidualKernel;
use crate::kernels::edac::config_1d::{fluid_field_names_1d, EdacNavierStokes1DConfig};

/// Fused split-form 1D EDAC Navier-Stokes kernel for `[u, p]`
/// (owns all equations; delegates to the part kernels).
pub struct KernelEdacNavierStokesSplit1D {
    pub config: EdacNavierStokes1DConfig,
}

impl KernelEdacNavierStokesSplit1D {
    pub fn new(config: EdacNavierStokes1DConfig) -> Self {
        Self { config }
    }
}

impl ResidualKernel for KernelEdacNavierStokesSplit1D {
    fn nfields(&self) -> usize {
        2
    }

    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names_1d()
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        KernelEdacMomentumConvectionSplit1D::new(self.config)
            .residual_integrand(ctx, state, equation, q, test_i)
            + KernelEdacPressureGradient1D::new(self.config)
                .residual_integrand(ctx, state, equation, q, test_i)
            + KernelEdacViscousStress1D::new(self.config)
                .residual_integrand(ctx, state, equation, q, test_i)
            + KernelEdacPressureDivergence1D::new(self.config)
                .residual_integrand(ctx, state, equation, q, test_i)
            + KernelEdacPressureAdvectionSplit1D::new(self.config)
                .residual_integrand(ctx, state, equation, q, test_i)
            + KernelEdacPressureDiffusion1D::new(self.config)
                .residual_integrand(ctx, state, equation, q, test_i)
    }

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        KernelEdacMomentumConvectionSplit1D::new(self.config)
            .jacobian_integrand(ctx, state, equation, unknown, q, test_i, trial_i)
            + KernelEdacPressureGradient1D::new(self.config)
                .jacobian_integrand(ctx, state, equation, unknown, q, test_i, trial_i)
            + KernelEdacViscousStress1D::new(self.config)
                .jacobian_integrand(ctx, state, equation, unknown, q, test_i, trial_i)
            + KernelEdacPressureDivergence1D::new(self.config)
                .jacobian_integrand(ctx, state, equation, unknown, q, test_i, trial_i)
            + KernelEdacPressureAdvectionSplit1D::new(self.config)
                .jacobian_integrand(ctx, state, equation, unknown, q, test_i, trial_i)
            + KernelEdacPressureDiffusion1D::new(self.config)
                .jacobian_integrand(ctx, state, equation, unknown, q, test_i, trial_i)
    }
}
