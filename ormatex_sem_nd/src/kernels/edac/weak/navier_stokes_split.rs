//! Fused split-form EDAC Navier-Stokes kernel (`[u, v, p]`, weak form).
//!
//! Mathematics: one kernel implementing the split advection formulation used
//! by `cavity_comp_tensor`-style solvers without assembling six separate
//! terms. Momentum: `1/2 [(u . grad) u + div(u u)] + grad(p)/rho - div(tau)`;
//! pressure: `rho c0^2 div(u) + 1/2 [(u . grad) p + div(u p)] - div(k grad(p))`.
//! Each integrand delegates to the corresponding `*_split` / conservative part
//! kernel, so the fused kernel equals their sum by construction.
//!
//! Boundary pairing: the conservative halves (`div(u u)`, `div(u p)`) need
//! their facet fluxes — use a [`SplitBoundaryFlux`](crate::kernels::edac::weak::split_boundary_flux::KernelEdacSplitBoundaryFlux2D)
//! default with [`Dong`](crate::kernels::edac::weak::dong_outflow::KernelEdacDongOutflow2D) or
//! [`directional-do-nothing`](crate::kernels::edac::weak::directional_do_nothing::KernelEdacDirectionalDoNothing2D)
//! `.with_split_flux(true)` outlets, plus no-slip/slip walls. Tensor
//! counterpart:
//! [`TensorKernelEdacNavierStokesSplit2D`](crate::kernels::edac::tensor::navier_stokes_split::TensorKernelEdacNavierStokesSplit2D).

use crate::common::{CellState, LocalCtx};

use super::momentum_convection_split::KernelEdacMomentumConvectionSplit2D;
use super::pressure_advection_split::KernelEdacPressureAdvectionSplit2D;
use super::pressure_diffusion::KernelEdacPressureDiffusion2D;
use super::pressure_divergence::KernelEdacPressureDivergence2D;
use super::pressure_gradient::KernelEdacPressureGradient2D;
use super::viscous_stress::KernelEdacViscousStress2D;
use crate::kernels::common::ResidualKernel;
use crate::kernels::edac::config::{fluid_field_names, EdacNavierStokes2DConfig};

/// Fused split-form EDAC Navier-Stokes kernel for `[u, v, p]`
/// (owns all equations; delegates to the part kernels).
pub struct KernelEdacNavierStokesSplit2D {
    pub config: EdacNavierStokes2DConfig,
}

impl KernelEdacNavierStokesSplit2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl ResidualKernel for KernelEdacNavierStokesSplit2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names()
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        KernelEdacMomentumConvectionSplit2D::new(self.config)
            .residual_integrand(ctx, state, equation, q, test_i)
            + KernelEdacPressureGradient2D::new(self.config)
                .residual_integrand(ctx, state, equation, q, test_i)
            + KernelEdacViscousStress2D::new(self.config)
                .residual_integrand(ctx, state, equation, q, test_i)
            + KernelEdacPressureDivergence2D::new(self.config)
                .residual_integrand(ctx, state, equation, q, test_i)
            + KernelEdacPressureAdvectionSplit2D::new(self.config)
                .residual_integrand(ctx, state, equation, q, test_i)
            + KernelEdacPressureDiffusion2D::new(self.config)
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
        KernelEdacMomentumConvectionSplit2D::new(self.config)
            .jacobian_integrand(ctx, state, equation, unknown, q, test_i, trial_i)
            + KernelEdacPressureGradient2D::new(self.config)
                .jacobian_integrand(ctx, state, equation, unknown, q, test_i, trial_i)
            + KernelEdacViscousStress2D::new(self.config)
                .jacobian_integrand(ctx, state, equation, unknown, q, test_i, trial_i)
            + KernelEdacPressureDivergence2D::new(self.config)
                .jacobian_integrand(ctx, state, equation, unknown, q, test_i, trial_i)
            + KernelEdacPressureAdvectionSplit2D::new(self.config)
                .jacobian_integrand(ctx, state, equation, unknown, q, test_i, trial_i)
            + KernelEdacPressureDiffusion2D::new(self.config)
                .jacobian_integrand(ctx, state, equation, unknown, q, test_i, trial_i)
    }
}
