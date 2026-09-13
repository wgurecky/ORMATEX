//! Shared parameters and helpers for the 1D EDAC pipe-flow kernels.
//!
//! All 1D EDAC volume kernels in this directory solve for `[u, p]` and share
//! [`EdacNavierStokes1DConfig`]. Laminar only (no Smagorinsky model); the
//! pressure diffusivity scales with the cell length like the 2D filter width.

use crate::common::{CellState, LocalCtx, TensorCtx};

/// Shared parameters for the decomposed two-field 1D EDAC terms.
#[derive(Clone, Copy, Debug)]
pub struct EdacNavierStokes1DConfig {
    pub rho: f64,
    pub nu: f64,
    pub c0: f64,
    pub pressure_diffusion_factor: f64,
}

impl EdacNavierStokes1DConfig {
    pub fn new(rho: f64, nu: f64, c0: f64) -> Self {
        assert!(
            rho.is_finite() && rho > 0.0,
            "density must be finite and positive"
        );
        assert!(
            nu.is_finite() && nu >= 0.0,
            "kinematic viscosity must be finite and nonnegative"
        );
        assert!(
            c0.is_finite() && c0 > 0.0,
            "artificial sound speed must be finite and positive"
        );
        Self {
            rho,
            nu,
            c0,
            pressure_diffusion_factor: 0.1,
        }
    }

    pub fn with_pressure_diffusion_factor(mut self, factor: f64) -> Self {
        assert!(
            factor.is_finite() && factor >= 0.0,
            "pressure diffusion factor must be finite and nonnegative"
        );
        self.pressure_diffusion_factor = factor;
        self
    }

    /// Cell length from the weak context (matches the 1D `cell_sizes` convention).
    fn filter_width(ctx: &LocalCtx) -> f64 {
        let length: f64 = ctx.wts.iter().zip(ctx.jdets).map(|(&w, &j)| w * j).sum();
        assert!(
            length.is_finite() && length > 0.0,
            "cell length must be positive"
        );
        length.sqrt()
    }

    pub(crate) fn pressure_diffusivity(&self, ctx: &LocalCtx) -> f64 {
        self.pressure_diffusion_factor * self.c0 * Self::filter_width(ctx)
    }

    pub(crate) fn pressure_diffusivity_tensor(&self, ctx: &TensorCtx<'_>) -> f64 {
        self.pressure_diffusion_factor * self.c0 * ctx.cell_size
    }

    /// Viscous stress `tau = 2 nu du/dx` at quadrature point `q`.
    pub(crate) fn stress(&self, state: &CellState, q: usize) -> f64 {
        2.0 * self.nu * state.grad(0, q, 0)
    }

    /// Gateaux derivative of [`stress`](Self::stress) in the `trial` direction.
    pub(crate) fn stress_jacobian(&self, trial_grad: f64, unknown: usize) -> f64 {
        if unknown != 0 {
            return 0.0;
        }
        2.0 * self.nu * trial_grad
    }

    /// Viscous stress for the tensor x-flux slot.
    pub(crate) fn stress_tensor(&self, state: &CellState<'_>, q: usize) -> f64 {
        self.stress(state, q)
    }

    /// Directional-derivative stress for the tensor action slot.
    pub(crate) fn stress_tensor_directional_derivative(
        &self,
        direction: &CellState<'_>,
        q: usize,
    ) -> f64 {
        2.0 * self.nu * direction.grad(0, q, 0)
    }
}

/// Ordered `[u, p]` field names shared by every 1D EDAC kernel in this directory.
pub(crate) fn fluid_field_names_1d() -> Option<Vec<String>> {
    Some(["u", "p"].into_iter().map(str::to_owned).collect())
}

/// Assert a weak 1D two-field fluid cell context.
pub(crate) fn check_weak_cell_1d(ctx: &LocalCtx, state: &CellState) {
    assert_eq!(ctx.gdim, 1, "1D EDAC terms require gdim == 1");
    assert_eq!(ctx.ncomp, 1, "fluid fields must be scalar fields");
    assert_eq!(state.nfields, 2, "fluid state must contain [u, p]");
}
