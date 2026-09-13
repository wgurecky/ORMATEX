//! Shared parameters and helpers for the 2D EDAC Navier-Stokes kernels.
//!
//! All EDAC volume kernels in this directory solve for `[u, v, p]` and share
//! [`EdacNavierStokes2DConfig`]. The free `pub(crate)` helpers below replace
//! the per-file `field_names` / `check` / `velocity` copies the split grew.

use crate::common::{CellState, FacetCtx, LocalCtx, ShapeFn, TensorCtx};

use super::smagorinsky_lilly::SmagorinskyLilly2D;

/// Shared parameters for the decomposed three-field EDAC Navier-Stokes terms.
#[derive(Clone, Copy, Debug)]
pub struct EdacNavierStokes2DConfig {
    pub rho: f64,
    pub nu: f64,
    pub c0: f64,
    pub pressure_diffusion_factor: f64,
    pub smagorinsky: SmagorinskyLilly2D,
}

impl EdacNavierStokes2DConfig {
    pub fn new(rho: f64, nu: f64, c0: f64, cs: f64) -> Self {
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
            smagorinsky: SmagorinskyLilly2D::new(cs),
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

    pub(crate) fn pressure_diffusivity(&self, ctx: &LocalCtx) -> f64 {
        self.pressure_diffusion_factor * self.c0 * self.smagorinsky.filter_width(ctx)
    }

    pub(crate) fn pressure_diffusivity_tensor(&self, ctx: &TensorCtx<'_>) -> f64 {
        self.pressure_diffusion_factor * self.c0 * self.smagorinsky.filter_width_tensor(ctx)
    }

    pub(crate) fn strain_component(state: &CellState, q: usize, i: usize, j: usize) -> f64 {
        0.5 * (state.grad(i, q, j) + state.grad(j, q, i))
    }

    pub(crate) fn stress(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        q: usize,
        i: usize,
        j: usize,
    ) -> f64 {
        let viscosity = self.nu + self.smagorinsky.eddy_viscosity(ctx, state, q);
        2.0 * viscosity * Self::strain_component(state, q, i, j)
    }

    pub(crate) fn stress_jacobian(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        q: usize,
        i: usize,
        j: usize,
        unknown: usize,
        trial: &ShapeFn<'_>,
    ) -> f64 {
        if unknown >= 2 {
            return 0.0;
        }
        let viscosity = self.nu + self.smagorinsky.eddy_viscosity(ctx, state, q);
        let dviscosity = self
            .smagorinsky
            .eddy_viscosity_gradient_derivative(ctx, state, q, unknown, 0)
            * trial.grad(q, 0)
            + self
                .smagorinsky
                .eddy_viscosity_gradient_derivative(ctx, state, q, unknown, 1)
                * trial.grad(q, 1);
        let dstrain = 0.5
            * ((i == unknown) as usize as f64 * trial.grad(q, j)
                + (j == unknown) as usize as f64 * trial.grad(q, i));
        2.0 * (viscosity * dstrain + dviscosity * Self::strain_component(state, q, i, j))
    }

    /// Both physical components of one viscous-stress row in a single pass.
    ///
    /// Pointwise callers need both columns; evaluating them together shares
    /// one viscosity/dviscosity evaluation instead of recomputing it per
    /// component (and per equation at the call site).
    pub(crate) fn stress_tensor_row(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        q: usize,
        i: usize,
    ) -> [f64; 2] {
        let viscosity = self.nu + self.smagorinsky.eddy_viscosity_tensor(ctx, state, q);
        [
            2.0 * viscosity * Self::strain_component(state, q, i, 0),
            2.0 * viscosity * Self::strain_component(state, q, i, 1),
        ]
    }

    /// Both physical components of one directional-derivative stress row.
    pub(crate) fn stress_tensor_row_directional_derivative(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        q: usize,
        i: usize,
    ) -> [f64; 2] {
        let viscosity = self.nu + self.smagorinsky.eddy_viscosity_tensor(ctx, state, q);
        let dviscosity = self
            .smagorinsky
            .eddy_viscosity_directional_derivative(ctx, state, direction, q);
        let dstrain = [
            0.5 * (direction.grad(i, q, 0) + direction.grad(0, q, i)),
            0.5 * (direction.grad(i, q, 1) + direction.grad(1, q, i)),
        ];
        [
            2.0 * (viscosity * dstrain[0] + dviscosity * Self::strain_component(state, q, i, 0)),
            2.0 * (viscosity * dstrain[1] + dviscosity * Self::strain_component(state, q, i, 1)),
        ]
    }
}

/// Ordered `[u, v, p]` field names shared by every EDAC kernel in this directory.
pub(crate) fn fluid_field_names() -> Option<Vec<String>> {
    Some(["u", "v", "p"].into_iter().map(str::to_owned).collect())
}

/// Assert a weak 2D three-field fluid cell context.
pub(crate) fn check_weak_cell(ctx: &LocalCtx, state: &CellState) {
    assert_eq!(ctx.gdim, 2, "EDAC Navier-Stokes terms require gdim == 2");
    assert_eq!(ctx.ncomp, 1, "fluid fields must be scalar fields");
    assert_eq!(state.nfields, 3, "fluid state must contain [u, v, p]");
}

/// Velocity components at quadrature point `q`.
pub(crate) fn velocity(state: &CellState, q: usize) -> [f64; 2] {
    [state.value(0, q), state.value(1, q)]
}

/// Assert a 2D three-field fluid facet context for boundary kernels.
pub(crate) fn check_boundary_facet(ctx: &FacetCtx, state: &CellState) {
    assert_eq!(ctx.gdim, 2, "EDAC boundary kernels require gdim == 2");
    assert_eq!(ctx.ncomp, 1, "fluid fields must be scalar fields");
    assert_eq!(state.nfields, 3, "fluid state must contain [u, v, p]");
}
