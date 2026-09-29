//! Shared parameters and helpers for the 2D EDAC Navier-Stokes kernels.
//!
//! All EDAC volume kernels in this directory solve for `[u, v, p]` and share
//! [`EdacNavierStokes2DConfig`]. The free `pub(crate)` helpers below replace
//! the per-file `field_names` / `check` / `velocity` copies the split grew.

use crate::common::{CellState, FacetCtx, LaneState, Lanes, LocalCtx, ShapeFn, TensorCtx, LANES};

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

    #[inline]
    pub(crate) fn pressure_diffusivity(&self, ctx: &LocalCtx) -> f64 {
        self.pressure_diffusion_factor * self.c0 * self.smagorinsky.filter_width(ctx)
    }

    /// Lane-packed pressure diffusivities in scalar operation order.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    ///
    /// # Returns
    /// Per-lane `(factor * c0) * width[l]` with scalar operation order.
    #[inline]
    pub(crate) fn pressure_diffusivities_tensor_lanes(&self, ctxs: &[TensorCtx<'_>]) -> Lanes {
        let widths = self.smagorinsky.filter_widths_tensor_lanes(ctxs);
        let mut out = [0.0; LANES];
        for l in 0..LANES {
            out[l] = self.pressure_diffusion_factor * self.c0 * widths[l];
        }
        out
    }

    /// Lane-packed viscous-stress row in scalar operation order.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed solution.
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `i` - momentum row (`0` or `1`).
    ///
    /// # Returns
    /// `(row_x, row_y)` lane vectors with `row = 2*visc*strain` in scalar order.
    #[inline]
    pub(crate) fn stress_tensor_row_lanes(
        &self,
        ctxs: &[TensorCtx<'_>],
        state: &LaneState<'_>,
        q: usize,
        i: usize,
    ) -> (Lanes, Lanes) {
        let nu_t = self
            .smagorinsky
            .eddy_viscosities_tensor_lanes(ctxs, state, q);
        let gi0 = state.grad(i, q, 0);
        let gi1 = state.grad(i, q, 1);
        let g0i = state.grad(0, q, i);
        let g1i = state.grad(1, q, i);
        let mut row0 = [0.0; LANES];
        let mut row1 = [0.0; LANES];
        for l in 0..LANES {
            let viscosity = self.nu + nu_t[l];
            let s0 = 0.5 * (gi0[l] + g0i[l]);
            let s1 = 0.5 * (gi1[l] + g1i[l]);
            row0[l] = 2.0 * viscosity * s0;
            row1[l] = 2.0 * viscosity * s1;
        }
        (row0, row1)
    }

    /// Lane-packed directional-derivative stress row in scalar operation order.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed linearization point.
    /// * `direction` - lane-packed Gateaux direction.
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `i` - momentum row (`0` or `1`).
    ///
    /// # Returns
    /// `(row_x, row_y)` lane vectors with `row = 2*(visc*dstrain+dvisc*strain)`
    /// in scalar order.
    #[inline]
    pub(crate) fn stress_tensor_row_directional_derivative_lanes(
        &self,
        ctxs: &[TensorCtx<'_>],
        state: &LaneState<'_>,
        direction: &LaneState<'_>,
        q: usize,
        i: usize,
    ) -> (Lanes, Lanes) {
        let (nu_t, dvisc) = self
            .smagorinsky
            .eddy_viscosity_and_derivative_lanes(ctxs, state, direction, q);
        let gi0 = state.grad(i, q, 0);
        let gi1 = state.grad(i, q, 1);
        let g0i = state.grad(0, q, i);
        let g1i = state.grad(1, q, i);
        let dgi0 = direction.grad(i, q, 0);
        let dgi1 = direction.grad(i, q, 1);
        let dg0i = direction.grad(0, q, i);
        let dg1i = direction.grad(1, q, i);
        let mut row0 = [0.0; LANES];
        let mut row1 = [0.0; LANES];
        for l in 0..LANES {
            let viscosity = self.nu + nu_t[l];
            let s0 = 0.5 * (gi0[l] + g0i[l]);
            let s1 = 0.5 * (gi1[l] + g1i[l]);
            let ds0 = 0.5 * (dgi0[l] + dg0i[l]);
            let ds1 = 0.5 * (dgi1[l] + dg1i[l]);
            row0[l] = 2.0 * (viscosity * ds0 + dvisc[l] * s0);
            row1[l] = 2.0 * (viscosity * ds1 + dvisc[l] * s1);
        }
        (row0, row1)
    }

    #[inline]
    pub(crate) fn strain_component(state: &CellState, q: usize, i: usize, j: usize) -> f64 {
        0.5 * (state.grad(i, q, j) + state.grad(j, q, i))
    }

    #[inline]
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

    #[inline]
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
}

/// Ordered `[u, v, p]` field names shared by every EDAC kernel in this directory.
#[inline]
pub(crate) fn fluid_field_names() -> Option<Vec<String>> {
    Some(["u", "v", "p"].into_iter().map(str::to_owned).collect())
}

/// Assert a weak 2D three-field fluid cell context.
#[inline]
pub(crate) fn check_weak_cell(ctx: &LocalCtx, state: &CellState) {
    assert_eq!(ctx.gdim, 2, "EDAC Navier-Stokes terms require gdim == 2");
    assert_eq!(ctx.ncomp, 1, "fluid fields must be scalar fields");
    assert_eq!(state.nfields, 3, "fluid state must contain [u, v, p]");
}

/// Velocity components at quadrature point `q`.
#[inline]
pub(crate) fn velocity(state: &CellState, q: usize) -> [f64; 2] {
    [state.value(0, q), state.value(1, q)]
}

/// Assert a 2D three-field fluid facet context for boundary kernels.
#[inline]
pub(crate) fn check_boundary_facet(ctx: &FacetCtx, state: &CellState) {
    assert_eq!(ctx.gdim, 2, "EDAC boundary kernels require gdim == 2");
    assert_eq!(ctx.ncomp, 1, "fluid fields must be scalar fields");
    assert_eq!(state.nfields, 3, "fluid state must contain [u, v, p]");
}
