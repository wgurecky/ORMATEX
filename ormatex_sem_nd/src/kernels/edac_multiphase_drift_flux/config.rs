//! Shared parameters for the 2D EDAC drift-flux kernels (`[u, v, p, alpha]`).
//!
//! Mixture model (Ishii 1975 dispersed-bubbly simplification, EDAC base):
//! mixture velocity `u_m` in the momentum balance, EDAC pressure evolution
//! with mixture density `rho_m(alpha) = a*rho_g + (1-a)*rho_l`, void transport
//! `u_g = C0*u_m + V_um` with the um-referenced slip `V_um = Vdj*rho_l/rho_m`
//! (Ishii-Zuber `Vdj` re-referenced from the volumetric flux to `u_m`, exact
//! for `C0 = 1`), buoyant gravity `(rho_m - rho_l)/rho_m`
//! so `alpha -> 0` recovers single-phase EDAC exactly, and Smagorinsky-Lilly
//! eddy viscosity on the mixture velocity (as in the base EDAC kernels).

use crate::common::{LaneState, Lanes, TensorCtx, LANES};
use crate::kernels::edac::smagorinsky_lilly::SmagorinskyLilly2D;

use super::closures::{clamp_alpha, DistributionParameter, IshiiZuberParams};

/// Index of the void fraction in the 2D drift-flux state.
pub const ALPHA_2D: usize = 3;

/// Shared parameters for the decomposed four-field 2D drift-flux terms.
#[derive(Clone, Copy, Debug)]
pub struct DriftFlux2DConfig {
    pub rho_l: f64,
    pub rho_g: f64,
    pub mu_l: f64,
    pub mu_g: f64,
    pub c0: f64,
    pub pressure_diffusion_factor: f64,
    pub smagorinsky: SmagorinskyLilly2D,
    /// Gravity vector (default `[0, -9.81]`); drift rises opposite to it.
    pub gravity: [f64; 2],
    pub ishii_zuber: IshiiZuberParams,
    pub distribution: DistributionParameter,
}

impl DriftFlux2DConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(rho_l: f64, rho_g: f64, mu_l: f64, mu_g: f64, c0: f64, cs: f64) -> Self {
        assert!(
            rho_l.is_finite() && rho_l > 0.0,
            "liquid density must be finite and positive"
        );
        assert!(
            rho_g.is_finite() && rho_g >= 0.0,
            "gas density must be finite and nonnegative"
        );
        assert!(
            mu_l.is_finite() && mu_l >= 0.0,
            "liquid viscosity must be finite and nonnegative"
        );
        assert!(
            mu_g.is_finite() && mu_g >= 0.0,
            "gas viscosity must be finite and nonnegative"
        );
        assert!(
            c0.is_finite() && c0 > 0.0,
            "artificial sound speed must be finite and positive"
        );
        Self {
            rho_l,
            rho_g,
            mu_l,
            mu_g,
            c0,
            pressure_diffusion_factor: 0.1,
            smagorinsky: SmagorinskyLilly2D::new(cs),
            gravity: [0.0, -9.81],
            ishii_zuber: IshiiZuberParams::default(),
            distribution: DistributionParameter::default(),
        }
    }

    pub fn with_gravity(mut self, gravity: [f64; 2]) -> Self {
        self.gravity = gravity;
        self
    }

    pub fn with_gravity_mag(mut self, g: f64) -> Self {
        self.gravity = [0.0, -g];
        self
    }

    pub fn with_pressure_diffusion_factor(mut self, factor: f64) -> Self {
        assert!(
            factor.is_finite() && factor >= 0.0,
            "pressure diffusion factor must be finite and nonnegative"
        );
        self.pressure_diffusion_factor = factor;
        self
    }

    pub fn with_ishii_zuber(mut self, params: IshiiZuberParams) -> Self {
        self.ishii_zuber = params;
        self
    }

    pub fn with_distribution(mut self, distribution: DistributionParameter) -> Self {
        self.distribution = distribution;
        self
    }

    /// Linear mixture density `rho_m(alpha)`.
    pub fn mixture_density(&self, alpha: f64) -> f64 {
        let a = clamp_alpha(alpha);
        a * self.rho_g + (1.0 - a) * self.rho_l
    }

    /// `d rho_m / d alpha` (constant for linear averaging).
    pub fn mixture_density_derivative(&self) -> f64 {
        self.rho_g - self.rho_l
    }

    /// Linear mixture dynamic viscosity `mu_m(alpha)`.
    pub fn mixture_dynamic_viscosity(&self, alpha: f64) -> f64 {
        let a = clamp_alpha(alpha);
        a * self.mu_g + (1.0 - a) * self.mu_l
    }

    /// Mixture kinematic viscosity `nu_m = mu_m / rho_m`.
    pub fn mixture_nu(&self, alpha: f64) -> f64 {
        self.mixture_dynamic_viscosity(alpha) / self.mixture_density(alpha)
    }

    /// `d nu_m / d alpha` for the linear average (quotient rule).
    pub fn mixture_nu_derivative(&self, alpha: f64) -> f64 {
        let rho = self.mixture_density(alpha);
        let mu = self.mixture_dynamic_viscosity(alpha);
        ((self.mu_g - self.mu_l) * rho - mu * self.mixture_density_derivative()) / (rho * rho)
    }

    /// Distribution parameter `C0` (state-independent).
    pub fn distribution_parameter(&self) -> f64 {
        self.distribution.value(self.rho_l, self.rho_g)
    }

    pub fn gravity_magnitude(&self) -> f64 {
        (self.gravity[0] * self.gravity[0] + self.gravity[1] * self.gravity[1]).sqrt()
    }

    /// Rise unit vector (opposite gravity); zero when gravity vanishes.
    pub fn rise_direction(&self) -> [f64; 2] {
        let g = self.gravity_magnitude();
        if g <= 0.0 {
            return [0.0, 0.0];
        }
        [-self.gravity[0] / g, -self.gravity[1] / g]
    }

    /// Um-referenced slip speed `V_um(a) = Vdj(a)*rho_l/rho_m(a)`.
    ///
    /// The Ishii-Zuber correlation measures drift against the volumetric flux
    /// `j`, but every kernel closes on the mass-averaged `u_m`
    /// (`u_g = C0*u_m + V*e`). Re-referencing by `rho_l/rho_m` (exact for
    /// `C0 = 1`: the `u_m`-closed void equation then transports void
    /// identically to Ishii's `j`-form) keeps one reference velocity
    /// throughout, so no `j - u_m` gap terms are needed anywhere. Unity at
    /// `alpha = 0` and for matched phases; regular at `alpha = 1` (zero).
    pub fn slip_speed(&self, alpha: f64) -> f64 {
        self.ishii_zuber
            .drift_speed(alpha, self.rho_l, self.rho_g, self.gravity_magnitude())
            * self.rho_l
            / self.mixture_density(alpha)
    }

    /// Drift-velocity vector (rise opposite gravity) at `alpha`.
    pub fn drift_velocity(&self, alpha: f64) -> [f64; 2] {
        let speed = self.slip_speed(alpha);
        let e = self.rise_direction();
        [e[0] * speed, e[1] * speed]
    }

    /// Hindered drift flux magnitude `F(a) = a*V_um(a)` (um-referenced).
    pub fn drift_flux(&self, alpha: f64) -> f64 {
        clamp_alpha(alpha) * self.slip_speed(alpha)
    }

    /// Exact drift-flux derivative `dF_um/da` (non-monotone hindered regime:
    /// the exact sign matters).
    ///
    /// Quotient rule on `F_dj*rho_l/rho_m` with `F_dj`, `dF_dj/da` the
    /// correlation values; exact (no frozen terms), so linearizations stay
    /// consistent with [`Self::drift_flux`].
    pub fn drift_flux_derivative(&self, alpha: f64) -> f64 {
        if alpha <= 0.0 || alpha >= 1.0 {
            return 0.0;
        }
        let rho = self.mixture_density(alpha);
        let f =
            self.ishii_zuber
                .drift_flux(alpha, self.rho_l, self.rho_g, self.gravity_magnitude());
        let df = self.ishii_zuber.drift_flux_derivative(
            alpha,
            self.rho_l,
            self.rho_g,
            self.gravity_magnitude(),
        );
        self.rho_l * (df * rho - f * self.mixture_density_derivative()) / (rho * rho)
    }

    /// Vapor-phase velocity from the slip relation `u_g = C0*u_m + V_drift`.
    ///
    /// Pointwise post-processing (and future coupling) helper; uses the same
    /// closure evaluation as the void-transport kernels.
    pub fn vapor_velocity(&self, alpha: f64, mixture: [f64; 2]) -> [f64; 2] {
        let c0 = self.distribution_parameter();
        let drift = self.drift_velocity(alpha);
        [c0 * mixture[0] + drift[0], c0 * mixture[1] + drift[1]]
    }

    /// Liquid-phase velocity from the mixture definition
    /// `rho_m*u_m = a*rho_g*u_g + (1-a)*rho_l*u_l`.
    ///
    /// Returns the mixture velocity when the liquid fraction is below
    /// [`MIN_LIQUID_FRACTION`](super::closures::MIN_LIQUID_FRACTION) (pure-gas
    /// limit where the inversion is singular).
    pub fn liquid_velocity(&self, alpha: f64, mixture: [f64; 2]) -> [f64; 2] {
        let liquid_fraction = 1.0 - clamp_alpha(alpha);
        if liquid_fraction <= super::closures::MIN_LIQUID_FRACTION {
            return mixture;
        }
        let vapor = self.vapor_velocity(alpha, mixture);
        let rho_m = self.mixture_density(alpha);
        [
            (rho_m * mixture[0] - clamp_alpha(alpha) * self.rho_g * vapor[0])
                / (liquid_fraction * self.rho_l),
            (rho_m * mixture[1] - clamp_alpha(alpha) * self.rho_g * vapor[1])
                / (liquid_fraction * self.rho_l),
        ]
    }

    /// Lane-packed hindered drift fluxes matching [`drift_flux`](Self::drift_flux).
    ///
    /// # Arguments
    /// * `alphas` - per-lane void fractions.
    ///
    /// # Returns
    /// Per-lane `F(a)`, bit-identical to the scalar path lane-by-lane.
    #[inline]
    pub(crate) fn drift_flux_lanes(&self, alphas: &Lanes) -> Lanes {
        let mut out = [0.0; LANES];
        for l in 0..LANES {
            out[l] = self.drift_flux(alphas[l]);
        }
        out
    }

    /// Lane-packed drift-flux derivatives matching [`drift_flux_derivative`](Self::drift_flux_derivative).
    ///
    /// # Arguments
    /// * `alphas` - per-lane void fractions.
    ///
    /// # Returns
    /// Per-lane `dF_um/da`, bit-identical to the scalar path lane-by-lane.
    #[inline]
    pub(crate) fn drift_flux_derivative_lanes(&self, alphas: &Lanes) -> Lanes {
        let mut out = [0.0; LANES];
        for l in 0..LANES {
            out[l] = self.drift_flux_derivative(alphas[l]);
        }
        out
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

    /// Lane-packed mixture viscous-stress row in scalar operation order.
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
        let alpha = state.value(ALPHA_2D, q);
        let gi0 = state.grad(i, q, 0);
        let gi1 = state.grad(i, q, 1);
        let g0i = state.grad(0, q, i);
        let g1i = state.grad(1, q, i);
        let mut row0 = [0.0; LANES];
        let mut row1 = [0.0; LANES];
        for l in 0..LANES {
            let viscosity = self.mixture_nu(alpha[l]) + nu_t[l];
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
        let (nu_t, dnu_t) = self
            .smagorinsky
            .eddy_viscosity_and_derivative_lanes(ctxs, state, direction, q);
        let alpha = state.value(ALPHA_2D, q);
        let dalpha = direction.value(ALPHA_2D, q);
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
            let viscosity = self.mixture_nu(alpha[l]) + nu_t[l];
            let dviscosity = self.mixture_nu_derivative(alpha[l]) * dalpha[l] + dnu_t[l];
            let s0 = 0.5 * (gi0[l] + g0i[l]);
            let s1 = 0.5 * (gi1[l] + g1i[l]);
            let ds0 = 0.5 * (dgi0[l] + dg0i[l]);
            let ds1 = 0.5 * (dgi1[l] + dg1i[l]);
            row0[l] = 2.0 * (viscosity * ds0 + dviscosity * s0);
            row1[l] = 2.0 * (viscosity * ds1 + dviscosity * s1);
        }
        (row0, row1)
    }
}

/// Ordered `[u, v, p, alpha]` field names shared by every 2D drift-flux kernel.
pub(crate) fn drift_field_names() -> Option<Vec<String>> {
    Some(
        ["u", "v", "p", "alpha"]
            .into_iter()
            .map(str::to_owned)
            .collect(),
    )
}
