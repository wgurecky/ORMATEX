//! Shared parameters for the 2D EDAC drift-flux kernels (`[u, v, p, alpha]`).
//!
//! Mixture model (Ishii 1975 dispersed-bubbly simplification, EDAC base):
//! mixture velocity `u_m` in the momentum balance, EDAC pressure evolution
//! with mixture density `rho_m(alpha) = a*rho_g + (1-a)*rho_l`, void transport
//! `u_g = C0*j + V_gj` with `j ~= u_m`, buoyant gravity `(rho_m - rho_l)/rho_m`
//! so `alpha -> 0` recovers single-phase EDAC exactly, and Smagorinsky-Lilly
//! eddy viscosity on the mixture velocity (as in the base EDAC kernels).

use crate::common::{CellState, TensorCtx};
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

    /// Drift-velocity vector (rise opposite gravity) at `alpha`.
    pub fn drift_velocity(&self, alpha: f64) -> [f64; 2] {
        let g = self.gravity_magnitude();
        let speed = self
            .ishii_zuber
            .drift_speed(alpha, self.rho_l, self.rho_g, g);
        let e = self.rise_direction();
        [e[0] * speed, e[1] * speed]
    }

    /// Hindered drift flux magnitude `F(a) = a*V_gj(a)`.
    pub fn drift_flux(&self, alpha: f64) -> f64 {
        self.ishii_zuber
            .drift_flux(alpha, self.rho_l, self.rho_g, self.gravity_magnitude())
    }

    /// Exact drift-flux derivative `dF/da` (sign-correct past `a = 4/11`).
    pub fn drift_flux_derivative(&self, alpha: f64) -> f64 {
        self.ishii_zuber.drift_flux_derivative(
            alpha,
            self.rho_l,
            self.rho_g,
            self.gravity_magnitude(),
        )
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

    pub(crate) fn pressure_diffusivity_tensor(&self, ctx: &TensorCtx<'_>) -> f64 {
        self.pressure_diffusion_factor * self.c0 * self.smagorinsky.filter_width_tensor(ctx)
    }

    pub(crate) fn strain_component(state: &CellState, q: usize, i: usize, j: usize) -> f64 {
        0.5 * (state.grad(i, q, j) + state.grad(j, q, i))
    }

    /// Mixture viscous-stress row with Smagorinsky eddy viscosity on `u_m`.
    ///
    /// `tau_ij = 2 (nu_m(alpha) + nu_t) S_ij`; shares one viscosity eval.
    pub(crate) fn stress_tensor_row(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        q: usize,
        i: usize,
    ) -> [f64; 2] {
        let viscosity = self.mixture_nu(state.value(ALPHA_2D, q))
            + self.smagorinsky.eddy_viscosity_tensor(ctx, state, q);
        [
            2.0 * viscosity * Self::strain_component(state, q, i, 0),
            2.0 * viscosity * Self::strain_component(state, q, i, 1),
        ]
    }

    pub(crate) fn stress_tensor_row_directional_derivative(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        q: usize,
        i: usize,
    ) -> [f64; 2] {
        let alpha = state.value(ALPHA_2D, q);
        let dalpha = direction.value(ALPHA_2D, q);
        let viscosity =
            self.mixture_nu(alpha) + self.smagorinsky.eddy_viscosity_tensor(ctx, state, q);
        let dviscosity = self.mixture_nu_derivative(alpha) * dalpha
            + self
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

/// Ordered `[u, v, p, alpha]` field names shared by every 2D drift-flux kernel.
pub(crate) fn drift_field_names() -> Option<Vec<String>> {
    Some(
        ["u", "v", "p", "alpha"]
            .into_iter()
            .map(str::to_owned)
            .collect(),
    )
}

/// Mixture velocity components at quadrature point `q`.
pub(crate) fn velocity(state: &CellState, q: usize) -> [f64; 2] {
    [state.value(0, q), state.value(1, q)]
}
