//! Shared parameters for the 1D EDAC drift-flux pipe kernels (`[u, p, alpha]`).
//!
//! Same mixture model as the 2D config but laminar (no Smagorinsky, matching
//! the base 1D EDAC kernels). Axial gravity comes from a space-dependent pipe
//! angle `theta(x)` supplied per-kernel as a `MaterialProperty`; the config
//! only stores `|g|`.

use crate::common::{CellState, TensorCtx};

use super::closures::{clamp_alpha, DistributionParameter, IshiiZuberParams};

/// Index of the void fraction in the 1D drift-flux state.
pub const ALPHA_1D: usize = 2;

/// Shared parameters for the decomposed three-field 1D drift-flux terms.
#[derive(Clone, Copy, Debug)]
pub struct DriftFlux1DConfig {
    pub rho_l: f64,
    pub rho_g: f64,
    pub mu_l: f64,
    pub mu_g: f64,
    pub c0: f64,
    pub pressure_diffusion_factor: f64,
    /// Gravity magnitude `|g|`; axial component is `-g*sin(theta(x))`.
    pub gravity: f64,
    pub ishii_zuber: IshiiZuberParams,
    pub distribution: DistributionParameter,
}

impl DriftFlux1DConfig {
    pub fn new(rho_l: f64, rho_g: f64, mu_l: f64, mu_g: f64, c0: f64) -> Self {
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
            gravity: 9.81,
            ishii_zuber: IshiiZuberParams::default(),
            distribution: DistributionParameter::default(),
        }
    }

    pub fn with_gravity(mut self, g: f64) -> Self {
        assert!(
            g.is_finite() && g >= 0.0,
            "gravity must be finite and nonnegative"
        );
        self.gravity = g;
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

    /// Axial drift speed `V*sin(theta)` (positive up-pipe).
    pub fn axial_drift_speed(&self, alpha: f64, theta: f64) -> f64 {
        self.ishii_zuber
            .drift_speed(alpha, self.rho_l, self.rho_g, self.gravity)
            * theta.sin()
    }

    /// Axial hindered drift flux `F(a)*sin(theta)` (positive up-pipe).
    pub fn axial_drift_flux(&self, alpha: f64, theta: f64) -> f64 {
        self.ishii_zuber
            .drift_flux(alpha, self.rho_l, self.rho_g, self.gravity)
            * theta.sin()
    }

    /// Exact axial drift-flux derivative `dF/da*sin(theta)`.
    pub fn axial_drift_flux_derivative(&self, alpha: f64, theta: f64) -> f64 {
        self.ishii_zuber
            .drift_flux_derivative(alpha, self.rho_l, self.rho_g, self.gravity)
            * theta.sin()
    }

    /// Vapor-phase velocity from the slip relation `u_g = C0*u + V_axial`.
    ///
    /// Pointwise post-processing (and future coupling) helper; `theta` is the
    /// pipe angle in radians from horizontal.
    pub fn vapor_velocity_1d(&self, alpha: f64, u: f64, theta: f64) -> f64 {
        self.distribution_parameter() * u + self.axial_drift_speed(alpha, theta)
    }

    /// Liquid-phase velocity from the mixture definition
    /// `rho_m*u = a*rho_g*u_g + (1-a)*rho_l*u_l`.
    ///
    /// Returns the mixture velocity when the liquid fraction is below
    /// [`MIN_LIQUID_FRACTION`](super::closures::MIN_LIQUID_FRACTION).
    pub fn liquid_velocity_1d(&self, alpha: f64, u: f64, theta: f64) -> f64 {
        let liquid_fraction = 1.0 - clamp_alpha(alpha);
        if liquid_fraction <= super::closures::MIN_LIQUID_FRACTION {
            return u;
        }
        let vapor = self.vapor_velocity_1d(alpha, u, theta);
        let rho_m = self.mixture_density(alpha);
        (rho_m * u - clamp_alpha(alpha) * self.rho_g * vapor) / (liquid_fraction * self.rho_l)
    }

    /// Axial gravity component `-g*sin(theta)` (positive up-pipe).
    pub fn axial_gravity(&self, theta: f64) -> f64 {
        -self.gravity * theta.sin()
    }

    pub(crate) fn pressure_diffusivity_tensor(&self, ctx: &TensorCtx<'_>) -> f64 {
        self.pressure_diffusion_factor * self.c0 * ctx.cell_size
    }

    /// Laminar viscous stress `tau = 2 nu_m(alpha) du/dx`.
    pub(crate) fn stress_tensor(&self, state: &CellState<'_>, q: usize) -> f64 {
        2.0 * self.mixture_nu(state.value(ALPHA_1D, q)) * state.grad(0, q, 0)
    }

    pub(crate) fn stress_tensor_directional_derivative(
        &self,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        q: usize,
    ) -> f64 {
        let alpha = state.value(ALPHA_1D, q);
        2.0 * self.mixture_nu(alpha) * direction.grad(0, q, 0)
            + 2.0
                * self.mixture_nu_derivative(alpha)
                * direction.value(ALPHA_1D, q)
                * state.grad(0, q, 0)
    }
}

/// Ordered `[u, p, alpha]` field names shared by every 1D drift-flux kernel.
pub(crate) fn drift_field_names_1d() -> Option<Vec<String>> {
    Some(["u", "p", "alpha"].into_iter().map(str::to_owned).collect())
}
