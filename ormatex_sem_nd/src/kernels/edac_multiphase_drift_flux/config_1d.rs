//! Shared parameters for the 1D EDAC drift-flux pipe kernels (`[u, p, alpha]`).
//!
//! Same mixture model as the 2D config but laminar (no Smagorinsky, matching
//! the base 1D EDAC kernels). Axial gravity comes from a space-dependent pipe
//! angle `theta(x)` supplied per-kernel as a `MaterialProperty`; the config
//! only stores `|g|`.

use crate::common::{LaneState, Lanes, TensorCtx, LANES};

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

    /// Um-referenced axial slip speed `V_um*sin(theta)` (positive up-pipe).
    ///
    /// Same `rho_l/rho_m` re-referencing as
    /// [`DriftFlux2DConfig::slip_speed`](super::config::DriftFlux2DConfig::slip_speed).
    pub fn slip_speed_1d(&self, alpha: f64) -> f64 {
        self.ishii_zuber
            .drift_speed(alpha, self.rho_l, self.rho_g, self.gravity)
            * self.rho_l
            / self.mixture_density(alpha)
    }

    /// Axial drift speed `V_um*sin(theta)` (positive up-pipe).
    pub fn axial_drift_speed(&self, alpha: f64, theta: f64) -> f64 {
        self.slip_speed_1d(alpha) * theta.sin()
    }

    /// Axial hindered drift flux `F_um(a)*sin(theta)` (positive up-pipe).
    pub fn axial_drift_flux(&self, alpha: f64, theta: f64) -> f64 {
        clamp_alpha(alpha) * self.slip_speed_1d(alpha) * theta.sin()
    }

    /// Exact axial drift-flux derivative `dF_um/da*sin(theta)`.
    pub fn axial_drift_flux_derivative(&self, alpha: f64, theta: f64) -> f64 {
        if alpha <= 0.0 || alpha >= 1.0 {
            return 0.0;
        }
        let rho = self.mixture_density(alpha);
        let f = self
            .ishii_zuber
            .drift_flux(alpha, self.rho_l, self.rho_g, self.gravity);
        let df =
            self.ishii_zuber
                .drift_flux_derivative(alpha, self.rho_l, self.rho_g, self.gravity);
        self.rho_l * (df * rho - f * self.mixture_density_derivative()) / (rho * rho) * theta.sin()
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

    /// Lane-packed axial drift fluxes matching [`axial_drift_flux`](Self::axial_drift_flux).
    ///
    /// # Arguments
    /// * `alphas` - per-lane void fractions.
    /// * `thetas` - per-lane pipe angles in radians from horizontal.
    ///
    /// # Returns
    /// Per-lane axial hindered drift flux, bit-identical to the scalar path lane-by-lane.
    #[inline]
    pub(crate) fn axial_drift_flux_lanes(&self, alphas: &Lanes, thetas: &Lanes) -> Lanes {
        let mut out = [0.0; LANES];
        for l in 0..LANES {
            out[l] = self.axial_drift_flux(alphas[l], thetas[l]);
        }
        out
    }

    /// Lane-packed axial drift-flux derivatives matching [`axial_drift_flux_derivative`](Self::axial_drift_flux_derivative).
    ///
    /// # Arguments
    /// * `alphas` - per-lane void fractions.
    /// * `thetas` - per-lane pipe angles in radians from horizontal.
    ///
    /// # Returns
    /// Per-lane `dF_um/da*sin(theta)`, bit-identical to the scalar path lane-by-lane.
    #[inline]
    pub(crate) fn axial_drift_flux_derivative_lanes(
        &self,
        alphas: &Lanes,
        thetas: &Lanes,
    ) -> Lanes {
        let mut out = [0.0; LANES];
        for l in 0..LANES {
            out[l] = self.axial_drift_flux_derivative(alphas[l], thetas[l]);
        }
        out
    }

    /// Lane-packed pressure diffusivities in scalar operation order.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    ///
    /// # Returns
    /// Per-lane `(factor * c0) * cell_size[l]` with scalar operation order.
    #[inline]
    pub(crate) fn pressure_diffusivities_tensor_lanes(&self, ctxs: &[TensorCtx<'_>]) -> Lanes {
        let mut out = [0.0; LANES];
        for l in 0..LANES {
            out[l] = self.pressure_diffusion_factor * self.c0 * ctxs[l].cell_size;
        }
        out
    }

    /// Lane-packed laminar viscous stresses in scalar operation order.
    ///
    /// # Arguments
    /// * `state` - lane-packed solution.
    /// * `q` - quadrature-point index shared by all lanes.
    ///
    /// # Returns
    /// Per-lane `2*nu_m*du/dx`, bit-identical to the scalar path lane-by-lane.
    #[inline]
    pub(crate) fn stress_tensor_lanes(&self, state: &LaneState<'_>, q: usize) -> Lanes {
        let alpha = state.value(ALPHA_1D, q);
        let grad = state.grad(0, q, 0);
        let mut out = [0.0; LANES];
        for l in 0..LANES {
            out[l] = 2.0 * self.mixture_nu(alpha[l]) * grad[l];
        }
        out
    }

    /// Lane-packed viscous-stress directional derivatives in scalar operation order.
    ///
    /// # Arguments
    /// * `state` - lane-packed linearization point.
    /// * `direction` - lane-packed Gateaux direction.
    /// * `q` - quadrature-point index shared by all lanes.
    ///
    /// # Returns
    /// Per-lane `2*nu_m*ddu/dx + 2*dnu_m/da*da*du/dx` in scalar operation order.
    #[inline]
    pub(crate) fn stress_tensor_directional_derivative_lanes(
        &self,
        state: &LaneState<'_>,
        direction: &LaneState<'_>,
        q: usize,
    ) -> Lanes {
        let alpha = state.value(ALPHA_1D, q);
        let grad = state.grad(0, q, 0);
        let dalpha = direction.value(ALPHA_1D, q);
        let dgrad = direction.grad(0, q, 0);
        let mut out = [0.0; LANES];
        for l in 0..LANES {
            out[l] = 2.0 * self.mixture_nu(alpha[l]) * dgrad[l]
                + 2.0 * self.mixture_nu_derivative(alpha[l]) * dalpha[l] * grad[l];
        }
        out
    }
}

/// Ordered `[u, p, alpha]` field names shared by every 1D drift-flux kernel.
pub(crate) fn drift_field_names_1d() -> Option<Vec<String>> {
    Some(["u", "p", "alpha"].into_iter().map(str::to_owned).collect())
}
