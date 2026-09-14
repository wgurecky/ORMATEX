//! Drift-flux closure relations (Ishii-Zuber drift velocity, distribution parameter).
//!
//! Citations:
//! * M. Ishii, *Thermo-Fluid Dynamic Theory of Two-Phase Flow* (1975) —
//!   drift-flux formulation `u_g = C0 * j + V_gj` relating the gas velocity to
//!   the volumetric flux `j` of the mixture.
//! * M. Ishii and N. Zuber, "Drag coefficient and relative velocity in bubbly,
//!   droplet or particulate flows," *AIChE J.* 25(5), 1979 —
//!   `V_gj = sqrt(2) * (sigma * g * drho / rho_l^2)^{1/4} * (1 - alpha)^{1.75}`
//!   bubble-regime drift speed used here.
//! * G. B. Dix, "Vapor void fractions for forced convection with subcooled
//!   boiling at low pressure," *GEAP-2047*, 1971 — distribution-parameter
//!   correlation; here reduced to the density-ratio form
//!   `C0 = 1.2 - 0.2 * sqrt(rho_g / rho_l)` of the Zuber-Findlay family which
//!   Dix's correlation approaches for fully developed flow. The full local
//!   quality/void-dependent Dix form is future work.
//! * S. Lopez de Bertodano, R. T. Lahey Jr. and O. C. Jones, "Turbulent
//!   bubbly two-phase flow data in a triangular duct," *Nucl. Eng. Des.* 1994;
//!   M. Burns et al., "The Favre averaged drag model for turbulent dispersion
//!   in Eulerian multi-phase flows," 2004 — Fickian turbulent-dispersion flux
//!   `-D_td * grad(alpha)`; here with the simplest constant `D_td`.

/// Clamp a void fraction to the physical range for closure evaluation.
#[inline(always)]
pub fn clamp_alpha(alpha: f64) -> f64 {
    alpha.clamp(0.0, 1.0)
}

/// Ishii-Zuber bubble-regime drift-speed parameters.
///
/// `vgj_scale` defaults to `sqrt(2)` (Ishii-Zuber); `sigma` is the surface
/// tension. The speed is aligned with the rise direction by the caller
/// (opposite the gravity vector; axial `sin(theta)` factor in 1D).
#[derive(Clone, Copy, Debug)]
pub struct IshiiZuberParams {
    pub sigma: f64,
    pub vgj_scale: f64,
}

impl Default for IshiiZuberParams {
    fn default() -> Self {
        Self {
            sigma: 0.0728,
            vgj_scale: std::f64::consts::SQRT_2,
        }
    }
}

impl IshiiZuberParams {
    pub fn new(sigma: f64) -> Self {
        assert!(
            sigma.is_finite() && sigma > 0.0,
            "surface tension must be finite and positive"
        );
        Self {
            sigma,
            vgj_scale: std::f64::consts::SQRT_2,
        }
    }

    pub fn with_vgj_scale(mut self, scale: f64) -> Self {
        assert!(
            scale.is_finite() && scale >= 0.0,
            "drift scale must be finite and nonnegative"
        );
        self.vgj_scale = scale;
        self
    }

    /// Drift speed magnitude `|V_gj|` at void fraction `alpha`.
    ///
    /// Harmathy terminal speed scaled by Ishii-Zuber hindrance `(1-a)^{1.75}`.
    /// Returns `0` for single-phase liquid (`g == 0`, `drho <= 0`, or
    /// `alpha >= 1`).
    pub fn drift_speed(&self, alpha: f64, rho_l: f64, rho_g: f64, g_mag: f64) -> f64 {
        let a = clamp_alpha(alpha);
        let drho = rho_l - rho_g;
        if g_mag <= 0.0 || drho <= 0.0 || a >= 1.0 {
            return 0.0;
        }
        let terminal = (self.sigma * g_mag * drho / (rho_l * rho_l)).powf(0.25);
        self.vgj_scale * terminal * (1.0 - a).powf(1.75)
    }

    /// Gas drift flux magnitude `F(a) = a * V_gj(a)` (clamped, bounded).
    pub fn drift_flux(&self, alpha: f64, rho_l: f64, rho_g: f64, g_mag: f64) -> f64 {
        clamp_alpha(alpha) * self.drift_speed(alpha, rho_l, rho_g, g_mag)
    }

    /// Exact derivative `dF/da` of the drift flux.
    ///
    /// `F(a) = S*a*(1-a)^{1.75}` is non-monotone (`dF/da < 0` past `a = 4/11`,
    /// the kinematic-shock regime), so linearizations must carry the exact
    /// sign. Regular at `a = 1` (`dF/da -> 0`); zero outside `[0, 1]` where
    /// the clamp freezes the flux.
    pub fn drift_flux_derivative(&self, alpha: f64, rho_l: f64, rho_g: f64, g_mag: f64) -> f64 {
        if alpha <= 0.0 || alpha >= 1.0 {
            return 0.0;
        }
        let drho = rho_l - rho_g;
        if g_mag <= 0.0 || drho <= 0.0 {
            return 0.0;
        }
        let v = self.drift_speed(alpha, rho_l, rho_g, g_mag);
        v * (1.0 - 1.75 * alpha / (1.0 - alpha))
    }
}

/// Liquid-fraction floor for the phase-velocity inversion below: when
/// `(1 - alpha)` is smaller, the liquid velocity is indistinguishable from
/// the mixture velocity and the division is skipped.
pub const MIN_LIQUID_FRACTION: f64 = 1e-12;

/// Distribution parameter `C0` in `u_g = C0 * j + V_gj`.
#[derive(Clone, Copy, Debug)]
pub enum DistributionParameter {
    /// Constant `C0` (default `1.0`; `1.1-1.2` typical for bubbly pipe flow).
    Constant(f64),
    /// Dix (1971) density-ratio form `1.2 - 0.2*sqrt(rho_g/rho_l)`.
    Dix,
}

impl Default for DistributionParameter {
    fn default() -> Self {
        Self::Constant(1.0)
    }
}

impl DistributionParameter {
    /// Evaluate `C0`. State-independent by construction, so Jacobians stay exact.
    pub fn value(&self, rho_l: f64, rho_g: f64) -> f64 {
        match *self {
            Self::Constant(c0) => c0,
            Self::Dix => {
                assert!(
                    rho_l > 0.0 && rho_g >= 0.0,
                    "densities must be nonnegative with positive liquid density"
                );
                1.2 - 0.2 * (rho_g / rho_l).sqrt()
            }
        }
    }
}
