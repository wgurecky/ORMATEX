//! Tensor free-surface boundary for drift-flux EDAC (`[u, v, p, alpha]`).
//!
//! Mathematics: the mixture momentum/pressure equations reuse the base EDAC
//! directional-do-nothing outflow (`-p n / rho_l` traction plus the
//! `max(-u.n, 0)` backflow penalty, with split-flux halves), so the pressure
//! datum is set weakly and no Dirichlet pressure pin is needed. On top of
//! that, the liquid no-penetration condition (`u_l . n = 0`) is enforced
//! weakly with a normal-direction penalty `gamma * (u_l.n) * n_i` on the two
//! momentum rows: the tangential liquid velocity stays free (slip-like) while
//! vapor/void keep a vertical component so gas vents. `u_l` is the drift-flux
//! closure inversion `rho_m*u_m = a*rho_g*u_g + (1-a)*rho_l*u_l` with
//! `u_g = C0*u_m + V_drift*e`, i.e.
//! `u_l.n = [(rho_m - a*rho_g*C0)*u_m.n - a*rho_g*V*en]/((1-a)*rho_l)`.
//! The void equation vents gas with a no-reentry flux
//! `1/2 C0 max(u.n, 0) alpha + F_vent(alpha) max(e.n, 0)`: the split-consistent
//! advective half (clipped so void never re-enters) plus an unhindered
//! degassing flux `F_vent = a*V0` with the terminal speed `V0` (no
//! `(1-a)^1.75` hindrance). Hindrance is an interior dispersed-flow effect;
//! at the surface gas coalesces and escapes to the atmosphere, and the
//! hindered flux `F -> 0` as `a -> 1` would choke venting exactly when the
//! plume needs it most (continuous injection cannot balance otherwise and
//! gas accumulates to `a -> 1` blowup). The linear vent is monotone
//! (`dF_vent/da = V0 > 0`, no kinematic shock) and has capacity `V0` per
//! unit area at `a = 1`. At `u.n = 0` the advective Jacobian uses the
//! outflow-side derivative, matching the directional-do-nothing convention.
use crate::common::{LaneState, Lanes, TensorFacetCtx, LANES};
use crate::kernels::common::StateTensorBoundaryIntegrator;
use crate::kernels::edac::weak::directional_do_nothing::{
    directional_jacobian_action, directional_residual,
};
use crate::kernels::edac_multiphase_drift_flux::closures::MIN_LIQUID_FRACTION;
use crate::kernels::edac_multiphase_drift_flux::config::{
    drift_field_names, DriftFlux2DConfig, ALPHA_2D,
};

/// Default normal penalty `gamma` for the liquid no-penetration term.
/// Residual units are traction (`m^2/s^2`), so `gamma` is a velocity scale:
/// `u_l.n ~ traction/gamma`. `1e6` keeps every Newton iteration near the
/// constraint manifold while staying soft enough for GMRES at late times
/// (larger gammas stall the linear solve once gas accumulates); exact
/// `u_l.n == 0` at the surface is enforced by projection in the bubble-plume
/// driver (see `project_liquid_no_penetration` there).
pub const DEFAULT_FREE_SURFACE_PENALTY: f64 = 1.0e6;

/// Tensor free-surface boundary for drift-flux EDAC (owns all four equations).
pub struct TensorDriftFreeSurface2D {
    pub config: DriftFlux2DConfig,
    pub penalty: f64,
}

impl TensorDriftFreeSurface2D {
    pub fn new(config: DriftFlux2DConfig) -> Self {
        Self {
            config,
            penalty: DEFAULT_FREE_SURFACE_PENALTY,
        }
    }

    pub fn with_penalty(mut self, penalty: f64) -> Self {
        assert!(
            penalty.is_finite() && penalty >= 0.0,
            "free-surface penalty must be finite and nonnegative"
        );
        self.penalty = penalty;
        self
    }

    /// Unhindered terminal slip speed `V0` for surface degassing.
    /// Interior kernels keep the hindered `(1-a)^1.75` speed; the surface
    /// vent must not choke as `a -> 1`. (`slip_speed(0)` equals the
    /// correlation value; the um-referenced rescaling is unity at `a = 0`.)
    fn vent_speed(&self) -> f64 {
        self.config.slip_speed(0.0)
    }

    /// Surface degassing flux `F_vent(a) = clamp(a)*V0` (linear, monotone).
    fn vent_flux(&self, alpha: f64) -> f64 {
        alpha.clamp(0.0, 1.0) * self.vent_speed()
    }

    /// `dF_vent/da` (`V0` interior, `0` outside where the clamp freezes).
    fn vent_flux_derivative(&self, alpha: f64) -> f64 {
        if alpha <= 0.0 || alpha >= 1.0 {
            0.0
        } else {
            self.vent_speed()
        }
    }

    /// Liquid normal velocity `u_l.n` from the closure inversion.
    /// Pure-gas / out-of-range states fall back to the mixture normal.
    fn liquid_normal(&self, mixture_normal: f64, alpha: f64, rise_normal: f64) -> f64 {
        if alpha <= 0.0 || alpha >= 1.0 {
            return mixture_normal;
        }
        let liquid_fraction = 1.0 - alpha;
        if liquid_fraction <= MIN_LIQUID_FRACTION {
            return mixture_normal;
        }
        let rho_m = self.config.mixture_density(alpha);
        let c0 = self.config.distribution_parameter();
        let speed = self.config.slip_speed(alpha);
        let numerator = mixture_normal * (rho_m - alpha * self.config.rho_g * c0)
            - alpha * self.config.rho_g * speed * rise_normal;
        numerator / (liquid_fraction * self.config.rho_l)
    }

    /// `d(u_l.n)/d(u_m.n)` (= `A`) and `d(u_l.n)/d(alpha)` at the state.
    fn liquid_normal_derivatives(
        &self,
        mixture_normal: f64,
        alpha: f64,
        rise_normal: f64,
    ) -> (f64, f64) {
        if alpha <= 0.0 || alpha >= 1.0 {
            return (1.0, 0.0);
        }
        let liquid_fraction = 1.0 - alpha;
        if liquid_fraction <= MIN_LIQUID_FRACTION {
            return (1.0, 0.0);
        }
        let rho_m = self.config.mixture_density(alpha);
        let drho = self.config.mixture_density_derivative();
        let c0 = self.config.distribution_parameter();
        let denom = liquid_fraction * self.config.rho_l;
        let a_factor = (rho_m - alpha * self.config.rho_g * c0) / denom;
        // N = un*(rho_m - a*rho_g*C0) - a*rho_g*V*en; d(aV)/da = dF/da,
        // both um-referenced (slip_speed / drift_flux_derivative).
        let speed = self.config.slip_speed(alpha);
        let dflux = self.config.drift_flux_derivative(alpha);
        let numerator = mixture_normal * (rho_m - alpha * self.config.rho_g * c0)
            - alpha * self.config.rho_g * speed * rise_normal;
        let d_numerator = mixture_normal * (drho - self.config.rho_g * c0)
            - self.config.rho_g * rise_normal * dflux;
        let d_alpha = (d_numerator * denom + numerator * self.config.rho_l) / (denom * denom);
        (a_factor, d_alpha)
    }

    /// Lane-packed surface degassing fluxes matching [`vent_flux`](Self::vent_flux).
    ///
    /// # Arguments
    /// * `alphas` - per-lane void fractions.
    ///
    /// # Returns
    /// Per-lane `F_vent`, bit-identical to the scalar path lane-by-lane.
    #[inline]
    fn vent_flux_lanes(&self, alphas: &Lanes) -> Lanes {
        let mut out = [0.0; LANES];
        for l in 0..LANES {
            out[l] = self.vent_flux(alphas[l]);
        }
        out
    }

    /// Lane-packed degassing-flux derivatives matching [`vent_flux_derivative`](Self::vent_flux_derivative).
    ///
    /// # Arguments
    /// * `alphas` - per-lane void fractions.
    ///
    /// # Returns
    /// Per-lane `dF_vent/da`, bit-identical to the scalar path lane-by-lane.
    #[inline]
    fn vent_flux_derivative_lanes(&self, alphas: &Lanes) -> Lanes {
        let mut out = [0.0; LANES];
        for l in 0..LANES {
            out[l] = self.vent_flux_derivative(alphas[l]);
        }
        out
    }

    /// Lane-packed liquid normal velocities matching [`liquid_normal`](Self::liquid_normal).
    ///
    /// # Arguments
    /// * `mixture_normals` - per-lane mixture normal velocities.
    /// * `alphas` - per-lane void fractions.
    /// * `rise_normals` - per-lane rise-vector normal components.
    ///
    /// # Returns
    /// Per-lane `u_l.n`, bit-identical to the scalar path lane-by-lane.
    #[inline]
    fn liquid_normal_lanes(
        &self,
        mixture_normals: &Lanes,
        alphas: &Lanes,
        rise_normals: &Lanes,
    ) -> Lanes {
        let mut out = [0.0; LANES];
        for l in 0..LANES {
            out[l] = self.liquid_normal(mixture_normals[l], alphas[l], rise_normals[l]);
        }
        out
    }

    /// Lane-packed liquid-normal derivatives matching [`liquid_normal_derivatives`](Self::liquid_normal_derivatives).
    ///
    /// # Arguments
    /// * `mixture_normals` - per-lane mixture normal velocities.
    /// * `alphas` - per-lane void fractions.
    /// * `rise_normals` - per-lane rise-vector normal components.
    ///
    /// # Returns
    /// Per-lane `(d(u_l.n)/d(u_m.n), d(u_l.n)/d(alpha))` pairs, bit-identical
    /// to the scalar path lane-by-lane.
    #[inline]
    fn liquid_normal_derivatives_lanes(
        &self,
        mixture_normals: &Lanes,
        alphas: &Lanes,
        rise_normals: &Lanes,
    ) -> (Lanes, Lanes) {
        let mut a_factor = [0.0; LANES];
        let mut d_alpha = [0.0; LANES];
        for l in 0..LANES {
            let (a, d) =
                self.liquid_normal_derivatives(mixture_normals[l], alphas[l], rise_normals[l]);
            a_factor[l] = a;
            d_alpha[l] = d;
        }
        (a_factor, d_alpha)
    }
}

impl StateTensorBoundaryIntegrator<2> for TensorDriftFreeSurface2D {
    fn nfields(&self) -> usize {
        4
    }

    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names()
    }

    /// Lane-packed free-surface trace residual for all lanes.
    ///
    /// Each lane evaluates exactly the scalar expression with its own facet
    /// context and state lane, preserving every branch (`un > 0` vent
    /// clipping, pure-gas fallbacks, `max(e.n, 0)` rise clipping).
    ///
    /// # Arguments
    /// * `ctxs` - one tensor facet context per lane, length [`LANES`].
    /// * `state` - lane-packed facet solution.
    /// * `equation` - output equation index shared by all lanes.
    /// * `q` - facet quadrature-point index shared by all lanes.
    /// * `out` - lane trace-flux slots. Overwritten.
    #[inline]
    fn tensor_residual(
        &self,
        ctxs: &[TensorFacetCtx<'_>],
        state: &LaneState<'_>,
        equation: usize,
        q: usize,
        out: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        let u0 = state.value(0, q);
        let u1 = state.value(1, q);
        let p = state.value(2, q);
        let a = state.value(ALPHA_2D, q);
        let e = self.config.rise_direction();
        let c0 = self.config.distribution_parameter();
        if equation == ALPHA_2D {
            let vent = self.vent_flux_lanes(a);
            for l in 0..LANES {
                let un = ctxs[l].normal[0] * u0[l] + ctxs[l].normal[1] * u1[l];
                let advective = if un > 0.0 { 0.5 * c0 * un * a[l] } else { 0.0 };
                let outward = (ctxs[l].normal[0] * e[0] + ctxs[l].normal[1] * e[1]).max(0.0);
                out[l] = advective + vent[l] * outward;
            }
            return;
        }
        if equation < 2 {
            let mut mixture = [0.0; LANES];
            let mut rise = [0.0; LANES];
            for l in 0..LANES {
                mixture[l] = ctxs[l].normal[0] * u0[l] + ctxs[l].normal[1] * u1[l];
                rise[l] = ctxs[l].normal[0] * e[0] + ctxs[l].normal[1] * e[1];
            }
            let uln = self.liquid_normal_lanes(&mixture, a, &rise);
            for l in 0..LANES {
                let base = directional_residual(
                    ctxs[l].normal,
                    [u0[l], u1[l]],
                    p[l],
                    self.config.rho_l,
                    true,
                    equation,
                );
                out[l] = base + self.penalty * uln[l] * ctxs[l].normal[equation];
            }
            return;
        }
        for l in 0..LANES {
            out[l] = directional_residual(
                ctxs[l].normal,
                [u0[l], u1[l]],
                p[l],
                self.config.rho_l,
                true,
                equation,
            );
        }
    }

    /// Lane-packed free-surface trace Jacobian action for all lanes.
    ///
    /// The advective void derivative uses the outflow-side branch
    /// (`un >= 0.0`), matching the scalar path.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor facet context per lane, length [`LANES`].
    /// * `state` - lane-packed linearization point.
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - output equation index shared by all lanes.
    /// * `q` - facet quadrature-point index shared by all lanes.
    /// * `out` - lane linearized trace-flux slots. Overwritten.
    #[inline]
    fn tensor_jacobian_action(
        &self,
        ctxs: &[TensorFacetCtx<'_>],
        state: &LaneState<'_>,
        direction: &LaneState<'_>,
        equation: usize,
        q: usize,
        out: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        let u0 = state.value(0, q);
        let u1 = state.value(1, q);
        let p = state.value(2, q);
        let a = state.value(ALPHA_2D, q);
        let du0 = direction.value(0, q);
        let du1 = direction.value(1, q);
        let dp = direction.value(2, q);
        let da = direction.value(ALPHA_2D, q);
        let e = self.config.rise_direction();
        let c0 = self.config.distribution_parameter();
        if equation == ALPHA_2D {
            let dvent = self.vent_flux_derivative_lanes(a);
            for l in 0..LANES {
                let un = ctxs[l].normal[0] * u0[l] + ctxs[l].normal[1] * u1[l];
                let dun = ctxs[l].normal[0] * du0[l] + ctxs[l].normal[1] * du1[l];
                let dadvective = if un >= 0.0 {
                    0.5 * c0 * (dun * a[l] + un * da[l])
                } else {
                    0.0
                };
                let outward = (ctxs[l].normal[0] * e[0] + ctxs[l].normal[1] * e[1]).max(0.0);
                out[l] = dadvective + dvent[l] * outward * da[l];
            }
            return;
        }
        if equation < 2 {
            let mut mixture = [0.0; LANES];
            let mut rise = [0.0; LANES];
            let mut dmixture = [0.0; LANES];
            for l in 0..LANES {
                mixture[l] = ctxs[l].normal[0] * u0[l] + ctxs[l].normal[1] * u1[l];
                rise[l] = ctxs[l].normal[0] * e[0] + ctxs[l].normal[1] * e[1];
                dmixture[l] = ctxs[l].normal[0] * du0[l] + ctxs[l].normal[1] * du1[l];
            }
            let (a_factor, d_alpha) = self.liquid_normal_derivatives_lanes(&mixture, a, &rise);
            for l in 0..LANES {
                let base = directional_jacobian_action(
                    ctxs[l].normal,
                    [u0[l], u1[l]],
                    [du0[l], du1[l]],
                    p[l],
                    dp[l],
                    self.config.rho_l,
                    true,
                    equation,
                );
                out[l] = base
                    + self.penalty
                        * (a_factor[l] * dmixture[l] + d_alpha[l] * da[l])
                        * ctxs[l].normal[equation];
            }
            return;
        }
        for l in 0..LANES {
            out[l] = directional_jacobian_action(
                ctxs[l].normal,
                [u0[l], u1[l]],
                [du0[l], du1[l]],
                p[l],
                dp[l],
                self.config.rho_l,
                true,
                equation,
            );
        }
    }
}
