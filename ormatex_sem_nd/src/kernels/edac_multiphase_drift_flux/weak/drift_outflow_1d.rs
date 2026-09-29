//! Drift-flux outflow endpoint kernel (1D, weak form).
//!
//! Mathematics: right-endpoint (`n = +1`) pairing fluxes for the split-form
//! 1D drift volumes, following the `-(volume f1 . n)` rule used by the 2D
//! split boundaries: `+u^2/2` (momentum), `+u p/2` (pressure), and
//! `+C0 u a/2 + F_um(a) sin(theta)` (void, with the um-referenced hindered
//! drift flux `F_um`, matching the volume drift kernel exactly). Attach only
//! at outflow facets; no backflow stabilization is included.
//! Precedent:
//! [`KernelAdvectionOutflow1D`](crate::kernels::KernelAdvectionOutflow1D)
//! documents that the 1D tensor operator reuses weak endpoint terms (the 1D
//! tensor path has no facet-kernel support).
use crate::common::{CellState, FacetCtx};

use crate::kernels::common::StateBoundaryIntegrator;
use crate::kernels::edac_multiphase_drift_flux::config_1d::{
    drift_field_names_1d, DriftFlux1DConfig,
};

/// 1D drift-flux outflow endpoint (owns all three equations at the facet).
pub struct DriftOutflow1D {
    pub config: DriftFlux1DConfig,
    /// Pipe angle `theta` at the outlet facet (radians from horizontal).
    pub theta: f64,
}

impl DriftOutflow1D {
    pub fn new(config: DriftFlux1DConfig, theta: f64) -> Self {
        assert!(theta.is_finite(), "outlet pipe angle must be finite");
        Self { config, theta }
    }

    fn check(&self, ctx: &FacetCtx, state: &CellState) {
        assert_eq!(ctx.gdim, 1, "DriftOutflow1D requires gdim == 1");
        assert_eq!(ctx.ncomp, 1, "drift fields must be scalar fields");
        assert_eq!(state.nfields, 3, "drift state must contain [u, p, alpha]");
    }
}

impl StateBoundaryIntegrator for DriftOutflow1D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names_1d()
    }

    fn residual_integrand(
        &self,
        ctx: &FacetCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        self.check(ctx, state);
        let test = ctx.test(test_i, 0).v(q);
        let u = state.value(0, q);
        match equation {
            0 => 0.5 * u * u * test,
            1 => 0.5 * u * state.value(1, q) * test,
            2 => {
                let a = state.value(2, q);
                (0.5 * self.config.distribution_parameter() * u * a
                    + self.config.axial_drift_flux(a, self.theta))
                    * test
            }
            _ => unreachable!(),
        }
    }

    fn jacobian_integrand(
        &self,
        ctx: &FacetCtx,
        state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        self.check(ctx, state);
        assert!(unknown < 3, "drift unknown field out of range");
        let test = ctx.test(test_i, 0).v(q);
        let trial = ctx.trial(trial_i, 0).v(q);
        let u = state.value(0, q);
        match (equation, unknown) {
            (0, 0) => u * trial * test,
            (1, 0) => 0.5 * trial * state.value(1, q) * test,
            (1, 1) => 0.5 * u * trial * test,
            (2, 0) => 0.5 * self.config.distribution_parameter() * trial * state.value(2, q) * test,
            (2, 2) => {
                (0.5 * self.config.distribution_parameter() * u
                    + self
                        .config
                        .axial_drift_flux_derivative(state.value(2, q), self.theta))
                    * trial
                    * test
            }
            _ => 0.0,
        }
    }
}
