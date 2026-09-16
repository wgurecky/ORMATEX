//! Tensor free-surface boundary for drift-flux EDAC (`[u, v, p, alpha]`).
//!
//! Mathematics: the mixture momentum/pressure equations reuse the base EDAC
//! directional-do-nothing outflow (`-p n / rho_l` traction plus the
//! `max(-u.n, 0)` backflow penalty, with split-flux halves), so the pressure
//! datum is set weakly and no Dirichlet pressure pin is needed. The mixture
//! velocity itself is left free: the liquid no-penetration condition
//! (`u_l . n = 0`) cannot be a strong Dirichlet condition because `u_l` is a
//! derived closure quantity, and near the surface the liquid fraction is
//! close to one (`u_l ~= u_m`), so the slip-like zero-flux treatment is the
//! consistent weak form. The void equation vents gas with a no-reentry flux
//! `1/2 C0 max(u.n, 0) alpha + F(alpha) max(e.n, 0)`: the split-consistent
//! advective half (clipped so void never re-enters) plus the full hindered
//! drift flux `F` of Ishii-Zuber (which points out wherever the rise vector
//! `e` has an outward component). At `u.n = 0` the Jacobian uses the
//! outflow-side derivative, matching the directional-do-nothing convention;
//! it carries the exact `dF/da` sign.
use crate::common::{CellState, TensorFacetCtx};
use crate::kernels::common::StateTensorBoundaryIntegrator;
use crate::kernels::edac::weak::directional_do_nothing::{
    directional_jacobian_action, directional_residual,
};
use crate::kernels::edac_multiphase_drift_flux::config::{
    drift_field_names, DriftFlux2DConfig, ALPHA_2D,
};

/// Tensor free-surface boundary for drift-flux EDAC (owns all four equations).
pub struct TensorDriftFreeSurface2D {
    pub config: DriftFlux2DConfig,
}

impl TensorDriftFreeSurface2D {
    pub fn new(config: DriftFlux2DConfig) -> Self {
        Self { config }
    }

    fn normal_velocity(ctx: &TensorFacetCtx<'_>, state: &CellState<'_>, q: usize) -> f64 {
        ctx.normal[0] * state.value(0, q) + ctx.normal[1] * state.value(1, q)
    }

    /// Outward rise component `max(e . n, 0)` (geometry-fixed, state-independent).
    fn outward_rise(&self, ctx: &TensorFacetCtx<'_>) -> f64 {
        let e = self.config.rise_direction();
        (ctx.normal[0] * e[0] + ctx.normal[1] * e[1]).max(0.0)
    }
}

impl StateTensorBoundaryIntegrator<2> for TensorDriftFreeSurface2D {
    fn nfields(&self) -> usize {
        4
    }

    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names()
    }

    fn tensor_residual(
        &self,
        ctx: &TensorFacetCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> f64 {
        if equation == ALPHA_2D {
            let un = Self::normal_velocity(ctx, state, q);
            let advective = if un > 0.0 {
                0.5 * self.config.distribution_parameter() * un * state.value(ALPHA_2D, q)
            } else {
                0.0
            };
            advective + self.config.drift_flux(state.value(ALPHA_2D, q)) * self.outward_rise(ctx)
        } else {
            directional_residual(
                ctx.normal,
                [state.value(0, q), state.value(1, q)],
                state.value(2, q),
                self.config.rho_l,
                true,
                equation,
            )
        }
    }

    fn tensor_jacobian_action(
        &self,
        ctx: &TensorFacetCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> f64 {
        if equation == ALPHA_2D {
            let un = Self::normal_velocity(ctx, state, q);
            let dun = ctx.normal[0] * direction.value(0, q) + ctx.normal[1] * direction.value(1, q);
            let a = state.value(ALPHA_2D, q);
            let da = direction.value(ALPHA_2D, q);
            // Outflow-side derivative at u.n = 0 (see module docs).
            let dadvective = if un >= 0.0 {
                0.5 * self.config.distribution_parameter() * (dun * a + un * da)
            } else {
                0.0
            };
            dadvective + self.config.drift_flux_derivative(a) * self.outward_rise(ctx) * da
        } else {
            directional_jacobian_action(
                ctx.normal,
                [state.value(0, q), state.value(1, q)],
                [direction.value(0, q), direction.value(1, q)],
                state.value(2, q),
                direction.value(2, q),
                self.config.rho_l,
                true,
                equation,
            )
        }
    }
}
