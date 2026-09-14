//! Tensor directional do-nothing outflow for drift-flux EDAC (`[u, v, p, alpha]`).
//!
//! Mathematics: borrows the base EDAC outflow — `-p n / rho_l` traction plus
//! the `max(-u.n, 0)` backflow penalty — with the reference liquid density so
//! the `alpha -> 0` limit matches the single-phase kernel exactly; with
//! `split_flux` it also supplies the conservative-half fluxes for split
//! momentum/pressure advection. The void equation gets the passive-scalar
//! split outflow `1/2 (u.n) alpha` (zero when `split_flux` is off).
use crate::common::{CellState, TensorFacetCtx};
use crate::kernels::common::StateTensorBoundaryIntegrator;
use crate::kernels::edac::weak::directional_do_nothing::{
    directional_jacobian_action, directional_residual,
};
use crate::kernels::edac_multiphase_drift_flux::config::{drift_field_names, ALPHA_2D};

/// Tensor directional do-nothing outflow for drift-flux EDAC.
pub struct TensorDriftDirectionalDoNothing2D {
    pub rho_l: f64,
    pub split_flux: bool,
}

impl TensorDriftDirectionalDoNothing2D {
    pub fn new(rho_l: f64) -> Self {
        assert!(
            rho_l.is_finite() && rho_l > 0.0,
            "reference density must be finite and positive"
        );
        Self {
            rho_l,
            split_flux: false,
        }
    }

    pub fn with_split_flux(mut self) -> Self {
        self.split_flux = true;
        self
    }
}

impl StateTensorBoundaryIntegrator<2> for TensorDriftDirectionalDoNothing2D {
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
            if !self.split_flux {
                return 0.0;
            }
            let normal_velocity =
                ctx.normal[0] * state.value(0, q) + ctx.normal[1] * state.value(1, q);
            return 0.5 * normal_velocity * state.value(ALPHA_2D, q);
        }
        let velocity = [state.value(0, q), state.value(1, q)];
        directional_residual(
            ctx.normal,
            velocity,
            state.value(2, q),
            self.rho_l,
            self.split_flux,
            equation,
        )
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
            if !self.split_flux {
                return 0.0;
            }
            let normal_velocity =
                ctx.normal[0] * state.value(0, q) + ctx.normal[1] * state.value(1, q);
            let direction_normal_velocity =
                ctx.normal[0] * direction.value(0, q) + ctx.normal[1] * direction.value(1, q);
            return 0.5
                * (direction_normal_velocity * state.value(ALPHA_2D, q)
                    + normal_velocity * direction.value(ALPHA_2D, q));
        }
        let velocity = [state.value(0, q), state.value(1, q)];
        let direction_velocity = [direction.value(0, q), direction.value(1, q)];
        directional_jacobian_action(
            ctx.normal,
            velocity,
            direction_velocity,
            state.value(2, q),
            direction.value(2, q),
            self.rho_l,
            self.split_flux,
            equation,
        )
    }
}
