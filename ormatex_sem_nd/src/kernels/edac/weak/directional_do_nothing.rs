//! Directional do-nothing outflow for EDAC (`[u, v, p]`).
//!
//! Mathematics: `-p n / rho` traction plus a backflow penalty active only for
//! incoming normal velocity (`max(-u.n, 0)`); at `u.n = 0` the Jacobian uses
//! the outflow-side derivative. With `split_flux`, also supplies the
//! conservative-half fluxes for split momentum/pressure advection, pairing
//! with split-form volume kernels (default SplitBoundaryFlux elsewhere).
use crate::common::{CellState, FacetCtx};
use crate::kernels::common::StateBoundaryIntegrator;
pub(crate) fn directional_field_names() -> Option<Vec<String>> {
    Some(["u", "v", "p"].into_iter().map(str::to_owned).collect())
}

pub(crate) fn directional_backflow(normal_velocity: f64) -> f64 {
    (-normal_velocity).max(0.0)
}

pub(crate) fn directional_backflow_derivative(normal_velocity: f64) -> f64 {
    if normal_velocity < 0.0 {
        -1.0
    } else {
        0.0
    }
}

pub(crate) fn directional_flux(normal: &[f64], velocity: [f64; 2]) -> [f64; 2] {
    let normal_velocity = normal[0] * velocity[0] + normal[1] * velocity[1];
    let factor = 0.5 * directional_backflow(normal_velocity);
    [factor * velocity[0], factor * velocity[1]]
}

pub(crate) fn directional_flux_derivative(
    normal: &[f64],
    velocity: [f64; 2],
    component: usize,
    unknown_velocity: usize,
) -> f64 {
    let normal_velocity = normal[0] * velocity[0] + normal[1] * velocity[1];
    let backflow = directional_backflow(normal_velocity);
    let dbackflow = directional_backflow_derivative(normal_velocity) * normal[unknown_velocity];
    0.5 * (dbackflow * velocity[component]
        + backflow * (component == unknown_velocity) as usize as f64)
}

pub(crate) fn directional_residual(
    normal: &[f64],
    velocity: [f64; 2],
    pressure: f64,
    rho: f64,
    split_flux: bool,
    equation: usize,
) -> f64 {
    let normal_velocity = normal[0] * velocity[0] + normal[1] * velocity[1];
    match equation {
        0 | 1 => {
            let split = if split_flux {
                0.5 * normal_velocity * velocity[equation]
            } else {
                0.0
            };
            split - pressure * normal[equation] / rho + directional_flux(normal, velocity)[equation]
        }
        2 => {
            if split_flux {
                0.5 * normal_velocity * pressure
            } else {
                0.0
            }
        }
        _ => unreachable!(),
    }
}

pub(crate) fn directional_jacobian_action(
    normal: &[f64],
    velocity: [f64; 2],
    direction_velocity: [f64; 2],
    pressure: f64,
    direction_pressure: f64,
    rho: f64,
    split_flux: bool,
    equation: usize,
) -> f64 {
    let normal_velocity = normal[0] * velocity[0] + normal[1] * velocity[1];
    let direction_normal_velocity =
        normal[0] * direction_velocity[0] + normal[1] * direction_velocity[1];
    match equation {
        0 | 1 => {
            let split = if split_flux {
                0.5 * (direction_normal_velocity * velocity[equation]
                    + normal_velocity * direction_velocity[equation])
            } else {
                0.0
            };
            split - direction_pressure * normal[equation] / rho
                + (0..2)
                    .map(|unknown| {
                        direction_velocity[unknown]
                            * directional_flux_derivative(normal, velocity, equation, unknown)
                    })
                    .sum::<f64>()
        }
        2 => {
            if split_flux {
                0.5 * (direction_normal_velocity * pressure + normal_velocity * direction_pressure)
            } else {
                0.0
            }
        }
        _ => unreachable!(),
    }
}

/// Directional do-nothing condition adapted to the monolithic EDAC momentum
/// residual.
///
/// The correction is active only for incoming normal velocity. In split mode,
/// this also supplies the conservative-half boundary fluxes for momentum and
/// pressure advection. At zero normal velocity, the Jacobian uses the
/// outflow-side derivative of the piecewise backflow correction.
pub struct KernelEdacDirectionalDoNothing2D {
    pub rho: f64,
    pub split_flux: bool,
}

impl KernelEdacDirectionalDoNothing2D {
    pub fn new(rho: f64) -> Self {
        assert!(
            rho.is_finite() && rho > 0.0,
            "density must be finite and positive"
        );
        Self {
            rho,
            split_flux: false,
        }
    }

    pub fn with_split_flux(mut self) -> Self {
        self.split_flux = true;
        self
    }

    fn check(ctx: &FacetCtx, state: &CellState) {
        assert_eq!(
            ctx.gdim, 2,
            "KernelEdacDirectionalDoNothing2D requires gdim == 2"
        );
        assert_eq!(ctx.ncomp, 1, "fluid fields must be scalar fields");
        assert_eq!(state.nfields, 3, "fluid state must contain [u, v, p]");
    }

    fn velocity(state: &CellState, q: usize) -> [f64; 2] {
        [state.value(0, q), state.value(1, q)]
    }
}

impl StateBoundaryIntegrator for KernelEdacDirectionalDoNothing2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        directional_field_names()
    }

    fn residual_integrand(
        &self,
        ctx: &FacetCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        Self::check(ctx, state);
        let test = ctx.test(test_i, 0).v(q);
        let velocity = Self::velocity(state, q);
        directional_residual(
            ctx.normal,
            velocity,
            state.value(2, q),
            self.rho,
            self.split_flux,
            equation,
        ) * test
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
        Self::check(ctx, state);
        assert!(unknown < 3, "fluid unknown field out of range");
        let test = ctx.test(test_i, 0).v(q);
        let trial = ctx.trial(trial_i, 0).v(q);
        let velocity = Self::velocity(state, q);
        let mut direction_velocity = [0.0; 2];
        if unknown < 2 {
            direction_velocity[unknown] = trial;
        }
        let direction_pressure = if unknown == 2 { trial } else { 0.0 };
        directional_jacobian_action(
            ctx.normal,
            velocity,
            direction_velocity,
            state.value(2, q),
            direction_pressure,
            self.rho,
            self.split_flux,
            equation,
        ) * test
    }
}
