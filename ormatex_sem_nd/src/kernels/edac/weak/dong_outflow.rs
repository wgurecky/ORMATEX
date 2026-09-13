//! Dong OBC-C outflow for EDAC (`[u, v, p]`).
//!
//! Mathematics: `u_n u / 2` consistency flux for the weak conservative half
//! of split momentum advection, Dong's smoothed OBC-C traction
//! (`tanh` switch on `u.n / (delta * U0)`), and the matching `u_n p / 2`
//! flux for split pressure advection. With `split_flux`, pairs with
//! split-form volume kernels; otherwise with conservative-form volumes.
//! Needs the SplitBoundaryFlux default on remaining facets.
use crate::common::{CellState, FacetCtx};
use crate::kernels::common::StateBoundaryIntegrator;
use crate::kernels::edac::config::fluid_field_names;
/// Dong OBC-C adapted to the monolithic EDAC momentum residual.
///
/// The boundary form includes the `u_n u / 2` consistency flux for the weak
/// conservative half of the split momentum term, Dong's OBC-C traction, and
/// the matching `u_n p / 2` flux for split pressure advection.
pub struct KernelEdacDongOutflow2D {
    pub rho: f64,
    pub delta: f64,
    pub velocity_scale: f64,
    pub split_flux: bool,
}

impl KernelEdacDongOutflow2D {
    pub fn new(rho: f64, delta: f64, velocity_scale: f64) -> Self {
        assert!(
            rho.is_finite() && rho > 0.0,
            "density must be finite and positive"
        );
        assert!(
            delta.is_finite() && delta > 0.0,
            "Dong smoothing parameter must be finite and positive"
        );
        assert!(
            velocity_scale.is_finite() && velocity_scale > 0.0,
            "Dong velocity scale must be finite and positive"
        );
        Self {
            rho,
            delta,
            velocity_scale,
            split_flux: false,
        }
    }

    pub fn with_split_flux(mut self) -> Self {
        self.split_flux = true;
        self
    }

    fn check(ctx: &FacetCtx, state: &CellState) {
        assert_eq!(ctx.gdim, 2, "KernelEdacDongOutflow2D requires gdim == 2");
        assert_eq!(ctx.ncomp, 1, "fluid fields must be scalar fields");
        assert_eq!(state.nfields, 3, "fluid state must contain [u, v, p]");
    }

    fn velocity(state: &CellState, q: usize) -> [f64; 2] {
        [state.value(0, q), state.value(1, q)]
    }

    pub(crate) fn switch(&self, normal_velocity: f64) -> f64 {
        0.5 * (1.0 - (normal_velocity / (self.delta * self.velocity_scale)).tanh())
    }

    pub(crate) fn switch_derivative(&self, normal_velocity: f64) -> f64 {
        let x = normal_velocity / (self.delta * self.velocity_scale);
        -0.5 * (1.0 - x.tanh().powi(2)) / (self.delta * self.velocity_scale)
    }

    pub(crate) fn dong_flux(&self, normal: &[f64], velocity: [f64; 2]) -> [f64; 2] {
        let normal_velocity = normal[0] * velocity[0] + normal[1] * velocity[1];
        let speed_squared = velocity[0] * velocity[0] + velocity[1] * velocity[1];
        let factor = 0.5 * self.switch(normal_velocity);
        [
            factor * (speed_squared * normal[0] + normal_velocity * velocity[0]),
            factor * (speed_squared * normal[1] + normal_velocity * velocity[1]),
        ]
    }

    pub(crate) fn dong_flux_derivative(
        &self,
        normal: &[f64],
        velocity: [f64; 2],
        component: usize,
        unknown_velocity: usize,
    ) -> f64 {
        let normal_velocity = normal[0] * velocity[0] + normal[1] * velocity[1];
        let speed_squared = velocity[0] * velocity[0] + velocity[1] * velocity[1];
        let switch = self.switch(normal_velocity);
        let switch_derivative = self.switch_derivative(normal_velocity);
        let kronecker = (component == unknown_velocity) as usize as f64;
        let derivative_bracket = 2.0 * velocity[unknown_velocity] * normal[component]
            + normal[unknown_velocity] * velocity[component]
            + normal_velocity * kronecker;
        0.5 * switch * derivative_bracket
            + 0.5
                * switch_derivative
                * normal[unknown_velocity]
                * (speed_squared * normal[component] + normal_velocity * velocity[component])
    }
}

impl StateBoundaryIntegrator for KernelEdacDongOutflow2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names()
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
        let normal_velocity = ctx.normal[0] * velocity[0] + ctx.normal[1] * velocity[1];
        match equation {
            0 | 1 => {
                let dong_flux = self.dong_flux(ctx.normal, velocity);
                ((if self.split_flux {
                    0.5 * normal_velocity * velocity[equation]
                } else {
                    0.0
                }) - state.value(2, q) * ctx.normal[equation] / self.rho
                    - dong_flux[equation])
                    * test
            }
            2 => {
                (if self.split_flux {
                    0.5 * normal_velocity * state.value(2, q)
                } else {
                    0.0
                }) * test
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
        Self::check(ctx, state);
        assert!(unknown < 3, "fluid unknown field out of range");
        let test = ctx.test(test_i, 0).v(q);
        let trial = ctx.trial(trial_i, 0).v(q);
        let velocity = Self::velocity(state, q);
        let normal_velocity = ctx.normal[0] * velocity[0] + ctx.normal[1] * velocity[1];
        match equation {
            0 | 1 => {
                let derivative = if unknown < 2 {
                    (if self.split_flux {
                        0.5 * (ctx.normal[unknown] * velocity[equation]
                            + normal_velocity * (equation == unknown) as usize as f64)
                    } else {
                        0.0
                    }) - self.dong_flux_derivative(ctx.normal, velocity, equation, unknown)
                } else {
                    -ctx.normal[equation] / self.rho
                };
                derivative * trial * test
            }
            2 => {
                let pressure = state.value(2, q);
                let derivative = if unknown < 2 {
                    if self.split_flux {
                        0.5 * ctx.normal[unknown] * pressure
                    } else {
                        0.0
                    }
                } else {
                    if self.split_flux {
                        0.5 * normal_velocity * trial
                    } else {
                        0.0
                    }
                };
                if unknown < 2 {
                    derivative * trial * test
                } else {
                    derivative * test
                }
            }
            _ => unreachable!(),
        }
    }
}
