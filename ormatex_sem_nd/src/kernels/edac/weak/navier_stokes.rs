//! Fused three-field EDAC Navier-Stokes kernel (`[u, v, p]`, conservative form).
//!
//! Mathematics: momentum uses `(u . grad) u + grad(p)/rho - div(tau)` with the
//! strong convection/pressure forms and the weak viscous form; pressure uses
//! `rho c0^2 div(u) + (u . grad) p - div(k grad(p))`. Tensor counterpart:
//! [`TensorKernelEdacNavierStokes2D`](crate::kernels::edac::tensor::navier_stokes::TensorKernelEdacNavierStokes2D).
//! For the split advection halves see the `*_split` kernels plus a split-flux
//! boundary (Dong or directional-do-nothing with split flux).
use crate::common::{CellState, LocalCtx};
use crate::kernels::common::ResidualKernel;
use crate::kernels::edac::config::{EdacNavierStokes2DConfig, check_weak_cell, fluid_field_names, velocity};
use crate::kernels::edac::smagorinsky_lilly::SmagorinskyLilly2D;

/// Fused conservative EDAC Navier-Stokes kernel: strong convection/pressure,
/// weak viscous and pressure-diffusion forms (owns all equations).
pub struct KernelEdacNavierStokes2D {
    pub rho: f64,
    pub nu: f64,
    pub c0: f64,
    pub pressure_diffusion_factor: f64,
    pub smagorinsky: SmagorinskyLilly2D,
}


impl KernelEdacNavierStokes2D {
    pub fn new(rho: f64, nu: f64, c0: f64, cs: f64) -> Self {
        assert!(
            rho.is_finite() && rho > 0.0,
            "density must be finite and positive"
        );
        assert!(
            nu.is_finite() && nu >= 0.0,
            "kinematic viscosity must be finite and nonnegative"
        );
        assert!(
            c0.is_finite() && c0 > 0.0,
            "artificial sound speed must be finite and positive"
        );
        Self {
            rho,
            nu,
            c0,
            pressure_diffusion_factor: 0.1,
            smagorinsky: SmagorinskyLilly2D::new(cs),
        }
    }

    pub fn with_pressure_diffusion_factor(mut self, factor: f64) -> Self {
        assert!(
            factor.is_finite() && factor >= 0.0,
            "pressure diffusion factor must be finite and nonnegative"
        );
        self.pressure_diffusion_factor = factor;
        self
    }

    fn config(&self) -> EdacNavierStokes2DConfig {
        EdacNavierStokes2DConfig {
            rho: self.rho,
            nu: self.nu,
            c0: self.c0,
            pressure_diffusion_factor: self.pressure_diffusion_factor,
            smagorinsky: self.smagorinsky,
        }
    }
}

impl ResidualKernel for KernelEdacNavierStokes2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names()
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        check_weak_cell(ctx, state);
        let test = ctx.test(test_i, 0);
        let velocity = velocity(state, q);
        match equation {
            0 | 1 => {
                let i = equation;
                let convection = velocity
                    .iter()
                    .enumerate()
                    .map(|(j, &u_j)| u_j * state.grad(i, q, j))
                    .sum::<f64>();
                let viscous = (0..2)
                    .map(|j| self.config().stress(ctx, state, q, i, j) * test.grad(q, j))
                    .sum::<f64>();
                convection * test.v(q) + state.grad(2, q, i) * test.v(q) / self.rho + viscous
            }
            2 => {
                let divergence = state.grad(0, q, 0) + state.grad(1, q, 1);
                let pressure_advection = velocity
                    .iter()
                    .enumerate()
                    .map(|(j, &u_j)| u_j * state.grad(2, q, j))
                    .sum::<f64>();
                let pressure_diffusion = self.config().pressure_diffusivity(ctx)
                    * (0..2)
                        .map(|j| state.grad(2, q, j) * test.grad(q, j))
                        .sum::<f64>();
                (self.rho * self.c0 * self.c0 * divergence + pressure_advection) * test.v(q)
                    + pressure_diffusion
            }
            _ => unreachable!(),
        }
    }

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        check_weak_cell(ctx, state);
        assert!(unknown < 3, "fluid unknown field out of range");
        let test = ctx.test(test_i, 0);
        let trial = ctx.trial(trial_i, 0);
        let velocity = velocity(state, q);
        match equation {
            0 | 1 => {
                let i = equation;
                let convection = if unknown < 2 {
                    trial.v(q) * state.grad(i, q, unknown)
                        + if unknown == i {
                            velocity
                                .iter()
                                .enumerate()
                                .map(|(j, &u_j)| u_j * trial.grad(q, j))
                                .sum::<f64>()
                        } else {
                            0.0
                        }
                } else {
                    0.0
                };
                let pressure = (unknown == 2) as usize as f64 * trial.grad(q, i);
                let viscous = (0..2)
                    .map(|j| {
                        self.config().stress_jacobian(ctx, state, q, i, j, unknown, &trial) * test.grad(q, j)
                    })
                    .sum::<f64>();
                convection * test.v(q) + pressure * test.v(q) / self.rho + viscous
            }
            2 => {
                let divergence = if unknown < 2 {
                    (unknown == 0) as usize as f64 * trial.grad(q, 0)
                        + (unknown == 1) as usize as f64 * trial.grad(q, 1)
                } else {
                    0.0
                };
                let pressure_advection = if unknown == 2 {
                    velocity
                        .iter()
                        .enumerate()
                        .map(|(j, &u_j)| u_j * trial.grad(q, j))
                        .sum::<f64>()
                } else if unknown < 2 {
                    trial.v(q) * state.grad(2, q, unknown)
                } else {
                    0.0
                };
                let pressure_diffusion = if unknown == 2 {
                    self.config().pressure_diffusivity(ctx)
                        * (0..2)
                            .map(|j| trial.grad(q, j) * test.grad(q, j))
                            .sum::<f64>()
                } else {
                    0.0
                };
                (self.rho * self.c0 * self.c0 * divergence + pressure_advection) * test.v(q)
                    + pressure_diffusion
            }
            _ => unreachable!(),
        }
    }
}
