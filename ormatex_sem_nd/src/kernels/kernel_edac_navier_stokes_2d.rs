use crate::common::{CellState, LocalCtx};

use super::kernel_common::ResidualKernel;
use super::kernel_smagorinsky_lilly_2d::SmagorinskyLilly2D;

/// Three-field entropically damped artifical compressibility (EDAC)
/// Navier-Stokes kernel with fields `[u, v, p]`.
///
/// The momentum convection and pressure gradient use their strong first-order
/// forms; viscous and EDAC pressure diffusion terms use the weak gradient form.
/// This keeps symmetry and pressure outlet conditions usable with the current
/// natural-boundary interface.
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

    fn check(ctx: &LocalCtx, state: &CellState) {
        assert_eq!(ctx.gdim, 2, "KernelEdacNavierStokes2D requires gdim == 2");
        assert_eq!(ctx.ncomp, 1, "fluid fields must be scalar fields");
        assert_eq!(state.nfields, 3, "fluid state must contain [u, v, p]");
    }

    fn velocity(state: &CellState, q: usize) -> [f64; 2] {
        [state.value(0, q), state.value(1, q)]
    }

    fn pressure_diffusivity(&self, ctx: &LocalCtx) -> f64 {
        self.pressure_diffusion_factor * self.c0 * self.smagorinsky.filter_width(ctx)
    }

    fn strain_component(state: &CellState, q: usize, i: usize, j: usize) -> f64 {
        0.5 * (state.grad(i, q, j) + state.grad(j, q, i))
    }

    fn stress(&self, ctx: &LocalCtx, state: &CellState, q: usize, i: usize, j: usize) -> f64 {
        let viscosity = self.nu + self.smagorinsky.eddy_viscosity(ctx, state, q);
        2.0 * viscosity * Self::strain_component(state, q, i, j)
    }

    fn stress_jacobian(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        q: usize,
        i: usize,
        j: usize,
        unknown: usize,
        trial: &crate::common::ShapeFn<'_>,
    ) -> f64 {
        if unknown >= 2 {
            return 0.0;
        }
        let viscosity = self.nu + self.smagorinsky.eddy_viscosity(ctx, state, q);
        let dviscosity = self
            .smagorinsky
            .eddy_viscosity_gradient_derivative(ctx, state, q, unknown, 0)
            * trial.grad(q, 0)
            + self
                .smagorinsky
                .eddy_viscosity_gradient_derivative(ctx, state, q, unknown, 1)
                * trial.grad(q, 1);
        let dstrain = 0.5
            * ((i == unknown) as usize as f64 * trial.grad(q, j)
                + (j == unknown) as usize as f64 * trial.grad(q, i));
        2.0 * (viscosity * dstrain + dviscosity * Self::strain_component(state, q, i, j))
    }
}

impl ResidualKernel for KernelEdacNavierStokes2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        Some(["u", "v", "p"].into_iter().map(str::to_owned).collect())
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        Self::check(ctx, state);
        let test = ctx.test(test_i, 0);
        let velocity = Self::velocity(state, q);
        match equation {
            0 | 1 => {
                let i = equation;
                let convection = velocity
                    .iter()
                    .enumerate()
                    .map(|(j, &u_j)| u_j * state.grad(i, q, j))
                    .sum::<f64>();
                let viscous = (0..2)
                    .map(|j| self.stress(ctx, state, q, i, j) * test.grad(q, j))
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
                let pressure_diffusion = self.pressure_diffusivity(ctx)
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
        Self::check(ctx, state);
        assert!(unknown < 3, "fluid unknown field out of range");
        let test = ctx.test(test_i, 0);
        let trial = ctx.trial(trial_i, 0);
        let velocity = Self::velocity(state, q);
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
                        self.stress_jacobian(ctx, state, q, i, j, unknown, &trial) * test.grad(q, j)
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
                    self.pressure_diffusivity(ctx)
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
