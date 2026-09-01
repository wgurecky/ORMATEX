use crate::common::{CellState, LocalCtx, TensorCtx};

use super::kernel_common::{ResidualKernel, TensorResidualKernel};
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

/// Tensor-product EDAC Navier-Stokes volume kernel.
pub struct TensorKernelEdacNavierStokes2D {
    pub rho: f64,
    pub nu: f64,
    pub c0: f64,
    pub pressure_diffusion_factor: f64,
    pub smagorinsky: SmagorinskyLilly2D,
}

impl TensorKernelEdacNavierStokes2D {
    pub fn new(rho: f64, nu: f64, c0: f64, cs: f64) -> Self {
        let kernel = KernelEdacNavierStokes2D::new(rho, nu, c0, cs);
        Self {
            rho: kernel.rho,
            nu: kernel.nu,
            c0: kernel.c0,
            pressure_diffusion_factor: kernel.pressure_diffusion_factor,
            smagorinsky: kernel.smagorinsky,
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

    fn pressure_diffusivity(&self, ctx: &TensorCtx<'_>) -> f64 {
        self.pressure_diffusion_factor * self.c0 * self.smagorinsky.filter_width_tensor(ctx)
    }

    fn stress(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        q: usize,
        i: usize,
        j: usize,
    ) -> f64 {
        2.0 * (self.nu + self.smagorinsky.eddy_viscosity_tensor(ctx, state, q))
            * KernelEdacNavierStokes2D::strain_component(state, q, i, j)
    }

    fn stress_directional_derivative(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        q: usize,
        i: usize,
        j: usize,
    ) -> f64 {
        let viscosity = self.nu + self.smagorinsky.eddy_viscosity_tensor(ctx, state, q);
        let dviscosity = self
            .smagorinsky
            .eddy_viscosity_directional_derivative(ctx, state, direction, q);
        let dstrain = 0.5 * (direction.grad(i, q, j) + direction.grad(j, q, i));
        2.0 * (viscosity * dstrain
            + dviscosity * KernelEdacNavierStokes2D::strain_component(state, q, i, j))
    }
}

impl TensorResidualKernel<2> for TensorKernelEdacNavierStokes2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        Some(["u", "v", "p"].into_iter().map(str::to_owned).collect())
    }

    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let velocity = KernelEdacNavierStokes2D::velocity(state, q);
        match equation {
            0 | 1 => [
                velocity[0] * state.grad(equation, q, 0)
                    + velocity[1] * state.grad(equation, q, 1)
                    + state.grad(2, q, equation) / self.rho,
                self.stress(ctx, state, q, equation, 0),
                self.stress(ctx, state, q, equation, 1),
            ],
            2 => {
                let coefficient = self.pressure_diffusivity(ctx);
                [
                    self.rho * self.c0 * self.c0 * (state.grad(0, q, 0) + state.grad(1, q, 1))
                        + velocity[0] * state.grad(2, q, 0)
                        + velocity[1] * state.grad(2, q, 1),
                    coefficient * state.grad(2, q, 0),
                    coefficient * state.grad(2, q, 1),
                ]
            }
            _ => unreachable!(),
        }
    }

    fn tensor_jacobian_action(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let velocity = KernelEdacNavierStokes2D::velocity(state, q);
        match equation {
            0 | 1 => [
                direction.value(0, q) * state.grad(equation, q, 0)
                    + direction.value(1, q) * state.grad(equation, q, 1)
                    + velocity[0] * direction.grad(equation, q, 0)
                    + velocity[1] * direction.grad(equation, q, 1)
                    + direction.grad(2, q, equation) / self.rho,
                self.stress_directional_derivative(ctx, state, direction, q, equation, 0),
                self.stress_directional_derivative(ctx, state, direction, q, equation, 1),
            ],
            2 => {
                let coefficient = self.pressure_diffusivity(ctx);
                [
                    self.rho
                        * self.c0
                        * self.c0
                        * (direction.grad(0, q, 0) + direction.grad(1, q, 1))
                        + direction.value(0, q) * state.grad(2, q, 0)
                        + direction.value(1, q) * state.grad(2, q, 1)
                        + velocity[0] * direction.grad(2, q, 0)
                        + velocity[1] * direction.grad(2, q, 1),
                    coefficient * direction.grad(2, q, 0),
                    coefficient * direction.grad(2, q, 1),
                ]
            }
            _ => unreachable!(),
        }
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
