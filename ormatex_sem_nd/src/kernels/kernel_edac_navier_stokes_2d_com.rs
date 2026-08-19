use crate::common::{CellState, LocalCtx, ShapeFn};

use super::kernel_common::ResidualKernel;
use super::kernel_smagorinsky_lilly_2d::SmagorinskyLilly2D;

/// Shared parameters for the decomposed three-field EDAC Navier-Stokes terms.
#[derive(Clone, Copy, Debug)]
pub struct EdacNavierStokes2DConfig {
    pub rho: f64,
    pub nu: f64,
    pub c0: f64,
    pub pressure_diffusion_factor: f64,
    pub smagorinsky: SmagorinskyLilly2D,
}

impl EdacNavierStokes2DConfig {
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
        assert_eq!(ctx.gdim, 2, "EDAC Navier-Stokes terms require gdim == 2");
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
        trial: &ShapeFn<'_>,
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

fn field_names() -> Option<Vec<String>> {
    Some(["u", "v", "p"].into_iter().map(str::to_owned).collect())
}

/// Momentum convection contribution for the `u` and `v` equations.
pub struct KernelEdacMomentumConvection2D {
    pub config: EdacNavierStokes2DConfig,
}

impl KernelEdacMomentumConvection2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl ResidualKernel for KernelEdacMomentumConvection2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        field_names()
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        EdacNavierStokes2DConfig::check(ctx, state);
        match equation {
            0 | 1 => {
                let velocity = EdacNavierStokes2DConfig::velocity(state, q);
                let convection = velocity
                    .iter()
                    .enumerate()
                    .map(|(j, &u_j)| u_j * state.grad(equation, q, j))
                    .sum::<f64>();
                convection * ctx.test(test_i, 0).v(q)
            }
            2 => 0.0,
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
        EdacNavierStokes2DConfig::check(ctx, state);
        assert!(unknown < 3, "fluid unknown field out of range");
        if equation == 2 {
            return 0.0;
        }
        let test = ctx.test(test_i, 0);
        if unknown >= 2 {
            return 0.0;
        }
        let trial = ctx.trial(trial_i, 0);
        let velocity = EdacNavierStokes2DConfig::velocity(state, q);
        let convection = trial.v(q) * state.grad(equation, q, unknown)
            + if unknown == equation {
                velocity
                    .iter()
                    .enumerate()
                    .map(|(j, &u_j)| u_j * trial.grad(q, j))
                    .sum::<f64>()
            } else {
                0.0
            };
        convection * test.v(q)
    }
}

/// Pressure-gradient contribution for the two momentum equations.
pub struct KernelEdacPressureGradient2D {
    pub config: EdacNavierStokes2DConfig,
}

impl KernelEdacPressureGradient2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl ResidualKernel for KernelEdacPressureGradient2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        field_names()
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        EdacNavierStokes2DConfig::check(ctx, state);
        match equation {
            0 | 1 => state.grad(2, q, equation) * ctx.test(test_i, 0).v(q) / self.config.rho,
            2 => 0.0,
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
        EdacNavierStokes2DConfig::check(ctx, state);
        assert!(unknown < 3, "fluid unknown field out of range");
        if equation >= 2 || unknown != 2 {
            return 0.0;
        }
        ctx.trial(trial_i, 0).grad(q, equation) * ctx.test(test_i, 0).v(q) / self.config.rho
    }
}

/// Weak viscous stress contribution for the two momentum equations.
pub struct KernelEdacViscousStress2D {
    pub config: EdacNavierStokes2DConfig,
}

impl KernelEdacViscousStress2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl ResidualKernel for KernelEdacViscousStress2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        field_names()
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        EdacNavierStokes2DConfig::check(ctx, state);
        match equation {
            0 | 1 => (0..2)
                .map(|j| {
                    self.config.stress(ctx, state, q, equation, j) * ctx.test(test_i, 0).grad(q, j)
                })
                .sum(),
            2 => 0.0,
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
        EdacNavierStokes2DConfig::check(ctx, state);
        assert!(unknown < 3, "fluid unknown field out of range");
        if equation >= 2 {
            return 0.0;
        }
        let trial = ctx.trial(trial_i, 0);
        (0..2)
            .map(|j| {
                self.config
                    .stress_jacobian(ctx, state, q, equation, j, unknown, &trial)
                    * ctx.test(test_i, 0).grad(q, j)
            })
            .sum()
    }
}

/// Artificial-compressibility divergence contribution for the pressure equation.
pub struct KernelEdacPressureDivergence2D {
    pub config: EdacNavierStokes2DConfig,
}

impl KernelEdacPressureDivergence2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl ResidualKernel for KernelEdacPressureDivergence2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        field_names()
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        EdacNavierStokes2DConfig::check(ctx, state);
        if equation != 2 {
            return 0.0;
        }
        let divergence = state.grad(0, q, 0) + state.grad(1, q, 1);
        self.config.rho * self.config.c0 * self.config.c0 * divergence * ctx.test(test_i, 0).v(q)
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
        EdacNavierStokes2DConfig::check(ctx, state);
        assert!(unknown < 3, "fluid unknown field out of range");
        if equation != 2 || unknown >= 2 {
            return 0.0;
        }
        self.config.rho
            * self.config.c0
            * self.config.c0
            * ctx.trial(trial_i, 0).grad(q, unknown)
            * ctx.test(test_i, 0).v(q)
    }
}

/// Pressure advection contribution for the pressure equation.
pub struct KernelEdacPressureAdvection2D {
    pub config: EdacNavierStokes2DConfig,
}

impl KernelEdacPressureAdvection2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl ResidualKernel for KernelEdacPressureAdvection2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        field_names()
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        EdacNavierStokes2DConfig::check(ctx, state);
        if equation != 2 {
            return 0.0;
        }
        let velocity = EdacNavierStokes2DConfig::velocity(state, q);
        let pressure_advection = velocity
            .iter()
            .enumerate()
            .map(|(j, &u_j)| u_j * state.grad(2, q, j))
            .sum::<f64>();
        pressure_advection * ctx.test(test_i, 0).v(q)
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
        EdacNavierStokes2DConfig::check(ctx, state);
        assert!(unknown < 3, "fluid unknown field out of range");
        if equation != 2 {
            return 0.0;
        }
        let trial = ctx.trial(trial_i, 0);
        let pressure_advection = if unknown == 2 {
            let velocity = EdacNavierStokes2DConfig::velocity(state, q);
            velocity
                .iter()
                .enumerate()
                .map(|(j, &u_j)| u_j * trial.grad(q, j))
                .sum::<f64>()
        } else {
            trial.v(q) * state.grad(2, q, unknown)
        };
        pressure_advection * ctx.test(test_i, 0).v(q)
    }
}

/// Weak pressure diffusion contribution for the pressure equation.
pub struct KernelEdacPressureDiffusion2D {
    pub config: EdacNavierStokes2DConfig,
}

impl KernelEdacPressureDiffusion2D {
    pub fn new(config: EdacNavierStokes2DConfig) -> Self {
        Self { config }
    }
}

impl ResidualKernel for KernelEdacPressureDiffusion2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        field_names()
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        EdacNavierStokes2DConfig::check(ctx, state);
        if equation != 2 {
            return 0.0;
        }
        let pressure_gradient = (0..2)
            .map(|j| state.grad(2, q, j) * ctx.test(test_i, 0).grad(q, j))
            .sum::<f64>();
        self.config.pressure_diffusivity(ctx) * pressure_gradient
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
        EdacNavierStokes2DConfig::check(ctx, state);
        assert!(unknown < 3, "fluid unknown field out of range");
        if equation != 2 || unknown != 2 {
            return 0.0;
        }
        let pressure_gradient = (0..2)
            .map(|j| ctx.trial(trial_i, 0).grad(q, j) * ctx.test(test_i, 0).grad(q, j))
            .sum::<f64>();
        self.config.pressure_diffusivity(ctx) * pressure_gradient
    }
}
