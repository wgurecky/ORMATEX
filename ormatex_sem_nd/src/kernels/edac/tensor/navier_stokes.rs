//! Tensor-product fused EDAC Navier-Stokes kernel (`[u, v, p]`).
//!
//! Mathematics: pointwise `(f0, f1x, f1y)` triples for the fused conservative
//! momentum and pressure equations (see
//! [`KernelEdacNavierStokes2D`](crate::kernels::edac::weak::navier_stokes::KernelEdacNavierStokes2D)
//! for the PDE terms), with the directional-derivative action for the Jacobian.
//! Owns all three equations. Weak counterpart:
//! [`KernelEdacNavierStokes2D`](crate::kernels::edac::weak::navier_stokes::KernelEdacNavierStokes2D).
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config::{EdacNavierStokes2DConfig, fluid_field_names, velocity};
use crate::kernels::edac::smagorinsky_lilly::SmagorinskyLilly2D;

/// Tensor fused EDAC Navier-Stokes kernel (sum-factorized counterpart, owns all equations).
pub struct TensorKernelEdacNavierStokes2D {
    pub rho: f64,
    pub nu: f64,
    pub c0: f64,
    pub pressure_diffusion_factor: f64,
    pub smagorinsky: SmagorinskyLilly2D,
}

impl TensorKernelEdacNavierStokes2D {
    pub fn new(rho: f64, nu: f64, c0: f64, cs: f64) -> Self {
        let config = EdacNavierStokes2DConfig::new(rho, nu, c0, cs);
        Self {
            rho: config.rho,
            nu: config.nu,
            c0: config.c0,
            pressure_diffusion_factor: config.pressure_diffusion_factor,
            smagorinsky: config.smagorinsky,
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

impl TensorResidualKernel<2> for TensorKernelEdacNavierStokes2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names()
    }

    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let velocity = velocity(state, q);
        match equation {
            0 | 1 => [
                velocity[0] * state.grad(equation, q, 0)
                    + velocity[1] * state.grad(equation, q, 1)
                    + state.grad(2, q, equation) / self.rho,
                self.config().stress_tensor_row(ctx, state, q, equation)[0],
                self.config().stress_tensor_row(ctx, state, q, equation)[1],
            ],
            2 => {
                let coefficient = self.config().pressure_diffusivity_tensor(ctx);
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
        let velocity = velocity(state, q);
        match equation {
            0 | 1 => [
                direction.value(0, q) * state.grad(equation, q, 0)
                    + direction.value(1, q) * state.grad(equation, q, 1)
                    + velocity[0] * direction.grad(equation, q, 0)
                    + velocity[1] * direction.grad(equation, q, 1)
                    + direction.grad(2, q, equation) / self.rho,
                self.config().stress_tensor_row_directional_derivative(ctx, state, direction, q, equation)[0],
                self.config().stress_tensor_row_directional_derivative(ctx, state, direction, q, equation)[1],
            ],
            2 => {
                let coefficient = self.config().pressure_diffusivity_tensor(ctx);
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
