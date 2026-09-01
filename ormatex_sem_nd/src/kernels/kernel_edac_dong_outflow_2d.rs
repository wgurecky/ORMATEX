use crate::common::{CellState, FacetCtx, TensorFacetCtx};

use super::kernel_common::{StateBoundaryIntegrator, StateTensorBoundaryIntegrator};

/// Boundary consistency flux for the weak conservative halves of split EDAC
/// advection. Use this on non-Dong boundaries where the normal flux is not
/// already supplied by another boundary condition.
pub struct KernelEdacSplitBoundaryFlux2D;

/// Tensor-product boundary consistency flux for split EDAC advection.
pub struct TensorKernelEdacSplitBoundaryFlux2D;

impl StateTensorBoundaryIntegrator<2> for TensorKernelEdacSplitBoundaryFlux2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        Some(["u", "v", "p"].into_iter().map(str::to_owned).collect())
    }

    fn tensor_residual(
        &self,
        ctx: &TensorFacetCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> f64 {
        let velocity = [state.value(0, q), state.value(1, q)];
        let normal_velocity = ctx.normal[0] * velocity[0] + ctx.normal[1] * velocity[1];
        let transported = if equation < 2 {
            velocity[equation]
        } else {
            state.value(2, q)
        };
        0.5 * normal_velocity * transported
    }

    fn tensor_jacobian_action(
        &self,
        ctx: &TensorFacetCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> f64 {
        let velocity = [state.value(0, q), state.value(1, q)];
        let direction_velocity = [direction.value(0, q), direction.value(1, q)];
        let normal_velocity = ctx.normal[0] * velocity[0] + ctx.normal[1] * velocity[1];
        let direction_normal_velocity =
            ctx.normal[0] * direction_velocity[0] + ctx.normal[1] * direction_velocity[1];
        let transported = if equation < 2 {
            velocity[equation]
        } else {
            state.value(2, q)
        };
        let direction_transported = if equation < 2 {
            direction_velocity[equation]
        } else {
            direction.value(2, q)
        };
        0.5 * (direction_normal_velocity * transported + normal_velocity * direction_transported)
    }
}

impl StateBoundaryIntegrator for KernelEdacSplitBoundaryFlux2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        Some(["u", "v", "p"].into_iter().map(str::to_owned).collect())
    }

    fn residual_integrand(
        &self,
        ctx: &FacetCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        assert_eq!(ctx.gdim, 2, "EDAC split boundary requires gdim == 2");
        assert_eq!(state.nfields, 3, "fluid state must contain [u, v, p]");
        let velocity = [state.value(0, q), state.value(1, q)];
        let normal_velocity = ctx.normal[0] * velocity[0] + ctx.normal[1] * velocity[1];
        let transported = if equation < 2 {
            velocity[equation]
        } else {
            state.value(2, q)
        };
        0.5 * normal_velocity * transported * ctx.test(test_i, 0).v(q)
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
        assert_eq!(ctx.gdim, 2, "EDAC split boundary requires gdim == 2");
        assert_eq!(state.nfields, 3, "fluid state must contain [u, v, p]");
        assert!(unknown < 3, "fluid unknown field out of range");
        let velocity = [state.value(0, q), state.value(1, q)];
        let normal_velocity = ctx.normal[0] * velocity[0] + ctx.normal[1] * velocity[1];
        let transported = if equation < 2 {
            velocity[equation]
        } else {
            state.value(2, q)
        };
        let derivative = if unknown < 2 {
            0.5 * (ctx.normal[unknown] * transported
                + normal_velocity * (equation == unknown) as usize as f64)
        } else if equation == 2 {
            0.5 * normal_velocity
        } else {
            0.0
        };
        derivative * ctx.trial(trial_i, 0).v(q) * ctx.test(test_i, 0).v(q)
    }
}

fn directional_field_names() -> Option<Vec<String>> {
    Some(["u", "v", "p"].into_iter().map(str::to_owned).collect())
}

fn directional_backflow(normal_velocity: f64) -> f64 {
    (-normal_velocity).max(0.0)
}

fn directional_backflow_derivative(normal_velocity: f64) -> f64 {
    if normal_velocity < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn directional_flux(normal: &[f64], velocity: [f64; 2]) -> [f64; 2] {
    let normal_velocity = normal[0] * velocity[0] + normal[1] * velocity[1];
    let factor = 0.5 * directional_backflow(normal_velocity);
    [factor * velocity[0], factor * velocity[1]]
}

fn directional_flux_derivative(
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

fn directional_residual(
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

fn directional_jacobian_action(
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

/// Tensor-product directional do-nothing boundary kernel for monolithic EDAC.
pub struct TensorKernelEdacDirectionalDoNothing2D {
    pub rho: f64,
    pub split_flux: bool,
}

impl TensorKernelEdacDirectionalDoNothing2D {
    pub fn new(rho: f64) -> Self {
        let kernel = KernelEdacDirectionalDoNothing2D::new(rho);
        Self {
            rho: kernel.rho,
            split_flux: kernel.split_flux,
        }
    }

    pub fn with_split_flux(mut self) -> Self {
        self.split_flux = true;
        self
    }
}

impl StateTensorBoundaryIntegrator<2> for TensorKernelEdacDirectionalDoNothing2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        directional_field_names()
    }

    fn tensor_residual(
        &self,
        ctx: &TensorFacetCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> f64 {
        let velocity = [state.value(0, q), state.value(1, q)];
        directional_residual(
            ctx.normal,
            velocity,
            state.value(2, q),
            self.rho,
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
        let velocity = [state.value(0, q), state.value(1, q)];
        let direction_velocity = [direction.value(0, q), direction.value(1, q)];
        directional_jacobian_action(
            ctx.normal,
            velocity,
            direction_velocity,
            state.value(2, q),
            direction.value(2, q),
            self.rho,
            self.split_flux,
            equation,
        )
    }
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

/// Tensor-product Dong OBC-C boundary kernel for monolithic EDAC.
pub struct TensorKernelEdacDongOutflow2D {
    pub rho: f64,
    pub delta: f64,
    pub velocity_scale: f64,
    pub split_flux: bool,
}

impl TensorKernelEdacDongOutflow2D {
    pub fn new(rho: f64, delta: f64, velocity_scale: f64) -> Self {
        let kernel = KernelEdacDongOutflow2D::new(rho, delta, velocity_scale);
        Self {
            rho: kernel.rho,
            delta: kernel.delta,
            velocity_scale: kernel.velocity_scale,
            split_flux: kernel.split_flux,
        }
    }

    pub fn with_split_flux(mut self) -> Self {
        self.split_flux = true;
        self
    }

    fn switch(&self, normal_velocity: f64) -> f64 {
        0.5 * (1.0 - (normal_velocity / (self.delta * self.velocity_scale)).tanh())
    }

    fn switch_derivative(&self, normal_velocity: f64) -> f64 {
        let x = normal_velocity / (self.delta * self.velocity_scale);
        -0.5 * (1.0 - x.tanh().powi(2)) / (self.delta * self.velocity_scale)
    }

    fn dong_flux(&self, normal: &[f64], velocity: [f64; 2]) -> [f64; 2] {
        let normal_velocity = normal[0] * velocity[0] + normal[1] * velocity[1];
        let speed_squared = velocity[0] * velocity[0] + velocity[1] * velocity[1];
        let factor = 0.5 * self.switch(normal_velocity);
        [
            factor * (speed_squared * normal[0] + normal_velocity * velocity[0]),
            factor * (speed_squared * normal[1] + normal_velocity * velocity[1]),
        ]
    }

    fn dong_flux_derivative(
        &self,
        normal: &[f64],
        velocity: [f64; 2],
        component: usize,
        unknown_velocity: usize,
    ) -> f64 {
        let normal_velocity = normal[0] * velocity[0] + normal[1] * velocity[1];
        let speed_squared = velocity[0] * velocity[0] + velocity[1] * velocity[1];
        let kronecker = (component == unknown_velocity) as usize as f64;
        let derivative_bracket = 2.0 * velocity[unknown_velocity] * normal[component]
            + normal[unknown_velocity] * velocity[component]
            + normal_velocity * kronecker;
        0.5 * self.switch(normal_velocity) * derivative_bracket
            + 0.5
                * self.switch_derivative(normal_velocity)
                * normal[unknown_velocity]
                * (speed_squared * normal[component] + normal_velocity * velocity[component])
    }
}

impl StateTensorBoundaryIntegrator<2> for TensorKernelEdacDongOutflow2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        Some(["u", "v", "p"].into_iter().map(str::to_owned).collect())
    }

    fn tensor_residual(
        &self,
        ctx: &TensorFacetCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> f64 {
        let velocity = [state.value(0, q), state.value(1, q)];
        let normal_velocity = ctx.normal[0] * velocity[0] + ctx.normal[1] * velocity[1];
        match equation {
            0 | 1 => {
                let dong_flux = self.dong_flux(ctx.normal, velocity);
                (if self.split_flux {
                    0.5 * normal_velocity * velocity[equation]
                } else {
                    0.0
                }) - state.value(2, q) * ctx.normal[equation] / self.rho
                    - dong_flux[equation]
            }
            2 => {
                if self.split_flux {
                    0.5 * normal_velocity * state.value(2, q)
                } else {
                    0.0
                }
            }
            _ => unreachable!(),
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
        let velocity = [state.value(0, q), state.value(1, q)];
        let direction_velocity = [direction.value(0, q), direction.value(1, q)];
        let normal_velocity = ctx.normal[0] * velocity[0] + ctx.normal[1] * velocity[1];
        let direction_normal_velocity =
            ctx.normal[0] * direction_velocity[0] + ctx.normal[1] * direction_velocity[1];
        match equation {
            0 | 1 => {
                let split = if self.split_flux {
                    0.5 * (direction_normal_velocity * velocity[equation]
                        + normal_velocity * direction_velocity[equation])
                } else {
                    0.0
                };
                split
                    - direction.value(2, q) * ctx.normal[equation] / self.rho
                    - (0..2)
                        .map(|unknown| {
                            direction_velocity[unknown]
                                * self.dong_flux_derivative(ctx.normal, velocity, equation, unknown)
                        })
                        .sum::<f64>()
            }
            2 => {
                if self.split_flux {
                    0.5 * (direction_normal_velocity * state.value(2, q)
                        + normal_velocity * direction.value(2, q))
                } else {
                    0.0
                }
            }
            _ => unreachable!(),
        }
    }
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

    fn switch(&self, normal_velocity: f64) -> f64 {
        0.5 * (1.0 - (normal_velocity / (self.delta * self.velocity_scale)).tanh())
    }

    fn switch_derivative(&self, normal_velocity: f64) -> f64 {
        let x = normal_velocity / (self.delta * self.velocity_scale);
        -0.5 * (1.0 - x.tanh().powi(2)) / (self.delta * self.velocity_scale)
    }

    fn dong_flux(&self, normal: &[f64], velocity: [f64; 2]) -> [f64; 2] {
        let normal_velocity = normal[0] * velocity[0] + normal[1] * velocity[1];
        let speed_squared = velocity[0] * velocity[0] + velocity[1] * velocity[1];
        let factor = 0.5 * self.switch(normal_velocity);
        [
            factor * (speed_squared * normal[0] + normal_velocity * velocity[0]),
            factor * (speed_squared * normal[1] + normal_velocity * velocity[1]),
        ]
    }

    fn dong_flux_derivative(
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
        Some(["u", "v", "p"].into_iter().map(str::to_owned).collect())
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
