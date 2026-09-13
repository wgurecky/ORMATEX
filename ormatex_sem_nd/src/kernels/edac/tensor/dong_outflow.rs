//! Tensor Dong OBC-C outflow for EDAC (`[u, v, p]`).
//!
//! Mathematics: `u_n u / 2` consistency flux for the weak conservative half
//! of split momentum advection, Dong's smoothed OBC-C traction
//! (`tanh` switch on `u.n / (delta * U0)`), and the matching `u_n p / 2`
//! flux for split pressure advection. With `split_flux`, pairs with
//! split-form volume kernels; otherwise with conservative-form volumes.
//! Needs the SplitBoundaryFlux default on remaining facets.
use crate::common::{CellState, TensorFacetCtx};
use crate::kernels::common::StateTensorBoundaryIntegrator;
use crate::kernels::edac::config::fluid_field_names;
use crate::kernels::edac::weak::dong_outflow::KernelEdacDongOutflow2D;

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

    pub(crate) fn as_weak(&self) -> KernelEdacDongOutflow2D {
        KernelEdacDongOutflow2D {
            rho: self.rho,
            delta: self.delta,
            velocity_scale: self.velocity_scale,
            split_flux: self.split_flux,
        }
    }
}

impl StateTensorBoundaryIntegrator<2> for TensorKernelEdacDongOutflow2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names()
    }

    fn tensor_residual(
        &self,
        ctx: &TensorFacetCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> f64 {
        let weak = self.as_weak();
        let velocity = [state.value(0, q), state.value(1, q)];
        let normal_velocity = ctx.normal[0] * velocity[0] + ctx.normal[1] * velocity[1];
        match equation {
            0 | 1 => {
                let dong_flux = weak.dong_flux(ctx.normal, velocity);
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
        let weak = self.as_weak();
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
                                * weak.dong_flux_derivative(ctx.normal, velocity, equation, unknown)
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
