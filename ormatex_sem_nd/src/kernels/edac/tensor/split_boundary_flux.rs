//! Split-advection consistency flux `u_n phi / 2` for split EDAC boundaries (tensor path).
//!
//! Mathematics: supplies the conservative-half facet flux `(u.n) phi / 2`
//! (momentum components and pressure) that the split volume terms integrate
//! by parts. Pair with split-form volume kernels as the default boundary;
//! outlets additionally need Dong or directional-do-nothing with split flux.
use crate::common::{CellState, TensorFacetCtx};
use crate::kernels::common::StateTensorBoundaryIntegrator;
use crate::kernels::edac::config::fluid_field_names;
/// Tensor-product boundary consistency flux for split EDAC advection.
pub struct TensorKernelEdacSplitBoundaryFlux2D;

impl StateTensorBoundaryIntegrator<2> for TensorKernelEdacSplitBoundaryFlux2D {
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
