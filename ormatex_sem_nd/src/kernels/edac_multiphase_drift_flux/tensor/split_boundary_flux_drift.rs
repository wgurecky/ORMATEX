//! Split-advection consistency flux `(u.n) phi / 2` for drift-flux (tensor path).
//!
//! Mathematics: conservative-half facet flux for split volumes — momentum,
//! pressure (as in the base kernel) plus the void fraction `alpha`. Pair with
//! split-form drift volumes as the default boundary; outlets additionally use
//! the drift directional-do-nothing kernel with split flux.
use crate::common::{CellState, TensorFacetCtx};
use crate::kernels::common::StateTensorBoundaryIntegrator;
use crate::kernels::edac_multiphase_drift_flux::config::{drift_field_names, ALPHA_2D};

/// Tensor boundary consistency flux for split drift-flux advection.
pub struct TensorDriftSplitBoundaryFlux2D;

impl StateTensorBoundaryIntegrator<2> for TensorDriftSplitBoundaryFlux2D {
    fn nfields(&self) -> usize {
        4
    }

    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names()
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
            state.value(equation, q)
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
        } else if equation == ALPHA_2D {
            state.value(ALPHA_2D, q)
        } else {
            state.value(2, q)
        };
        let direction_transported = if equation < 2 {
            direction_velocity[equation]
        } else if equation == ALPHA_2D {
            direction.value(ALPHA_2D, q)
        } else {
            direction.value(2, q)
        };
        0.5 * (direction_normal_velocity * transported + normal_velocity * direction_transported)
    }
}
