//! Split-advection consistency flux `u_n phi / 2` for split EDAC boundaries.
//!
//! Mathematics: supplies the conservative-half facet flux `(u.n) phi / 2`
//! (momentum components and pressure) that the split volume terms integrate
//! by parts. Pair with split-form volume kernels as the default boundary;
//! outlets additionally need Dong or directional-do-nothing with split flux.
use crate::common::{CellState, FacetCtx};
use crate::kernels::common::StateBoundaryIntegrator;
use crate::kernels::edac::config::fluid_field_names;
/// Boundary consistency flux for the weak conservative halves of split EDAC
/// advection. Use this on non-Dong boundaries where the normal flux is not
/// already supplied by another boundary condition.
pub struct KernelEdacSplitBoundaryFlux2D;

impl StateBoundaryIntegrator for KernelEdacSplitBoundaryFlux2D {
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
