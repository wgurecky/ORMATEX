//! Split-advection consistency flux `u_n phi / 2` for split EDAC boundaries (tensor path).
//!
//! Mathematics: supplies the conservative-half facet flux `(u.n) phi / 2`
//! (momentum components and pressure) that the split volume terms integrate
//! by parts. Pair with split-form volume kernels as the default boundary;
//! outlets additionally need Dong or directional-do-nothing with split flux.
use crate::common::{LaneState, Lanes, TensorFacetCtx, LANES};
use crate::kernels::common::StateTensorBoundaryIntegrator;
use crate::kernels::edac::config::fluid_field_names;
/// Tensor-product boundary consistency flux for split EDAC advection.
pub struct TensorKernelEdacSplitBoundaryFlux2D;

impl StateTensorBoundaryIntegrator<2> for TensorKernelEdacSplitBoundaryFlux2D {
    #[inline]
    fn nfields(&self) -> usize {
        3
    }

    #[inline]
    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names()
    }

    /// Lane-packed split-flux residual for all lanes.
    ///
    /// Per lane `l` computes exactly the scalar expression with
    /// `&ctxs[l]` (same operations in the same order).
    ///
    /// # Arguments
    /// * `ctxs` - one tensor facet context per lane, length [`LANES`].
    /// * `state` - lane-packed facet solution.
    /// * `equation` - output equation shared by all lanes.
    /// * `q` - facet quadrature-point index shared by all lanes.
    /// * `out` - lane trace-flux slots. Overwritten.
    #[inline]
    fn tensor_residual(
        &self,
        ctxs: &[TensorFacetCtx<'_>],
        state: &LaneState<'_>,
        equation: usize,
        q: usize,
        out: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        let u0 = state.value(0, q);
        let u1 = state.value(1, q);
        let transported = if equation < 2 {
            state.value(equation, q)
        } else {
            state.value(2, q)
        };
        for l in 0..LANES {
            let normal = ctxs[l].normal;
            let normal_velocity = normal[0] * u0[l] + normal[1] * u1[l];
            out[l] = 0.5 * normal_velocity * transported[l];
        }
    }

    /// Lane-packed split-flux Jacobian action for all lanes.
    ///
    /// Per lane `l` computes exactly the scalar expression with
    /// `&ctxs[l]` (same operations in the same order).
    ///
    /// # Arguments
    /// * `ctxs` - one tensor facet context per lane, length [`LANES`].
    /// * `state` - lane-packed linearization point.
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - output equation shared by all lanes.
    /// * `q` - facet quadrature-point index shared by all lanes.
    /// * `out` - lane linearized trace-flux slots. Overwritten.
    #[inline]
    fn tensor_jacobian_action(
        &self,
        ctxs: &[TensorFacetCtx<'_>],
        state: &LaneState<'_>,
        direction: &LaneState<'_>,
        equation: usize,
        q: usize,
        out: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        let u0 = state.value(0, q);
        let u1 = state.value(1, q);
        let du0 = direction.value(0, q);
        let du1 = direction.value(1, q);
        let transported = if equation < 2 {
            state.value(equation, q)
        } else {
            state.value(2, q)
        };
        let direction_transported = if equation < 2 {
            direction.value(equation, q)
        } else {
            direction.value(2, q)
        };
        for l in 0..LANES {
            let normal = ctxs[l].normal;
            let normal_velocity = normal[0] * u0[l] + normal[1] * u1[l];
            let direction_normal_velocity = normal[0] * du0[l] + normal[1] * du1[l];
            out[l] = 0.5
                * (direction_normal_velocity * transported[l]
                    + normal_velocity * direction_transported[l]);
        }
    }
}
