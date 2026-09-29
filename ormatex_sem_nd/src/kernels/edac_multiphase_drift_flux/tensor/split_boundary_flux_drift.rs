//! Split-advection consistency flux `(u.n) phi / 2` for drift-flux (tensor path).
//!
//! Mathematics: conservative-half facet flux for split volumes — momentum,
//! pressure (as in the base kernel) plus the void fraction `alpha`. Pair with
//! split-form drift volumes as the default boundary; outlets additionally use
//! the drift directional-do-nothing kernel with split flux.
use crate::common::{LaneState, Lanes, TensorFacetCtx, LANES};
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

    /// Lane-packed split-flux residual for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor facet context per lane, length [`LANES`].
    /// * `state` - lane-packed facet solution.
    /// * `equation` - output equation index shared by all lanes.
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
        if equation < 2 {
            let v = state.value(equation, q);
            for l in 0..LANES {
                let normal_velocity = ctxs[l].normal[0] * u0[l] + ctxs[l].normal[1] * u1[l];
                out[l] = 0.5 * normal_velocity * v[l];
            }
        } else {
            let t = state.value(equation, q);
            for l in 0..LANES {
                let normal_velocity = ctxs[l].normal[0] * u0[l] + ctxs[l].normal[1] * u1[l];
                out[l] = 0.5 * normal_velocity * t[l];
            }
        }
    }

    /// Lane-packed split-flux Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor facet context per lane, length [`LANES`].
    /// * `state` - lane-packed linearization point.
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - output equation index shared by all lanes.
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
        if equation < 2 {
            let v = state.value(equation, q);
            let dv = direction.value(equation, q);
            for l in 0..LANES {
                let normal_velocity = ctxs[l].normal[0] * u0[l] + ctxs[l].normal[1] * u1[l];
                let direction_normal_velocity =
                    ctxs[l].normal[0] * du0[l] + ctxs[l].normal[1] * du1[l];
                out[l] = 0.5 * (direction_normal_velocity * v[l] + normal_velocity * dv[l]);
            }
        } else if equation == ALPHA_2D {
            let a = state.value(ALPHA_2D, q);
            let da = direction.value(ALPHA_2D, q);
            for l in 0..LANES {
                let normal_velocity = ctxs[l].normal[0] * u0[l] + ctxs[l].normal[1] * u1[l];
                let direction_normal_velocity =
                    ctxs[l].normal[0] * du0[l] + ctxs[l].normal[1] * du1[l];
                out[l] = 0.5 * (direction_normal_velocity * a[l] + normal_velocity * da[l]);
            }
        } else {
            let p = state.value(2, q);
            let dp = direction.value(2, q);
            for l in 0..LANES {
                let normal_velocity = ctxs[l].normal[0] * u0[l] + ctxs[l].normal[1] * u1[l];
                let direction_normal_velocity =
                    ctxs[l].normal[0] * du0[l] + ctxs[l].normal[1] * du1[l];
                out[l] = 0.5 * (direction_normal_velocity * p[l] + normal_velocity * dp[l]);
            }
        }
    }
}
