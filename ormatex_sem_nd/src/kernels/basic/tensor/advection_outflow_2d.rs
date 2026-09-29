use crate::common::{LaneState, Lanes, TensorFacetCtx, LANES};
use crate::material::FrozenFacetField;

use crate::kernels::common::StateTensorBoundaryIntegrator;

use crate::kernels::basic::weak::advection_outflow_2d::{
    outflow_flux, outflow_flux_action, KernelAdvectionOutflow2D,
};

/// Tensor directional advective outflow for a frozen 2D velocity field.
///
/// Mathematics: pointwise trace flux `max(u.n, 0) * c` per transported
/// equation (zero backflow concentration); the action carries the same
/// outflow factor on the direction value. Weak counterpart:
/// [`KernelAdvectionOutflow2D`].
/// Tensor-product 2D advective-outflow kernel (sum-factorized counterpart).
pub struct TensorKernelAdvectionOutflow2D(pub KernelAdvectionOutflow2D);

impl TensorKernelAdvectionOutflow2D {
    pub fn new(ux: FrozenFacetField, uy: FrozenFacetField) -> Self {
        Self(KernelAdvectionOutflow2D::new(ux, uy))
    }
    pub fn with_field_names<I, S>(ux: FrozenFacetField, uy: FrozenFacetField, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self(KernelAdvectionOutflow2D::with_field_names(ux, uy, names))
    }
}

impl StateTensorBoundaryIntegrator<2> for TensorKernelAdvectionOutflow2D {
    #[inline]
    fn nfields(&self) -> usize {
        <KernelAdvectionOutflow2D as crate::kernels::common::StateBoundaryIntegrator>::nfields(
            &self.0,
        )
    }
    #[inline]
    fn field_names(&self) -> Option<Vec<String>> {
        <KernelAdvectionOutflow2D as crate::kernels::common::StateBoundaryIntegrator>::field_names(
            &self.0,
        )
    }

    /// Lane-packed advective-outflow residual for all lanes.
    ///
    /// Per lane `l` computes exactly the scalar expression with
    /// `&ctxs[l]` (same operations in the same order, including the
    /// `max(u.n, 0)` outflow select over the lane's frozen velocity).
    ///
    /// # Arguments
    /// * `ctxs` - one tensor facet context per lane, length [`LANES`].
    /// * `state` - lane-packed transported solution.
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
        let c = state.value(equation, q);
        for l in 0..LANES {
            let normal = ctxs[l].normal;
            let facet = ctxs[l].facet.local_index;
            let normal_velocity =
                normal[0] * self.0.ux.value(facet, q) + normal[1] * self.0.uy.value(facet, q);
            out[l] = outflow_flux(normal_velocity, c[l]);
        }
    }

    /// Lane-packed advective-outflow Jacobian action for all lanes.
    ///
    /// Per lane `l` computes exactly the scalar expression with
    /// `&ctxs[l]` (same operations in the same order).
    ///
    /// # Arguments
    /// * `ctxs` - one tensor facet context per lane, length [`LANES`].
    /// * `state` - lane-packed linearization point (only the field count is
    ///   checked, mirroring the scalar path).
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
        assert_eq!(
            state.nfields,
            <KernelAdvectionOutflow2D as crate::kernels::common::StateBoundaryIntegrator>::nfields(
                &self.0
            ),
            "kernel/state field count mismatch"
        );
        let dc = direction.value(equation, q);
        for l in 0..LANES {
            let normal = ctxs[l].normal;
            let facet = ctxs[l].facet.local_index;
            let normal_velocity =
                normal[0] * self.0.ux.value(facet, q) + normal[1] * self.0.uy.value(facet, q);
            out[l] = outflow_flux_action(normal_velocity, dc[l]);
        }
    }
}
