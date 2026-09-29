//! Stationary tensor perfect-slip EDAC wall closure (zero normal velocity).
//!
use crate::common::{LaneState, Lanes, TensorFacetCtx, LANES};
use crate::kernels::common::StateTensorBoundaryIntegrator;
use crate::kernels::edac::config::fluid_field_names;

/// Tensor-product stationary perfect-slip wall closure for monolithic EDAC.
pub struct TensorKernelEdacSlipWall2D;

impl TensorKernelEdacSlipWall2D {
    pub fn new() -> Self {
        Self
    }
}

impl StateTensorBoundaryIntegrator<2> for TensorKernelEdacSlipWall2D {
    #[inline]
    fn nfields(&self) -> usize {
        3
    }

    #[inline]
    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names()
    }

    /// Lane-packed slip-wall residual (identically zero).
    ///
    /// # Arguments
    /// * `ctxs` - one tensor facet context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed facet solution (unused).
    /// * `equation` - output equation shared by all lanes (unused).
    /// * `q` - facet quadrature-point index shared by all lanes (unused).
    /// * `out` - lane trace-flux slots. Overwritten with zero.
    #[inline]
    fn tensor_residual(
        &self,
        ctxs: &[TensorFacetCtx<'_>],
        _state: &LaneState<'_>,
        _equation: usize,
        _q: usize,
        out: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        *out = [0.0; LANES];
    }

    /// Lane-packed slip-wall Jacobian action (identically zero).
    ///
    /// # Arguments
    /// * `ctxs` - one tensor facet context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed linearization point (unused).
    /// * `direction` - lane-packed Gateaux direction (unused).
    /// * `equation` - output equation shared by all lanes (unused).
    /// * `q` - facet quadrature-point index shared by all lanes (unused).
    /// * `out` - lane linearized trace-flux slots. Overwritten with zero.
    #[inline]
    fn tensor_jacobian_action(
        &self,
        ctxs: &[TensorFacetCtx<'_>],
        _state: &LaneState<'_>,
        _direction: &LaneState<'_>,
        _equation: usize,
        _q: usize,
        out: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        *out = [0.0; LANES];
    }
}
