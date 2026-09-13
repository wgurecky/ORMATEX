//! Stationary tensor perfect-slip EDAC wall closure (zero normal velocity).
//!
use crate::common::{CellState, TensorFacetCtx};
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
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names()
    }

    fn tensor_residual(
        &self,
        _ctx: &TensorFacetCtx<'_>,
        _state: &CellState<'_>,
        _equation: usize,
        _q: usize,
    ) -> f64 {
        0.0
    }

    fn tensor_jacobian_action(
        &self,
        _ctx: &TensorFacetCtx<'_>,
        _state: &CellState<'_>,
        _direction: &CellState<'_>,
        _equation: usize,
        _q: usize,
    ) -> f64 {
        0.0
    }
}
