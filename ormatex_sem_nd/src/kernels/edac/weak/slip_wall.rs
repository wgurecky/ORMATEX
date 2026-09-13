//! Stationary perfect-slip EDAC wall closure (zero normal velocity).
//!
use crate::common::{CellState, FacetCtx};
use crate::kernels::common::StateBoundaryIntegrator;
use crate::kernels::edac::config::{check_boundary_facet, fluid_field_names};

/// Stationary perfect-slip wall closure for the conforming EDAC SEM
/// formulation.
///
/// The zero normal velocity is enforced strongly through `DofReduction2D`; the
/// weak viscous term then supplies the natural zero tangential traction.
pub struct KernelEdacSlipWall2D;

impl KernelEdacSlipWall2D {
    pub fn new() -> Self {
        Self
    }
}

impl StateBoundaryIntegrator for KernelEdacSlipWall2D {
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
        _equation: usize,
        _q: usize,
        _test_i: usize,
    ) -> f64 {
        check_boundary_facet(ctx, state);
        0.0
    }

    fn jacobian_integrand(
        &self,
        ctx: &FacetCtx,
        state: &CellState,
        _equation: usize,
        unknown: usize,
        _q: usize,
        _test_i: usize,
        _trial_i: usize,
    ) -> f64 {
        check_boundary_facet(ctx, state);
        assert!(unknown < 3, "fluid unknown field out of range");
        0.0
    }
}
