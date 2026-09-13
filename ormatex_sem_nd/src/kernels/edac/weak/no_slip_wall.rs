//! Stationary no-slip EDAC wall closure (zero inviscid flux).
//!
use crate::common::{CellState, FacetCtx};
use crate::kernels::common::StateBoundaryIntegrator;
use crate::kernels::edac::config::{check_boundary_facet, fluid_field_names};

/// Stationary no-slip wall closure for the conforming EDAC SEM formulation.
///
/// The velocity value is enforced strongly through `DofReduction2D`. This
/// kernel supplies the zero inviscid wall flux in place of the split advection
/// flux; the weak viscous term supplies the unconstrained wall traction.
pub struct KernelEdacNoSlipWall2D;

impl KernelEdacNoSlipWall2D {
    pub fn new() -> Self {
        Self
    }
}

impl StateBoundaryIntegrator for KernelEdacNoSlipWall2D {
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
