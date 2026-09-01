use crate::common::{CellState, FacetCtx, TensorFacetCtx};

use super::kernel_common::{StateBoundaryIntegrator, StateTensorBoundaryIntegrator};

fn field_names() -> Option<Vec<String>> {
    Some(["u", "v", "p"].into_iter().map(str::to_owned).collect())
}

fn check(ctx: &FacetCtx, state: &CellState) {
    assert_eq!(ctx.gdim, 2, "EDAC wall boundary requires gdim == 2");
    assert_eq!(ctx.ncomp, 1, "fluid fields must be scalar fields");
    assert_eq!(state.nfields, 3, "fluid state must contain [u, v, p]");
}

/// Stationary no-slip wall closure for the conforming EDAC SEM formulation.
///
/// The velocity value is enforced strongly through `DofReduction2D`. This
/// kernel supplies the zero inviscid wall flux in place of the split advection
/// flux; the weak viscous term supplies the unconstrained wall traction.
pub struct KernelEdacNoSlipWall2D;

/// Tensor-product stationary no-slip wall closure for monolithic EDAC.
pub struct TensorKernelEdacNoSlipWall2D;

/// Stationary perfect-slip wall closure for the conforming EDAC SEM
/// formulation.
///
/// The zero normal velocity is enforced strongly through `DofReduction2D`; the
/// weak viscous term then supplies the natural zero tangential traction.
pub struct KernelEdacSlipWall2D;

/// Tensor-product stationary perfect-slip wall closure for monolithic EDAC.
pub struct TensorKernelEdacSlipWall2D;

impl KernelEdacNoSlipWall2D {
    pub fn new() -> Self {
        Self
    }
}

impl TensorKernelEdacNoSlipWall2D {
    pub fn new() -> Self {
        Self
    }
}

impl KernelEdacSlipWall2D {
    pub fn new() -> Self {
        Self
    }
}

impl TensorKernelEdacSlipWall2D {
    pub fn new() -> Self {
        Self
    }
}

impl StateBoundaryIntegrator for KernelEdacNoSlipWall2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        field_names()
    }

    fn residual_integrand(
        &self,
        ctx: &FacetCtx,
        state: &CellState,
        _equation: usize,
        _q: usize,
        _test_i: usize,
    ) -> f64 {
        check(ctx, state);
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
        check(ctx, state);
        assert!(unknown < 3, "fluid unknown field out of range");
        0.0
    }
}

impl StateTensorBoundaryIntegrator<2> for TensorKernelEdacNoSlipWall2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        field_names()
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

impl StateBoundaryIntegrator for KernelEdacSlipWall2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        field_names()
    }

    fn residual_integrand(
        &self,
        ctx: &FacetCtx,
        state: &CellState,
        _equation: usize,
        _q: usize,
        _test_i: usize,
    ) -> f64 {
        check(ctx, state);
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
        check(ctx, state);
        assert!(unknown < 3, "fluid unknown field out of range");
        0.0
    }
}

impl StateTensorBoundaryIntegrator<2> for TensorKernelEdacSlipWall2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        field_names()
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
