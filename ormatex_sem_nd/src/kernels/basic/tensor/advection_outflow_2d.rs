use crate::common::{CellState, TensorFacetCtx};
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
    fn nfields(&self) -> usize {
        <KernelAdvectionOutflow2D as crate::kernels::common::StateBoundaryIntegrator>::nfields(
            &self.0,
        )
    }
    fn field_names(&self) -> Option<Vec<String>> {
        <KernelAdvectionOutflow2D as crate::kernels::common::StateBoundaryIntegrator>::field_names(
            &self.0,
        )
    }
    fn tensor_residual(
        &self,
        ctx: &TensorFacetCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> f64 {
        let normal_velocity = ctx.normal[0] * self.0.ux.value(ctx.facet.local_index, q)
            + ctx.normal[1] * self.0.uy.value(ctx.facet.local_index, q);
        outflow_flux(normal_velocity, state.value(equation, q))
    }
    fn tensor_jacobian_action(
        &self,
        ctx: &TensorFacetCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> f64 {
        assert_eq!(
            state.nfields,
            <KernelAdvectionOutflow2D as crate::kernels::common::StateBoundaryIntegrator>::nfields(
                &self.0
            ),
            "kernel/state field count mismatch"
        );
        let normal_velocity = ctx.normal[0] * self.0.ux.value(ctx.facet.local_index, q)
            + ctx.normal[1] * self.0.uy.value(ctx.facet.local_index, q);
        outflow_flux_action(normal_velocity, direction.value(equation, q))
    }
}
