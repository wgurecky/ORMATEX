//! Tensor directional do-nothing outflow for EDAC (`[u, v, p]`).
//!
//! Mathematics: `-p n / rho` traction plus a backflow penalty active only for
//! incoming normal velocity (`max(-u.n, 0)`); at `u.n = 0` the Jacobian uses
//! the outflow-side derivative. With `split_flux`, also supplies the
//! conservative-half fluxes for split momentum/pressure advection, pairing
//! with split-form volume kernels (default SplitBoundaryFlux elsewhere).
use crate::common::{CellState, TensorFacetCtx};
use crate::kernels::common::StateTensorBoundaryIntegrator;
use crate::kernels::edac::weak::directional_do_nothing::KernelEdacDirectionalDoNothing2D;
use crate::kernels::edac::weak::directional_do_nothing::{
    directional_field_names, directional_jacobian_action, directional_residual,
};

/// Tensor-product directional do-nothing boundary kernel for monolithic EDAC.
pub struct TensorKernelEdacDirectionalDoNothing2D {
    pub rho: f64,
    pub split_flux: bool,
}

impl TensorKernelEdacDirectionalDoNothing2D {
    pub fn new(rho: f64) -> Self {
        let kernel = KernelEdacDirectionalDoNothing2D::new(rho);
        Self {
            rho: kernel.rho,
            split_flux: kernel.split_flux,
        }
    }

    pub fn with_split_flux(mut self) -> Self {
        self.split_flux = true;
        self
    }
}

impl StateTensorBoundaryIntegrator<2> for TensorKernelEdacDirectionalDoNothing2D {
    fn nfields(&self) -> usize {
        3
    }

    fn field_names(&self) -> Option<Vec<String>> {
        directional_field_names()
    }

    fn tensor_residual(
        &self,
        ctx: &TensorFacetCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> f64 {
        let velocity = [state.value(0, q), state.value(1, q)];
        directional_residual(
            ctx.normal,
            velocity,
            state.value(2, q),
            self.rho,
            self.split_flux,
            equation,
        )
    }

    fn tensor_jacobian_action(
        &self,
        ctx: &TensorFacetCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> f64 {
        let velocity = [state.value(0, q), state.value(1, q)];
        let direction_velocity = [direction.value(0, q), direction.value(1, q)];
        directional_jacobian_action(
            ctx.normal,
            velocity,
            direction_velocity,
            state.value(2, q),
            direction.value(2, q),
            self.rho,
            self.split_flux,
            equation,
        )
    }
}
