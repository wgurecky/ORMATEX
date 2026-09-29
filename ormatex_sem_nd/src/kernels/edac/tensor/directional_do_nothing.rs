//! Tensor directional do-nothing outflow for EDAC (`[u, v, p]`).
//!
//! Mathematics: `-p n / rho` traction plus a backflow penalty active only for
//! incoming normal velocity (`max(-u.n, 0)`); at `u.n = 0` the Jacobian uses
//! the outflow-side derivative. With `split_flux`, also supplies the
//! conservative-half fluxes for split momentum/pressure advection, pairing
//! with split-form volume kernels (default SplitBoundaryFlux elsewhere).
use crate::common::{LaneState, Lanes, TensorFacetCtx, LANES};
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
    #[inline]
    fn nfields(&self) -> usize {
        3
    }

    #[inline]
    fn field_names(&self) -> Option<Vec<String>> {
        directional_field_names()
    }

    /// Lane-packed directional do-nothing residual for all lanes.
    ///
    /// Per lane `l` computes exactly the scalar expression with
    /// `&ctxs[l]` (same operations in the same order, including the
    /// `min(u.n, 0)` backflow select).
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
        let p = state.value(2, q);
        for l in 0..LANES {
            out[l] = directional_residual(
                ctxs[l].normal,
                [u0[l], u1[l]],
                p[l],
                self.rho,
                self.split_flux,
                equation,
            );
        }
    }

    /// Lane-packed directional do-nothing Jacobian action for all lanes.
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
        let p = state.value(2, q);
        let du0 = direction.value(0, q);
        let du1 = direction.value(1, q);
        let dp = direction.value(2, q);
        for l in 0..LANES {
            out[l] = directional_jacobian_action(
                ctxs[l].normal,
                [u0[l], u1[l]],
                [du0[l], du1[l]],
                p[l],
                dp[l],
                self.rho,
                self.split_flux,
                equation,
            );
        }
    }
}
