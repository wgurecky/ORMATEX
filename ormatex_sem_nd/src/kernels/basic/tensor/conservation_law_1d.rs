use crate::common::{LaneState, Lanes, TensorCtx, LANES};

use crate::kernels::common::{FluxKernel1D, TensorResidualKernel};

use crate::kernels::basic::weak::conservation_law_1d::KernelConservationLaw1D;

/// Tensor generic Galerkin flux for `U_t + dF(U)/dx = 0`.
///
/// Mathematics: the triple is `(0, -F(U), 0)` evaluated pointwise through the
/// [`FluxKernel1D`] callback; the action contracts `dF/dU` with the direction
/// values. Generic over `FluxKernel1D`. Weak counterpart:
/// [`KernelConservationLaw1D`].
/// Tensor-product 1D conservation-law kernel (sum-factorized counterpart).
pub struct TensorKernelConservationLaw1D<F>(pub KernelConservationLaw1D<F>);

impl<F> TensorKernelConservationLaw1D<F> {
    /// Build the tensor kernel from a pointwise 1D flux.
    ///
    /// # Arguments
    /// * `flux` - flux implementation shared with the weak kernel.
    ///
    /// # Returns
    /// Tensor kernel wrapping `flux` in a [`KernelConservationLaw1D`].
    pub fn new(flux: F) -> Self {
        Self(KernelConservationLaw1D::new(flux))
    }
}

impl<F: FluxKernel1D + Send + Sync> TensorResidualKernel<1> for TensorKernelConservationLaw1D<F> {
    #[inline]
    fn nfields(&self) -> usize {
        self.0.flux.nfields()
    }

    #[inline]
    fn field_names(&self) -> Option<Vec<String>> {
        self.0.flux.field_names()
    }

    /// Lane-packed conservation-law residual for all lanes.
    ///
    /// Calls the pointwise flux callback once per lane with that lane's
    /// [`TensorCtx`] and scalar [`StateView`](crate::common::StateView)
    /// (`state.lane(l)`); no per-lane copy is needed.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed solution.
    /// * `equation` - output equation shared by all lanes.
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0`/`f1x`/`f1y` - lane output slots. Overwritten.
    #[inline]
    fn tensor_residual(
        &self,
        ctxs: &[TensorCtx<'_>],
        state: &LaneState<'_>,
        equation: usize,
        q: usize,
        f0: &mut Lanes,
        f1x: &mut Lanes,
        f1y: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = -self.0.flux.flux(&ctxs[l], state.lane(l), equation, q);
            f1y[l] = 0.0;
        }
    }

    /// Lane-packed conservation-law Jacobian action for all lanes.
    ///
    /// Per lane contracts `dF/dU` with the direction values in unknown order
    /// (`acc += jac * dv`, starting from `0.0`), then negates.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed linearization point.
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - output equation shared by all lanes.
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0`/`f1x`/`f1y` - lane output slots. Overwritten.
    #[inline]
    fn tensor_jacobian_action(
        &self,
        ctxs: &[TensorCtx<'_>],
        state: &LaneState<'_>,
        direction: &LaneState<'_>,
        equation: usize,
        q: usize,
        f0: &mut Lanes,
        f1x: &mut Lanes,
        f1y: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        let nf = self.0.flux.nfields();
        for l in 0..LANES {
            let lane = state.lane(l);
            let dir = direction.lane(l);
            let mut flux_action = 0.0;
            for unknown in 0..nf {
                flux_action += self
                    .0
                    .flux
                    .flux_jacobian(&ctxs[l], lane, equation, unknown, q)
                    * dir.value(unknown, q);
            }
            f0[l] = 0.0;
            f1x[l] = -flux_action;
            f1y[l] = 0.0;
        }
    }
}
