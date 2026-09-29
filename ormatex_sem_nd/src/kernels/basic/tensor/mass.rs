use crate::common::{LaneState, Lanes, TensorCtx, LANES};

use crate::kernels::common::TensorResidualKernel;

use crate::kernels::basic::weak::mass::KernelMass;

/// Tensor mass kernel, GDIM-generic.
///
/// Mathematics: the triple is `(u, 0, 0)` with the action `(du, 0, 0)`;
/// feeds the lumped-mass path. Weak counterpart:
/// [`KernelMass`].
/// Tensor-product mass kernel (sum-factorized counterpart).
pub struct TensorKernelMass(pub KernelMass);

impl TensorKernelMass {
    pub fn new() -> Self {
        Self(KernelMass::new())
    }
}

impl<const GDIM: usize> TensorResidualKernel<GDIM> for TensorKernelMass {
    /// Lane-packed mass residual for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed solution.
    /// * `equation` - unused (single output).
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0`/`f1x`/`f1y` - lane output slots. Overwritten.
    #[inline]
    fn tensor_residual(
        &self,
        ctxs: &[TensorCtx<'_>],
        state: &LaneState<'_>,
        _equation: usize,
        q: usize,
        f0: &mut Lanes,
        f1x: &mut Lanes,
        f1y: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        let v = state.value(0, q);
        for l in 0..LANES {
            f0[l] = v[l];
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }

    /// Lane-packed mass Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed linearization point (unused).
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - unused (single output).
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0`/`f1x`/`f1y` - lane output slots. Overwritten.
    #[inline]
    fn tensor_jacobian_action(
        &self,
        ctxs: &[TensorCtx<'_>],
        _state: &LaneState<'_>,
        direction: &LaneState<'_>,
        _equation: usize,
        q: usize,
        f0: &mut Lanes,
        f1x: &mut Lanes,
        f1y: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        let dv = direction.value(0, q);
        for l in 0..LANES {
            f0[l] = dv[l];
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }
}
