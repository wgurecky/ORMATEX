use faer::sparse::SparseColMat;

use crate::common::{LaneState, Lanes, TensorCtx, LANES};

use crate::kernels::common::TensorResidualKernel;

use crate::kernels::basic::weak::linear_reaction::KernelLinearReaction;

/// Tensor sparse linear reaction `-R u`, GDIM-generic.
///
/// Mathematics: the triple is `(-sum_j R[eq][j]*u_j, 0, 0)` over the sparse
/// `(equation, unknown)` interaction pattern (missing entries are zero); the
/// action substitutes direction values. Weak counterpart:
/// [`KernelLinearReaction`].
/// Tensor-product linear reaction kernel (sum-factorized counterpart).
pub struct TensorKernelLinearReaction(pub KernelLinearReaction);

impl TensorKernelLinearReaction {
    pub fn new(rates: SparseColMat<usize, f64>) -> Self {
        Self(KernelLinearReaction::new(rates))
    }
    pub fn with_field_names<I, S>(rates: SparseColMat<usize, f64>, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self(KernelLinearReaction::with_field_names(rates, names))
    }
}

impl<const GDIM: usize> TensorResidualKernel<GDIM> for TensorKernelLinearReaction {
    #[inline]
    fn nfields(&self) -> usize {
        self.0.interactions.len()
    }
    #[inline]
    fn field_names(&self) -> Option<Vec<String>> {
        self.0.field_names.clone()
    }

    /// Lane-packed linear-reaction residual for all lanes.
    ///
    /// Accumulates `coeff * value` over the sparse interaction pattern in
    /// stored order, matching the scalar `iter().map(..).sum()` order
    /// per lane, then negates.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
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
        let mut acc = [0.0; LANES];
        for &(unknown, coefficient) in &self.0.interactions[equation] {
            let v = state.value(unknown, q);
            for l in 0..LANES {
                acc[l] += coefficient * v[l];
            }
        }
        for l in 0..LANES {
            f0[l] = -acc[l];
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }

    /// Lane-packed linear-reaction Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed linearization point (unused).
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - output equation shared by all lanes.
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0`/`f1x`/`f1y` - lane output slots. Overwritten.
    #[inline]
    fn tensor_jacobian_action(
        &self,
        ctxs: &[TensorCtx<'_>],
        _state: &LaneState<'_>,
        direction: &LaneState<'_>,
        equation: usize,
        q: usize,
        f0: &mut Lanes,
        f1x: &mut Lanes,
        f1y: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        let mut acc = [0.0; LANES];
        for &(unknown, coefficient) in &self.0.interactions[equation] {
            let dv = direction.value(unknown, q);
            for l in 0..LANES {
                acc[l] += coefficient * dv[l];
            }
        }
        for l in 0..LANES {
            f0[l] = -acc[l];
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }
}
