use faer::sparse::SparseColMat;

use crate::common::{CellState, TensorCtx};

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
    fn nfields(&self) -> usize {
        self.0.interactions.len()
    }
    fn field_names(&self) -> Option<Vec<String>> {
        self.0.field_names.clone()
    }
    fn tensor_residual(
        &self,
        _ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        [
            -self.0.interactions[equation]
                .iter()
                .map(|&(unknown, coefficient)| coefficient * state.value(unknown, q))
                .sum::<f64>(),
            0.0,
            0.0,
        ]
    }
    fn tensor_jacobian_action(
        &self,
        _ctx: &TensorCtx<'_>,
        _state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        [
            -self.0.interactions[equation]
                .iter()
                .map(|&(unknown, coefficient)| coefficient * direction.value(unknown, q))
                .sum::<f64>(),
            0.0,
            0.0,
        ]
    }

}
