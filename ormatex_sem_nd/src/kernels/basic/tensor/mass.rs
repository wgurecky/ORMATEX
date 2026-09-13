use crate::common::{CellState, TensorCtx};

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
    fn tensor_residual(
        &self,
        _ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        _equation: usize,
        q: usize,
    ) -> [f64; 3] {
        [state.value(0, q), 0.0, 0.0]
    }
    fn tensor_jacobian_action(
        &self,
        _ctx: &TensorCtx<'_>,
        _state: &CellState<'_>,
        direction: &CellState<'_>,
        _equation: usize,
        q: usize,
    ) -> [f64; 3] {
        [direction.value(0, q), 0.0, 0.0]
    }
}
