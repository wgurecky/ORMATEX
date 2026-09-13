use crate::common::{CellState, TensorCtx};

use crate::kernels::common::TensorResidualKernel;

use crate::kernels::basic::weak::volume_source::KernelVolumeSource;

/// Tensor constant volumetric source, GDIM-generic.
///
/// Mathematics: the triple is `(-val, 0, 0)` matching the `ResidualKernel`
/// sign convention (sources move to the LHS); the action is zero. Weak
/// counterpart: [`KernelVolumeSource`].
/// Tensor-product volumetric source kernel (sum-factorized counterpart).
pub struct TensorKernelVolumeSource(pub KernelVolumeSource);

impl TensorKernelVolumeSource {
    pub fn new(val: f64) -> Self {
        Self(KernelVolumeSource::new(val))
    }
}

impl<const GDIM: usize> TensorResidualKernel<GDIM> for TensorKernelVolumeSource {
    fn tensor_residual(
        &self,
        _ctx: &TensorCtx<'_>,
        _state: &CellState<'_>,
        _equation: usize,
        _q: usize,
    ) -> [f64; 3] {
        [-self.0.val, 0.0, 0.0]
    }
    fn tensor_jacobian_action(
        &self,
        _ctx: &TensorCtx<'_>,
        _state: &CellState<'_>,
        _direction: &CellState<'_>,
        _equation: usize,
        _q: usize,
    ) -> [f64; 3] {
        [0.0, 0.0, 0.0]
    }
}
