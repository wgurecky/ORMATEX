use crate::common::{CellState, LocalCtx, TensorCtx};

use super::kernel_common::{LinearForm, ResidualKernel, TensorResidualKernel};

/// Constant volumetric source `f(x) = val`.
pub struct KernelVolumeSource {
    pub val: f64,
}

impl KernelVolumeSource {
    pub fn new(val: f64) -> Self {
        Self { val }
    }
}

impl LinearForm for KernelVolumeSource {
    fn integrand(&self, ctx: &LocalCtx, _equation: usize, q: usize, test_i: usize) -> f64 {
        assert_eq!(ctx.ncomp, 1, "KernelVolumeSource: scalar only (ncomp==1)");
        self.val * ctx.test(test_i, 0).v(q)
    }
}

impl ResidualKernel for KernelVolumeSource {
    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        _state: &CellState,
        _equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        -self.val * ctx.test(test_i, 0).v(q)
    }

    fn jacobian_integrand(
        &self,
        _ctx: &LocalCtx,
        _state: &CellState,
        _equation: usize,
        _unknown: usize,
        _q: usize,
        _test_i: usize,
        _trial_i: usize,
    ) -> f64 {
        0.0
    }
}

/// Tensor-product volumetric source kernel.
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
