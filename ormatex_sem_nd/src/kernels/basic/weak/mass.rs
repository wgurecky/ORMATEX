use crate::common::{CellState, LocalCtx, TensorCtx};

use crate::kernels::common::{BilinearForm, ResidualKernel};

/// Scalar mass kernel: integral of `u * v`.
///
/// Serves as `BilinearForm` (with tensor fast path), `ResidualKernel`, and the
/// tensor `(u, 0, 0)` triple feeding lumped-mass assembly. Tensor mirror:
/// [`TensorKernelMass`](crate::kernels::basic::tensor::mass::TensorKernelMass).
pub struct KernelMass {}

impl KernelMass {
    pub fn new() -> Self {
        Self {}
    }
}

impl BilinearForm for KernelMass {
    fn supports_tensor_bilinear_1d(&self) -> bool {
        true
    }

    fn supports_tensor_bilinear(&self) -> bool {
        true
    }

    fn tensor_bilinear(
        &self,
        _ctx: &TensorCtx<'_>,
        _equation: usize,
        _unknown: usize,
        _q: usize,
        trial_value: f64,
        _trial_grad: [f64; 2],
    ) -> [f64; 3] {
        [trial_value, 0.0, 0.0]
    }

    fn integrand(
        &self,
        ctx: &LocalCtx,
        _equation: usize,
        _unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        assert_eq!(ctx.ncomp, 1, "KernelMass: scalar only (ncomp==1)");
        ctx.test(test_i, 0).v(q) * ctx.trial(trial_i, 0).v(q)
    }
}

impl ResidualKernel for KernelMass {
    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        _equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        state.value(0, q) * ctx.test(test_i, 0).v(q)
    }

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx,
        _state: &CellState,
        _equation: usize,
        _unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        ctx.test(test_i, 0).v(q) * ctx.trial(trial_i, 0).v(q)
    }
}
