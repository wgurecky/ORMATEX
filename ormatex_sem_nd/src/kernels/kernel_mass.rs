use crate::common::{CellState, LocalCtx};

use super::kernel_common::{BilinearForm, ResidualKernel};

/// Scalar mass kernel: integral of `u * v`.
pub struct KernelMass {}

impl KernelMass {
    pub fn new() -> Self {
        Self {}
    }
}

impl BilinearForm for KernelMass {
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
