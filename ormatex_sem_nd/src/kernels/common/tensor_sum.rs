//! Statically dispatched sums of tensor residual kernels.
//!
//! [`TensorResidualKernelSum`] chains kernels with `.with()` while
//! keeping pointwise evaluation statically dispatched; the
//! [`fuse_tensor_kernels`] macro is sugar over its builder.
use super::traits::TensorResidualKernel;
use crate::common::{CellState, TensorCtx};


/// Statically dispatched additive composition of tensor residual kernels.
///
/// Calling [`Self::with`] nests another concrete kernel in the type, so the
/// pointwise sum remains statically dispatched and inlinable.
pub struct TensorResidualKernelSum<A, B = ()> {
    first: A,
    second: B,
}

impl<A> TensorResidualKernelSum<A, ()> {
    pub fn from_kernel(kernel: A) -> Self {
        Self {
            first: kernel,
            second: (),
        }
    }
}

impl<A, B> TensorResidualKernelSum<A, B> {
    pub fn with<C>(self, kernel: C) -> TensorResidualKernelSum<Self, C> {
        TensorResidualKernelSum {
            first: self,
            second: kernel,
        }
    }
}

impl<const GDIM: usize, A> TensorResidualKernel<GDIM> for TensorResidualKernelSum<A, ()>
where
    A: TensorResidualKernel<GDIM>,
{
    fn nfields(&self) -> usize {
        self.first.nfields()
    }

    fn owns_equation(&self, equation: usize) -> bool {
        self.first.owns_equation(equation)
    }

    fn field_names(&self) -> Option<Vec<String>> {
        self.first.field_names()
    }

    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        self.first.tensor_residual(ctx, state, equation, q)
    }

    fn tensor_jacobian_action(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        self.first
            .tensor_jacobian_action(ctx, state, direction, equation, q)
    }
}

impl<const GDIM: usize, A, B> TensorResidualKernel<GDIM> for TensorResidualKernelSum<A, B>
where
    A: TensorResidualKernel<GDIM>,
    B: TensorResidualKernel<GDIM>,
{
    fn nfields(&self) -> usize {
        assert_eq!(
            self.first.nfields(),
            self.second.nfields(),
            "tensor residual kernel sum field count mismatch"
        );
        self.first.nfields()
    }

    fn field_names(&self) -> Option<Vec<String>> {
        let first = self.first.field_names();
        let second = self.second.field_names();
        match (first, second) {
            (Some(first), Some(second)) => {
                assert_eq!(
                    first, second,
                    "tensor residual kernel sum field names/order mismatch"
                );
                Some(first)
            }
            (Some(first), None) => Some(first),
            (None, Some(second)) => Some(second),
            (None, None) => None,
        }
    }

    fn owns_equation(&self, equation: usize) -> bool {
        self.first.owns_equation(equation) || self.second.owns_equation(equation)
    }

    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        // ponytail: skip zero blocks; non-owning leaves return [0; 3] by contract.
        if !self.second.owns_equation(equation) {
            return self.first.tensor_residual(ctx, state, equation, q);
        }
        if !self.first.owns_equation(equation) {
            return self.second.tensor_residual(ctx, state, equation, q);
        }
        let first = self.first.tensor_residual(ctx, state, equation, q);
        let second = self.second.tensor_residual(ctx, state, equation, q);
        [
            first[0] + second[0],
            first[1] + second[1],
            first[2] + second[2],
        ]
    }

    fn tensor_jacobian_action(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        // ponytail: skip zero blocks; non-owning leaves return [0; 3] by contract.
        if !self.second.owns_equation(equation) {
            return self
                .first
                .tensor_jacobian_action(ctx, state, direction, equation, q);
        }
        if !self.first.owns_equation(equation) {
            return self
                .second
                .tensor_jacobian_action(ctx, state, direction, equation, q);
        }
        let first = self
            .first
            .tensor_jacobian_action(ctx, state, direction, equation, q);
        let second = self
            .second
            .tensor_jacobian_action(ctx, state, direction, equation, q);
        [
            first[0] + second[0],
            first[1] + second[1],
            first[2] + second[2],
        ]
    }
}

/// Fuse tensor kernels into one statically dispatched sum.
///
/// `fuse_tensor_kernels!(a, b, c)` expands to the equivalent chained
/// `TensorResidualKernelSum::from_kernel(a).with(b).with(c)`, so the fused
/// pointwise evaluation stays statically dispatched and inlinable. Field
/// count/name checks still apply through the `Sum` implementation.
///
/// A proc macro is deliberately not used here: this is pure syntactic sugar
/// over the existing builder, with no new crate, dependencies, or generated
/// code to debug.
#[macro_export]
macro_rules! fuse_tensor_kernels {
    () => {
        compile_error!("fuse_tensor_kernels! requires at least one kernel")
    };
    ($first:expr) => {
        $crate::TensorResidualKernelSum::from_kernel($first)
    };
    ($first:expr, $($rest:expr),+ $(,)?) => {
        $crate::TensorResidualKernelSum::from_kernel($first)
            $(.with($rest))+
    };
}
