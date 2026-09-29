//! Statically dispatched sums of tensor residual kernels.
//!
//! [`TensorResidualKernelSum`] chains kernels with `.with()` while
//! keeping pointwise evaluation statically dispatched; the
//! [`fuse_tensor_kernels`] macro is sugar over its builder.
use super::traits::TensorResidualKernel;
use crate::common::{LaneState, Lanes, TensorCtx, LANES};

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
    #[inline]
    fn nfields(&self) -> usize {
        self.first.nfields()
    }

    #[inline]
    fn owns_equation(&self, equation: usize) -> bool {
        self.first.owns_equation(equation)
    }

    #[inline]
    fn field_names(&self) -> Option<Vec<String>> {
        self.first.field_names()
    }

    /// Lane-packed residual forwarding to the single child.
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
        self.first
            .tensor_residual(ctxs, state, equation, q, f0, f1x, f1y)
    }

    /// Lane-packed Jacobian action forwarding to the single child.
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
        self.first
            .tensor_jacobian_action(ctxs, state, direction, equation, q, f0, f1x, f1y)
    }
}

impl<const GDIM: usize, A, B> TensorResidualKernel<GDIM> for TensorResidualKernelSum<A, B>
where
    A: TensorResidualKernel<GDIM>,
    B: TensorResidualKernel<GDIM>,
{
    #[inline]
    fn nfields(&self) -> usize {
        assert_eq!(
            self.first.nfields(),
            self.second.nfields(),
            "tensor residual kernel sum field count mismatch"
        );
        self.first.nfields()
    }

    #[inline]
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

    #[inline]
    fn owns_equation(&self, equation: usize) -> bool {
        self.first.owns_equation(equation) || self.second.owns_equation(equation)
    }

    /// Lane-packed residual mirroring the scalar `owns_equation` skips.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed solution.
    /// * `equation` - output equation shared by all lanes.
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0`/`f1x`/`f1y` - lane output slots. Overwritten; when both
    ///   children own the equation each lane holds `first[l] + second[l]`.
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
        if !self.second.owns_equation(equation) {
            self.first
                .tensor_residual(ctxs, state, equation, q, f0, f1x, f1y);
            return;
        }
        if !self.first.owns_equation(equation) {
            self.second
                .tensor_residual(ctxs, state, equation, q, f0, f1x, f1y);
            return;
        }
        let mut a0 = [0.0; LANES];
        let mut a1x = [0.0; LANES];
        let mut a1y = [0.0; LANES];
        let mut b0 = [0.0; LANES];
        let mut b1x = [0.0; LANES];
        let mut b1y = [0.0; LANES];
        self.first
            .tensor_residual(ctxs, state, equation, q, &mut a0, &mut a1x, &mut a1y);
        self.second
            .tensor_residual(ctxs, state, equation, q, &mut b0, &mut b1x, &mut b1y);
        for l in 0..LANES {
            f0[l] = a0[l] + b0[l];
            f1x[l] = a1x[l] + b1x[l];
            f1y[l] = a1y[l] + b1y[l];
        }
    }

    /// Lane-packed Jacobian action mirroring the scalar `owns_equation` skips.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed linearization point.
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - output equation shared by all lanes.
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0`/`f1x`/`f1y` - lane output slots. Overwritten; when both
    ///   children own the equation each lane holds `first[l] + second[l]`.
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
        if !self.second.owns_equation(equation) {
            self.first
                .tensor_jacobian_action(ctxs, state, direction, equation, q, f0, f1x, f1y);
            return;
        }
        if !self.first.owns_equation(equation) {
            self.second
                .tensor_jacobian_action(ctxs, state, direction, equation, q, f0, f1x, f1y);
            return;
        }
        let mut a0 = [0.0; LANES];
        let mut a1x = [0.0; LANES];
        let mut a1y = [0.0; LANES];
        let mut b0 = [0.0; LANES];
        let mut b1x = [0.0; LANES];
        let mut b1y = [0.0; LANES];
        self.first.tensor_jacobian_action(
            ctxs, state, direction, equation, q, &mut a0, &mut a1x, &mut a1y,
        );
        self.second.tensor_jacobian_action(
            ctxs, state, direction, equation, q, &mut b0, &mut b1x, &mut b1y,
        );
        for l in 0..LANES {
            f0[l] = a0[l] + b0[l];
            f1x[l] = a1x[l] + b1x[l];
            f1y[l] = a1y[l] + b1y[l];
        }
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
