//! Shared kernel traits and default local assembly loops.

use crate::common::{CellState, FacetCtx, LocalCtx, TensorCtx, TensorFacetCtx};
use std::collections::HashMap;
use std::sync::Arc;

/// Per-cell bilinear-form kernel.
pub trait BilinearForm {
    /// Number of scalar equation/unknown fields in this form.
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the form.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    /// Bare integrand for one equation/unknown block and test/trial pair.
    fn integrand(
        &self,
        ctx: &LocalCtx,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64;

    /// Whether this form implements tensor-product bilinear evaluation in its
    /// supported geometric dimension.
    fn supports_tensor_bilinear(&self) -> bool {
        false
    }

    /// Whether this form implements tensor-product bilinear evaluation on an
    /// interval. This is separate from the 2D capability because a kernel may
    /// support only one geometric dimension.
    fn supports_tensor_bilinear_1d(&self) -> bool {
        false
    }

    /// Return the pointwise action of one trial basis function.
    ///
    /// The returned values are `(f0, f1_x, f1_y)` for
    /// `f0 * test + f1 . grad(test)`. In 1D, only `f0` and `f1_x` are used.
    /// `trial_value` and `trial_grad` are the value and physical gradient of
    /// the trial basis at `q`.
    fn tensor_bilinear(
        &self,
        _ctx: &TensorCtx<'_>,
        _equation: usize,
        _unknown: usize,
        _q: usize,
        _trial_value: f64,
        _trial_grad: [f64; 2],
    ) -> [f64; 3] {
        unreachable!("bilinear form does not support tensor-product evaluation")
    }

    /// Assemble a field-major local block matrix.
    fn assemble_local(&self, ctx: &LocalCtx, out: &mut [f64]) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        let local_size = nf * n;
        assert!(nf > 0, "bilinear form must contain at least one field");
        assert_eq!(
            out.len(),
            local_size * local_size,
            "local matrix size mismatch"
        );
        for equation in 0..nf {
            for unknown in 0..nf {
                for ti in 0..n {
                    for si in 0..n {
                        let mut acc = 0.0;
                        for q in 0..ctx.npts {
                            acc += ctx.wts[q]
                                * ctx.jdets[q]
                                * self.integrand(ctx, equation, unknown, q, ti, si);
                        }
                        let row = equation * n + ti;
                        let col = unknown * n + si;
                        out[row * local_size + col] = acc;
                    }
                }
            }
        }
    }
}

/// Per-cell linear-form kernel (RHS contribution).
pub trait LinearForm {
    /// Number of scalar equation fields in this form.
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the form.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    /// Bare integrand for one equation and test dof.
    fn integrand(&self, ctx: &LocalCtx, equation: usize, q: usize, test_i: usize) -> f64;

    /// Assemble a field-major local RHS.
    fn assemble_local_rhs(&self, ctx: &LocalCtx, out: &mut [f64]) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        assert!(nf > 0, "linear form must contain at least one field");
        assert_eq!(out.len(), nf * n, "local RHS size mismatch");
        for equation in 0..nf {
            for ti in 0..n {
                let mut acc = 0.0;
                for q in 0..ctx.npts {
                    acc += ctx.wts[q] * ctx.jdets[q] * self.integrand(ctx, equation, q, ti);
                }
                out[equation * n + ti] = acc;
            }
        }
    }
}

/// State-aware cell kernel for residual and Jacobian assembly.
pub trait ResidualKernel {
    /// Number of scalar PDE fields/equations in this kernel.
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the kernel.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    /// Number of compact state fields consumed by this term.
    fn input_nfields(&self) -> usize {
        self.nfields()
    }

    /// Number of compact residual fields produced by this term.
    fn output_nfields(&self) -> usize {
        self.nfields()
    }

    /// Optional ordered names for compact state fields.
    fn input_field_names(&self) -> Option<Vec<String>> {
        self.field_names()
    }

    /// Optional ordered names for compact residual fields.
    fn output_field_names(&self) -> Option<Vec<String>> {
        self.field_names()
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64;

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64;

    fn assemble_local_residual(&self, ctx: &LocalCtx, state: &CellState, out: &mut [f64]) {
        let ni = self.input_nfields();
        let no = self.output_nfields();
        let n = ctx.ndofs;
        assert_eq!(state.nfields, ni, "kernel/state field count mismatch");
        assert_eq!(out.len(), no * n, "local residual size mismatch");
        for equation in 0..no {
            for ti in 0..n {
                let mut acc = 0.0;
                for q in 0..ctx.npts {
                    acc += ctx.wts[q]
                        * ctx.jdets[q]
                        * self.residual_integrand(ctx, state, equation, q, ti);
                }
                out[equation * n + ti] = acc;
            }
        }
    }

    fn assemble_local_jacobian(&self, ctx: &LocalCtx, state: &CellState, out: &mut [f64]) {
        let ni = self.input_nfields();
        let no = self.output_nfields();
        let n = ctx.ndofs;
        let row_size = no * n;
        let col_size = ni * n;
        assert_eq!(state.nfields, ni, "kernel/state field count mismatch");
        assert_eq!(
            out.len(),
            row_size * col_size,
            "local Jacobian size mismatch"
        );
        for equation in 0..no {
            for unknown in 0..ni {
                for ti in 0..n {
                    for si in 0..n {
                        let mut acc = 0.0;
                        for q in 0..ctx.npts {
                            acc += ctx.wts[q]
                                * ctx.jdets[q]
                                * self.jacobian_integrand(ctx, state, equation, unknown, q, ti, si);
                        }
                        let row = equation * n + ti;
                        let col = unknown * n + si;
                        out[row * col_size + col] = acc;
                    }
                }
            }
        }
    }

    fn apply_local_jacobian(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        direction: &[f64],
        out: &mut [f64],
    ) {
        let ni = self.input_nfields();
        let no = self.output_nfields();
        let n = ctx.ndofs;
        let input_size = ni * n;
        let output_size = no * n;
        assert_eq!(state.nfields, ni, "kernel/state field count mismatch");
        assert_eq!(direction.len(), input_size, "local direction size mismatch");
        assert_eq!(
            out.len(),
            output_size,
            "local Jacobian action size mismatch"
        );
        for equation in 0..no {
            for ti in 0..n {
                let mut acc = 0.0;
                for unknown in 0..ni {
                    for si in 0..n {
                        for q in 0..ctx.npts {
                            acc += ctx.wts[q]
                                * ctx.jdets[q]
                                * self.jacobian_integrand(ctx, state, equation, unknown, q, ti, si)
                                * direction[unknown * n + si];
                        }
                    }
                }
                out[equation * n + ti] = acc;
            }
        }
    }
}

/// Statically dispatched tensor-product residual kernel.
///
/// `GDIM` is part of the type, so tensor assembly never needs a capability
/// query to select its volume path.
pub trait TensorResidualKernel<const GDIM: usize>: Send + Sync {
    fn nfields(&self) -> usize {
        1
    }
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }
    fn input_nfields(&self) -> usize {
        self.nfields()
    }
    fn output_nfields(&self) -> usize {
        self.nfields()
    }
    fn input_field_names(&self) -> Option<Vec<String>> {
        self.field_names()
    }
    fn output_field_names(&self) -> Option<Vec<String>> {
        self.field_names()
    }
    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3];
    fn tensor_jacobian_action(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3];
}

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

    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
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

/// Assemble one 1D tensor-product cell from pointwise weak-form fluxes.
pub(crate) fn assemble_tensor_residual_1d<K: TensorResidualKernel<1>>(
    kernel: &K,
    ctx: &TensorCtx<'_>,
    state: &CellState<'_>,
    out: &mut [f64],
) {
    let ni = kernel.input_nfields();
    let no = kernel.output_nfields();
    let n = ctx.n1d;
    assert_eq!(
        ctx.geometric_dimension(),
        1,
        "1D tensor context requires gdim == 1"
    );
    assert_eq!(state.nfields, ni, "kernel/state field count mismatch");
    assert_eq!(
        state.npts, ctx.npts,
        "tensor state/context point count mismatch"
    );
    assert_eq!(out.len(), no * n, "local 1D tensor residual size mismatch");
    out.fill(0.0);

    for equation in 0..no {
        for q in 0..n {
            let [f0, f1, _] = kernel.tensor_residual(ctx, state, equation, q);
            let weight = ctx.wdet[q];
            let local = ctx.q_to_local[q];
            out[equation * n + local] += weight * f0;
            let reference_flux = ctx.jinv[q] * f1;
            for a in 0..n {
                let test = ctx.q_to_local[a];
                out[equation * n + test] +=
                    weight * ctx.differentiation[q * n + a] * reference_flux;
            }
        }
    }
}

/// Assemble one tensor-product cell from pointwise weak-form fluxes.
pub(crate) fn assemble_tensor_residual<K: TensorResidualKernel<2>>(
    kernel: &K,
    ctx: &TensorCtx<'_>,
    state: &CellState<'_>,
    out: &mut [f64],
) {
    let ni = kernel.input_nfields();
    let no = kernel.output_nfields();
    let n = ctx.n1d * ctx.n1d;
    assert_eq!(state.nfields, ni, "kernel/state field count mismatch");
    assert_eq!(
        state.npts, ctx.npts,
        "tensor state/context point count mismatch"
    );
    assert_eq!(out.len(), no * n, "local tensor residual size mismatch");
    out.fill(0.0);

    for equation in 0..no {
        for j in 0..ctx.n1d {
            for i in 0..ctx.n1d {
                let q = j * ctx.n1d + i;
                let [f0, f1_x, f1_y] = kernel.tensor_residual(ctx, state, equation, q);
                let weight = ctx.wdet[q];
                let local = ctx.q_to_local[q];
                out[equation * n + local] += weight * f0;

                let jinv = &ctx.jinv[q * 4..q * 4 + 4];
                let reference_x = jinv[0] * f1_x + jinv[1] * f1_y;
                let reference_y = jinv[2] * f1_x + jinv[3] * f1_y;
                for a in 0..ctx.n1d {
                    let x_local = ctx.q_to_local[j * ctx.n1d + a];
                    out[equation * n + x_local] +=
                        weight * ctx.differentiation[i * ctx.n1d + a] * reference_x;
                    let y_local = ctx.q_to_local[a * ctx.n1d + i];
                    out[equation * n + y_local] +=
                        weight * ctx.differentiation[j * ctx.n1d + a] * reference_y;
                }
            }
        }
    }
}

/// Apply one local trial column of a 1D tensor-product bilinear form.
pub(crate) fn apply_tensor_bilinear_column_1d<K: BilinearForm>(
    kernel: &K,
    ctx: &TensorCtx<'_>,
    unknown: usize,
    trial: &CellState<'_>,
    out: &mut [f64],
) {
    let nf = kernel.nfields();
    let n = ctx.n1d;
    assert_eq!(
        ctx.geometric_dimension(),
        1,
        "1D tensor context requires gdim == 1"
    );
    assert_eq!(trial.nfields, nf, "kernel/trial field count mismatch");
    assert_eq!(
        trial.npts, ctx.npts,
        "tensor trial/context point count mismatch"
    );
    assert_eq!(
        out.len(),
        nf * n,
        "local 1D tensor bilinear action size mismatch"
    );
    out.fill(0.0);

    for equation in 0..nf {
        for q in 0..n {
            let contribution = kernel.tensor_bilinear(
                ctx,
                equation,
                unknown,
                q,
                trial.value(unknown, q),
                [trial.grad(unknown, q, 0), 0.0],
            );
            let weight = ctx.wdet[q];
            let local = ctx.q_to_local[q];
            out[equation * n + local] += weight * contribution[0];
            let reference_flux = ctx.jinv[q] * contribution[1];
            for a in 0..n {
                let test = ctx.q_to_local[a];
                out[equation * n + test] +=
                    weight * ctx.differentiation[q * n + a] * reference_flux;
            }
        }
    }
}

/// Apply one local trial column of a tensor-product bilinear form.
pub(crate) fn apply_tensor_bilinear_column<K: BilinearForm>(
    kernel: &K,
    ctx: &TensorCtx<'_>,
    unknown: usize,
    trial: &CellState<'_>,
    out: &mut [f64],
) {
    let nf = kernel.nfields();
    let n = ctx.n1d * ctx.n1d;
    assert_eq!(trial.nfields, nf, "kernel/trial field count mismatch");
    assert_eq!(
        trial.npts, ctx.npts,
        "tensor trial/context point count mismatch"
    );
    assert_eq!(
        out.len(),
        nf * n,
        "local tensor bilinear action size mismatch"
    );
    out.fill(0.0);

    for equation in 0..nf {
        for j in 0..ctx.n1d {
            for i in 0..ctx.n1d {
                let q = j * ctx.n1d + i;
                let contribution = kernel.tensor_bilinear(
                    ctx,
                    equation,
                    unknown,
                    q,
                    trial.value(unknown, q),
                    [trial.grad(unknown, q, 0), trial.grad(unknown, q, 1)],
                );
                let weight = ctx.wdet[q];
                let local = ctx.q_to_local[q];
                out[equation * n + local] += weight * contribution[0];

                let jinv = &ctx.jinv[q * 4..q * 4 + 4];
                let reference_x = jinv[0] * contribution[1] + jinv[1] * contribution[2];
                let reference_y = jinv[2] * contribution[1] + jinv[3] * contribution[2];
                for a in 0..ctx.n1d {
                    let x_local = ctx.q_to_local[j * ctx.n1d + a];
                    out[equation * n + x_local] +=
                        weight * ctx.differentiation[i * ctx.n1d + a] * reference_x;
                    let y_local = ctx.q_to_local[a * ctx.n1d + i];
                    out[equation * n + y_local] +=
                        weight * ctx.differentiation[j * ctx.n1d + a] * reference_y;
                }
            }
        }
    }
}

/// Apply a 1D tensor-product pointwise Jacobian action on one cell.
pub(crate) fn apply_tensor_jacobian_1d<K: TensorResidualKernel<1>>(
    kernel: &K,
    ctx: &TensorCtx<'_>,
    state: &CellState<'_>,
    direction: &CellState<'_>,
    out: &mut [f64],
) {
    let ni = kernel.input_nfields();
    let no = kernel.output_nfields();
    let n = ctx.n1d;
    assert_eq!(
        ctx.geometric_dimension(),
        1,
        "1D tensor context requires gdim == 1"
    );
    assert_eq!(state.nfields, ni, "kernel/state field count mismatch");
    assert_eq!(
        direction.nfields, ni,
        "kernel/direction field count mismatch"
    );
    assert_eq!(
        state.npts, ctx.npts,
        "tensor state/context point count mismatch"
    );
    assert_eq!(
        direction.npts, ctx.npts,
        "tensor direction/context point count mismatch"
    );
    assert_eq!(
        out.len(),
        no * n,
        "local 1D tensor Jacobian action size mismatch"
    );
    out.fill(0.0);

    for equation in 0..no {
        for q in 0..n {
            let [f0, f1, _] = kernel.tensor_jacobian_action(ctx, state, direction, equation, q);
            let weight = ctx.wdet[q];
            let local = ctx.q_to_local[q];
            out[equation * n + local] += weight * f0;
            let reference_flux = ctx.jinv[q] * f1;
            for a in 0..n {
                let test = ctx.q_to_local[a];
                out[equation * n + test] +=
                    weight * ctx.differentiation[q * n + a] * reference_flux;
            }
        }
    }
}

/// Apply a tensor-product pointwise Jacobian action on one cell.
pub(crate) fn apply_tensor_jacobian<K: TensorResidualKernel<2>>(
    kernel: &K,
    ctx: &TensorCtx<'_>,
    state: &CellState<'_>,
    direction: &CellState<'_>,
    out: &mut [f64],
) {
    let ni = kernel.input_nfields();
    let no = kernel.output_nfields();
    let n = ctx.n1d * ctx.n1d;
    assert_eq!(state.nfields, ni, "kernel/state field count mismatch");
    assert_eq!(
        direction.nfields, ni,
        "kernel/direction field count mismatch"
    );
    assert_eq!(
        state.npts, ctx.npts,
        "tensor state/context point count mismatch"
    );
    assert_eq!(
        direction.npts, ctx.npts,
        "tensor direction/context point count mismatch"
    );
    assert_eq!(
        out.len(),
        no * n,
        "local tensor Jacobian action size mismatch"
    );
    out.fill(0.0);

    for equation in 0..no {
        for j in 0..ctx.n1d {
            for i in 0..ctx.n1d {
                let q = j * ctx.n1d + i;
                let [f0, f1_x, f1_y] =
                    kernel.tensor_jacobian_action(ctx, state, direction, equation, q);
                let weight = ctx.wdet[q];
                let local = ctx.q_to_local[q];
                out[equation * n + local] += weight * f0;

                let jinv = &ctx.jinv[q * 4..q * 4 + 4];
                let reference_x = jinv[0] * f1_x + jinv[1] * f1_y;
                let reference_y = jinv[2] * f1_x + jinv[3] * f1_y;
                for a in 0..ctx.n1d {
                    let x_local = ctx.q_to_local[j * ctx.n1d + a];
                    out[equation * n + x_local] +=
                        weight * ctx.differentiation[i * ctx.n1d + a] * reference_x;
                    let y_local = ctx.q_to_local[a * ctx.n1d + i];
                    out[equation * n + y_local] +=
                        weight * ctx.differentiation[j * ctx.n1d + a] * reference_y;
                }
            }
        }
    }
}

/// Additive composition of state-aware cell kernels.
///
/// Each child must describe the same ordered system fields. The parent SEM
/// assembly still interpolates the state and scatters the local result once;
/// only the pointwise residual or Jacobian integrands are summed here.
pub struct ResidualKernelSum<'a> {
    kernels: Vec<Box<dyn ResidualKernel + Send + Sync + 'a>>,
    nfields: usize,
    field_names: Option<Vec<String>>,
}

impl<'a> ResidualKernelSum<'a> {
    /// Build a sum from heterogeneous residual kernels.
    pub fn new(kernels: Vec<Box<dyn ResidualKernel + Send + Sync + 'a>>) -> Self {
        let mut kernels = kernels.into_iter();
        let first = kernels
            .next()
            .expect("residual kernel sum must contain at least one kernel");
        let nfields = first.nfields();
        assert!(
            nfields > 0,
            "residual kernel sum must contain at least one field"
        );
        let mut field_names = first.field_names();
        Self::validate_field_names(&field_names, nfields);

        let mut sum = Self {
            kernels: vec![first],
            nfields,
            field_names: field_names.take(),
        };
        for kernel in kernels {
            sum.push(kernel);
        }
        sum
    }

    /// Start a sum with one concrete kernel.
    pub fn from_kernel<K>(kernel: K) -> Self
    where
        K: ResidualKernel + Send + Sync + 'a,
    {
        let kernel: Box<dyn ResidualKernel + Send + Sync + 'a> = Box::new(kernel);
        Self::new(vec![kernel])
    }

    /// Add one concrete kernel to this sum.
    pub fn with<K>(mut self, kernel: K) -> Self
    where
        K: ResidualKernel + Send + Sync + 'a,
    {
        let kernel: Box<dyn ResidualKernel + Send + Sync + 'a> = Box::new(kernel);
        self.push(kernel);
        self
    }

    fn push(&mut self, kernel: Box<dyn ResidualKernel + Send + Sync + 'a>) {
        assert_eq!(
            kernel.nfields(),
            self.nfields,
            "residual kernel sum field count mismatch"
        );
        let names = kernel.field_names();
        Self::validate_field_names(&names, self.nfields);
        match (&mut self.field_names, names) {
            (Some(expected), Some(actual)) => assert_eq!(
                *expected, actual,
                "residual kernel sum field names/order mismatch"
            ),
            (None, Some(actual)) => self.field_names = Some(actual),
            _ => {}
        }
        self.kernels.push(kernel);
    }

    fn validate_field_names(names: &Option<Vec<String>>, nfields: usize) {
        if let Some(names) = names {
            assert_eq!(
                names.len(),
                nfields,
                "residual kernel field-name count does not match field count"
            );
        }
    }
}

impl ResidualKernel for ResidualKernelSum<'_> {
    fn nfields(&self) -> usize {
        self.nfields
    }

    fn field_names(&self) -> Option<Vec<String>> {
        self.field_names.clone()
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        self.kernels
            .iter()
            .map(|kernel| kernel.residual_integrand(ctx, state, equation, q, test_i))
            .sum()
    }

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        self.kernels
            .iter()
            .map(|kernel| {
                kernel.jacobian_integrand(ctx, state, equation, unknown, q, test_i, trial_i)
            })
            .sum()
    }
}

/// Named composition for terms with different local field selections.
///
/// The union state is interpolated once by the SEM layer. Each child receives
/// a zero-copy local view through `CellState::field_indices`, so existing
/// compact kernels keep their original positional field contracts.
pub struct ResidualKernelSet<'a> {
    kernels: Vec<Box<dyn ResidualKernel + Send + Sync + 'a>>,
    fields: Vec<String>,
    input_maps: Vec<Vec<usize>>,
    output_maps: Vec<Vec<usize>>,
}

impl<'a> ResidualKernelSet<'a> {
    pub fn from_kernel<K>(kernel: K) -> Self
    where
        K: ResidualKernel + Send + Sync + 'a,
    {
        Self::new(vec![Box::new(kernel)])
    }

    pub fn new(kernels: Vec<Box<dyn ResidualKernel + Send + Sync + 'a>>) -> Self {
        assert!(
            !kernels.is_empty(),
            "residual kernel set must contain a kernel"
        );
        let mut set = Self {
            kernels: Vec::new(),
            fields: Vec::new(),
            input_maps: Vec::new(),
            output_maps: Vec::new(),
        };
        for kernel in kernels {
            set.push(kernel);
        }
        set
    }

    pub fn with<K>(mut self, kernel: K) -> Self
    where
        K: ResidualKernel + Send + Sync + 'a,
    {
        self.push(Box::new(kernel));
        self
    }

    fn push(&mut self, kernel: Box<dyn ResidualKernel + Send + Sync + 'a>) {
        let input_names = kernel
            .input_field_names()
            .expect("heterogeneous residual terms require named input fields");
        let output_names = kernel
            .output_field_names()
            .expect("heterogeneous residual terms require named output fields");
        assert_eq!(input_names.len(), kernel.input_nfields());
        assert_eq!(output_names.len(), kernel.output_nfields());
        let input_maps = input_names
            .iter()
            .map(|name| self.union_field(name))
            .collect();
        let output_maps = output_names
            .iter()
            .map(|name| self.union_field(name))
            .collect();
        self.kernels.push(kernel);
        self.input_maps.push(input_maps);
        self.output_maps.push(output_maps);
    }

    fn union_field(&mut self, name: &str) -> usize {
        if let Some(index) = self.fields.iter().position(|field| field == name) {
            index
        } else {
            let index = self.fields.len();
            self.fields.push(name.to_owned());
            index
        }
    }

    fn child_state<'s>(state: &'s CellState<'s>, map: &'s [usize]) -> CellState<'s> {
        CellState {
            nfields: map.len(),
            npts: state.npts,
            gdim: state.gdim,
            values: state.values,
            grads: state.grads,
            field_indices: map,
        }
    }
}

impl ResidualKernel for ResidualKernelSet<'_> {
    fn nfields(&self) -> usize {
        self.fields.len()
    }

    fn field_names(&self) -> Option<Vec<String>> {
        Some(self.fields.clone())
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        self.kernels
            .iter()
            .zip(&self.output_maps)
            .zip(&self.input_maps)
            .map(|((kernel, output_map), input_map)| {
                output_map
                    .iter()
                    .position(|&field| field == equation)
                    .map_or(0.0, |local_equation| {
                        let child = Self::child_state(state, input_map);
                        kernel.residual_integrand(ctx, &child, local_equation, q, test_i)
                    })
            })
            .sum()
    }

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        self.kernels
            .iter()
            .zip(&self.output_maps)
            .zip(&self.input_maps)
            .map(|((kernel, output_map), input_map)| {
                let Some(local_equation) = output_map.iter().position(|&field| field == equation)
                else {
                    return 0.0;
                };
                let Some(local_unknown) = input_map.iter().position(|&field| field == unknown)
                else {
                    return 0.0;
                };
                let child = Self::child_state(state, input_map);
                kernel.jacobian_integrand(
                    ctx,
                    &child,
                    local_equation,
                    local_unknown,
                    q,
                    test_i,
                    trial_i,
                )
            })
            .sum()
    }
}

/// Named tensor composition for terms with different local field selections.
pub struct TensorResidualKernelSet<'a, const GDIM: usize> {
    kernels: Vec<Box<dyn TensorResidualKernel<GDIM> + 'a>>,
    fields: Vec<String>,
    input_maps: Vec<Vec<usize>>,
    output_maps: Vec<Vec<usize>>,
}

impl<'a, const GDIM: usize> TensorResidualKernelSet<'a, GDIM> {
    pub fn from_kernel<K>(kernel: K) -> Self
    where
        K: TensorResidualKernel<GDIM> + 'a,
    {
        Self::new(vec![Box::new(kernel)])
    }

    pub fn new(kernels: Vec<Box<dyn TensorResidualKernel<GDIM> + 'a>>) -> Self {
        assert!(
            !kernels.is_empty(),
            "tensor residual kernel set must contain a kernel"
        );
        let mut set = Self {
            kernels: Vec::new(),
            fields: Vec::new(),
            input_maps: Vec::new(),
            output_maps: Vec::new(),
        };
        for kernel in kernels {
            set.push(kernel);
        }
        set
    }

    pub fn with<K>(mut self, kernel: K) -> Self
    where
        K: TensorResidualKernel<GDIM> + 'a,
    {
        self.push(Box::new(kernel));
        self
    }

    fn push(&mut self, kernel: Box<dyn TensorResidualKernel<GDIM> + 'a>) {
        let input_names = kernel
            .input_field_names()
            .expect("heterogeneous tensor terms require named input fields");
        let output_names = kernel
            .output_field_names()
            .expect("heterogeneous tensor terms require named output fields");
        assert_eq!(input_names.len(), kernel.input_nfields());
        assert_eq!(output_names.len(), kernel.output_nfields());
        let input_maps = input_names
            .iter()
            .map(|name| self.union_field(name))
            .collect();
        let output_maps = output_names
            .iter()
            .map(|name| self.union_field(name))
            .collect();
        self.kernels.push(kernel);
        self.input_maps.push(input_maps);
        self.output_maps.push(output_maps);
    }

    fn union_field(&mut self, name: &str) -> usize {
        if let Some(index) = self.fields.iter().position(|field| field == name) {
            index
        } else {
            let index = self.fields.len();
            self.fields.push(name.to_owned());
            index
        }
    }
}

impl<const GDIM: usize> TensorResidualKernel<GDIM> for TensorResidualKernelSet<'_, GDIM> {
    fn nfields(&self) -> usize {
        self.fields.len()
    }

    fn field_names(&self) -> Option<Vec<String>> {
        Some(self.fields.clone())
    }

    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let mut result = [0.0; 3];
        for ((kernel, output_map), input_map) in self
            .kernels
            .iter()
            .zip(&self.output_maps)
            .zip(&self.input_maps)
        {
            let Some(local_equation) = output_map.iter().position(|&field| field == equation)
            else {
                continue;
            };
            let child = CellState {
                nfields: input_map.len(),
                npts: state.npts,
                gdim: state.gdim,
                values: state.values,
                grads: state.grads,
                field_indices: input_map,
            };
            let contribution = kernel.tensor_residual(ctx, &child, local_equation, q);
            result[0] += contribution[0];
            result[1] += contribution[1];
            result[2] += contribution[2];
        }
        result
    }

    fn tensor_jacobian_action(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let mut result = [0.0; 3];
        for ((kernel, output_map), input_map) in self
            .kernels
            .iter()
            .zip(&self.output_maps)
            .zip(&self.input_maps)
        {
            let Some(local_equation) = output_map.iter().position(|&field| field == equation)
            else {
                continue;
            };
            let child_state = CellState {
                nfields: input_map.len(),
                npts: state.npts,
                gdim: state.gdim,
                values: state.values,
                grads: state.grads,
                field_indices: input_map,
            };
            let child_direction = CellState {
                nfields: input_map.len(),
                npts: direction.npts,
                gdim: direction.gdim,
                values: direction.values,
                grads: direction.grads,
                field_indices: input_map,
            };
            let contribution = kernel.tensor_jacobian_action(
                ctx,
                &child_state,
                &child_direction,
                local_equation,
                q,
            );
            result[0] += contribution[0];
            result[1] += contribution[1];
            result[2] += contribution[2];
        }
        result
    }
}

/// Pointwise flux provider for one-dimensional conservation laws.
pub trait FluxKernel1D {
    fn nfields(&self) -> usize;

    /// Optional ordered names for fields whose meaning is part of the flux.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    fn flux(&self, ctx: &LocalCtx, state: &CellState, equation: usize, q: usize) -> f64;

    fn flux_jacobian(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
    ) -> f64;
}

/// Boundary integrator trait for Neumann and Robin forms.
pub trait BoundaryIntegrator {
    /// Number of scalar equation/unknown fields in this boundary form.
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the boundary form.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    fn integrand_rhs(&self, ctx: &FacetCtx, equation: usize, q: usize, test_i: usize) -> f64;

    fn integrand_mat(
        &self,
        _ctx: &FacetCtx,
        _equation: usize,
        _unknown: usize,
        _q: usize,
        _test_i: usize,
        _trial_i: usize,
    ) -> f64 {
        0.0
    }

    fn assemble_facet_rhs(&self, ctx: &FacetCtx, rhs: &mut [f64]) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        assert!(
            nf > 0,
            "boundary integrator must contain at least one field"
        );
        assert_eq!(rhs.len(), nf * n, "local boundary RHS size mismatch");
        for equation in 0..nf {
            for ti in 0..n {
                let mut acc = 0.0;
                for q in 0..ctx.npts {
                    acc +=
                        ctx.wts[q] * ctx.jfacet_det[q] * self.integrand_rhs(ctx, equation, q, ti);
                }
                rhs[equation * n + ti] = acc;
            }
        }
    }

    fn assemble_facet_mat(&self, ctx: &FacetCtx, mat: &mut [f64]) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        let local_size = nf * n;
        assert_eq!(
            mat.len(),
            local_size * local_size,
            "local boundary matrix size mismatch"
        );
        for equation in 0..nf {
            for unknown in 0..nf {
                for ti in 0..n {
                    for si in 0..n {
                        let mut acc = 0.0;
                        for q in 0..ctx.npts {
                            acc += ctx.wts[q]
                                * ctx.jfacet_det[q]
                                * self.integrand_mat(ctx, equation, unknown, q, ti, si);
                        }
                        let row = equation * n + ti;
                        let col = unknown * n + si;
                        mat[row * local_size + col] = acc;
                    }
                }
            }
        }
    }
}

/// State-aware nonlinear boundary form.
///
/// Unlike [`BoundaryIntegrator`], this form receives the interpolated state at
/// facet quadrature points and can therefore represent nonlinear conditions
/// such as backflow-stabilizing outflow traction.
pub trait StateBoundaryIntegrator: Send + Sync {
    /// Number of scalar equation/unknown fields in this boundary form.
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the form.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    fn input_nfields(&self) -> usize {
        self.nfields()
    }

    fn output_nfields(&self) -> usize {
        self.nfields()
    }

    fn input_field_names(&self) -> Option<Vec<String>> {
        self.field_names()
    }

    fn output_field_names(&self) -> Option<Vec<String>> {
        self.field_names()
    }

    fn residual_integrand(
        &self,
        ctx: &FacetCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64;

    fn jacobian_integrand(
        &self,
        ctx: &FacetCtx,
        state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64;

    fn apply_local_jacobian(
        &self,
        ctx: &FacetCtx,
        state: &CellState,
        direction: &[f64],
        out: &mut [f64],
    ) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        let local_size = nf * n;
        assert_eq!(
            direction.len(),
            local_size,
            "boundary direction size mismatch"
        );
        assert_eq!(out.len(), local_size, "boundary action size mismatch");
        for equation in 0..nf {
            for ti in 0..n {
                let mut acc = 0.0;
                for unknown in 0..nf {
                    for si in 0..n {
                        for q in 0..ctx.npts {
                            acc += ctx.wts[q]
                                * ctx.jfacet_det[q]
                                * self.jacobian_integrand(ctx, state, equation, unknown, q, ti, si)
                                * direction[unknown * n + si];
                        }
                    }
                }
                out[equation * n + ti] = acc;
            }
        }
    }

    fn assemble_local_residual(&self, ctx: &FacetCtx, state: &CellState, out: &mut [f64]) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        assert!(
            nf > 0,
            "state boundary form must contain at least one field"
        );
        assert_eq!(state.nfields, nf, "boundary/state field count mismatch");
        assert_eq!(out.len(), nf * n, "local boundary residual size mismatch");
        for equation in 0..nf {
            for ti in 0..n {
                let mut acc = 0.0;
                for q in 0..ctx.npts {
                    acc += ctx.wts[q]
                        * ctx.jfacet_det[q]
                        * self.residual_integrand(ctx, state, equation, q, ti);
                }
                out[equation * n + ti] = acc;
            }
        }
    }

    fn assemble_local_jacobian(&self, ctx: &FacetCtx, state: &CellState, out: &mut [f64]) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        let local_size = nf * n;
        assert_eq!(state.nfields, nf, "boundary/state field count mismatch");
        assert_eq!(
            out.len(),
            local_size * local_size,
            "local boundary Jacobian size mismatch"
        );
        for equation in 0..nf {
            for unknown in 0..nf {
                for ti in 0..n {
                    for si in 0..n {
                        let mut acc = 0.0;
                        for q in 0..ctx.npts {
                            acc += ctx.wts[q]
                                * ctx.jfacet_det[q]
                                * self.jacobian_integrand(ctx, state, equation, unknown, q, ti, si);
                        }
                        let row = equation * n + ti;
                        let col = unknown * n + si;
                        out[row * local_size + col] = acc;
                    }
                }
            }
        }
    }
}

/// Statically dispatched tensor-product state boundary kernel.
///
/// `GDIM` is part of the type so tensor boundary assembly can select this
/// interface without probing the weak [`StateBoundaryIntegrator`] API.
pub trait StateTensorBoundaryIntegrator<const GDIM: usize>: Send + Sync {
    /// Number of scalar equation/unknown fields in this boundary form.
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the form.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    fn input_nfields(&self) -> usize {
        self.nfields()
    }

    fn output_nfields(&self) -> usize {
        self.nfields()
    }

    fn input_field_names(&self) -> Option<Vec<String>> {
        self.field_names()
    }

    fn output_field_names(&self) -> Option<Vec<String>> {
        self.field_names()
    }

    /// Whether evaluating this boundary condition needs state gradients.
    fn tensor_requires_gradients(&self) -> bool {
        false
    }

    /// Return the pointwise trace residual for one equation.
    fn tensor_residual(
        &self,
        ctx: &TensorFacetCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> f64;

    /// Return the pointwise trace Jacobian action for one equation.
    fn tensor_jacobian_action(
        &self,
        ctx: &TensorFacetCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> f64;
}

/// Immutable state-dependent boundary terms selected by mesh entity index.
#[derive(Clone, Default)]
pub struct StateBoundaryTerms {
    default: Option<Arc<dyn StateBoundaryIntegrator>>,
    overrides: HashMap<usize, Arc<dyn StateBoundaryIntegrator>>,
}

impl StateBoundaryTerms {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_default<K>(mut self, kernel: K) -> Self
    where
        K: StateBoundaryIntegrator + 'static,
    {
        self.default = Some(Arc::new(kernel));
        self
    }

    pub fn with_entities<K, I>(mut self, entities: I, kernel: K) -> Self
    where
        K: StateBoundaryIntegrator + 'static,
        I: IntoIterator<Item = usize>,
    {
        let kernel: Arc<dyn StateBoundaryIntegrator> = Arc::new(kernel);
        for entity in entities {
            assert!(
                self.overrides.insert(entity, Arc::clone(&kernel)).is_none(),
                "state boundary entity configured more than once"
            );
        }
        self
    }

    pub(crate) fn kernel_for(&self, entity: usize) -> Option<&dyn StateBoundaryIntegrator> {
        self.overrides
            .get(&entity)
            .or(self.default.as_ref())
            .map(AsRef::as_ref)
    }
}

/// Immutable tensor boundary terms selected by mesh entity index.
///
/// This is deliberately separate from [`StateBoundaryTerms`]: weak boundary
/// kernels remain usable without also implementing a tensor boundary kernel.
#[derive(Clone, Default)]
pub struct StateTensorBoundaryTerms<const GDIM: usize> {
    default: Option<Arc<dyn StateTensorBoundaryIntegrator<GDIM>>>,
    overrides: HashMap<usize, Arc<dyn StateTensorBoundaryIntegrator<GDIM>>>,
}

impl<const GDIM: usize> StateTensorBoundaryTerms<GDIM> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_default<K>(mut self, kernel: K) -> Self
    where
        K: StateTensorBoundaryIntegrator<GDIM> + 'static,
    {
        self.default = Some(Arc::new(kernel));
        self
    }

    pub fn with_entities<K, I>(mut self, entities: I, kernel: K) -> Self
    where
        K: StateTensorBoundaryIntegrator<GDIM> + 'static,
        I: IntoIterator<Item = usize>,
    {
        let kernel: Arc<dyn StateTensorBoundaryIntegrator<GDIM>> = Arc::new(kernel);
        for entity in entities {
            assert!(
                self.overrides.insert(entity, Arc::clone(&kernel)).is_none(),
                "tensor state boundary entity configured more than once"
            );
        }
        self
    }

    pub(crate) fn kernel_for(
        &self,
        entity: usize,
    ) -> Option<&dyn StateTensorBoundaryIntegrator<GDIM>> {
        self.overrides
            .get(&entity)
            .or(self.default.as_ref())
            .map(AsRef::as_ref)
    }
}
