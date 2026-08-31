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

    /// Whether this kernel implements tensor-product residual evaluation in its
    /// supported geometric dimension.
    fn supports_tensor_residual(&self) -> bool {
        false
    }

    /// Whether this kernel implements tensor-product residual evaluation on an
    /// interval.
    fn supports_tensor_residual_1d(&self) -> bool {
        false
    }

    /// Whether this kernel implements tensor-product Jacobian actions in its
    /// supported geometric dimension.
    fn supports_tensor_jacobian(&self) -> bool {
        false
    }

    /// Whether this kernel implements tensor-product Jacobian actions on an
    /// interval.
    fn supports_tensor_jacobian_1d(&self) -> bool {
        false
    }

    /// Return the weak-form pointwise contribution for a tensor-product cell.
    ///
    /// The returned values are `(f0, f1_x, f1_y)` for
    /// `f0 * test + f1 . grad(test)`. In 1D, only `f0` and `f1_x` are used.
    /// Kernels that do not set
    /// [`Self::supports_tensor_residual`] do not call this method.
    fn tensor_residual(
        &self,
        _ctx: &TensorCtx<'_>,
        _state: &CellState<'_>,
        _equation: usize,
        _q: usize,
    ) -> [f64; 3] {
        unreachable!("kernel does not support tensor-product residuals")
    }

    /// Return the weak-form pointwise Jacobian action for a tensor-product cell.
    ///
    /// `direction` contains the pointwise value and gradient of the complete
    /// perturbation. Kernels that do not set [`Self::supports_tensor_jacobian`]
    /// do not call this method.
    fn tensor_jacobian_action(
        &self,
        _ctx: &TensorCtx<'_>,
        _state: &CellState<'_>,
        _direction: &CellState<'_>,
        _equation: usize,
        _q: usize,
    ) -> [f64; 3] {
        unreachable!("kernel does not support tensor-product Jacobian actions")
    }

    fn assemble_local_residual(&self, ctx: &LocalCtx, state: &CellState, out: &mut [f64]) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        assert_eq!(state.nfields, nf, "kernel/state field count mismatch");
        assert_eq!(out.len(), nf * n, "local residual size mismatch");
        for equation in 0..nf {
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
        let nf = self.nfields();
        let n = ctx.ndofs;
        let local_size = nf * n;
        assert_eq!(state.nfields, nf, "kernel/state field count mismatch");
        assert_eq!(
            out.len(),
            local_size * local_size,
            "local Jacobian size mismatch"
        );
        for equation in 0..nf {
            for unknown in 0..nf {
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
                        out[row * local_size + col] = acc;
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
        let nf = self.nfields();
        let n = ctx.ndofs;
        let local_size = nf * n;
        assert_eq!(state.nfields, nf, "kernel/state field count mismatch");
        assert_eq!(direction.len(), local_size, "local direction size mismatch");
        assert_eq!(out.len(), local_size, "local Jacobian action size mismatch");
        for equation in 0..nf {
            for ti in 0..n {
                let mut acc = 0.0;
                for unknown in 0..nf {
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

/// Assemble one 1D tensor-product cell from pointwise weak-form fluxes.
pub(crate) fn assemble_tensor_residual_1d<K: ResidualKernel>(
    kernel: &K,
    ctx: &TensorCtx<'_>,
    state: &CellState<'_>,
    out: &mut [f64],
) {
    let nf = kernel.nfields();
    let n = ctx.n1d;
    assert_eq!(
        ctx.geometric_dimension(),
        1,
        "1D tensor context requires gdim == 1"
    );
    assert_eq!(state.nfields, nf, "kernel/state field count mismatch");
    assert_eq!(
        state.npts, ctx.npts,
        "tensor state/context point count mismatch"
    );
    assert_eq!(out.len(), nf * n, "local 1D tensor residual size mismatch");
    out.fill(0.0);

    for equation in 0..nf {
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
pub(crate) fn assemble_tensor_residual<K: ResidualKernel>(
    kernel: &K,
    ctx: &TensorCtx<'_>,
    state: &CellState<'_>,
    out: &mut [f64],
) {
    let nf = kernel.nfields();
    let n = ctx.n1d * ctx.n1d;
    assert_eq!(state.nfields, nf, "kernel/state field count mismatch");
    assert_eq!(
        state.npts, ctx.npts,
        "tensor state/context point count mismatch"
    );
    assert_eq!(out.len(), nf * n, "local tensor residual size mismatch");
    out.fill(0.0);

    for equation in 0..nf {
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
pub(crate) fn apply_tensor_jacobian_1d<K: ResidualKernel>(
    kernel: &K,
    ctx: &TensorCtx<'_>,
    state: &CellState<'_>,
    direction: &CellState<'_>,
    out: &mut [f64],
) {
    let nf = kernel.nfields();
    let n = ctx.n1d;
    assert_eq!(
        ctx.geometric_dimension(),
        1,
        "1D tensor context requires gdim == 1"
    );
    assert_eq!(state.nfields, nf, "kernel/state field count mismatch");
    assert_eq!(
        direction.nfields, nf,
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
        nf * n,
        "local 1D tensor Jacobian action size mismatch"
    );
    out.fill(0.0);

    for equation in 0..nf {
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
pub(crate) fn apply_tensor_jacobian<K: ResidualKernel>(
    kernel: &K,
    ctx: &TensorCtx<'_>,
    state: &CellState<'_>,
    direction: &CellState<'_>,
    out: &mut [f64],
) {
    let nf = kernel.nfields();
    let n = ctx.n1d * ctx.n1d;
    assert_eq!(state.nfields, nf, "kernel/state field count mismatch");
    assert_eq!(
        direction.nfields, nf,
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
        nf * n,
        "local tensor Jacobian action size mismatch"
    );
    out.fill(0.0);

    for equation in 0..nf {
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

    fn supports_tensor_residual(&self) -> bool {
        self.kernels
            .iter()
            .all(|kernel| kernel.supports_tensor_residual())
    }

    fn supports_tensor_residual_1d(&self) -> bool {
        self.kernels
            .iter()
            .all(|kernel| kernel.supports_tensor_residual_1d())
    }

    fn supports_tensor_jacobian(&self) -> bool {
        self.kernels
            .iter()
            .all(|kernel| kernel.supports_tensor_jacobian())
    }

    fn supports_tensor_jacobian_1d(&self) -> bool {
        self.kernels
            .iter()
            .all(|kernel| kernel.supports_tensor_jacobian_1d())
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

    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let mut sum = [0.0; 3];
        for kernel in &self.kernels {
            let contribution = kernel.tensor_residual(ctx, state, equation, q);
            for (sum, contribution) in sum.iter_mut().zip(contribution) {
                *sum += contribution;
            }
        }
        sum
    }

    fn tensor_jacobian_action(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let mut sum = [0.0; 3];
        for kernel in &self.kernels {
            let contribution = kernel.tensor_jacobian_action(ctx, state, direction, equation, q);
            for (sum, contribution) in sum.iter_mut().zip(contribution) {
                *sum += contribution;
            }
        }
        sum
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

    /// Whether this boundary kernel supplies a sum-factorization-compatible
    /// pointwise residual action.
    fn supports_tensor_residual(&self) -> bool {
        false
    }

    /// Whether this boundary kernel supplies a sum-factorization-compatible
    /// pointwise Jacobian action.
    fn supports_tensor_jacobian(&self) -> bool {
        false
    }

    /// Whether the tensor boundary hooks need the state gradient field.
    ///
    /// Value-only conditions, such as advective and Dong outflow fluxes, can
    /// leave this disabled and avoid gathering unused cell gradients.
    fn tensor_requires_gradients(&self) -> bool {
        false
    }

    /// Return the pointwise trace residual for one equation.
    fn tensor_residual(
        &self,
        _ctx: &TensorFacetCtx<'_>,
        _state: &CellState<'_>,
        _equation: usize,
        _q: usize,
    ) -> f64 {
        unreachable!("state boundary kernel does not support tensor evaluation")
    }

    /// Return the pointwise trace Jacobian action for one equation.
    fn tensor_jacobian_action(
        &self,
        _ctx: &TensorFacetCtx<'_>,
        _state: &CellState<'_>,
        _direction: &CellState<'_>,
        _equation: usize,
        _q: usize,
    ) -> f64 {
        unreachable!("state boundary kernel does not support tensor evaluation")
    }

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
