//! Sum-factorized bilinear-column assembly loops (crate-internal).
//!
//! Drives [`BilinearForm::tensor_bilinear`] pointwise physics over GLL cells
//! in 1D and 2D. Used by the bilinear assembly paths; volume residuals and
//! Jacobian actions of the residual operators use the lane-packed
//! [`TensorResidualKernel`](crate::kernels::common::TensorResidualKernel)
//! batch pipeline instead.
use super::traits::BilinearForm;
use crate::common::{CellState, TensorCtx};
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
