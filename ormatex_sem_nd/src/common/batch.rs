//! SIMD-over-element batch helpers (libCEED-inspired E-vector batching).
//!
//! Irregular global restriction stays scalar; dense tensor interpolation,
//! geometry transforms and integration run across independent element lanes
//! with lane-contiguous scratch `[field/q/local][lane]`.
//!
//! `SIMD_CELL_WIDTH` is re-exported from `cell` for a single source of truth.

use pulp::{Arch, Simd, WithSimd};

use crate::common::cell::CellData;

/// Lane width shared by every batched helper in this module.
pub(crate) use super::cell::SIMD_CELL_WIDTH;

/// Fused multiply-add over contiguous lanes inside one pulp dispatch.
///
/// This is the only vectorized primitive the batch helpers need; calling it
/// directly (rather than `crate::simd::axpy`) keeps one `Arch` dispatch per
/// batch function instead of one per `(field, q, a)` contraction term.
#[inline(always)]
fn axpy_lanes<S: Simd>(simd: S, target: &mut [f64], factor: f64, source: &[f64]) {
    debug_assert_eq!(target.len(), source.len());
    let splat = simd.splat_f64s(factor);
    let (target, target_tail) = S::as_mut_simd_f64s(target);
    let (source, source_tail) = S::as_simd_f64s(source);
    for (target, source) in target.iter_mut().zip(source) {
        *target = simd.mul_add_f64s(splat, *source, *target);
    }
    for (target, source) in target_tail.iter_mut().zip(source_tail) {
        *target += factor * *source;
    }
}

/// 1D tensor interpolation across lanes: E-vector coefficients to quadrature
/// values and physical gradients.
///
/// Inputs are lane-contiguous coefficients `[(f*ndofs+d)*W+lane]`.
/// Outputs are lane-contiguous values/grads `[(f*npts+q)*W+lane]`.
/// `cell_indices[lane]` selects per-lane `jinv`; `q_to_local` from `tensor`.
///
/// `values` and `grads` must hold `nfields*npts*W` doubles; only the first
/// `nlanes` lanes of each row are written (`nlanes <= W`). Reference gradients
/// come from one 1D contraction per point; the cached per-lane inverse
/// Jacobian maps them to physical gradients afterwards.
pub(crate) fn interpolate_batch_1d(
    cell_data: &CellData,
    nfields: usize,
    coeffs: &[f64],
    values: &mut [f64],
    grads: &mut [f64],
    cell_indices: &[usize],
    nlanes: usize,
) {
    Arch::new().dispatch(InterpolateBatch1d {
        cell_data,
        nfields,
        coeffs,
        values,
        grads,
        cell_indices,
        nlanes,
    });
}

/// Single-dispatch 1D tensor interpolation (see [`interpolate_batch_1d`]).
struct InterpolateBatch1d<'a> {
    cell_data: &'a CellData,
    nfields: usize,
    coeffs: &'a [f64],
    values: &'a mut [f64],
    grads: &'a mut [f64],
    cell_indices: &'a [usize],
    nlanes: usize,
}

impl WithSimd for InterpolateBatch1d<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let w = SIMD_CELL_WIDTH;
        let tensor = self
            .cell_data
            .tensor
            .as_ref()
            .expect("tensor batch requires tensor data");
        let n1d = tensor.n1d;
        let npts = self.cell_data.npts;
        debug_assert_eq!(npts, n1d);
        for f in 0..self.nfields {
            // values[q][lane] = coeff[q_to_local[q]][lane]
            for q in 0..npts {
                let local = tensor.q_to_local[q];
                let src = &self.coeffs[(f * self.cell_data.ndofs + local) * w..(f * self.cell_data.ndofs + local) * w + self.nlanes];
                let dst = &mut self.values[(f * npts + q) * w..(f * npts + q) * w + self.nlanes];
                dst[..self.nlanes].copy_from_slice(&src[..self.nlanes]);
            }
            // grad[q][lane] = jinv[q,lane] * sum_a D[q,a]*value[a][lane].
            // GLL differentiation matrices are dense, so every term runs.
            for q in 0..n1d {
                let dst_off = (f * npts + q) * w;
                for lane in 0..self.nlanes {
                    self.grads[dst_off + lane] = 0.0;
                }
                for a in 0..n1d {
                    let factor = tensor.differentiation[q * n1d + a];
                    // ponytail: lane slices are contiguous; no temp copy needed.
                    axpy_lanes(
                        simd,
                        &mut self.grads[dst_off..dst_off + self.nlanes],
                        factor,
                        &self.values[(f * npts + a) * w..(f * npts + a) * w + self.nlanes],
                    );
                }
                // physical transform per lane (scalar, auto-vectorizable)
                for lane in 0..self.nlanes {
                    let jinv = self.cell_data.jinv_cache[self.cell_indices[lane] * npts + q];
                    self.grads[dst_off + lane] *= jinv;
                }
            }
        }
    }
}

/// 2D tensor interpolation across lanes: values plus both gradient components.
///
/// Layouts mirror the 1D version with `gdim=2` grads
/// `[(f*2+d)*npts+q)*W+lane]`. Reference `x`/`y` gradients use separate 1D
/// contractions over rows/columns followed by one per-lane `jinv` transform.
/// Buffer contract matches [`interpolate_batch_1d`] with `npts == n1d*n1d`.
pub(crate) fn interpolate_batch_2d(
    cell_data: &CellData,
    nfields: usize,
    coeffs: &[f64],
    values: &mut [f64],
    grads: &mut [f64],
    cell_indices: &[usize],
    nlanes: usize,
) {
    Arch::new().dispatch(InterpolateBatch2d {
        cell_data,
        nfields,
        coeffs,
        values,
        grads,
        cell_indices,
        nlanes,
    });
}

/// Single-dispatch 2D tensor interpolation (see [`interpolate_batch_2d`]).
struct InterpolateBatch2d<'a> {
    cell_data: &'a CellData,
    nfields: usize,
    coeffs: &'a [f64],
    values: &'a mut [f64],
    grads: &'a mut [f64],
    cell_indices: &'a [usize],
    nlanes: usize,
}

impl WithSimd for InterpolateBatch2d<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let w = SIMD_CELL_WIDTH;
        let tensor = self
            .cell_data
            .tensor
            .as_ref()
            .expect("tensor batch requires tensor data");
        let n1d = tensor.n1d;
        let npts = self.cell_data.npts;
        debug_assert_eq!(npts, n1d * n1d);
        for f in 0..self.nfields {
            for q in 0..npts {
                let local = tensor.q_to_local[q];
                self.values[(f * npts + q) * w..(f * npts + q) * w + self.nlanes]
                    .copy_from_slice(&self.coeffs[(f * self.cell_data.ndofs + local) * w..(f * self.cell_data.ndofs + local) * w + self.nlanes]);
            }
            // Reference grads via axpy over lanes; GLL rows are dense.
            for q in 0..npts {
                self.grads[(f * 2 * npts + q) * w..(f * 2 * npts + q) * w + self.nlanes].fill(0.0);
                self.grads[((f * 2 + 1) * npts + q) * w..((f * 2 + 1) * npts + q) * w + self.nlanes].fill(0.0);
            }
            for j in 0..n1d {
                for i in 0..n1d {
                    let q = j * n1d + i;
                    // gx[q] += sum_a D[i,a]*v[j,a]
                    for a in 0..n1d {
                        axpy_lanes(
                            simd,
                            &mut self.grads[(f * 2 * npts + q) * w..(f * 2 * npts + q) * w + self.nlanes],
                            tensor.differentiation[i * n1d + a],
                            &self.values[(f * npts + j * n1d + a) * w
                                ..(f * npts + j * n1d + a) * w + self.nlanes],
                        );
                        axpy_lanes(
                            simd,
                            &mut self.grads[((f * 2 + 1) * npts + q) * w..((f * 2 + 1) * npts + q) * w + self.nlanes],
                            tensor.differentiation[j * n1d + a],
                            &self.values[(f * npts + a * n1d + i) * w
                                ..(f * npts + a * n1d + i) * w + self.nlanes],
                        );
                    }
                }
            }
            // physical transform per (q,lane)
            for q in 0..npts {
                for lane in 0..self.nlanes {
                    let rx = self.grads[(f * 2 * npts + q) * w + lane];
                    let ry = self.grads[((f * 2 + 1) * npts + q) * w + lane];
                    let j = &self.cell_data.jinv_cache
                        [(self.cell_indices[lane] * npts + q) * 4..(self.cell_indices[lane] * npts + q + 1) * 4];
                    self.grads[(f * 2 * npts + q) * w + lane] = j[0] * rx + j[2] * ry;
                    self.grads[((f * 2 + 1) * npts + q) * w + lane] = j[1] * rx + j[3] * ry;
                }
            }
        }
    }
}

/// 1D batched integration: lane-packed fluxes -> lane-packed local actions.
///
/// `f0,f1`: `[(eq*npts+q)*W+lane]` pointwise fluxes; `out`:
/// `[(eq*ndofs+local)*W+lane]`, zeroed by the caller and accumulated into
/// (`+=`, never overwritten). Each point folds `wdet` and the per-lane `jinv`
/// into a reference flux, then applies the transposed 1D contraction.
/// `cell_indices` and `nlanes` follow [`interpolate_batch_1d`].
pub(crate) fn integrate_batch_1d(
    cell_data: &CellData,
    noutputs: usize,
    f0: &[f64],
    f1: &[f64],
    out: &mut [f64],
    cell_indices: &[usize],
    nlanes: usize,
) {
    Arch::new().dispatch(IntegrateBatch1d {
        cell_data,
        noutputs,
        f0,
        f1,
        out,
        cell_indices,
        nlanes,
    });
}

/// Single-dispatch 1D batched integration (see [`integrate_batch_1d`]).
struct IntegrateBatch1d<'a> {
    cell_data: &'a CellData,
    noutputs: usize,
    f0: &'a [f64],
    f1: &'a [f64],
    out: &'a mut [f64],
    cell_indices: &'a [usize],
    nlanes: usize,
}

impl WithSimd for IntegrateBatch1d<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let w = SIMD_CELL_WIDTH;
        let tensor = self.cell_data.tensor.as_ref().expect("tensor batch needs tensor");
        let n1d = tensor.n1d;
        let npts = self.cell_data.npts;
        for eq in 0..self.noutputs {
            for q in 0..n1d {
                // gather per-lane wdet*jinv*f1 into tmp, plus wdet*f0
                let mut ref_flux = [0.0f64; SIMD_CELL_WIDTH];
                for lane in 0..self.nlanes {
                    let wdet = self.cell_data.wdet_cache[self.cell_indices[lane] * npts + q];
                    let jinv = self.cell_data.jinv_cache[self.cell_indices[lane] * npts + q];
                    ref_flux[lane] = wdet * jinv * self.f1[(eq * npts + q) * w + lane];
                    self.out[(eq * self.cell_data.ndofs + tensor.q_to_local[q]) * w + lane] +=
                        wdet * self.f0[(eq * npts + q) * w + lane];
                }
                // GLL differentiation rows are dense; every term runs.
                for a in 0..n1d {
                    let test = tensor.q_to_local[a];
                    axpy_lanes(
                        simd,
                        &mut self.out[(eq * self.cell_data.ndofs + test) * w..(eq * self.cell_data.ndofs + test) * w + self.nlanes],
                        tensor.differentiation[q * n1d + a],
                        &ref_flux[..self.nlanes],
                    );
                }
            }
        }
    }
}

/// 2D batched integration: lane-packed fluxes to lane-packed local actions.
///
/// `f0,fx,fy`: `[(eq*npts+q)*W+lane]` pointwise weak-form fluxes; `out`:
/// `[(eq*ndofs+local)*W+lane]`, zeroed by the caller and accumulated into.
/// Each point folds `wdet` and the per-lane `jinv` into `x`/`y` reference
/// fluxes, then applies the transposed 1D contractions over rows/columns.
/// `cell_indices` and `nlanes` follow [`interpolate_batch_2d`].
pub(crate) fn integrate_batch_2d(
    cell_data: &CellData,
    noutputs: usize,
    f0: &[f64],
    fx: &[f64],
    fy: &[f64],
    out: &mut [f64],
    cell_indices: &[usize],
    nlanes: usize,
) {
    Arch::new().dispatch(IntegrateBatch2d {
        cell_data,
        noutputs,
        f0,
        fx,
        fy,
        out,
        cell_indices,
        nlanes,
    });
}

/// Single-dispatch 2D batched integration (see [`integrate_batch_2d`]).
struct IntegrateBatch2d<'a> {
    cell_data: &'a CellData,
    noutputs: usize,
    f0: &'a [f64],
    fx: &'a [f64],
    fy: &'a [f64],
    out: &'a mut [f64],
    cell_indices: &'a [usize],
    nlanes: usize,
}

impl WithSimd for IntegrateBatch2d<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let w = SIMD_CELL_WIDTH;
        let tensor = self.cell_data.tensor.as_ref().expect("tensor batch needs tensor");
        let n1d = tensor.n1d;
        let npts = self.cell_data.npts;
        for eq in 0..self.noutputs {
            for j in 0..n1d {
                for i in 0..n1d {
                    let q = j * n1d + i;
                    let mut ref_x = [0.0f64; SIMD_CELL_WIDTH];
                    let mut ref_y = [0.0f64; SIMD_CELL_WIDTH];
                    for lane in 0..self.nlanes {
                        let wdet = self.cell_data.wdet_cache[self.cell_indices[lane] * npts + q];
                        let jinv = &self.cell_data.jinv_cache
                            [(self.cell_indices[lane] * npts + q) * 4..(self.cell_indices[lane] * npts + q + 1) * 4];
                        let f1x = self.fx[(eq * npts + q) * w + lane];
                        let f1y = self.fy[(eq * npts + q) * w + lane];
                        ref_x[lane] = wdet * (jinv[0] * f1x + jinv[1] * f1y);
                        ref_y[lane] = wdet * (jinv[2] * f1x + jinv[3] * f1y);
                        self.out[(eq * self.cell_data.ndofs + tensor.q_to_local[q]) * w + lane] +=
                            wdet * self.f0[(eq * npts + q) * w + lane];
                    }
                    // GLL differentiation rows are dense; every term runs.
                    for a in 0..n1d {
                        let xl = tensor.q_to_local[j * n1d + a];
                        axpy_lanes(
                            simd,
                            &mut self.out[(eq * self.cell_data.ndofs + xl) * w..(eq * self.cell_data.ndofs + xl) * w + self.nlanes],
                            tensor.differentiation[i * n1d + a],
                            &ref_x[..self.nlanes],
                        );
                        let yl = tensor.q_to_local[a * n1d + i];
                        axpy_lanes(
                            simd,
                            &mut self.out[(eq * self.cell_data.ndofs + yl) * w..(eq * self.cell_data.ndofs + yl) * w + self.nlanes],
                            tensor.differentiation[j * n1d + a],
                            &ref_y[..self.nlanes],
                        );
                    }
                }
            }
        }
    }
}

/// Copy lane-packed values/grads out to per-lane contiguous buffers for the
/// scalar-fallback physics path.
///
/// Reads `[(f*npts+q)*W+lane]` values and `[(f*gdim+d)*npts+q)*W+lane]` grads
/// into `[lane][field][q]` / `[lane][field][dir][q]` scalar views. This
/// transpose is pure overhead once physics vectorizes over lanes; it exists
/// only because the default pointwise kernels evaluate one lane at a time.
pub(crate) fn extract_lanes(
    lane_values: &[f64],
    lane_grads: &[f64],
    gdim: usize,
    nfields: usize,
    npts: usize,
    w: usize,
    nlanes: usize,
    scalar_values: &mut [f64],
    scalar_grads: &mut [f64],
) {
    for lane in 0..nlanes {
        for f in 0..nfields {
            for q in 0..npts {
                scalar_values[lane * nfields * npts + f * npts + q] =
                    lane_values[(f * npts + q) * w + lane];
                for d in 0..gdim {
                    scalar_grads[lane * nfields * gdim * npts + (f * gdim + d) * npts + q] =
                        lane_grads[((f * gdim + d) * npts + q) * w + lane];
                }
            }
        }
    }
}

/// Per-worker scratch for one SIMD-over-element lane group.
///
/// Lane-packed buffers use `[(field/q/local)*W + lane]`; scalar buffers hold
/// per-lane contiguous copies for fallback physics.
pub(crate) struct TensorLaneScratch {
    pub packed_coeffs: Vec<f64>,
    pub lane_values: Vec<f64>,
    pub lane_grads: Vec<f64>,
    pub scalar_values: Vec<f64>,
    pub scalar_grads: Vec<f64>,
    pub flux0: Vec<f64>,
    pub flux1x: Vec<f64>,
    pub flux1y: Vec<f64>,
    pub packed_out: Vec<f64>,
    /// Contiguous per-cell transpose of one lane for direct global scatter.
    pub cell_local: Vec<f64>,
    /// Global cell index per lane.
    pub cell_idx: Vec<usize>,
}

impl TensorLaneScratch {
    /// Allocate zeroed scratch for one lane group: `ninputs` state fields on
    /// `ndofs` locals / `npts` points, `noutputs` residual fields, `gdim`
    /// gradient directions, all at [`SIMD_CELL_WIDTH`] lanes. Reused across
    /// batches by one Rayon worker; never shared between threads.
    pub(crate) fn new(ninputs: usize, noutputs: usize, ndofs: usize, npts: usize, gdim: usize) -> Self {
        let w = SIMD_CELL_WIDTH;
        Self {
            packed_coeffs: vec![0.0; ninputs * ndofs * w],
            lane_values: vec![0.0; ninputs * npts * w],
            lane_grads: vec![0.0; ninputs * gdim * npts * w],
            scalar_values: vec![0.0; w * ninputs * npts],
            scalar_grads: vec![0.0; w * ninputs * gdim * npts],
            flux0: vec![0.0; noutputs * npts * w],
            flux1x: vec![0.0; noutputs * npts * w],
            flux1y: vec![0.0; noutputs * npts * w],
            packed_out: vec![0.0; noutputs * ndofs * w],
            cell_local: vec![0.0; noutputs * ndofs],
            cell_idx: vec![0usize; w],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{integrate_batch_1d, interpolate_batch_1d, SIMD_CELL_WIDTH};

    #[test]
    /// Guards the layout constants the batch helpers hard-code.
    fn lane_layout_helpers_smoke() {
        assert_eq!(SIMD_CELL_WIDTH, 8);
        let _ = integrate_batch_1d;
        let _ = interpolate_batch_1d;
    }
}
