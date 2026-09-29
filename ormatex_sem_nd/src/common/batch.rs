//! SIMD-over-element batch helpers (libCEED-inspired E-vector batching).
//!
//! Irregular global restriction stays scalar; dense tensor interpolation,
//! geometry transforms and integration run across independent element lanes
//! with lane-contiguous scratch `[field/q/local][lane]`.
//!
//! `SIMD_CELL_WIDTH` is re-exported from `cell` for a single source of truth.

use pulp::{Arch, Simd, WithSimd};
use std::cell::RefCell;
use std::sync::OnceLock;

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

/// 1D tensor interpolation with lane-packed geometry.
///
/// Dispatches once per call on `n1d` to a const-generic specialization
/// ([`interpolate_batch_1d_packed_n`], register-blocked, bit-identical) for
/// `n1d in 2..=8`, falling back to [`interpolate_batch_1d_packed_dynamic`]
/// otherwise. Arithmetic matches [`interpolate_batch_1d_packed_dynamic`] term-for-term
/// (same `axpy_lanes` FMA order, same plain `* jinv` transform), only the
/// geometry comes from the packed buffer and all `W` lanes are written.
/// Assumes full `W` lanes; padded lanes compute but are discarded by the
/// row-sorted scatter.
///
/// # Arguments
/// * `cell_data` - tensor data (differentiation, permutation).
/// * `nfields` - field count.
/// * `coeffs` - lane-packed coefficients `[(f*ndofs+d)*W+lane]`.
/// * `values` - lane-packed values `[(f*npts+q)*W+lane]`. Overwritten.
/// * `grads` - lane-packed grads `[(f*npts+q)*W+lane]`. Overwritten.
/// * `jinv_packed` - packed scalar inverse Jacobians `[q*W+lane]`.
pub(crate) fn interpolate_batch_1d_packed(
    cell_data: &CellData,
    nfields: usize,
    coeffs: &[f64],
    values: &mut [f64],
    grads: &mut [f64],
    jinv_packed: &[f64],
) {
    let w = SIMD_CELL_WIDTH;
    let npts = cell_data.npts;
    let ndofs = cell_data.ndofs;
    debug_assert_eq!(coeffs.len(), nfields * ndofs * w);
    debug_assert_eq!(values.len(), nfields * npts * w);
    debug_assert_eq!(grads.len(), nfields * npts * w);
    debug_assert_eq!(jinv_packed.len(), npts * w);
    let n1d = cell_data
        .tensor
        .as_ref()
        .expect("tensor batch requires tensor data")
        .n1d;
    match n1d {
        2 => interpolate_batch_1d_packed_n::<2>(
            cell_data,
            nfields,
            coeffs,
            values,
            grads,
            jinv_packed,
        ),
        3 => interpolate_batch_1d_packed_n::<3>(
            cell_data,
            nfields,
            coeffs,
            values,
            grads,
            jinv_packed,
        ),
        4 => interpolate_batch_1d_packed_n::<4>(
            cell_data,
            nfields,
            coeffs,
            values,
            grads,
            jinv_packed,
        ),
        5 => interpolate_batch_1d_packed_n::<5>(
            cell_data,
            nfields,
            coeffs,
            values,
            grads,
            jinv_packed,
        ),
        6 => interpolate_batch_1d_packed_n::<6>(
            cell_data,
            nfields,
            coeffs,
            values,
            grads,
            jinv_packed,
        ),
        7 => interpolate_batch_1d_packed_n::<7>(
            cell_data,
            nfields,
            coeffs,
            values,
            grads,
            jinv_packed,
        ),
        8 => interpolate_batch_1d_packed_n::<8>(
            cell_data,
            nfields,
            coeffs,
            values,
            grads,
            jinv_packed,
        ),
        _ => interpolate_batch_1d_packed_dynamic(
            cell_data,
            nfields,
            coeffs,
            values,
            grads,
            jinv_packed,
        ),
    }
}

/// Const-`N` 1D packed interpolation entry point (`N` = `n1d` = `npts`).
///
/// Single pulp `Arch` dispatch into [`interpolate_1d_const_inner`]; FMA runs
/// through `simd.mul_add_f64s` exactly as in the dynamic path.
///
/// # Arguments
/// Same as [`interpolate_batch_1d_packed`]; requires `tensor.n1d == N` and
/// `npts == N`.
///
/// # Returns
/// Nothing; `values`/`grads` are overwritten.
pub(crate) fn interpolate_batch_1d_packed_n<const N: usize>(
    cell_data: &CellData,
    nfields: usize,
    coeffs: &[f64],
    values: &mut [f64],
    grads: &mut [f64],
    jinv_packed: &[f64],
) {
    Arch::new().dispatch(InterpolateBatch1dPackedN::<N> {
        cell_data,
        nfields,
        coeffs,
        values,
        grads,
        jinv_packed,
    });
}

/// Dynamic-`n1d` 1D packed interpolation fallback (also used for `n1d > 8`).
///
/// Bitwise reference for the const specializations; matches
/// [`interpolate_batch_1d_packed_dynamic`] term-for-term on full lane groups.
///
/// # Arguments
/// Same as [`interpolate_batch_1d_packed`].
///
/// # Returns
/// Nothing; `values`/`grads` are overwritten.
pub(crate) fn interpolate_batch_1d_packed_dynamic(
    cell_data: &CellData,
    nfields: usize,
    coeffs: &[f64],
    values: &mut [f64],
    grads: &mut [f64],
    jinv_packed: &[f64],
) {
    Arch::new().dispatch(InterpolateBatch1dPacked {
        cell_data,
        nfields,
        coeffs,
        values,
        grads,
        jinv_packed,
    });
}

/// Single-dispatch const-`N` 1D packed interpolation (see
/// [`interpolate_batch_1d_packed_n`]).
struct InterpolateBatch1dPackedN<'a, const N: usize> {
    cell_data: &'a CellData,
    nfields: usize,
    coeffs: &'a [f64],
    values: &'a mut [f64],
    grads: &'a mut [f64],
    jinv_packed: &'a [f64],
}

impl<const N: usize> WithSimd for InterpolateBatch1dPackedN<'_, N> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        interpolate_1d_const_inner::<S, N>(
            simd,
            self.cell_data,
            self.nfields,
            self.coeffs,
            self.values,
            self.grads,
            self.jinv_packed,
        );
    }
}

/// Register-blocked const-`N` 1D interpolation kernel.
///
/// Contraction layout: for each point `q`, `ref[q] = SUM_a D[q][a]*v[a]`
/// accumulates in a stack `[f64; W]` array (`N` const so the `a` loop unrolls)
/// with the same `simd.mul_add_f64s(splat(D), source, acc)` FMA sequence in
/// the same `a = 0..N` order as the dynamic zero-then-`axpy_lanes` loop, then
/// the geometric transform `g = ref * jinv` (plain multiply, same order)
/// fuses on the store path. Fixed-size array accesses and `get_unchecked`
/// indexing keep bounds checks out of the inner loops.
///
/// Bit-identity vs the dynamic path: the per-lane FMA stream is
/// `fma(D[0],v[0],0), fma(D[1],v[1],acc), ...` in the same order (starting
/// from `0.0`, `mul_add(s,x,0) == s*x` up to the sign of zero, identically in
/// both), and the transform is the same plain multiply, so every output lane
/// is bitwise equal.
///
/// # Arguments
/// * `simd` - pulp SIMD token for FMA (`mul_add_f64s`).
/// * `cell_data` - tensor data; requires `tensor.n1d == N`, `npts == N`.
/// * `nfields` - field count.
/// * `coeffs` - lane-packed coefficients `[(f*ndofs+d)*W+lane]`.
/// * `values` - lane-packed values `[(f*npts+q)*W+lane]`. Overwritten.
/// * `grads` - lane-packed grads `[(f*npts+q)*W+lane]`. Overwritten.
/// * `jinv_packed` - packed scalar inverse Jacobians `[q*W+lane]`.
///
/// # Returns
/// Nothing; `values`/`grads` are overwritten.
#[inline(always)]
fn interpolate_1d_const_inner<S: Simd, const N: usize>(
    simd: S,
    cell_data: &CellData,
    nfields: usize,
    coeffs: &[f64],
    values: &mut [f64],
    grads: &mut [f64],
    jinv_packed: &[f64],
) {
    const W: usize = SIMD_CELL_WIDTH;
    let tensor = cell_data
        .tensor
        .as_ref()
        .expect("tensor batch requires tensor data");
    debug_assert_eq!(tensor.n1d, N);
    let ndofs = cell_data.ndofs;
    debug_assert_eq!(cell_data.npts, N);
    let diff = tensor.differentiation.as_slice();
    let q_to_local = tensor.q_to_local.as_slice();
    debug_assert_eq!(diff.len(), N * N);
    debug_assert_eq!(q_to_local.len(), N);
    let coeffs_ptr = coeffs.as_ptr();
    let values_ptr = values.as_mut_ptr();
    let grads_ptr = grads.as_mut_ptr();
    let jinv_ptr = jinv_packed.as_ptr();
    for f in 0..nfields {
        for q in 0..N {
            unsafe {
                let local = *q_to_local.get_unchecked(q);
                debug_assert!(local < ndofs);
                let src = coeffs_ptr.add((f * ndofs + local) * W);
                let dst = values_ptr.add((f * N + q) * W);
                std::ptr::copy_nonoverlapping(src, dst, W);
            }
        }
        for q in 0..N {
            let mut acc = [0.0f64; SIMD_CELL_WIDTH];
            for a in 0..N {
                unsafe {
                    let dx = *diff.get_unchecked(q * N + a);
                    let vx = std::slice::from_raw_parts(values_ptr.add((f * N + a) * W), W);
                    axpy_lanes(simd, &mut acc, dx, vx);
                }
            }
            unsafe {
                let gbase = (f * N + q) * W;
                for lane in 0..SIMD_CELL_WIDTH {
                    let r = *acc.get_unchecked(lane);
                    let j = *jinv_ptr.add(q * W + lane);
                    *grads_ptr.add(gbase + lane) = r * j;
                }
            }
        }
    }
}

/// Single-dispatch 1D packed interpolation, dynamic `n1d` (see
/// [`interpolate_batch_1d_packed_dynamic`]).
struct InterpolateBatch1dPacked<'a> {
    cell_data: &'a CellData,
    nfields: usize,
    coeffs: &'a [f64],
    values: &'a mut [f64],
    grads: &'a mut [f64],
    jinv_packed: &'a [f64],
}

impl WithSimd for InterpolateBatch1dPacked<'_> {
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
            for q in 0..npts {
                let local = tensor.q_to_local[q];
                self.values[(f * npts + q) * w..(f * npts + q) * w + w].copy_from_slice(
                    &self.coeffs[(f * self.cell_data.ndofs + local) * w
                        ..(f * self.cell_data.ndofs + local) * w + w],
                );
            }
            for q in 0..n1d {
                let dst_off = (f * npts + q) * w;
                for lane in 0..w {
                    self.grads[dst_off + lane] = 0.0;
                }
                for a in 0..n1d {
                    let factor = tensor.differentiation[q * n1d + a];
                    axpy_lanes(
                        simd,
                        &mut self.grads[dst_off..dst_off + w],
                        factor,
                        &self.values[(f * npts + a) * w..(f * npts + a) * w + w],
                    );
                }
                for lane in 0..w {
                    unsafe {
                        let jinv = *self.jinv_packed.get_unchecked(q * w + lane);
                        *self.grads.get_unchecked_mut(dst_off + lane) *= jinv;
                    }
                }
            }
        }
    }
}

/// 1D batched integration with lane-packed geometry.
///
/// Dispatches once per call on `n1d` to a const-generic specialization
/// ([`integrate_batch_1d_packed_n`]) for `n1d in 2..=8`, falling back to
/// [`integrate_batch_1d_packed_dynamic`] otherwise. Arithmetic matches
/// [`integrate_batch_1d_packed_dynamic`] term-for-term (same reference-flux expressions,
/// same `axpy_lanes` FMA order). `out` is accumulated into (`+=`).
/// Assumes full `W` lanes.
///
/// # Arguments
/// * `cell_data` - tensor data.
/// * `noutputs` - equation count.
/// * `f0`/`f1` - lane-packed fluxes `[(eq*npts+q)*W+lane]`.
/// * `out` - lane-packed actions `[(eq*ndofs+local)*W+lane]`. Accumulated into.
/// * `wdet_packed` - packed measures `[q*W+lane]`.
/// * `jinv_packed` - packed scalar inverse Jacobians `[q*W+lane]`.
pub(crate) fn integrate_batch_1d_packed(
    cell_data: &CellData,
    noutputs: usize,
    f0: &[f64],
    f1: &[f64],
    out: &mut [f64],
    wdet_packed: &[f64],
    jinv_packed: &[f64],
) {
    let w = SIMD_CELL_WIDTH;
    let npts = cell_data.npts;
    let ndofs = cell_data.ndofs;
    debug_assert_eq!(f0.len(), noutputs * npts * w);
    debug_assert_eq!(f1.len(), noutputs * npts * w);
    debug_assert_eq!(out.len(), noutputs * ndofs * w);
    debug_assert_eq!(wdet_packed.len(), npts * w);
    debug_assert_eq!(jinv_packed.len(), npts * w);
    let n1d = cell_data
        .tensor
        .as_ref()
        .expect("tensor batch needs tensor")
        .n1d;
    match n1d {
        2 => integrate_batch_1d_packed_n::<2>(
            cell_data,
            noutputs,
            f0,
            f1,
            out,
            wdet_packed,
            jinv_packed,
        ),
        3 => integrate_batch_1d_packed_n::<3>(
            cell_data,
            noutputs,
            f0,
            f1,
            out,
            wdet_packed,
            jinv_packed,
        ),
        4 => integrate_batch_1d_packed_n::<4>(
            cell_data,
            noutputs,
            f0,
            f1,
            out,
            wdet_packed,
            jinv_packed,
        ),
        5 => integrate_batch_1d_packed_n::<5>(
            cell_data,
            noutputs,
            f0,
            f1,
            out,
            wdet_packed,
            jinv_packed,
        ),
        6 => integrate_batch_1d_packed_n::<6>(
            cell_data,
            noutputs,
            f0,
            f1,
            out,
            wdet_packed,
            jinv_packed,
        ),
        7 => integrate_batch_1d_packed_n::<7>(
            cell_data,
            noutputs,
            f0,
            f1,
            out,
            wdet_packed,
            jinv_packed,
        ),
        8 => integrate_batch_1d_packed_n::<8>(
            cell_data,
            noutputs,
            f0,
            f1,
            out,
            wdet_packed,
            jinv_packed,
        ),
        _ => integrate_batch_1d_packed_dynamic(
            cell_data,
            noutputs,
            f0,
            f1,
            out,
            wdet_packed,
            jinv_packed,
        ),
    }
}

/// Const-`N` 1D packed integration entry point (`N` = `n1d` = `npts`).
///
/// Single pulp `Arch` dispatch into [`integrate_1d_const_inner`]; FMA runs
/// through `simd.mul_add_f64s` exactly as in the dynamic path.
///
/// # Arguments
/// Same as [`integrate_batch_1d_packed`]; requires `tensor.n1d == N` and
/// `npts == N`.
///
/// # Returns
/// Nothing; `out` is accumulated into (`+=`).
pub(crate) fn integrate_batch_1d_packed_n<const N: usize>(
    cell_data: &CellData,
    noutputs: usize,
    f0: &[f64],
    f1: &[f64],
    out: &mut [f64],
    wdet_packed: &[f64],
    jinv_packed: &[f64],
) {
    Arch::new().dispatch(IntegrateBatch1dPackedN::<N> {
        cell_data,
        noutputs,
        f0,
        f1,
        out,
        wdet_packed,
        jinv_packed,
    });
}

/// Dynamic-`n1d` 1D packed integration fallback (also used for `n1d > 8`).
///
/// Bitwise reference for the const specialization; unit tests compare
/// against it directly.
///
/// # Arguments
/// Same as [`integrate_batch_1d_packed`].
///
/// # Returns
/// Nothing; `out` is accumulated into (`+=`).
pub(crate) fn integrate_batch_1d_packed_dynamic(
    cell_data: &CellData,
    noutputs: usize,
    f0: &[f64],
    f1: &[f64],
    out: &mut [f64],
    wdet_packed: &[f64],
    jinv_packed: &[f64],
) {
    Arch::new().dispatch(IntegrateBatch1dPacked {
        cell_data,
        noutputs,
        f0,
        f1,
        out,
        wdet_packed,
        jinv_packed,
    });
}

/// Single-dispatch const-`N` 1D packed integration (see
/// [`integrate_batch_1d_packed_n`]).
struct IntegrateBatch1dPackedN<'a, const N: usize> {
    cell_data: &'a CellData,
    noutputs: usize,
    f0: &'a [f64],
    f1: &'a [f64],
    out: &'a mut [f64],
    wdet_packed: &'a [f64],
    jinv_packed: &'a [f64],
}

impl<const N: usize> WithSimd for IntegrateBatch1dPackedN<'_, N> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        integrate_1d_const_inner::<S, N>(
            simd,
            self.cell_data,
            self.noutputs,
            self.f0,
            self.f1,
            self.out,
            self.wdet_packed,
            self.jinv_packed,
        );
    }
}

/// Const-`N` 1D integration kernel with the CURRENT scatter accumulation order.
///
/// For each `(eq, q)` in row-major order: the per-lane reference flux
/// `ref = (wdet*jinv)*f1` and the mass term `out[q2l[q]] += wdet*f0` compute
/// in registers with the same plain-arithmetic expression order as the
/// dynamic path, then the transposed contraction `out[q2l[a]] += D[q][a]*ref`
/// runs for `a = 0..N` with the same `axpy_lanes` FMA. `N` const lets the `a`
/// loop unroll and keeps bounds checks out of the inner loops; `ref` lives on
/// the stack.
///
/// Bit-identical to the dynamic path: loop order, per-term FMA vs plain-op
/// choice, and expression order are all unchanged; only bounds checks are
/// removed.
///
/// # Arguments
/// * `simd` - pulp SIMD token for FMA (`mul_add_f64s`).
/// * `cell_data` - tensor data; requires `tensor.n1d == N`, `npts == N`.
/// * `noutputs` - equation count.
/// * `f0`/`f1` - lane-packed fluxes `[(eq*npts+q)*W+lane]`.
/// * `out` - lane-packed actions `[(eq*ndofs+local)*W+lane]`. Accumulated into.
/// * `wdet_packed` - packed measures `[q*W+lane]`.
/// * `jinv_packed` - packed scalar inverse Jacobians `[q*W+lane]`.
///
/// # Returns
/// Nothing; `out` is accumulated into (`+=`).
#[inline(always)]
fn integrate_1d_const_inner<S: Simd, const N: usize>(
    simd: S,
    cell_data: &CellData,
    noutputs: usize,
    f0: &[f64],
    f1: &[f64],
    out: &mut [f64],
    wdet_packed: &[f64],
    jinv_packed: &[f64],
) {
    const W: usize = SIMD_CELL_WIDTH;
    let tensor = cell_data
        .tensor
        .as_ref()
        .expect("tensor batch needs tensor");
    debug_assert_eq!(tensor.n1d, N);
    let ndofs = cell_data.ndofs;
    debug_assert_eq!(cell_data.npts, N);
    let diff = tensor.differentiation.as_slice();
    let q_to_local = tensor.q_to_local.as_slice();
    debug_assert_eq!(diff.len(), N * N);
    debug_assert_eq!(q_to_local.len(), N);
    let f0_ptr = f0.as_ptr();
    let f1_ptr = f1.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let wdet_ptr = wdet_packed.as_ptr();
    let jinv_ptr = jinv_packed.as_ptr();
    for eq in 0..noutputs {
        for q in 0..N {
            let mut ref_flux = [0.0f64; SIMD_CELL_WIDTH];
            unsafe {
                let mass_local = *q_to_local.get_unchecked(q);
                debug_assert!(mass_local < ndofs);
                for lane in 0..SIMD_CELL_WIDTH {
                    let wdet = *wdet_ptr.add(q * W + lane);
                    let jinv = *jinv_ptr.add(q * W + lane);
                    *ref_flux.get_unchecked_mut(lane) =
                        wdet * jinv * *f1_ptr.add((eq * N + q) * W + lane);
                    let dst = out_ptr.add((eq * ndofs + mass_local) * W + lane);
                    *dst += wdet * *f0_ptr.add((eq * N + q) * W + lane);
                }
            }
            for a in 0..N {
                unsafe {
                    let test = *q_to_local.get_unchecked(a);
                    debug_assert!(test < ndofs);
                    let dst =
                        std::slice::from_raw_parts_mut(out_ptr.add((eq * ndofs + test) * W), W);
                    axpy_lanes(simd, dst, *diff.get_unchecked(q * N + a), &ref_flux);
                }
            }
        }
    }
}

/// Single-dispatch 1D packed integration, dynamic `n1d` (see
/// [`integrate_batch_1d_packed_dynamic`]).
struct IntegrateBatch1dPacked<'a> {
    cell_data: &'a CellData,
    noutputs: usize,
    f0: &'a [f64],
    f1: &'a [f64],
    out: &'a mut [f64],
    wdet_packed: &'a [f64],
    jinv_packed: &'a [f64],
}

impl WithSimd for IntegrateBatch1dPacked<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let w = SIMD_CELL_WIDTH;
        let tensor = self
            .cell_data
            .tensor
            .as_ref()
            .expect("tensor batch needs tensor");
        let n1d = tensor.n1d;
        let npts = self.cell_data.npts;
        for eq in 0..self.noutputs {
            for q in 0..n1d {
                let mut ref_flux = [0.0f64; SIMD_CELL_WIDTH];
                for lane in 0..w {
                    unsafe {
                        let wdet = *self.wdet_packed.get_unchecked(q * w + lane);
                        let jinv = *self.jinv_packed.get_unchecked(q * w + lane);
                        ref_flux[lane] =
                            wdet * jinv * *self.f1.get_unchecked((eq * npts + q) * w + lane);
                        *self.out.get_unchecked_mut(
                            (eq * self.cell_data.ndofs + tensor.q_to_local[q]) * w + lane,
                        ) += wdet * *self.f0.get_unchecked((eq * npts + q) * w + lane);
                    }
                }
                for a in 0..n1d {
                    let test = tensor.q_to_local[a];
                    axpy_lanes(
                        simd,
                        &mut self.out[(eq * self.cell_data.ndofs + test) * w
                            ..(eq * self.cell_data.ndofs + test) * w + w],
                        tensor.differentiation[q * n1d + a],
                        &ref_flux[..w],
                    );
                }
            }
        }
    }
}

/// Per-worker scratch for one SIMD-over-element lane group.
///
/// All buffers are lane-packed (`[(field/q/local)*W + lane]`).
pub(crate) struct TensorLaneScratch {
    pub packed_coeffs: Vec<f64>,
    pub lane_values: Vec<f64>,
    pub lane_grads: Vec<f64>,
    pub flux0: Vec<f64>,
    pub flux1x: Vec<f64>,
    pub flux1y: Vec<f64>,
    pub packed_out: Vec<f64>,
}

impl TensorLaneScratch {
    /// Allocate zeroed scratch for one lane group: `ninputs` state fields on
    /// `ndofs` locals / `npts` points, `noutputs` residual fields, `gdim`
    /// gradient directions, all at [`SIMD_CELL_WIDTH`] lanes. Reused across
    /// batches by one Rayon worker; never shared between threads.
    pub(crate) fn new(
        ninputs: usize,
        noutputs: usize,
        ndofs: usize,
        npts: usize,
        gdim: usize,
    ) -> Self {
        let w = SIMD_CELL_WIDTH;
        Self {
            packed_coeffs: vec![0.0; ninputs * ndofs * w],
            lane_values: vec![0.0; ninputs * npts * w],
            lane_grads: vec![0.0; ninputs * gdim * npts * w],
            flux0: vec![0.0; noutputs * npts * w],
            flux1x: vec![0.0; noutputs * npts * w],
            flux1y: vec![0.0; noutputs * npts * w],
            packed_out: vec![0.0; noutputs * ndofs * w],
        }
    }

    /// Resize every buffer to the exact requested dimensions.
    ///
    /// `Vec::resize` only reallocates when the new length exceeds capacity,
    /// so steady-state calls with fixed dimensions perform no allocation.
    /// Every buffer is fully overwritten before being read (`packed_out` is
    /// explicitly `fill(0.0)`-ed before the accumulating integrate step), so
    /// stale contents from a previous call are harmless.
    pub(crate) fn ensure(
        &mut self,
        ninputs: usize,
        noutputs: usize,
        ndofs: usize,
        npts: usize,
        gdim: usize,
    ) {
        let w = SIMD_CELL_WIDTH;
        self.packed_coeffs.resize(ninputs * ndofs * w, 0.0);
        self.lane_values.resize(ninputs * npts * w, 0.0);
        self.lane_grads.resize(ninputs * gdim * npts * w, 0.0);
        self.flux0.resize(noutputs * npts * w, 0.0);
        self.flux1x.resize(noutputs * npts * w, 0.0);
        self.flux1y.resize(noutputs * npts * w, 0.0);
        self.packed_out.resize(noutputs * ndofs * w, 0.0);
    }
}

/// Persistent per-thread scratch for the tensor batch paths.
///
/// Holds both SIMD lane groups (state and direction), so one thread reuses
/// the same allocations across chunks, colors, and successive
/// residual/Jacobian calls. Shared between the residual and Jacobian paths,
/// which may request different sizes; [`TensorWorkerScratch::ensure`] resizes
/// everything to the caller's exact dimensions (reallocating only on growth).
pub(crate) struct TensorWorkerScratch {
    /// Lane group for state interpolation and residual fluxes.
    pub state_lane: TensorLaneScratch,
    /// Lane group for direction interpolation and Jacobian fluxes.
    pub dir_lane: TensorLaneScratch,
}

impl TensorWorkerScratch {
    /// Allocate an empty worker scratch; call [`TensorWorkerScratch::ensure`]
    /// before first use.
    pub(crate) fn new() -> Self {
        Self {
            state_lane: TensorLaneScratch::new(0, 0, 0, 0, 2),
            dir_lane: TensorLaneScratch::new(0, 0, 0, 0, 2),
        }
    }

    /// Resize every buffer to the exact requested dimensions (see
    /// [`TensorLaneScratch::ensure`]). Only grows allocations; shrinking
    /// truncates lengths in place. Every buffer is fully overwritten before
    /// being read within one chunk visit, so stale contents are harmless.
    pub(crate) fn ensure(
        &mut self,
        ninputs: usize,
        noutputs: usize,
        ndofs: usize,
        npts: usize,
        gdim: usize,
    ) {
        self.state_lane.ensure(ninputs, noutputs, ndofs, npts, gdim);
        self.dir_lane.ensure(ninputs, noutputs, ndofs, npts, gdim);
    }
}

thread_local! {
    static TENSOR_WORKER_SCRATCH: RefCell<TensorWorkerScratch> =
        RefCell::new(TensorWorkerScratch::new());
}

/// Run `f` with the calling thread's persistent [`TensorWorkerScratch`],
/// resized to the exact requested dimensions.
///
/// Uses `try_borrow_mut()`; on re-entrant use (already borrowed on this
/// thread) falls back to a freshly allocated local scratch so behavior stays
/// correct at the cost of one allocation in a path that should not occur in
/// the current call graph.
///
/// # Arguments
/// * `ninputs` - number of input/state fields the buffers must hold.
/// * `noutputs` - number of output/residual fields the buffers must hold.
/// * `ndofs` - local basis functions per cell.
/// * `npts` - quadrature points per cell.
/// * `gdim` - geometric dimension (gradient components per field).
/// * `f` - closure receiving the ensured scratch.
/// # Returns
/// The closure's return value.
pub(crate) fn with_tensor_worker_scratch<R>(
    ninputs: usize,
    noutputs: usize,
    ndofs: usize,
    npts: usize,
    gdim: usize,
    f: impl FnOnce(&mut TensorWorkerScratch) -> R,
) -> R {
    let mut f = Some(f);
    // `try_with` fails only while the thread is tearing down;
    // `try_borrow_mut` fails only on re-entrant use of the scratch on this
    // thread. Either way, fall back to a freshly allocated local scratch so
    // behavior stays correct.
    let hit = TENSOR_WORKER_SCRATCH.try_with(|cell| {
        let mut guard = cell.try_borrow_mut().ok()?;
        guard.ensure(ninputs, noutputs, ndofs, npts, gdim);
        Some((f.take().unwrap())(&mut guard))
    });
    match hit {
        Ok(Some(value)) => value,
        _ => {
            let mut local = TensorWorkerScratch::new();
            local.ensure(ninputs, noutputs, ndofs, npts, gdim);
            (f.take().unwrap())(&mut local)
        }
    }
}

/// Sentinel for eliminated DOFs in packed gather indices.
///
/// Packed indices equal to this (`u32::MAX`) read the packed prescribed value
/// (state gather) or zero (direction gather).
pub(crate) const TENSOR_SENTINEL: u32 = u32::MAX;

/// Precomputed row-sorted E-vector scatter table for one output selection.
///
/// Layout: the row-sorted E-vector holds `nslots` doubles per direction column
/// (`ncols * nslots`, column `c` at `c * nslots`), where slots are ordered by
/// row (`row_ptr[r]..row_ptr[r+1]` is row `r`'s contiguous segment) and within
/// a row by `(color, pos, local)` — exactly the old color-sequential `+=`
/// order, so `out[r] = sum(seg)` with `let mut s = 0.0; for v in seg { s += v }`
/// is bit-identical to the old batch-major reduction. `dest` inverts that
/// order: for batch `b`, `dest[b * block .. (b+1) * block]` maps the
/// lane-packed local index `(pos * ndofs + local) * W + lane` to its slot
/// (`u32::MAX` sentinel for padded lanes / eliminated DOFs, never written).
/// Each real slot is the image of exactly one `(batch, inner)` pair, so
/// different batches write disjoint slots and phase 1 needs no atomics.
pub(crate) struct SortedRowScatter {
    /// CSR row starts, length `total_size + 1` (slots per column).
    pub row_ptr: Vec<u32>,
    /// Per-batch destination slots, length `nbatches * block`.
    pub dest: Vec<u32>,
    /// Slot count per column (`row_ptr[total]`).
    pub nslots: usize,
}

impl SortedRowScatter {
    /// Build the row-sorted scatter table for one output-field selection.
    ///
    /// # Arguments
    /// * `restriction` - element restriction (maps, offsets, sources).
    /// * `color_of` - color index per cell (orders slots within a row).
    /// * `nbatches` - batch count of the flat natural-order plan.
    /// * `selection_outputs` - global field ids receiving the scatter.
    /// * `ndofs` - local DOFs per cell.
    ///
    /// # Returns
    /// Scatter table with per-row slots sorted by `(color, position, local)`
    /// and the inverted per-batch `dest` map.
    pub(crate) fn build(
        restriction: &super::restriction::ElementRestriction,
        color_of: &[usize],
        nbatches: usize,
        selection_outputs: &[usize],
        ndofs: usize,
    ) -> Self {
        let w = SIMD_CELL_WIDTH;
        let noutputs = selection_outputs.len();
        let total = restriction.total_size();
        let ncells = restriction.cell_count();
        let block = noutputs * ndofs * w;
        assert!(
            nbatches * block < u32::MAX as usize,
            "E-vector too large for u32"
        );
        let mut tmp: Vec<Vec<(u32, u32, u32)>> = vec![Vec::new(); total];
        for cell in 0..ncells {
            let batch = cell / w;
            let lane = cell % w;
            debug_assert!(batch < nbatches);
            let color = color_of[cell] as u32;
            for (pos, &field) in selection_outputs.iter().enumerate() {
                let source = restriction.field_map_source(field);
                let map = restriction.cell_map(source, cell);
                debug_assert_eq!(map.len(), ndofs);
                let base = restriction.offset(field);
                for (local, &reduced) in map.iter().enumerate() {
                    if let Some(r) = reduced {
                        let row = base + r;
                        let idx = (batch * block + (pos * ndofs + local) * w + lane) as u32;
                        let order = (pos * ndofs + local) as u32;
                        tmp[row].push((color, order, idx));
                    }
                }
            }
        }
        for entries in tmp.iter_mut() {
            entries.sort_unstable();
        }
        let mut row_ptr = Vec::with_capacity(total + 1);
        let mut entry_idx: Vec<u32> = Vec::new();
        row_ptr.push(0);
        for entries in &tmp {
            for &(_, _, idx) in entries {
                entry_idx.push(idx);
            }
            row_ptr.push(entry_idx.len() as u32);
        }
        let nslots = entry_idx.len();
        // Invert `entry_idx` (slot order) into per-batch destinations.
        // Each real batch-major index appears exactly once (one global row per
        // real lane), so each slot has exactly one writer per column and
        // batches write disjoint slot sets; padded/eliminated stay sentinel.
        let mut dest = vec![u32::MAX; nbatches * block];
        for (slot, idx) in entry_idx.iter().enumerate() {
            assert_eq!(dest[*idx as usize], u32::MAX);
            dest[*idx as usize] = slot as u32;
        }
        Self {
            row_ptr,
            dest,
            nslots,
        }
    }
}

/// Raw-pointer writer for the row-sorted E-vector (phase 1 scatter).
///
/// Each slot is written exactly once per column by exactly one batch, so
/// concurrent batch writes are disjoint and need no atomics. The type cannot
/// enforce this; [`SortedRowScatter::build`]'s inversion argument does.
#[derive(Clone, Copy)]
pub(crate) struct DisjointSortedEvec {
    ptr: *mut f64,
    nslots: usize,
}

// SAFETY: sound only for slot-disjoint concurrent use; callers uphold this by
// writing each batch's `dest`-mapped slots exactly once per column (disjoint
// across batches by construction, see `SortedRowScatter`).
unsafe impl Send for DisjointSortedEvec {}
unsafe impl Sync for DisjointSortedEvec {}

impl DisjointSortedEvec {
    /// Wrap a row-sorted E-vector for disjoint slot writes.
    ///
    /// # Arguments
    /// * `slice` - row-sorted E-vector (`ncols * nslots`). Must outlive all uses.
    /// * `nslots` - slots per column.
    ///
    /// # Returns
    /// Writer handle; concurrent writes must target disjoint slots.
    pub(crate) unsafe fn new(slice: &mut [f64], nslots: usize) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            nslots,
        }
    }

    /// Write one slot of one column.
    ///
    /// # Arguments
    /// * `slot` - slot index (`< nslots`).
    /// * `column` - direction column.
    /// * `value` - value to store (overwrites).
    ///
    /// SAFETY: same requirements as the handle itself: concurrent writes must
    /// target disjoint `(column, slot)` pairs.
    #[inline(always)]
    pub(crate) unsafe fn write(&self, slot: u32, column: usize, value: f64) {
        debug_assert!((slot as usize) < self.nslots);
        *self.ptr.add(column * self.nslots + slot as usize) = value;
    }
}

/// One full SIMD-over-element batch with lane-packed geometry and restriction.
///
/// `cells` holds the global cell per lane. `jinv` packs
/// `jinv[(q*4+k)*W+lane]` copied from `CellData::jinv_cache`, `wdet` packs
/// `wdet[q*W+lane]` from `CellData::wdet_cache`. `map_idx[s][local*W+lane]`
/// holds the reduced DOF (or [`TENSOR_SENTINEL`]) for restriction source `s`,
/// `map_presc[s][local*W+lane]` the prescribed value (or `0.0`).
pub(crate) struct TensorBatch {
    /// Global cell per lane (`cells[nreal..W]` pads with the last real cell).
    pub cells: [usize; SIMD_CELL_WIDTH],
    /// Real lanes in this batch (`1..=W`); padded lanes compute but are discarded.
    pub nreal: usize,
    /// Packed inverse Jacobians `[(q*4+k)*W+lane]`.
    pub jinv: Vec<f64>,
    /// Packed weighted measures `[q*W+lane]`.
    pub wdet: Vec<f64>,
    /// Packed reduced indices per source `[local*W+lane]`.
    pub map_idx: Vec<Vec<u32>>,
    /// Packed prescribed values per source `[local*W+lane]`.
    pub map_presc: Vec<Vec<f64>>,
}

/// Precomputed flat tensor batch plan for a 1D or 2D problem.
///
/// Batches run over cells in natural cell-index order (`ceil(ncells / W)`
/// batches, padding only the last one); lanes are independent so per-cell
/// results are unchanged. `color_of` keeps each cell's greedy-color id solely
/// to order [`SortedRowScatter`] row reductions bit-identically to the old
/// color-sequential scatter. The packed `jinv` layout is fixed per build:
/// scalar `jinv` (`[q*W+lane]`) for 1D, four-component `jinv`
/// (`[(q*4+k)*W+lane]`) for 2D.
pub(crate) struct TensorBatchPlan {
    /// Flat batches in natural cell order.
    pub batches: Vec<TensorBatch>,
    /// Greedy-color index per cell (orders row entries, not batches).
    pub color_of: Vec<usize>,
    /// Local DOFs per cell.
    pub ndofs: usize,
}

impl TensorBatchPlan {
    /// Build a plan from cached geometry and restriction.
    ///
    /// Packs per-lane geometry (`jinv`, `wdet`) and restriction indices plus
    /// prescribed values for every natural-order chunk; the trailing partial
    /// chunk is padded with its last cell (`nreal` marks real lanes).
    ///
    /// # Arguments
    /// * `cell_data` - cached Jacobians, measures, and sizes.
    /// * `restriction` - maps, prescribed values, and coloring.
    ///
    /// # Returns
    /// Plan with one batch per chunk covering every cell.
    pub(crate) fn build(
        cell_data: &CellData,
        restriction: &super::restriction::ElementRestriction,
    ) -> Self {
        let w = SIMD_CELL_WIDTH;
        let ndofs = cell_data.ndofs;
        let npts = cell_data.npts;
        let n_sources = restriction.n_sources();
        debug_assert!(n_sources >= 1);
        // Non-uniform ndofs cannot occur by construction: every cell map has
        // `cell_size() == ndofs` (see `ElementRestriction::new`), and
        // `cell_size()` equals `cell_data.ndofs`. Partial chunks are padded
        // with the last cell instead of taking a scalar path.
        assert_eq!(
            restriction.cell_size(),
            ndofs,
            "tensor batch plan requires uniform ndofs"
        );
        let ncells = restriction.cell_count();
        // One-time validation for the `get_unchecked` gathers below: every
        // packed reduced index must fit the global restriction, so
        // `off + r` stays in bounds for any consistent field offset layout.
        let mut max_reduced: usize = 0;
        let mut any_reduced = false;
        for s in 0..n_sources {
            for cell in 0..ncells {
                for &r in restriction.cell_map(s, cell) {
                    if let Some(v) = r {
                        any_reduced = true;
                        max_reduced = max_reduced.max(v);
                    }
                }
            }
        }
        if any_reduced {
            assert!(
                max_reduced < restriction.total_size(),
                "tensor batch plan reduced index out of range"
            );
        }
        let mut color_of = vec![usize::MAX; ncells];
        for (color, cells) in restriction.cell_colors().iter().enumerate() {
            for &cell in cells {
                color_of[cell] = color;
            }
        }
        debug_assert!(color_of.iter().all(|&c| c != usize::MAX));
        let mut batches = Vec::with_capacity(ncells.div_ceil(w));
        for batch_idx in 0..ncells.div_ceil(w) {
            let start = batch_idx * w;
            let end = (start + w).min(ncells);
            let nreal = end - start;
            debug_assert!(nreal > 0 && nreal <= w);
            let pad_cell = end - 1;
            let mut cells = [pad_cell; SIMD_CELL_WIDTH];
            for (lane, cell) in (start..end).enumerate() {
                cells[lane] = cell;
            }
            let mut jinv = vec![0.0; npts * 4 * w];
            let mut wdet = vec![0.0; npts * w];
            for (lane, &cell) in cells.iter().enumerate() {
                for q in 0..npts {
                    wdet[q * w + lane] = cell_data.wdet_cache[cell * npts + q];
                    let base = (cell * npts + q) * 4;
                    for k in 0..4 {
                        jinv[(q * 4 + k) * w + lane] = cell_data.jinv_cache[base + k];
                    }
                }
            }
            let mut map_idx = Vec::with_capacity(n_sources);
            let mut map_presc = Vec::with_capacity(n_sources);
            for s in 0..n_sources {
                let mut idx = vec![0u32; ndofs * w];
                let mut presc = vec![0.0; ndofs * w];
                for local in 0..ndofs {
                    for (lane, &cell) in cells.iter().enumerate() {
                        let r = restriction.cell_map(s, cell)[local];
                        idx[local * w + lane] = match r {
                            Some(v) => {
                                debug_assert!((v as u64) < u32::MAX as u64);
                                v as u32
                            }
                            None => TENSOR_SENTINEL,
                        };
                        presc[local * w + lane] =
                            restriction.cell_prescribed(s, cell)[local].unwrap_or(0.0);
                    }
                }
                map_idx.push(idx);
                map_presc.push(presc);
            }
            batches.push(TensorBatch {
                cells,
                nreal,
                jinv,
                wdet,
                map_idx,
                map_presc,
            });
        }
        Self {
            batches,
            color_of,
            ndofs,
        }
    }

    /// Build a 1D plan from cached geometry and restriction.
    ///
    /// Same natural-order batching, last-cell padding, and packed restriction
    /// as [`TensorBatchPlan::build`], but packs the scalar 1D inverse
    /// Jacobians (`jinv[q*W+lane]` from `CellData::jinv_cache[cell*npts+q]`)
    /// instead of the four-component 2D tensors.
    ///
    /// # Arguments
    /// * `cell_data` - cached Jacobians, measures, and sizes.
    /// * `restriction` - maps, prescribed values, and coloring.
    ///
    /// # Returns
    /// Plan with one batch per chunk covering every cell (`gdim == 1`).
    pub(crate) fn build_1d(
        cell_data: &CellData,
        restriction: &super::restriction::ElementRestriction,
    ) -> Self {
        let w = SIMD_CELL_WIDTH;
        let ndofs = cell_data.ndofs;
        let npts = cell_data.npts;
        let n_sources = restriction.n_sources();
        debug_assert!(n_sources >= 1);
        assert_eq!(
            restriction.cell_size(),
            ndofs,
            "tensor batch plan requires uniform ndofs"
        );
        let ncells = restriction.cell_count();
        let mut max_reduced: usize = 0;
        let mut any_reduced = false;
        for s in 0..n_sources {
            for cell in 0..ncells {
                for &r in restriction.cell_map(s, cell) {
                    if let Some(v) = r {
                        any_reduced = true;
                        max_reduced = max_reduced.max(v);
                    }
                }
            }
        }
        if any_reduced {
            assert!(
                max_reduced < restriction.total_size(),
                "tensor batch plan reduced index out of range"
            );
        }
        let mut color_of = vec![usize::MAX; ncells];
        for (color, cells) in restriction.cell_colors().iter().enumerate() {
            for &cell in cells {
                color_of[cell] = color;
            }
        }
        debug_assert!(color_of.iter().all(|&c| c != usize::MAX));
        let mut batches = Vec::with_capacity(ncells.div_ceil(w));
        for batch_idx in 0..ncells.div_ceil(w) {
            let start = batch_idx * w;
            let end = (start + w).min(ncells);
            let nreal = end - start;
            debug_assert!(nreal > 0 && nreal <= w);
            let pad_cell = end - 1;
            let mut cells = [pad_cell; SIMD_CELL_WIDTH];
            for (lane, cell) in (start..end).enumerate() {
                cells[lane] = cell;
            }
            let mut jinv = vec![0.0; npts * w];
            let mut wdet = vec![0.0; npts * w];
            for (lane, &cell) in cells.iter().enumerate() {
                for q in 0..npts {
                    jinv[q * w + lane] = cell_data.jinv_cache[cell * npts + q];
                    wdet[q * w + lane] = cell_data.wdet_cache[cell * npts + q];
                }
            }
            let mut map_idx = Vec::with_capacity(n_sources);
            let mut map_presc = Vec::with_capacity(n_sources);
            for s in 0..n_sources {
                let mut idx = vec![0u32; ndofs * w];
                let mut presc = vec![0.0; ndofs * w];
                for local in 0..ndofs {
                    for (lane, &cell) in cells.iter().enumerate() {
                        let r = restriction.cell_map(s, cell)[local];
                        idx[local * w + lane] = match r {
                            Some(v) => {
                                debug_assert!((v as u64) < u32::MAX as u64);
                                v as u32
                            }
                            None => TENSOR_SENTINEL,
                        };
                        presc[local * w + lane] =
                            restriction.cell_prescribed(s, cell)[local].unwrap_or(0.0);
                    }
                }
                map_idx.push(idx);
                map_presc.push(presc);
            }
            batches.push(TensorBatch {
                cells,
                nreal,
                jinv,
                wdet,
                map_idx,
                map_presc,
            });
        }
        Self {
            batches,
            color_of,
            ndofs,
        }
    }
}

/// Gather state coefficients from packed plan indices using a contiguous slice.
///
/// # Arguments
/// * `batch` - batch with packed indices/prescribed values.
/// * `sources` - restriction source per input position.
/// * `offsets` - global row offset per input position.
/// * `state_slice` - contiguous state column.
/// * `out` - lane-packed coefficients `[(pos*ndofs+local)*W+lane]`. Overwritten.
/// * `ndofs` - local DOFs per cell.
pub(crate) fn gather_state_packed_slice(
    batch: &TensorBatch,
    sources: &[usize],
    offsets: &[usize],
    state_slice: &[f64],
    out: &mut [f64],
    ndofs: usize,
) {
    let w = SIMD_CELL_WIDTH;
    debug_assert_eq!(sources.len(), offsets.len());
    debug_assert_eq!(out.len(), sources.len() * ndofs * w);
    for s in sources {
        debug_assert!(*s < batch.map_idx.len());
        debug_assert_eq!(batch.map_idx[*s].len(), ndofs * w);
        debug_assert_eq!(batch.map_presc[*s].len(), ndofs * w);
    }
    for (pos, (&s, &off)) in sources.iter().zip(offsets.iter()).enumerate() {
        let idx = &batch.map_idx[s];
        let presc = &batch.map_presc[s];
        for local in 0..ndofs {
            let dst = (pos * ndofs + local) * w;
            let src_base = local * w;
            for lane in 0..w {
                let r = idx[src_base + lane];
                // SAFETY: bounds validated by plan construction; inner loop
                // uses unchecked access to avoid bounds checks.
                unsafe {
                    *out.get_unchecked_mut(dst + lane) = if r != TENSOR_SENTINEL {
                        debug_assert!(off + (r as usize) < state_slice.len());
                        *state_slice.get_unchecked(off + r as usize)
                    } else {
                        *presc.get_unchecked(src_base + lane)
                    };
                }
            }
        }
    }
}

/// Gather state coefficients from packed plan indices via generic `MatRef`.
///
/// Same as [`gather_state_packed_slice`] but reads through `state[(row,0)]`
/// for non-contiguous columns; results are bit-identical.
///
/// # Arguments
/// * `batch` - batch with packed indices/prescribed values.
/// * `sources` - restriction source per input position.
/// * `offsets` - global row offset per input position.
/// * `state` - global state (`N×1`).
/// * `out` - lane-packed coefficients. Overwritten.
/// * `ndofs` - local DOFs per cell.
pub(crate) fn gather_state_packed_matref(
    batch: &TensorBatch,
    sources: &[usize],
    offsets: &[usize],
    state: faer::MatRef<'_, f64>,
    out: &mut [f64],
    ndofs: usize,
) {
    let w = SIMD_CELL_WIDTH;
    debug_assert_eq!(sources.len(), offsets.len());
    debug_assert_eq!(out.len(), sources.len() * ndofs * w);
    for s in sources {
        debug_assert!(*s < batch.map_idx.len());
        debug_assert_eq!(batch.map_idx[*s].len(), ndofs * w);
        debug_assert_eq!(batch.map_presc[*s].len(), ndofs * w);
    }
    for (pos, (&s, &off)) in sources.iter().zip(offsets.iter()).enumerate() {
        let idx = &batch.map_idx[s];
        let presc = &batch.map_presc[s];
        for local in 0..ndofs {
            let dst = (pos * ndofs + local) * w;
            let src_base = local * w;
            for lane in 0..w {
                let r = idx[src_base + lane];
                out[dst + lane] = if r != TENSOR_SENTINEL {
                    state[(off + r as usize, 0)]
                } else {
                    presc[src_base + lane]
                };
            }
        }
    }
}

/// Gather state coefficients from packed plan indices.
///
/// Dispatches to the contiguous-slice or generic path; both are bit-identical.
///
/// # Arguments
/// * `batch` - batch with packed indices/prescribed values.
/// * `sources` - restriction source per input position.
/// * `offsets` - global row offset per input position.
/// * `state` - global state (`N×1`).
/// * `out` - lane-packed coefficients. Overwritten.
/// * `ndofs` - local DOFs per cell.
pub(crate) fn gather_state_batch_from_plan(
    batch: &TensorBatch,
    sources: &[usize],
    offsets: &[usize],
    state: faer::MatRef<'_, f64>,
    out: &mut [f64],
    ndofs: usize,
) {
    debug_assert_eq!(sources.len(), offsets.len());
    debug_assert_eq!(out.len(), sources.len() * ndofs * SIMD_CELL_WIDTH);
    if let Some(slice) = state.col(0).try_as_col_major().map(|c| c.as_slice()) {
        gather_state_packed_slice(batch, sources, offsets, slice, out, ndofs);
    } else {
        gather_state_packed_matref(batch, sources, offsets, state, out, ndofs);
    }
}

/// Gather direction coefficients from packed plan indices (eliminated as zero).
///
/// # Arguments
/// * `batch` - batch with packed indices.
/// * `sources` - restriction source per input position.
/// * `offsets` - global row offset per input position.
/// * `direction` - global direction column.
/// * `column` - direction column index.
/// * `out` - lane-packed coefficients. Overwritten.
/// * `ndofs` - local DOFs per cell.
pub(crate) fn gather_direction_batch_from_plan(
    batch: &TensorBatch,
    sources: &[usize],
    offsets: &[usize],
    direction: faer::MatRef<'_, f64>,
    column: usize,
    out: &mut [f64],
    ndofs: usize,
) {
    let w = SIMD_CELL_WIDTH;
    debug_assert_eq!(sources.len(), offsets.len());
    debug_assert_eq!(out.len(), sources.len() * ndofs * w);
    debug_assert!(column < direction.ncols());
    for s in sources {
        debug_assert!(*s < batch.map_idx.len());
        debug_assert_eq!(batch.map_idx[*s].len(), ndofs * w);
    }
    if let Some(slice) = direction
        .col(column)
        .try_as_col_major()
        .map(|c| c.as_slice())
    {
        for (pos, (&s, &off)) in sources.iter().zip(offsets.iter()).enumerate() {
            let idx = &batch.map_idx[s];
            for local in 0..ndofs {
                let dst = (pos * ndofs + local) * w;
                let src_base = local * w;
                for lane in 0..w {
                    let r = unsafe { *idx.get_unchecked(src_base + lane) };
                    unsafe {
                        *out.get_unchecked_mut(dst + lane) = if r != TENSOR_SENTINEL {
                            debug_assert!(off + (r as usize) < slice.len());
                            *slice.get_unchecked(off + r as usize)
                        } else {
                            0.0
                        };
                    }
                }
            }
        }
    } else {
        for (pos, (&s, &off)) in sources.iter().zip(offsets.iter()).enumerate() {
            let idx = &batch.map_idx[s];
            for local in 0..ndofs {
                let dst = (pos * ndofs + local) * w;
                let src_base = local * w;
                for lane in 0..w {
                    let r = idx[src_base + lane];
                    out[dst + lane] = if r != TENSOR_SENTINEL {
                        direction[(off + r as usize, column)]
                    } else {
                        0.0
                    };
                }
            }
        }
    }
}

/// 2D tensor interpolation with lane-packed geometry.
///
/// Dispatches once per call on `n1d` to a const-generic specialization
/// ([`interpolate_batch_2d_packed_n`], register-blocked, bit-identical) for
/// `n1d in 2..=8`, falling back to [`interpolate_batch_2d_packed_dynamic`]
/// otherwise. See [`interpolate_2d_const_inner`] for the contraction layout
/// and the bit-identity argument. Assumes full `W` lanes.
///
/// # Arguments
/// * `cell_data` - tensor data (differentiation, permutation).
/// * `nfields` - field count.
/// * `coeffs` - lane-packed coefficients `[(f*ndofs+d)*W+lane]`.
/// * `values` - lane-packed values `[(f*npts+q)*W+lane]`. Overwritten.
/// * `grads` - lane-packed grads `[((f*2+d)*npts+q)*W+lane]`. Overwritten.
/// * `jinv_packed` - packed inverse Jacobians `[(q*4+k)*W+lane]`.
pub(crate) fn interpolate_batch_2d_packed(
    cell_data: &CellData,
    nfields: usize,
    coeffs: &[f64],
    values: &mut [f64],
    grads: &mut [f64],
    jinv_packed: &[f64],
) {
    let w = SIMD_CELL_WIDTH;
    let npts = cell_data.npts;
    let ndofs = cell_data.ndofs;
    debug_assert_eq!(coeffs.len(), nfields * ndofs * w);
    debug_assert_eq!(values.len(), nfields * npts * w);
    debug_assert_eq!(grads.len(), nfields * 2 * npts * w);
    debug_assert_eq!(jinv_packed.len(), npts * 4 * w);
    let n1d = cell_data
        .tensor
        .as_ref()
        .expect("tensor batch requires tensor data")
        .n1d;
    match n1d {
        2 => interpolate_batch_2d_packed_n::<2>(
            cell_data,
            nfields,
            coeffs,
            values,
            grads,
            jinv_packed,
        ),
        3 => interpolate_batch_2d_packed_n::<3>(
            cell_data,
            nfields,
            coeffs,
            values,
            grads,
            jinv_packed,
        ),
        4 => interpolate_batch_2d_packed_n::<4>(
            cell_data,
            nfields,
            coeffs,
            values,
            grads,
            jinv_packed,
        ),
        5 => interpolate_batch_2d_packed_n::<5>(
            cell_data,
            nfields,
            coeffs,
            values,
            grads,
            jinv_packed,
        ),
        6 => interpolate_batch_2d_packed_n::<6>(
            cell_data,
            nfields,
            coeffs,
            values,
            grads,
            jinv_packed,
        ),
        7 => interpolate_batch_2d_packed_n::<7>(
            cell_data,
            nfields,
            coeffs,
            values,
            grads,
            jinv_packed,
        ),
        8 => interpolate_batch_2d_packed_n::<8>(
            cell_data,
            nfields,
            coeffs,
            values,
            grads,
            jinv_packed,
        ),
        _ => interpolate_batch_2d_packed_dynamic(
            cell_data,
            nfields,
            coeffs,
            values,
            grads,
            jinv_packed,
        ),
    }
}

/// Const-`N` 2D packed interpolation entry point (`N` = `n1d`, `npts` = `N*N`).
///
/// Single pulp `Arch` dispatch into [`interpolate_2d_const_inner`]; FMA runs
/// through `simd.mul_add_f64s` exactly as in the dynamic path.
///
/// # Arguments
/// Same as [`interpolate_batch_2d_packed`]; requires `tensor.n1d == N` and
/// `npts == N*N`.
///
/// # Returns
/// Nothing; `values`/`grads` are overwritten.
pub(crate) fn interpolate_batch_2d_packed_n<const N: usize>(
    cell_data: &CellData,
    nfields: usize,
    coeffs: &[f64],
    values: &mut [f64],
    grads: &mut [f64],
    jinv_packed: &[f64],
) {
    Arch::new().dispatch(InterpolateBatch2dPackedN::<N> {
        cell_data,
        nfields,
        coeffs,
        values,
        grads,
        jinv_packed,
    });
}

/// Dynamic-`n1d` 2D packed interpolation fallback (also used for `n1d > 8`).
///
/// Bitwise reference for the const specializations; unit tests compare
/// against it directly.
///
/// # Arguments
/// Same as [`interpolate_batch_2d_packed`].
///
/// # Returns
/// Nothing; `values`/`grads` are overwritten.
pub(crate) fn interpolate_batch_2d_packed_dynamic(
    cell_data: &CellData,
    nfields: usize,
    coeffs: &[f64],
    values: &mut [f64],
    grads: &mut [f64],
    jinv_packed: &[f64],
) {
    Arch::new().dispatch(InterpolateBatch2dPacked {
        cell_data,
        nfields,
        coeffs,
        values,
        grads,
        jinv_packed,
    });
}

/// Single-dispatch const-`N` 2D packed interpolation (see
/// [`interpolate_batch_2d_packed_n`]).
struct InterpolateBatch2dPackedN<'a, const N: usize> {
    cell_data: &'a CellData,
    nfields: usize,
    coeffs: &'a [f64],
    values: &'a mut [f64],
    grads: &'a mut [f64],
    jinv_packed: &'a [f64],
}

impl<const N: usize> WithSimd for InterpolateBatch2dPackedN<'_, N> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        interpolate_2d_const_inner::<S, N>(
            simd,
            self.cell_data,
            self.nfields,
            self.coeffs,
            self.values,
            self.grads,
            self.jinv_packed,
        );
    }
}

/// Register-blocked const-`N` 2D interpolation kernel.
///
/// Contraction layout: for each output point `q = (j, i)` and each lane-vector,
/// `rx[q] = Σ_a D[i][a]*v[j][a]` and `ry[q] = Σ_a D[j][a]*v[a][i]` accumulate
/// in stack `acc` arrays (`[f64; W]`, `N` const so the `a` loop unrolls) with
/// the same `simd.mul_add_f64s(splat(D), source, acc)` FMA sequence in the same
/// `a = 0..N` order as the dynamic zero-then-`axpy_lanes` loop, then the
/// geometric transform `gx = j0*rx + j2*ry`, `gy = j1*rx + j3*ry` (plain
/// mul/add, same expression order, no `mul_add`) fuses on the store path, so
/// reference gradients never round-trip through `grads` memory. Fixed-size
/// array accesses and `get_unchecked` indexing keep bounds checks out of the
/// inner loops; `grads` is stored exactly once per `(q, dir)`.
///
/// Bit-identity vs the dynamic path: the per-lane FMA stream is
/// `fma(D[0],v[0],0), fma(D[1],v[1],acc), ...` in the same order (starting from
/// `0.0`, `mul_add(s,x,0) == s*x` up to the sign of zero, identically in both),
/// and the transform expressions are textually identical plain arithmetic, so
/// every output lane is bitwise equal.
///
/// # Arguments
/// * `simd` - pulp SIMD token for FMA (`mul_add_f64s`).
/// * `cell_data` - tensor data; requires `tensor.n1d == N`, `npts == N*N`.
/// * `nfields` - field count.
/// * `coeffs` - lane-packed coefficients `[(f*ndofs+d)*W+lane]`.
/// * `values` - lane-packed values `[(f*npts+q)*W+lane]`. Overwritten.
/// * `grads` - lane-packed grads `[((f*2+d)*npts+q)*W+lane]`. Overwritten.
/// * `jinv_packed` - packed inverse Jacobians `[(q*4+k)*W+lane]`.
///
/// # Returns
/// Nothing; `values`/`grads` are overwritten.
#[inline(always)]
fn interpolate_2d_const_inner<S: Simd, const N: usize>(
    simd: S,
    cell_data: &CellData,
    nfields: usize,
    coeffs: &[f64],
    values: &mut [f64],
    grads: &mut [f64],
    jinv_packed: &[f64],
) {
    const W: usize = SIMD_CELL_WIDTH;
    let tensor = cell_data
        .tensor
        .as_ref()
        .expect("tensor batch requires tensor data");
    debug_assert_eq!(tensor.n1d, N);
    let npts = N * N;
    let ndofs = cell_data.ndofs;
    debug_assert_eq!(cell_data.npts, npts);
    let diff = tensor.differentiation.as_slice();
    let q_to_local = tensor.q_to_local.as_slice();
    debug_assert_eq!(diff.len(), N * N);
    debug_assert_eq!(q_to_local.len(), npts);
    let coeffs_ptr = coeffs.as_ptr();
    let values_ptr = values.as_mut_ptr();
    let grads_ptr = grads.as_mut_ptr();
    let jinv_ptr = jinv_packed.as_ptr();
    for f in 0..nfields {
        // Values copy (q_to_local permutation), identical to the dynamic path.
        for q in 0..npts {
            unsafe {
                let local = *q_to_local.get_unchecked(q);
                debug_assert!(local < ndofs);
                let src = coeffs_ptr.add((f * ndofs + local) * W);
                let dst = values_ptr.add((f * npts + q) * W);
                std::ptr::copy_nonoverlapping(src, dst, W);
            }
        }
        for j in 0..N {
            for i in 0..N {
                let q = j * N + i;
                // Register-blocked reference gradients: same FMA order as the
                // dynamic axpy chain, accumulated on the stack, stored once.
                let mut acc_x = [0.0f64; SIMD_CELL_WIDTH];
                let mut acc_y = [0.0f64; SIMD_CELL_WIDTH];
                for a in 0..N {
                    unsafe {
                        let dx = *diff.get_unchecked(i * N + a);
                        let dy = *diff.get_unchecked(j * N + a);
                        let vx = std::slice::from_raw_parts(
                            values_ptr.add((f * npts + j * N + a) * W),
                            W,
                        );
                        let vy = std::slice::from_raw_parts(
                            values_ptr.add((f * npts + a * N + i) * W),
                            W,
                        );
                        axpy_lanes(simd, &mut acc_x, dx, vx);
                        axpy_lanes(simd, &mut acc_y, dy, vy);
                    }
                }
                // Fused geometric transform with the single store per (q, dir).
                unsafe {
                    let jbase = (q * 4) * W;
                    let rx_base = (f * 2 * npts + q) * W;
                    let ry_base = ((f * 2 + 1) * npts + q) * W;
                    for lane in 0..SIMD_CELL_WIDTH {
                        let rx = *acc_x.get_unchecked(lane);
                        let ry = *acc_y.get_unchecked(lane);
                        let j0 = *jinv_ptr.add(jbase + lane);
                        let j1 = *jinv_ptr.add(jbase + W + lane);
                        let j2 = *jinv_ptr.add(jbase + 2 * W + lane);
                        let j3 = *jinv_ptr.add(jbase + 3 * W + lane);
                        *grads_ptr.add(rx_base + lane) = j0 * rx + j2 * ry;
                        *grads_ptr.add(ry_base + lane) = j1 * rx + j3 * ry;
                    }
                }
            }
        }
    }
}

/// Single-dispatch 2D packed interpolation, dynamic `n1d` (see
/// [`interpolate_batch_2d_packed_dynamic`]).
struct InterpolateBatch2dPacked<'a> {
    cell_data: &'a CellData,
    nfields: usize,
    coeffs: &'a [f64],
    values: &'a mut [f64],
    grads: &'a mut [f64],
    jinv_packed: &'a [f64],
}

impl WithSimd for InterpolateBatch2dPacked<'_> {
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
                self.values[(f * npts + q) * w..(f * npts + q) * w + w].copy_from_slice(
                    &self.coeffs[(f * self.cell_data.ndofs + local) * w
                        ..(f * self.cell_data.ndofs + local) * w + w],
                );
            }
            for q in 0..npts {
                self.grads[(f * 2 * npts + q) * w..(f * 2 * npts + q) * w + w].fill(0.0);
                self.grads[((f * 2 + 1) * npts + q) * w..((f * 2 + 1) * npts + q) * w + w]
                    .fill(0.0);
            }
            for j in 0..n1d {
                for i in 0..n1d {
                    let q = j * n1d + i;
                    for a in 0..n1d {
                        axpy_lanes(
                            simd,
                            &mut self.grads[(f * 2 * npts + q) * w..(f * 2 * npts + q) * w + w],
                            tensor.differentiation[i * n1d + a],
                            &self.values
                                [(f * npts + j * n1d + a) * w..(f * npts + j * n1d + a) * w + w],
                        );
                        axpy_lanes(
                            simd,
                            &mut self.grads
                                [((f * 2 + 1) * npts + q) * w..((f * 2 + 1) * npts + q) * w + w],
                            tensor.differentiation[j * n1d + a],
                            &self.values
                                [(f * npts + a * n1d + i) * w..(f * npts + a * n1d + i) * w + w],
                        );
                    }
                }
            }
            // physical transform over all W lanes, plain a*b+c*d, no FMA.
            for q in 0..npts {
                let rx_base = (f * 2 * npts + q) * w;
                let ry_base = ((f * 2 + 1) * npts + q) * w;
                let jbase = (q * 4) * w;
                for lane in 0..w {
                    unsafe {
                        let rx = *self.grads.get_unchecked(rx_base + lane);
                        let ry = *self.grads.get_unchecked(ry_base + lane);
                        let j0 = *self.jinv_packed.get_unchecked(jbase + lane);
                        let j1 = *self.jinv_packed.get_unchecked(jbase + w + lane);
                        let j2 = *self.jinv_packed.get_unchecked(jbase + 2 * w + lane);
                        let j3 = *self.jinv_packed.get_unchecked(jbase + 3 * w + lane);
                        *self.grads.get_unchecked_mut(rx_base + lane) = j0 * rx + j2 * ry;
                        *self.grads.get_unchecked_mut(ry_base + lane) = j1 * rx + j3 * ry;
                    }
                }
            }
        }
    }
}

/// 2D batched integration with lane-packed geometry.
///
/// Dispatches once per call on `n1d` to a const-generic specialization
/// ([`integrate_batch_2d_packed_n`]) for `n1d in 2..=8`, falling back to
/// [`integrate_batch_2d_packed_dynamic`] otherwise. The specialization keeps
/// the current scatter accumulation order bit-identically (see
/// [`integrate_2d_const_inner`]). `out` is accumulated into (`+=`).
///
/// # Arguments
/// * `cell_data` - tensor data.
/// * `noutputs` - equation count.
/// * `f0`/`fx`/`fy` - lane-packed fluxes `[(eq*npts+q)*W+lane]`.
/// * `out` - lane-packed actions `[(eq*ndofs+local)*W+lane]`. Accumulated into.
/// * `wdet_packed` - packed measures `[q*W+lane]`.
/// * `jinv_packed` - packed inverse Jacobians `[(q*4+k)*W+lane]`.
pub(crate) fn integrate_batch_2d_packed(
    cell_data: &CellData,
    noutputs: usize,
    f0: &[f64],
    fx: &[f64],
    fy: &[f64],
    out: &mut [f64],
    wdet_packed: &[f64],
    jinv_packed: &[f64],
) {
    let w = SIMD_CELL_WIDTH;
    let npts = cell_data.npts;
    let ndofs = cell_data.ndofs;
    debug_assert_eq!(f0.len(), noutputs * npts * w);
    debug_assert_eq!(fx.len(), noutputs * npts * w);
    debug_assert_eq!(fy.len(), noutputs * npts * w);
    debug_assert_eq!(out.len(), noutputs * ndofs * w);
    debug_assert_eq!(wdet_packed.len(), npts * w);
    debug_assert_eq!(jinv_packed.len(), npts * 4 * w);
    let n1d = cell_data
        .tensor
        .as_ref()
        .expect("tensor batch needs tensor")
        .n1d;
    match n1d {
        2 => integrate_batch_2d_packed_n::<2>(
            cell_data,
            noutputs,
            f0,
            fx,
            fy,
            out,
            wdet_packed,
            jinv_packed,
        ),
        3 => integrate_batch_2d_packed_n::<3>(
            cell_data,
            noutputs,
            f0,
            fx,
            fy,
            out,
            wdet_packed,
            jinv_packed,
        ),
        4 => integrate_batch_2d_packed_n::<4>(
            cell_data,
            noutputs,
            f0,
            fx,
            fy,
            out,
            wdet_packed,
            jinv_packed,
        ),
        5 => integrate_batch_2d_packed_n::<5>(
            cell_data,
            noutputs,
            f0,
            fx,
            fy,
            out,
            wdet_packed,
            jinv_packed,
        ),
        6 => integrate_batch_2d_packed_n::<6>(
            cell_data,
            noutputs,
            f0,
            fx,
            fy,
            out,
            wdet_packed,
            jinv_packed,
        ),
        7 => integrate_batch_2d_packed_n::<7>(
            cell_data,
            noutputs,
            f0,
            fx,
            fy,
            out,
            wdet_packed,
            jinv_packed,
        ),
        8 => integrate_batch_2d_packed_n::<8>(
            cell_data,
            noutputs,
            f0,
            fx,
            fy,
            out,
            wdet_packed,
            jinv_packed,
        ),
        _ => integrate_batch_2d_packed_dynamic(
            cell_data,
            noutputs,
            f0,
            fx,
            fy,
            out,
            wdet_packed,
            jinv_packed,
        ),
    }
}

/// Const-`N` 2D packed integration entry point (`N` = `n1d`, `npts` = `N*N`).
///
/// Single pulp `Arch` dispatch into [`integrate_2d_const_inner`]; FMA runs
/// through `simd.mul_add_f64s` exactly as in the dynamic path.
///
/// # Arguments
/// Same as [`integrate_batch_2d_packed`]; requires `tensor.n1d == N` and
/// `npts == N*N`.
///
/// # Returns
/// Nothing; `out` is accumulated into (`+=`).
pub(crate) fn integrate_batch_2d_packed_n<const N: usize>(
    cell_data: &CellData,
    noutputs: usize,
    f0: &[f64],
    fx: &[f64],
    fy: &[f64],
    out: &mut [f64],
    wdet_packed: &[f64],
    jinv_packed: &[f64],
) {
    Arch::new().dispatch(IntegrateBatch2dPackedN::<N> {
        cell_data,
        noutputs,
        f0,
        fx,
        fy,
        out,
        wdet_packed,
        jinv_packed,
    });
}

/// Dynamic-`n1d` 2D packed integration fallback (also used for `n1d > 8`).
///
/// Bitwise reference for the const scatter specialization; unit tests compare
/// against it directly.
///
/// # Arguments
/// Same as [`integrate_batch_2d_packed`].
///
/// # Returns
/// Nothing; `out` is accumulated into (`+=`).
pub(crate) fn integrate_batch_2d_packed_dynamic(
    cell_data: &CellData,
    noutputs: usize,
    f0: &[f64],
    fx: &[f64],
    fy: &[f64],
    out: &mut [f64],
    wdet_packed: &[f64],
    jinv_packed: &[f64],
) {
    Arch::new().dispatch(IntegrateBatch2dPacked {
        cell_data,
        noutputs,
        f0,
        fx,
        fy,
        out,
        wdet_packed,
        jinv_packed,
    });
}

/// Single-dispatch const-`N` 2D packed integration (see
/// [`integrate_batch_2d_packed_n`]).
struct IntegrateBatch2dPackedN<'a, const N: usize> {
    cell_data: &'a CellData,
    noutputs: usize,
    f0: &'a [f64],
    fx: &'a [f64],
    fy: &'a [f64],
    out: &'a mut [f64],
    wdet_packed: &'a [f64],
    jinv_packed: &'a [f64],
}

impl<const N: usize> WithSimd for IntegrateBatch2dPackedN<'_, N> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        integrate_2d_const_inner::<S, N>(
            simd,
            self.cell_data,
            self.noutputs,
            self.f0,
            self.fx,
            self.fy,
            self.out,
            self.wdet_packed,
            self.jinv_packed,
        );
    }
}

/// Const-`N` 2D integration kernel with the CURRENT scatter accumulation order.
///
/// For each `(eq, q=(j,i))` in row-major order: per-lane reference fluxes
/// `ref_x = wdet*(j0*fx + j1*fy)`, `ref_y = wdet*(j2*fx + j3*fy)` (plain
/// arithmetic, same expression order as the dynamic path) and the mass term
/// `out[q2l[q]] += wdet*f0` compute in registers, then the transposed
/// contractions `out[q2l[j][a]] += D[i][a]*ref_x` / `out[q2l[a][i]] +=
/// D[j][a]*ref_y` run for `a = 0..N` (x then y per `a`) with the same
/// `axpy_lanes` FMA as today. `N` const lets the `a` loop unroll and keeps
/// bounds checks out of the inner loops; `ref_x`/`ref_y` live on the stack.
///
/// Bit-identical to the dynamic path: loop order, per-term FMA vs plain-op
/// choice, and expression order are all unchanged; only bounds checks are
/// removed.
///
/// # Arguments
/// * `simd` - pulp SIMD token for FMA (`mul_add_f64s`).
/// * `cell_data` - tensor data; requires `tensor.n1d == N`, `npts == N*N`.
/// * `noutputs` - equation count.
/// * `f0`/`fx`/`fy` - lane-packed fluxes `[(eq*npts+q)*W+lane]`.
/// * `out` - lane-packed actions `[(eq*ndofs+local)*W+lane]`. Accumulated into.
/// * `wdet_packed` - packed measures `[q*W+lane]`.
/// * `jinv_packed` - packed inverse Jacobians `[(q*4+k)*W+lane]`.
///
/// # Returns
/// Nothing; `out` is accumulated into (`+=`).
#[inline(always)]
fn integrate_2d_const_inner<S: Simd, const N: usize>(
    simd: S,
    cell_data: &CellData,
    noutputs: usize,
    f0: &[f64],
    fx: &[f64],
    fy: &[f64],
    out: &mut [f64],
    wdet_packed: &[f64],
    jinv_packed: &[f64],
) {
    const W: usize = SIMD_CELL_WIDTH;
    let tensor = cell_data
        .tensor
        .as_ref()
        .expect("tensor batch needs tensor");
    debug_assert_eq!(tensor.n1d, N);
    let npts = N * N;
    let ndofs = cell_data.ndofs;
    debug_assert_eq!(cell_data.npts, npts);
    let diff = tensor.differentiation.as_slice();
    let q_to_local = tensor.q_to_local.as_slice();
    debug_assert_eq!(diff.len(), N * N);
    debug_assert_eq!(q_to_local.len(), npts);
    let f0_ptr = f0.as_ptr();
    let fx_ptr = fx.as_ptr();
    let fy_ptr = fy.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let wdet_ptr = wdet_packed.as_ptr();
    let jinv_ptr = jinv_packed.as_ptr();
    for eq in 0..noutputs {
        for j in 0..N {
            for i in 0..N {
                let q = j * N + i;
                let mut ref_x = [0.0f64; SIMD_CELL_WIDTH];
                let mut ref_y = [0.0f64; SIMD_CELL_WIDTH];
                unsafe {
                    let mass_local = *q_to_local.get_unchecked(q);
                    debug_assert!(mass_local < ndofs);
                    for lane in 0..SIMD_CELL_WIDTH {
                        let wdet = *wdet_ptr.add(q * W + lane);
                        let jbase = (q * 4) * W;
                        let j0 = *jinv_ptr.add(jbase + lane);
                        let j1 = *jinv_ptr.add(jbase + W + lane);
                        let j2 = *jinv_ptr.add(jbase + 2 * W + lane);
                        let j3 = *jinv_ptr.add(jbase + 3 * W + lane);
                        let f1x = *fx_ptr.add((eq * npts + q) * W + lane);
                        let f1y = *fy_ptr.add((eq * npts + q) * W + lane);
                        *ref_x.get_unchecked_mut(lane) = wdet * (j0 * f1x + j1 * f1y);
                        *ref_y.get_unchecked_mut(lane) = wdet * (j2 * f1x + j3 * f1y);
                        let dst = out_ptr.add((eq * ndofs + mass_local) * W + lane);
                        *dst += wdet * *f0_ptr.add((eq * npts + q) * W + lane);
                    }
                }
                for a in 0..N {
                    unsafe {
                        let xl = *q_to_local.get_unchecked(j * N + a);
                        debug_assert!(xl < ndofs);
                        let xdst =
                            std::slice::from_raw_parts_mut(out_ptr.add((eq * ndofs + xl) * W), W);
                        axpy_lanes(simd, xdst, *diff.get_unchecked(i * N + a), &ref_x);
                        let yl = *q_to_local.get_unchecked(a * N + i);
                        debug_assert!(yl < ndofs);
                        let ydst =
                            std::slice::from_raw_parts_mut(out_ptr.add((eq * ndofs + yl) * W), W);
                        axpy_lanes(simd, ydst, *diff.get_unchecked(j * N + a), &ref_y);
                    }
                }
            }
        }
    }
}

/// Single-dispatch 2D packed integration, dynamic `n1d` (see
/// [`integrate_batch_2d_packed_dynamic`]).
struct IntegrateBatch2dPacked<'a> {
    cell_data: &'a CellData,
    noutputs: usize,
    f0: &'a [f64],
    fx: &'a [f64],
    fy: &'a [f64],
    out: &'a mut [f64],
    wdet_packed: &'a [f64],
    jinv_packed: &'a [f64],
}

impl WithSimd for IntegrateBatch2dPacked<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let w = SIMD_CELL_WIDTH;
        let tensor = self
            .cell_data
            .tensor
            .as_ref()
            .expect("tensor batch needs tensor");
        let n1d = tensor.n1d;
        let npts = self.cell_data.npts;
        for eq in 0..self.noutputs {
            for j in 0..n1d {
                for i in 0..n1d {
                    let q = j * n1d + i;
                    let mut ref_x = [0.0f64; SIMD_CELL_WIDTH];
                    let mut ref_y = [0.0f64; SIMD_CELL_WIDTH];
                    for lane in 0..w {
                        unsafe {
                            let wdet = *self.wdet_packed.get_unchecked(q * w + lane);
                            let jbase = (q * 4) * w;
                            let j0 = *self.jinv_packed.get_unchecked(jbase + lane);
                            let j1 = *self.jinv_packed.get_unchecked(jbase + w + lane);
                            let j2 = *self.jinv_packed.get_unchecked(jbase + 2 * w + lane);
                            let j3 = *self.jinv_packed.get_unchecked(jbase + 3 * w + lane);
                            let f1x = *self.fx.get_unchecked((eq * npts + q) * w + lane);
                            let f1y = *self.fy.get_unchecked((eq * npts + q) * w + lane);
                            ref_x[lane] = wdet * (j0 * f1x + j1 * f1y);
                            ref_y[lane] = wdet * (j2 * f1x + j3 * f1y);
                            *self.out.get_unchecked_mut(
                                (eq * self.cell_data.ndofs + tensor.q_to_local[q]) * w + lane,
                            ) += wdet * *self.f0.get_unchecked((eq * npts + q) * w + lane);
                        }
                    }
                    for a in 0..n1d {
                        let xl = tensor.q_to_local[j * n1d + a];
                        axpy_lanes(
                            simd,
                            &mut self.out[(eq * self.cell_data.ndofs + xl) * w
                                ..(eq * self.cell_data.ndofs + xl) * w + w],
                            tensor.differentiation[i * n1d + a],
                            &ref_x[..w],
                        );
                        let yl = tensor.q_to_local[a * n1d + i];
                        axpy_lanes(
                            simd,
                            &mut self.out[(eq * self.cell_data.ndofs + yl) * w
                                ..(eq * self.cell_data.ndofs + yl) * w + w],
                            tensor.differentiation[j * n1d + a],
                            &ref_y[..w],
                        );
                    }
                }
            }
        }
    }
}

/// Default cap (MiB) for the interpolated batch-state cache.
///
/// The lane-packed values+grads for all batches (`≈ ncells·ninputs·3·npts·8`
/// bytes) only pay off while they fit comfortably in the last-level cache;
/// beyond that, streaming them each apply is slower than recomputing by fresh
/// gather+interpolate. Overridable via `ORMATEX_TENSOR_STATE_CACHE_MB`.
pub(crate) const TENSOR_STATE_CACHE_DEFAULT_MB: u64 = 8;

/// Interpolated-state byte budget parsed once from `ORMATEX_TENSOR_STATE_CACHE_MB`.
///
/// Falls back to [`TENSOR_STATE_CACHE_DEFAULT_MB`] when unset or unparsable;
/// `0` disables the interpolated cache (the owned state copy is still kept).
///
/// # Returns
/// Budget in MiB.
fn tensor_state_cache_limit_mb() -> u64 {
    static LIMIT_MB: OnceLock<u64> = OnceLock::new();
    *LIMIT_MB.get_or_init(|| {
        std::env::var("ORMATEX_TENSOR_STATE_CACHE_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(TENSOR_STATE_CACHE_DEFAULT_MB)
    })
}

/// Interpolated-state byte budget for [`TensorStateCache::batch_states`].
///
/// # Returns
/// Budget in bytes (`MiB * 1024 * 1024`); `0` disables the interpolated cache.
pub(crate) fn tensor_state_cache_limit_bytes() -> u64 {
    tensor_state_cache_limit_mb().saturating_mul(1024 * 1024)
}

/// Exact interpolated-state footprint for one cache build.
///
/// # Arguments
/// * `nbatches` - batch count of the flat natural-order plan.
/// * `ninputs` - state field count.
/// * `npts` - quadrature points per cell.
/// * `gdim` - geometric dimension (gradient components per field).
///
/// # Returns
/// `nbatches·ninputs·(1+gdim)·npts·W·8` bytes (values plus gradients).
pub(crate) fn tensor_interpolated_state_bytes_gdim(
    nbatches: usize,
    ninputs: usize,
    npts: usize,
    gdim: usize,
) -> u64 {
    nbatches as u64 * ninputs as u64 * (1 + gdim) as u64 * npts as u64 * SIMD_CELL_WIDTH as u64 * 8
}

/// Exact interpolated-state footprint for one 2D cache build.
///
/// Delegates to [`tensor_interpolated_state_bytes_gdim`] with `gdim == 2`;
/// kept so the 2D path is untouched.
///
/// # Arguments
/// * `nbatches` - batch count of the flat natural-order plan.
/// * `ninputs` - state field count.
/// * `npts` - quadrature points per cell.
///
/// # Returns
/// `nbatches·ninputs·3·npts·W·8` bytes (values plus two gradient components).
pub(crate) fn tensor_interpolated_state_bytes(nbatches: usize, ninputs: usize, npts: usize) -> u64 {
    tensor_interpolated_state_bytes_gdim(nbatches, ninputs, npts, 2)
}

/// Cached lane-packed linearization state for one batch.
///
/// `values`/`grads` use the lane-packed layouts produced by
/// [`interpolate_batch_2d_packed`].
pub(crate) struct CachedBatchState {
    /// Lane-packed values `[(f*npts+q)*W+lane]`.
    pub values: Vec<f64>,
    /// Lane-packed grads `[((f*2+d)*npts+q)*W+lane]`.
    pub grads: Vec<f64>,
}

/// Owned linearization-state cache for tensor Jacobian actions.
///
/// Holds a copy of the state vector plus, when small enough to stay useful,
/// per-batch lane-packed values/grads, computed once in
/// `prepare_linearization` and reused across `apply_jacobian` calls when the
/// state matches bitwise. There is no pointer fast path: a caller that
/// mutates the state buffer in place must observe a miss and recompute.
///
/// The interpolated states are only built when their footprint
/// ([`tensor_interpolated_state_bytes`]) fits the budget from
/// [`tensor_state_cache_limit_bytes`]; otherwise the prepared apply path
/// re-interpolates from the owned copy (still no state comparison, still
/// fused boundary/epilogue), bit-identically.
pub(crate) struct TensorStateCache {
    /// Copy of the linearized state vector.
    pub state_copy: Vec<f64>,
    /// State rows at prepare time.
    pub nrows: usize,
    /// State cols at prepare time.
    pub ncols: usize,
    /// Resolved input field ids the cache was built for.
    pub inputs: Vec<usize>,
    /// Per-batch lane-packed states in flat natural batch order, or `None`
    /// when above the size budget (prepared path re-interpolates instead).
    pub batch_states: Option<Vec<CachedBatchState>>,
}

impl TensorStateCache {
    /// Whether the interpolated per-batch states were built.
    ///
    /// # Returns
    /// `true` when the prepared path can reuse cached interpolation.
    pub(crate) fn has_batch_states(&self) -> bool {
        self.batch_states.is_some()
    }
    /// Check whether `state` matches the cached linearization point.
    ///
    /// Always compares bitwise against the stored copy: a tight
    /// `to_bits` loop over contiguous slices, or indexed access when the
    /// state column is non-contiguous.
    ///
    /// # Arguments
    /// * `state` - candidate state (`N×1`).
    /// * `inputs` - current resolved input field ids; must equal the cached ids.
    ///
    /// # Returns
    /// `true` only on full bitwise equality of field ids, shape, and values.
    pub(crate) fn matches(&self, state: faer::MatRef<'_, f64>, inputs: &[usize]) -> bool {
        if inputs != self.inputs.as_slice() {
            return false;
        }
        if state.nrows() != self.nrows || state.ncols() != self.ncols {
            return false;
        }
        if let Some(slice) = state.col(0).try_as_col_major().map(|c| c.as_slice()) {
            if slice.len() != self.state_copy.len() {
                return false;
            }
            for (a, b) in slice.iter().zip(self.state_copy.iter()) {
                if a.to_bits() != b.to_bits() {
                    return false;
                }
            }
            true
        } else {
            for r in 0..state.nrows() {
                if state[(r, 0)].to_bits() != self.state_copy[r].to_bits() {
                    return false;
                }
            }
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        integrate_batch_1d_packed_dynamic, integrate_batch_1d_packed_n,
        integrate_batch_2d_packed_dynamic, integrate_batch_2d_packed_n,
        interpolate_batch_1d_packed_dynamic, interpolate_batch_1d_packed_n,
        interpolate_batch_2d_packed_dynamic, interpolate_batch_2d_packed_n, SIMD_CELL_WIDTH,
    };
    use crate::common::cell::{CellData, TensorProductData};
    use rlst::DynArray;

    #[test]
    /// Guards the layout constants the batch helpers hard-code.
    fn lane_layout_helpers_smoke() {
        assert_eq!(SIMD_CELL_WIDTH, 8);
    }

    /// Minimal `CellData` carrying only what the 2D packed kernels read
    /// (`npts`, `ndofs`, `tensor`); remaining caches are empty.
    ///
    /// # Arguments
    /// * `n1d` - 1D GLL points per direction.
    /// * `differentiation` - row-major `n1d x n1d` matrix.
    /// * `q_to_local` - quadrature-to-local permutation (`npts` entries).
    ///
    /// # Returns
    /// `CellData` with `ndofs = npts = n1d*n1d`.
    fn test_cell_data_2d(
        n1d: usize,
        differentiation: Vec<f64>,
        q_to_local: Vec<usize>,
    ) -> CellData {
        let npts = n1d * n1d;
        CellData {
            wts: vec![1.0; npts],
            npts,
            ndofs: npts,
            table: DynArray::<f64, 4>::from_shape([1, 1, 1, 1]),
            reference_values: vec![0.0; npts * npts],
            nodal_quadrature: (0..npts).collect(),
            jinv_cache: Vec::new(),
            jdets_cache: Vec::new(),
            wdet_cache: Vec::new(),
            cell_sizes: Vec::new(),
            physical_points_cache: Vec::new(),
            tensor: Some(TensorProductData {
                n1d,
                differentiation,
                q_to_local,
            }),
        }
    }

    /// SplitMix64 step for deterministic test randomness (no extra deps).
    ///
    /// # Arguments
    /// * `state` - in-place RNG state.
    ///
    /// # Returns
    /// Next `u64` draw.
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform `f64` in `[-1, 1)` from SplitMix64.
    ///
    /// # Arguments
    /// * `state` - in-place RNG state.
    ///
    /// # Returns
    /// Pseudorandom double in `[-1, 1)`.
    fn rand_unit(state: &mut u64) -> f64 {
        let bits = splitmix64(state) >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64)) * 2.0 - 1.0
    }

    /// Bitwise comparison of the const-`N` interpolate path vs the dynamic one.
    ///
    /// Uses random coefficients, a dense random differentiation matrix, a
    /// reversed `q_to_local` permutation (non-identity mapping), and random
    /// packed inverse Jacobians with O(1) entries.
    ///
    /// # Arguments
    /// * `seed` - RNG seed (varied per `N` by the caller).
    fn check_interpolate_const_bitwise<const N: usize>(seed: u64) {
        const W: usize = SIMD_CELL_WIDTH;
        let npts = N * N;
        let mut rng = seed;
        let differentiation: Vec<f64> = (0..N * N).map(|_| rand_unit(&mut rng)).collect();
        // Reversed permutation exercises non-identity q_to_local.
        let q_to_local: Vec<usize> = (0..npts).rev().collect();
        let cell_data = test_cell_data_2d(N, differentiation, q_to_local);
        let nfields = 3;
        let coeffs: Vec<f64> = (0..nfields * npts * W)
            .map(|_| rand_unit(&mut rng))
            .collect();
        let jinv: Vec<f64> = (0..npts * 4 * W)
            .map(|_| 0.5 + 0.5 * rand_unit(&mut rng))
            .collect();
        let mut values_c = vec![7.0; nfields * npts * W];
        let mut grads_c = vec![7.0; nfields * 2 * npts * W];
        let mut values_d = values_c.clone();
        let mut grads_d = grads_c.clone();
        interpolate_batch_2d_packed_n::<N>(
            &cell_data,
            nfields,
            &coeffs,
            &mut values_c,
            &mut grads_c,
            &jinv,
        );
        interpolate_batch_2d_packed_dynamic(
            &cell_data,
            nfields,
            &coeffs,
            &mut values_d,
            &mut grads_d,
            &jinv,
        );
        assert_eq!(values_c.len(), values_d.len());
        assert_eq!(grads_c.len(), grads_d.len());
        for (a, b) in values_c.iter().zip(values_d.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "values differ bitwise");
        }
        for (a, b) in grads_c.iter().zip(grads_d.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "grads differ bitwise");
        }
    }

    /// Bitwise comparison of the const-`N` scatter integrate path vs dynamic.
    ///
    /// Random fluxes/geometry plus random nonzero initial `out` (both paths
    /// accumulate into the same starting `out`, testing accumulation order).
    ///
    /// # Arguments
    /// * `seed` - RNG seed (varied per `N` by the caller).
    fn check_integrate_const_bitwise<const N: usize>(seed: u64) {
        const W: usize = SIMD_CELL_WIDTH;
        let npts = N * N;
        let mut rng = seed;
        let differentiation: Vec<f64> = (0..N * N).map(|_| rand_unit(&mut rng)).collect();
        let q_to_local: Vec<usize> = (0..npts).rev().collect();
        let cell_data = test_cell_data_2d(N, differentiation, q_to_local);
        let noutputs = 2;
        let f0: Vec<f64> = (0..noutputs * npts * W)
            .map(|_| rand_unit(&mut rng))
            .collect();
        let fx: Vec<f64> = (0..noutputs * npts * W)
            .map(|_| rand_unit(&mut rng))
            .collect();
        let fy: Vec<f64> = (0..noutputs * npts * W)
            .map(|_| rand_unit(&mut rng))
            .collect();
        let wdet: Vec<f64> = (0..npts * W)
            .map(|_| 0.1 + rand_unit(&mut rng).abs())
            .collect();
        let jinv: Vec<f64> = (0..npts * 4 * W)
            .map(|_| 0.5 + 0.5 * rand_unit(&mut rng))
            .collect();
        let out_init: Vec<f64> = (0..noutputs * npts * W)
            .map(|_| rand_unit(&mut rng))
            .collect();
        let mut out_c = out_init.clone();
        let mut out_d = out_init.clone();
        integrate_batch_2d_packed_n::<N>(
            &cell_data, noutputs, &f0, &fx, &fy, &mut out_c, &wdet, &jinv,
        );
        integrate_batch_2d_packed_dynamic(
            &cell_data, noutputs, &f0, &fx, &fy, &mut out_d, &wdet, &jinv,
        );
        assert_eq!(out_c.len(), out_d.len());
        for (a, b) in out_c.iter().zip(out_d.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "integrate out differs bitwise");
        }
    }

    #[test]
    /// Const-`N` interpolate paths are bitwise equal to the dynamic fallback.
    fn interpolate_const_specialization_bitwise() {
        check_interpolate_const_bitwise::<2>(0x1234);
        check_interpolate_const_bitwise::<3>(0x1235);
        check_interpolate_const_bitwise::<4>(0x1236);
        check_interpolate_const_bitwise::<5>(0x1237);
        check_interpolate_const_bitwise::<6>(0x1238);
        check_interpolate_const_bitwise::<7>(0x1239);
        check_interpolate_const_bitwise::<8>(0x123A);
    }

    #[test]
    /// Const-`N` scatter integrate paths are bitwise equal to the dynamic fallback.
    fn integrate_const_specialization_bitwise() {
        check_integrate_const_bitwise::<2>(0xABCD);
        check_integrate_const_bitwise::<3>(0xABCE);
        check_integrate_const_bitwise::<4>(0xABCF);
        check_integrate_const_bitwise::<5>(0xABD0);
        check_integrate_const_bitwise::<6>(0xABD1);
        check_integrate_const_bitwise::<7>(0xABD2);
        check_integrate_const_bitwise::<8>(0xABD3);
    }

    /// Minimal `CellData` carrying only what the 1D packed kernels read
    /// (`npts`, `ndofs`, `tensor`); remaining caches are empty.
    ///
    /// # Arguments
    /// * `n1d` - 1D GLL points (`npts == ndofs == n1d`).
    /// * `differentiation` - row-major `n1d x n1d` matrix.
    /// * `q_to_local` - quadrature-to-local permutation (`n1d` entries).
    ///
    /// # Returns
    /// `CellData` with `ndofs = npts = n1d`.
    fn test_cell_data_1d(
        n1d: usize,
        differentiation: Vec<f64>,
        q_to_local: Vec<usize>,
    ) -> CellData {
        CellData {
            wts: vec![1.0; n1d],
            npts: n1d,
            ndofs: n1d,
            table: DynArray::<f64, 4>::from_shape([1, 1, 1, 1]),
            reference_values: vec![0.0; n1d * n1d],
            nodal_quadrature: (0..n1d).collect(),
            jinv_cache: Vec::new(),
            jdets_cache: Vec::new(),
            wdet_cache: Vec::new(),
            cell_sizes: Vec::new(),
            physical_points_cache: Vec::new(),
            tensor: Some(TensorProductData {
                n1d,
                differentiation,
                q_to_local,
            }),
        }
    }

    /// Bitwise comparison of the const-`N` 1D interpolate path vs dynamic.
    ///
    /// Uses random coefficients, a dense random differentiation matrix, a
    /// reversed `q_to_local` permutation (non-identity mapping), and random
    /// packed scalar inverse Jacobians with O(1) entries.
    ///
    /// # Arguments
    /// * `seed` - RNG seed (varied per `N` by the caller).
    fn check_interpolate_1d_const_bitwise<const N: usize>(seed: u64) {
        const W: usize = SIMD_CELL_WIDTH;
        let mut rng = seed;
        let differentiation: Vec<f64> = (0..N * N).map(|_| rand_unit(&mut rng)).collect();
        // Reversed permutation exercises non-identity q_to_local.
        let q_to_local: Vec<usize> = (0..N).rev().collect();
        let cell_data = test_cell_data_1d(N, differentiation, q_to_local);
        let nfields = 3;
        let coeffs: Vec<f64> = (0..nfields * N * W).map(|_| rand_unit(&mut rng)).collect();
        let jinv: Vec<f64> = (0..N * W)
            .map(|_| 0.5 + 0.5 * rand_unit(&mut rng))
            .collect();
        let mut values_c = vec![7.0; nfields * N * W];
        let mut grads_c = vec![7.0; nfields * N * W];
        let mut values_d = values_c.clone();
        let mut grads_d = grads_c.clone();
        interpolate_batch_1d_packed_n::<N>(
            &cell_data,
            nfields,
            &coeffs,
            &mut values_c,
            &mut grads_c,
            &jinv,
        );
        interpolate_batch_1d_packed_dynamic(
            &cell_data,
            nfields,
            &coeffs,
            &mut values_d,
            &mut grads_d,
            &jinv,
        );
        assert_eq!(values_c.len(), values_d.len());
        assert_eq!(grads_c.len(), grads_d.len());
        for (a, b) in values_c.iter().zip(values_d.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "values differ bitwise");
        }
        for (a, b) in grads_c.iter().zip(grads_d.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "grads differ bitwise");
        }
    }

    /// Bitwise comparison of the const-`N` 1D integrate path vs dynamic.
    ///
    /// Random fluxes/geometry plus random nonzero initial `out` (both paths
    /// accumulate into the same starting `out`, testing accumulation order).
    ///
    /// # Arguments
    /// * `seed` - RNG seed (varied per `N` by the caller).
    fn check_integrate_1d_const_bitwise<const N: usize>(seed: u64) {
        const W: usize = SIMD_CELL_WIDTH;
        let mut rng = seed;
        let differentiation: Vec<f64> = (0..N * N).map(|_| rand_unit(&mut rng)).collect();
        let q_to_local: Vec<usize> = (0..N).rev().collect();
        let cell_data = test_cell_data_1d(N, differentiation, q_to_local);
        let noutputs = 2;
        let f0: Vec<f64> = (0..noutputs * N * W).map(|_| rand_unit(&mut rng)).collect();
        let f1: Vec<f64> = (0..noutputs * N * W).map(|_| rand_unit(&mut rng)).collect();
        let wdet: Vec<f64> = (0..N * W)
            .map(|_| 0.1 + rand_unit(&mut rng).abs())
            .collect();
        let jinv: Vec<f64> = (0..N * W)
            .map(|_| 0.5 + 0.5 * rand_unit(&mut rng))
            .collect();
        let out_init: Vec<f64> = (0..noutputs * N * W).map(|_| rand_unit(&mut rng)).collect();
        let mut out_c = out_init.clone();
        let mut out_d = out_init.clone();
        integrate_batch_1d_packed_n::<N>(&cell_data, noutputs, &f0, &f1, &mut out_c, &wdet, &jinv);
        integrate_batch_1d_packed_dynamic(&cell_data, noutputs, &f0, &f1, &mut out_d, &wdet, &jinv);
        assert_eq!(out_c.len(), out_d.len());
        for (a, b) in out_c.iter().zip(out_d.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "integrate out differs bitwise");
        }
    }

    #[test]
    /// Const-`N` 1D interpolate paths are bitwise equal to the dynamic fallback.
    fn interpolate_1d_const_specialization_bitwise() {
        check_interpolate_1d_const_bitwise::<2>(0x2234);
        check_interpolate_1d_const_bitwise::<3>(0x2235);
        check_interpolate_1d_const_bitwise::<4>(0x2236);
        check_interpolate_1d_const_bitwise::<5>(0x2237);
        check_interpolate_1d_const_bitwise::<6>(0x2238);
        check_interpolate_1d_const_bitwise::<7>(0x2239);
        check_interpolate_1d_const_bitwise::<8>(0x223A);
    }

    #[test]
    /// Const-`N` 1D integrate paths are bitwise equal to the dynamic fallback.
    fn integrate_1d_const_specialization_bitwise() {
        check_integrate_1d_const_bitwise::<2>(0xBBCD);
        check_integrate_1d_const_bitwise::<3>(0xBBCE);
        check_integrate_1d_const_bitwise::<4>(0xBBCF);
        check_integrate_1d_const_bitwise::<5>(0xBBD0);
        check_integrate_1d_const_bitwise::<6>(0xBBD1);
        check_integrate_1d_const_bitwise::<7>(0xBBD2);
        check_integrate_1d_const_bitwise::<8>(0xBBD3);
    }
}
