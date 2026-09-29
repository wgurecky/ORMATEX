//! Shared sum-factorized tensor volume pass (1D/2D).
//!
//! The 1D (`SEM1DProblem`) and 2D (`SEM2DProblem`) tensor drivers were
//! near-verbatim copies differing only in the `TensorResidualKernel<GDIM>`
//! bound, the packed interpolate/integrate calls, gdim-scaled scratch sizes,
//! and the problem type. This module hoists that skeleton behind the
//! [`TensorVolumeAdapter`] trait so the arithmetic appears exactly once.
//! `SEM1DProblem` implements `TensorVolumeAdapter<1>` and `SEM2DProblem`
//! implements `TensorVolumeAdapter<2>`; the `sem_1d/tensor.rs` and
//! `sem_2d/tensor.rs` files keep thin adapters plus public entry points.
//! Static dispatch only (no `dyn` in hot loops).

use std::sync::{Arc, Mutex, RwLock};

use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};
use rayon::prelude::*;

use super::batch::{
    gather_direction_batch_from_plan, gather_state_batch_from_plan,
    integrate_batch_1d_packed, integrate_batch_2d_packed, interpolate_batch_1d_packed,
    interpolate_batch_2d_packed, tensor_interpolated_state_bytes,
    tensor_interpolated_state_bytes_gdim, with_tensor_worker_scratch, CachedBatchState,
    DisjointSortedEvec, SortedRowScatter, TensorBatch, TensorBatchPlan, TensorStateCache,
    TensorWorkerScratch, SIMD_CELL_WIDTH,
};
use super::cell::CellData;
use super::jacobian_pattern::JacobianPatternCache;
use super::reduction::FieldDofLayout;
use super::restriction::ElementRestriction;
use super::{push_rectangular_local_matrix_triplets, LaneState, Lanes, TensorCtx, LANES};
use crate::fields::FieldRegistry;
use crate::jacobian::RowEpilogue;
use crate::kernels::common::TensorResidualKernel;

/// Cap for the reusable row-sorted E-vector scratch pool.
///
/// Bounds memory while keeping steady-state tensor passes allocation-free
/// (one `nslots * ncols` buffer per in-flight pass).
pub(crate) const EVEC_POOL_CAP: usize = 8;

/// State source for one tensor batch.
///
/// `Fresh` gathers and interpolates the state into scratch inside the batch;
/// `Cached` borrows the lane-packed values/grads prepared once by the
/// state-cache builder.
#[derive(Clone, Copy)]
pub(crate) enum BatchStateInput<'a> {
    /// Linearization state to gather and interpolate (`N×1`).
    Fresh(faer::MatRef<'a, f64>),
    /// Pre-interpolated lane-packed state for this batch.
    Cached(&'a CachedBatchState),
}

/// Bundled phase-1 arguments for [`phase1_fill_sorted`].
///
/// Groups the per-pass arguments (besides the kernel) for the phase-1 pass,
/// so dispatch takes them once instead of repeating the full argument list
/// per call.
pub(crate) struct TensorPhase1Ctx<'a, 'e> {
    /// Evaluation time.
    pub time: f64,
    /// State field count.
    pub ninputs: usize,
    /// Equation count.
    pub noutputs: usize,
    /// Restriction source per input position.
    pub sources: &'a [usize],
    /// Global row offset per input position.
    pub input_offsets: &'a [usize],
    /// Linearization state, or `None` when cached.
    pub fresh_state: Option<faer::MatRef<'a, f64>>,
    /// Linearization cache, or `None` when fresh.
    pub cache: Option<&'a TensorStateCache>,
    /// Global directions, or `None` for the residual.
    pub direction: Option<faer::MatRef<'a, f64>>,
    /// Row-sorted destination table.
    pub scatter: &'a SortedRowScatter,
    /// Row-sorted E-vector (`nslots * ncols`). Fully overwritten.
    pub evec_sorted: &'e mut [f64],
}

/// Per-dimension adapter for the shared tensor volume pass.
///
/// Implemented by `SEM1DProblem` (`GDIM = 1`) and `SEM2DProblem`
/// (`GDIM = 2`). All hot-loop dispatch is static via the `GDIM` const
/// parameter and the `interpolate_packed` / `integrate_packed` fns.
pub(crate) trait TensorVolumeAdapter<const GDIM: usize> {
    /// Access cached cell geometry and tensor-product tables.
    fn cell_data(&self) -> &CellData;
    /// Access the element restriction (maps, coloring, offsets).
    fn restriction(&self) -> &ElementRestriction;
    /// Access the flat natural-order lane batch plan.
    fn batch_plan(&self) -> &TensorBatchPlan;
    /// Access the reusable row-sorted E-vector pool.
    fn evec_pool(&self) -> &Mutex<Vec<Vec<f64>>>;
    /// Access the lazily built row-sorted scatter cache.
    fn scatter_cache(&self) -> &RwLock<Vec<(Vec<usize>, Arc<SortedRowScatter>)>>;
    /// Access the assembled-Jacobian CSC pattern cache.
    fn pattern_cache(&self) -> &JacobianPatternCache;
    /// Access the ordered field registry.
    fn fields(&self) -> &FieldRegistry;
    /// Build the pointwise tensor context for one cell.
    ///
    /// # Arguments
    /// * `time` - evaluation time.
    /// * `cell` - global cell index.
    fn tensor_ctx(&self, time: f64, cell: usize) -> TensorCtx<'_>;
    /// Lane-packed interpolation for this dimension.
    ///
    /// # Arguments
    /// * `cd` - tensor cell data.
    /// * `ninputs` - field count.
    /// * `coeffs` - lane-packed coefficients.
    /// * `values` - lane-packed values. Overwritten.
    /// * `grads` - lane-packed grads. Overwritten.
    /// * `jinv` - packed inverse Jacobians.
    fn interpolate_packed(
        cd: &CellData,
        ninputs: usize,
        coeffs: &[f64],
        values: &mut [f64],
        grads: &mut [f64],
        jinv: &[f64],
    );
    /// Lane-packed integration for this dimension.
    ///
    /// Takes all three flux slots; the 1D implementation ignores `f1y`
    /// (still evaluated as zeros per the lane contract).
    ///
    /// # Arguments
    /// * `cd` - tensor cell data.
    /// * `noutputs` - equation count.
    /// * `f0`/`f1x`/`f1y` - lane-packed fluxes.
    /// * `out` - lane-packed actions. Accumulated into.
    /// * `wdet` - packed measures.
    /// * `jinv` - packed inverse Jacobians.
    #[allow(clippy::too_many_arguments)]
    fn integrate_packed(
        cd: &CellData,
        noutputs: usize,
        f0: &[f64],
        f1x: &[f64],
        f1y: &[f64],
        out: &mut [f64],
        wdet: &[f64],
        jinv: &[f64],
    );
}

/// 1D packed interpolation adapter.
///
/// # Arguments
/// Same as [`TensorVolumeAdapter::interpolate_packed`].
pub(crate) fn interpolate_packed_1d(
    cd: &CellData,
    ninputs: usize,
    coeffs: &[f64],
    values: &mut [f64],
    grads: &mut [f64],
    jinv: &[f64],
) {
    interpolate_batch_1d_packed(cd, ninputs, coeffs, values, grads, jinv);
}

/// 2D packed interpolation adapter.
///
/// # Arguments
/// Same as [`TensorVolumeAdapter::interpolate_packed`].
pub(crate) fn interpolate_packed_2d(
    cd: &CellData,
    ninputs: usize,
    coeffs: &[f64],
    values: &mut [f64],
    grads: &mut [f64],
    jinv: &[f64],
) {
    interpolate_batch_2d_packed(cd, ninputs, coeffs, values, grads, jinv);
}

/// 1D packed integration adapter (ignores `f1y`).
///
/// # Arguments
/// Same as [`TensorVolumeAdapter::integrate_packed`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn integrate_packed_1d(
    cd: &CellData,
    noutputs: usize,
    f0: &[f64],
    f1x: &[f64],
    _f1y: &[f64],
    out: &mut [f64],
    wdet: &[f64],
    jinv: &[f64],
) {
    integrate_batch_1d_packed(cd, noutputs, f0, f1x, out, wdet, jinv);
}

/// 2D packed integration adapter.
///
/// # Arguments
/// Same as [`TensorVolumeAdapter::integrate_packed`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn integrate_packed_2d(
    cd: &CellData,
    noutputs: usize,
    f0: &[f64],
    f1x: &[f64],
    f1y: &[f64],
    out: &mut [f64],
    wdet: &[f64],
    jinv: &[f64],
) {
    integrate_batch_2d_packed(cd, noutputs, f0, f1x, f1y, out, wdet, jinv);
}

/// Lane-packed pointwise flux evaluation for one tensor batch (residual).
///
/// The single supported pointwise path: [`TensorResidualKernel::tensor_residual`]
/// runs directly on lane-packed views. The 1D assembler reads only
/// `flux0`/`flux1x`; `flux1y` is still written (zeros) per the lane contract.
///
/// # Arguments
/// * `kernel` - tensor residual kernel.
/// * `ctxs` - one tensor context per lane.
/// * `ninputs` - state field count.
/// * `noutputs` - equation count.
/// * `npts` - quadrature points per cell.
/// * `lane_values`/`lane_grads` - lane-packed state.
/// * `flux0`/`flux1x`/`flux1y` - lane-packed flux outputs. Overwritten.
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_tensor_residual_fluxes<K, const GDIM: usize>(
    kernel: &K,
    ctxs: &[TensorCtx; SIMD_CELL_WIDTH],
    ninputs: usize,
    noutputs: usize,
    npts: usize,
    lane_values: &[f64],
    lane_grads: &[f64],
    flux0: &mut [f64],
    flux1x: &mut [f64],
    flux1y: &mut [f64],
) where
    K: TensorResidualKernel<GDIM> + Sync,
{
    let w = SIMD_CELL_WIDTH;
    debug_assert_eq!(LANES, w);
    let lane_state = LaneState {
        nfields: ninputs,
        npts,
        gdim: GDIM,
        values: lane_values,
        grads: lane_grads,
        field_indices: &[],
    };
    for eq in 0..noutputs {
        for q in 0..npts {
            let o = (eq * npts + q) * w;
            let f0: &mut Lanes = (&mut flux0[o..o + w]).try_into().expect("flux lane width");
            let f1x: &mut Lanes = (&mut flux1x[o..o + w]).try_into().expect("flux lane width");
            let f1y: &mut Lanes = (&mut flux1y[o..o + w]).try_into().expect("flux lane width");
            kernel.tensor_residual(ctxs, &lane_state, eq, q, f0, f1x, f1y);
        }
    }
}

/// Lane-packed pointwise flux evaluation for one tensor batch (Jacobian action).
///
/// Same loop nest as [`eval_tensor_residual_fluxes`] with
/// [`tensor_jacobian_action`](TensorResidualKernel::tensor_jacobian_action).
///
/// # Arguments
/// * `kernel` - tensor residual kernel.
/// * `ctxs` - one tensor context per lane.
/// * `ninputs` - state field count.
/// * `noutputs` - equation count.
/// * `npts` - quadrature points per cell.
/// * `state_values`/`state_grads` - lane-packed linearization point.
/// * `dir_values`/`dir_grads` - lane-packed Gateaux direction.
/// * `flux0`/`flux1x`/`flux1y` - lane-packed flux outputs. Overwritten.
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_tensor_jacobian_fluxes<K, const GDIM: usize>(
    kernel: &K,
    ctxs: &[TensorCtx; SIMD_CELL_WIDTH],
    ninputs: usize,
    noutputs: usize,
    npts: usize,
    state_values: &[f64],
    state_grads: &[f64],
    dir_values: &[f64],
    dir_grads: &[f64],
    flux0: &mut [f64],
    flux1x: &mut [f64],
    flux1y: &mut [f64],
) where
    K: TensorResidualKernel<GDIM> + Sync,
{
    let w = SIMD_CELL_WIDTH;
    debug_assert_eq!(LANES, w);
    let lane_state = LaneState {
        nfields: ninputs,
        npts,
        gdim: GDIM,
        values: state_values,
        grads: state_grads,
        field_indices: &[],
    };
    let lane_dir = LaneState {
        nfields: ninputs,
        npts,
        gdim: GDIM,
        values: dir_values,
        grads: dir_grads,
        field_indices: &[],
    };
    for eq in 0..noutputs {
        for q in 0..npts {
            let o = (eq * npts + q) * w;
            let f0: &mut Lanes = (&mut flux0[o..o + w]).try_into().expect("flux lane width");
            let f1x: &mut Lanes = (&mut flux1x[o..o + w]).try_into().expect("flux lane width");
            let f1y: &mut Lanes = (&mut flux1y[o..o + w]).try_into().expect("flux lane width");
            kernel.tensor_jacobian_action(ctxs, &lane_state, &lane_dir, eq, q, f0, f1x, f1y);
        }
    }
}

/// Fetch the cached row-sorted scatter table for one output selection,
/// building and caching it on first use.
///
/// # Arguments
/// * `adapter` - dimension adapter.
/// * `selection_outputs` - global field ids receiving the scatter.
///
/// # Returns
/// Shared `SortedRowScatter` (slots ordered by row, then `(color, pos, local)`).
pub(crate) fn sorted_scatter_for<A, const GDIM: usize>(
    adapter: &A,
    selection_outputs: &[usize],
) -> Arc<SortedRowScatter>
where
    A: TensorVolumeAdapter<GDIM>,
{
    {
        let cache = adapter.scatter_cache().read().unwrap();
        for (key, entries) in cache.iter() {
            if key.as_slice() == selection_outputs {
                return entries.clone();
            }
        }
    }
    let plan = adapter.batch_plan();
    let built = Arc::new(SortedRowScatter::build(
        adapter.restriction(),
        &plan.color_of,
        plan.batches.len(),
        selection_outputs,
        plan.ndofs,
    ));
    let mut cache = adapter.scatter_cache().write().unwrap();
    for (key, entries) in cache.iter() {
        if key.as_slice() == selection_outputs {
            return entries.clone();
        }
    }
    cache.push((selection_outputs.to_vec(), built.clone()));
    built
}

/// Take a reusable row-sorted E-vector buffer of exactly `len` doubles.
///
/// # Arguments
/// * `adapter` - dimension adapter.
/// * `len` - required buffer length (`nslots * ncols`).
///
/// # Returns
/// Pooled buffer (resized if reused at a different length), or fresh.
pub(crate) fn acquire_evec<A, const GDIM: usize>(adapter: &A, len: usize) -> Vec<f64>
where
    A: TensorVolumeAdapter<GDIM>,
{
    let mut pool = adapter.evec_pool().lock().unwrap();
    match pool.pop() {
        Some(mut buf) => {
            if buf.len() != len {
                buf.resize(len, 0.0);
            }
            buf
        }
        None => vec![0.0; len],
    }
}

/// Return an E-vector buffer to the pool (capped to bound memory).
///
/// # Arguments
/// * `adapter` - dimension adapter.
/// * `buf` - buffer to recycle.
pub(crate) fn release_evec<A, const GDIM: usize>(adapter: &A, buf: Vec<f64>)
where
    A: TensorVolumeAdapter<GDIM>,
{
    let mut pool = adapter.evec_pool().lock().unwrap();
    if pool.len() < EVEC_POOL_CAP {
        pool.push(buf);
    }
}

/// Outer tensor pass: two row-sorted E-vector phases over natural-order batches.
///
/// Phase 1 ([`phase1_fill_sorted`]) runs one parallel region over ALL batches
/// (no color grouping): each batch gathers, interpolates, evaluates pointwise
/// fluxes, integrates into thread-local scratch, and scatters once per real
/// lane into its disjoint row-sorted slots (see [`SortedRowScatter`]). State
/// interpolation is shared across direction columns within a batch. Phase 2
/// ([`reduce_sorted_evec_into_out`]) reduces contiguous row segments
/// `out[r] = sum(seg)` in slot order (bit-identical to the old
/// color-sequential scatter, thread-count independent). `out` is fully
/// overwritten: contiguous columns are reduced in place, non-contiguous
/// `out` goes through a private column-major buffer copied back afterwards.
/// Exactly one of `fresh_state` / `cache` must be `Some`, selecting the
/// per-batch [`BatchStateInput`]; `direction` selects the residual (`None`)
/// vs a Jacobian action (`Some`).
///
/// # Arguments
/// * `adapter` - dimension adapter.
/// * `time` - evaluation time.
/// * `kernel` - tensor residual kernel.
/// * `ninputs` - state field count (scratch sizing).
/// * `noutputs` - equation count (scratch sizing).
/// * `sources` - restriction source per input position.
/// * `input_offsets` - global row offset per input position.
/// * `selection_outputs` - global field ids receiving the reduction.
/// * `fresh_state` - linearization state (`N×1`), or `None` when cached.
/// * `cache` - linearization cache, or `None` when fresh.
/// * `direction` - global directions, or `None` for the residual.
/// * `out` - global output (`N×ncols`, `ncols == 1` for the residual). Fully overwritten.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_tensor_pass<A, K, const GDIM: usize>(
    adapter: &A,
    time: f64,
    kernel: &K,
    ninputs: usize,
    noutputs: usize,
    sources: &[usize],
    input_offsets: &[usize],
    selection_outputs: &[usize],
    fresh_state: Option<faer::MatRef<'_, f64>>,
    cache: Option<&TensorStateCache>,
    direction: Option<faer::MatRef<'_, f64>>,
    out: MatMut<'_, f64>,
) where
    A: TensorVolumeAdapter<GDIM> + Sync,
    K: TensorResidualKernel<GDIM> + Sync,
{
    assert!(
        fresh_state.is_some() ^ cache.is_some(),
        "tensor pass needs exactly one state source"
    );
    let ncols = direction.map(|d| d.ncols()).unwrap_or(1);
    assert_eq!(out.ncols(), ncols, "output column count mismatch");
    if ncols == 0 {
        return;
    }
    let scatter = sorted_scatter_for(adapter, selection_outputs);
    // Phase 1: one parallel region over all batches into the row-sorted
    // E-vector (`nslots * ncols`, column `c` at `c * nslots`).
    let mut evec_sorted = acquire_evec(adapter, scatter.nslots * ncols);
    let ctx = TensorPhase1Ctx {
        time,
        ninputs,
        noutputs,
        sources,
        input_offsets,
        fresh_state,
        cache,
        direction,
        scatter: &scatter,
        evec_sorted: &mut evec_sorted,
    };
    phase1_fill_sorted(adapter, kernel, ctx);
    // Phase 2: shared streaming segmented-sum reduction (fully overwrites
    // `out`; see `reduce_sorted_evec_into_out` for the ordering argument).
    reduce_sorted_evec_into_out(
        &scatter,
        &evec_sorted,
        ncols,
        None,
        RowEpilogue::None,
        adapter.evec_pool(),
        out,
    );
    release_evec(adapter, evec_sorted);
}

/// Phase 1: scatter every batch's integrated local actions into the
/// row-sorted E-vector through `scatter.dest`.
///
/// After the dimension's packed integrate into thread-local scratch, each real
/// value is stored once at `evec_sorted[c * nslots + dest[i]]`; padded /
/// eliminated lanes (`dest == u32::MAX`) are skipped. Slots are disjoint
/// across batches (see [`SortedRowScatter`]), so the parallel writes use a
/// [`DisjointSortedEvec`] raw-pointer handle with no atomics.
///
/// Bit-identity: the integrated scratch block is produced by the exact
/// same gather → interpolate → pointwise → integrate arithmetic as before,
/// only permuted into row-sorted slots.
///
/// # Arguments
/// * `adapter` - dimension adapter.
/// * `kernel` - tensor residual kernel.
/// * `ctx` - bundled phase-1 arguments (see [`TensorPhase1Ctx`]).
pub(crate) fn phase1_fill_sorted<A, K, const GDIM: usize>(
    adapter: &A,
    kernel: &K,
    ctx: TensorPhase1Ctx<'_, '_>,
) where
    A: TensorVolumeAdapter<GDIM> + Sync,
    K: TensorResidualKernel<GDIM> + Sync,
{
    let TensorPhase1Ctx {
        time,
        ninputs,
        noutputs,
        sources,
        input_offsets,
        fresh_state,
        cache,
        direction,
        scatter,
        evec_sorted,
    } = ctx;
    let cd = adapter.cell_data();
    let ncols = direction.map(|d| d.ncols()).unwrap_or(1);
    let block = noutputs * cd.ndofs * SIMD_CELL_WIDTH;
    let plan = adapter.batch_plan();
    let nbatches = plan.batches.len();
    debug_assert_eq!(evec_sorted.len(), scatter.nslots * ncols);
    debug_assert_eq!(scatter.dest.len(), nbatches * block);
    let batches = &plan.batches;
    debug_assert_eq!(batches.len(), nbatches);
    // SAFETY: `dest` maps each real `(batch, inner)` pair to a unique slot
    // (inversion of the row-sorted order; padded/eliminated stay sentinel),
    // so per column each slot has exactly one writer and batch slot sets
    // are disjoint. No other access to `evec_sorted` occurs during the
    // parallel region; each slot is written exactly once per column.
    let writer = unsafe { DisjointSortedEvec::new(evec_sorted, scatter.nslots) };
    let dest_all = scatter.dest.as_slice();
    batches
        .par_iter()
        .enumerate()
        .for_each(|(batch_idx, batch)| {
            let dest_batch = &dest_all[batch_idx * block..(batch_idx + 1) * block];
            let state_input = match (fresh_state, cache) {
                (Some(state), None) => BatchStateInput::Fresh(state),
                (None, Some(cache)) => {
                    // The unprepared path only forwards a cache whose
                    // interpolated states were built (see
                    // `apply_tensor_jacobian_with_layout_cached`).
                    let states = cache.batch_states.as_ref().expect(
                        "tensor pass needs cached batch states; use fresh state instead",
                    );
                    BatchStateInput::Cached(&states[batch_idx])
                }
                _ => unreachable!("tensor pass needs exactly one state source"),
            };
            with_tensor_worker_scratch(ninputs, noutputs, cd.ndofs, cd.npts, GDIM, |scratch| {
                process_tensor_batch_sorted(
                    adapter,
                    time,
                    kernel,
                    batch,
                    state_input,
                    sources,
                    input_offsets,
                    direction,
                    ninputs,
                    noutputs,
                    dest_batch,
                    writer,
                    scratch,
                );
            });
        });
}

/// One batch of the tensor volume skeleton into row-sorted slots.
///
/// Same gather → interpolate → pointwise → integrate arithmetic, but the
/// integrated block is staged in thread-local `packed_out` scratch
/// (`dir_lane` for Jacobian columns, `state_lane` for the residual) and then
/// scattered once per real lane to `sorted[c * nslots + dest[i]]`.
/// `direction` selects the residual (`None`, single column 0) vs a Jacobian
/// action (`Some`).
///
/// # Arguments
/// * `adapter` - dimension adapter.
/// * `time` - evaluation time for the adapter's `tensor_ctx`.
/// * `kernel` - tensor residual kernel.
/// * `batch` - precomputed lane-packed geometry and restriction.
/// * `state_input` - fresh state to gather+interpolate, or cached lane state.
/// * `sources` - restriction source per input position.
/// * `input_offsets` - global row offset per input position.
/// * `direction` - global directions (`N×ncols`), or `None` for the residual.
/// * `ninputs` - state field count.
/// * `noutputs` - equation count.
/// * `dest_batch` - this batch's destination slots (`block`). Borrowed.
/// * `sorted` - row-sorted E-vector writer (disjoint slots).
/// * `scratch` - calling thread's persistent lane scratch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_tensor_batch_sorted<A, K, const GDIM: usize>(
    adapter: &A,
    time: f64,
    kernel: &K,
    batch: &TensorBatch,
    state_input: BatchStateInput<'_>,
    sources: &[usize],
    input_offsets: &[usize],
    direction: Option<faer::MatRef<'_, f64>>,
    ninputs: usize,
    noutputs: usize,
    dest_batch: &[u32],
    sorted: DisjointSortedEvec,
    scratch: &mut TensorWorkerScratch,
) where
    A: TensorVolumeAdapter<GDIM> + Sync,
    K: TensorResidualKernel<GDIM> + Sync,
{
    let cd = adapter.cell_data();
    let (npts, ndofs) = (cd.npts, cd.ndofs);
    debug_assert!((1..=SIMD_CELL_WIDTH).contains(&batch.nreal));
    let TensorWorkerScratch {
        state_lane,
        dir_lane,
    } = scratch;
    if let BatchStateInput::Fresh(state) = state_input {
        gather_state_batch_from_plan(
            batch,
            sources,
            input_offsets,
            state,
            &mut state_lane.packed_coeffs,
            ndofs,
        );
        A::interpolate_packed(
            cd,
            ninputs,
            &state_lane.packed_coeffs,
            &mut state_lane.lane_values,
            &mut state_lane.lane_grads,
            &batch.jinv,
        );
    }
    let (state_values, state_grads): (&[f64], &[f64]) = match &state_input {
        BatchStateInput::Fresh(_) => (
            state_lane.lane_values.as_slice(),
            state_lane.lane_grads.as_slice(),
        ),
        BatchStateInput::Cached(cached) => (cached.values.as_slice(), cached.grads.as_slice()),
    };
    let ctxs: [TensorCtx; SIMD_CELL_WIDTH] =
        std::array::from_fn(|l| adapter.tensor_ctx(time, batch.cells[l]));
    match direction {
        None => {
            eval_tensor_residual_fluxes::<K, GDIM>(
                kernel,
                &ctxs,
                ninputs,
                noutputs,
                npts,
                state_values,
                state_grads,
                &mut state_lane.flux0,
                &mut state_lane.flux1x,
                &mut state_lane.flux1y,
            );
            let tmp = &mut state_lane.packed_out;
            debug_assert_eq!(tmp.len(), dest_batch.len());
            tmp.fill(0.0);
            A::integrate_packed(
                cd,
                noutputs,
                &state_lane.flux0,
                &state_lane.flux1x,
                &state_lane.flux1y,
                tmp,
                &batch.wdet,
                &batch.jinv,
            );
            for (i, &slot) in dest_batch.iter().enumerate() {
                if slot != u32::MAX {
                    // SAFETY: slot disjoint across batches per column.
                    unsafe { sorted.write(slot, 0, tmp[i]) };
                }
            }
        }
        Some(direction) => {
            let ncols = direction.ncols();
            for column in 0..ncols {
                gather_direction_batch_from_plan(
                    batch,
                    sources,
                    input_offsets,
                    direction,
                    column,
                    &mut dir_lane.packed_coeffs,
                    ndofs,
                );
                A::interpolate_packed(
                    cd,
                    ninputs,
                    &dir_lane.packed_coeffs,
                    &mut dir_lane.lane_values,
                    &mut dir_lane.lane_grads,
                    &batch.jinv,
                );
                eval_tensor_jacobian_fluxes::<K, GDIM>(
                    kernel,
                    &ctxs,
                    ninputs,
                    noutputs,
                    npts,
                    state_values,
                    state_grads,
                    &dir_lane.lane_values,
                    &dir_lane.lane_grads,
                    &mut dir_lane.flux0,
                    &mut dir_lane.flux1x,
                    &mut dir_lane.flux1y,
                );
                let tmp = &mut dir_lane.packed_out;
                debug_assert_eq!(tmp.len(), dest_batch.len());
                tmp.fill(0.0);
                A::integrate_packed(
                    cd,
                    noutputs,
                    &dir_lane.flux0,
                    &dir_lane.flux1x,
                    &dir_lane.flux1y,
                    tmp,
                    &batch.wdet,
                    &batch.jinv,
                );
                for (i, &slot) in dest_batch.iter().enumerate() {
                    if slot != u32::MAX {
                        // SAFETY: slot disjoint across batches per column.
                        unsafe { sorted.write(slot, column, tmp[i]) };
                    }
                }
            }
        }
    }
}

/// Sum-factorized tensor residual with a caller-provided field layout.
///
/// Mathematics: `R(U) = Σ_e E_eᵀ r_e` with sum-factorized GLL
/// interpolation/integration and SIMD-over-element batching. `layout` avoids
/// recomputation when the caller already holds it.
///
/// # Arguments
/// * `adapter` - dimension adapter.
/// * `time` - evaluation time.
/// * `kernel` - tensor residual kernel.
/// * `state` - global state (`N×1`).
/// * `layout` - field layout.
///
/// # Returns
/// Global residual vector.
pub(crate) fn assemble_tensor_residual_with_layout<A, K, const GDIM: usize>(
    adapter: &A,
    time: f64,
    kernel: &K,
    state: MatRef<f64>,
    layout: &FieldDofLayout,
) -> Vec<f64>
where
    A: TensorVolumeAdapter<GDIM> + Sync,
    K: TensorResidualKernel<GDIM> + Sync,
{
    assert_eq!(state.nrows(), layout.total_size, "state size mismatch");
    assert_eq!(
        state.ncols(),
        1,
        "tensor residual requires one state column"
    );
    let selection = adapter.fields().resolve_selection(
        kernel.input_nfields(),
        kernel.input_field_names(),
        kernel.output_nfields(),
        kernel.output_field_names(),
        "tensor residual kernel",
    );
    let ninputs = selection.inputs.len();
    let noutputs = selection.outputs.len();
    let input_offsets: Vec<_> = selection
        .inputs
        .iter()
        .map(|&field| layout.offsets[field])
        .collect();
    let sources: Vec<usize> = selection
        .inputs
        .iter()
        .map(|&g| adapter.restriction().field_map_source(g))
        .collect();
    let mut residual = vec![0.0; layout.total_size];
    {
        let mut out =
            MatMut::from_column_major_slice_mut(residual.as_mut_slice(), layout.total_size, 1);
        execute_tensor_pass(
            adapter,
            time,
            kernel,
            ninputs,
            noutputs,
            &sources,
            &input_offsets,
            &selection.outputs,
            Some(state),
            None,
            None,
            out.rb_mut(),
        );
    }
    residual
}

/// Batched unit-impulse columns through the lane-packed pipeline.
///
/// Mathematics: column `(unknown, trial)` of each cell block is the
/// lane-packed tensor action on a unit impulse, scattered as triplets
/// in cell-major order via `push_rectangular_local_matrix_triplets`.
/// Same sparsity as the weak Jacobian; cheaper formation. For each
/// batch the state is interpolated once (lane-packed); then every
/// `(unknown, trial)` sets the direction coefficients to the unit
/// impulse in all real lanes and runs lane interpolate, lane kernel
/// action, and lane integrate. Only real lanes are pushed, in the same
/// per-cell `(equation, test, unknown, trial)` order as the assembled
/// pattern cache expects.
///
/// # Arguments
/// * `adapter` - dimension adapter.
/// * `time` - evaluation time.
/// * `kernel` - tensor residual kernel.
/// * `state` - global state (`N×1`).
/// * `layout` - field layout.
///
/// # Returns
/// Assembled sparse Jacobian.
pub(crate) fn assemble_tensor_jacobian_with_layout<A, K, const GDIM: usize>(
    adapter: &A,
    time: f64,
    kernel: &K,
    state: MatRef<f64>,
    layout: &FieldDofLayout,
) -> SparseColMat<usize, f64>
where
    A: TensorVolumeAdapter<GDIM> + Sync,
    K: TensorResidualKernel<GDIM> + Sync,
{
    let selection = adapter.fields().resolve_selection(
        kernel.input_nfields(),
        kernel.input_field_names(),
        kernel.output_nfields(),
        kernel.output_field_names(),
        "tensor residual kernel",
    );
    let ninputs = selection.inputs.len();
    let noutputs = selection.outputs.len();
    let input_offsets: Vec<_> = selection
        .inputs
        .iter()
        .map(|&field| layout.offsets[field])
        .collect();
    let output_offsets: Vec<_> = selection
        .outputs
        .iter()
        .map(|&field| layout.offsets[field])
        .collect();
    let sources: Vec<usize> = selection
        .inputs
        .iter()
        .map(|&g| adapter.restriction().field_map_source(g))
        .collect();
    let cd = adapter.cell_data();
    let (npts, ndofs) = (cd.npts, cd.ndofs);
    let w = SIMD_CELL_WIDTH;
    let row_size = noutputs * ndofs;
    let col_size = ninputs * ndofs;
    let per_lane = row_size * col_size;
    let batches: Vec<Vec<Triplet<usize, usize, f64>>> = adapter
        .batch_plan()
        .batches
        .par_iter()
        .map_init(
            || {
                (
                    vec![0.0; ninputs * ndofs * w],
                    vec![0.0; ninputs * npts * w],
                    vec![0.0; ninputs * GDIM * npts * w],
                    vec![0.0; ninputs * ndofs * w],
                    vec![0.0; ninputs * npts * w],
                    vec![0.0; ninputs * GDIM * npts * w],
                    vec![0.0; noutputs * npts * w],
                    vec![0.0; noutputs * npts * w],
                    vec![0.0; noutputs * npts * w],
                    vec![0.0; noutputs * ndofs * w],
                    vec![0.0; w * per_lane],
                )
            },
            |(
                state_coeffs,
                state_values,
                state_grads,
                dir_coeffs,
                dir_values,
                dir_grads,
                flux0,
                flux1x,
                flux1y,
                packed_out,
                local_mats,
            ),
             batch| {
                let nreal = batch.nreal;
                gather_state_batch_from_plan(
                    batch,
                    &sources,
                    &input_offsets,
                    state,
                    state_coeffs,
                    ndofs,
                );
                A::interpolate_packed(
                    cd,
                    ninputs,
                    state_coeffs,
                    state_values,
                    state_grads,
                    &batch.jinv,
                );
                let ctxs: [TensorCtx; SIMD_CELL_WIDTH] =
                    std::array::from_fn(|l| adapter.tensor_ctx(time, batch.cells[l]));
                local_mats.fill(0.0);
                for unknown in 0..ninputs {
                    for trial in 0..ndofs {
                        dir_coeffs.fill(0.0);
                        for l in 0..nreal {
                            dir_coeffs[(unknown * ndofs + trial) * w + l] = 1.0;
                        }
                        A::interpolate_packed(
                            cd,
                            ninputs,
                            dir_coeffs,
                            dir_values,
                            dir_grads,
                            &batch.jinv,
                        );
                        eval_tensor_jacobian_fluxes::<K, GDIM>(
                            kernel,
                            &ctxs,
                            ninputs,
                            noutputs,
                            npts,
                            state_values,
                            state_grads,
                            dir_values,
                            dir_grads,
                            flux0,
                            flux1x,
                            flux1y,
                        );
                        packed_out.fill(0.0);
                        A::integrate_packed(
                            cd,
                            noutputs,
                            flux0,
                            flux1x,
                            flux1y,
                            packed_out,
                            &batch.wdet,
                            &batch.jinv,
                        );
                        for l in 0..nreal {
                            let base = l * per_lane;
                            for equation in 0..noutputs {
                                for test in 0..ndofs {
                                    local_mats[base
                                        + (equation * ndofs + test) * col_size
                                        + unknown * ndofs
                                        + trial] =
                                        packed_out[(equation * ndofs + test) * w + l];
                                }
                            }
                        }
                    }
                }
                let mut triplets = Vec::with_capacity(nreal * per_lane);
                for l in 0..nreal {
                    let cell_index = batch.cells[l];
                    let (field_maps, _) =
                        adapter.restriction().maps_for(cell_index, &selection.inputs);
                    let (row_maps, _) =
                        adapter.restriction().maps_for(cell_index, &selection.outputs);
                    push_rectangular_local_matrix_triplets(
                        &mut triplets,
                        &local_mats[l * per_lane..(l + 1) * per_lane],
                        &row_maps,
                        &field_maps,
                        noutputs,
                        ninputs,
                        &output_offsets,
                        &input_offsets,
                        // ponytail: keep explicit zeros; stable CSC pattern across states.
                        |_| true,
                    );
                }
                triplets
            },
        )
        .collect();
    let triplets: Vec<_> = batches.into_iter().flatten().collect();
    let system_size = layout.total_size;
    adapter.pattern_cache().assemble(
        system_size,
        &selection.inputs,
        &selection.outputs,
        triplets,
    )
}

/// Build an owned linearization-state cache with an explicit byte budget.
///
/// Always copies the state vector; the per-batch lane-packed values+grads
/// are only built when their footprint
/// ([`tensor_interpolated_state_bytes_gdim`]) fits `limit_bytes`. Above the
/// budget the prepared path re-interpolates from the owned copy instead,
/// which stays bit-identical while avoiding last-level-cache thrash.
///
/// # Arguments
/// * `adapter` - dimension adapter.
/// * `kernel` - tensor kernel selecting input fields.
/// * `state` - linearization point (`N×1`).
/// * `limit_bytes` - interpolated-state budget in bytes; `0` disables the
///   interpolated states (only the owned state copy is kept).
/// * `layout_total` - total system size for the state-size check.
///
/// # Returns
/// Owned `TensorStateCache`, with per-batch lane-packed states only when
/// their footprint fits `limit_bytes`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_tensor_state_cache_with_limit<A, K, const GDIM: usize>(
    adapter: &A,
    kernel: &K,
    state: MatRef<f64>,
    limit_bytes: u64,
    layout_total: usize,
    input_offsets: &[usize],
    sources: &[usize],
    ninputs: usize,
) -> TensorStateCache
where
    A: TensorVolumeAdapter<GDIM> + Sync,
    K: TensorResidualKernel<GDIM> + Sync,
{
    let cd = adapter.cell_data();
    let ndofs = cd.ndofs;
    let npts = cd.npts;
    assert_eq!(state.nrows(), layout_total, "state size mismatch");
    assert_eq!(state.ncols(), 1, "cache requires one state column");
    let state_copy: Vec<f64> =
        if let Some(slice) = state.col(0).try_as_col_major().map(|c| c.as_slice()) {
            slice.to_vec()
        } else {
            (0..state.nrows()).map(|r| state[(r, 0)]).collect()
        };
    let nrows = state.nrows();
    let ncols = state.ncols();
    let selection = adapter.fields().resolve_selection(
        kernel.input_nfields(),
        kernel.input_field_names(),
        kernel.output_nfields(),
        kernel.output_field_names(),
        "tensor residual kernel",
    );
    let nbatches = adapter.batch_plan().batches.len();
    let needed = if GDIM == 2 {
        tensor_interpolated_state_bytes(nbatches, ninputs, npts)
    } else {
        tensor_interpolated_state_bytes_gdim(nbatches, ninputs, npts, GDIM)
    };
    let batch_states: Option<Vec<CachedBatchState>> = if needed <= limit_bytes {
        Some(
            adapter
                .batch_plan()
                .batches
                .par_iter()
                .map(|batch| {
                    let w = SIMD_CELL_WIDTH;
                    let mut packed = vec![0.0; ninputs * ndofs * w];
                    gather_state_batch_from_plan(
                        batch,
                        sources,
                        input_offsets,
                        state,
                        &mut packed,
                        ndofs,
                    );
                    let mut values = vec![0.0; ninputs * npts * w];
                    let mut grads = vec![0.0; ninputs * GDIM * npts * w];
                    A::interpolate_packed(
                        cd,
                        ninputs,
                        &packed,
                        &mut values,
                        &mut grads,
                        &batch.jinv,
                    );
                    CachedBatchState { values, grads }
                })
                .collect(),
        )
    } else {
        None
    };
    TensorStateCache {
        state_copy,
        nrows,
        ncols,
        inputs: selection.inputs.clone(),
        batch_states,
    }
}

/// Matrix-free Jacobian action with a caller-provided layout and optional cache.
///
/// # Arguments
/// * `adapter` - dimension adapter.
/// * `time` - evaluation time.
/// * `kernel` - tensor residual kernel.
/// * `state` - linearization state (`N×1`).
/// * `direction` - global directions.
/// * `layout` - field layout.
/// * `out` - global output. Fully overwritten.
/// * `cache` - linearization cache, or `None`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_tensor_jacobian_with_layout_cached<A, K, const GDIM: usize>(
    adapter: &A,
    time: f64,
    kernel: &K,
    state: MatRef<f64>,
    direction: MatRef<f64>,
    layout: &FieldDofLayout,
    out: MatMut<'_, f64>,
    cache: Option<&TensorStateCache>,
) where
    A: TensorVolumeAdapter<GDIM> + Sync,
    K: TensorResidualKernel<GDIM> + Sync,
{
    let selection = adapter.fields().resolve_selection(
        kernel.input_nfields(),
        kernel.input_field_names(),
        kernel.output_nfields(),
        kernel.output_field_names(),
        "tensor residual kernel",
    );
    let ninputs = selection.inputs.len();
    let noutputs = selection.outputs.len();
    let input_offsets: Vec<_> = selection
        .inputs
        .iter()
        .map(|&field| layout.offsets[field])
        .collect();
    assert_eq!(state.nrows(), layout.total_size, "state size mismatch");
    assert_eq!(
        state.ncols(),
        1,
        "matrix-free Jacobian requires one state column"
    );
    assert_eq!(
        direction.nrows(),
        layout.total_size,
        "direction size mismatch"
    );
    assert_eq!(out.nrows(), layout.total_size, "output size mismatch");
    assert_eq!(
        out.ncols(),
        direction.ncols(),
        "output column count mismatch"
    );
    let ncols = direction.ncols();
    if ncols == 0 {
        return;
    }
    let sources: Vec<usize> = selection
        .inputs
        .iter()
        .map(|&g| adapter.restriction().field_map_source(g))
        .collect();
    // Without interpolated states the cache is simply not used here; the
    // prepared path re-interpolates from the owned copy instead.
    let use_cache = cache
        .map(|c| c.has_batch_states() && c.matches(state, &selection.inputs))
        .unwrap_or(false);
    // One shared skeleton (`execute_tensor_pass` →
    // `process_tensor_batch_sorted`) covers every variant: resolve the
    // state source once; cached/fresh stay bit-identical.
    let (fresh_state, cache_ref) = if use_cache {
        (None, cache)
    } else {
        (Some(state), None)
    };
    execute_tensor_pass(
        adapter,
        time,
        kernel,
        ninputs,
        noutputs,
        &sources,
        &input_offsets,
        &selection.outputs,
        fresh_state,
        cache_ref,
        Some(direction),
        out,
    );
}

/// One row of the phase-2 reduction: segment sum, optional boundary add,
/// optional `-M^{-1}` epilogue.
///
/// Computes exactly the old per-row sequence: `acc = seg_sum` (`let mut s =
/// 0.0; for v in seg { s += v }`, whose slot order equals the old `entry_idx`
/// order, hence bit-identical), then `acc = acc + boundary` when present
/// (elementwise, the same single addition faer's `out += boundary` performs
/// per row), then `-(acc * m_inv)` for [`RowEpilogue::NegScale`] (the SIMD
/// order `neg(mul(v, f))` of `simd::scale_negate_in_place`; the scalar tail
/// `(-v) * f` is bit-identical since negation is exact and IEEE
/// multiplication is sign-symmetric) or `acc` for [`RowEpilogue::None`].
/// Shared by the contiguous and non-contiguous branches of
/// [`reduce_sorted_evec_into_out`].
///
/// # Arguments
/// * `seg` - this row's contiguous E-vector segment for one column.
/// * `boundary` - boundary action value for `(row, column)`, or `None`.
/// * `m_inv` - inverse-mass diagonal value for the row, or `None`.
///
/// # Returns
/// The reduced row value to write to `out`.
#[inline(always)]
fn reduce_row(seg: &[f64], boundary: Option<f64>, m_inv: Option<f64>) -> f64 {
    let mut acc = 0.0;
    for v in seg {
        acc += *v;
    }
    if let Some(b) = boundary {
        acc = acc + b;
    }
    if let Some(m) = m_inv {
        -(acc * m)
    } else {
        acc
    }
}

/// Shared phase-2 row reduction for the row-sorted E-vector.
///
/// For each row `r` and column `c`, computes exactly the old sequence:
/// `v = seg_sum` (`let mut s = 0.0; for v in seg { s += v }` over
/// `evec_sorted[c * nslots + row_ptr[r] .. c * nslots + row_ptr[r+1]]`, whose
/// slot order equals the old `entry_idx` order, hence bit-identical), then
/// `v = v + boundary[(r, c)]` when `boundary` is present (elementwise, the
/// same single addition faer's `out += boundary` performs per row), then
/// `out[(r, c)] = -(v * m_inv[r])` for [`RowEpilogue::NegScale`] (the SIMD
/// order `neg(mul(v, f))` of `simd::scale_negate_in_place`; the scalar tail
/// `(-v) * f` is bit-identical since negation is exact and IEEE
/// multiplication is sign-symmetric) or `out = v` for [`RowEpilogue::None`].
/// Multi-column directions are handled per column. `out` is fully
/// overwritten; contiguous columns reduce in place (parallel over row chunks,
/// so each chunk reads a contiguous E-vector range), non-contiguous `out`
/// reduces into a pooled column-major buffer (`evec_pool`) copied back
/// afterwards.
///
/// # Arguments
/// * `scatter` - row-sorted destination table (row segments + slot count).
/// * `evec_sorted` - row-sorted E-vector (`ncols * nslots`). Borrowed read-only.
/// * `ncols` - direction column count.
/// * `boundary` - optional boundary action (`N×ncols`), added per row when present.
/// * `epilogue` - optional fused `-M^{-1}` row scaling.
/// * `evec_pool` - reusable E-vector scratch pool.
/// * `out` - global output (`N×ncols`). Fully overwritten.
pub(crate) fn reduce_sorted_evec_into_out(
    scatter: &SortedRowScatter,
    evec_sorted: &[f64],
    ncols: usize,
    boundary: Option<faer::MatRef<'_, f64>>,
    epilogue: RowEpilogue<'_>,
    evec_pool: &std::sync::Mutex<Vec<Vec<f64>>>,
    mut out: faer::MatMut<'_, f64>,
) {
    let total = out.nrows();
    assert_eq!(out.ncols(), ncols, "output column count mismatch");
    if ncols == 0 {
        return;
    }
    assert_eq!(
        scatter.row_ptr.len(),
        total + 1,
        "scatter/output row mismatch"
    );
    assert_eq!(
        evec_sorted.len(),
        scatter.nslots * ncols,
        "sorted E-vector size mismatch"
    );
    if let Some(b) = boundary {
        assert_eq!(b.nrows(), total, "boundary row mismatch");
        assert_eq!(b.ncols(), ncols, "boundary column mismatch");
    }
    let m_inv: Option<&[f64]> = match epilogue {
        RowEpilogue::None => None,
        RowEpilogue::NegScale(m) => {
            assert_eq!(m.len(), total, "mass/output size mismatch");
            Some(m)
        }
    };
    let row_ptr = scatter.row_ptr.as_slice();
    let nslots = scatter.nslots;
    let mut contiguous = true;
    for c in 0..ncols {
        if out
            .rb()
            .col(c)
            .try_as_col_major()
            .map(|v| v.as_slice().as_ptr())
            .is_none()
        {
            contiguous = false;
            break;
        }
    }
    // Per-column boundary slices when contiguous (the prepared path always
    // builds an owned contiguous boundary `Mat`, so this hits; `&[f64]` is
    // `Sync` and safe to share across the row-parallel reduction).
    let boundary_contig: Vec<Option<&[f64]>> = (0..ncols)
        .map(|c| boundary.and_then(|b| b.col(c).try_as_col_major().map(|v| v.as_slice())))
        .collect();
    if contiguous {
        for c in 0..ncols {
            let evec_col = &evec_sorted[c * nslots..(c + 1) * nslots];
            let bptr = boundary_contig[c];
            let has_boundary = boundary.is_some();
            let mcol = m_inv;
            let boundary_ref = boundary;
            let col = out.rb_mut().col_mut(c);
            let slice = col
                .try_as_col_major_mut()
                .expect("column checked contiguous")
                .as_slice_mut();
            debug_assert_eq!(slice.len(), total);
            slice
                .par_chunks_mut(512)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    for (i, slot) in chunk.iter_mut().enumerate() {
                        let r = chunk_idx * 512 + i;
                        // Chunks partition the slice exactly, so `r` is always
                        // in bounds; the old `if r >= total { break; }` was dead.
                        debug_assert!(r < total);
                        let s = row_ptr[r] as usize;
                        let e = row_ptr[r + 1] as usize;
                        let boundary_val = if has_boundary {
                            Some(if let Some(bslice) = bptr {
                                bslice[r]
                            } else {
                                boundary_ref.unwrap()[(r, c)]
                            })
                        } else {
                            None
                        };
                        *slot = reduce_row(&evec_col[s..e], boundary_val, mcol.map(|m| m[r]));
                    }
                });
        }
    } else {
        // Pooled column-major scratch (`total * ncols`, fully overwritten by
        // the reduction below, so pooled stale contents are harmless).
        let mut actions = {
            let mut pool = evec_pool.lock().unwrap();
            match pool.pop() {
                Some(mut buf) => {
                    if buf.len() != total * ncols {
                        buf.resize(total * ncols, 0.0);
                    }
                    buf
                }
                None => vec![0.0; total * ncols],
            }
        };
        {
            let boundary_ref = boundary;
            actions
                .par_chunks_mut(total)
                .enumerate()
                .for_each(|(c, col_actions)| {
                    let evec_col = &evec_sorted[c * nslots..(c + 1) * nslots];
                    let bptr = boundary_contig[c];
                    let has_boundary = boundary_ref.is_some();
                    let mcol = m_inv;
                    col_actions
                        .par_chunks_mut(512)
                        .enumerate()
                        .for_each(|(chunk_idx, chunk)| {
                            for (i, slot) in chunk.iter_mut().enumerate() {
                                let r = chunk_idx * 512 + i;
                                // Chunks partition the column exactly; see above.
                                debug_assert!(r < total);
                                let s = row_ptr[r] as usize;
                                let e = row_ptr[r + 1] as usize;
                                let boundary_val = if has_boundary {
                                    Some(if let Some(bslice) = bptr {
                                        bslice[r]
                                    } else {
                                        boundary_ref.unwrap()[(r, c)]
                                    })
                                } else {
                                    None
                                };
                                *slot =
                                    reduce_row(&evec_col[s..e], boundary_val, mcol.map(|m| m[r]));
                            }
                        });
                });
        }
        for c in 0..ncols {
            for r in 0..total {
                out[(r, c)] = actions[c * total + r];
            }
        }
        {
            let mut pool = evec_pool.lock().unwrap();
            if pool.len() < EVEC_POOL_CAP {
                pool.push(actions);
            }
        }
    }
}
