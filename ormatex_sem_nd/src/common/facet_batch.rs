//! Facet-batched tensor boundary evaluation over eight-lane facet batches.
//!
//! The tensor boundary driver (see [`super::boundary`]) groups the active
//! boundary facets into batches per boundary kernel and evaluates each batch
//! in chunks of at most [`LANES`] facets with lane-packed facet traces. All
//! pointwise kernel values are computed first (phase A, order-independent and
//! parallelizable) and stored per facet; the driver then scatters them in
//! exactly the historical accumulation order (phase B), so results stay
//! bitwise identical while the kernel calls run eight facets at a time.
//!
//! Lane-packed trace interpolation replays the scalar
//! `tensor_boundary_state_into` / `tensor_boundary_direction_into` updates per
//! lane (same field/facet-dof/quadrature order), so extracted lane values are
//! bitwise equal to the scalar interpolation. Padded lanes (chunks shorter
//! than [`LANES`]) copy the last real lane's traces and context and are never
//! scattered.

use std::collections::HashSet;

use faer::prelude::MatRef;
use rayon::prelude::*;

use super::boundary::{
    fill_maps_for_cell, tensor_kernel_key, QuadStateBoundaryCache, QuadStateBoundaryFacet,
    ResolvedTensorBoundary,
};
use super::contexts::{LaneState, TensorFacetCtx, LANES};
use crate::kernels::common::{StateTensorBoundaryIntegrator, StateTensorBoundaryTerms};

/// Inactive-facet sentinel for the pointwise index tables.
pub(crate) const INACTIVE_FACET: usize = usize::MAX;

/// Active boundary facets owned by one kernel, in ascending facet order.
pub(crate) struct FacetBatch<'t> {
    /// Boundary kernel shared by every facet in this batch.
    pub(crate) kernel: &'t dyn StateTensorBoundaryIntegrator<2>,
    /// Fat-pointer identity of `kernel` (see [`tensor_kernel_key`]).
    pub(crate) key: (usize, usize),
    /// Index into the pre-resolved tensor entries.
    pub(crate) entry: usize,
    /// Global facet positions (`cache.facets` indices) in ascending order.
    pub(crate) facets: Vec<usize>,
}

/// Group active facet positions by kernel identity.
///
/// Inputs: boundary cache, boundary terms, pre-resolved entries. Purpose: one
/// pass over `cache.facets` collecting, per distinct kernel (first-seen
/// order), the ascending facet positions it owns; facets without a kernel are
/// skipped. Computing the plan is `O(nfacets)` pointer lookups plus small
/// batch vectors, so it is rebuilt per call instead of cached. Output:
/// per-kernel batches for lane chunking.
pub(crate) fn plan_facet_batches<'t>(
    cache: &QuadStateBoundaryCache,
    terms: &'t StateTensorBoundaryTerms<2>,
    resolved: &[ResolvedTensorBoundary],
) -> Vec<FacetBatch<'t>> {
    let mut batches: Vec<FacetBatch<'t>> = Vec::new();
    for (position, facet) in cache.facets.iter().enumerate() {
        let Some(kernel) = terms.kernel_for(facet.facet.local_index) else {
            continue;
        };
        let key = tensor_kernel_key(kernel);
        if let Some(batch) = batches.iter_mut().find(|batch| batch.key == key) {
            batch.facets.push(position);
            continue;
        }
        let entry = resolved
            .iter()
            .position(|entry| entry.key == key)
            .expect("boundary kernel missing from pre-resolved entries");
        batches.push(FacetBatch {
            kernel,
            key,
            entry,
            facets: vec![position],
        });
    }
    batches
}

/// One lane-chunk work item: `len` (`1..=LANES`) consecutive facet positions
/// of batch `batch` starting at `start`.
#[derive(Clone, Copy)]
pub(crate) struct FacetChunk {
    pub(crate) batch: usize,
    pub(crate) start: usize,
    pub(crate) len: usize,
}

/// Split every batch into consecutive chunks of at most [`LANES`] facets.
///
/// Inputs: per-kernel batches. Purpose: fixed lane-chunk work items for phase
/// A (serial or parallel). Output: chunk descriptors in batch order.
pub(crate) fn chunk_facet_batches(batches: &[FacetBatch<'_>]) -> Vec<FacetChunk> {
    let mut chunks = Vec::new();
    for (batch, entry) in batches.iter().enumerate() {
        let mut start = 0;
        while start < entry.facets.len() {
            let len = (entry.facets.len() - start).min(LANES);
            chunks.push(FacetChunk { batch, start, len });
            start += len;
        }
    }
    chunks
}

/// Per-facet input/output DOF maps for one boundary evaluation.
///
/// Vectors are indexed by global facet position; inactive positions hold empty
/// maps. Built once per call and shared by phase A (all columns) and phase B.
pub(crate) struct FacetMapTable<'m> {
    pub(crate) input: Vec<Vec<&'m [Option<usize>]>>,
    pub(crate) prescribed: Vec<Vec<&'m [Option<f64>]>>,
    pub(crate) output: Vec<Vec<&'m [Option<usize>]>>,
}

/// Build the per-facet map table (see [`FacetMapTable`]).
///
/// Inputs: cache, batches with resolved entries, field DOF/prescribed maps.
/// Purpose: hoist the per-facet map lookups out of the column loops to one
/// lookup per facet per call. Output: map table indexed by facet position.
pub(crate) fn build_facet_map_table<'m>(
    cache: &QuadStateBoundaryCache,
    batches: &[FacetBatch<'_>],
    resolved: &[ResolvedTensorBoundary],
    field_reduced_dofs: &'m [Vec<Vec<Option<usize>>>],
    field_prescribed_values: &'m [Vec<Vec<Option<f64>>>],
) -> FacetMapTable<'m> {
    let nfacets = cache.facets.len();
    let mut table = FacetMapTable {
        input: vec![Vec::new(); nfacets],
        prescribed: vec![Vec::new(); nfacets],
        output: vec![Vec::new(); nfacets],
    };
    let mut input_buf = Vec::new();
    let mut prescribed_buf = Vec::new();
    let mut output_buf = Vec::new();
    for batch in batches {
        let entry = &resolved[batch.entry];
        for &position in &batch.facets {
            let cell = cache.facets[position].cell_index;
            fill_maps_for_cell(field_reduced_dofs, &entry.inputs, cell, &mut input_buf);
            fill_maps_for_cell(
                field_prescribed_values,
                &entry.inputs,
                cell,
                &mut prescribed_buf,
            );
            fill_maps_for_cell(field_reduced_dofs, &entry.outputs, cell, &mut output_buf);
            table.input[position] = std::mem::take(&mut input_buf);
            table.prescribed[position] = std::mem::take(&mut prescribed_buf);
            table.output[position] = std::mem::take(&mut output_buf);
        }
    }
    table
}

/// Reusable lane-chunk workspace (phase A).
///
/// Buffers are resized per chunk dimensions and reused across chunks (serial)
/// or per thread (parallel); nothing here scales with the global system size.
pub(crate) struct FacetChunkScratch<'c> {
    packed_state: Vec<f64>,
    packed_state_grads: Vec<f64>,
    packed_dir: Vec<f64>,
    packed_dir_grads: Vec<f64>,
    ctxs: Vec<TensorFacetCtx<'c>>,
}

impl<'c> FacetChunkScratch<'c> {
    /// Empty workspace; buffers grow on first use.
    pub(crate) fn new() -> Self {
        Self {
            packed_state: Vec::new(),
            packed_state_grads: Vec::new(),
            packed_dir: Vec::new(),
            packed_dir_grads: Vec::new(),
            ctxs: Vec::with_capacity(LANES),
        }
    }

    /// Resize buffers for `ninputs` fields and `npts` facet points.
    ///
    /// Inputs: field count and facet quadrature count. Purpose: grow (never
    /// shrink) the packed buffers; callers zero what they use.
    /// Output: none.
    fn ensure(&mut self, ninputs: usize, npts: usize) {
        self.packed_state.resize(LANES * ninputs * npts, 0.0);
        self.packed_state_grads
            .resize(LANES * ninputs * 2 * npts, 0.0);
        self.packed_dir.resize(LANES * ninputs * npts, 0.0);
        self.packed_dir_grads
            .resize(LANES * ninputs * 2 * npts, 0.0);
    }
}

/// Copy packed lane `source` into lane `target` over `nslots` slots.
///
/// Inputs: packed buffer, slot count, lane indices. Purpose: padded lanes
/// duplicate the last real lane exactly. Output: none (buffer updated).
fn copy_lane(packed: &mut [f64], nslots: usize, source: usize, target: usize) {
    for slot in 0..nslots {
        packed[slot * LANES + target] = packed[slot * LANES + source];
    }
}

/// Interpolate one lane's state trace into lane-packed buffers.
///
/// Same field/facet-dof/quadrature order as `tensor_boundary_state_into`, so
/// lane `lane` is bitwise equal to the scalar interpolation of that facet.
///
/// Inputs: facet, its input/prescribed maps, nonlinear state, input offsets,
/// dimensions, lane index, packed buffers (pre-zeroed by the caller), and the
/// gradient flag. Purpose: one lane of [`pack_state_chunk`]. Output: none
/// (packed lane slots accumulated).
#[allow(clippy::too_many_arguments)]
fn pack_state_lane(
    facet: &QuadStateBoundaryFacet,
    input_maps: &[&[Option<usize>]],
    prescribed: &[&[Option<f64>]],
    state: MatRef<'_, f64>,
    input_offsets: &[usize],
    ninputs: usize,
    npts: usize,
    lane: usize,
    packed_values: &mut [f64],
    packed_grads: &mut [f64],
    include_gradients: bool,
) {
    for field in 0..ninputs {
        for (facet_i, &cell_i) in facet.cell_indices.iter().enumerate() {
            let coefficient = input_maps[field][cell_i]
                .map_or(prescribed[field][cell_i].unwrap_or(0.0), |reduced| {
                    state[(input_offsets[field] + reduced, 0)]
                });
            for q in 0..npts {
                packed_values[(field * npts + q) * LANES + lane] +=
                    coefficient * facet.values[facet_i * npts + q];
            }
        }
        if include_gradients {
            for (cell_i, cell_grads) in facet.cell_grads.chunks_exact(2 * npts).enumerate() {
                let coefficient = input_maps[field][cell_i]
                    .map_or(prescribed[field][cell_i].unwrap_or(0.0), |reduced| {
                        state[(input_offsets[field] + reduced, 0)]
                    });
                for q in 0..npts {
                    for gd in 0..2 {
                        packed_grads[((field * 2 + gd) * npts + q) * LANES + lane] +=
                            coefficient * cell_grads[gd * npts + q];
                    }
                }
            }
        }
    }
}

/// Interpolate state traces for one chunk into lane-packed buffers.
///
/// Inputs: cache, chunk facet positions, resolved entry, map table, evaluation
/// time, nonlinear state, scratch (buffers resized and zeroed here, contexts
/// rebuilt). Purpose: lane-packed counterpart of
/// `tensor_boundary_state_into` with the same per-lane order, so each lane is
/// bitwise equal to the scalar interpolation; padded lanes copy the last real
/// lane and per-lane contexts copy the last real facet. Output: none (scratch
/// holds packed traces and one context per lane on return).
#[allow(clippy::too_many_arguments)]
fn pack_state_chunk<'c>(
    cache: &'c QuadStateBoundaryCache,
    positions: &[usize],
    entry: &ResolvedTensorBoundary,
    maps: &FacetMapTable<'_>,
    time: f64,
    state: MatRef<'_, f64>,
    scratch: &mut FacetChunkScratch<'c>,
) {
    let ninputs = entry.inputs.len();
    let npts = cache.wts.len();
    let include_gradients = entry.include_gradients;
    let nreal = positions.len();
    scratch.ensure(ninputs, npts);
    scratch.packed_state.fill(0.0);
    if include_gradients {
        scratch.packed_state_grads.fill(0.0);
    }
    for (lane, &position) in positions.iter().enumerate() {
        let facet = &cache.facets[position];
        pack_state_lane(
            facet,
            &maps.input[position],
            &maps.prescribed[position],
            state,
            &entry.input_offsets,
            ninputs,
            npts,
            lane,
            &mut scratch.packed_state,
            &mut scratch.packed_state_grads,
            include_gradients,
        );
    }
    for lane in nreal..LANES {
        copy_lane(&mut scratch.packed_state, ninputs * npts, nreal - 1, lane);
        if include_gradients {
            copy_lane(
                &mut scratch.packed_state_grads,
                ninputs * 2 * npts,
                nreal - 1,
                lane,
            );
        }
    }
    scratch.ctxs.clear();
    for lane in 0..LANES {
        let source = &cache.facets[positions[lane.min(nreal - 1)]];
        scratch.ctxs.push(TensorFacetCtx {
            time,
            facet: source.facet,
            npts,
            wts: &cache.wts,
            jfacet_det: &source.jfacet_det,
            points: &source.points,
            normal: &source.normal,
        });
    }
}

/// Interpolate direction-column traces for one chunk into lane-packed buffers.
///
/// Same per-lane order as `tensor_boundary_direction_into`, so each lane is
/// bitwise equal to the scalar interpolation of that facet and column; padded
/// lanes copy the last real lane.
///
/// Inputs: cache, chunk positions, resolved entry, map table, direction
/// matrix, its column, scratch (buffers resized and zeroed here). Purpose: one
/// column of [`eval_action_chunk`]. Output: none (scratch holds packed
/// direction traces on return).
#[allow(clippy::too_many_arguments)]
fn pack_direction_chunk(
    cache: &QuadStateBoundaryCache,
    positions: &[usize],
    entry: &ResolvedTensorBoundary,
    maps: &FacetMapTable<'_>,
    direction: MatRef<'_, f64>,
    column: usize,
    scratch: &mut FacetChunkScratch<'_>,
) {
    let ninputs = entry.inputs.len();
    let npts = cache.wts.len();
    let include_gradients = entry.include_gradients;
    let nreal = positions.len();
    scratch.ensure(ninputs, npts);
    scratch.packed_dir.fill(0.0);
    if include_gradients {
        scratch.packed_dir_grads.fill(0.0);
    }
    for (lane, &position) in positions.iter().enumerate() {
        let facet = &cache.facets[position];
        let input_maps = &maps.input[position];
        for field in 0..ninputs {
            for (facet_i, &cell_i) in facet.cell_indices.iter().enumerate() {
                let Some(reduced) = input_maps[field][cell_i] else {
                    continue;
                };
                let coefficient = direction[(entry.input_offsets[field] + reduced, column)];
                for q in 0..npts {
                    scratch.packed_dir[(field * npts + q) * LANES + lane] +=
                        coefficient * facet.values[facet_i * npts + q];
                }
            }
            if include_gradients {
                for (cell_i, cell_grads) in facet.cell_grads.chunks_exact(2 * npts).enumerate() {
                    let Some(reduced) = input_maps[field][cell_i] else {
                        continue;
                    };
                    let coefficient = direction[(entry.input_offsets[field] + reduced, column)];
                    for q in 0..npts {
                        for gd in 0..2 {
                            scratch.packed_dir_grads
                                [((field * 2 + gd) * npts + q) * LANES + lane] +=
                                coefficient * cell_grads[gd * npts + q];
                        }
                    }
                }
            }
        }
    }
    for lane in nreal..LANES {
        copy_lane(&mut scratch.packed_dir, ninputs * npts, nreal - 1, lane);
        if include_gradients {
            copy_lane(
                &mut scratch.packed_dir_grads,
                ninputs * 2 * npts,
                nreal - 1,
                lane,
            );
        }
    }
}

/// Interpolate one lane's unit-impulse trace for column `(unknown, reduced)`.
///
/// Same facet-dof/quadrature order as the assembled-Jacobian impulse build, so
/// the lane trace is bitwise equal; lanes without this column keep zeros
/// (their outputs are never stored).
///
/// Inputs: facet, its input maps, impulse column, presence flag, dimensions,
/// lane index, gradient flag, packed direction buffers (pre-zeroed by the
/// caller). Purpose: one lane of the assembled phase A. Output: none (packed
/// lane slots accumulated).
#[allow(clippy::too_many_arguments)]
fn pack_impulse_lane(
    facet: &QuadStateBoundaryFacet,
    input_maps: &[&[Option<usize>]],
    unknown: usize,
    reduced: usize,
    column_present: bool,
    npts: usize,
    lane: usize,
    include_gradients: bool,
    packed_dir: &mut [f64],
    packed_dir_grads: &mut [f64],
) {
    if !column_present {
        return;
    }
    for (facet_i, &basis_cell_i) in facet.cell_indices.iter().enumerate() {
        if input_maps[unknown][basis_cell_i] != Some(reduced) {
            continue;
        }
        for q in 0..npts {
            packed_dir[(unknown * npts + q) * LANES + lane] += facet.values[facet_i * npts + q];
        }
    }
    if include_gradients {
        for (cell_i, cell_grads) in facet.cell_grads.chunks_exact(2 * npts).enumerate() {
            if input_maps[unknown][cell_i] != Some(reduced) {
                continue;
            }
            for q in 0..npts {
                for gd in 0..2 {
                    packed_dir_grads[((unknown * 2 + gd) * npts + q) * LANES + lane] +=
                        cell_grads[gd * npts + q];
                }
            }
        }
    }
}

/// Pointwise residual/action values for one chunk:
/// `blocks[(sub * noutputs + equation) * npts + q]`.
pub(crate) struct LaneChunkOut {
    pub(crate) positions: Vec<usize>,
    pub(crate) noutputs: usize,
    pub(crate) blocks: Vec<f64>,
}

/// Evaluate residual fluxes for one chunk (phase A).
///
/// Inputs: cache, batch, resolved entry, map table, chunk positions, time,
/// nonlinear state, scratch. Purpose: pack state traces, call the lane-packed
/// residual, and return owned pointwise blocks. Output: chunk blocks for the
/// real lanes (padded lanes never stored).
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_residual_chunk<'c>(
    cache: &'c QuadStateBoundaryCache,
    batch: &FacetBatch<'_>,
    entry: &ResolvedTensorBoundary,
    maps: &FacetMapTable<'_>,
    positions: &[usize],
    time: f64,
    state: MatRef<'_, f64>,
    scratch: &mut FacetChunkScratch<'c>,
) -> LaneChunkOut {
    let ninputs = entry.inputs.len();
    let noutputs = entry.outputs.len();
    let npts = cache.wts.len();
    pack_state_chunk(cache, positions, entry, maps, time, state, scratch);
    let lane_state = LaneState {
        nfields: ninputs,
        npts,
        gdim: 2,
        values: &scratch.packed_state,
        grads: if entry.include_gradients {
            &scratch.packed_state_grads
        } else {
            &[]
        },
        field_indices: &[],
    };
    let mut blocks = vec![0.0; positions.len() * noutputs * npts];

    for equation in 0..noutputs {
        for q in 0..npts {
            let mut lanes = [0.0; LANES];
            batch
                .kernel
                .tensor_residual(&scratch.ctxs, &lane_state, equation, q, &mut lanes);
            for (sub, _) in positions.iter().enumerate() {
                blocks[(sub * noutputs + equation) * npts + q] = lanes[sub];
            }
        }
    }

    LaneChunkOut {
        positions: positions.to_vec(),
        noutputs,
        blocks,
    }
}

/// Evaluate Jacobian actions for one chunk and direction column (phase A).
///
/// Inputs: cache, batch, resolved entry, map table, chunk positions, time,
/// nonlinear state, direction matrix, its column, scratch. Purpose: pack state
/// and direction traces, call the lane-packed action, and return owned
/// pointwise blocks.
/// Output: chunk blocks for the real lanes (padded lanes never stored).
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_action_chunk<'c>(
    cache: &'c QuadStateBoundaryCache,
    batch: &FacetBatch<'_>,
    entry: &ResolvedTensorBoundary,
    maps: &FacetMapTable<'_>,
    positions: &[usize],
    time: f64,
    state: MatRef<'_, f64>,
    direction: MatRef<'_, f64>,
    column: usize,
    scratch: &mut FacetChunkScratch<'c>,
) -> LaneChunkOut {
    let ninputs = entry.inputs.len();
    let noutputs = entry.outputs.len();
    let npts = cache.wts.len();
    pack_state_chunk(cache, positions, entry, maps, time, state, scratch);
    pack_direction_chunk(cache, positions, entry, maps, direction, column, scratch);
    let lane_state = LaneState {
        nfields: ninputs,
        npts,
        gdim: 2,
        values: &scratch.packed_state,
        grads: if entry.include_gradients {
            &scratch.packed_state_grads
        } else {
            &[]
        },
        field_indices: &[],
    };
    let lane_direction = LaneState {
        nfields: ninputs,
        npts,
        gdim: 2,
        values: &scratch.packed_dir,
        grads: if entry.include_gradients {
            &scratch.packed_dir_grads
        } else {
            &[]
        },
        field_indices: &[],
    };
    let mut blocks = vec![0.0; positions.len() * noutputs * npts];

    for equation in 0..noutputs {
        for q in 0..npts {
            let mut lanes = [0.0; LANES];
            batch.kernel.tensor_jacobian_action(
                &scratch.ctxs,
                &lane_state,
                &lane_direction,
                equation,
                q,
                &mut lanes,
            );
            for (sub, _) in positions.iter().enumerate() {
                blocks[(sub * noutputs + equation) * npts + q] = lanes[sub];
            }
        }
    }

    LaneChunkOut {
        positions: positions.to_vec(),
        noutputs,
        blocks,
    }
}

/// Discover ordered unit-impulse columns per facet position.
///
/// Inputs: cache, batches, resolved entries, map table. Purpose: same
/// `(unknown, reduced)` discovery order as the assembled Jacobian
/// (`unknown` outer, trace-dof inner, first-seen dedup), shared by phase A
/// (impulse packing) and phase B (triplet emission). Output: column lists
/// indexed by facet position (empty for inactive facets).
pub(crate) fn boundary_column_lists(
    cache: &QuadStateBoundaryCache,
    batches: &[FacetBatch<'_>],
    resolved: &[ResolvedTensorBoundary],
    maps: &FacetMapTable<'_>,
) -> Vec<Vec<(usize, usize)>> {
    let mut columns = vec![Vec::new(); cache.facets.len()];
    for batch in batches {
        let entry = &resolved[batch.entry];
        for &position in &batch.facets {
            let facet = &cache.facets[position];
            let input_maps = &maps.input[position];
            let mut seen = HashSet::new();
            let mut list = Vec::new();
            for unknown in 0..entry.inputs.len() {
                for &cell_i in &facet.cell_indices {
                    let Some(reduced) = input_maps[unknown][cell_i] else {
                        continue;
                    };
                    if !seen.insert((unknown, reduced)) {
                        continue;
                    }
                    list.push((unknown, reduced));
                }
            }
            columns[position] = list;
        }
    }
    columns
}

/// Pointwise actions for one assembled chunk:
/// `blocks[((j * nlanes + sub) * noutputs + equation) * npts + q]`, where `j`
/// runs over the chunk's impulse-column positions.
pub(crate) struct AssembledChunkOut {
    pub(crate) positions: Vec<usize>,
    pub(crate) noutputs: usize,
    pub(crate) blocks: Vec<f64>,
}

/// Evaluate assembled-Jacobian actions for one chunk (phase A).
///
/// Inputs: cache, batch, resolved entry, map table, chunk positions, global
/// column lists, time, nonlinear state, scratch. Purpose: pack state traces,
/// then per impulse-column position pack impulse traces and call the
/// lane-packed action.
/// Output: owned pointwise blocks (lanes missing a column position compute on
/// zeros but are never stored).
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_assembled_chunk<'c>(
    cache: &'c QuadStateBoundaryCache,
    batch: &FacetBatch<'_>,
    entry: &ResolvedTensorBoundary,
    maps: &FacetMapTable<'_>,
    positions: &[usize],
    columns: &[Vec<(usize, usize)>],
    time: f64,
    state: MatRef<'_, f64>,
    scratch: &mut FacetChunkScratch<'c>,
) -> AssembledChunkOut {
    let ninputs = entry.inputs.len();
    let noutputs = entry.outputs.len();
    let npts = cache.wts.len();
    let include_gradients = entry.include_gradients;
    pack_state_chunk(cache, positions, entry, maps, time, state, scratch);
    let lane_state = LaneState {
        nfields: ninputs,
        npts,
        gdim: 2,
        values: &scratch.packed_state,
        grads: if include_gradients {
            &scratch.packed_state_grads
        } else {
            &[]
        },
        field_indices: &[],
    };
    let nlanes = positions.len();
    let maxcols = positions
        .iter()
        .map(|&position| columns[position].len())
        .max()
        .unwrap_or(0);
    let mut blocks = vec![0.0; maxcols * nlanes * noutputs * npts];
    for j in 0..maxcols {
        scratch.packed_dir.fill(0.0);
        if include_gradients {
            scratch.packed_dir_grads.fill(0.0);
        }
        for (lane, &position) in positions.iter().enumerate() {
            let lane_columns = &columns[position];
            let present = j < lane_columns.len();
            let (unknown, reduced) = if present { lane_columns[j] } else { (0, 0) };
            pack_impulse_lane(
                &cache.facets[position],
                &maps.input[position],
                unknown,
                reduced,
                present,
                npts,
                lane,
                include_gradients,
                &mut scratch.packed_dir,
                &mut scratch.packed_dir_grads,
            );
        }
        let lane_direction = LaneState {
            nfields: ninputs,
            npts,
            gdim: 2,
            values: &scratch.packed_dir,
            grads: if include_gradients {
                &scratch.packed_dir_grads
            } else {
                &[]
            },
            field_indices: &[],
        };

        for equation in 0..noutputs {
            for q in 0..npts {
                let mut lanes = [0.0; LANES];
                batch.kernel.tensor_jacobian_action(
                    &scratch.ctxs,
                    &lane_state,
                    &lane_direction,
                    equation,
                    q,
                    &mut lanes,
                );
                for (sub, _) in positions.iter().enumerate() {
                    blocks[((j * nlanes + sub) * noutputs + equation) * npts + q] = lanes[sub];
                }
            }
        }
    }
    AssembledChunkOut {
        positions: positions.to_vec(),
        noutputs,
        blocks,
    }
}

/// Stored pointwise values for the residual/apply paths, indexed by facet.
///
/// `data[offset[p] + equation * npts + q]` holds facet `p`'s flux/action;
/// `offset` is [`INACTIVE_FACET`] for facets without a kernel.
pub(crate) struct FacetPointwise {
    pub(crate) data: Vec<f64>,
    pub(crate) offset: Vec<usize>,
    pub(crate) entry_of: Vec<usize>,
}

/// Allocate the pointwise table (see [`FacetPointwise`]).
///
/// Inputs: cache, batches, resolved entries. Purpose: one flat store sized by
/// the boundary workload (`sum noutputs * npts`) with ascending offsets for
/// determinism. Output: zeroed table (filled by phase A chunk merges).
pub(crate) fn alloc_facet_pointwise(
    cache: &QuadStateBoundaryCache,
    batches: &[FacetBatch<'_>],
    resolved: &[ResolvedTensorBoundary],
) -> FacetPointwise {
    let npts = cache.wts.len();
    let mut offset = vec![INACTIVE_FACET; cache.facets.len()];
    let mut entry_of = vec![INACTIVE_FACET; cache.facets.len()];
    for batch in batches {
        for &position in &batch.facets {
            entry_of[position] = batch.entry;
        }
    }
    let mut total = 0;
    for position in 0..cache.facets.len() {
        if entry_of[position] == INACTIVE_FACET {
            continue;
        }
        let noutputs = resolved[entry_of[position]].outputs.len();
        offset[position] = total;
        total += noutputs * npts;
    }
    FacetPointwise {
        data: vec![0.0; total],
        offset,
        entry_of,
    }
}

/// Merge one chunk's blocks into the pointwise table.
///
/// Inputs: table, chunk output, facet `npts`. Purpose: gather
/// order-independent phase-A results into per-facet storage for the
/// order-sensitive phase-B scatter. Output: none (table blocks overwritten).
pub(crate) fn merge_chunk_blocks(table: &mut FacetPointwise, out: &LaneChunkOut, npts: usize) {
    let size = out.noutputs * npts;
    for (sub, &position) in out.positions.iter().enumerate() {
        let dst = table.offset[position];
        table.data[dst..dst + size].copy_from_slice(&out.blocks[sub * size..(sub + 1) * size]);
    }
}

/// Stored pointwise actions for the assembled Jacobian, indexed by facet and
/// impulse-column position:
/// `data[offset[p] + (j * noutputs + equation) * npts + q]`.
pub(crate) struct AssembledPointwise {
    pub(crate) data: Vec<f64>,
    pub(crate) offset: Vec<usize>,
    pub(crate) entry_of: Vec<usize>,
}

/// Allocate the assembled pointwise table (see [`AssembledPointwise`]).
///
/// Inputs: cache, batches, resolved entries, per-facet column lists. Purpose:
/// flat store with ascending offsets (see [`alloc_facet_pointwise`]).
/// Output: zeroed table.
pub(crate) fn alloc_assembled_pointwise(
    cache: &QuadStateBoundaryCache,
    batches: &[FacetBatch<'_>],
    resolved: &[ResolvedTensorBoundary],
    columns: &[Vec<(usize, usize)>],
) -> AssembledPointwise {
    let npts = cache.wts.len();
    let mut offset = vec![INACTIVE_FACET; cache.facets.len()];
    let mut entry_of = vec![INACTIVE_FACET; cache.facets.len()];
    for batch in batches {
        for &position in &batch.facets {
            entry_of[position] = batch.entry;
        }
    }
    let mut total = 0;
    for position in 0..cache.facets.len() {
        if entry_of[position] == INACTIVE_FACET {
            continue;
        }
        let noutputs = resolved[entry_of[position]].outputs.len();
        offset[position] = total;
        total += columns[position].len() * noutputs * npts;
    }
    AssembledPointwise {
        data: vec![0.0; total],
        offset,
        entry_of,
    }
}

/// Merge one assembled chunk's blocks (see [`merge_chunk_blocks`]).
///
/// Inputs: table, chunk output, global column lists, facet `npts`. Purpose:
/// gather the chunk's `[column][lane][equation][q]` blocks into per-facet
/// storage (only real column positions; padded/missing lanes are dropped).
/// Output: none (table blocks overwritten).
pub(crate) fn merge_assembled_blocks(
    table: &mut AssembledPointwise,
    out: &AssembledChunkOut,
    columns: &[Vec<(usize, usize)>],
    npts: usize,
) {
    let nlanes = out.positions.len();
    for (sub, &position) in out.positions.iter().enumerate() {
        let ncol = columns[position].len();
        let base = table.offset[position];
        for j in 0..ncol {
            for equation in 0..out.noutputs {
                let src = ((j * nlanes + sub) * out.noutputs + equation) * npts;
                let dst = base + (j * out.noutputs + equation) * npts;
                table.data[dst..dst + npts].copy_from_slice(&out.blocks[src..src + npts]);
            }
        }
    }
}

/// Run residual phase A over all chunks, serially.
///
/// Inputs: cache, batches, chunks, resolved entries, map table, time, state,
/// table. Purpose: evaluate every chunk with a scratch workspace reused across
/// chunks and merge blocks into the table. Output: none (table filled).
#[allow(clippy::too_many_arguments)]
pub(crate) fn residual_phase_a_serial(
    cache: &QuadStateBoundaryCache,
    batches: &[FacetBatch<'_>],
    chunks: &[FacetChunk],
    resolved: &[ResolvedTensorBoundary],
    maps: &FacetMapTable<'_>,
    time: f64,
    state: MatRef<'_, f64>,
    table: &mut FacetPointwise,
) {
    let mut scratch = FacetChunkScratch::new();
    let npts = cache.wts.len();
    for chunk in chunks {
        let batch = &batches[chunk.batch];
        let entry = &resolved[batch.entry];
        let positions = &batch.facets[chunk.start..chunk.start + chunk.len];
        let out = eval_residual_chunk(
            cache,
            batch,
            entry,
            maps,
            positions,
            time,
            state,
            &mut scratch,
        );
        merge_chunk_blocks(table, &out, npts);
    }
}

/// Run residual phase A over all chunks in parallel (see
/// [`residual_phase_a_serial`]).
///
/// Each thread owns its scratch; chunk outputs are merged serially on return.
/// Call inside the boundary thread pool.
#[allow(clippy::too_many_arguments)]
pub(crate) fn residual_phase_a_parallel(
    cache: &QuadStateBoundaryCache,
    batches: &[FacetBatch<'_>],
    chunks: &[FacetChunk],
    resolved: &[ResolvedTensorBoundary],
    maps: &FacetMapTable<'_>,
    time: f64,
    state: MatRef<'_, f64>,
    table: &mut FacetPointwise,
) {
    let outs: Vec<LaneChunkOut> = chunks
        .par_iter()
        .map_init(FacetChunkScratch::new, |scratch, chunk: &FacetChunk| {
            let batch = &batches[chunk.batch];
            let entry = &resolved[batch.entry];
            let positions = &batch.facets[chunk.start..chunk.start + chunk.len];
            eval_residual_chunk(cache, batch, entry, maps, positions, time, state, scratch)
        })
        .collect();
    let npts = cache.wts.len();
    for out in &outs {
        merge_chunk_blocks(table, out, npts);
    }
}

/// Run action phase A for one direction column over all chunks, serially.
///
/// Inputs: cache, batches, chunks, resolved entries, map table, time, state,
/// direction matrix, its column, table. Purpose: same as
/// [`residual_phase_a_serial`] for Jacobian actions. Output: none (table
/// filled for this column).
#[allow(clippy::too_many_arguments)]
pub(crate) fn action_phase_a_serial(
    cache: &QuadStateBoundaryCache,
    batches: &[FacetBatch<'_>],
    chunks: &[FacetChunk],
    resolved: &[ResolvedTensorBoundary],
    maps: &FacetMapTable<'_>,
    time: f64,
    state: MatRef<'_, f64>,
    direction: MatRef<'_, f64>,
    column: usize,
    table: &mut FacetPointwise,
) {
    let mut scratch = FacetChunkScratch::new();
    let npts = cache.wts.len();
    for chunk in chunks {
        let batch = &batches[chunk.batch];
        let entry = &resolved[batch.entry];
        let positions = &batch.facets[chunk.start..chunk.start + chunk.len];
        let out = eval_action_chunk(
            cache,
            batch,
            entry,
            maps,
            positions,
            time,
            state,
            direction,
            column,
            &mut scratch,
        );
        merge_chunk_blocks(table, &out, npts);
    }
}

/// Run action phase A for one direction column over all chunks in parallel
/// (see [`action_phase_a_serial`]).
///
/// Each thread owns its scratch; chunk outputs are merged serially on return.
/// Call inside the boundary thread pool.
#[allow(clippy::too_many_arguments)]
pub(crate) fn action_phase_a_parallel(
    cache: &QuadStateBoundaryCache,
    batches: &[FacetBatch<'_>],
    chunks: &[FacetChunk],
    resolved: &[ResolvedTensorBoundary],
    maps: &FacetMapTable<'_>,
    time: f64,
    state: MatRef<'_, f64>,
    direction: MatRef<'_, f64>,
    column: usize,
    table: &mut FacetPointwise,
) {
    let outs: Vec<LaneChunkOut> = chunks
        .par_iter()
        .map_init(FacetChunkScratch::new, |scratch, chunk: &FacetChunk| {
            let batch = &batches[chunk.batch];
            let entry = &resolved[batch.entry];
            let positions = &batch.facets[chunk.start..chunk.start + chunk.len];
            eval_action_chunk(
                cache, batch, entry, maps, positions, time, state, direction, column, scratch,
            )
        })
        .collect();
    let npts = cache.wts.len();
    for out in &outs {
        merge_chunk_blocks(table, out, npts);
    }
}

/// Run assembled-Jacobian phase A over all chunks, serially.
///
/// Inputs: cache, batches, chunks, resolved entries, map table, column lists,
/// time, state, table. Purpose: evaluate every chunk's impulse actions with a
/// reused scratch workspace and merge blocks into the table. Output: none
/// (table filled).
#[allow(clippy::too_many_arguments)]
pub(crate) fn assembled_phase_a_serial(
    cache: &QuadStateBoundaryCache,
    batches: &[FacetBatch<'_>],
    chunks: &[FacetChunk],
    resolved: &[ResolvedTensorBoundary],
    maps: &FacetMapTable<'_>,
    columns: &[Vec<(usize, usize)>],
    time: f64,
    state: MatRef<'_, f64>,
    table: &mut AssembledPointwise,
) {
    let mut scratch = FacetChunkScratch::new();
    let npts = cache.wts.len();
    for chunk in chunks {
        let batch = &batches[chunk.batch];
        let entry = &resolved[batch.entry];
        let positions = &batch.facets[chunk.start..chunk.start + chunk.len];
        let out = eval_assembled_chunk(
            cache,
            batch,
            entry,
            maps,
            positions,
            columns,
            time,
            state,
            &mut scratch,
        );
        merge_assembled_blocks(table, &out, columns, npts);
    }
}

#[cfg(test)]
mod tests {
    use super::super::boundary::tensor_boundary_direction_into;
    use super::super::boundary::tensor_boundary_state_into;
    use super::*;
    use crate::kernels::common::StateTensorBoundaryIntegrator;
    use faer::prelude::Mat;

    /// Deterministic pseudo-random value in `[lo, hi)`.
    fn prng_range(seed: &mut u64, lo: f64, hi: f64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (*seed >> 11) | 0x3FF0_0000_0000_0000;
        lo + (hi - lo) * (f64::from_bits(bits) - 1.0)
    }

    /// Build a fake boundary facet with random trace data.
    fn fake_facet(seed: &mut u64, local_index: usize, normal: [f64; 2]) -> QuadStateBoundaryFacet {
        let npts = 4;
        let ncell = 9;
        let cell_indices = vec![0, 4, 8];
        let nfacet = cell_indices.len();
        let values = (0..nfacet * npts)
            .map(|_| prng_range(seed, -2.0, 2.0))
            .collect();
        let cell_grads = (0..ncell * 2 * npts)
            .map(|_| prng_range(seed, -3.0, 3.0))
            .collect();
        let points = (0..npts * 2).map(|i| 0.1 * i as f64 - 0.2).collect();
        QuadStateBoundaryFacet {
            facet: crate::regions::FacetMeta {
                local_index,
                physical_region: None,
            },
            cell_index: 0,
            facet_dofs: vec![0, 1, 2],
            cell_indices,
            values,
            grads: Vec::new(),
            cell_grads,
            jfacet_det: vec![0.5; npts],
            points,
            normal,
        }
    }

    /// Test fixture: three distinct facets, mixed eliminated/prescribed DOFs.
    struct Fixture {
        cache: QuadStateBoundaryCache,
        field_maps: Vec<Vec<Vec<Option<usize>>>>,
        prescribed: Vec<Vec<Vec<Option<f64>>>>,
        input_offsets: Vec<usize>,
        state: Mat<f64>,
        resolved: Vec<ResolvedTensorBoundary>,
    }

    impl Fixture {
        fn build(include_gradients: bool) -> Self {
            let mut seed = 0x1234_5678u64;
            let cache = QuadStateBoundaryCache {
                wts: vec![0.25, 0.5, 0.75, 1.0],
                facets: vec![
                    fake_facet(&mut seed, 3, [1.0, 0.0]),
                    fake_facet(&mut seed, 5, [0.0, 1.0]),
                    fake_facet(&mut seed, 7, [0.6, 0.8]),
                ],
                cell_facets: vec![vec![0, 1, 2]],
            };
            // Three fields over a nine-DOF cell: field 1 eliminates DOF 4
            // (prescribed 0.5), field 2 eliminates DOF 8 (no prescription).
            let mut field_maps = Vec::new();
            let mut prescribed = Vec::new();
            for field in 0..3 {
                let mut map = vec![Some(0); 9];
                let mut pre = vec![None; 9];
                for dof in 0..9 {
                    map[dof] = Some(field * 9 + dof);
                }
                if field == 1 {
                    map[4] = None;
                    pre[4] = Some(0.5);
                }
                if field == 2 {
                    map[8] = None;
                }
                field_maps.push(vec![map]);
                prescribed.push(vec![pre]);
            }
            let input_offsets = vec![0, 0, 0];
            let mut state = Mat::<f64>::zeros(27, 1);
            for i in 0..27 {
                state[(i, 0)] = 0.1 * (i as f64 + 1.0) - 1.4;
            }
            let resolved = vec![ResolvedTensorBoundary {
                key: (1, 2),
                inputs: vec![0, 1, 2],
                outputs: vec![0, 1, 2],
                input_offsets: input_offsets.clone(),
                output_offsets: input_offsets.clone(),
                include_gradients,
            }];
            Self {
                cache,
                field_maps,
                prescribed,
                input_offsets,
                state,
                resolved,
            }
        }

        fn maps(&self) -> FacetMapTable<'_> {
            let entry = &self.resolved[0];
            let mut table = FacetMapTable {
                input: vec![Vec::new(); 3],
                prescribed: vec![Vec::new(); 3],
                output: vec![Vec::new(); 3],
            };
            let mut a = Vec::new();
            let mut b = Vec::new();
            let mut c = Vec::new();
            for position in 0..3 {
                fill_maps_for_cell(&self.field_maps, &entry.inputs, 0, &mut a);
                fill_maps_for_cell(&self.prescribed, &entry.inputs, 0, &mut b);
                fill_maps_for_cell(&self.field_maps, &entry.outputs, 0, &mut c);
                table.input[position] = std::mem::take(&mut a);
                table.prescribed[position] = std::mem::take(&mut b);
                table.output[position] = std::mem::take(&mut c);
            }
            table
        }
    }

    /// Assert two scalars are bitwise equal.
    fn assert_bits_eq(a: f64, b: f64, what: &str) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{what}: {a:e} != {b:e} (bits {:x} != {:x})",
            a.to_bits(),
            b.to_bits()
        );
    }
    /// Replicate one scalar trace across all `LANES` lanes.
    ///
    /// # Arguments
    /// * `trace` - scalar trace buffer (`[slot]`).
    ///
    /// # Returns
    /// Lane-packed buffer (`[slot * LANES + lane]`) with every lane equal.
    fn replicate_lane(trace: &[f64]) -> Vec<f64> {
        let mut packed = vec![0.0; trace.len() * LANES];
        for (slot, &v) in trace.iter().enumerate() {
            for lane in 0..LANES {
                packed[slot * LANES + lane] = v;
            }
        }
        packed
    }

    #[test]
    fn packed_state_matches_scalar_interpolation() {
        for include_gradients in [false, true] {
            let fixture = Fixture::build(include_gradients);
            let npts = fixture.cache.wts.len();
            let maps = fixture.maps();
            let entry = &fixture.resolved[0];
            let positions = vec![0, 1, 2];
            let mut scratch = FacetChunkScratch::new();
            pack_state_chunk(
                &fixture.cache,
                &positions,
                entry,
                &maps,
                0.25,
                fixture.state.as_ref(),
                &mut scratch,
            );
            assert_eq!(scratch.ctxs.len(), LANES);
            for (lane, &position) in positions.iter().enumerate() {
                let facet = &fixture.cache.facets[position];
                let mut expected_values = vec![0.0; 3 * npts];
                let mut expected_grads = vec![0.0; 3 * 2 * npts];
                tensor_boundary_state_into(
                    facet,
                    3,
                    &maps.input[position],
                    &maps.prescribed[position],
                    fixture.state.as_ref(),
                    &fixture.input_offsets,
                    npts,
                    include_gradients,
                    &mut expected_values,
                    &mut expected_grads,
                );
                for field in 0..3 {
                    for q in 0..npts {
                        assert_bits_eq(
                            scratch.packed_state[(field * npts + q) * LANES + lane],
                            expected_values[field * npts + q],
                            &format!("value grad={include_gradients} lane={lane} f={field} q={q}"),
                        );
                        if include_gradients {
                            for gd in 0..2 {
                                assert_bits_eq(
                                    scratch.packed_state_grads
                                        [((field * 2 + gd) * npts + q) * LANES + lane],
                                    expected_grads[(field * 2 + gd) * npts + q],
                                    &format!("grad lane={lane} f={field} d={gd} q={q}"),
                                );
                            }
                        }
                    }
                }
                // Per-lane context borrows the lane's own facet data.
                assert_eq!(
                    scratch.ctxs[lane].facet.local_index,
                    facet.facet.local_index
                );
                assert_eq!(scratch.ctxs[lane].normal, facet.normal);
            }
            // Padded lanes copy the last real facet's context and traces.
            for lane in 3..LANES {
                assert_eq!(scratch.ctxs[lane].facet.local_index, 7);
                for slot in 0..3 * npts {
                    assert_bits_eq(
                        scratch.packed_state[slot * LANES + lane],
                        scratch.packed_state[slot * LANES + 2],
                        &format!("padded value lane={lane} slot={slot}"),
                    );
                }
            }
        }
    }

    #[test]
    fn packed_direction_matches_scalar_interpolation() {
        for include_gradients in [false, true] {
            let fixture = Fixture::build(include_gradients);
            let npts = fixture.cache.wts.len();
            let maps = fixture.maps();
            let entry = &fixture.resolved[0];
            let positions = vec![0, 1, 2];
            let mut direction = Mat::<f64>::zeros(27, 2);
            for i in 0..27 {
                direction[(i, 0)] = 0.05 * (i as f64) - 0.3;
                direction[(i, 1)] = -0.07 * (i as f64) + 0.2;
            }
            let mut scratch = FacetChunkScratch::new();
            scratch.ensure(3, npts);
            for column in 0..2 {
                pack_direction_chunk(
                    &fixture.cache,
                    &positions,
                    entry,
                    &maps,
                    direction.as_ref(),
                    column,
                    &mut scratch,
                );
                for (lane, &position) in positions.iter().enumerate() {
                    let facet = &fixture.cache.facets[position];
                    let mut expected_values = vec![0.0; 3 * npts];
                    let mut expected_grads = vec![0.0; 3 * 2 * npts];
                    tensor_boundary_direction_into(
                        facet,
                        3,
                        &maps.input[position],
                        direction.as_ref(),
                        &fixture.input_offsets,
                        column,
                        npts,
                        include_gradients,
                        &mut expected_values,
                        &mut expected_grads,
                    );
                    for field in 0..3 {
                        for q in 0..npts {
                            assert_bits_eq(
                                scratch.packed_dir[(field * npts + q) * LANES + lane],
                                expected_values[field * npts + q],
                                &format!("dir grad={include_gradients} col={column} lane={lane}"),
                            );
                        }
                    }
                }
            }
        }
    }

    /// Chunk residual blocks match the lane kernel evaluated per facet.
    ///
    /// For every facet position the expected value comes from the lane-packed
    /// kernel on that facet's trace replicated across all lanes (lane 0 is
    /// compared). This keeps covering the chunk pack -> kernel -> block
    /// wiring without a scalar kernel path.
    #[test]
    fn residual_chunk_matches_lane_kernel() {
        let kernel =
            crate::kernels::edac::tensor::split_boundary_flux::TensorKernelEdacSplitBoundaryFlux2D;
        for include_gradients in [false, true] {
            let fixture = Fixture::build(include_gradients);
            let npts = fixture.cache.wts.len();
            let maps = fixture.maps();
            let entry = &fixture.resolved[0];
            let positions = vec![0, 1, 2];
            let batch = FacetBatch {
                kernel: &kernel as &dyn StateTensorBoundaryIntegrator<2>,
                key: (1, 2),
                entry: 0,
                facets: positions.clone(),
            };
            let mut scratch = FacetChunkScratch::new();
            let out = eval_residual_chunk(
                &fixture.cache,
                &batch,
                entry,
                &maps,
                &positions,
                0.25,
                fixture.state.as_ref(),
                &mut scratch,
            );
            for (sub, &position) in positions.iter().enumerate() {
                let facet = &fixture.cache.facets[position];
                let mut trace_values = vec![0.0; 3 * npts];
                let mut trace_grads = vec![0.0; 3 * 2 * npts];
                tensor_boundary_state_into(
                    facet,
                    3,
                    &maps.input[position],
                    &maps.prescribed[position],
                    fixture.state.as_ref(),
                    &fixture.input_offsets,
                    npts,
                    include_gradients,
                    &mut trace_values,
                    &mut trace_grads,
                );
                let ctx = TensorFacetCtx {
                    time: 0.25,
                    facet: facet.facet,
                    npts,
                    wts: &fixture.cache.wts,
                    jfacet_det: &facet.jfacet_det,
                    points: &facet.points,
                    normal: &facet.normal,
                };
                let ctxs: [TensorFacetCtx<'_>; LANES] = std::array::from_fn(|_| TensorFacetCtx {
                    time: ctx.time,
                    facet: ctx.facet,
                    npts: ctx.npts,
                    wts: ctx.wts,
                    jfacet_det: ctx.jfacet_det,
                    points: ctx.points,
                    normal: ctx.normal,
                });
                let packed_values = replicate_lane(&trace_values);
                let packed_grads = if include_gradients {
                    replicate_lane(&trace_grads)
                } else {
                    Vec::new()
                };
                let lane_state = LaneState {
                    nfields: 3,
                    npts,
                    gdim: 2,
                    values: &packed_values,
                    grads: &packed_grads,
                    field_indices: &[],
                };
                for equation in 0..3 {
                    for q in 0..npts {
                        let mut lanes = [0.0; LANES];
                        kernel.tensor_residual(&ctxs, &lane_state, equation, q, &mut lanes);
                        assert_bits_eq(
                            out.blocks[(sub * 3 + equation) * npts + q],
                            lanes[0],
                            &format!("residual lane={sub} eq={equation} q={q}"),
                        );
                    }
                }
            }
        }
    }

    /// Assembled chunk blocks match the lane kernel evaluated per facet column.
    ///
    /// For every facet position and unit-impulse column the expected value
    /// comes from the lane-packed Jacobian action on that facet's state and
    /// direction traces replicated across all lanes (lane 0 is compared).
    /// This keeps covering the impulse pack -> kernel -> block wiring without
    /// a scalar kernel path.
    #[test]
    fn assembled_chunk_matches_lane_kernel() {
        let kernel =
            crate::kernels::edac::tensor::split_boundary_flux::TensorKernelEdacSplitBoundaryFlux2D;
        for include_gradients in [false, true] {
            let fixture = Fixture::build(include_gradients);
            let npts = fixture.cache.wts.len();
            let maps = fixture.maps();
            let entry = &fixture.resolved[0];
            let positions = vec![0, 1, 2];
            // Column lists over the fixture maps (all DOFs present except the
            // eliminated ones).
            let batch_probe = FacetBatch {
                kernel: &kernel as &dyn StateTensorBoundaryIntegrator<2>,
                key: (1, 2),
                entry: 0,
                facets: positions.clone(),
            };
            let batches = [batch_probe];
            let columns = boundary_column_lists(&fixture.cache, &batches, &fixture.resolved, &maps);
            let batch = FacetBatch {
                kernel: &kernel as &dyn StateTensorBoundaryIntegrator<2>,
                key: (1, 2),
                entry: 0,
                facets: positions.clone(),
            };
            let mut scratch = FacetChunkScratch::new();
            let out = eval_assembled_chunk(
                &fixture.cache,
                &batch,
                entry,
                &maps,
                &positions,
                &columns,
                0.25,
                fixture.state.as_ref(),
                &mut scratch,
            );
            for (sub, &position) in positions.iter().enumerate() {
                let facet = &fixture.cache.facets[position];
                let mut state_values = vec![0.0; 3 * npts];
                let mut state_grads = vec![0.0; 3 * 2 * npts];
                tensor_boundary_state_into(
                    facet,
                    3,
                    &maps.input[position],
                    &maps.prescribed[position],
                    fixture.state.as_ref(),
                    &fixture.input_offsets,
                    npts,
                    include_gradients,
                    &mut state_values,
                    &mut state_grads,
                );
                let ctx = TensorFacetCtx {
                    time: 0.25,
                    facet: facet.facet,
                    npts,
                    wts: &fixture.cache.wts,
                    jfacet_det: &facet.jfacet_det,
                    points: &facet.points,
                    normal: &facet.normal,
                };
                let ctxs: [TensorFacetCtx<'_>; LANES] = std::array::from_fn(|_| TensorFacetCtx {
                    time: ctx.time,
                    facet: ctx.facet,
                    npts: ctx.npts,
                    wts: ctx.wts,
                    jfacet_det: ctx.jfacet_det,
                    points: ctx.points,
                    normal: ctx.normal,
                });
                let packed_state_values = replicate_lane(&state_values);
                let packed_state_grads = if include_gradients {
                    replicate_lane(&state_grads)
                } else {
                    Vec::new()
                };
                let lane_state = LaneState {
                    nfields: 3,
                    npts,
                    gdim: 2,
                    values: &packed_state_values,
                    grads: &packed_state_grads,
                    field_indices: &[],
                };
                for (j, &(unknown, reduced)) in columns[position].iter().enumerate() {
                    let mut dir_values = vec![0.0; 3 * npts];
                    let mut dir_grads = vec![0.0; 3 * 2 * npts];
                    for (facet_i, &basis_cell_i) in facet.cell_indices.iter().enumerate() {
                        if maps.input[position][unknown][basis_cell_i] != Some(reduced) {
                            continue;
                        }
                        for q in 0..npts {
                            dir_values[unknown * npts + q] += facet.values[facet_i * npts + q];
                        }
                    }
                    if include_gradients {
                        for (cell_i, cell_grads) in
                            facet.cell_grads.chunks_exact(2 * npts).enumerate()
                        {
                            if maps.input[position][unknown][cell_i] != Some(reduced) {
                                continue;
                            }
                            for q in 0..npts {
                                for gd in 0..2 {
                                    dir_grads[(unknown * 2 + gd) * npts + q] +=
                                        cell_grads[gd * npts + q];
                                }
                            }
                        }
                    }
                    let packed_dir_values = replicate_lane(&dir_values);
                    let packed_dir_grads = if include_gradients {
                        replicate_lane(&dir_grads)
                    } else {
                        Vec::new()
                    };
                    let lane_dir = LaneState {
                        nfields: 3,
                        npts,
                        gdim: 2,
                        values: &packed_dir_values,
                        grads: &packed_dir_grads,
                        field_indices: &[],
                    };
                    for equation in 0..3 {
                        for q in 0..npts {
                            let mut lanes = [0.0; LANES];
                            kernel.tensor_jacobian_action(
                                &ctxs,
                                &lane_state,
                                &lane_dir,
                                equation,
                                q,
                                &mut lanes,
                            );
                            assert_bits_eq(
                                out.blocks[((j * 3 + sub) * 3 + equation) * npts + q],
                                lanes[0],
                                &format!("assembled lane={sub} col={j} eq={equation} q={q}"),
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn phase_a_is_deterministic() {
        let kernel =
            crate::kernels::edac::tensor::split_boundary_flux::TensorKernelEdacSplitBoundaryFlux2D;
        let fixture = Fixture::build(false);
        let maps = fixture.maps();
        let entry = &fixture.resolved[0];
        let positions = vec![0, 1, 2];
        let batch = FacetBatch {
            kernel: &kernel as &dyn StateTensorBoundaryIntegrator<2>,
            key: (1, 2),
            entry: 0,
            facets: positions.clone(),
        };
        // Residual blocks are identical across repeated evaluations.
        let mut first_scratch = FacetChunkScratch::new();
        let first = eval_residual_chunk(
            &fixture.cache,
            &batch,
            entry,
            &maps,
            &positions,
            0.25,
            fixture.state.as_ref(),
            &mut first_scratch,
        );
        let mut second_scratch = FacetChunkScratch::new();
        let second = eval_residual_chunk(
            &fixture.cache,
            &batch,
            entry,
            &maps,
            &positions,
            0.25,
            fixture.state.as_ref(),
            &mut second_scratch,
        );
        assert_eq!(first.blocks.len(), second.blocks.len());
        for (i, (&a, &b)) in first.blocks.iter().zip(&second.blocks).enumerate() {
            assert_bits_eq(a, b, &format!("residual determinism block {i}"));
        }
        // Assembled blocks are identical across repeated evaluations.
        let batches = [FacetBatch {
            kernel: &kernel as &dyn StateTensorBoundaryIntegrator<2>,
            key: (1, 2),
            entry: 0,
            facets: positions.clone(),
        }];
        let columns = boundary_column_lists(&fixture.cache, &batches, &fixture.resolved, &maps);
        let first_a = eval_assembled_chunk(
            &fixture.cache,
            &batch,
            entry,
            &maps,
            &positions,
            &columns,
            0.25,
            fixture.state.as_ref(),
            &mut first_scratch,
        );
        let second_a = eval_assembled_chunk(
            &fixture.cache,
            &batch,
            entry,
            &maps,
            &positions,
            &columns,
            0.25,
            fixture.state.as_ref(),
            &mut second_scratch,
        );
        assert_eq!(first_a.blocks.len(), second_a.blocks.len());
        for (i, (&a, &b)) in first_a.blocks.iter().zip(&second_a.blocks).enumerate() {
            assert_bits_eq(a, b, &format!("assembled determinism block {i}"));
        }
    }
}
