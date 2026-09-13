//! Sum-factorized tensor-product (`TensorResidualKernel`) paths (2D).
use crate::common::{
    apply_quad_state_tensor_boundary_terms_cached,
    assemble_quad_state_tensor_boundary_jacobian_cached,
    assemble_quad_state_tensor_boundary_residual_cached, interpolate_tensor_cell_coefficients, interpolate_tensor_cell_state, push_rectangular_local_matrix_triplets, CellState, DisjointOut,
    FieldDofLayout,
    TensorCtx, rayon_cell_chunk_size,
};
use crate::common::batch::{
    TensorLaneScratch, extract_lanes, integrate_batch_2d, interpolate_batch_2d,
};
use crate::common::batch::SIMD_CELL_WIDTH;
use crate::kernels::common::{
    apply_tensor_jacobian, assemble_tensor_residual, StateTensorBoundaryTerms,
    TensorResidualKernel,
};
use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};

use ndelement::types::ReferenceCellType;
use ndmesh::traits::Mesh;
use rayon::prelude::*;


use super::problem::SEM2DProblem;

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> SEM2DProblem<M> {
    /// Sum-factorized tensor residual with a caller-provided field layout.
    ///
    /// Mathematics: `R(U) = Σ_e E_eᵀ r_e` with 2D sum-factorized GLL
    /// interpolation/integration (`O(p³)` per quad vs `O(p⁴)` weak) and
    /// SIMD-over-element batching. `layout` avoids recomputation when the
    /// caller already holds it; external callers should use
    /// [`TensorResidualOps::assemble_tensor_residual`][crate::sem_traits::TensorResidualOps].
    pub(crate) fn assemble_tensor_residual_with_layout<K: TensorResidualKernel<2> + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        layout: &FieldDofLayout,
    ) -> Vec<f64>
    where
        M: Sync,
    {
        let selection = self.fields.resolve_selection(
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

        let cd = &self.cell_data;
        let local_stride = noutputs * cd.ndofs;
        // ponytail: one color-sequential region scatters straight into the
        // global residual; no actions buffer, no second reduction pass.
        let mut residual = vec![0.0; layout.total_size];
        {
            let out = unsafe { DisjointOut::new(&mut residual) };
            for color_cells in self.restriction.cell_colors() {
                color_cells
                    .par_chunks(SIMD_CELL_WIDTH)
                    .for_each_init(
                        || {
                            (
                                vec![0.0; ninputs * cd.ndofs],
                                vec![0.0; ninputs * cd.npts],
                                vec![0.0; ninputs * 2 * cd.npts],
                                vec![0.0; local_stride],
                                TensorLaneScratch::new(
                                    ninputs,
                                    noutputs,
                                    cd.ndofs,
                                    cd.npts,
                                    2,
                                ),
                            )
                        },
                        |(coefficients, field_values, field_grads, local, lane),
                         chunk: &[usize]| {
                            let w = SIMD_CELL_WIDTH;
                            let uniform = chunk.len() == w
                                && chunk.iter().all(|&cell| {
                                    self.cell_reduced_dofs[0][cell].len() == cd.ndofs
                                });
                            if !uniform {
                                for &cell in chunk {
                                    let (field_maps, field_prescribed) =
                                        self.restriction.maps_for(cell, &selection.inputs);
                                    let ndofs = field_maps[0].len();
                                    let cell_size = noutputs * ndofs;
                                    let state_cell = interpolate_tensor_cell_state(
                                        cd,
                                        2,
                                        ninputs,
                                        &field_maps,
                                        &field_prescribed,
                                        &input_offsets,
                                        state,
                                        &mut coefficients[..ninputs * cd.ndofs],
                                        &mut field_values[..ninputs * cd.npts],
                                        &mut field_grads[..ninputs * 2 * cd.npts],
                                        cell,
                                    );
                                    let ctx = self.tensor_ctx(time, cell);
                                    assemble_tensor_residual(
                                        kernel,
                                        &ctx,
                                        &state_cell,
                                        &mut local[..cell_size],
                                    );
                                    // SAFETY: same-color cells are row-disjoint.
                                    unsafe {
                                        self.restriction.scatter_add_column(
                                            cell,
                                            &selection.outputs,
                                            &local[..cell_size],
                                            out,
                                        )
                                    };
                                }
                                return;
                            }
                            self.restriction.gather_state_batch(
                                chunk,
                                &selection.inputs,
                                &input_offsets,
                                state,
                                &mut lane.packed_coeffs,
                                w,
                            );
                            interpolate_batch_2d(
                                cd,
                                ninputs,
                                &lane.packed_coeffs,
                                &mut lane.lane_values,
                                &mut lane.lane_grads,
                                chunk,
                                w,
                            );
                            extract_lanes(
                                &lane.lane_values,
                                &lane.lane_grads,
                                2,
                                ninputs,
                                cd.npts,
                                w,
                                w,
                                &mut lane.scalar_values,
                                &mut lane.scalar_grads,
                            );
                            let ctxs: Vec<TensorCtx> =
                                chunk.iter().map(|&c| self.tensor_ctx(time, c)).collect();
                            let states: Vec<CellState> = (0..w)
                                .map(|l| CellState {
                                    nfields: ninputs,
                                    npts: cd.npts,
                                    gdim: 2,
                                    values: &lane.scalar_values
                                        [l * ninputs * cd.npts..(l + 1) * ninputs * cd.npts],
                                    grads: &lane.scalar_grads[l * ninputs * 2 * cd.npts
                                        ..(l + 1) * ninputs * 2 * cd.npts],
                                    field_indices: &[],
                                })
                                .collect();
                            let mut t0 = [0.0f64; SIMD_CELL_WIDTH];
                            let mut t1x = [0.0f64; SIMD_CELL_WIDTH];
                            let mut t1y = [0.0f64; SIMD_CELL_WIDTH];
                            for eq in 0..noutputs {
                                for q in 0..cd.npts {
                                    kernel.tensor_residual_batch(
                                        &ctxs,
                                        &states,
                                        eq,
                                        q,
                                        &mut t0[..w],
                                        &mut t1x[..w],
                                        &mut t1y[..w],
                                    );
                                    for l in 0..w {
                                        lane.flux0[(eq * cd.npts + q) * w + l] = t0[l];
                                        lane.flux1x[(eq * cd.npts + q) * w + l] = t1x[l];
                                        lane.flux1y[(eq * cd.npts + q) * w + l] = t1y[l];
                                    }
                                }
                            }
                            lane.packed_out.fill(0.0);
                            integrate_batch_2d(
                                cd,
                                noutputs,
                                &lane.flux0,
                                &lane.flux1x,
                                &lane.flux1y,
                                &mut lane.packed_out,
                                chunk,
                                w,
                            );
                            for (l, &cell) in chunk.iter().enumerate() {
                                for o in 0..local_stride {
                                    lane.cell_local[o] = lane.packed_out[o * w + l];
                                }
                                // SAFETY: same-color cells are row-disjoint.
                                unsafe {
                                    self.restriction.scatter_add_column(
                                        cell,
                                        &selection.outputs,
                                        &lane.cell_local,
                                        out,
                                    )
                                };
                            }
                        },
                    );
            }
        }
        residual
    }

    /// Tensor state-boundary residual via the cached quad tensor assembly.
    ///
    /// Added to the tensor volume residual by the tensor/mixed operators when
    /// `StateTensorBoundaryTerms` are attached.
    pub(crate) fn assemble_state_tensor_boundary_residual(
        &self,
        time: f64,
        state: MatRef<f64>,
        terms: &StateTensorBoundaryTerms<2>,
    ) -> Vec<f64> {
        assemble_quad_state_tensor_boundary_residual_cached(
            &self.state_boundary_cache,
            &self.fields,
            time,
            state,
            &self.cell_reduced_dofs,
            &self.cell_prescribed_values,
            &self.field_layout().offsets,
            &self.restriction,
            terms,
        )
    }

    /// Tensor state-boundary Jacobian used by the tensor/mixed operators.
    pub(crate) fn assemble_state_tensor_boundary_jacobian(
        &self,
        time: f64,
        state: MatRef<f64>,
        terms: &StateTensorBoundaryTerms<2>,
    ) -> SparseColMat<usize, f64> {
        assemble_quad_state_tensor_boundary_jacobian_cached(
            &self.state_boundary_cache,
            &self.fields,
            time,
            state,
            &self.cell_reduced_dofs,
            &self.cell_prescribed_values,
            &self.field_layout().offsets,
            terms,
        )
    }

    /// Matrix-free action of the tensor state-boundary Jacobian.
    pub(crate) fn apply_state_tensor_boundary_jacobian(
        &self,
        time: f64,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        terms: &StateTensorBoundaryTerms<2>,
    ) -> Mat<f64> {
        apply_quad_state_tensor_boundary_terms_cached(
            &self.state_boundary_cache,
            &self.fields,
            time,
            state,
            direction,
            &self.cell_reduced_dofs,
            &self.cell_prescribed_values,
            &self.field_layout().offsets,
            &self.restriction,
            terms,
        )
    }

    /// Tensor Jacobian with a caller-provided layout, via unit-impulse columns.
    ///
    /// Mathematics: column `(unknown, trial)` of each cell block is the tensor
    /// action on a unit impulse, scattered as triplets. Same sparsity as the
    /// weak Jacobian; cheaper formation. External callers:
    /// [`TensorResidualOps::assemble_tensor_jacobian`][crate::sem_traits::TensorResidualOps].
    pub(crate) fn assemble_tensor_jacobian_with_layout<K: TensorResidualKernel<2> + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        layout: &FieldDofLayout,
    ) -> SparseColMat<usize, f64>
    where
        M: Sync,
    {
        let chunk = rayon_cell_chunk_size(self.cell_reduced_dofs[0].len());
        let selection = self.fields.resolve_selection(
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
        let cd = &self.cell_data;
        let row_size = noutputs * cd.ndofs;
        let col_size = ninputs * cd.ndofs;
        let batches: Vec<Vec<Triplet<usize, usize, f64>>> = self.cell_reduced_dofs[0]
            .par_chunks(chunk)
            .enumerate()
            .map_init(
                || {
                    (
                        vec![0.0; col_size],
                        vec![0.0; ninputs * cd.npts],
                        vec![0.0; ninputs * 2 * cd.npts],
                        vec![0.0; ninputs * cd.npts],
                        vec![0.0; ninputs * 2 * cd.npts],
                        vec![0.0; row_size * col_size],
                        vec![0.0; col_size.max(row_size)],
                    )
                },
                |(
                    state_coefficients,
                    field_values,
                    field_grads,
                    direction_values,
                    direction_grads,
                    local_matrix,
                    local_direction,
                ),
                 (batch_index, reduced_batch)| {
                    let mut triplets =
                        Vec::with_capacity(reduced_batch.len() * row_size * col_size);
                    for (cell_offset, reduced_dofs) in reduced_batch.iter().enumerate() {
                        let ndofs = reduced_dofs.len();
                        let row_cell_size = noutputs * ndofs;
                        let col_cell_size = ninputs * ndofs;
                        let cell_index = batch_index * chunk + cell_offset;
                        let (field_maps, field_prescribed) =
                            self.restriction.maps_for(cell_index, &selection.inputs);
                        let state_cell = interpolate_tensor_cell_state(
                            cd,
                            2,
                            ninputs,
                            &field_maps,
                            &field_prescribed,
                            &input_offsets,
                            state,
                            &mut state_coefficients[..col_cell_size],
                            &mut field_values[..ninputs * cd.npts],
                            &mut field_grads[..ninputs * 2 * cd.npts],
                            cell_index,
                        );
                        let ctx = self.tensor_ctx(time, cell_index);
                        local_matrix[..row_cell_size * col_cell_size].fill(0.0);
                        for unknown in 0..ninputs {
                            for trial in 0..ndofs {
                                local_direction[..col_cell_size].fill(0.0);
                                local_direction[unknown * ndofs + trial] = 1.0;
                                let direction_cell = interpolate_tensor_cell_coefficients(
                                    cd,
                                    2,
                                    ninputs,
                                    &local_direction[..col_cell_size],
                                    &mut direction_values[..ninputs * cd.npts],
                                    &mut direction_grads[..ninputs * 2 * cd.npts],
                                    cell_index,
                                );
                                apply_tensor_jacobian(
                                    kernel,
                                    &ctx,
                                    &state_cell,
                                    &direction_cell,
                                    &mut local_direction[..row_cell_size],
                                );
                                for equation in 0..noutputs {
                                    for test in 0..ndofs {
                                        local_matrix[(equation * ndofs + test) * col_cell_size
                                            + unknown * ndofs
                                            + trial] = local_direction[equation * ndofs + test];
                                    }
                                }
                            }
                        }
                        let (row_maps, _) =
                            self.restriction.maps_for(cell_index, &selection.outputs);
                        push_rectangular_local_matrix_triplets(
                            &mut triplets,
                            &local_matrix[..row_cell_size * col_cell_size],
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
        self.jacobian_pattern_cache.assemble(
            system_size,
            &selection.inputs,
            &selection.outputs,
            triplets,
        )
    }

    /// Matrix-free tensor Jacobian action with a caller-provided layout.
    ///
    /// Inputs: `state` (`N×1`), `direction` (`N×ncols`); `out` (`N×ncols`) is
    /// fully overwritten via color-disjoint scatters. External callers:
    /// [`TensorResidualOps::apply_tensor_jacobian_into`][crate::sem_traits::TensorResidualOps].
    pub(crate) fn apply_tensor_jacobian_with_layout<K: TensorResidualKernel<2> + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        layout: &FieldDofLayout,
        mut out: MatMut<'_, f64>,
    ) where
        M: Sync,
    {
        let selection = self.fields.resolve_selection(
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
        let cd = &self.cell_data;
        let input_size = ninputs * cd.ndofs;
        let output_size = noutputs * cd.ndofs;
        let ncols = direction.ncols();
        if ncols == 0 {
            return;
        }
        // ponytail: column-major global actions scattered in one region per
        // column; no cell-major buffer, no second reduction pass. The final
        // copy fully overwrites `out`, so no pre-zeroing is needed.
        let total = layout.total_size;
        let mut actions = vec![0.0; total * ncols];
        {
            let mut outs = Vec::with_capacity(ncols);
            for column_actions in actions.chunks_mut(total) {
                // SAFETY: column slices are disjoint; scatters within one
                // color are row-disjoint (see `cell_colors`).
                outs.push(unsafe { DisjointOut::new(column_actions) });
            }
            for color_cells in self.restriction.cell_colors() {
                color_cells
                    .par_chunks(SIMD_CELL_WIDTH)
                    .for_each_init(
                        || {
                            (
                                vec![0.0; input_size],
                                vec![0.0; ninputs * cd.npts],
                                vec![0.0; ninputs * 2 * cd.npts],
                                vec![0.0; ninputs * cd.npts],
                                vec![0.0; ninputs * 2 * cd.npts],
                                vec![0.0; input_size.max(output_size)],
                                vec![0.0; output_size],
                                TensorLaneScratch::new(
                                    ninputs,
                                    noutputs,
                                    cd.ndofs,
                                    cd.npts,
                                    2,
                                ),
                                TensorLaneScratch::new(
                                    ninputs,
                                    noutputs,
                                    cd.ndofs,
                                    cd.npts,
                                    2,
                                ),
                            )
                        },
                        |(tail_state_coeffs, tail_state_values, tail_state_grads, tail_dir_values, tail_dir_grads, tail_dir, tail_action, state_lane, dir_lane),
                         chunk: &[usize]| {
                            let w = SIMD_CELL_WIDTH;
                            let uniform = chunk.len() == w
                                && chunk.iter().all(|&cell| {
                                    self.cell_reduced_dofs[0][cell].len() == cd.ndofs
                                });
                            if !uniform {
                                for &cell in chunk {
                                    let (field_maps, _) =
                                        self.restriction.maps_for(cell, &selection.inputs);
                                    let ndofs = field_maps[0].len();
                                    let input_cell_size = ninputs * ndofs;
                                    let output_cell_size = noutputs * ndofs;
                                    self.restriction.gather_state(
                                        cell,
                                        &selection.inputs,
                                        &input_offsets,
                                        state,
                                        &mut tail_state_coeffs[..input_cell_size],
                                    );
                                    let state_cell = interpolate_tensor_cell_coefficients(
                                        cd,
                                        2,
                                        ninputs,
                                        &tail_state_coeffs[..input_cell_size],
                                        &mut tail_state_values[..ninputs * cd.npts],
                                        &mut tail_state_grads[..ninputs * 2 * cd.npts],
                                        cell,
                                    );
                                    let ctx = self.tensor_ctx(time, cell);
                                    for (column, &column_out) in outs.iter().enumerate() {
                                        self.restriction.gather_direction_column(
                                            cell,
                                            &selection.inputs,
                                            &input_offsets,
                                            direction,
                                            column,
                                            &mut tail_dir[..input_cell_size],
                                        );
                                        let direction_cell = interpolate_tensor_cell_coefficients(
                                            cd,
                                            2,
                                            ninputs,
                                            &tail_dir[..input_cell_size],
                                            &mut tail_dir_values[..ninputs * cd.npts],
                                            &mut tail_dir_grads[..ninputs * 2 * cd.npts],
                                            cell,
                                        );
                                        apply_tensor_jacobian(
                                            kernel,
                                            &ctx,
                                            &state_cell,
                                            &direction_cell,
                                            &mut tail_action[..output_cell_size],
                                        );
                                        // SAFETY: same-color cells are row-disjoint.
                                        unsafe {
                                            self.restriction.scatter_add_column(
                                                cell,
                                                &selection.outputs,
                                                &tail_action[..output_cell_size],
                                                column_out,
                                            )
                                        };
                                    }
                                }
                                return;
                            }
                            self.restriction.gather_state_batch(
                                chunk,
                                &selection.inputs,
                                &input_offsets,
                                state,
                                &mut state_lane.packed_coeffs,
                                w,
                            );
                            interpolate_batch_2d(
                                cd,
                                ninputs,
                                &state_lane.packed_coeffs,
                                &mut state_lane.lane_values,
                                &mut state_lane.lane_grads,
                                chunk,
                                w,
                            );
                            extract_lanes(
                                &state_lane.lane_values,
                                &state_lane.lane_grads,
                                2,
                                ninputs,
                                cd.npts,
                                w,
                                w,
                                &mut state_lane.scalar_values,
                                &mut state_lane.scalar_grads,
                            );
                            let ctxs: Vec<TensorCtx> =
                                chunk.iter().map(|&c| self.tensor_ctx(time, c)).collect();
                            let states: Vec<CellState> = (0..w)
                                .map(|l| CellState {
                                    nfields: ninputs,
                                    npts: cd.npts,
                                    gdim: 2,
                                    values: &state_lane.scalar_values
                                        [l * ninputs * cd.npts..(l + 1) * ninputs * cd.npts],
                                    grads: &state_lane.scalar_grads[l * ninputs * 2 * cd.npts
                                        ..(l + 1) * ninputs * 2 * cd.npts],
                                    field_indices: &[],
                                })
                                .collect();
                            let mut t0 = [0.0f64; SIMD_CELL_WIDTH];
                            let mut t1x = [0.0f64; SIMD_CELL_WIDTH];
                            let mut t1y = [0.0f64; SIMD_CELL_WIDTH];
                            for (column, &column_out) in outs.iter().enumerate() {
                                self.restriction.gather_direction_batch(
                                    chunk,
                                    &selection.inputs,
                                    &input_offsets,
                                    direction,
                                    column,
                                    &mut dir_lane.packed_coeffs,
                                    w,
                                );
                                interpolate_batch_2d(
                                    cd,
                                    ninputs,
                                    &dir_lane.packed_coeffs,
                                    &mut dir_lane.lane_values,
                                    &mut dir_lane.lane_grads,
                                    chunk,
                                    w,
                                );
                                extract_lanes(
                                    &dir_lane.lane_values,
                                    &dir_lane.lane_grads,
                                    2,
                                    ninputs,
                                    cd.npts,
                                    w,
                                    w,
                                    &mut dir_lane.scalar_values,
                                    &mut dir_lane.scalar_grads,
                                );
                                let dirs: Vec<CellState> = (0..w)
                                    .map(|l| CellState {
                                        nfields: ninputs,
                                        npts: cd.npts,
                                        gdim: 2,
                                        values: &dir_lane.scalar_values
                                            [l * ninputs * cd.npts..(l + 1) * ninputs * cd.npts],
                                        grads: &dir_lane.scalar_grads[l * ninputs * 2 * cd.npts
                                            ..(l + 1) * ninputs * 2 * cd.npts],
                                        field_indices: &[],
                                    })
                                    .collect();
                                for eq in 0..noutputs {
                                    for q in 0..cd.npts {
                                        kernel.tensor_jacobian_action_batch(
                                            &ctxs,
                                            &states,
                                            &dirs,
                                            eq,
                                            q,
                                            &mut t0[..w],
                                            &mut t1x[..w],
                                            &mut t1y[..w],
                                        );
                                        for l in 0..w {
                                            dir_lane.flux0[(eq * cd.npts + q) * w + l] = t0[l];
                                            dir_lane.flux1x[(eq * cd.npts + q) * w + l] = t1x[l];
                                            dir_lane.flux1y[(eq * cd.npts + q) * w + l] = t1y[l];
                                        }
                                    }
                                }
                                dir_lane.packed_out.fill(0.0);
                                integrate_batch_2d(
                                    cd,
                                    noutputs,
                                    &dir_lane.flux0,
                                    &dir_lane.flux1x,
                                    &dir_lane.flux1y,
                                    &mut dir_lane.packed_out,
                                    chunk,
                                    w,
                                );
                                for (l, &cell) in chunk.iter().enumerate() {
                                    for o in 0..output_size {
                                        dir_lane.cell_local[o] = dir_lane.packed_out[o * w + l];
                                    }
                                    // SAFETY: same-color cells are row-disjoint.
                                    unsafe {
                                        self.restriction.scatter_add_column(
                                            cell,
                                            &selection.outputs,
                                            &dir_lane.cell_local,
                                            column_out,
                                        )
                                    };
                                }
                            }
                        },
                    );
            }
        }
        for column in 0..ncols {
            for row in 0..total {
                out[(row, column)] = actions[column * total + row];
            }
        }
    }
}

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync>
    crate::sem_traits::TensorResidualOps<2> for SEM2DProblem<M>
{
    fn assemble_tensor_residual<K: TensorResidualKernel<2> + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
    ) -> Vec<f64> {
        let layout = self.field_layout();
        self.assemble_tensor_residual_with_layout(time, kernel, state, &layout)
    }

    fn assemble_tensor_jacobian<K: TensorResidualKernel<2> + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
    ) -> SparseColMat<usize, f64> {
        let layout = self.field_layout();
        self.assemble_tensor_jacobian_with_layout(time, kernel, state, &layout)
    }

    fn apply_tensor_jacobian_into<K: TensorResidualKernel<2> + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        out: MatMut<'_, f64>,
    ) {
        let layout = self.field_layout();
        self.apply_tensor_jacobian_with_layout(time, kernel, state, direction, &layout, out);
    }
}
