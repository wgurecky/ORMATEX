//! Weak-form (`ResidualKernel`) residual/Jacobian paths (2D).
use crate::common::{
    apply_quad_state_boundary_terms_cached,
    assemble_quad_state_boundary_jacobian, assemble_quad_state_boundary_residual_cached, push_rectangular_local_matrix_triplets, StateBoundaryContributions, rayon_cell_chunk_size,
};
use crate::kernels::common::{
    ResidualKernel, StateBoundaryTerms,
};
use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};

use ndelement::types::ReferenceCellType;
use ndmesh::traits::Mesh;
use rayon::prelude::*;


use super::problem::SEM2DProblem;
use crate::sem_traits::WeakResidualOps;

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> SEM2DProblem<M> {
    /// Boundary-only residual via the cached quad state-boundary assembly.
    ///
    /// Same inputs as [`assemble_state_boundary`](Self::assemble_state_boundary);
    /// skips Jacobian formation. Added to the volume residual by
    /// [`assemble_complete_residual`](Self::assemble_complete_residual).
    pub(crate) fn assemble_state_boundary_residual(
        &self,
        time: f64,
        state: MatRef<f64>,
        terms: &StateBoundaryTerms,
    ) -> Vec<f64> {
        assemble_quad_state_boundary_residual_cached(
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

    /// Complete weak residual: volume plus state-dependent boundary terms.
    ///
    /// Mathematics: `R(u) = R_vol(u) + R_bdry(u)`. Used by
    /// `SEM2DResidualOperator` when `with_state_boundary` terms are present;
    /// prefer the operator API over calling this directly.
    pub(crate) fn assemble_complete_residual<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        terms: &StateBoundaryTerms,
    ) -> Vec<f64>
    where
        M: Sync,
    {
        let mut residual = self.assemble_residual(time, kernel, state);
        let boundary = self.assemble_state_boundary_residual(time, state, terms);
        for (volume, boundary) in residual.iter_mut().zip(boundary) {
            *volume += boundary;
        }
        residual
    }

    /// Complete weak Jacobian: `dR_vol/du + dR_bdry/du` as a sparse matrix.
    ///
    /// Used by `SEM2DResidualOperator` with boundary terms; prefer the
    /// operator API and the matrix-free action inside solves.
    pub(crate) fn assemble_complete_jacobian<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        terms: &StateBoundaryTerms,
    ) -> SparseColMat<usize, f64>
    where
        M: Sync,
    {
        let volume = self.assemble_residual_jacobian(time, kernel, state);
        let boundary = self.assemble_state_boundary_jacobian(time, state, terms);
        volume.as_ref() + boundary.as_ref()
    }

    /// Complete weak Jacobian action: volume plus boundary contributions.
    pub(crate) fn apply_complete_jacobian<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        terms: &StateBoundaryTerms,
    ) -> Mat<f64>
    where
        M: Sync,
    {
        let mut action = self.apply_jacobian(time, kernel, state, direction);
        let layout = self.field_layout();
        action += apply_quad_state_boundary_terms_cached(
            &self.state_boundary_cache,
            &self.fields,
            time,
            state,
            direction,
            &self.cell_reduced_dofs,
            &self.cell_prescribed_values,
            &layout.offsets,
            terms,
        );
        action
    }

    /// Assemble state-dependent natural-boundary residual and Jacobian.
    ///
    /// Inputs: `time`, single-column `state`, boundary `terms`. Output:
    /// [`StateBoundaryContributions`] with the global residual and its sparse
    /// Jacobian, via the cached quad assembly. Use together with
    /// [`WeakResidualOps`] volume terms
    /// (see `assemble_complete_*` and the residual operators).
    pub fn assemble_state_boundary(
        &self,
        time: f64,
        state: MatRef<'_, f64>,
        terms: &StateBoundaryTerms,
    ) -> StateBoundaryContributions {
        let layout = self.field_layout();
        assert_eq!(state.nrows(), layout.total_size, "state size mismatch");
        assert_eq!(
            state.ncols(),
            1,
            "boundary assembly requires one state column"
        );
        StateBoundaryContributions {
            residual: assemble_quad_state_boundary_residual_cached(
                &self.state_boundary_cache,
                &self.fields,
                time,
                state,
                &self.cell_reduced_dofs,
                &self.cell_prescribed_values,
                &layout.offsets,
                terms,
            ),
            jacobian: self.assemble_state_boundary_jacobian(time, state, terms),
        }
    }

    /// Matrix-free action of the state-dependent boundary Jacobian.
    ///
    /// Inputs: single-column `state` plus `direction` (`N×ncols`); output is
    /// the `N×ncols` action via the cached quad assembly.
    pub fn apply_state_boundary_jacobian(
        &self,
        time: f64,
        state: MatRef<'_, f64>,
        direction: MatRef<'_, f64>,
        terms: &StateBoundaryTerms,
    ) -> Mat<f64> {
        let layout = self.field_layout();
        apply_quad_state_boundary_terms_cached(
            &self.state_boundary_cache,
            &self.fields,
            time,
            state,
            direction,
            &self.cell_reduced_dofs,
            &self.cell_prescribed_values,
            &layout.offsets,
            terms,
        )
    }

    /// Boundary-only Jacobian used by [`assemble_complete_jacobian`](Self::assemble_complete_jacobian).
    pub(crate) fn assemble_state_boundary_jacobian(
        &self,
        time: f64,
        state: MatRef<'_, f64>,
        terms: &StateBoundaryTerms,
    ) -> SparseColMat<usize, f64> {
        let layout = self.field_layout();
        assemble_quad_state_boundary_jacobian(
            &self.mesh,
            &self.family,
            self.p,
            &self.metadata,
            &self.fields,
            time,
            state,
            &self.cell_reduced_dofs,
            &self.cell_prescribed_values,
            &layout.offsets,
            terms,
        )
    }
}

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync>
    crate::sem_traits::WeakResidualOps for SEM2DProblem<M>
{
    fn assemble_residual<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
    ) -> Vec<f64>
    where
        M: Sync,
    {
        // ponytail: dynamic Rayon tile; SIMD width stays separate (SIMD_CELL_WIDTH).
        let chunk = rayon_cell_chunk_size(self.cell_reduced_dofs[0].len());
        let selection = self.resolve_residual_selection(kernel, "residual kernel");
        let ninputs = selection.inputs.len();
        let noutputs = selection.outputs.len();
        let layout = self.field_layout();
        let input_offsets: Vec<_> = selection
            .inputs
            .iter()
            .map(|&field| layout.offsets[field])
            .collect();
        assert_eq!(state.nrows(), layout.total_size, "state size mismatch");
        assert_eq!(
            state.ncols(),
            1,
            "residual assembly requires one state column"
        );
        let gdim = self.mesh.geometry_dim();
        let cd = &self.cell_data;
        let ncells = self.cell_reduced_dofs[0].len();
        let local_stride = noutputs * cd.ndofs;
        // ponytail: single cell-major buffer filled in place; E^T reduction is parallel.
        let mut actions = vec![0.0; ncells * local_stride];
        actions
            .par_chunks_mut((local_stride * chunk).max(1))
            .zip(self.cell_reduced_dofs[0].par_chunks(chunk))
            .enumerate()
            .for_each_init(
                || {
                    (
                        vec![0.0; cd.ndofs * gdim * cd.npts],
                        vec![0.0; ninputs * cd.npts],
                        vec![0.0; ninputs * gdim * cd.npts],
                        vec![0.0; local_stride],
                    )
                },
                |(basis_grads, field_values, field_grads, local),
                 (batch_index, (batch_actions, reduced_batch))| {
                    for (cell_offset, reduced_dofs) in reduced_batch.iter().enumerate() {
                        let ndofs = reduced_dofs.len();
                        let cell_size = noutputs * ndofs;
                        let cell_index = batch_index * chunk + cell_offset;
                        let (field_maps, field_prescribed) =
                            self.restriction.maps_for(cell_index, &selection.inputs);
                        self.populate_cell_grads(
                            cell_index,
                            ndofs,
                            &mut basis_grads[..ndofs * gdim * cd.npts],
                        );
                        let state_cell = self.prepare_cell_state(
                            ninputs,
                            &field_maps,
                            &field_prescribed,
                            &input_offsets,
                            state,
                            &basis_grads[..ndofs * gdim * cd.npts],
                            field_values,
                            field_grads,
                        );
                        let ctx = self.cell_ctx(
                            time,
                            cell_index,
                            ndofs,
                            &basis_grads[..ndofs * gdim * cd.npts],
                        );
                        kernel.assemble_local_residual(&ctx, &state_cell, &mut local[..cell_size]);
                        batch_actions[cell_offset * local_stride
                            ..cell_offset * local_stride + cell_size]
                            .copy_from_slice(&local[..cell_size]);
                    }
                },
            );
        self.restriction
            .transpose_reduce(&actions, local_stride, local_stride, 1, &selection.outputs)
    }

    fn assemble_residual_jacobian<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
    ) -> SparseColMat<usize, f64>
    where
        M: Sync,
    {
        let chunk = rayon_cell_chunk_size(self.cell_reduced_dofs[0].len());
        let selection = self.resolve_residual_selection(kernel, "residual kernel");
        let ninputs = selection.inputs.len();
        let noutputs = selection.outputs.len();
        let layout = self.field_layout();
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
        assert_eq!(state.nrows(), layout.total_size, "state size mismatch");
        assert_eq!(
            state.ncols(),
            1,
            "Jacobian assembly requires one state column"
        );
        let gdim = self.mesh.geometry_dim();
        let cd = &self.cell_data;
        let row_size = noutputs * cd.ndofs;
        let col_size = ninputs * cd.ndofs;
        let batches: Vec<Vec<Triplet<usize, usize, f64>>> = self.cell_reduced_dofs[0]
            .par_chunks(chunk)
            .enumerate()
            .map_init(
                || {
                    (
                        vec![0.0; cd.ndofs * gdim * cd.npts],
                        vec![0.0; ninputs * cd.npts],
                        vec![0.0; ninputs * gdim * cd.npts],
                        vec![0.0; row_size * col_size],
                    )
                },
                |(basis_grads, field_values, field_grads, local), (batch_index, reduced_batch)| {
                    let mut triplets =
                        Vec::with_capacity(reduced_batch.len() * row_size * col_size);
                    for (cell_offset, reduced_dofs) in reduced_batch.iter().enumerate() {
                        let ndofs = reduced_dofs.len();
                        let row_cell_size = noutputs * ndofs;
                        let col_cell_size = ninputs * ndofs;
                        let cell_index = batch_index * chunk + cell_offset;
                        let (field_maps, field_prescribed) =
                            self.cell_field_maps_for(cell_index, &selection.inputs);
                        self.populate_cell_grads(
                            cell_index,
                            ndofs,
                            &mut basis_grads[..ndofs * gdim * cd.npts],
                        );
                        let state_cell = self.prepare_cell_state(
                            ninputs,
                            &field_maps,
                            &field_prescribed,
                            &input_offsets,
                            state,
                            &basis_grads[..ndofs * gdim * cd.npts],
                            field_values,
                            field_grads,
                        );
                        let ctx = self.cell_ctx(
                            time,
                            cell_index,
                            ndofs,
                            &basis_grads[..ndofs * gdim * cd.npts],
                        );
                        local[..row_cell_size * col_cell_size].fill(0.0);
                        kernel.assemble_local_jacobian(
                            &ctx,
                            &state_cell,
                            &mut local[..row_cell_size * col_cell_size],
                        );
                        let (row_maps, _) =
                            self.restriction.maps_for(cell_index, &selection.outputs);
                        push_rectangular_local_matrix_triplets(
                            &mut triplets,
                            &local[..row_cell_size * col_cell_size],
                            &row_maps,
                            &field_maps,
                            noutputs,
                            ninputs,
                            &output_offsets,
                            &input_offsets,
                            |value| value != 0.0,
                        );
                    }
                    triplets
                },
            )
            .collect();
        let triplets: Vec<_> = batches.into_iter().flatten().collect();
        let system_size = layout.total_size;
        SparseColMat::try_new_from_triplets(system_size, system_size, &triplets).unwrap()
    }

    fn apply_jacobian<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
    ) -> Mat<f64>
    where
        M: Sync,
    {
        let mut out = Mat::<f64>::zeros(self.system_size(), direction.ncols());
        self.apply_jacobian_into(time, kernel, state, direction, out.as_mut());
        out
    }

    /// Apply `dR/du(state)` into caller-provided storage.
    fn apply_jacobian_into<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        mut out: MatMut<'_, f64>,
    ) where
        M: Sync,
    {
        let chunk = rayon_cell_chunk_size(self.cell_reduced_dofs[0].len());
        let selection = self.resolve_residual_selection(kernel, "residual kernel");
        let ninputs = selection.inputs.len();
        let noutputs = selection.outputs.len();
        let layout = self.field_layout();
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
        out.fill(0.0);
        let gdim = self.mesh.geometry_dim();
        let cd = &self.cell_data;
        let input_size = ninputs * cd.ndofs;
        let output_size = noutputs * cd.ndofs;
        let ncols = direction.ncols();
        let action_stride = output_size * ncols;
        if ncols == 0 {
            return;
        }
        let ncells = self.cell_reduced_dofs[0].len();
        // ponytail: preallocated cell-major actions filled in place (no Vec<Vec>).
        let mut actions = vec![0.0; ncells * action_stride];
        actions
            .par_chunks_mut((action_stride * chunk).max(1))
            .zip(self.cell_reduced_dofs[0].par_chunks(chunk))
            .enumerate()
            .for_each_init(
                || {
                    (
                        vec![0.0; cd.ndofs * gdim * cd.npts],
                        vec![0.0; ninputs * cd.npts],
                        vec![0.0; ninputs * gdim * cd.npts],
                        vec![0.0; input_size],
                        vec![0.0; output_size],
                    )
                },
                |(basis_grads, field_values, field_grads, local_direction, local_action),
                 (batch_index, (batch_actions, reduced_batch))| {
                    for (cell_offset, reduced_dofs) in reduced_batch.iter().enumerate() {
                        let ndofs = reduced_dofs.len();
                        let input_cell_size = ninputs * ndofs;
                        let output_cell_size = noutputs * ndofs;
                        let cell_index = batch_index * chunk + cell_offset;
                        let (field_maps, field_prescribed) =
                            self.restriction.maps_for(cell_index, &selection.inputs);
                        self.populate_cell_grads(
                            cell_index,
                            ndofs,
                            &mut basis_grads[..ndofs * gdim * cd.npts],
                        );
                        let state_cell = self.prepare_cell_state(
                            ninputs,
                            &field_maps,
                            &field_prescribed,
                            &input_offsets,
                            state,
                            &basis_grads[..ndofs * gdim * cd.npts],
                            field_values,
                            field_grads,
                        );
                        let ctx = self.cell_ctx(
                            time,
                            cell_index,
                            ndofs,
                            &basis_grads[..ndofs * gdim * cd.npts],
                        );
                        for column in 0..ncols {
                            self.restriction.gather_direction_column(
                                cell_index,
                                &selection.inputs,
                                &input_offsets,
                                direction,
                                column,
                                &mut local_direction[..input_cell_size],
                            );
                            kernel.apply_local_jacobian(
                                &ctx,
                                &state_cell,
                                &local_direction[..input_cell_size],
                                &mut local_action[..output_cell_size],
                            );
                            let start = cell_offset * action_stride + column * output_size;
                            batch_actions[start..start + output_cell_size]
                                .copy_from_slice(&local_action[..output_cell_size]);
                        }
                    }
                },
            );
        let reduced = self.restriction.transpose_reduce(
            &actions,
            action_stride,
            output_size,
            ncols,
            &selection.outputs,
        );
        for column in 0..ncols {
            for row in 0..layout.total_size {
                out[(row, column)] = reduced[row * ncols + column];
            }
        }
    }

}
