//! Weak-form (`ResidualKernel`) residual/Jacobian paths (1D).
use crate::common::{
    push_rectangular_local_matrix_triplets, CellState, FacetCtx, StateBoundaryContributions, rayon_cell_chunk_size,
};
use crate::kernels::common::{ResidualKernel, StateBoundaryTerms};
use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};

use ndelement::{
    traits::{ElementFamily, FiniteElement},
    types::ReferenceCellType,
};
use ndfunctionspace::{traits::FunctionSpace, FunctionSpaceImpl};
use ndmesh::traits::{Entity, Geometry, Mesh, Point, Topology};
use rayon::prelude::*;
use rlst::{rlst_dynamic_array, DynArray};


use super::problem::SEM1DProblem;
use crate::sem_traits::WeakResidualOps;

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> SEM1DProblem<M> {
    /// Shared endpoint loop for state-dependent natural boundaries.
    ///
    /// Mathematics: at each boundary point evaluates the `StateBoundaryTerms`
    /// kernel on the trace state and scatters a 1-DOF residual/Jacobian block.
    /// `include_residual` / `include_jacobian` select which outputs are built
    /// so callers pay for one pass only. Applicable to weak-form boundary
    /// fluxes (Robin/outflow); tensor boundaries live in [`tensor`](super::tensor).
    pub(crate) fn assemble_state_boundary_impl(
        &self,
        time: f64,
        state: MatRef<'_, f64>,
        terms: &StateBoundaryTerms,
        include_residual: bool,
        include_jacobian: bool,
    ) -> (Option<Vec<f64>>, Option<SparseColMat<usize, f64>>) {
        let layout = self.field_layout();
        let nfields = self.fields.len();
        assert_eq!(state.nrows(), layout.total_size, "state size mismatch");
        assert_eq!(state.ncols(), 1, "boundary state requires one column");
        let space = FunctionSpaceImpl::new(&self.mesh, &self.family);
        let element = self.family.element(ReferenceCellType::Interval);
        let mut residual = include_residual.then(|| vec![0.0; layout.total_size]);
        let mut triplets = Vec::new();
        let mut coord = [0.0; 1];

        for point in self.mesh.entity_iter(ReferenceCellType::Point) {
            let point_index = point.local_index();
            let Some(kernel) = terms.kernel_for(point_index) else {
                continue;
            };
            let topology = point.topology();
            let mut cells = topology.connected_entity_iter(ReferenceCellType::Interval);
            let Some(cell_index) = cells.next() else {
                continue;
            };
            if cells.next().is_some() {
                continue;
            }
            point.geometry().points().next().unwrap().coords(&mut coord);
            let cell = self
                .mesh
                .entity(ReferenceCellType::Interval, cell_index)
                .unwrap();
            let local_point = cell
                .topology()
                .sub_entity_iter(ReferenceCellType::Point)
                .position(|index| index == point_index)
                .expect("boundary point missing from owning interval");
            let normal = if local_point == 0 { -1.0 } else { 1.0 };
            assert_eq!(
                kernel.nfields(),
                nfields,
                "state boundary field count mismatch"
            );
            if let Some(names) = kernel.field_names() {
                assert_eq!(
                    names,
                    self.fields.names(),
                    "state boundary field names/order mismatch"
                );
            }
            let mut reference_point = rlst_dynamic_array!(f64, [1, 1]);
            *reference_point.get_mut([0, 0]).unwrap() = local_point as f64;
            let mut table = DynArray::<f64, 4>::from_shape(element.tabulate_array_shape(1, 1));
            element.tabulate(&reference_point, 1, &mut table);
            let cell_dofs = space
                .entity_closure_dofs(ReferenceCellType::Interval, cell_index)
                .unwrap();
            let point_dofs = space
                .entity_closure_dofs(ReferenceCellType::Point, point_index)
                .unwrap();
            assert_eq!(point_dofs.len(), 1, "a scalar boundary point has one DOF");
            let cell_dof = cell_dofs
                .iter()
                .position(|&dof| dof == point_dofs[0])
                .expect("point DOF missing from owning interval");
            let values = [*table.get([0, 0, cell_dof, 0]).unwrap()];
            let grad = [*table.get([1, 0, cell_dof, 0]).unwrap()
                * self.cell_data.jinv_cache[cell_index * self.cell_data.npts]];
            let points = [coord[0]];
            let normals = [normal];
            let ctx = FacetCtx {
                time,
                facet: self.metadata.facet(point_index),
                tdim: 0,
                gdim: 1,
                ncomp: 1,
                npts: 1,
                ndofs: 1,
                wts: &[1.0],
                jfacet_det: &[1.0],
                points: &points,
                normal: &normals,
                values: &values,
                grads: &grad,
            };
            let (field_maps, field_prescribed) = self.cell_field_maps(cell_index, nfields);
            let mut state_values = vec![0.0; nfields];
            let mut state_grads = vec![0.0; nfields];
            for field in 0..nfields {
                state_values[field] = field_maps[field][cell_dof].map_or(
                    field_prescribed[field][cell_dof].unwrap_or(0.0),
                    |reduced| state[(layout.offsets[field] + reduced, 0)],
                ) * values[0];
                for (local_i, _) in cell_dofs.iter().enumerate() {
                    let coefficient = field_maps[field][local_i]
                        .map_or(field_prescribed[field][local_i].unwrap_or(0.0), |reduced| {
                            state[(layout.offsets[field] + reduced, 0)]
                        });
                    state_grads[field] += coefficient
                        * *table.get([1, 0, local_i, 0]).unwrap()
                        * self.cell_data.jinv_cache[cell_index * self.cell_data.npts];
                }
            }
            let facet_state = CellState {
                nfields,
                npts: 1,
                gdim: 1,
                values: &state_values,
                grads: &state_grads,
                field_indices: &[],
            };
            let mut local_residual = include_residual.then(|| vec![0.0; nfields]);
            let mut local_jacobian = include_jacobian.then(|| vec![0.0; nfields * nfields]);
            if let Some(local_residual) = local_residual.as_mut() {
                kernel.assemble_local_residual(&ctx, &facet_state, local_residual);
            }
            if let Some(local_jacobian) = local_jacobian.as_mut() {
                kernel.assemble_local_jacobian(&ctx, &facet_state, local_jacobian);
            }
            for equation in 0..nfields {
                let Some(reduced) = field_maps[equation][cell_dof] else {
                    continue;
                };
                if let Some(residual) = residual.as_mut() {
                    residual[layout.offsets[equation] + reduced] +=
                        local_residual.as_ref().unwrap()[equation];
                }
                if let Some(local_jacobian) = local_jacobian.as_ref() {
                    for unknown in 0..nfields {
                        let Some(reduced_unknown) = field_maps[unknown][cell_dof] else {
                            continue;
                        };
                        let value = local_jacobian[equation * nfields + unknown];
                        if value != 0.0 {
                            triplets.push(Triplet::new(
                                layout.offsets[equation] + reduced,
                                layout.offsets[unknown] + reduced_unknown,
                                value,
                            ));
                        }
                    }
                }
            }
        }
        let jacobian = include_jacobian.then(|| {
            SparseColMat::try_new_from_triplets(layout.total_size, layout.total_size, &triplets)
                .unwrap()
        });
        (residual, jacobian)
    }

    /// Assemble state-dependent natural-boundary residual and Jacobian.
    ///
    /// Inputs: `time`, single-column `state`, boundary `terms` selecting a
    /// kernel per endpoint. Output: [`StateBoundaryContributions`] with the
    /// global residual and its sparse Jacobian. Use together with
    /// [`WeakResidualOps`] volume terms to
    /// form a complete residual (see `assemble_complete_*` and the residual
    /// operators in [`operators`](super::operators)).
    pub fn assemble_state_boundary(
        &self,
        time: f64,
        state: MatRef<'_, f64>,
        terms: &StateBoundaryTerms,
    ) -> StateBoundaryContributions {
        let (residual, jacobian) =
            self.assemble_state_boundary_impl(time, state, terms, true, true);
        StateBoundaryContributions {
            residual: residual.unwrap(),
            jacobian: jacobian.unwrap(),
        }
    }

    /// Boundary-only residual used by [`assemble_complete_residual`](Self::assemble_complete_residual).
    ///
    /// Same inputs as [`assemble_state_boundary`](Self::assemble_state_boundary);
    /// skips Jacobian formation.
    pub(crate) fn assemble_state_boundary_residual(
        &self,
        time: f64,
        state: MatRef<f64>,
        terms: &StateBoundaryTerms,
    ) -> Vec<f64> {
        self.assemble_state_boundary_impl(time, state, terms, true, false)
            .0
            .unwrap()
    }

    /// Boundary-only Jacobian used by [`assemble_complete_jacobian`](Self::assemble_complete_jacobian).
    pub(crate) fn assemble_state_boundary_jacobian(
        &self,
        time: f64,
        state: MatRef<'_, f64>,
        terms: &StateBoundaryTerms,
    ) -> SparseColMat<usize, f64> {
        self.assemble_state_boundary_impl(time, state, terms, false, true)
            .1
            .unwrap()
    }

    /// Complete weak residual: volume plus state-dependent boundary terms.
    ///
    /// Mathematics: `R(u) = R_vol(u) + R_bdry(u)`. Used by
    /// `SEM1DResidualOperator` when `with_state_boundary` terms are present;
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

    /// Matrix-free action of the state-dependent boundary Jacobian.
    ///
    /// Inputs: single-column `state` plus `direction` (`N×ncols`); output is
    /// the `N×ncols` action. Matrix-free counterpart of the boundary Jacobian
    /// assembly, added to the volume action by `apply_complete_jacobian`.
    pub fn apply_state_boundary_jacobian(
        &self,
        time: f64,
        state: MatRef<'_, f64>,
        direction: MatRef<'_, f64>,
        terms: &StateBoundaryTerms,
    ) -> Mat<f64> {
        let layout = self.field_layout();
        let nfields = self.fields.len();
        assert_eq!(state.nrows(), layout.total_size, "state size mismatch");
        assert_eq!(state.ncols(), 1, "boundary state requires one column");
        assert_eq!(
            direction.nrows(),
            layout.total_size,
            "boundary direction size mismatch"
        );
        let space = FunctionSpaceImpl::new(&self.mesh, &self.family);
        let element = self.family.element(ReferenceCellType::Interval);
        let mut out = Mat::<f64>::zeros(layout.total_size, direction.ncols());
        let mut coord = [0.0; 1];
        for point in self.mesh.entity_iter(ReferenceCellType::Point) {
            let point_index = point.local_index();
            let Some(kernel) = terms.kernel_for(point_index) else {
                continue;
            };
            let topology = point.topology();
            let mut cells = topology.connected_entity_iter(ReferenceCellType::Interval);
            let Some(cell_index) = cells.next() else {
                continue;
            };
            if cells.next().is_some() {
                continue;
            }
            point.geometry().points().next().unwrap().coords(&mut coord);
            let cell = self
                .mesh
                .entity(ReferenceCellType::Interval, cell_index)
                .unwrap();
            let local_point = cell
                .topology()
                .sub_entity_iter(ReferenceCellType::Point)
                .position(|index| index == point_index)
                .expect("boundary point missing from owning interval");
            let normal = if local_point == 0 { -1.0 } else { 1.0 };
            let mut reference_point = rlst_dynamic_array!(f64, [1, 1]);
            *reference_point.get_mut([0, 0]).unwrap() = local_point as f64;
            let mut table = DynArray::<f64, 4>::from_shape(element.tabulate_array_shape(1, 1));
            element.tabulate(&reference_point, 1, &mut table);
            let cell_dofs = space
                .entity_closure_dofs(ReferenceCellType::Interval, cell_index)
                .unwrap();
            let point_dofs = space
                .entity_closure_dofs(ReferenceCellType::Point, point_index)
                .unwrap();
            let cell_dof = cell_dofs
                .iter()
                .position(|&dof| dof == point_dofs[0])
                .unwrap();
            let values = [*table.get([0, 0, cell_dof, 0]).unwrap()];
            let grad = [*table.get([1, 0, cell_dof, 0]).unwrap()
                * self.cell_data.jinv_cache[cell_index * self.cell_data.npts]];
            let points = [coord[0]];
            let normals = [normal];
            let ctx = FacetCtx {
                time,
                facet: self.metadata.facet(point_index),
                tdim: 0,
                gdim: 1,
                ncomp: 1,
                npts: 1,
                ndofs: 1,
                wts: &[1.0],
                jfacet_det: &[1.0],
                points: &points,
                normal: &normals,
                values: &values,
                grads: &grad,
            };
            let (field_maps, field_prescribed) = self.cell_field_maps(cell_index, nfields);
            let mut state_values = vec![0.0; nfields];
            let mut state_grads = vec![0.0; nfields];
            for field in 0..nfields {
                state_values[field] = field_maps[field][cell_dof].map_or(
                    field_prescribed[field][cell_dof].unwrap_or(0.0),
                    |reduced| state[(layout.offsets[field] + reduced, 0)],
                ) * values[0];
                for (local_i, _) in cell_dofs.iter().enumerate() {
                    let coefficient = field_maps[field][local_i]
                        .map_or(field_prescribed[field][local_i].unwrap_or(0.0), |reduced| {
                            state[(layout.offsets[field] + reduced, 0)]
                        });
                    state_grads[field] += coefficient
                        * *table.get([1, 0, local_i, 0]).unwrap()
                        * self.cell_data.jinv_cache[cell_index * self.cell_data.npts];
                }
            }
            let facet_state = CellState {
                nfields,
                npts: 1,
                gdim: 1,
                values: &state_values,
                grads: &state_grads,
                field_indices: &[],
            };
            let mut local_direction = vec![0.0; nfields];
            let mut local_action = vec![0.0; nfields];
            for column in 0..direction.ncols() {
                for field in 0..nfields {
                    local_direction[field] = field_maps[field][cell_dof].map_or(0.0, |reduced| {
                        direction[(layout.offsets[field] + reduced, column)]
                    });
                }
                kernel.apply_local_jacobian(
                    &ctx,
                    &facet_state,
                    &local_direction,
                    &mut local_action,
                );
                for field in 0..nfields {
                    if let Some(reduced) = field_maps[field][cell_dof] {
                        out[(layout.offsets[field] + reduced, column)] += local_action[field];
                    }
                }
            }
        }
        out
    }

    /// Complete weak Jacobian: `dR_vol/du + dR_bdry/du` as a sparse matrix.
    ///
    /// Used by `SEM1DResidualOperator` with boundary terms; prefer the
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
        self.apply_jacobian(time, kernel, state, direction)
            + self.apply_state_boundary_jacobian(time, state, direction, terms)
    }

}

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync>
    crate::sem_traits::WeakResidualOps for SEM1DProblem<M>
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
                        vec![0.0; cd.ndofs * cd.npts],
                        vec![0.0; ninputs * cd.npts],
                        vec![0.0; ninputs * cd.npts],
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
                            &mut basis_grads[..ndofs * cd.npts],
                        );
                        let state_cell = self.prepare_cell_state(
                            ninputs,
                            &field_maps,
                            &field_prescribed,
                            &input_offsets,
                            state,
                            &basis_grads[..ndofs * cd.npts],
                            field_values,
                            field_grads,
                        );
                        let ctx =
                            self.cell_ctx(time, cell_index, ndofs, &basis_grads[..ndofs * cd.npts]);
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
        let cd = &self.cell_data;
        let row_size = noutputs * cd.ndofs;
        let col_size = ninputs * cd.ndofs;
        let batches: Vec<Vec<Triplet<usize, usize, f64>>> = self.cell_reduced_dofs[0]
            .par_chunks(chunk)
            .enumerate()
            .map_init(
                || {
                    (
                        vec![0.0; cd.ndofs * cd.npts],
                        vec![0.0; ninputs * cd.npts],
                        vec![0.0; ninputs * cd.npts],
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
                            self.restriction.maps_for(cell_index, &selection.inputs);
                        self.populate_cell_grads(
                            cell_index,
                            ndofs,
                            &mut basis_grads[..ndofs * cd.npts],
                        );
                        let state_cell = self.prepare_cell_state(
                            ninputs,
                            &field_maps,
                            &field_prescribed,
                            &input_offsets,
                            state,
                            &basis_grads[..ndofs * cd.npts],
                            field_values,
                            field_grads,
                        );
                        let ctx =
                            self.cell_ctx(time, cell_index, ndofs, &basis_grads[..ndofs * cd.npts]);
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
        let cd = &self.cell_data;
        let ncells = self.cell_reduced_dofs[0].len();
        let input_size = ninputs * cd.ndofs;
        let output_size = noutputs * cd.ndofs;
        let ncols = direction.ncols();
        let action_stride = output_size * ncols;
        // ponytail: zero-column actions reduce to nothing; avoids 0-stride chunks.
        if ncols == 0 {
            return;
        }
        // ponytail: preallocated cell-major actions filled in place (no Vec<Vec>).
        let mut actions = vec![0.0; ncells * action_stride];
        actions
            .par_chunks_mut((action_stride * chunk).max(1))
            .zip(self.cell_reduced_dofs[0].par_chunks(chunk))
            .enumerate()
            .for_each_init(
                || {
                    (
                        vec![0.0; cd.ndofs * cd.npts],
                        vec![0.0; ninputs * cd.npts],
                        vec![0.0; ninputs * cd.npts],
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
                            &mut basis_grads[..ndofs * cd.npts],
                        );
                        let state_cell = self.prepare_cell_state(
                            ninputs,
                            &field_maps,
                            &field_prescribed,
                            &input_offsets,
                            state,
                            &basis_grads[..ndofs * cd.npts],
                            field_values,
                            field_grads,
                        );
                        let ctx =
                            self.cell_ctx(time, cell_index, ndofs, &basis_grads[..ndofs * cd.npts]);
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
