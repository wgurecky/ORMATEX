//! State-independent bilinear/linear paths (`BilinearForm`/`LinearForm`, 2D).
use crate::common::{
    add_dirichlet_rhs_correction, assemble_lumped_mass, assemble_quad_boundaries, interpolate_tensor_cell_coefficients,
    push_local_matrix_triplets, scatter_local_vector,
    BoundaryContributions, BoundaryFacet,
    FieldDofLayout, rayon_cell_chunk_size,
};
use crate::kernels::common::{
    apply_tensor_bilinear_column, BilinearForm, BoundaryIntegrator, LinearForm,
};
use faer::sparse::{SparseColMat, Triplet};

use ndelement::types::ReferenceCellType;
use ndmesh::traits::Mesh;
use rayon::prelude::*;


use super::problem::SEM2DProblem;

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> SEM2DProblem<M> {
    /// Tensor fast path for [`assemble_bilinear`](crate::sem_traits::BilinearOps).
    ///
    /// Mathematics: each unit-trial column is pushed through
    /// `apply_tensor_bilinear_column` and scattered as triplets. Used only
    /// when the form reports `supports_tensor_bilinear()`; otherwise the
    /// trait method uses full quadrature. Same matrix, cheaper formation.
    pub(crate) fn assemble_tensor_bilinear<K: BilinearForm + Sync>(
        &self,
        time: f64,
        kernel: &K,
        layout: &FieldDofLayout,
    ) -> SparseColMat<usize, f64>
    where
        M: Sync,
    {
        let chunk = rayon_cell_chunk_size(self.cell_reduced_dofs[0].len());
        let names = kernel.field_names();
        let selection = self.fields.resolve_selection(
            kernel.nfields(),
            names.clone(),
            kernel.nfields(),
            names,
            "bilinear form",
        );
        let nfields = selection.inputs.len();
        let offsets: Vec<_> = selection
            .inputs
            .iter()
            .map(|&field| layout.offsets[field])
            .collect();
        let cd = &self.cell_data;
        let local_size = nfields * cd.ndofs;
        let batches: Vec<Vec<Triplet<usize, usize, f64>>> = self.cell_reduced_dofs[0]
            .par_chunks(chunk)
            .enumerate()
            .map_init(
                || {
                    (
                        vec![0.0; local_size],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; nfields * 2 * cd.npts],
                        vec![0.0; local_size],
                        vec![0.0; local_size * local_size],
                    )
                },
                |(trial_coefficients, trial_values, trial_grads, local_action, local_matrix),
                 (batch_index, reduced_batch)| {
                    let mut triplets =
                        Vec::with_capacity(reduced_batch.len() * local_size * local_size);
                    for (cell_offset, reduced_dofs) in reduced_batch.iter().enumerate() {
                        let ndofs = reduced_dofs.len();
                        let cell_size = nfields * ndofs;
                        let cell_index = batch_index * chunk + cell_offset;
                        let (field_maps, _) =
                            self.cell_field_maps_for(cell_index, &selection.inputs);
                        let ctx = self.tensor_ctx(time, cell_index);
                        local_matrix[..cell_size * cell_size].fill(0.0);
                        for unknown in 0..nfields {
                            for trial_dof in 0..ndofs {
                                trial_coefficients[..cell_size].fill(0.0);
                                trial_coefficients[unknown * ndofs + trial_dof] = 1.0;
                                let trial_state = interpolate_tensor_cell_coefficients(
                                    cd,
                                    2,
                                    nfields,
                                    &trial_coefficients[..cell_size],
                                    &mut trial_values[..nfields * cd.npts],
                                    &mut trial_grads[..nfields * 2 * cd.npts],
                                    cell_index,
                                );
                                apply_tensor_bilinear_column(
                                    kernel,
                                    &ctx,
                                    unknown,
                                    &trial_state,
                                    &mut local_action[..cell_size],
                                );
                                for equation in 0..nfields {
                                    for test in 0..ndofs {
                                        local_matrix[(equation * ndofs + test) * cell_size
                                            + unknown * ndofs
                                            + trial_dof] = local_action[equation * ndofs + test];
                                    }
                                }
                            }
                        }
                        push_local_matrix_triplets(
                            &mut triplets,
                            &local_matrix[..cell_size * cell_size],
                            &field_maps,
                            nfields,
                            &offsets,
                            |value| value.abs() > 1e-12,
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

    /// Tensor fast path for the Dirichlet correction (`-A_fb·u_b`).
    ///
    /// Same correction as the weak loop in
    /// [`apply_dirichlet_rhs_correction`](crate::sem_traits::BilinearOps),
    /// evaluated column-by-column through the tensor bilinear action. Used
    /// only when the form reports `supports_tensor_bilinear()`.
    pub(crate) fn apply_tensor_dirichlet_rhs_correction<K: BilinearForm>(
        &self,
        time: f64,
        kernel: &K,
        rhs: &mut [f64],
        layout: &FieldDofLayout,
    ) {
        let nfields = kernel.nfields();
        let cd = &self.cell_data;
        let local_size = nfields * cd.ndofs;
        let mut trial_coefficients = vec![0.0; local_size];
        let mut trial_values = vec![0.0; nfields * cd.npts];
        let mut trial_grads = vec![0.0; nfields * 2 * cd.npts];
        let mut local_action = vec![0.0; local_size];

        for cell_index in 0..self.cell_reduced_dofs[0].len() {
            let (field_maps, field_prescribed) = self.cell_field_maps(cell_index, nfields);
            if !field_prescribed
                .iter()
                .any(|values| values.iter().any(Option::is_some))
            {
                continue;
            }
            let ndofs = field_maps[0].len();
            let ctx = self.tensor_ctx(time, cell_index);
            for unknown in 0..nfields {
                for trial in 0..ndofs {
                    let Some(value) = field_prescribed[unknown][trial] else {
                        continue;
                    };
                    trial_coefficients[..local_size].fill(0.0);
                    trial_coefficients[unknown * ndofs + trial] = 1.0;
                    let trial_state = interpolate_tensor_cell_coefficients(
                        cd,
                        2,
                        nfields,
                        &trial_coefficients[..local_size],
                        &mut trial_values[..nfields * cd.npts],
                        &mut trial_grads[..nfields * 2 * cd.npts],
                        cell_index,
                    );
                    apply_tensor_bilinear_column(
                        kernel,
                        &ctx,
                        unknown,
                        &trial_state,
                        &mut local_action[..local_size],
                    );
                    for equation in 0..nfields {
                        for (local_test, &reduced) in field_maps[equation].iter().enumerate() {
                            if let Some(reduced) = reduced {
                                rhs[layout.offsets[equation] + reduced] -=
                                    value * local_action[equation * ndofs + local_test];
                            }
                        }
                    }
                }
            }
        }
    }

    /// Assemble selected natural-boundary kernels on quad facets.
    ///
    /// Inputs: `time` and a `select` closure receiving each boundary
    /// [`BoundaryFacet`] (ndmesh facet index and midpoint) and returning a
    /// kernel or `None` (adiabatic facet). Output: [`BoundaryContributions`]
    /// with the RHS and sparse matrix blocks; prescribed-DOF columns fold
    /// `-A_fb·u_b` into the RHS. Use for Neumann/Robin fluxes;
    /// state-dependent fluxes live in [`weak`](super::weak).
    pub fn assemble_boundary<'a, F>(&self, time: f64, select: F) -> BoundaryContributions
    where
        F: FnMut(BoundaryFacet) -> Option<&'a dyn BoundaryIntegrator>,
    {
        assemble_quad_boundaries(
            &self.mesh,
            &self.family,
            self.p,
            &self.metadata,
            &self.fields,
            time,
            |field, full| self.target_field_dof(field, full),
            |field, full| self.prescribed_field_dof(field, full),
            |field| self.field_reduced_size(field),
            select,
        )
    }
}

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync>
    crate::sem_traits::BilinearOps for SEM2DProblem<M>
{
    fn assemble_bilinear<K: BilinearForm + Sync>(
        &self,
        time: f64,
        kernel: &K,
    ) -> SparseColMat<usize, f64>
    where
        M: Sync,
    {
        let chunk = rayon_cell_chunk_size(self.cell_reduced_dofs[0].len());
        let names = kernel.field_names();
        let selection = self.fields.resolve_selection(
            kernel.nfields(),
            names.clone(),
            kernel.nfields(),
            names,
            "bilinear form",
        );
        let nfields = selection.inputs.len();
        let layout = self.field_layout();
        let offsets: Vec<_> = selection
            .inputs
            .iter()
            .map(|&field| layout.offsets[field])
            .collect();
        if kernel.supports_tensor_bilinear() && self.cell_data.tensor.is_some() {
            return self.assemble_tensor_bilinear(time, kernel, &layout);
        }
        let gdim = self.mesh.geometry_dim();
        let cd = &self.cell_data;
        let max_ndofs = cd.ndofs;

        let local_size = nfields * max_ndofs;
        let npts = cd.npts;
        let batches: Vec<Vec<Triplet<usize, usize, f64>>> = self.cell_reduced_dofs[0]
            .par_chunks(chunk)
            .enumerate()
            .map_init(
                || {
                    (
                        vec![0.0_f64; max_ndofs * gdim * npts],
                        vec![0.0_f64; local_size * local_size],
                    )
                },
                |(grads_buf, local_mat), (batch_index, reduced_batch)| {
                    let mut triplets =
                        Vec::with_capacity(reduced_batch.len() * local_size * local_size);
                    for (cell_offset, reduced_dofs) in reduced_batch.iter().enumerate() {
                        let ndofs = reduced_dofs.len();
                        debug_assert!(ndofs == cd.ndofs, "ndofs mismatch: cell vs CellData");
                        let cell_size = nfields * ndofs;
                        let cell_index = batch_index * chunk + cell_offset;
                        let (field_maps, _) =
                            self.cell_field_maps_for(cell_index, &selection.inputs);
                        let ctx = self.prepare_cell_ctx(
                            time,
                            cell_index,
                            ndofs,
                            &mut grads_buf[..ndofs * gdim * npts],
                        );
                        let mat_slice = &mut local_mat[..cell_size * cell_size];
                        mat_slice.fill(0.0);
                        kernel.assemble_local(&ctx, mat_slice);

                        push_local_matrix_triplets(
                            &mut triplets,
                            mat_slice,
                            &field_maps,
                            nfields,
                            &offsets,
                            |value| value.abs() > 1e-12,
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

    fn assemble_linear<K: LinearForm>(&self, time: f64, kernel: &K) -> Vec<f64> {
        let selection = self.fields.resolve_selection(
            kernel.nfields(),
            kernel.field_names(),
            kernel.nfields(),
            kernel.field_names(),
            "linear form",
        );
        let nfields = selection.outputs.len();
        let layout = self.field_layout();
        let offsets: Vec<_> = selection
            .outputs
            .iter()
            .map(|&field| layout.offsets[field])
            .collect();
        let gdim = self.mesh.geometry_dim();
        let cd = &self.cell_data;
        let mut grads_buf = vec![0.0_f64; cd.ndofs * gdim * cd.npts];
        let mut local_rhs = vec![0.0_f64; nfields * cd.ndofs];
        let mut rhs = vec![0.0_f64; layout.total_size];

        for c in 0..self.cell_reduced_dofs[0].len() {
            let reduced_dofs = &self.cell_reduced_dofs[0][c];
            let ndofs = reduced_dofs.len();
            let (field_maps, _) = self.cell_field_maps_for(c, &selection.outputs);
            let npts = cd.npts;
            let ctx = self.prepare_cell_ctx(time, c, ndofs, &mut grads_buf[..ndofs * gdim * npts]);
            kernel.assemble_local_rhs(&ctx, &mut local_rhs[..nfields * ndofs]);
            scatter_local_vector(
                &mut rhs,
                &local_rhs[..nfields * ndofs],
                &field_maps,
                nfields,
                &offsets,
            );
        }
        rhs
    }

    /// Assemble a linear RHS and apply the nonzero Dirichlet correction.
    fn assemble_linear_with_dirichlet<B, L>(
        &self,
        time: f64,
        bilinear: &B,
        linear: &L,
    ) -> Vec<f64>
    where
        B: BilinearForm,
        L: LinearForm,
    {
        assert_eq!(
            bilinear.nfields(),
            linear.nfields(),
            "bilinear and linear field counts must match"
        );
        let mut rhs = self.assemble_linear(time, linear);
        self.apply_dirichlet_rhs_correction(time, bilinear, &mut rhs);
        rhs
    }
    /// Apply the prescribed-DOF contribution `-A_fb u_b` to a reduced RHS.
    ///
    /// Call this after assembling a source RHS and before solving a linear
    /// problem with nonzero Dirichlet values. Homogeneous Dirichlet values and
    /// problems without Dirichlet reduction are no-ops.
    fn apply_dirichlet_rhs_correction<K: BilinearForm>(
        &self,
        time: f64,
        kernel: &K,
        rhs: &mut [f64],
    ) {
        let nfields = kernel.nfields();
        assert!(nfields > 0, "bilinear form must contain at least one field");
        self.validate_fields(nfields, kernel.field_names(), "bilinear form");
        let layout = self.field_layout();
        assert_eq!(rhs.len(), layout.total_size, "RHS size mismatch");
        if !self
            .cell_prescribed_values
            .iter()
            .flatten()
            .flatten()
            .any(Option::is_some)
        {
            return;
        }
        if kernel.supports_tensor_bilinear() && self.cell_data.tensor.is_some() {
            self.apply_tensor_dirichlet_rhs_correction(time, kernel, rhs, &layout);
            return;
        }
        let cd = &self.cell_data;
        let gdim = self.mesh.geometry_dim();
        let mut grads = vec![0.0; cd.ndofs * gdim * cd.npts];
        let local_size = nfields * cd.ndofs;
        let mut local = vec![0.0; local_size * local_size];
        for cell_index in 0..self.cell_reduced_dofs[0].len() {
            let reduced_dofs = &self.cell_reduced_dofs[0][cell_index];
            let (field_maps, field_prescribed) = self.cell_field_maps(cell_index, nfields);
            if !field_prescribed
                .iter()
                .any(|values| values.iter().any(Option::is_some))
            {
                continue;
            }
            let ndofs = reduced_dofs.len();
            let ctx = self.prepare_cell_ctx(
                time,
                cell_index,
                ndofs,
                &mut grads[..ndofs * gdim * cd.npts],
            );
            local.fill(0.0);
            kernel.assemble_local(&ctx, &mut local[..local_size * local_size]);
            add_dirichlet_rhs_correction(
                rhs,
                &local[..local_size * local_size],
                &field_maps,
                &field_prescribed,
                nfields,
                &layout.offsets,
            );
        }
    }

    fn assemble_lumped_mass(&self) -> SparseColMat<usize, f64> {
        let nfields = self.fields.len();
        let layout = self.field_layout();
        assemble_lumped_mass(
            &self.cell_data,
            &self.cell_reduced_dofs,
            &layout.offsets,
            nfields,
        )
    }

}
