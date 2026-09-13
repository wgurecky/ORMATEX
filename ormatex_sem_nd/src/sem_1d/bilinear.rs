//! State-independent bilinear/linear paths (`BilinearForm`/`LinearForm`, 1D).
use crate::common::{
    add_dirichlet_rhs_correction, assemble_lumped_mass, interpolate_tensor_cell_coefficients,
    push_local_matrix_triplets, rayon_cell_chunk_size, scatter_local_vector, BoundaryContributions,
    FacetCtx, FieldDofLayout,
};
use crate::kernels::common::{
    apply_tensor_bilinear_column_1d, BilinearForm, BoundaryIntegrator, LinearForm,
};
use faer::sparse::{SparseColMat, Triplet};

use ndelement::{
    traits::{ElementFamily, FiniteElement},
    types::ReferenceCellType,
};
use ndfunctionspace::{traits::FunctionSpace, FunctionSpaceImpl};
use ndmesh::traits::{Entity, Geometry, Mesh, Point, Topology};
use rayon::prelude::*;
use rlst::{rlst_dynamic_array, DynArray};

use super::problem::BoundaryPoint;
use super::problem::SEM1DProblem;

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> SEM1DProblem<M> {
    /// Tensor fast path for [`assemble_bilinear`](crate::sem_traits::BilinearOps).
    ///
    /// Mathematics: each unit-trial column is pushed through
    /// `apply_tensor_bilinear_column_1d` and scattered as triplets. Used only
    /// when the form reports `supports_tensor_bilinear_1d()`; otherwise the
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
                        vec![0.0; nfields * cd.npts],
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
                                    1,
                                    nfields,
                                    &trial_coefficients[..cell_size],
                                    &mut trial_values[..nfields * cd.npts],
                                    &mut trial_grads[..nfields * cd.npts],
                                    cell_index,
                                );
                                apply_tensor_bilinear_column_1d(
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
    /// only when the form reports `supports_tensor_bilinear_1d()`.
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
        let mut trial_grads = vec![0.0; nfields * cd.npts];
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
                        1,
                        nfields,
                        &trial_coefficients[..local_size],
                        &mut trial_values[..nfields * cd.npts],
                        &mut trial_grads[..nfields * cd.npts],
                        cell_index,
                    );
                    apply_tensor_bilinear_column_1d(
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

    /// Assemble selected natural-boundary kernels at 1D endpoints.
    ///
    /// Inputs: `time` and a `select` closure receiving each boundary
    /// [`BoundaryPoint`] (index, coordinate, outward normal, physical region)
    /// and returning a kernel or `None` (adiabatic point). Output:
    /// [`BoundaryContributions`] with the RHS and sparse matrix blocks;
    /// prescribed-DOF columns fold `-A_fb·u_b` into the RHS. Use for
    /// Neumann/Robin fluxes; state-dependent fluxes live in
    /// [`weak`](super::weak).
    pub fn assemble_boundary<'a, F>(&self, time: f64, mut select: F) -> BoundaryContributions
    where
        F: FnMut(BoundaryPoint) -> Option<&'a dyn BoundaryIntegrator>,
    {
        let space = FunctionSpaceImpl::new(&self.mesh, &self.family);
        let element = self.family.element(ReferenceCellType::Interval);
        let mut rhs = Vec::new();
        let nfields = self.fields.len();
        let mut selected = false;
        let mut triplets = Vec::new();
        let mut coord = [0.0; 1];

        for point in self.mesh.entity_iter(ReferenceCellType::Point) {
            let point_index = point.local_index();
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
            let Some(kernel) = select(BoundaryPoint {
                index: point_index,
                coordinate: coord[0],
                normal,
                physical_region: self.metadata.facet(point_index).physical_region,
            }) else {
                continue;
            };
            if !selected {
                self.validate_fields(nfields, kernel.field_names(), "boundary integrator");
                selected = true;
                let layout = self.field_layout();
                rhs.resize(layout.total_size, 0.0);
            } else {
                self.validate_fields(nfields, kernel.field_names(), "boundary integrator");
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
            let points = [coord[0]];
            let normal = [normal];
            let grads = [*table.get([1, 0, cell_dof, 0]).unwrap()
                * self.cell_data.jinv_cache[cell_index * self.cell_data.npts]];
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
                normal: &normal,
                values: &values,
                grads: &grads,
            };
            let mut local_rhs = vec![0.0; nfields];
            let mut local_mat = vec![0.0; nfields * nfields];
            kernel.assemble_facet_rhs(&ctx, &mut local_rhs);
            kernel.assemble_facet_mat(&ctx, &mut local_mat);
            let layout = self.field_layout();
            for equation in 0..nfields {
                let Some(reduced) = self.target_field_dof(equation, point_dofs[0]) else {
                    continue;
                };
                for unknown in 0..nfields {
                    if let Some(value) = self.prescribed_field_dof(unknown, point_dofs[0]) {
                        rhs[layout.offsets[equation] + reduced] -=
                            local_mat[equation * nfields + unknown] * value;
                    }
                    let Some(reduced_unknown) = self.target_field_dof(unknown, point_dofs[0])
                    else {
                        continue;
                    };
                    let value = local_mat[equation * nfields + unknown];
                    if value != 0.0 {
                        triplets.push(Triplet::new(
                            layout.offsets[equation] + reduced,
                            layout.offsets[unknown] + reduced_unknown,
                            value,
                        ));
                    }
                }
                rhs[layout.offsets[equation] + reduced] += local_rhs[equation];
            }
        }
        if rhs.is_empty() {
            rhs.resize(self.system_size(), 0.0);
        }
        let system_size = self.system_size();
        BoundaryContributions {
            rhs,
            mat: SparseColMat::try_new_from_triplets(system_size, system_size, &triplets).unwrap(),
        }
    }
}

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync> crate::sem_traits::BilinearOps
    for SEM1DProblem<M>
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
        if kernel.supports_tensor_bilinear_1d() && self.cell_data.tensor.is_some() {
            return self.assemble_tensor_bilinear(time, kernel, &layout);
        }
        let cd = &self.cell_data;
        let local_size = nfields * cd.ndofs;
        let batches: Vec<Vec<Triplet<usize, usize, f64>>> = self.cell_reduced_dofs[0]
            .par_chunks(chunk)
            .enumerate()
            .map_init(
                || {
                    (
                        vec![0.0; cd.ndofs * cd.npts],
                        vec![0.0; local_size * local_size],
                    )
                },
                |(grads, local), (batch_index, reduced_batch)| {
                    let mut triplets =
                        Vec::with_capacity(reduced_batch.len() * local_size * local_size);
                    for (cell_offset, reduced_dofs) in reduced_batch.iter().enumerate() {
                        let ndofs = reduced_dofs.len();
                        let cell_size = nfields * ndofs;
                        let cell_index = batch_index * chunk + cell_offset;
                        let (field_maps, _) =
                            self.cell_field_maps_for(cell_index, &selection.inputs);
                        let ctx = self.prepare_cell_ctx(
                            time,
                            cell_index,
                            ndofs,
                            &mut grads[..ndofs * cd.npts],
                        );
                        local[..cell_size * cell_size].fill(0.0);
                        kernel.assemble_local(&ctx, &mut local[..cell_size * cell_size]);
                        push_local_matrix_triplets(
                            &mut triplets,
                            &local[..cell_size * cell_size],
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
        let cd = &self.cell_data;
        let mut grads = vec![0.0; cd.ndofs * cd.npts];
        let mut local = vec![0.0; nfields * cd.ndofs];
        let mut rhs = vec![0.0; layout.total_size];

        for c in 0..self.cell_reduced_dofs[0].len() {
            let reduced_dofs = &self.cell_reduced_dofs[0][c];
            let ndofs = reduced_dofs.len();
            let (field_maps, _) = self.cell_field_maps_for(c, &selection.outputs);
            let ctx = self.prepare_cell_ctx(time, c, ndofs, &mut grads[..ndofs * cd.npts]);
            kernel.assemble_local_rhs(&ctx, &mut local[..nfields * ndofs]);
            scatter_local_vector(
                &mut rhs,
                &local[..nfields * ndofs],
                &field_maps,
                nfields,
                &offsets,
            );
        }
        rhs
    }

    /// Assemble a linear RHS and apply the nonzero Dirichlet correction.
    fn assemble_linear_with_dirichlet<B, L>(&self, time: f64, bilinear: &B, linear: &L) -> Vec<f64>
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
        if kernel.supports_tensor_bilinear_1d() && self.cell_data.tensor.is_some() {
            self.apply_tensor_dirichlet_rhs_correction(time, kernel, rhs, &layout);
            return;
        }
        let cd = &self.cell_data;
        let mut grads = vec![0.0; cd.ndofs * cd.npts];
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
            let ctx = self.prepare_cell_ctx(time, cell_index, ndofs, &mut grads[..ndofs * cd.npts]);
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
