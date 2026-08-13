use crate::common::{
    add_dirichlet_rhs_correction, assemble_lumped_mass, cell_ctx, interpolate_cell_state,
    push_local_matrix_triplets, scatter_local_vector, BoundaryContributions, CellData, CellState,
    FacetCtx, LocalCtx, ReducedDofMap, CELL_BATCH_SIZE,
};
use crate::kernels::kernel_common::{BilinearForm, BoundaryIntegrator, LinearForm, ResidualKernel};
use crate::material::MeshMetadata;
use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};

use ndelement::{
    ciarlet::{LagrangeElementFamily, LagrangeVariant},
    traits::{ElementFamily, FiniteElement, MappedFiniteElement},
    types::{Continuity, ReferenceCellType},
};
use ndfunctionspace::{traits::FunctionSpace, FunctionSpaceImpl};
use ndmesh::traits::{Entity, Geometry, GeometryMap, Mesh, Point, Topology};
use quadraturerules::{single_integral_quadrature, Domain, QuadratureRule};
use rayon::prelude::*;
use rlst::{rlst_dynamic_array, DynArray};

/// DOF reduction for a 1D interval mesh. Facets are `Point` entity indices.
#[derive(Clone, Debug)]
pub enum DofReduction1D {
    None,
    /// Identify the second endpoint facet with the first endpoint facet.
    Periodic {
        facets: [usize; 2],
    },
    /// Eliminate every closure DOF on selected endpoint facets and prescribe
    /// its value. Each pair is `(facet_index, prescribed_value)`.
    Dirichlet {
        facets: Vec<(usize, f64)>,
    },
}

/// Geometry used to select a 1D natural-boundary kernel.
#[derive(Clone, Copy, Debug)]
pub struct BoundaryPoint {
    pub index: usize,
    pub coordinate: f64,
    pub normal: f64,
    pub physical_region: Option<crate::material::PhysicalRegion>,
}

fn build_dof_map_1d<F>(n: usize, reduction: DofReduction1D, boundary_dof: F) -> ReducedDofMap
where
    F: Fn(usize) -> usize,
{
    match reduction {
        DofReduction1D::None => ReducedDofMap::identity(n),
        DofReduction1D::Dirichlet { facets } => ReducedDofMap::from_dirichlet_values(
            n,
            facets
                .into_iter()
                .map(|(facet, value)| (boundary_dof(facet), value)),
        ),
        DofReduction1D::Periodic { facets } => {
            let master = boundary_dof(facets[0]);
            let slave = boundary_dof(facets[1]);
            assert_ne!(master, slave, "periodic facets must be distinct");
            ReducedDofMap::from_representatives(
                (0..n)
                    .map(|dof| if dof == slave { master } else { dof })
                    .collect(),
            )
        }
    }
}

/// 1D GLL spectral-element problem on interval meshes.
pub struct SEM1DProblem<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> {
    mesh: M,
    family: LagrangeElementFamily<f64>,
    cell_data: CellData,
    cell_reduced_dofs: Vec<Vec<Option<usize>>>,
    cell_prescribed_values: Vec<Vec<Option<f64>>>,
    dof_map: ReducedDofMap,
    dof_x: Vec<f64>,
    metadata: MeshMetadata,
}

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> SEM1DProblem<M> {
    pub fn new(mesh: M, p: usize, reduction: DofReduction1D) -> Self {
        Self::new_with_metadata(mesh, p, reduction, MeshMetadata::default())
    }

    pub fn new_with_metadata(
        mesh: M,
        p: usize,
        reduction: DofReduction1D,
        metadata: MeshMetadata,
    ) -> Self {
        assert!(p >= 1, "polynomial degree p must be >= 1");
        assert_eq!(mesh.topology_dim(), 1, "SEM1DProblem: mesh tdim must be 1");
        assert_eq!(mesh.geometry_dim(), 1, "SEM1DProblem: mesh gdim must be 1");
        assert_eq!(
            mesh.entity_types(1),
            &[ReferenceCellType::Interval],
            "SEM1DProblem supports interval meshes only"
        );
        metadata.validate(
            mesh.entity_count(ReferenceCellType::Interval),
            mesh.entity_count(ReferenceCellType::Point),
        );

        let family =
            LagrangeElementFamily::<f64>::new(p, Continuity::Standard, LagrangeVariant::GLL);
        let space = FunctionSpaceImpl::new(&mesh, &family);
        let n = space.process_size();
        let element = family.element(ReferenceCellType::Interval);
        let (qpts, wts) = single_integral_quadrature(
            QuadratureRule::GaussLobattoLegendre,
            Domain::Interval,
            element.lagrange_superdegree().saturating_sub(1),
        )
        .unwrap();
        let npts = wts.len();
        let mut pts = rlst_dynamic_array!(f64, [1, npts]);
        for q in 0..npts {
            *pts.get_mut([0, q]).unwrap() = qpts[2 * q + 1];
        }
        let mut table = DynArray::<f64, 4>::from_shape(element.tabulate_array_shape(1, npts));
        element.tabulate(&pts, 1, &mut table);

        let ncells = mesh.entity_count(ReferenceCellType::Interval);
        let gmap = mesh.geometry_map(ReferenceCellType::Interval, 1, &pts);
        let mut jac_scratch = rlst_dynamic_array!(f64, [1, 1, npts]);
        let mut jinv_scratch = rlst_dynamic_array!(f64, [1, 1, npts]);
        let mut jdet_scratch = vec![0.0; npts];
        let mut jinv_cache = rlst_dynamic_array!(f64, [1, 1, npts, ncells]);
        let mut jdets_cache = vec![0.0; ncells * npts];
        let mut physical_points_cache = vec![0.0; ncells * npts];
        let mut dof_x = vec![f64::NAN; n];
        let mut physical_pts = rlst_dynamic_array!(f64, [1, npts]);

        for cell in mesh.entity_iter(ReferenceCellType::Interval) {
            let c = cell.local_index();
            gmap.jacobians_inverses_dets(c, &mut jac_scratch, &mut jinv_scratch, &mut jdet_scratch);
            gmap.physical_points(c, &mut physical_pts);
            for q in 0..npts {
                *jinv_cache.get_mut([0, 0, q, c]).unwrap() = *jinv_scratch.get([0, 0, q]).unwrap();
                jdets_cache[c * npts + q] = jdet_scratch[q];
                physical_points_cache[c * npts + q] = *physical_pts.get([0, q]).unwrap();
            }
            let cell_dofs = space
                .entity_closure_dofs(ReferenceCellType::Interval, c)
                .unwrap();
            for (local_dof, &full_dof) in cell_dofs.iter().enumerate() {
                let q = (0..npts)
                    .find(|&q| *table.get([0, q, local_dof, 0]).unwrap() > 1.0 - 1e-12)
                    .expect("GLL basis dof has no nodal quadrature point");
                let x = *physical_pts.get([0, q]).unwrap();
                if dof_x[full_dof].is_nan() {
                    dof_x[full_dof] = x;
                } else {
                    assert!(
                        (dof_x[full_dof] - x).abs() < 1e-12,
                        "inconsistent coordinate for DOF {full_dof}"
                    );
                }
            }
        }
        drop(gmap);
        assert!(
            dof_x.iter().all(|x| !x.is_nan()),
            "every GLL DOF must have a coordinate"
        );

        let mut reference_values = vec![0.0; element.dim() * npts];
        let mut nodal_quadrature = vec![0; element.dim()];
        for dof in 0..element.dim() {
            for q in 0..npts {
                reference_values[dof * npts + q] = *table.get([0, q, dof, 0]).unwrap();
            }
            nodal_quadrature[dof] = (0..npts)
                .find(|&q| reference_values[dof * npts + q] > 1.0 - 1e-12)
                .expect("GLL basis dof has no nodal quadrature point");
        }
        let cell_data = CellData {
            wts,
            npts,
            ndofs: element.dim(),
            table,
            reference_values,
            nodal_quadrature,
            jinv_cache,
            jdets_cache,
            physical_points_cache,
        };

        let boundary_dof = |facet_index: usize| -> usize {
            let facet = mesh
                .entity(ReferenceCellType::Point, facet_index)
                .expect("boundary facet index out of range");
            let topology = facet.topology();
            let mut cells = topology.connected_entity_iter(ReferenceCellType::Interval);
            assert!(
                cells.next().is_some() && cells.next().is_none(),
                "facet {facet_index} must be a boundary point"
            );
            let dofs = space
                .entity_closure_dofs(ReferenceCellType::Point, facet_index)
                .unwrap();
            assert_eq!(dofs.len(), 1, "a scalar endpoint must have one DOF");
            dofs[0]
        };

        let dof_map = build_dof_map_1d(n, reduction, boundary_dof);
        let cell_dofs: Vec<Vec<usize>> = (0..ncells)
            .map(|cell| {
                space
                    .entity_closure_dofs(ReferenceCellType::Interval, cell)
                    .unwrap()
                    .to_vec()
            })
            .collect();
        let (cell_reduced_dofs, cell_prescribed_values) = dof_map.map_cells(&cell_dofs);

        Self {
            mesh,
            family,
            cell_data,
            cell_reduced_dofs,
            cell_prescribed_values,
            dof_map,
            dof_x,
            metadata,
        }
    }

    pub fn target_dof(&self, full: usize) -> Option<usize> {
        self.dof_map.target(full)
    }

    pub fn mesh(&self) -> &M {
        &self.mesh
    }

    pub fn family(&self) -> &LagrangeElementFamily<f64> {
        &self.family
    }

    pub fn reduced_size(&self) -> usize {
        self.dof_map.reduced_size()
    }

    pub fn dof_positions(&self) -> Vec<f64> {
        let mut out = vec![f64::NAN; self.reduced_size()];
        for (full, &x) in self.dof_x.iter().enumerate() {
            if let Some(reduced) = self.target_dof(full) {
                if out[reduced].is_nan() {
                    out[reduced] = x;
                }
            }
        }
        debug_assert!(out.iter().all(|x| !x.is_nan()));
        out
    }

    fn populate_cell_grads(&self, cell_index: usize, ndofs: usize, grads: &mut [f64]) {
        let cd = &self.cell_data;
        for dof_i in 0..ndofs {
            for q in 0..cd.npts {
                grads[dof_i * cd.npts + q] = *cd.jinv_cache.get([0, 0, q, cell_index]).unwrap()
                    * *cd.table.get([1, q, dof_i, 0]).unwrap();
            }
        }
    }

    fn cell_ctx<'a>(
        &'a self,
        time: f64,
        cell_index: usize,
        ndofs: usize,
        grads: &'a [f64],
    ) -> LocalCtx<'a> {
        cell_ctx(
            &self.cell_data,
            &self.metadata,
            1,
            1,
            time,
            cell_index,
            ndofs,
            grads,
        )
    }

    fn prepare_cell_ctx<'a>(
        &'a self,
        time: f64,
        cell_index: usize,
        ndofs: usize,
        grads: &'a mut [f64],
    ) -> LocalCtx<'a> {
        self.populate_cell_grads(cell_index, ndofs, grads);
        self.cell_ctx(time, cell_index, ndofs, grads)
    }

    fn prepare_cell_state<'a>(
        &self,
        nfields: usize,
        reduced_dofs: &[Option<usize>],
        prescribed_values: &[Option<f64>],
        state: MatRef<'_, f64>,
        basis_grads: &[f64],
        values: &'a mut [f64],
        field_grads: &'a mut [f64],
    ) -> CellState<'a> {
        interpolate_cell_state(
            &self.cell_data,
            1,
            self.reduced_size(),
            nfields,
            reduced_dofs,
            prescribed_values,
            state,
            basis_grads,
            values,
            field_grads,
        )
    }

    pub fn assemble_system_residual_at<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
    ) -> Vec<f64>
    where
        M: Sync,
    {
        let nfields = kernel.nfields();
        assert!(nfields > 0, "kernel must contain at least one field");
        assert_eq!(
            state.nrows(),
            nfields * self.reduced_size(),
            "state size mismatch"
        );
        assert_eq!(
            state.ncols(),
            1,
            "residual assembly requires one state column"
        );
        let cd = &self.cell_data;
        let local_stride = nfields * cd.ndofs;
        let batches: Vec<Vec<f64>> = self
            .cell_reduced_dofs
            .par_chunks(CELL_BATCH_SIZE)
            .enumerate()
            .map_init(
                || {
                    (
                        vec![0.0; cd.ndofs * cd.npts],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; local_stride],
                    )
                },
                |(basis_grads, field_values, field_grads, local), (batch_index, reduced_batch)| {
                    let mut batch = vec![0.0; reduced_batch.len() * local_stride];
                    for (cell_offset, reduced_dofs) in reduced_batch.iter().enumerate() {
                        let ndofs = reduced_dofs.len();
                        let cell_size = nfields * ndofs;
                        let cell_index = batch_index * CELL_BATCH_SIZE + cell_offset;
                        self.populate_cell_grads(
                            cell_index,
                            ndofs,
                            &mut basis_grads[..ndofs * cd.npts],
                        );
                        let state_cell = self.prepare_cell_state(
                            nfields,
                            reduced_dofs,
                            &self.cell_prescribed_values[cell_index],
                            state,
                            &basis_grads[..ndofs * cd.npts],
                            field_values,
                            field_grads,
                        );
                        let ctx =
                            self.cell_ctx(time, cell_index, ndofs, &basis_grads[..ndofs * cd.npts]);
                        kernel.assemble_local_residual(&ctx, &state_cell, &mut local[..cell_size]);
                        batch[cell_offset * local_stride..cell_offset * local_stride + cell_size]
                            .copy_from_slice(&local[..cell_size]);
                    }
                    batch
                },
            )
            .collect();
        let mut residual = vec![0.0; nfields * self.reduced_size()];
        for (batch_index, batch) in batches.into_iter().enumerate() {
            let cell_start = batch_index * CELL_BATCH_SIZE;
            for (cell_offset, reduced_dofs) in self.cell_reduced_dofs
                [cell_start..(cell_start + CELL_BATCH_SIZE).min(self.cell_reduced_dofs.len())]
                .iter()
                .enumerate()
            {
                let ndofs = reduced_dofs.len();
                let cell_size = nfields * ndofs;
                let local =
                    &batch[cell_offset * local_stride..cell_offset * local_stride + cell_size];
                scatter_local_vector(
                    &mut residual,
                    local,
                    reduced_dofs,
                    nfields,
                    self.reduced_size(),
                );
            }
        }
        residual
    }

    pub fn assemble_residual<K: ResidualKernel + Sync>(
        &self,
        kernel: &K,
        state: MatRef<f64>,
    ) -> Vec<f64>
    where
        M: Sync,
    {
        assert_eq!(
            kernel.nfields(),
            1,
            "scalar residual requires a one-field kernel"
        );
        self.assemble_system_residual_at(0.0, kernel, state)
    }

    pub fn assemble_system_residual<K: ResidualKernel + Sync>(
        &self,
        kernel: &K,
        state: MatRef<f64>,
    ) -> Vec<f64>
    where
        M: Sync,
    {
        self.assemble_system_residual_at(0.0, kernel, state)
    }

    pub fn assemble_system_residual_jacobian_at<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
    ) -> SparseColMat<usize, f64>
    where
        M: Sync,
    {
        let nfields = kernel.nfields();
        assert!(nfields > 0, "kernel must contain at least one field");
        let n = self.reduced_size();
        assert_eq!(state.nrows(), nfields * n, "state size mismatch");
        assert_eq!(
            state.ncols(),
            1,
            "Jacobian assembly requires one state column"
        );
        let cd = &self.cell_data;
        let local_size = nfields * cd.ndofs;
        let batches: Vec<Vec<Triplet<usize, usize, f64>>> = self
            .cell_reduced_dofs
            .par_chunks(CELL_BATCH_SIZE)
            .enumerate()
            .map_init(
                || {
                    (
                        vec![0.0; cd.ndofs * cd.npts],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; local_size * local_size],
                    )
                },
                |(basis_grads, field_values, field_grads, local), (batch_index, reduced_batch)| {
                    let mut triplets =
                        Vec::with_capacity(reduced_batch.len() * local_size * local_size);
                    for (cell_offset, reduced_dofs) in reduced_batch.iter().enumerate() {
                        let ndofs = reduced_dofs.len();
                        let cell_size = nfields * ndofs;
                        let cell_index = batch_index * CELL_BATCH_SIZE + cell_offset;
                        self.populate_cell_grads(
                            cell_index,
                            ndofs,
                            &mut basis_grads[..ndofs * cd.npts],
                        );
                        let state_cell = self.prepare_cell_state(
                            nfields,
                            reduced_dofs,
                            &self.cell_prescribed_values[cell_index],
                            state,
                            &basis_grads[..ndofs * cd.npts],
                            field_values,
                            field_grads,
                        );
                        let ctx =
                            self.cell_ctx(time, cell_index, ndofs, &basis_grads[..ndofs * cd.npts]);
                        local[..cell_size * cell_size].fill(0.0);
                        kernel.assemble_local_jacobian(
                            &ctx,
                            &state_cell,
                            &mut local[..cell_size * cell_size],
                        );
                        push_local_matrix_triplets(
                            &mut triplets,
                            &local[..cell_size * cell_size],
                            reduced_dofs,
                            nfields,
                            n,
                            |value| value != 0.0,
                        );
                    }
                    triplets
                },
            )
            .collect();
        let triplets: Vec<_> = batches.into_iter().flatten().collect();
        let system_size = nfields * n;
        SparseColMat::try_new_from_triplets(system_size, system_size, &triplets).unwrap()
    }

    pub fn assemble_residual_jacobian<K: ResidualKernel + Sync>(
        &self,
        kernel: &K,
        state: MatRef<f64>,
    ) -> SparseColMat<usize, f64>
    where
        M: Sync,
    {
        assert_eq!(
            kernel.nfields(),
            1,
            "scalar Jacobian requires a one-field kernel"
        );
        self.assemble_system_residual_jacobian_at(0.0, kernel, state)
    }

    pub fn assemble_system_residual_jacobian<K: ResidualKernel + Sync>(
        &self,
        kernel: &K,
        state: MatRef<f64>,
    ) -> SparseColMat<usize, f64>
    where
        M: Sync,
    {
        self.assemble_system_residual_jacobian_at(0.0, kernel, state)
    }

    pub fn apply_system_jacobian_matfree_at<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
    ) -> Mat<f64>
    where
        M: Sync,
    {
        let nfields = kernel.nfields();
        assert!(nfields > 0, "kernel must contain at least one field");
        let n = self.reduced_size();
        assert_eq!(state.nrows(), nfields * n, "state size mismatch");
        assert_eq!(
            state.ncols(),
            1,
            "matrix-free Jacobian requires one state column"
        );
        assert_eq!(direction.nrows(), nfields * n, "direction size mismatch");
        let cd = &self.cell_data;
        let local_size = nfields * cd.ndofs;
        let ncols = direction.ncols();
        let action_stride = local_size * ncols;
        let batches: Vec<Vec<f64>> = self
            .cell_reduced_dofs
            .par_chunks(CELL_BATCH_SIZE)
            .enumerate()
            .map_init(
                || {
                    (
                        vec![0.0; cd.ndofs * cd.npts],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; local_size],
                        vec![0.0; local_size],
                    )
                },
                |(basis_grads, field_values, field_grads, local_direction, local_action),
                 (batch_index, reduced_batch)| {
                    let mut batch = vec![0.0; reduced_batch.len() * action_stride];
                    for (cell_offset, reduced_dofs) in reduced_batch.iter().enumerate() {
                        let ndofs = reduced_dofs.len();
                        let cell_size = nfields * ndofs;
                        let cell_index = batch_index * CELL_BATCH_SIZE + cell_offset;
                        self.populate_cell_grads(
                            cell_index,
                            ndofs,
                            &mut basis_grads[..ndofs * cd.npts],
                        );
                        let state_cell = self.prepare_cell_state(
                            nfields,
                            reduced_dofs,
                            &self.cell_prescribed_values[cell_index],
                            state,
                            &basis_grads[..ndofs * cd.npts],
                            field_values,
                            field_grads,
                        );
                        let ctx =
                            self.cell_ctx(time, cell_index, ndofs, &basis_grads[..ndofs * cd.npts]);
                        for column in 0..ncols {
                            for field in 0..nfields {
                                for (local_i, &reduced_i) in reduced_dofs.iter().enumerate() {
                                    local_direction[field * ndofs + local_i] = reduced_i
                                        .map_or(0.0, |i| direction[(field * n + i, column)]);
                                }
                            }
                            kernel.apply_local_jacobian(
                                &ctx,
                                &state_cell,
                                &local_direction[..cell_size],
                                &mut local_action[..cell_size],
                            );
                            let start = cell_offset * action_stride + column * local_size;
                            batch[start..start + cell_size]
                                .copy_from_slice(&local_action[..cell_size]);
                        }
                    }
                    batch
                },
            )
            .collect();
        let mut out = Mat::<f64>::zeros(nfields * n, direction.ncols());
        for (batch_index, batch) in batches.into_iter().enumerate() {
            let cell_start = batch_index * CELL_BATCH_SIZE;
            for (cell_offset, reduced_dofs) in self.cell_reduced_dofs
                [cell_start..(cell_start + CELL_BATCH_SIZE).min(self.cell_reduced_dofs.len())]
                .iter()
                .enumerate()
            {
                let ndofs = reduced_dofs.len();
                for column in 0..ncols {
                    let start = cell_offset * action_stride + column * local_size;
                    let local = &batch[start..start + nfields * ndofs];
                    for field in 0..nfields {
                        for (local_i, &reduced_i) in reduced_dofs.iter().enumerate() {
                            if let Some(reduced_i) = reduced_i {
                                out[(field * n + reduced_i, column)] +=
                                    local[field * ndofs + local_i];
                            }
                        }
                    }
                }
            }
        }
        out
    }

    pub fn apply_jacobian_matfree<K: ResidualKernel + Sync>(
        &self,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
    ) -> Mat<f64>
    where
        M: Sync,
    {
        assert_eq!(
            kernel.nfields(),
            1,
            "scalar Jacobian requires a one-field kernel"
        );
        self.apply_system_jacobian_matfree_at(0.0, kernel, state, direction)
    }

    pub fn apply_system_jacobian_matfree<K: ResidualKernel + Sync>(
        &self,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
    ) -> Mat<f64>
    where
        M: Sync,
    {
        self.apply_system_jacobian_matfree_at(0.0, kernel, state, direction)
    }

    pub fn assemble_system_bilinear_at<K: BilinearForm + Sync>(
        &self,
        time: f64,
        kernel: &K,
    ) -> SparseColMat<usize, f64>
    where
        M: Sync,
    {
        let nfields = kernel.nfields();
        assert!(nfields > 0, "bilinear form must contain at least one field");
        let n = self.reduced_size();
        let cd = &self.cell_data;
        let local_size = nfields * cd.ndofs;
        let batches: Vec<Vec<Triplet<usize, usize, f64>>> = self
            .cell_reduced_dofs
            .par_chunks(CELL_BATCH_SIZE)
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
                        let cell_index = batch_index * CELL_BATCH_SIZE + cell_offset;
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
                            reduced_dofs,
                            nfields,
                            n,
                            |value| value.abs() > 1e-12,
                        );
                    }
                    triplets
                },
            )
            .collect();
        let triplets: Vec<_> = batches.into_iter().flatten().collect();
        let system_size = nfields * n;
        SparseColMat::try_new_from_triplets(system_size, system_size, &triplets).unwrap()
    }

    pub fn assemble_bilinear<K: BilinearForm + Sync>(&self, kernel: &K) -> SparseColMat<usize, f64>
    where
        M: Sync,
    {
        assert_eq!(
            kernel.nfields(),
            1,
            "scalar matrix requires a one-field form"
        );
        self.assemble_system_bilinear_at(0.0, kernel)
    }

    pub fn assemble_system_bilinear<K: BilinearForm + Sync>(
        &self,
        kernel: &K,
    ) -> SparseColMat<usize, f64>
    where
        M: Sync,
    {
        self.assemble_system_bilinear_at(0.0, kernel)
    }

    pub fn assemble_system_linear_at<K: LinearForm>(&self, time: f64, kernel: &K) -> Vec<f64> {
        let nfields = kernel.nfields();
        assert!(nfields > 0, "linear form must contain at least one field");
        let n = self.reduced_size();
        let cd = &self.cell_data;
        let mut grads = vec![0.0; cd.ndofs * cd.npts];
        let mut local = vec![0.0; nfields * cd.ndofs];
        let mut rhs = vec![0.0; nfields * n];

        for (c, reduced_dofs) in self.cell_reduced_dofs.iter().enumerate() {
            let ndofs = reduced_dofs.len();
            let ctx = self.prepare_cell_ctx(time, c, ndofs, &mut grads[..ndofs * cd.npts]);
            kernel.assemble_local_rhs(&ctx, &mut local[..nfields * ndofs]);
            scatter_local_vector(
                &mut rhs,
                &local[..nfields * ndofs],
                reduced_dofs,
                nfields,
                n,
            );
        }
        rhs
    }

    pub fn assemble_linear<K: LinearForm>(&self, kernel: &K) -> Vec<f64> {
        assert_eq!(kernel.nfields(), 1, "scalar RHS requires a one-field form");
        self.assemble_system_linear_at(0.0, kernel)
    }

    pub fn assemble_system_linear<K: LinearForm>(&self, kernel: &K) -> Vec<f64> {
        self.assemble_system_linear_at(0.0, kernel)
    }

    /// Assemble a linear RHS and apply the nonzero Dirichlet correction.
    pub fn assemble_system_linear_with_dirichlet_at<B, L>(
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
        let mut rhs = self.assemble_system_linear_at(time, linear);
        self.apply_dirichlet_rhs_correction_at(time, bilinear, &mut rhs);
        rhs
    }

    pub fn assemble_linear_with_dirichlet<B, L>(&self, bilinear: &B, linear: &L) -> Vec<f64>
    where
        B: BilinearForm,
        L: LinearForm,
    {
        assert_eq!(
            bilinear.nfields(),
            1,
            "scalar matrix requires a one-field form"
        );
        assert_eq!(linear.nfields(), 1, "scalar RHS requires a one-field form");
        self.assemble_system_linear_with_dirichlet_at(0.0, bilinear, linear)
    }

    /// Apply the prescribed-DOF contribution `-A_fb u_b` to a reduced RHS.
    ///
    /// Call this after assembling a source RHS and before solving a linear
    /// problem with nonzero Dirichlet values. Homogeneous Dirichlet values and
    /// problems without Dirichlet reduction are no-ops.
    pub fn apply_dirichlet_rhs_correction_at<K: BilinearForm>(
        &self,
        time: f64,
        kernel: &K,
        rhs: &mut [f64],
    ) {
        let nfields = kernel.nfields();
        assert!(nfields > 0, "bilinear form must contain at least one field");
        assert_eq!(
            rhs.len(),
            nfields * self.reduced_size(),
            "RHS size mismatch"
        );
        if !self
            .cell_prescribed_values
            .iter()
            .flatten()
            .any(Option::is_some)
        {
            return;
        }
        let cd = &self.cell_data;
        let mut grads = vec![0.0; cd.ndofs * cd.npts];
        let local_size = nfields * cd.ndofs;
        let mut local = vec![0.0; local_size * local_size];
        for (cell_index, reduced_dofs) in self.cell_reduced_dofs.iter().enumerate() {
            let prescribed = &self.cell_prescribed_values[cell_index];
            if !prescribed.iter().any(Option::is_some) {
                continue;
            }
            let ndofs = reduced_dofs.len();
            let ctx = self.prepare_cell_ctx(time, cell_index, ndofs, &mut grads[..ndofs * cd.npts]);
            local.fill(0.0);
            kernel.assemble_local(&ctx, &mut local[..local_size * local_size]);
            add_dirichlet_rhs_correction(
                rhs,
                &local[..local_size * local_size],
                reduced_dofs,
                prescribed,
                nfields,
                self.reduced_size(),
            );
        }
    }

    pub fn apply_dirichlet_rhs_correction<K: BilinearForm>(&self, kernel: &K, rhs: &mut [f64]) {
        self.apply_dirichlet_rhs_correction_at(0.0, kernel, rhs);
    }

    /// Assemble block-diagonal lumped GLL mass for `nfields` scalar fields.
    pub fn assemble_system_lumped_mass(&self, nfields: usize) -> SparseColMat<usize, f64> {
        assemble_lumped_mass(
            &self.cell_data,
            &self.cell_reduced_dofs,
            self.reduced_size(),
            nfields,
        )
    }

    /// Assemble the diagonal GLL mass matrix for one scalar field.
    pub fn assemble_lumped_mass(&self) -> SparseColMat<usize, f64> {
        self.assemble_system_lumped_mass(1)
    }

    pub fn assemble_boundary_at<'a, F>(&self, time: f64, mut select: F) -> BoundaryContributions
    where
        F: FnMut(BoundaryPoint) -> Option<&'a dyn BoundaryIntegrator>,
    {
        let space = FunctionSpaceImpl::new(&self.mesh, &self.family);
        let element = self.family.element(ReferenceCellType::Interval);
        let mut rhs = Vec::new();
        let mut nfields = 1;
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
            if rhs.is_empty() {
                nfields = kernel.nfields();
                assert!(
                    nfields > 0,
                    "boundary integrator must contain at least one field"
                );
                rhs.resize(nfields * self.reduced_size(), 0.0);
            } else {
                assert_eq!(
                    kernel.nfields(),
                    nfields,
                    "all selected boundary integrators must have the same field count"
                );
            }
            let mut reference_point = rlst_dynamic_array!(f64, [1, 1]);
            *reference_point.get_mut([0, 0]).unwrap() = local_point as f64;
            let mut table = DynArray::<f64, 4>::from_shape(element.tabulate_array_shape(0, 1));
            element.tabulate(&reference_point, 0, &mut table);
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
                grads: &[],
            };
            let mut local_rhs = vec![0.0; nfields];
            let mut local_mat = vec![0.0; nfields * nfields];
            kernel.assemble_facet_rhs(&ctx, &mut local_rhs);
            kernel.assemble_facet_mat(&ctx, &mut local_mat);
            if let Some(reduced) = self.target_dof(point_dofs[0]) {
                for equation in 0..nfields {
                    rhs[equation * self.reduced_size() + reduced] += local_rhs[equation];
                    for unknown in 0..nfields {
                        let value = local_mat[equation * nfields + unknown];
                        if value != 0.0 {
                            triplets.push(Triplet::new(
                                equation * self.reduced_size() + reduced,
                                unknown * self.reduced_size() + reduced,
                                value,
                            ));
                        }
                    }
                }
            }
        }
        if rhs.is_empty() {
            rhs.resize(self.reduced_size(), 0.0);
        }
        let system_size = nfields * self.reduced_size();
        BoundaryContributions {
            rhs,
            mat: SparseColMat::try_new_from_triplets(system_size, system_size, &triplets).unwrap(),
        }
    }

    pub fn assemble_boundary<'a, F>(&self, select: F) -> BoundaryContributions
    where
        F: FnMut(BoundaryPoint) -> Option<&'a dyn BoundaryIntegrator>,
    {
        self.assemble_boundary_at(0.0, select)
    }
}
