use crate::common::{BoundaryContributions, CellData, CellState, FacetCtx, LocalCtx};
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
use rlst::{rlst_dynamic_array, DynArray};

/// DOF reduction for a 1D interval mesh. Facets are `Point` entity indices.
#[derive(Clone, Debug)]
pub enum DofReduction1D {
    None,
    /// Identify the second endpoint facet with the first endpoint facet.
    Periodic {
        facets: [usize; 2],
    },
    /// Eliminate every closure DOF on selected endpoint facets.
    Dirichlet {
        facets_to_eliminate: Vec<usize>,
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

/// 1D GLL spectral-element problem on interval meshes.
pub struct SEM1DProblem<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> {
    pub mesh: M,
    pub family: LagrangeElementFamily<f64>,
    cell_data: CellData,
    cell_dofs: Vec<Vec<usize>>,
    cell_reduced_dofs: Vec<Vec<Option<usize>>>,
    dof_lut: Vec<Option<usize>>,
    n_reduced: usize,
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
            pts,
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

        let (dof_lut, n_reduced) = match reduction {
            DofReduction1D::None => ((0..n).map(Some).collect(), n),
            DofReduction1D::Dirichlet {
                facets_to_eliminate,
            } => {
                let mut eliminated = std::collections::HashSet::new();
                for facet in facets_to_eliminate {
                    eliminated.insert(boundary_dof(facet));
                }
                let mut reduced = 0;
                let lut: Vec<Option<usize>> = (0..n)
                    .map(|dof| {
                        if eliminated.contains(&dof) {
                            None
                        } else {
                            let out = Some(reduced);
                            reduced += 1;
                            out
                        }
                    })
                    .collect();
                (lut, reduced)
            }
            DofReduction1D::Periodic { facets } => {
                let master = boundary_dof(facets[0]);
                let slave = boundary_dof(facets[1]);
                assert_ne!(master, slave, "periodic facets must be distinct");
                let canonical: Vec<usize> = (0..n)
                    .map(|dof| if dof == slave { master } else { dof })
                    .collect();
                let canonical_set: Vec<usize> =
                    (0..n).filter(|&dof| canonical[dof] == dof).collect();
                let reduced_index: std::collections::HashMap<usize, usize> = canonical_set
                    .iter()
                    .enumerate()
                    .map(|(reduced, &dof)| (dof, reduced))
                    .collect();
                let lut = canonical
                    .iter()
                    .map(|dof| Some(reduced_index[dof]))
                    .collect();
                (lut, canonical_set.len())
            }
        };
        let cell_dofs: Vec<Vec<usize>> = (0..ncells)
            .map(|cell| {
                space
                    .entity_closure_dofs(ReferenceCellType::Interval, cell)
                    .unwrap()
                    .to_vec()
            })
            .collect();
        let cell_reduced_dofs = cell_dofs
            .iter()
            .map(|dofs| dofs.iter().map(|&dof| dof_lut[dof]).collect())
            .collect();

        Self {
            mesh,
            family,
            cell_data,
            cell_dofs,
            cell_reduced_dofs,
            dof_lut,
            n_reduced,
            dof_x,
            metadata,
        }
    }

    pub fn target_dof(&self, full: usize) -> Option<usize> {
        self.dof_lut[full]
    }

    pub fn reduced_size(&self) -> usize {
        self.n_reduced
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
        let cd = &self.cell_data;
        LocalCtx {
            time,
            cell: self.metadata.cell(cell_index),
            tdim: 1,
            gdim: 1,
            ncomp: 1,
            npts: cd.npts,
            ndofs,
            wts: &cd.wts,
            jdets: &cd.jdets_cache[cell_index * cd.npts..(cell_index + 1) * cd.npts],
            points: &cd.physical_points_cache[cell_index * cd.npts..(cell_index + 1) * cd.npts],
            values: &cd.reference_values,
            grads: &grads[..ndofs * cd.npts],
        }
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
        state: MatRef<'_, f64>,
        basis_grads: &[f64],
        values: &'a mut [f64],
        field_grads: &'a mut [f64],
    ) -> CellState<'a> {
        let cd = &self.cell_data;
        let n = self.reduced_size();
        values[..nfields * cd.npts].fill(0.0);
        field_grads[..nfields * cd.npts].fill(0.0);
        for field in 0..nfields {
            for (local_i, &reduced) in reduced_dofs.iter().enumerate() {
                let coefficient = reduced.map_or(0.0, |i| state[(field * n + i, 0)]);
                for q in 0..cd.npts {
                    values[field * cd.npts + q] +=
                        coefficient * cd.reference_values[local_i * cd.npts + q];
                    field_grads[field * cd.npts + q] +=
                        coefficient * basis_grads[local_i * cd.npts + q];
                }
            }
        }
        CellState {
            nfields,
            npts: cd.npts,
            gdim: 1,
            values: &values[..nfields * cd.npts],
            grads: &field_grads[..nfields * cd.npts],
        }
    }

    pub fn assemble_system_residual_at<K: ResidualKernel>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
    ) -> Vec<f64> {
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
        let mut basis_grads = vec![0.0; cd.ndofs * cd.npts];
        let mut field_values = vec![0.0; nfields * cd.npts];
        let mut field_grads = vec![0.0; nfields * cd.npts];
        let mut local = vec![0.0; nfields * cd.ndofs];
        let mut residual = vec![0.0; nfields * self.reduced_size()];
        for (c, (dofs, reduced_dofs)) in self
            .cell_dofs
            .iter()
            .zip(&self.cell_reduced_dofs)
            .enumerate()
        {
            let ndofs = dofs.len();
            self.populate_cell_grads(c, ndofs, &mut basis_grads[..ndofs * cd.npts]);
            let state_cell = self.prepare_cell_state(
                nfields,
                reduced_dofs,
                state,
                &basis_grads[..ndofs * cd.npts],
                &mut field_values,
                &mut field_grads,
            );
            let ctx = self.cell_ctx(time, c, ndofs, &basis_grads[..ndofs * cd.npts]);
            kernel.assemble_local_residual(&ctx, &state_cell, &mut local[..nfields * ndofs]);
            for field in 0..nfields {
                for (local_i, &reduced_i) in reduced_dofs.iter().enumerate() {
                    if let Some(reduced_i) = reduced_i {
                        residual[field * self.reduced_size() + reduced_i] +=
                            local[field * ndofs + local_i];
                    }
                }
            }
        }
        residual
    }

    pub fn assemble_residual<K: ResidualKernel>(&self, kernel: &K, state: MatRef<f64>) -> Vec<f64> {
        assert_eq!(
            kernel.nfields(),
            1,
            "scalar residual requires a one-field kernel"
        );
        self.assemble_system_residual_at(0.0, kernel, state)
    }

    pub fn assemble_system_residual<K: ResidualKernel>(
        &self,
        kernel: &K,
        state: MatRef<f64>,
    ) -> Vec<f64> {
        self.assemble_system_residual_at(0.0, kernel, state)
    }

    pub fn assemble_system_residual_jacobian_at<K: ResidualKernel>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
    ) -> SparseColMat<usize, f64> {
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
        let mut basis_grads = vec![0.0; cd.ndofs * cd.npts];
        let mut field_values = vec![0.0; nfields * cd.npts];
        let mut field_grads = vec![0.0; nfields * cd.npts];
        let local_size = nfields * cd.ndofs;
        let mut local = vec![0.0; local_size * local_size];
        let mut triplets = Vec::with_capacity(self.cell_dofs.len() * local_size * local_size);
        for (c, (dofs, reduced_dofs)) in self
            .cell_dofs
            .iter()
            .zip(&self.cell_reduced_dofs)
            .enumerate()
        {
            let ndofs = dofs.len();
            self.populate_cell_grads(c, ndofs, &mut basis_grads[..ndofs * cd.npts]);
            let state_cell = self.prepare_cell_state(
                nfields,
                reduced_dofs,
                state,
                &basis_grads[..ndofs * cd.npts],
                &mut field_values,
                &mut field_grads,
            );
            let ctx = self.cell_ctx(time, c, ndofs, &basis_grads[..ndofs * cd.npts]);
            local[..local_size * local_size].fill(0.0);
            kernel.assemble_local_jacobian(
                &ctx,
                &state_cell,
                &mut local[..local_size * local_size],
            );
            for equation in 0..nfields {
                for (ti, &reduced_i) in reduced_dofs.iter().enumerate() {
                    let Some(reduced_i) = reduced_i else {
                        continue;
                    };
                    for unknown in 0..nfields {
                        for (si, &reduced_j) in reduced_dofs.iter().enumerate() {
                            if let Some(reduced_j) = reduced_j {
                                let row = equation * ndofs + ti;
                                let col = unknown * ndofs + si;
                                let value = local[row * local_size + col];
                                if value != 0.0 {
                                    triplets.push(Triplet::new(
                                        equation * n + reduced_i,
                                        unknown * n + reduced_j,
                                        value,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        let system_size = nfields * n;
        SparseColMat::try_new_from_triplets(system_size, system_size, &triplets).unwrap()
    }

    pub fn assemble_residual_jacobian<K: ResidualKernel>(
        &self,
        kernel: &K,
        state: MatRef<f64>,
    ) -> SparseColMat<usize, f64> {
        assert_eq!(
            kernel.nfields(),
            1,
            "scalar Jacobian requires a one-field kernel"
        );
        self.assemble_system_residual_jacobian_at(0.0, kernel, state)
    }

    pub fn assemble_system_residual_jacobian<K: ResidualKernel>(
        &self,
        kernel: &K,
        state: MatRef<f64>,
    ) -> SparseColMat<usize, f64> {
        self.assemble_system_residual_jacobian_at(0.0, kernel, state)
    }

    pub fn apply_system_jacobian_matfree_at<K: ResidualKernel>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
    ) -> Mat<f64> {
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
        let mut basis_grads = vec![0.0; cd.ndofs * cd.npts];
        let mut field_values = vec![0.0; nfields * cd.npts];
        let mut field_grads = vec![0.0; nfields * cd.npts];
        let local_size = nfields * cd.ndofs;
        let mut local_direction = vec![0.0; local_size];
        let mut local_action = vec![0.0; local_size];
        let mut out = Mat::<f64>::zeros(nfields * n, direction.ncols());
        for (c, (dofs, reduced_dofs)) in self
            .cell_dofs
            .iter()
            .zip(&self.cell_reduced_dofs)
            .enumerate()
        {
            let ndofs = dofs.len();
            self.populate_cell_grads(c, ndofs, &mut basis_grads[..ndofs * cd.npts]);
            let state_cell = self.prepare_cell_state(
                nfields,
                reduced_dofs,
                state,
                &basis_grads[..ndofs * cd.npts],
                &mut field_values,
                &mut field_grads,
            );
            let ctx = self.cell_ctx(time, c, ndofs, &basis_grads[..ndofs * cd.npts]);
            for column in 0..direction.ncols() {
                for field in 0..nfields {
                    for (local_i, &reduced_i) in reduced_dofs.iter().enumerate() {
                        local_direction[field * ndofs + local_i] =
                            reduced_i.map_or(0.0, |i| direction[(field * n + i, column)]);
                    }
                }
                kernel.apply_local_jacobian(
                    &ctx,
                    &state_cell,
                    &local_direction[..local_size],
                    &mut local_action[..local_size],
                );
                for field in 0..nfields {
                    for (local_i, &reduced_i) in reduced_dofs.iter().enumerate() {
                        if let Some(reduced_i) = reduced_i {
                            out[(field * n + reduced_i, column)] +=
                                local_action[field * ndofs + local_i];
                        }
                    }
                }
            }
        }
        out
    }

    pub fn apply_jacobian_matfree<K: ResidualKernel>(
        &self,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
    ) -> Mat<f64> {
        assert_eq!(
            kernel.nfields(),
            1,
            "scalar Jacobian requires a one-field kernel"
        );
        self.apply_system_jacobian_matfree_at(0.0, kernel, state, direction)
    }

    pub fn apply_system_jacobian_matfree<K: ResidualKernel>(
        &self,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
    ) -> Mat<f64> {
        self.apply_system_jacobian_matfree_at(0.0, kernel, state, direction)
    }

    pub fn assemble_system_bilinear_at<K: BilinearForm>(
        &self,
        time: f64,
        kernel: &K,
    ) -> SparseColMat<usize, f64> {
        let nfields = kernel.nfields();
        assert!(nfields > 0, "bilinear form must contain at least one field");
        let n = self.reduced_size();
        let cd = &self.cell_data;
        let mut grads = vec![0.0; cd.ndofs * cd.npts];
        let local_size = nfields * cd.ndofs;
        let mut local = vec![0.0; local_size * local_size];
        let mut triplets = Vec::with_capacity(self.cell_dofs.len() * local_size * local_size);

        for (c, (dofs, reduced_dofs)) in self
            .cell_dofs
            .iter()
            .zip(&self.cell_reduced_dofs)
            .enumerate()
        {
            let ndofs = dofs.len();
            let ctx = self.prepare_cell_ctx(time, c, ndofs, &mut grads[..ndofs * cd.npts]);
            let local = &mut local[..local_size * local_size];
            local.fill(0.0);
            kernel.assemble_local(&ctx, local);
            for equation in 0..nfields {
                for (ti, &reduced_i) in reduced_dofs.iter().enumerate() {
                    let Some(reduced_i) = reduced_i else {
                        continue;
                    };
                    for unknown in 0..nfields {
                        for (si, &reduced_j) in reduced_dofs.iter().enumerate() {
                            if let Some(reduced_j) = reduced_j {
                                let row = equation * ndofs + ti;
                                let col = unknown * ndofs + si;
                                let value = local[row * local_size + col];
                                if value.abs() > 1e-12 {
                                    triplets.push(Triplet::new(
                                        equation * n + reduced_i,
                                        unknown * n + reduced_j,
                                        value,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        let system_size = nfields * n;
        SparseColMat::try_new_from_triplets(system_size, system_size, &triplets).unwrap()
    }

    pub fn assemble_bilinear<K: BilinearForm>(&self, kernel: &K) -> SparseColMat<usize, f64> {
        assert_eq!(
            kernel.nfields(),
            1,
            "scalar matrix requires a one-field form"
        );
        self.assemble_system_bilinear_at(0.0, kernel)
    }

    pub fn assemble_system_bilinear<K: BilinearForm>(
        &self,
        kernel: &K,
    ) -> SparseColMat<usize, f64> {
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

        for (c, (dofs, reduced_dofs)) in self
            .cell_dofs
            .iter()
            .zip(&self.cell_reduced_dofs)
            .enumerate()
        {
            let ndofs = dofs.len();
            let ctx = self.prepare_cell_ctx(time, c, ndofs, &mut grads[..ndofs * cd.npts]);
            kernel.assemble_local_rhs(&ctx, &mut local[..nfields * ndofs]);
            for field in 0..nfields {
                for (local_i, &reduced_i) in reduced_dofs.iter().enumerate() {
                    if let Some(reduced_i) = reduced_i {
                        rhs[field * n + reduced_i] += local[field * ndofs + local_i];
                    }
                }
            }
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

    /// Assemble block-diagonal lumped GLL mass for `nfields` scalar fields.
    pub fn assemble_system_lumped_mass(&self, nfields: usize) -> SparseColMat<usize, f64> {
        assert!(nfields > 0, "mass requires at least one field");
        let n = self.reduced_size();
        let cd = &self.cell_data;
        let mut triplets = Vec::with_capacity(self.cell_dofs.len() * cd.ndofs * nfields);
        for field in 0..nfields {
            for (cell, reduced_dofs) in self.cell_reduced_dofs.iter().enumerate() {
                for (local_dof, &reduced) in reduced_dofs.iter().enumerate() {
                    if let Some(reduced) = reduced {
                        let q = cd.nodal_quadrature[local_dof];
                        let index = field * n + reduced;
                        triplets.push(Triplet::new(
                            index,
                            index,
                            cd.wts[q] * cd.jdets_cache[cell * cd.npts + q],
                        ));
                    }
                }
            }
        }
        let system_size = nfields * n;
        SparseColMat::try_new_from_triplets(system_size, system_size, &triplets).unwrap()
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
                physical_region: self
                    .metadata
                    .facet_regions
                    .get(point_index)
                    .copied()
                    .flatten(),
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
