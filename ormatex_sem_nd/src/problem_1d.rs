use crate::common::*;
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
}

/// 1D GLL spectral-element problem on interval meshes.
pub struct FiniteElement1DProblem<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> {
    pub mesh: M,
    pub family: LagrangeElementFamily<f64>,
    cell_data: CellData,
    cell_dofs: Vec<Vec<usize>>,
    cell_reduced_dofs: Vec<Vec<Option<usize>>>,
    dof_lut: Vec<Option<usize>>,
    n_reduced: usize,
    dof_x: Vec<f64>,
}

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> FiniteElement1DProblem<M> {
    pub fn new(mesh: M, p: usize, reduction: DofReduction1D) -> Self {
        assert!(p >= 1, "polynomial degree p must be >= 1");
        assert_eq!(
            mesh.topology_dim(),
            1,
            "FiniteElement1DProblem: mesh tdim must be 1"
        );
        assert_eq!(
            mesh.geometry_dim(),
            1,
            "FiniteElement1DProblem: mesh gdim must be 1"
        );
        assert_eq!(
            mesh.entity_types(1),
            &[ReferenceCellType::Interval],
            "FiniteElement1DProblem supports interval meshes only"
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

    fn cell_ctx<'a>(&'a self, cell_index: usize, ndofs: usize, grads: &'a [f64]) -> LocalCtx<'a> {
        let cd = &self.cell_data;
        LocalCtx {
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
        cell_index: usize,
        ndofs: usize,
        grads: &'a mut [f64],
    ) -> LocalCtx<'a> {
        self.populate_cell_grads(cell_index, ndofs, grads);
        self.cell_ctx(cell_index, ndofs, grads)
    }

    fn prepare_cell_field<'a>(
        &self,
        reduced_dofs: &[Option<usize>],
        state: MatRef<'_, f64>,
        basis_grads: &[f64],
        values: &'a mut [f64],
        field_grads: &'a mut [f64],
    ) -> CellField<'a> {
        let cd = &self.cell_data;
        values[..cd.npts].fill(0.0);
        field_grads[..cd.npts].fill(0.0);
        for (local_i, &reduced) in reduced_dofs.iter().enumerate() {
            let coefficient = reduced.map_or(0.0, |i| state[(i, 0)]);
            for q in 0..cd.npts {
                values[q] += coefficient * cd.reference_values[local_i * cd.npts + q];
                field_grads[q] += coefficient * basis_grads[local_i * cd.npts + q];
            }
        }
        CellField {
            npts: cd.npts,
            gdim: 1,
            values: &values[..cd.npts],
            grads: &field_grads[..cd.npts],
        }
    }

    pub fn assemble_residual<K: ResidualKernel>(&self, kernel: &K, state: MatRef<f64>) -> Vec<f64> {
        assert_eq!(state.nrows(), self.reduced_size(), "state size mismatch");
        assert_eq!(
            state.ncols(),
            1,
            "residual assembly requires one state column"
        );
        let cd = &self.cell_data;
        let mut basis_grads = vec![0.0; cd.ndofs * cd.npts];
        let mut field_values = vec![0.0; cd.npts];
        let mut field_grads = vec![0.0; cd.npts];
        let mut local = vec![0.0; cd.ndofs];
        let mut residual = vec![0.0; self.reduced_size()];
        for (c, (dofs, reduced_dofs)) in self
            .cell_dofs
            .iter()
            .zip(&self.cell_reduced_dofs)
            .enumerate()
        {
            let ndofs = dofs.len();
            self.populate_cell_grads(c, ndofs, &mut basis_grads[..ndofs * cd.npts]);
            let field = self.prepare_cell_field(
                reduced_dofs,
                state,
                &basis_grads[..ndofs * cd.npts],
                &mut field_values,
                &mut field_grads,
            );
            let ctx = self.cell_ctx(c, ndofs, &basis_grads[..ndofs * cd.npts]);
            kernel.assemble_local_residual(&ctx, &field, &mut local[..ndofs]);
            for (local_i, &reduced_i) in reduced_dofs.iter().enumerate() {
                if let Some(reduced_i) = reduced_i {
                    residual[reduced_i] += local[local_i];
                }
            }
        }
        residual
    }

    pub fn assemble_residual_jacobian<K: ResidualKernel>(
        &self,
        kernel: &K,
        state: MatRef<f64>,
    ) -> SparseColMat<usize, f64> {
        assert_eq!(state.nrows(), self.reduced_size(), "state size mismatch");
        assert_eq!(
            state.ncols(),
            1,
            "Jacobian assembly requires one state column"
        );
        let cd = &self.cell_data;
        let mut basis_grads = vec![0.0; cd.ndofs * cd.npts];
        let mut field_values = vec![0.0; cd.npts];
        let mut field_grads = vec![0.0; cd.npts];
        let mut local = vec![0.0; cd.ndofs * cd.ndofs];
        let mut triplets = Vec::with_capacity(self.cell_dofs.len() * cd.ndofs * cd.ndofs);
        for (c, (dofs, reduced_dofs)) in self
            .cell_dofs
            .iter()
            .zip(&self.cell_reduced_dofs)
            .enumerate()
        {
            let ndofs = dofs.len();
            self.populate_cell_grads(c, ndofs, &mut basis_grads[..ndofs * cd.npts]);
            let field = self.prepare_cell_field(
                reduced_dofs,
                state,
                &basis_grads[..ndofs * cd.npts],
                &mut field_values,
                &mut field_grads,
            );
            let ctx = self.cell_ctx(c, ndofs, &basis_grads[..ndofs * cd.npts]);
            kernel.assemble_local_jacobian(&ctx, &field, &mut local[..ndofs * ndofs]);
            for (ti, &reduced_i) in reduced_dofs.iter().enumerate() {
                let Some(reduced_i) = reduced_i else {
                    continue;
                };
                for (si, &reduced_j) in reduced_dofs.iter().enumerate() {
                    if let Some(reduced_j) = reduced_j {
                        let value = local[ti * ndofs + si];
                        if value != 0.0 {
                            triplets.push(Triplet::new(reduced_i, reduced_j, value));
                        }
                    }
                }
            }
        }
        let n = self.reduced_size();
        SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
    }

    pub fn apply_jacobian_matfree<K: ResidualKernel>(
        &self,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
    ) -> Mat<f64> {
        assert_eq!(state.nrows(), self.reduced_size(), "state size mismatch");
        assert_eq!(
            state.ncols(),
            1,
            "matrix-free Jacobian requires one state column"
        );
        assert_eq!(
            direction.nrows(),
            self.reduced_size(),
            "direction size mismatch"
        );
        let cd = &self.cell_data;
        let mut basis_grads = vec![0.0; cd.ndofs * cd.npts];
        let mut field_values = vec![0.0; cd.npts];
        let mut field_grads = vec![0.0; cd.npts];
        let mut local_direction = vec![0.0; cd.ndofs];
        let mut local_action = vec![0.0; cd.ndofs];
        let mut out = Mat::<f64>::zeros(self.reduced_size(), direction.ncols());
        for (c, (dofs, reduced_dofs)) in self
            .cell_dofs
            .iter()
            .zip(&self.cell_reduced_dofs)
            .enumerate()
        {
            let ndofs = dofs.len();
            self.populate_cell_grads(c, ndofs, &mut basis_grads[..ndofs * cd.npts]);
            let field = self.prepare_cell_field(
                reduced_dofs,
                state,
                &basis_grads[..ndofs * cd.npts],
                &mut field_values,
                &mut field_grads,
            );
            let ctx = self.cell_ctx(c, ndofs, &basis_grads[..ndofs * cd.npts]);
            for column in 0..direction.ncols() {
                for (local_i, &reduced_i) in reduced_dofs.iter().enumerate() {
                    local_direction[local_i] = reduced_i.map_or(0.0, |i| direction[(i, column)]);
                }
                kernel.apply_local_jacobian(
                    &ctx,
                    &field,
                    &local_direction[..ndofs],
                    &mut local_action[..ndofs],
                );
                for (local_i, &reduced_i) in reduced_dofs.iter().enumerate() {
                    if let Some(reduced_i) = reduced_i {
                        out[(reduced_i, column)] += local_action[local_i];
                    }
                }
            }
        }
        out
    }

    pub fn assemble_bilinear<K: BilinearForm>(&self, kernel: &K) -> SparseColMat<usize, f64> {
        let cd = &self.cell_data;
        let mut grads = vec![0.0; cd.ndofs * cd.npts];
        let mut local = vec![0.0; cd.ndofs * cd.ndofs];
        let mut triplets = Vec::with_capacity(self.cell_dofs.len() * cd.ndofs * cd.ndofs);

        for (c, (dofs, reduced_dofs)) in self
            .cell_dofs
            .iter()
            .zip(&self.cell_reduced_dofs)
            .enumerate()
        {
            let ndofs = dofs.len();
            let ctx = self.prepare_cell_ctx(c, ndofs, &mut grads[..ndofs * cd.npts]);
            let local = &mut local[..ndofs * ndofs];
            local.fill(0.0);
            kernel.assemble_local(&ctx, local);
            for (ti, &reduced_i) in reduced_dofs.iter().enumerate() {
                let Some(reduced_i) = reduced_i else {
                    continue;
                };
                for (si, &reduced_j) in reduced_dofs.iter().enumerate() {
                    if let Some(reduced_j) = reduced_j {
                        let value = local[ti * ndofs + si];
                        if value.abs() > 1e-12 {
                            triplets.push(Triplet::new(reduced_i, reduced_j, value));
                        }
                    }
                }
            }
        }
        let n = self.reduced_size();
        SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
    }

    pub fn assemble_linear<K: LinearForm>(&self, kernel: &K) -> Vec<f64> {
        let cd = &self.cell_data;
        let mut grads = vec![0.0; cd.ndofs * cd.npts];
        let mut local = vec![0.0; cd.ndofs];
        let mut rhs = vec![0.0; self.reduced_size()];

        for (c, (dofs, reduced_dofs)) in self
            .cell_dofs
            .iter()
            .zip(&self.cell_reduced_dofs)
            .enumerate()
        {
            let ndofs = dofs.len();
            let ctx = self.prepare_cell_ctx(c, ndofs, &mut grads[..ndofs * cd.npts]);
            kernel.assemble_local_rhs(&ctx, &mut local[..ndofs]);
            for (local_i, &reduced_i) in reduced_dofs.iter().enumerate() {
                if let Some(reduced_i) = reduced_i {
                    rhs[reduced_i] += local[local_i];
                }
            }
        }
        rhs
    }

    /// Assemble the diagonal GLL mass matrix using collocated nodal quadrature.
    pub fn assemble_lumped_mass(&self) -> SparseColMat<usize, f64> {
        let cd = &self.cell_data;
        let mut triplets = Vec::with_capacity(self.cell_dofs.len() * cd.ndofs);
        for (cell, reduced_dofs) in self.cell_reduced_dofs.iter().enumerate() {
            for (local_dof, &reduced) in reduced_dofs.iter().enumerate() {
                if let Some(reduced) = reduced {
                    let q = cd.nodal_quadrature[local_dof];
                    triplets.push(Triplet::new(
                        reduced,
                        reduced,
                        cd.wts[q] * cd.jdets_cache[cell * cd.npts + q],
                    ));
                }
            }
        }
        let n = self.reduced_size();
        SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
    }

    pub fn assemble_boundary<'a, F>(&self, mut select: F) -> BoundaryContributions
    where
        F: FnMut(BoundaryPoint) -> Option<&'a dyn BoundaryIntegrator>,
    {
        let space = FunctionSpaceImpl::new(&self.mesh, &self.family);
        let element = self.family.element(ReferenceCellType::Interval);
        let mut rhs = vec![0.0; self.reduced_size()];
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
            }) else {
                continue;
            };
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
            let mut local_rhs = [0.0];
            let mut local_mat = [0.0];
            kernel.assemble_facet_rhs(&ctx, &mut local_rhs);
            kernel.assemble_facet_mat(&ctx, &mut local_mat);
            if let Some(reduced) = self.target_dof(point_dofs[0]) {
                rhs[reduced] += local_rhs[0];
                if local_mat[0] != 0.0 {
                    triplets.push(Triplet::new(reduced, reduced, local_mat[0]));
                }
            }
        }
        let n = self.reduced_size();
        BoundaryContributions {
            rhs,
            mat: SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap(),
        }
    }
}
