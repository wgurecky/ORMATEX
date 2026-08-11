use crate::common::{
    assemble_quad_boundaries, BoundaryContributions, BoundaryFacet, CellData, CellState, LocalCtx,
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
use ndmesh::traits::{Entity, GeometryMap, Mesh, Topology};
use quadraturerules::{single_integral_quadrature, Domain, QuadratureRule};
use rayon::prelude::*;
use rlst::{rlst_dynamic_array, DynArray};

// ponytail: fixed batches reuse scratch without creating a task per cell; tune only after profiling.
const CELL_BATCH_SIZE: usize = 32;

/// DOF reduction selector for a 2D quadrilateral problem.
///
/// * `Periodic` -- identify left<->right and bottom<->top boundary vertices
///   (cyclic), system size = (# unique canonical vertices).
/// * `None` -- retain all dofs.
/// * `Dirichlet { facets_to_eliminate }` -- eliminate every DOF on the listed
///   boundary interval facets.
#[derive(Clone, Debug)]
pub enum DofReduction2D {
    /// Identify left<->right and bottom<->top boundary vertices (cyclic).
    Periodic,
    /// Retain all dofs.
    None,
    /// Eliminate every closure DOF on the listed boundary facets (homogeneous
    /// Dirichlet), including high-order edge DOFs.
    Dirichlet { facets_to_eliminate: Vec<usize> },
}

// `CellData` (per-cell-type quadrature + tabulation + jacobian caches) is
// shared with the 1D example via `ex_nd_common::CellData`.  See there for
// the struct definition + per-field docs.

// =============================================================================
// SEM2DProblem
// =============================================================================

/// 2D GLL spectral-element problem on quadrilateral meshes.
pub struct SEM2DProblem<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> {
    pub mesh: M,
    pub family: LagrangeElementFamily<f64>,
    p: usize,
    cell_data: CellData,
    cell_dofs: Vec<Vec<usize>>,
    cell_reduced_dofs: Vec<Vec<Option<usize>>>,
    bc: DofReduction2D,
    /// Precomputed `full -> Option<reduced>` DOF map (BC reduction LUT).
    /// Built once in `new`; `target_dof` is O(1), `apply_bc` is O(nnz).
    dof_lut: Vec<Option<usize>>,
    /// Number of reduced dofs (count of unique canonical vertices after
    /// periodic identification).  Distinct from `dof_lut.len()` when BCs
    /// identify boundary dofs.
    n_reduced: usize,
    /// (x, y) position of each full GLL nodal DOF, used to build the periodic
    /// identification LUT and to expose `dof_positions()`.
    dof_xy: Vec<(f64, f64)>,
    metadata: MeshMetadata,
}

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> SEM2DProblem<M> {
    /// Build a GLL quadrilateral spectral-element problem.
    pub fn new(mesh: M, p: usize, bc: DofReduction2D) -> Self {
        Self::new_with_metadata(mesh, p, bc, MeshMetadata::default())
    }

    pub fn new_with_metadata(
        mesh: M,
        p: usize,
        bc: DofReduction2D,
        metadata: MeshMetadata,
    ) -> Self {
        assert!(p >= 1, "polynomial degree p must be >= 1");
        let family =
            LagrangeElementFamily::<f64>::new(p, Continuity::Standard, LagrangeVariant::GLL);
        let tdim = mesh.topology_dim();
        let gdim = mesh.geometry_dim();
        assert_eq!(tdim, 2, "SEM2DProblem: mesh tdim must be 2");
        assert_eq!(gdim, 2, "SEM2DProblem: mesh gdim must be 2 (2D-in-2D)");
        metadata.validate(
            mesh.entity_count(ReferenceCellType::Quadrilateral),
            mesh.entity_count(ReferenceCellType::Interval),
        );

        let space = FunctionSpaceImpl::new(&mesh, &family);
        let n = space.process_size();

        assert_eq!(
            mesh.entity_types(tdim),
            &[ReferenceCellType::Quadrilateral],
            "SEM2DProblem supports quadrilateral meshes only"
        );
        let cell_type = ReferenceCellType::Quadrilateral;
        let element = family.element(cell_type);
        let ndofs = element.dim();

        let order = element.lagrange_superdegree().saturating_sub(1);
        let (qx, qw) = single_integral_quadrature(
            QuadratureRule::GaussLobattoLegendre,
            Domain::Interval,
            order,
        )
        .unwrap();
        let n1d = qw.len();
        let xs_1d: Vec<f64> = (0..n1d).map(|i| qx[2 * i + 1]).collect();
        let npts = n1d * n1d;
        let mut pts = rlst_dynamic_array!(f64, [2, npts]);
        let mut wts = vec![0.0_f64; npts];
        for j in 0..n1d {
            for i in 0..n1d {
                let k = j * n1d + i;
                *pts.get_mut([0, k]).unwrap() = xs_1d[i];
                *pts.get_mut([1, k]).unwrap() = xs_1d[j];
                wts[k] = qw[i] * qw[j];
            }
        }

        let mut table = DynArray::<f64, 4>::from_shape(element.tabulate_array_shape(1, npts));
        element.tabulate(&pts, 1, &mut table);

        let ncells = mesh.entity_count(cell_type);
        let gmap = mesh.geometry_map(cell_type, 1, &pts);
        let mut jac_scratch = rlst_dynamic_array!(f64, [gdim, tdim, npts]);
        let mut jinv_scratch = rlst_dynamic_array!(f64, [tdim, gdim, npts]);
        let mut jdet_scratch = vec![0.0_f64; npts];
        let mut jinv_cache = rlst_dynamic_array!(f64, [tdim, gdim, npts, ncells]);
        let mut jdets_cache = vec![0.0_f64; ncells * npts];
        let mut physical_points_cache = vec![0.0_f64; ncells * npts * gdim];
        let mut dof_xy = vec![(f64::NAN, f64::NAN); n];
        let mut physical_pts = rlst_dynamic_array!(f64, [gdim, npts]);
        // The cell's local_index() indexes the quadrilateral Jacobian cache.
        for cell in mesh.entity_iter(cell_type) {
            let c = cell.local_index();
            gmap.jacobians_inverses_dets(c, &mut jac_scratch, &mut jinv_scratch, &mut jdet_scratch);
            for td in 0..tdim {
                for gd in 0..gdim {
                    for q in 0..npts {
                        *jinv_cache.get_mut([td, gd, q, c]).unwrap() =
                            *jinv_scratch.get([td, gd, q]).unwrap();
                    }
                }
            }
            for q in 0..npts {
                jdets_cache[c * npts + q] = jdet_scratch[q];
            }
            gmap.physical_points(c, &mut physical_pts);
            for q in 0..npts {
                for gd in 0..gdim {
                    physical_points_cache[(c * npts + q) * gdim + gd] =
                        *physical_pts.get([gd, q]).unwrap();
                }
            }
            let cell_dofs = space.entity_closure_dofs(cell_type, c).unwrap();
            for (local_dof, &full_dof) in cell_dofs.iter().enumerate() {
                let q = (0..npts)
                    .find(|&q| *table.get([0, q, local_dof, 0]).unwrap() > 1.0 - 1e-12)
                    .expect("GLL basis dof has no nodal quadrature point");
                let xy = (
                    *physical_pts.get([0, q]).unwrap(),
                    *physical_pts.get([1, q]).unwrap(),
                );
                let old = dof_xy[full_dof];
                if old.0.is_nan() {
                    dof_xy[full_dof] = xy;
                } else {
                    assert!(
                        (old.0 - xy.0).abs() < 1e-12 && (old.1 - xy.1).abs() < 1e-12,
                        "inconsistent coordinates for global dof {full_dof}"
                    );
                }
            }
        }
        drop(gmap);

        let mut reference_values = vec![0.0; ndofs * npts];
        let mut nodal_quadrature = vec![0; ndofs];
        for dof in 0..ndofs {
            for q in 0..npts {
                reference_values[dof * npts + q] = *table.get([0, q, dof, 0]).unwrap();
            }
            nodal_quadrature[dof] = (0..npts)
                .find(|&q| reference_values[dof * npts + q] > 1.0 - 1e-12)
                .expect("GLL basis dof has no nodal quadrature point");
        }
        let cell_data = CellData {
            pts,
            wts,
            npts,
            table,
            reference_values,
            nodal_quadrature,
            jinv_cache,
            jdets_cache,
            physical_points_cache,
            ndofs,
        };

        assert!(
            dof_xy.iter().all(|(x, y)| !x.is_nan() && !y.is_nan()),
            "every GLL dof must have a physical coordinate"
        );

        // --- DOF reduction LUT (full dof -> Option<reduced>).  Three cases:
        //   * `Periodic` -- identify every GLL node by snapped XY position
        //     (left<->right, top<->bottom).
        //   * `None` -- retain all dofs.
        //   * `Dirichlet { facets_to_eliminate }` -- eliminate every closure
        //     DOF on those boundary facets (`None` in the LUT), keep the rest
        //     contiguously-indexed.
        use std::collections::HashMap;
        let (dof_lut, n_reduced) = match &bc {
            DofReduction2D::Periodic => {
                const EPS: f64 = 1e-9;
                let snap = |c: f64| -> i64 {
                    let s = if c < EPS || c > 1.0 - EPS { 0.0 } else { c };
                    (s * 1e9).round() as i64
                };
                let mut canonical: HashMap<(i64, i64), usize> = HashMap::new();
                let mut canonical_dof = vec![0; n];
                for full in 0..n {
                    let (x, y) = dof_xy[full];
                    let key = (snap(x), snap(y));
                    canonical_dof[full] = *canonical.entry(key).or_insert(full);
                }
                let mut canonical_set: Vec<usize> =
                    (0..n).filter(|&full| canonical_dof[full] == full).collect();
                canonical_set.sort_unstable();
                let mut reduced_index: HashMap<usize, usize> = HashMap::new();
                for (i, &vi) in canonical_set.iter().enumerate() {
                    reduced_index.insert(vi, i);
                }
                let lut: Vec<Option<usize>> = (0..n)
                    .map(|d| {
                        let canon = canonical_dof[d];
                        Some(reduced_index[&canon])
                    })
                    .collect();
                (lut, canonical_set.len())
            }
            DofReduction2D::None => ((0..n).map(Some).collect(), n),
            DofReduction2D::Dirichlet {
                facets_to_eliminate,
            } => {
                use std::collections::HashSet;
                let mut eliminated_dofs = HashSet::new();
                for &facet_index in facets_to_eliminate {
                    let facet = mesh
                        .entity(ReferenceCellType::Interval, facet_index)
                        .expect("Dirichlet facet index out of range");
                    let topology = facet.topology();
                    let mut cells =
                        topology.connected_entity_iter(ReferenceCellType::Quadrilateral);
                    assert!(
                        cells.next().is_some() && cells.next().is_none(),
                        "Dirichlet facet {facet_index} must be a boundary interval"
                    );
                    for dof in space
                        .entity_closure_dofs(ReferenceCellType::Interval, facet_index)
                        .unwrap()
                    {
                        eliminated_dofs.insert(dof);
                    }
                }
                let mut lut: Vec<Option<usize>> = Vec::with_capacity(n);
                let mut reduced = 0;
                for d in 0..n {
                    if eliminated_dofs.contains(&d) {
                        lut.push(None);
                    } else {
                        lut.push(Some(reduced));
                        reduced += 1;
                    }
                }
                (lut, reduced)
            }
        };
        let cell_dofs: Vec<Vec<usize>> = (0..ncells)
            .map(|cell| {
                space
                    .entity_closure_dofs(ReferenceCellType::Quadrilateral, cell)
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
            p,
            cell_data,
            cell_dofs,
            cell_reduced_dofs,
            bc,
            dof_lut,
            n_reduced,
            dof_xy,
            metadata,
        }
    }

    /// O(1) full -> Option<reduced> DOF lookup.
    pub fn target_dof(&self, full: usize) -> Option<usize> {
        self.dof_lut[full]
    }

    /// Number of reduced dofs (post-BC) = count of unique canonical
    /// vertices after periodic identification.
    pub fn reduced_size(&self) -> usize {
        self.n_reduced
    }

    /// (x, y) position of each reduced dof, in reduced-index order.
    pub fn dof_positions(&self) -> Vec<(f64, f64)> {
        let nr = self.reduced_size();
        let mut out = vec![(f64::NAN, f64::NAN); nr];
        // Snap x>1-eps and y>1-eps to 0 only under `Periodic`.
        let do_snap = matches!(self.bc, DofReduction2D::Periodic);
        let snap = |c: f64| if do_snap && c > 1.0 - 1e-9 { 0.0 } else { c };
        for full in 0..self.dof_lut.len() {
            if let Some(r) = self.dof_lut[full] {
                let (x, y) = self.dof_xy[full];
                out[r] = (snap(x), snap(y));
            }
        }
        debug_assert!(out.iter().all(|(x, y)| !x.is_nan() && !y.is_nan()));
        out
    }

    fn populate_cell_grads(&self, cell_index: usize, ndofs: usize, grads: &mut [f64]) {
        let cd = &self.cell_data;
        let npts = cd.npts;
        let gdim = self.mesh.geometry_dim();
        let tdim = self.mesh.topology_dim();
        for dof_i in 0..ndofs {
            for q in 0..npts {
                for gd in 0..gdim {
                    let mut acc = 0.0;
                    for td in 0..tdim {
                        acc += *cd.jinv_cache.get([td, gd, q, cell_index]).unwrap()
                            * *cd.table.get([1 + td, q, dof_i, 0]).unwrap();
                    }
                    grads[(dof_i * gdim + gd) * npts + q] = acc;
                }
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
        let npts = cd.npts;
        let gdim = self.mesh.geometry_dim();
        let tdim = self.mesh.topology_dim();
        LocalCtx {
            time,
            cell: self.metadata.cell(cell_index),
            tdim,
            gdim,
            ncomp: 1,
            npts,
            ndofs,
            wts: &cd.wts,
            jdets: &cd.jdets_cache[cell_index * npts..(cell_index + 1) * npts],
            points: &cd.physical_points_cache
                [cell_index * npts * gdim..(cell_index + 1) * npts * gdim],
            values: &cd.reference_values,
            grads: &grads[..ndofs * gdim * npts],
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
        _ndofs: usize,
        reduced_dofs: &[Option<usize>],
        state: MatRef<'_, f64>,
        basis_grads: &[f64],
        values: &'a mut [f64],
        field_grads: &'a mut [f64],
    ) -> CellState<'a> {
        let cd = &self.cell_data;
        let npts = cd.npts;
        let gdim = self.mesh.geometry_dim();
        let n = self.reduced_size();
        values[..nfields * npts].fill(0.0);
        field_grads[..nfields * gdim * npts].fill(0.0);
        for field in 0..nfields {
            for (local_i, &reduced) in reduced_dofs.iter().enumerate() {
                let coefficient = reduced.map_or(0.0, |i| state[(field * n + i, 0)]);
                for q in 0..npts {
                    values[field * npts + q] +=
                        coefficient * cd.reference_values[local_i * npts + q];
                    for gd in 0..gdim {
                        field_grads[(field * gdim + gd) * npts + q] +=
                            coefficient * basis_grads[(local_i * gdim + gd) * npts + q];
                    }
                }
            }
        }
        CellState {
            nfields,
            npts,
            gdim,
            values: &values[..nfields * npts],
            grads: &field_grads[..nfields * gdim * npts],
        }
    }

    /// Assemble the positive weak spatial residual `R(U)` of a state-aware
    /// kernel. Semi-discrete systems apply the lumped inverse mass separately.
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
        let n = self.reduced_size();
        assert_eq!(state.nrows(), nfields * n, "state size mismatch");
        assert_eq!(
            state.ncols(),
            1,
            "residual assembly requires one state column"
        );
        let gdim = self.mesh.geometry_dim();
        let cd = &self.cell_data;
        let local_stride = nfields * cd.ndofs;
        let batches: Vec<Vec<f64>> = self
            .cell_dofs
            .par_chunks(CELL_BATCH_SIZE)
            .zip(self.cell_reduced_dofs.par_chunks(CELL_BATCH_SIZE))
            .enumerate()
            .map_init(
                || {
                    (
                        vec![0.0; cd.ndofs * gdim * cd.npts],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; nfields * gdim * cd.npts],
                        vec![0.0; local_stride],
                    )
                },
                |(basis_grads, field_values, field_grads, local),
                 (batch_index, (dofs_batch, reduced_batch))| {
                    let mut batch = vec![0.0; dofs_batch.len() * local_stride];
                    for (cell_offset, (dofs, reduced_dofs)) in
                        dofs_batch.iter().zip(reduced_batch).enumerate()
                    {
                        let ndofs = dofs.len();
                        let cell_size = nfields * ndofs;
                        let cell_index = batch_index * CELL_BATCH_SIZE + cell_offset;
                        self.populate_cell_grads(
                            cell_index,
                            ndofs,
                            &mut basis_grads[..ndofs * gdim * cd.npts],
                        );
                        let state_cell = self.prepare_cell_state(
                            nfields,
                            ndofs,
                            reduced_dofs,
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
                        batch[cell_offset * local_stride..cell_offset * local_stride + cell_size]
                            .copy_from_slice(&local[..cell_size]);
                    }
                    batch
                },
            )
            .collect();
        let mut residual = vec![0.0; nfields * n];
        for (batch_index, batch) in batches.into_iter().enumerate() {
            let cell_start = batch_index * CELL_BATCH_SIZE;
            for (cell_offset, reduced_dofs) in self.cell_reduced_dofs
                [cell_start..(cell_start + CELL_BATCH_SIZE).min(self.cell_reduced_dofs.len())]
                .iter()
                .enumerate()
            {
                let ndofs = self.cell_dofs[cell_start + cell_offset].len();
                let cell_size = nfields * ndofs;
                let local =
                    &batch[cell_offset * local_stride..cell_offset * local_stride + cell_size];
                for field in 0..nfields {
                    for (local_i, &reduced_i) in reduced_dofs.iter().enumerate() {
                        if let Some(reduced_i) = reduced_i {
                            residual[field * n + reduced_i] += local[field * ndofs + local_i];
                        }
                    }
                }
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

    /// Assemble `dR/du` for a state-aware kernel at `state`.
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
        let gdim = self.mesh.geometry_dim();
        let cd = &self.cell_data;
        let local_size = nfields * cd.ndofs;
        let batches: Vec<Vec<Triplet<usize, usize, f64>>> = self
            .cell_dofs
            .par_chunks(CELL_BATCH_SIZE)
            .zip(self.cell_reduced_dofs.par_chunks(CELL_BATCH_SIZE))
            .enumerate()
            .map_init(
                || {
                    (
                        vec![0.0; cd.ndofs * gdim * cd.npts],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; nfields * gdim * cd.npts],
                        vec![0.0; local_size * local_size],
                    )
                },
                |(basis_grads, field_values, field_grads, local),
                 (batch_index, (dofs_batch, reduced_batch))| {
                    let mut triplets =
                        Vec::with_capacity(dofs_batch.len() * local_size * local_size);
                    for (cell_offset, (dofs, reduced_dofs)) in
                        dofs_batch.iter().zip(reduced_batch).enumerate()
                    {
                        let ndofs = dofs.len();
                        let cell_size = nfields * ndofs;
                        let cell_index = batch_index * CELL_BATCH_SIZE + cell_offset;
                        self.populate_cell_grads(
                            cell_index,
                            ndofs,
                            &mut basis_grads[..ndofs * gdim * cd.npts],
                        );
                        let state_cell = self.prepare_cell_state(
                            nfields,
                            ndofs,
                            reduced_dofs,
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
                        local[..cell_size * cell_size].fill(0.0);
                        kernel.assemble_local_jacobian(
                            &ctx,
                            &state_cell,
                            &mut local[..cell_size * cell_size],
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
                                            let value = local[row * cell_size + col];
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

    /// Apply `dR/du(state)` to one or more direction columns without building
    /// a global sparse Jacobian. The local action is direct by default.
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
        let gdim = self.mesh.geometry_dim();
        let cd = &self.cell_data;
        let local_size = nfields * cd.ndofs;
        let ncols = direction.ncols();
        let action_stride = local_size * ncols;
        let batches: Vec<Vec<f64>> = self
            .cell_dofs
            .par_chunks(CELL_BATCH_SIZE)
            .zip(self.cell_reduced_dofs.par_chunks(CELL_BATCH_SIZE))
            .enumerate()
            .map_init(
                || {
                    (
                        vec![0.0; cd.ndofs * gdim * cd.npts],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; nfields * gdim * cd.npts],
                        vec![0.0; local_size],
                        vec![0.0; local_size],
                    )
                },
                |(basis_grads, field_values, field_grads, local_direction, local_action),
                 (batch_index, (dofs_batch, reduced_batch))| {
                    let mut batch = vec![0.0; dofs_batch.len() * action_stride];
                    for (cell_offset, (dofs, reduced_dofs)) in
                        dofs_batch.iter().zip(reduced_batch).enumerate()
                    {
                        let ndofs = dofs.len();
                        let cell_size = nfields * ndofs;
                        let cell_index = batch_index * CELL_BATCH_SIZE + cell_offset;
                        self.populate_cell_grads(
                            cell_index,
                            ndofs,
                            &mut basis_grads[..ndofs * gdim * cd.npts],
                        );
                        let state_cell = self.prepare_cell_state(
                            nfields,
                            ndofs,
                            reduced_dofs,
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
                let ndofs = self.cell_dofs[cell_start + cell_offset].len();
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

    /// Assemble a reduced sparse matrix using `kernel.assemble_local` per
    /// quadrilateral. Periodic DOFs are combined and eliminated DOFs omitted
    /// while scattering, so no full-size matrix is exposed.
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
        let gdim = self.mesh.geometry_dim();
        let cd = &self.cell_data;
        let max_ndofs = cd.ndofs;

        let local_size = nfields * max_ndofs;
        let npts = cd.npts;
        let batches: Vec<Vec<Triplet<usize, usize, f64>>> = self
            .cell_dofs
            .par_chunks(CELL_BATCH_SIZE)
            .zip(self.cell_reduced_dofs.par_chunks(CELL_BATCH_SIZE))
            .enumerate()
            .map_init(
                || {
                    (
                        vec![0.0_f64; max_ndofs * gdim * npts],
                        vec![0.0_f64; local_size * local_size],
                    )
                },
                |(grads_buf, local_mat), (batch_index, (dofs_batch, reduced_batch))| {
                    let mut triplets =
                        Vec::with_capacity(dofs_batch.len() * local_size * local_size);
                    for (cell_offset, (dofs, reduced_dofs)) in
                        dofs_batch.iter().zip(reduced_batch).enumerate()
                    {
                        let ndofs = dofs.len();
                        debug_assert!(ndofs == cd.ndofs, "ndofs mismatch: cell vs CellData");
                        let cell_size = nfields * ndofs;
                        let cell_index = batch_index * CELL_BATCH_SIZE + cell_offset;
                        let ctx = self.prepare_cell_ctx(
                            time,
                            cell_index,
                            ndofs,
                            &mut grads_buf[..ndofs * gdim * npts],
                        );
                        let mat_slice = &mut local_mat[..cell_size * cell_size];
                        mat_slice.fill(0.0);
                        kernel.assemble_local(&ctx, mat_slice);

                        for equation in 0..nfields {
                            for (ti, &reduced_i) in reduced_dofs.iter().enumerate() {
                                let Some(reduced_i) = reduced_i else {
                                    continue;
                                };
                                for unknown in 0..nfields {
                                    for (si, &reduced_j) in reduced_dofs.iter().enumerate() {
                                        let Some(reduced_j) = reduced_j else {
                                            continue;
                                        };
                                        let row = equation * ndofs + ti;
                                        let col = unknown * ndofs + si;
                                        let entry = mat_slice[row * cell_size + col];
                                        if entry.abs() > 1e-12 {
                                            triplets.push(Triplet::new(
                                                equation * n + reduced_i,
                                                unknown * n + reduced_j,
                                                entry,
                                            ));
                                        }
                                    }
                                }
                            }
                        }
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

    /// Assemble a reduced volume RHS from a user-defined `LinearForm`.
    pub fn assemble_system_linear_at<K: LinearForm>(&self, time: f64, kernel: &K) -> Vec<f64> {
        let nfields = kernel.nfields();
        assert!(nfields > 0, "linear form must contain at least one field");
        let n = self.reduced_size();
        let gdim = self.mesh.geometry_dim();
        let cd = &self.cell_data;
        let mut grads_buf = vec![0.0_f64; cd.ndofs * gdim * cd.npts];
        let mut local_rhs = vec![0.0_f64; nfields * cd.ndofs];
        let mut rhs = vec![0.0_f64; nfields * n];

        for (c, (dofs, reduced_dofs)) in self
            .cell_dofs
            .iter()
            .zip(&self.cell_reduced_dofs)
            .enumerate()
        {
            let ndofs = dofs.len();
            let npts = cd.npts;
            let ctx = self.prepare_cell_ctx(time, c, ndofs, &mut grads_buf[..ndofs * gdim * npts]);
            kernel.assemble_local_rhs(&ctx, &mut local_rhs[..nfields * ndofs]);
            for field in 0..nfields {
                for (local_i, &reduced_i) in reduced_dofs.iter().enumerate() {
                    if let Some(reduced_i) = reduced_i {
                        rhs[field * n + reduced_i] += local_rhs[field * ndofs + local_i];
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

    /// Assemble selected natural-boundary kernels. The selector receives the
    /// `ndmesh` facet index and midpoint; returning `None` leaves it adiabatic.
    pub fn assemble_boundary<'a, F>(&self, select: F) -> BoundaryContributions
    where
        F: FnMut(BoundaryFacet) -> Option<&'a dyn BoundaryIntegrator>,
    {
        assemble_quad_boundaries(
            &self.mesh,
            &self.family,
            self.p,
            self.reduced_size(),
            &self.metadata.facet_regions,
            0.0,
            |full| self.target_dof(full),
            select,
        )
    }

    pub fn assemble_boundary_at<'a, F>(&self, time: f64, select: F) -> BoundaryContributions
    where
        F: FnMut(BoundaryFacet) -> Option<&'a dyn BoundaryIntegrator>,
    {
        assemble_quad_boundaries(
            &self.mesh,
            &self.family,
            self.p,
            self.reduced_size(),
            &self.metadata.facet_regions,
            time,
            |full| self.target_dof(full),
            select,
        )
    }
}
