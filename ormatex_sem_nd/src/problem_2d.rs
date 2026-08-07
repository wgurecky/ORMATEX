use crate::common::*;
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
use rlst::{rlst_dynamic_array, DynArray};

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
// FiniteElement2DProblem
// =============================================================================

/// 2D GLL spectral-element problem on quadrilateral meshes.
pub struct FiniteElement2DProblem<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> {
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
}

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> FiniteElement2DProblem<M> {
    /// Build a GLL quadrilateral spectral-element problem.
    pub fn new(mesh: M, p: usize, bc: DofReduction2D) -> Self {
        assert!(p >= 1, "polynomial degree p must be >= 1");
        let family =
            LagrangeElementFamily::<f64>::new(p, Continuity::Standard, LagrangeVariant::GLL);
        let tdim = mesh.topology_dim();
        let gdim = mesh.geometry_dim();
        assert_eq!(tdim, 2, "FiniteElement2DProblem: mesh tdim must be 2");
        assert_eq!(
            gdim, 2,
            "FiniteElement2DProblem: mesh gdim must be 2 (2D-in-2D)"
        );

        let space = FunctionSpaceImpl::new(&mesh, &family);
        let n = space.process_size();

        assert_eq!(
            mesh.entity_types(tdim),
            &[ReferenceCellType::Quadrilateral],
            "FiniteElement2DProblem supports quadrilateral meshes only"
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

    fn cell_ctx<'a>(&'a self, cell_index: usize, ndofs: usize, grads: &'a [f64]) -> LocalCtx<'a> {
        let cd = &self.cell_data;
        let npts = cd.npts;
        let gdim = self.mesh.geometry_dim();
        let tdim = self.mesh.topology_dim();
        LocalCtx {
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
        cell_index: usize,
        ndofs: usize,
        grads: &'a mut [f64],
    ) -> LocalCtx<'a> {
        self.populate_cell_grads(cell_index, ndofs, grads);
        self.cell_ctx(cell_index, ndofs, grads)
    }

    fn prepare_cell_field<'a>(
        &self,
        _ndofs: usize,
        reduced_dofs: &[Option<usize>],
        state: MatRef<'_, f64>,
        basis_grads: &[f64],
        values: &'a mut [f64],
        field_grads: &'a mut [f64],
    ) -> CellField<'a> {
        let cd = &self.cell_data;
        let npts = cd.npts;
        let gdim = self.mesh.geometry_dim();
        values[..npts].fill(0.0);
        field_grads[..gdim * npts].fill(0.0);
        for (local_i, &reduced) in reduced_dofs.iter().enumerate() {
            let coefficient = reduced.map_or(0.0, |i| state[(i, 0)]);
            for q in 0..npts {
                values[q] += coefficient * cd.reference_values[local_i * npts + q];
                for gd in 0..gdim {
                    field_grads[gd * npts + q] +=
                        coefficient * basis_grads[(local_i * gdim + gd) * npts + q];
                }
            }
        }
        CellField {
            npts,
            gdim,
            values: &values[..npts],
            grads: &field_grads[..gdim * npts],
        }
    }

    /// Assemble the positive weak spatial residual `R(u)` of a state-aware
    /// kernel. Semi-discrete systems apply the lumped inverse mass separately.
    pub fn assemble_residual<K: ResidualKernel>(&self, kernel: &K, state: MatRef<f64>) -> Vec<f64> {
        assert_eq!(state.nrows(), self.reduced_size(), "state size mismatch");
        assert_eq!(
            state.ncols(),
            1,
            "residual assembly requires one state column"
        );
        let gdim = self.mesh.geometry_dim();
        let cd = &self.cell_data;
        let mut basis_grads = vec![0.0; cd.ndofs * gdim * cd.npts];
        let mut field_values = vec![0.0; cd.npts];
        let mut field_grads = vec![0.0; gdim * cd.npts];
        let mut local = vec![0.0; cd.ndofs];
        let mut residual = vec![0.0; self.reduced_size()];

        for (c, (dofs, reduced_dofs)) in self
            .cell_dofs
            .iter()
            .zip(&self.cell_reduced_dofs)
            .enumerate()
        {
            let ndofs = dofs.len();
            self.populate_cell_grads(c, ndofs, &mut basis_grads[..ndofs * gdim * cd.npts]);
            let field = self.prepare_cell_field(
                ndofs,
                reduced_dofs,
                state,
                &basis_grads[..ndofs * gdim * cd.npts],
                &mut field_values,
                &mut field_grads,
            );
            let ctx = self.cell_ctx(c, ndofs, &basis_grads[..ndofs * gdim * cd.npts]);
            kernel.assemble_local_residual(&ctx, &field, &mut local[..ndofs]);
            for (local_i, &reduced_i) in reduced_dofs.iter().enumerate() {
                if let Some(reduced_i) = reduced_i {
                    residual[reduced_i] += local[local_i];
                }
            }
        }
        residual
    }

    /// Assemble `dR/du` for a state-aware kernel at `state`.
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
        let gdim = self.mesh.geometry_dim();
        let cd = &self.cell_data;
        let mut basis_grads = vec![0.0; cd.ndofs * gdim * cd.npts];
        let mut field_values = vec![0.0; cd.npts];
        let mut field_grads = vec![0.0; gdim * cd.npts];
        let mut local = vec![0.0; cd.ndofs * cd.ndofs];
        let mut triplets = Vec::with_capacity(self.cell_dofs.len() * cd.ndofs * cd.ndofs);

        for (c, (dofs, reduced_dofs)) in self
            .cell_dofs
            .iter()
            .zip(&self.cell_reduced_dofs)
            .enumerate()
        {
            let ndofs = dofs.len();
            self.populate_cell_grads(c, ndofs, &mut basis_grads[..ndofs * gdim * cd.npts]);
            let field = self.prepare_cell_field(
                ndofs,
                reduced_dofs,
                state,
                &basis_grads[..ndofs * gdim * cd.npts],
                &mut field_values,
                &mut field_grads,
            );
            let ctx = self.cell_ctx(c, ndofs, &basis_grads[..ndofs * gdim * cd.npts]);
            kernel.assemble_local_jacobian(&ctx, &field, &mut local[..ndofs * ndofs]);
            for (ti, &reduced_i) in reduced_dofs.iter().enumerate() {
                let Some(reduced_i) = reduced_i else {
                    continue;
                };
                for (si, &reduced_j) in reduced_dofs.iter().enumerate() {
                    let Some(reduced_j) = reduced_j else {
                        continue;
                    };
                    let value = local[ti * ndofs + si];
                    if value != 0.0 {
                        triplets.push(Triplet::new(reduced_i, reduced_j, value));
                    }
                }
            }
        }
        let n = self.reduced_size();
        SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
    }

    /// Apply `dR/du(state)` to one or more direction columns without building
    /// a global sparse Jacobian. The local action is direct by default.
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
        let gdim = self.mesh.geometry_dim();
        let cd = &self.cell_data;
        let mut basis_grads = vec![0.0; cd.ndofs * gdim * cd.npts];
        let mut field_values = vec![0.0; cd.npts];
        let mut field_grads = vec![0.0; gdim * cd.npts];
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
            self.populate_cell_grads(c, ndofs, &mut basis_grads[..ndofs * gdim * cd.npts]);
            let field = self.prepare_cell_field(
                ndofs,
                reduced_dofs,
                state,
                &basis_grads[..ndofs * gdim * cd.npts],
                &mut field_values,
                &mut field_grads,
            );
            let ctx = self.cell_ctx(c, ndofs, &basis_grads[..ndofs * gdim * cd.npts]);
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

    /// Assemble a reduced sparse matrix using `kernel.assemble_local` per
    /// quadrilateral. Periodic DOFs are combined and eliminated DOFs omitted
    /// while scattering, so no full-size matrix is exposed.
    pub fn assemble_bilinear<K: BilinearForm>(&self, kernel: &K) -> SparseColMat<usize, f64> {
        let gdim = self.mesh.geometry_dim();
        let cd = &self.cell_data;
        let max_ndofs = cd.ndofs;

        // Per-call reused scratch (sized once, reused for every cell).
        let mut grads_buf = vec![0.0_f64; max_ndofs * gdim * cd.npts];
        let mut local_mat = vec![0.0_f64; max_ndofs * max_ndofs];
        let mut triplets = Vec::with_capacity(self.cell_dofs.len() * cd.ndofs * cd.ndofs);

        let npts = cd.npts;
        grads_buf.resize(cd.ndofs * gdim * npts, 0.0);
        for (c, (dofs, reduced_dofs)) in self
            .cell_dofs
            .iter()
            .zip(&self.cell_reduced_dofs)
            .enumerate()
        {
            let ndofs = dofs.len();
            debug_assert!(ndofs == cd.ndofs, "ndofs mismatch: cell vs CellData");

            let ctx = self.prepare_cell_ctx(c, ndofs, &mut grads_buf[..ndofs * gdim * npts]);
            {
                let mat_slice = &mut local_mat[..ndofs * ndofs];
                mat_slice.fill(0.0);
                kernel.assemble_local(&ctx, mat_slice);
            }

            // Scatter directly to reduced indices.
            for (ti, &reduced_i) in reduced_dofs.iter().enumerate() {
                let Some(reduced_i) = reduced_i else {
                    continue;
                };
                for (si, &reduced_j) in reduced_dofs.iter().enumerate() {
                    let Some(reduced_j) = reduced_j else {
                        continue;
                    };
                    let entry = local_mat[ti * ndofs + si];
                    if entry.abs() > 1e-12 {
                        triplets.push(Triplet::new(reduced_i, reduced_j, entry));
                    }
                }
            }
        }
        let n = self.reduced_size();
        SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
    }

    /// Assemble a reduced volume RHS from a user-defined `LinearForm`.
    pub fn assemble_linear<K: LinearForm>(&self, kernel: &K) -> Vec<f64> {
        let gdim = self.mesh.geometry_dim();
        let cd = &self.cell_data;
        let mut grads_buf = vec![0.0_f64; cd.ndofs * gdim * cd.npts];
        let mut local_rhs = vec![0.0_f64; cd.ndofs];
        let mut rhs = vec![0.0_f64; self.reduced_size()];

        for (c, (dofs, reduced_dofs)) in self
            .cell_dofs
            .iter()
            .zip(&self.cell_reduced_dofs)
            .enumerate()
        {
            let ndofs = dofs.len();
            let npts = cd.npts;
            let ctx = self.prepare_cell_ctx(c, ndofs, &mut grads_buf[..ndofs * gdim * npts]);
            kernel.assemble_local_rhs(&ctx, &mut local_rhs[..ndofs]);
            for (local_i, &reduced_i) in reduced_dofs.iter().enumerate() {
                if let Some(reduced_i) = reduced_i {
                    rhs[reduced_i] += local_rhs[local_i];
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
            |full| self.target_dof(full),
            select,
        )
    }
}
