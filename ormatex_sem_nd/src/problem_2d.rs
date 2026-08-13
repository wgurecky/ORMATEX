use crate::common::{
    add_dirichlet_rhs_correction, assemble_lumped_mass, assemble_quad_boundaries, cell_ctx,
    interpolate_cell_state, push_local_matrix_triplets, scatter_local_vector,
    BoundaryContributions, BoundaryFacet, CellData, CellState, LocalCtx, ReducedDofMap,
    CELL_BATCH_SIZE,
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

/// DOF reduction selector for a 2D quadrilateral problem.
#[derive(Clone, Debug)]
pub enum DofReduction2D {
    /// Identify explicitly paired boundary facets related by translation.
    ///
    /// Each pair contains ndmesh `Interval` local indices. Every paired facet
    /// must have the same number of closure DOFs and differ only by one
    /// translation; orientation is inferred from its endpoint coordinates.
    /// Pair every conforming facet segment, not just each outer boundary side.
    Periodic {
        facet_pairs: Vec<[usize; 2]>,
        tolerance: f64,
    },
    /// Retain all dofs.
    None,
    /// Eliminate every closure DOF on the listed boundary facets and prescribe
    /// its value. Each pair is `(facet_index, prescribed_value)`.
    /// This includes high-order edge DOFs.
    Dirichlet { facets: Vec<(usize, f64)> },
}

struct DisjointSet {
    parent: Vec<usize>,
}

impl DisjointSet {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
        }
    }

    fn find(&mut self, item: usize) -> usize {
        if self.parent[item] != item {
            self.parent[item] = self.find(self.parent[item]);
        }
        self.parent[item]
    }

    fn union(&mut self, a: usize, b: usize) {
        let a = self.find(a);
        let b = self.find(b);
        if a != b {
            self.parent[a.max(b)] = a.min(b);
        }
    }
}

fn build_dof_map_2d<F>(
    n: usize,
    dof_xy: &[(f64, f64)],
    bc: &DofReduction2D,
    facet_data: F,
) -> ReducedDofMap
where
    F: Fn(usize) -> ([[f64; 2]; 2], Vec<usize>),
{
    match bc {
        DofReduction2D::Periodic {
            facet_pairs,
            tolerance,
        } => {
            assert!(
                tolerance.is_finite() && *tolerance > 0.0,
                "periodic tolerance must be finite and positive"
            );
            assert!(
                !facet_pairs.is_empty(),
                "periodic reduction requires at least one facet pair"
            );
            let mut paired_facets = std::collections::HashSet::new();
            let mut equivalence = DisjointSet::new(n);

            for &[source_facet, target_facet] in facet_pairs {
                assert_ne!(
                    source_facet, target_facet,
                    "periodic facets must be distinct"
                );
                assert!(
                    paired_facets.insert(source_facet) && paired_facets.insert(target_facet),
                    "a periodic facet may appear in only one pair"
                );

                let (source_endpoints, source_dofs) = facet_data(source_facet);
                let (target_endpoints, target_dofs) = facet_data(target_facet);
                let distance = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
                assert!(
                    distance(source_endpoints[0], source_endpoints[1]) > *tolerance,
                    "periodic facet {source_facet} has zero length"
                );
                assert!(
                    distance(target_endpoints[0], target_endpoints[1]) > *tolerance,
                    "periodic facet {target_facet} has zero length"
                );
                let aligned = [
                    target_endpoints[0][0] - source_endpoints[0][0],
                    target_endpoints[0][1] - source_endpoints[0][1],
                ];
                let reversed = [
                    target_endpoints[1][0] - source_endpoints[0][0],
                    target_endpoints[1][1] - source_endpoints[0][1],
                ];
                let translation = if distance(
                    [
                        source_endpoints[1][0] + aligned[0],
                        source_endpoints[1][1] + aligned[1],
                    ],
                    target_endpoints[1],
                ) <= *tolerance
                {
                    aligned
                } else if distance(
                    [
                        source_endpoints[1][0] + reversed[0],
                        source_endpoints[1][1] + reversed[1],
                    ],
                    target_endpoints[0],
                ) <= *tolerance
                {
                    reversed
                } else {
                    panic!(
                        "periodic facets {source_facet} and {target_facet} are not related by a translation"
                    );
                };

                assert_eq!(
                    source_dofs.len(),
                    target_dofs.len(),
                    "periodic facets must have matching closure DOF counts"
                );
                let mut matched = vec![false; target_dofs.len()];
                for &source in &source_dofs {
                    let source_xy = dof_xy[source];
                    let expected = (source_xy.0 + translation[0], source_xy.1 + translation[1]);
                    // ponytail: p + 1 facet nodes make direct matching O(p^2); index only if p becomes large.
                    let matches: Vec<_> = target_dofs
                        .iter()
                        .enumerate()
                        .filter(|(_, &target)| {
                            let target_xy = dof_xy[target];
                            (target_xy.0 - expected.0).hypot(target_xy.1 - expected.1) <= *tolerance
                        })
                        .collect();
                    assert_eq!(
                        matches.len(),
                        1,
                        "periodic source DOF {source} must match exactly one target DOF"
                    );
                    let (target_index, &target) = matches[0];
                    assert!(
                        !matched[target_index],
                        "periodic target DOF {target} matched more than once"
                    );
                    matched[target_index] = true;
                    equivalence.union(source, target);
                }
                assert!(
                    matched.iter().all(|&used| used),
                    "every periodic target DOF must be matched"
                );
            }
            ReducedDofMap::from_representatives((0..n).map(|dof| equivalence.find(dof)).collect())
        }
        DofReduction2D::None => ReducedDofMap::identity(n),
        DofReduction2D::Dirichlet { facets } => {
            let mut prescribed = Vec::new();
            for &(facet, value) in facets {
                let (_, dofs) = facet_data(facet);
                prescribed.extend(dofs.into_iter().map(|dof| (dof, value)));
            }
            ReducedDofMap::from_dirichlet_values(n, prescribed)
        }
    }
}

/// 2D GLL spectral-element problem on quadrilateral meshes.
pub struct SEM2DProblem<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> {
    mesh: M,
    family: LagrangeElementFamily<f64>,
    p: usize,
    cell_data: CellData,
    cell_reduced_dofs: Vec<Vec<Option<usize>>>,
    cell_prescribed_values: Vec<Vec<Option<f64>>>,
    dof_map: ReducedDofMap,
    /// (x, y) position of each full GLL nodal DOF.
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

        let facet_data = |facet_index: usize| -> ([[f64; 2]; 2], Vec<usize>) {
            let facet = mesh
                .entity(ReferenceCellType::Interval, facet_index)
                .expect("facet index out of range");
            let topology = facet.topology();
            let mut cells = topology.connected_entity_iter(ReferenceCellType::Quadrilateral);
            assert!(
                cells.next().is_some() && cells.next().is_none(),
                "facet {facet_index} must be a boundary interval"
            );
            let mut vertices = topology.sub_entity_iter(ReferenceCellType::Point);
            let endpoint = |vertex| {
                let point = mesh
                    .entity(ReferenceCellType::Point, vertex)
                    .expect("facet endpoint out of range");
                let mut xy = [0.0; 2];
                point.geometry().points().next().unwrap().coords(&mut xy);
                xy
            };
            let endpoints = [
                endpoint(vertices.next().expect("facet has no first endpoint")),
                endpoint(vertices.next().expect("facet has no second endpoint")),
            ];
            assert!(
                vertices.next().is_none(),
                "facet {facet_index} must have two endpoints"
            );
            let dofs = space
                .entity_closure_dofs(ReferenceCellType::Interval, facet_index)
                .unwrap()
                .to_vec();
            (endpoints, dofs)
        };
        let dof_map = build_dof_map_2d(n, &dof_xy, &bc, facet_data);
        let cell_dofs: Vec<Vec<usize>> = (0..ncells)
            .map(|cell| {
                space
                    .entity_closure_dofs(ReferenceCellType::Quadrilateral, cell)
                    .unwrap()
                    .to_vec()
            })
            .collect();
        let (cell_reduced_dofs, cell_prescribed_values) = dof_map.map_cells(&cell_dofs);

        Self {
            mesh,
            family,
            p,
            cell_data,
            cell_reduced_dofs,
            cell_prescribed_values,
            dof_map,
            dof_xy,
            metadata,
        }
    }

    /// O(1) full -> Option<reduced> DOF lookup.
    pub fn target_dof(&self, full: usize) -> Option<usize> {
        self.dof_map.target(full)
    }

    pub fn mesh(&self) -> &M {
        &self.mesh
    }

    pub fn family(&self) -> &LagrangeElementFamily<f64> {
        &self.family
    }

    /// Number of retained reduced DOFs after boundary-condition reduction.
    pub fn reduced_size(&self) -> usize {
        self.dof_map.reduced_size()
    }

    /// Representative (x, y) position of each reduced DOF, in reduced-index order.
    pub fn dof_positions(&self) -> Vec<(f64, f64)> {
        let nr = self.reduced_size();
        let mut out = vec![(f64::NAN, f64::NAN); nr];
        for full in 0..self.dof_map.full_size() {
            if let Some(r) = self.dof_map.target(full) {
                let (x, y) = self.dof_xy[full];
                if out[r].0.is_nan() {
                    out[r] = (x, y);
                }
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
        cell_ctx(
            &self.cell_data,
            &self.metadata,
            self.mesh.topology_dim(),
            self.mesh.geometry_dim(),
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
        _ndofs: usize,
        reduced_dofs: &[Option<usize>],
        prescribed_values: &[Option<f64>],
        state: MatRef<'_, f64>,
        basis_grads: &[f64],
        values: &'a mut [f64],
        field_grads: &'a mut [f64],
    ) -> CellState<'a> {
        interpolate_cell_state(
            &self.cell_data,
            self.mesh.geometry_dim(),
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
            .cell_reduced_dofs
            .par_chunks(CELL_BATCH_SIZE)
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
                |(basis_grads, field_values, field_grads, local), (batch_index, reduced_batch)| {
                    let mut batch = vec![0.0; reduced_batch.len() * local_stride];
                    for (cell_offset, reduced_dofs) in reduced_batch.iter().enumerate() {
                        let ndofs = reduced_dofs.len();
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
                            &self.cell_prescribed_values[cell_index],
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
                let ndofs = reduced_dofs.len();
                let cell_size = nfields * ndofs;
                let local =
                    &batch[cell_offset * local_stride..cell_offset * local_stride + cell_size];
                scatter_local_vector(&mut residual, local, reduced_dofs, nfields, n);
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
            .cell_reduced_dofs
            .par_chunks(CELL_BATCH_SIZE)
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
                            &mut basis_grads[..ndofs * gdim * cd.npts],
                        );
                        let state_cell = self.prepare_cell_state(
                            nfields,
                            ndofs,
                            reduced_dofs,
                            &self.cell_prescribed_values[cell_index],
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
            .cell_reduced_dofs
            .par_chunks(CELL_BATCH_SIZE)
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
                 (batch_index, reduced_batch)| {
                    let mut batch = vec![0.0; reduced_batch.len() * action_stride];
                    for (cell_offset, reduced_dofs) in reduced_batch.iter().enumerate() {
                        let ndofs = reduced_dofs.len();
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
                            &self.cell_prescribed_values[cell_index],
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
            .cell_reduced_dofs
            .par_chunks(CELL_BATCH_SIZE)
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

                        push_local_matrix_triplets(
                            &mut triplets,
                            mat_slice,
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

        for (c, reduced_dofs) in self.cell_reduced_dofs.iter().enumerate() {
            let ndofs = reduced_dofs.len();
            let npts = cd.npts;
            let ctx = self.prepare_cell_ctx(time, c, ndofs, &mut grads_buf[..ndofs * gdim * npts]);
            kernel.assemble_local_rhs(&ctx, &mut local_rhs[..nfields * ndofs]);
            scatter_local_vector(
                &mut rhs,
                &local_rhs[..nfields * ndofs],
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
        let gdim = self.mesh.geometry_dim();
        let mut grads = vec![0.0; cd.ndofs * gdim * cd.npts];
        let local_size = nfields * cd.ndofs;
        let mut local = vec![0.0; local_size * local_size];
        for (cell_index, reduced_dofs) in self.cell_reduced_dofs.iter().enumerate() {
            let prescribed = &self.cell_prescribed_values[cell_index];
            if !prescribed.iter().any(Option::is_some) {
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
            &self.metadata,
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
            &self.metadata,
            time,
            |full| self.target_dof(full),
            select,
        )
    }
}
