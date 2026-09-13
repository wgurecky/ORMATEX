//! Core `SEM2DProblem` type, construction, and cell helpers.
use crate::common::{
    build_quad_state_boundary_cache, cell_ctx,
    interpolate_cell_state, CellData, CellState, ElementRestriction,
    FieldDofLayout, LocalCtx, QuadStateBoundaryCache, ReducedDofMap,
    TensorCtx, TensorProductData,
};
use crate::fields::{FieldRegistry, FieldSelection, FieldValues};
use crate::kernels::common::ResidualKernel;
use crate::mesh::MeshMetadata;
use faer::prelude::*;

use ndelement::{
    ciarlet::{LagrangeElementFamily, LagrangeVariant},
    traits::{ElementFamily, FiniteElement, MappedFiniteElement},
    types::{Continuity, ReferenceCellType},
};
use ndfunctionspace::{traits::FunctionSpace, FunctionSpaceImpl};
use ndmesh::traits::{Entity, Geometry, GeometryMap, Mesh, Point, Topology};
use quadraturerules::{single_integral_quadrature, Domain, QuadratureRule};
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
    /// Eliminate explicitly listed full-space DOFs and prescribe their values.
    /// This is useful when a boundary value changes at a corner.
    DirichletValues { values: Vec<(usize, f64)> },
    /// Apply one reduction policy per scalar field. The number of policies
    /// must match the kernel field count during system assembly.
    FieldSpecific { reductions: Vec<DofReduction2D> },
}

struct DisjointSet {
    pub(crate) parent: Vec<usize>,
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
        DofReduction2D::DirichletValues { values } => {
            ReducedDofMap::from_dirichlet_values(n, values.iter().copied())
        }
        DofReduction2D::FieldSpecific { .. } => {
            panic!("field-specific reductions must be passed at the outer level")
        }
    }
}

/// 2D GLL spectral-element problem on quadrilateral meshes.
pub struct SEM2DProblem<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> {
    pub(crate) mesh: M,
    pub(crate) fields: FieldRegistry,
    pub(crate) family: LagrangeElementFamily<f64>,
    pub(crate) p: usize,
    pub(crate) cell_data: CellData,
    pub(crate) cell_reduced_dofs: Vec<Vec<Vec<Option<usize>>>>,
    pub(crate) cell_prescribed_values: Vec<Vec<Vec<Option<f64>>>>,
    pub(crate) restriction: ElementRestriction,
    pub(crate) dof_map: ReducedDofMap,
    pub(crate) field_dof_maps: Vec<ReducedDofMap>,
    /// (x, y) position of each full GLL nodal DOF.
    pub(crate) dof_xy: Vec<(f64, f64)>,
    pub(crate) metadata: MeshMetadata,
    pub(crate) state_boundary_cache: QuadStateBoundaryCache,
}


impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> SEM2DProblem<M> {
    /// Build a 2D GLL spectral-element problem on a quadrilateral mesh.
    ///
    /// Inputs: `mesh` (tdim = gdim = 2, quadrilaterals only), `p >= 1` (tensor
    /// GLL basis of degree `p` per direction), `fields` in system-vector
    /// order, and a boundary-condition `bc` (none/periodic/Dirichlet,
    /// optionally per field). Precomputes reference tables, Jacobians, the
    /// 1D differentiation matrix for [`tensor`](super::tensor) sum
    /// factorization, and the state-boundary cache used by [`weak`](super::weak).
    pub fn new(mesh: M, p: usize, fields: FieldRegistry, bc: DofReduction2D) -> Self {
        Self::new_with_metadata(mesh, p, fields, bc, MeshMetadata::default())
    }

    /// Build a problem with named scalar fields and mesh metadata.
    ///
    /// Same as [`new`](Self::new) plus per-cell/per-facet `metadata`
    /// (physical regions used by boundary selection).
    pub fn new_with_metadata(
        mesh: M,
        p: usize,
        fields: FieldRegistry,
        bc: DofReduction2D,
        metadata: MeshMetadata,
    ) -> Self {
        assert!(p >= 1, "polynomial degree p must be >= 1");
        let family =
            LagrangeElementFamily::<f64>::new(p, Continuity::Standard, LagrangeVariant::Gll);
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
        let mut jinv_cache = vec![0.0_f64; ncells * npts * tdim * gdim];
        let mut jdets_cache = vec![0.0_f64; ncells * npts];
        let mut wdet_cache = vec![0.0_f64; ncells * npts];
        let mut cell_sizes = vec![0.0_f64; ncells];
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
                        let value = *jinv_scratch.get([td, gd, q]).unwrap();
                        jinv_cache[(c * npts + q) * tdim * gdim + td * gdim + gd] = value;
                    }
                }
            }
            for q in 0..npts {
                jdets_cache[c * npts + q] = jdet_scratch[q];
                wdet_cache[c * npts + q] = wts[q] * jdet_scratch[q];
            }
            let cell_measure: f64 = wts
                .iter()
                .zip(&jdets_cache[c * npts..(c + 1) * npts])
                .map(|(&weight, &jdet)| weight * jdet)
                .sum();
            cell_sizes[c] = cell_measure.sqrt();
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
        for dof in 0..ndofs {
            for q in 0..npts {
                let value = reference_values[dof * npts + q];
                let expected = if nodal_quadrature[dof] == q { 1.0 } else { 0.0 };
                assert!(
                    (value - expected).abs() < 1e-10,
                    "GLL basis is not collocated at quadrature point {q} for local DOF {dof}"
                );
            }
        }

        let interval_element = family.element(ReferenceCellType::Interval);
        let mut interval_pts = rlst_dynamic_array!(f64, [1, n1d]);
        for i in 0..n1d {
            *interval_pts.get_mut([0, i]).unwrap() = xs_1d[i];
        }
        let mut interval_table =
            DynArray::<f64, 4>::from_shape(interval_element.tabulate_array_shape(1, n1d));
        interval_element.tabulate(&interval_pts, 1, &mut interval_table);
        let mut differentiation = vec![0.0; n1d * n1d];
        let mut interval_nodal_quadrature = vec![0; n1d];
        for local in 0..n1d {
            interval_nodal_quadrature[local] = (0..n1d)
                .find(|&q| *interval_table.get([0, q, local, 0]).unwrap() > 1.0 - 1e-12)
                .expect("interval GLL basis dof has no nodal quadrature point");
        }
        for q in 0..n1d {
            for local_dof in 0..n1d {
                differentiation[q * n1d + interval_nodal_quadrature[local_dof]] =
                    *interval_table.get([1, q, local_dof, 0]).unwrap();
            }
        }
        let mut q_to_local = vec![usize::MAX; npts];
        for (local, &q) in nodal_quadrature.iter().enumerate() {
            assert_eq!(q_to_local[q], usize::MAX, "duplicate GLL node mapping");
            q_to_local[q] = local;
        }
        assert!(q_to_local.iter().all(|&local| local != usize::MAX));
        let tensor = TensorProductData {
            n1d,
            differentiation,
            q_to_local,
        };
        let cell_data = CellData {
            wts,
            npts,
            table,
            reference_values,
            nodal_quadrature,
            jinv_cache,
            jdets_cache,
            wdet_cache,
            cell_sizes,
            physical_points_cache,
            ndofs,
            tensor: Some(tensor),
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
        let reductions = match bc {
            DofReduction2D::FieldSpecific { reductions } => {
                assert!(
                    !reductions.is_empty(),
                    "field-specific reductions cannot be empty"
                );
                assert_eq!(
                    reductions.len(),
                    fields.len(),
                    "field-specific reduction count must match problem field count"
                );
                reductions
            }
            reduction => vec![reduction],
        };
        let field_dof_maps: Vec<_> = reductions
            .into_iter()
            .map(|reduction| build_dof_map_2d(n, &dof_xy, &reduction, &facet_data))
            .collect();
        let dof_map = field_dof_maps[0].clone();
        let cell_dofs: Vec<Vec<usize>> = (0..ncells)
            .map(|cell| {
                space
                    .entity_closure_dofs(ReferenceCellType::Quadrilateral, cell)
                    .unwrap()
                    .to_vec()
            })
            .collect();
        let mut cell_reduced_dofs = Vec::with_capacity(field_dof_maps.len());
        let mut cell_prescribed_values = Vec::with_capacity(field_dof_maps.len());
        for map in &field_dof_maps {
            let (reduced, prescribed) = map.map_cells(&cell_dofs);
            cell_reduced_dofs.push(reduced);
            cell_prescribed_values.push(prescribed);
        }
        let field_sizes: Vec<_> = if field_dof_maps.len() == 1 {
            vec![field_dof_maps[0].reduced_size(); fields.len()]
        } else {
            field_dof_maps
                .iter()
                .map(ReducedDofMap::reduced_size)
                .collect()
        };
        let restriction =
            ElementRestriction::new(&cell_reduced_dofs, &cell_prescribed_values, &field_sizes);
        let state_boundary_cache = build_quad_state_boundary_cache(&mesh, &family, p, &metadata);
        Self {
            mesh,
            fields,
            family,
            p,
            cell_data,
            cell_reduced_dofs,
            cell_prescribed_values,
            restriction,
            dof_map,
            field_dof_maps,
            dof_xy,
            metadata,
            state_boundary_cache,
        }
    }

    /// O(1) full → `Option<reduced>` DOF lookup.
    pub fn target_dof(&self, full: usize) -> Option<usize> {
        self.dof_map.target(full)
    }

    /// Return the problem's ordered field registry.
    pub fn fields(&self) -> &FieldRegistry {
        &self.fields
    }

    /// Return the numeric ID for a named field.
    pub fn field_id(&self, name: &str) -> Option<usize> {
        self.fields.id(name)
    }

    pub fn field_names(&self) -> &[String] {
        self.fields.names()
    }

    pub fn target_field_dof(&self, field: usize, full: usize) -> Option<usize> {
        assert!(field < self.fields.len(), "field index out of range");
        self.field_dof_maps
            .get(if self.field_dof_maps.len() == 1 {
                0
            } else {
                field
            })
            .expect("field index out of range")
            .target(full)
    }

    pub(crate) fn prescribed_field_dof(&self, field: usize, full: usize) -> Option<f64> {
        assert!(field < self.fields.len(), "field index out of range");
        self.field_dof_maps
            .get(if self.field_dof_maps.len() == 1 {
                0
            } else {
                field
            })
            .expect("field index out of range")
            .prescribed(full)
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

    pub fn system_size(&self) -> usize {
        self.field_layout().total_size
    }

    pub fn field_offset(&self, field: usize) -> usize {
        self.field_layout().offsets[field]
    }

    pub fn field_reduced_size(&self, field: usize) -> usize {
        assert!(field < self.fields.len(), "field index out of range");
        self.field_dof_maps
            .get(if self.field_dof_maps.len() == 1 {
                0
            } else {
                field
            })
            .expect("field index out of range")
            .reduced_size()
    }

    pub fn field_dof_positions(&self, field: usize) -> Vec<(f64, f64)> {
        assert!(field < self.fields.len(), "field index out of range");
        let map = self
            .field_dof_maps
            .get(if self.field_dof_maps.len() == 1 {
                0
            } else {
                field
            })
            .expect("field index out of range");
        let mut out = vec![(f64::NAN, f64::NAN); map.reduced_size()];
        for full in 0..self.dof_map.full_size() {
            if let Some(reduced) = map.target(full) {
                let xy = self.dof_xy[full];
                if out[reduced].0.is_nan() {
                    out[reduced] = xy;
                }
            }
        }
        out
    }

    /// Get one named field's reduced values and representative positions.
    pub fn field_values(
        &self,
        name: &str,
        state: MatRef<'_, f64>,
    ) -> Option<FieldValues<(f64, f64)>> {
        let field = self.field_id(name)?;
        assert_eq!(state.nrows(), self.system_size(), "state size mismatch");
        assert_eq!(
            state.ncols(),
            1,
            "field extraction requires one state column"
        );
        let positions = self.field_dof_positions(field);
        let offset = self.field_offset(field);
        let values = (0..positions.len())
            .map(|local| state[(offset + local, 0)])
            .collect();
        Some(FieldValues { positions, values })
    }

    pub(crate) fn field_layout(&self) -> FieldDofLayout {
        FieldDofLayout::new(
            &self.dof_map,
            (self.field_dof_maps.len() > 1).then_some(&self.field_dof_maps),
            self.fields.len(),
        )
    }

    pub(crate) fn validate_fields(
        &self,
        nfields: usize,
        names: Option<Vec<String>>,
        context: &str,
    ) {
        assert_eq!(
            nfields,
            self.fields.len(),
            "{context} field count does not match the SEM problem"
        );
        if let Some(names) = names {
            assert_eq!(
                names,
                self.fields.names(),
                "{context} field names/order do not match the SEM problem"
            );
        }
    }

    pub(crate) fn cell_field_maps(
        &self,
        cell: usize,
        nfields: usize,
    ) -> (Vec<&[Option<usize>]>, Vec<&[Option<f64>]>) {
        if self.cell_reduced_dofs.len() == 1 {
            (
                vec![&self.cell_reduced_dofs[0][cell]; nfields],
                vec![&self.cell_prescribed_values[0][cell]; nfields],
            )
        } else {
            assert_eq!(self.cell_reduced_dofs.len(), nfields);
            (
                (0..nfields)
                    .map(|field| self.cell_reduced_dofs[field][cell].as_slice())
                    .collect(),
                (0..nfields)
                    .map(|field| self.cell_prescribed_values[field][cell].as_slice())
                    .collect(),
            )
        }
    }

    pub(crate) fn cell_field_maps_for(
        &self,
        cell: usize,
        fields: &[usize],
    ) -> (Vec<&[Option<usize>]>, Vec<&[Option<f64>]>) {
        if self.cell_reduced_dofs.len() == 1 {
            (
                vec![&self.cell_reduced_dofs[0][cell]; fields.len()],
                vec![&self.cell_prescribed_values[0][cell]; fields.len()],
            )
        } else {
            (
                fields
                    .iter()
                    .map(|&field| self.cell_reduced_dofs[field][cell].as_slice())
                    .collect(),
                fields
                    .iter()
                    .map(|&field| self.cell_prescribed_values[field][cell].as_slice())
                    .collect(),
            )
        }
    }

    pub(crate) fn resolve_residual_selection<K: ResidualKernel>(
        &self,
        kernel: &K,
        context: &str,
    ) -> FieldSelection {
        self.fields.resolve_selection(
            kernel.input_nfields(),
            kernel.input_field_names(),
            kernel.output_nfields(),
            kernel.output_field_names(),
            context,
        )
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

    pub(crate) fn populate_cell_grads(&self, cell_index: usize, ndofs: usize, grads: &mut [f64]) {
        let cd = &self.cell_data;
        let npts = cd.npts;
        let gdim = self.mesh.geometry_dim();
        let tdim = self.mesh.topology_dim();
        for dof_i in 0..ndofs {
            for q in 0..npts {
                for gd in 0..gdim {
                    let mut acc = 0.0;
                    for td in 0..tdim {
                        acc += cd.jinv_cache
                            [(cell_index * npts + q) * tdim * gdim + td * gdim + gd]
                            * *cd.table.get([1 + td, q, dof_i, 0]).unwrap();
                    }
                    grads[(dof_i * gdim + gd) * npts + q] = acc;
                }
            }
        }
    }

    pub(crate) fn cell_ctx<'a>(
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

    pub(crate) fn tensor_ctx<'a>(&'a self, time: f64, cell_index: usize) -> TensorCtx<'a> {
        let cd = &self.cell_data;
        let tensor = cd
            .tensor
            .as_ref()
            .expect("tensor-product context requires tensor data");
        let jinv_start = cell_index * cd.npts * 4;
        TensorCtx {
            time,
            cell: self.metadata.cell(cell_index),
            n1d: tensor.n1d,
            npts: cd.npts,
            wts: &cd.wts,
            jdets: &cd.jdets_cache[cell_index * cd.npts..(cell_index + 1) * cd.npts],
            wdet: &cd.wdet_cache[cell_index * cd.npts..(cell_index + 1) * cd.npts],
            points: &cd.physical_points_cache
                [cell_index * cd.npts * 2..(cell_index + 1) * cd.npts * 2],
            differentiation: &tensor.differentiation,
            q_to_local: &tensor.q_to_local,
            jinv: &cd.jinv_cache[jinv_start..jinv_start + cd.npts * 4],
            cell_size: cd.cell_sizes[cell_index],
        }
    }

    pub(crate) fn prepare_cell_ctx<'a>(
        &'a self,
        time: f64,
        cell_index: usize,
        ndofs: usize,
        grads: &'a mut [f64],
    ) -> LocalCtx<'a> {
        self.populate_cell_grads(cell_index, ndofs, grads);
        self.cell_ctx(time, cell_index, ndofs, grads)
    }

    pub(crate) fn prepare_cell_state<'a>(
        &self,
        nfields: usize,
        field_reduced_dofs: &[&[Option<usize>]],
        field_prescribed_values: &[&[Option<f64>]],
        field_offsets: &[usize],
        state: MatRef<'_, f64>,
        basis_grads: &[f64],
        values: &'a mut [f64],
        field_grads: &'a mut [f64],
    ) -> CellState<'a> {
        interpolate_cell_state(
            &self.cell_data,
            self.mesh.geometry_dim(),
            nfields,
            field_reduced_dofs,
            field_prescribed_values,
            field_offsets,
            state,
            basis_grads,
            values,
            field_grads,
        )
    }
}
