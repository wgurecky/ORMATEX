//! Core `SEM1DProblem` type, construction, and cell helpers.
use crate::common::{
    cell_ctx, interpolate_cell_state, CellData, CellState, ElementRestriction,
    FieldDofLayout, LocalCtx, ReducedDofMap, TensorCtx,
    TensorProductData,
};
use crate::fields::{FieldRegistry, FieldSelection, FieldValues};
use crate::kernels::kernel_common::ResidualKernel;
use crate::mesh::{MeshMetadata, PhysicalRegion};
use faer::prelude::*;

use ndelement::{
    ciarlet::{LagrangeElementFamily, LagrangeVariant},
    traits::{ElementFamily, FiniteElement, MappedFiniteElement},
    types::{Continuity, ReferenceCellType},
};
use ndfunctionspace::{traits::FunctionSpace, FunctionSpaceImpl};
use ndmesh::traits::{Entity, GeometryMap, Mesh, Topology};
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
    /// Eliminate every closure DOF on selected endpoint facets and prescribe
    /// its value. Each pair is `(facet_index, prescribed_value)`.
    Dirichlet {
        facets: Vec<(usize, f64)>,
    },
    /// Apply one reduction policy per scalar field. The number of policies
    /// must match the kernel field count during system assembly.
    FieldSpecific {
        reductions: Vec<DofReduction1D>,
    },
}

/// Geometry used to select a 1D natural-boundary kernel.
#[derive(Clone, Copy, Debug)]
pub struct BoundaryPoint {
    pub index: usize,
    pub coordinate: f64,
    pub normal: f64,
    pub physical_region: Option<PhysicalRegion>,
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
        DofReduction1D::FieldSpecific { .. } => {
            panic!("field-specific reductions must be passed at the outer level")
        }
    }
}

/// 1D GLL spectral-element problem on interval meshes.
pub struct SEM1DProblem<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> {
    pub(crate) mesh: M,
    pub(crate) fields: FieldRegistry,
    pub(crate) family: LagrangeElementFamily<f64>,
    pub(crate) cell_data: CellData,
    pub(crate) cell_reduced_dofs: Vec<Vec<Vec<Option<usize>>>>,
    pub(crate) cell_prescribed_values: Vec<Vec<Vec<Option<f64>>>>,
    pub(crate) restriction: ElementRestriction,
    pub(crate) dof_map: ReducedDofMap,
    pub(crate) field_dof_maps: Vec<ReducedDofMap>,
    pub(crate) dof_x: Vec<f64>,
    pub(crate) metadata: MeshMetadata,
}


impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> SEM1DProblem<M> {
    /// Build a 1D GLL spectral-element problem on an interval mesh.
    ///
    /// Inputs: `mesh` (tdim = gdim = 1, intervals only), `p >= 1` (GLL degree;
    /// quadrature collocated so one node per basis DOF), `fields` in
    /// system-vector order, and a DOF `reduction` (none/periodic/Dirichlet,
    /// optionally per field). Precomputes reference tables, Jacobians, and
    /// the tensor-product differentiation matrix shared by [`weak`](super::weak)
    /// and [`tensor`](super::tensor) assembly.
    pub fn new(mesh: M, p: usize, fields: FieldRegistry, reduction: DofReduction1D) -> Self {
        Self::new_with_metadata(mesh, p, fields, reduction, MeshMetadata::default())
    }

    /// Build a problem with named scalar fields and mesh metadata.
    ///
    /// Same as [`new`](Self::new) plus per-cell/per-facet `metadata`
    /// (physical regions used by boundary selection).
    pub fn new_with_metadata(
        mesh: M,
        p: usize,
        fields: FieldRegistry,
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
            LagrangeElementFamily::<f64>::new(p, Continuity::Standard, LagrangeVariant::Gll);
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
        let mut jinv_cache = vec![0.0; ncells * npts];
        let mut jdets_cache = vec![0.0; ncells * npts];
        let mut wdet_cache = vec![0.0; ncells * npts];
        let mut cell_sizes = vec![0.0; ncells];
        let mut physical_points_cache = vec![0.0; ncells * npts];
        let mut dof_x = vec![f64::NAN; n];
        let mut physical_pts = rlst_dynamic_array!(f64, [1, npts]);

        for cell in mesh.entity_iter(ReferenceCellType::Interval) {
            let c = cell.local_index();
            gmap.jacobians_inverses_dets(c, &mut jac_scratch, &mut jinv_scratch, &mut jdet_scratch);
            gmap.physical_points(c, &mut physical_pts);
            for q in 0..npts {
                jinv_cache[c * npts + q] = *jinv_scratch.get([0, 0, q]).unwrap();
                jdets_cache[c * npts + q] = jdet_scratch[q];
                wdet_cache[c * npts + q] = wts[q] * jdet_scratch[q];
                physical_points_cache[c * npts + q] = *physical_pts.get([0, q]).unwrap();
            }
            cell_sizes[c] = wdet_cache[c * npts..(c + 1) * npts]
                .iter()
                .sum::<f64>()
                .sqrt();
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
        assert_eq!(
            element.dim(),
            npts,
            "1D tensor assembly requires one GLL quadrature point per basis DOF"
        );
        let mut differentiation = vec![0.0; npts * npts];
        let mut q_to_local = vec![usize::MAX; npts];
        for (local, &q) in nodal_quadrature.iter().enumerate() {
            assert_eq!(q_to_local[q], usize::MAX, "duplicate GLL node mapping");
            q_to_local[q] = local;
            for derivative_q in 0..npts {
                differentiation[derivative_q * npts + q] =
                    *table.get([1, derivative_q, local, 0]).unwrap();
            }
        }
        assert!(q_to_local.iter().all(|&local| local != usize::MAX));
        let cell_data = CellData {
            wts,
            npts,
            ndofs: element.dim(),
            table,
            reference_values,
            nodal_quadrature,
            jinv_cache,
            jdets_cache,
            wdet_cache,
            cell_sizes,
            physical_points_cache,
            tensor: Some(TensorProductData {
                n1d: npts,
                differentiation,
                q_to_local,
            }),
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

        let reductions = match reduction {
            DofReduction1D::FieldSpecific { reductions } => {
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
            .map(|reduction| build_dof_map_1d(n, reduction, &boundary_dof))
            .collect();
        let dof_map = field_dof_maps[0].clone();
        let cell_dofs: Vec<Vec<usize>> = (0..ncells)
            .map(|cell| {
                space
                    .entity_closure_dofs(ReferenceCellType::Interval, cell)
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

        Self {
            mesh,
            fields,
            family,
            cell_data,
            cell_reduced_dofs,
            cell_prescribed_values,
            restriction,
            dof_map,
            field_dof_maps,
            dof_x,
            metadata,
        }
    }

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

    pub fn field_dof_positions(&self, field: usize) -> Vec<f64> {
        assert!(field < self.fields.len(), "field index out of range");
        let map = self
            .field_dof_maps
            .get(if self.field_dof_maps.len() == 1 {
                0
            } else {
                field
            })
            .expect("field index out of range");
        let mut out = vec![f64::NAN; map.reduced_size()];
        for (full, &x) in self.dof_x.iter().enumerate() {
            if let Some(reduced) = map.target(full) {
                if out[reduced].is_nan() {
                    out[reduced] = x;
                }
            }
        }
        out
    }

    /// Get one named field's reduced values and representative positions.
    pub fn field_values(&self, name: &str, state: MatRef<'_, f64>) -> Option<FieldValues<f64>> {
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

    pub(crate) fn populate_cell_grads(&self, cell_index: usize, ndofs: usize, grads: &mut [f64]) {
        let cd = &self.cell_data;
        for dof_i in 0..ndofs {
            for q in 0..cd.npts {
                grads[dof_i * cd.npts + q] = cd.jinv_cache[cell_index * cd.npts + q]
                    * *cd.table.get([1, q, dof_i, 0]).unwrap();
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
            1,
            1,
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
        TensorCtx {
            time,
            cell: self.metadata.cell(cell_index),
            n1d: tensor.n1d,
            npts: cd.npts,
            wts: &cd.wts,
            jdets: &cd.jdets_cache[cell_index * cd.npts..(cell_index + 1) * cd.npts],
            wdet: &cd.wdet_cache[cell_index * cd.npts..(cell_index + 1) * cd.npts],
            points: &cd.physical_points_cache[cell_index * cd.npts..(cell_index + 1) * cd.npts],
            differentiation: &tensor.differentiation,
            q_to_local: &tensor.q_to_local,
            jinv: &cd.jinv_cache[cell_index * cd.npts..(cell_index + 1) * cd.npts],
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
            1,
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
