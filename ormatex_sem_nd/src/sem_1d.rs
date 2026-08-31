use crate::common::{
    add_dirichlet_rhs_correction, assemble_lumped_mass, cell_ctx, interpolate_cell_state,
    interpolate_tensor_cell_coefficients, interpolate_tensor_cell_state,
    push_local_matrix_triplets, scatter_local_vector, BoundaryContributions, CellData, CellState,
    FacetCtx, FieldDofLayout, LocalCtx, ReducedDofMap, StateBoundaryContributions, TensorCtx,
    TensorProductData, CELL_BATCH_SIZE,
};
use crate::fields::{FieldRegistry, FieldValues};
use crate::jacobian::CompleteResidualOperator;
use crate::kernels::kernel_common::{
    apply_tensor_bilinear_column_1d, apply_tensor_jacobian_1d, assemble_tensor_residual_1d,
    BilinearForm, BoundaryIntegrator, LinearForm, ResidualKernel, StateBoundaryTerms,
};
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
        DofReduction1D::FieldSpecific { .. } => {
            panic!("field-specific reductions must be passed at the outer level")
        }
    }
}

/// 1D GLL spectral-element problem on interval meshes.
pub struct SEM1DProblem<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> {
    mesh: M,
    fields: FieldRegistry,
    family: LagrangeElementFamily<f64>,
    cell_data: CellData,
    cell_reduced_dofs: Vec<Vec<Vec<Option<usize>>>>,
    cell_prescribed_values: Vec<Vec<Vec<Option<f64>>>>,
    dof_map: ReducedDofMap,
    field_dof_maps: Vec<ReducedDofMap>,
    dof_x: Vec<f64>,
    metadata: MeshMetadata,
}

pub struct SEM1DResidualOperator<'p, 'k, 'b, M, K>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
{
    problem: &'p SEM1DProblem<M>,
    kernel: &'k K,
    time: f64,
    terms: Option<&'b StateBoundaryTerms>,
}

impl<'p, 'k, 'b, M, K> SEM1DResidualOperator<'p, 'k, 'b, M, K>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
    K: ResidualKernel + Sync,
{
    pub fn system_size(&self) -> usize {
        self.problem.system_size()
    }

    pub fn residual(&self, state: MatRef<f64>) -> Vec<f64> {
        match &self.terms {
            Some(terms) => {
                self.problem
                    .assemble_complete_residual(self.time, self.kernel, state, terms)
            }
            None => self
                .problem
                .assemble_residual(self.time, self.kernel, state),
        }
    }

    pub fn assemble_jacobian(&self, state: MatRef<f64>) -> SparseColMat<usize, f64> {
        match &self.terms {
            Some(terms) => {
                self.problem
                    .assemble_complete_jacobian(self.time, self.kernel, state, terms)
            }
            None => self
                .problem
                .assemble_residual_jacobian(self.time, self.kernel, state),
        }
    }

    pub fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        match &self.terms {
            Some(terms) => self.problem.apply_complete_jacobian(
                self.time,
                self.kernel,
                state,
                direction,
                terms,
            ),
            None => self
                .problem
                .apply_jacobian(self.time, self.kernel, state, direction),
        }
    }

    pub fn at_time(mut self, time: f64) -> Self {
        self.time = time;
        self
    }

    pub fn with_state_boundary<'terms>(
        self,
        terms: &'terms StateBoundaryTerms,
    ) -> SEM1DResidualOperator<'p, 'k, 'terms, M, K> {
        SEM1DResidualOperator {
            problem: self.problem,
            kernel: self.kernel,
            time: self.time,
            terms: Some(terms),
        }
    }
}

impl<M, K> CompleteResidualOperator for SEM1DResidualOperator<'_, '_, '_, M, K>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
    K: ResidualKernel + Sync,
{
    fn system_size(&self) -> usize {
        SEM1DResidualOperator::system_size(self)
    }

    fn residual(&self, state: MatRef<f64>) -> Vec<f64> {
        SEM1DResidualOperator::residual(self, state)
    }

    fn assemble_jacobian(&self, state: MatRef<f64>) -> SparseColMat<usize, f64> {
        SEM1DResidualOperator::assemble_jacobian(self, state)
    }

    fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        SEM1DResidualOperator::apply_jacobian(self, state, direction)
    }

    fn apply_jacobian_into(
        &self,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        mut out: MatMut<'_, f64>,
    ) {
        self.problem
            .apply_jacobian_into(self.time, self.kernel, state, direction, out.rb_mut());
        if let Some(terms) = self.terms {
            out += self
                .problem
                .apply_state_boundary_jacobian(self.time, state, direction, terms);
        }
    }
}

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> SEM1DProblem<M> {
    pub fn residual_operator<'p, 'k, K: ResidualKernel + Sync>(
        &'p self,
        kernel: &'k K,
    ) -> SEM1DResidualOperator<'p, 'k, 'static, M, K>
    where
        M: Sync,
    {
        self.validate_fields(kernel.nfields(), kernel.field_names(), "residual kernel");
        SEM1DResidualOperator {
            problem: self,
            kernel,
            time: 0.0,
            terms: None,
        }
    }
    /// Build a problem with named scalar fields in system-vector order.
    pub fn new(mesh: M, p: usize, fields: FieldRegistry, reduction: DofReduction1D) -> Self {
        Self::new_with_metadata(mesh, p, fields, reduction, MeshMetadata::default())
    }

    /// Build a problem with named scalar fields and mesh metadata.
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

        Self {
            mesh,
            fields,
            family,
            cell_data,
            cell_reduced_dofs,
            cell_prescribed_values,
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

    fn prescribed_field_dof(&self, field: usize, full: usize) -> Option<f64> {
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

    fn field_layout(&self) -> FieldDofLayout {
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

    fn cell_field_maps(
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
                grads[dof_i * cd.npts + q] = cd.jinv_cache[cell_index * cd.npts + q]
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

    fn tensor_ctx<'a>(&'a self, time: f64, cell_index: usize) -> TensorCtx<'a> {
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

    pub fn assemble_residual<K: ResidualKernel + Sync>(
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
        self.validate_fields(nfields, kernel.field_names(), "residual kernel");
        let layout = self.field_layout();
        assert_eq!(state.nrows(), layout.total_size, "state size mismatch");
        assert_eq!(
            state.ncols(),
            1,
            "residual assembly requires one state column"
        );
        if kernel.supports_tensor_residual_1d() && self.cell_data.tensor.is_some() {
            return self.assemble_tensor_residual(time, kernel, state, &layout);
        }
        let cd = &self.cell_data;
        let local_stride = nfields * cd.ndofs;
        let batches: Vec<Vec<f64>> = self.cell_reduced_dofs[0]
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
                        let (field_maps, field_prescribed) =
                            self.cell_field_maps(cell_index, nfields);
                        self.populate_cell_grads(
                            cell_index,
                            ndofs,
                            &mut basis_grads[..ndofs * cd.npts],
                        );
                        let state_cell = self.prepare_cell_state(
                            nfields,
                            &field_maps,
                            &field_prescribed,
                            &layout.offsets,
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
        let mut residual = vec![0.0; layout.total_size];
        for (batch_index, batch) in batches.into_iter().enumerate() {
            let cell_start = batch_index * CELL_BATCH_SIZE;
            for (cell_offset, reduced_dofs) in self.cell_reduced_dofs[0]
                [cell_start..(cell_start + CELL_BATCH_SIZE).min(self.cell_reduced_dofs[0].len())]
                .iter()
                .enumerate()
            {
                let ndofs = reduced_dofs.len();
                let cell_size = nfields * ndofs;
                let local =
                    &batch[cell_offset * local_stride..cell_offset * local_stride + cell_size];
                let (field_maps, _) = self.cell_field_maps(cell_start + cell_offset, nfields);
                scatter_local_vector(&mut residual, local, &field_maps, nfields, &layout.offsets);
            }
        }
        residual
    }

    fn assemble_tensor_residual<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        layout: &FieldDofLayout,
    ) -> Vec<f64>
    where
        M: Sync,
    {
        let nfields = kernel.nfields();
        let cd = &self.cell_data;
        let local_stride = nfields * cd.ndofs;
        let batches: Vec<Vec<f64>> = self.cell_reduced_dofs[0]
            .par_chunks(CELL_BATCH_SIZE)
            .enumerate()
            .map_init(
                || {
                    (
                        vec![0.0; local_stride],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; local_stride],
                    )
                },
                |(coefficients, field_values, field_grads, local), (batch_index, reduced_batch)| {
                    let mut batch = vec![0.0; reduced_batch.len() * local_stride];
                    for (cell_offset, reduced_dofs) in reduced_batch.iter().enumerate() {
                        let ndofs = reduced_dofs.len();
                        let cell_size = nfields * ndofs;
                        let cell_index = batch_index * CELL_BATCH_SIZE + cell_offset;
                        let (field_maps, field_prescribed) =
                            self.cell_field_maps(cell_index, nfields);
                        let state_cell = interpolate_tensor_cell_state(
                            cd,
                            1,
                            nfields,
                            &field_maps,
                            &field_prescribed,
                            &layout.offsets,
                            state,
                            &mut coefficients[..cell_size],
                            &mut field_values[..nfields * cd.npts],
                            &mut field_grads[..nfields * cd.npts],
                            cell_index,
                        );
                        let ctx = self.tensor_ctx(time, cell_index);
                        assemble_tensor_residual_1d(
                            kernel,
                            &ctx,
                            &state_cell,
                            &mut local[..cell_size],
                        );
                        batch[cell_offset * local_stride..cell_offset * local_stride + cell_size]
                            .copy_from_slice(&local[..cell_size]);
                    }
                    batch
                },
            )
            .collect();
        let mut residual = vec![0.0; layout.total_size];
        for (batch_index, batch) in batches.into_iter().enumerate() {
            let cell_start = batch_index * CELL_BATCH_SIZE;
            for (cell_offset, reduced_dofs) in self.cell_reduced_dofs[0]
                [cell_start..(cell_start + CELL_BATCH_SIZE).min(self.cell_reduced_dofs[0].len())]
                .iter()
                .enumerate()
            {
                let ndofs = reduced_dofs.len();
                let local = &batch
                    [cell_offset * local_stride..cell_offset * local_stride + nfields * ndofs];
                let (field_maps, _) = self.cell_field_maps(cell_start + cell_offset, nfields);
                scatter_local_vector(&mut residual, local, &field_maps, nfields, &layout.offsets);
            }
        }
        residual
    }

    fn assemble_state_boundary_impl(
        &self,
        time: f64,
        state: MatRef<'_, f64>,
        terms: &StateBoundaryTerms,
        include_residual: bool,
        include_jacobian: bool,
    ) -> (Option<Vec<f64>>, Option<SparseColMat<usize, f64>>) {
        let layout = self.field_layout();
        let nfields = self.fields.len();
        assert_eq!(state.nrows(), layout.total_size, "state size mismatch");
        assert_eq!(state.ncols(), 1, "boundary state requires one column");
        let space = FunctionSpaceImpl::new(&self.mesh, &self.family);
        let element = self.family.element(ReferenceCellType::Interval);
        let mut residual = include_residual.then(|| vec![0.0; layout.total_size]);
        let mut triplets = Vec::new();
        let mut coord = [0.0; 1];

        for point in self.mesh.entity_iter(ReferenceCellType::Point) {
            let point_index = point.local_index();
            let Some(kernel) = terms.kernel_for(point_index) else {
                continue;
            };
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
            assert_eq!(
                kernel.nfields(),
                nfields,
                "state boundary field count mismatch"
            );
            if let Some(names) = kernel.field_names() {
                assert_eq!(
                    names,
                    self.fields.names(),
                    "state boundary field names/order mismatch"
                );
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
            let grad = [*table.get([1, 0, cell_dof, 0]).unwrap()
                * self.cell_data.jinv_cache[cell_index * self.cell_data.npts]];
            let points = [coord[0]];
            let normals = [normal];
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
                normal: &normals,
                values: &values,
                grads: &grad,
            };
            let (field_maps, field_prescribed) = self.cell_field_maps(cell_index, nfields);
            let mut state_values = vec![0.0; nfields];
            let mut state_grads = vec![0.0; nfields];
            for field in 0..nfields {
                state_values[field] = field_maps[field][cell_dof].map_or(
                    field_prescribed[field][cell_dof].unwrap_or(0.0),
                    |reduced| state[(layout.offsets[field] + reduced, 0)],
                ) * values[0];
                for (local_i, _) in cell_dofs.iter().enumerate() {
                    let coefficient = field_maps[field][local_i]
                        .map_or(field_prescribed[field][local_i].unwrap_or(0.0), |reduced| {
                            state[(layout.offsets[field] + reduced, 0)]
                        });
                    state_grads[field] += coefficient
                        * *table.get([1, 0, local_i, 0]).unwrap()
                        * self.cell_data.jinv_cache[cell_index * self.cell_data.npts];
                }
            }
            let facet_state = CellState {
                nfields,
                npts: 1,
                gdim: 1,
                values: &state_values,
                grads: &state_grads,
            };
            let mut local_residual = include_residual.then(|| vec![0.0; nfields]);
            let mut local_jacobian = include_jacobian.then(|| vec![0.0; nfields * nfields]);
            if let Some(local_residual) = local_residual.as_mut() {
                kernel.assemble_local_residual(&ctx, &facet_state, local_residual);
            }
            if let Some(local_jacobian) = local_jacobian.as_mut() {
                kernel.assemble_local_jacobian(&ctx, &facet_state, local_jacobian);
            }
            for equation in 0..nfields {
                let Some(reduced) = field_maps[equation][cell_dof] else {
                    continue;
                };
                if let Some(residual) = residual.as_mut() {
                    residual[layout.offsets[equation] + reduced] +=
                        local_residual.as_ref().unwrap()[equation];
                }
                if let Some(local_jacobian) = local_jacobian.as_ref() {
                    for unknown in 0..nfields {
                        let Some(reduced_unknown) = field_maps[unknown][cell_dof] else {
                            continue;
                        };
                        let value = local_jacobian[equation * nfields + unknown];
                        if value != 0.0 {
                            triplets.push(Triplet::new(
                                layout.offsets[equation] + reduced,
                                layout.offsets[unknown] + reduced_unknown,
                                value,
                            ));
                        }
                    }
                }
            }
        }
        let jacobian = include_jacobian.then(|| {
            SparseColMat::try_new_from_triplets(layout.total_size, layout.total_size, &triplets)
                .unwrap()
        });
        (residual, jacobian)
    }

    pub fn assemble_state_boundary(
        &self,
        time: f64,
        state: MatRef<'_, f64>,
        terms: &StateBoundaryTerms,
    ) -> StateBoundaryContributions {
        let (residual, jacobian) =
            self.assemble_state_boundary_impl(time, state, terms, true, true);
        StateBoundaryContributions {
            residual: residual.unwrap(),
            jacobian: jacobian.unwrap(),
        }
    }

    fn assemble_state_boundary_residual(
        &self,
        time: f64,
        state: MatRef<f64>,
        terms: &StateBoundaryTerms,
    ) -> Vec<f64> {
        self.assemble_state_boundary_impl(time, state, terms, true, false)
            .0
            .unwrap()
    }

    fn assemble_state_boundary_jacobian(
        &self,
        time: f64,
        state: MatRef<'_, f64>,
        terms: &StateBoundaryTerms,
    ) -> SparseColMat<usize, f64> {
        self.assemble_state_boundary_impl(time, state, terms, false, true)
            .1
            .unwrap()
    }

    fn assemble_complete_residual<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        terms: &StateBoundaryTerms,
    ) -> Vec<f64>
    where
        M: Sync,
    {
        let mut residual = self.assemble_residual(time, kernel, state);
        let boundary = self.assemble_state_boundary_residual(time, state, terms);
        for (volume, boundary) in residual.iter_mut().zip(boundary) {
            *volume += boundary;
        }
        residual
    }

    pub fn apply_state_boundary_jacobian(
        &self,
        time: f64,
        state: MatRef<'_, f64>,
        direction: MatRef<'_, f64>,
        terms: &StateBoundaryTerms,
    ) -> Mat<f64> {
        let layout = self.field_layout();
        let nfields = self.fields.len();
        assert_eq!(state.nrows(), layout.total_size, "state size mismatch");
        assert_eq!(state.ncols(), 1, "boundary state requires one column");
        assert_eq!(
            direction.nrows(),
            layout.total_size,
            "boundary direction size mismatch"
        );
        let space = FunctionSpaceImpl::new(&self.mesh, &self.family);
        let element = self.family.element(ReferenceCellType::Interval);
        let mut out = Mat::<f64>::zeros(layout.total_size, direction.ncols());
        let mut coord = [0.0; 1];
        for point in self.mesh.entity_iter(ReferenceCellType::Point) {
            let point_index = point.local_index();
            let Some(kernel) = terms.kernel_for(point_index) else {
                continue;
            };
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
            let cell_dof = cell_dofs
                .iter()
                .position(|&dof| dof == point_dofs[0])
                .unwrap();
            let values = [*table.get([0, 0, cell_dof, 0]).unwrap()];
            let grad = [*table.get([1, 0, cell_dof, 0]).unwrap()
                * self.cell_data.jinv_cache[cell_index * self.cell_data.npts]];
            let points = [coord[0]];
            let normals = [normal];
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
                normal: &normals,
                values: &values,
                grads: &grad,
            };
            let (field_maps, field_prescribed) = self.cell_field_maps(cell_index, nfields);
            let mut state_values = vec![0.0; nfields];
            let mut state_grads = vec![0.0; nfields];
            for field in 0..nfields {
                state_values[field] = field_maps[field][cell_dof].map_or(
                    field_prescribed[field][cell_dof].unwrap_or(0.0),
                    |reduced| state[(layout.offsets[field] + reduced, 0)],
                ) * values[0];
                for (local_i, _) in cell_dofs.iter().enumerate() {
                    let coefficient = field_maps[field][local_i]
                        .map_or(field_prescribed[field][local_i].unwrap_or(0.0), |reduced| {
                            state[(layout.offsets[field] + reduced, 0)]
                        });
                    state_grads[field] += coefficient
                        * *table.get([1, 0, local_i, 0]).unwrap()
                        * self.cell_data.jinv_cache[cell_index * self.cell_data.npts];
                }
            }
            let facet_state = CellState {
                nfields,
                npts: 1,
                gdim: 1,
                values: &state_values,
                grads: &state_grads,
            };
            let mut local_direction = vec![0.0; nfields];
            let mut local_action = vec![0.0; nfields];
            for column in 0..direction.ncols() {
                for field in 0..nfields {
                    local_direction[field] = field_maps[field][cell_dof].map_or(0.0, |reduced| {
                        direction[(layout.offsets[field] + reduced, column)]
                    });
                }
                kernel.apply_local_jacobian(
                    &ctx,
                    &facet_state,
                    &local_direction,
                    &mut local_action,
                );
                for field in 0..nfields {
                    if let Some(reduced) = field_maps[field][cell_dof] {
                        out[(layout.offsets[field] + reduced, column)] += local_action[field];
                    }
                }
            }
        }
        out
    }

    pub fn assemble_residual_jacobian<K: ResidualKernel + Sync>(
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
        self.validate_fields(nfields, kernel.field_names(), "residual kernel");
        let layout = self.field_layout();
        assert_eq!(state.nrows(), layout.total_size, "state size mismatch");
        assert_eq!(
            state.ncols(),
            1,
            "Jacobian assembly requires one state column"
        );
        if kernel.supports_tensor_jacobian_1d() && self.cell_data.tensor.is_some() {
            return self.assemble_tensor_jacobian(time, kernel, state, &layout);
        }
        let cd = &self.cell_data;
        let local_size = nfields * cd.ndofs;
        let batches: Vec<Vec<Triplet<usize, usize, f64>>> = self.cell_reduced_dofs[0]
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
                        let (field_maps, field_prescribed) =
                            self.cell_field_maps(cell_index, nfields);
                        self.populate_cell_grads(
                            cell_index,
                            ndofs,
                            &mut basis_grads[..ndofs * cd.npts],
                        );
                        let state_cell = self.prepare_cell_state(
                            nfields,
                            &field_maps,
                            &field_prescribed,
                            &layout.offsets,
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
                            &field_maps,
                            nfields,
                            &layout.offsets,
                            |value| value != 0.0,
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

    fn assemble_tensor_jacobian<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        layout: &FieldDofLayout,
    ) -> SparseColMat<usize, f64>
    where
        M: Sync,
    {
        let nfields = kernel.nfields();
        let cd = &self.cell_data;
        let local_size = nfields * cd.ndofs;
        let batches: Vec<Vec<Triplet<usize, usize, f64>>> = self.cell_reduced_dofs[0]
            .par_chunks(CELL_BATCH_SIZE)
            .enumerate()
            .map_init(
                || {
                    (
                        vec![0.0; local_size],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; local_size * local_size],
                        vec![0.0; local_size],
                    )
                },
                |(
                    state_coefficients,
                    state_values,
                    state_grads,
                    direction_values,
                    direction_grads,
                    local_matrix,
                    local_direction,
                ),
                 (batch_index, reduced_batch)| {
                    let mut triplets =
                        Vec::with_capacity(reduced_batch.len() * local_size * local_size);
                    for (cell_offset, reduced_dofs) in reduced_batch.iter().enumerate() {
                        let ndofs = reduced_dofs.len();
                        let cell_size = nfields * ndofs;
                        let cell_index = batch_index * CELL_BATCH_SIZE + cell_offset;
                        let (field_maps, field_prescribed) =
                            self.cell_field_maps(cell_index, nfields);
                        let state_cell = interpolate_tensor_cell_state(
                            cd,
                            1,
                            nfields,
                            &field_maps,
                            &field_prescribed,
                            &layout.offsets,
                            state,
                            &mut state_coefficients[..cell_size],
                            &mut state_values[..nfields * cd.npts],
                            &mut state_grads[..nfields * cd.npts],
                            cell_index,
                        );
                        let ctx = self.tensor_ctx(time, cell_index);
                        local_matrix[..cell_size * cell_size].fill(0.0);
                        for unknown in 0..nfields {
                            for trial in 0..ndofs {
                                local_direction[..cell_size].fill(0.0);
                                local_direction[unknown * ndofs + trial] = 1.0;
                                let direction_cell = interpolate_tensor_cell_coefficients(
                                    cd,
                                    1,
                                    nfields,
                                    &local_direction[..cell_size],
                                    &mut direction_values[..nfields * cd.npts],
                                    &mut direction_grads[..nfields * cd.npts],
                                    cell_index,
                                );
                                apply_tensor_jacobian_1d(
                                    kernel,
                                    &ctx,
                                    &state_cell,
                                    &direction_cell,
                                    &mut local_direction[..cell_size],
                                );
                                for equation in 0..nfields {
                                    for test in 0..ndofs {
                                        local_matrix[(equation * ndofs + test) * cell_size
                                            + unknown * ndofs
                                            + trial] = local_direction[equation * ndofs + test];
                                    }
                                }
                            }
                        }
                        push_local_matrix_triplets(
                            &mut triplets,
                            &local_matrix[..cell_size * cell_size],
                            &field_maps,
                            nfields,
                            &layout.offsets,
                            |value| value != 0.0,
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

    fn assemble_complete_jacobian<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        terms: &StateBoundaryTerms,
    ) -> SparseColMat<usize, f64>
    where
        M: Sync,
    {
        let volume = self.assemble_residual_jacobian(time, kernel, state);
        let boundary = self.assemble_state_boundary_jacobian(time, state, terms);
        volume.as_ref() + boundary.as_ref()
    }

    pub fn apply_jacobian<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
    ) -> Mat<f64>
    where
        M: Sync,
    {
        let mut out = Mat::<f64>::zeros(self.system_size(), direction.ncols());
        self.apply_jacobian_into(time, kernel, state, direction, out.as_mut());
        out
    }

    /// Apply `dR/du(state)` into caller-provided storage.
    pub fn apply_jacobian_into<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        mut out: MatMut<'_, f64>,
    ) where
        M: Sync,
    {
        let nfields = kernel.nfields();
        assert!(nfields > 0, "kernel must contain at least one field");
        self.validate_fields(nfields, kernel.field_names(), "residual kernel");
        let layout = self.field_layout();
        assert_eq!(state.nrows(), layout.total_size, "state size mismatch");
        assert_eq!(
            state.ncols(),
            1,
            "matrix-free Jacobian requires one state column"
        );
        assert_eq!(
            direction.nrows(),
            layout.total_size,
            "direction size mismatch"
        );
        assert_eq!(out.nrows(), layout.total_size, "output size mismatch");
        assert_eq!(
            out.ncols(),
            direction.ncols(),
            "output column count mismatch"
        );
        out.fill(0.0);
        if kernel.supports_tensor_jacobian_1d() && self.cell_data.tensor.is_some() {
            self.apply_tensor_jacobian_into(time, kernel, state, direction, &layout, out);
            return;
        }
        let cd = &self.cell_data;
        let local_size = nfields * cd.ndofs;
        let ncols = direction.ncols();
        let action_stride = local_size * ncols;
        let batches: Vec<Vec<f64>> = self.cell_reduced_dofs[0]
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
                        let (field_maps, field_prescribed) =
                            self.cell_field_maps(cell_index, nfields);
                        self.populate_cell_grads(
                            cell_index,
                            ndofs,
                            &mut basis_grads[..ndofs * cd.npts],
                        );
                        let state_cell = self.prepare_cell_state(
                            nfields,
                            &field_maps,
                            &field_prescribed,
                            &layout.offsets,
                            state,
                            &basis_grads[..ndofs * cd.npts],
                            field_values,
                            field_grads,
                        );
                        let ctx =
                            self.cell_ctx(time, cell_index, ndofs, &basis_grads[..ndofs * cd.npts]);
                        for column in 0..ncols {
                            for field in 0..nfields {
                                for (local_i, &reduced_i) in field_maps[field].iter().enumerate() {
                                    local_direction[field * ndofs + local_i] = reduced_i
                                        .map_or(0.0, |i| {
                                            direction[(layout.offsets[field] + i, column)]
                                        });
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
        for (batch_index, batch) in batches.into_iter().enumerate() {
            let cell_start = batch_index * CELL_BATCH_SIZE;
            for (cell_offset, reduced_dofs) in self.cell_reduced_dofs[0]
                [cell_start..(cell_start + CELL_BATCH_SIZE).min(self.cell_reduced_dofs[0].len())]
                .iter()
                .enumerate()
            {
                let ndofs = reduced_dofs.len();
                let (field_maps, _) = self.cell_field_maps(cell_start + cell_offset, nfields);
                for column in 0..ncols {
                    let start = cell_offset * action_stride + column * local_size;
                    let local = &batch[start..start + nfields * ndofs];
                    for field in 0..nfields {
                        for (local_i, &reduced_i) in field_maps[field].iter().enumerate() {
                            if let Some(reduced_i) = reduced_i {
                                out[(layout.offsets[field] + reduced_i, column)] +=
                                    local[field * ndofs + local_i];
                            }
                        }
                    }
                }
            }
        }
    }

    fn apply_tensor_jacobian_into<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        layout: &FieldDofLayout,
        mut out: MatMut<'_, f64>,
    ) where
        M: Sync,
    {
        let nfields = kernel.nfields();
        let cd = &self.cell_data;
        let local_size = nfields * cd.ndofs;
        let ncols = direction.ncols();
        let action_stride = local_size * ncols;
        let batches: Vec<Vec<f64>> = self.cell_reduced_dofs[0]
            .par_chunks(CELL_BATCH_SIZE)
            .enumerate()
            .map_init(
                || {
                    (
                        vec![0.0; nfields * cd.ndofs],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; local_size],
                        vec![0.0; local_size],
                    )
                },
                |(
                    state_coefficients,
                    state_values,
                    state_grads,
                    direction_values,
                    direction_grads,
                    local_direction,
                    local_action,
                ),
                 (batch_index, reduced_batch)| {
                    let mut batch = vec![0.0; reduced_batch.len() * action_stride];
                    for (cell_offset, reduced_dofs) in reduced_batch.iter().enumerate() {
                        let ndofs = reduced_dofs.len();
                        let cell_size = nfields * ndofs;
                        let cell_index = batch_index * CELL_BATCH_SIZE + cell_offset;
                        let (field_maps, field_prescribed) =
                            self.cell_field_maps(cell_index, nfields);
                        let state_cell = interpolate_tensor_cell_state(
                            cd,
                            1,
                            nfields,
                            &field_maps,
                            &field_prescribed,
                            &layout.offsets,
                            state,
                            &mut state_coefficients[..cell_size],
                            &mut state_values[..nfields * cd.npts],
                            &mut state_grads[..nfields * cd.npts],
                            cell_index,
                        );
                        let ctx = self.tensor_ctx(time, cell_index);
                        for column in 0..ncols {
                            for field in 0..nfields {
                                for (local, &reduced) in field_maps[field].iter().enumerate() {
                                    local_direction[field * ndofs + local] =
                                        reduced.map_or(0.0, |reduced| {
                                            direction[(layout.offsets[field] + reduced, column)]
                                        });
                                }
                            }
                            let direction_cell = interpolate_tensor_cell_coefficients(
                                cd,
                                1,
                                nfields,
                                &local_direction[..cell_size],
                                &mut direction_values[..nfields * cd.npts],
                                &mut direction_grads[..nfields * cd.npts],
                                cell_index,
                            );
                            apply_tensor_jacobian_1d(
                                kernel,
                                &ctx,
                                &state_cell,
                                &direction_cell,
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
        for (batch_index, batch) in batches.into_iter().enumerate() {
            let cell_start = batch_index * CELL_BATCH_SIZE;
            for (cell_offset, reduced_dofs) in self.cell_reduced_dofs[0]
                [cell_start..(cell_start + CELL_BATCH_SIZE).min(self.cell_reduced_dofs[0].len())]
                .iter()
                .enumerate()
            {
                let ndofs = reduced_dofs.len();
                let (field_maps, _) = self.cell_field_maps(cell_start + cell_offset, nfields);
                for column in 0..ncols {
                    let start = cell_offset * action_stride + column * local_size;
                    let local_action = &batch[start..start + nfields * ndofs];
                    for field in 0..nfields {
                        for (local_dof, &reduced) in field_maps[field].iter().enumerate() {
                            if let Some(reduced) = reduced {
                                out[(layout.offsets[field] + reduced, column)] +=
                                    local_action[field * ndofs + local_dof];
                            }
                        }
                    }
                }
            }
        }
    }

    fn apply_complete_jacobian<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        terms: &StateBoundaryTerms,
    ) -> Mat<f64>
    where
        M: Sync,
    {
        self.apply_jacobian(time, kernel, state, direction)
            + self.apply_state_boundary_jacobian(time, state, direction, terms)
    }

    pub fn assemble_bilinear<K: BilinearForm + Sync>(
        &self,
        time: f64,
        kernel: &K,
    ) -> SparseColMat<usize, f64>
    where
        M: Sync,
    {
        let nfields = kernel.nfields();
        assert!(nfields > 0, "bilinear form must contain at least one field");
        self.validate_fields(nfields, kernel.field_names(), "bilinear form");
        let layout = self.field_layout();
        if kernel.supports_tensor_bilinear_1d() && self.cell_data.tensor.is_some() {
            return self.assemble_tensor_bilinear(time, kernel, &layout);
        }
        let cd = &self.cell_data;
        let local_size = nfields * cd.ndofs;
        let batches: Vec<Vec<Triplet<usize, usize, f64>>> = self.cell_reduced_dofs[0]
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
                        let (field_maps, _) = self.cell_field_maps(cell_index, nfields);
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
                            &layout.offsets,
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

    fn assemble_tensor_bilinear<K: BilinearForm + Sync>(
        &self,
        time: f64,
        kernel: &K,
        layout: &FieldDofLayout,
    ) -> SparseColMat<usize, f64>
    where
        M: Sync,
    {
        let nfields = kernel.nfields();
        let cd = &self.cell_data;
        let local_size = nfields * cd.ndofs;
        let batches: Vec<Vec<Triplet<usize, usize, f64>>> = self.cell_reduced_dofs[0]
            .par_chunks(CELL_BATCH_SIZE)
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
                        let cell_index = batch_index * CELL_BATCH_SIZE + cell_offset;
                        let (field_maps, _) = self.cell_field_maps(cell_index, nfields);
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
                            &layout.offsets,
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

    pub fn assemble_linear<K: LinearForm>(&self, time: f64, kernel: &K) -> Vec<f64> {
        let nfields = kernel.nfields();
        assert!(nfields > 0, "linear form must contain at least one field");
        self.validate_fields(nfields, kernel.field_names(), "linear form");
        let layout = self.field_layout();
        let cd = &self.cell_data;
        let mut grads = vec![0.0; cd.ndofs * cd.npts];
        let mut local = vec![0.0; nfields * cd.ndofs];
        let mut rhs = vec![0.0; layout.total_size];

        for c in 0..self.cell_reduced_dofs[0].len() {
            let reduced_dofs = &self.cell_reduced_dofs[0][c];
            let ndofs = reduced_dofs.len();
            let (field_maps, _) = self.cell_field_maps(c, nfields);
            let ctx = self.prepare_cell_ctx(time, c, ndofs, &mut grads[..ndofs * cd.npts]);
            kernel.assemble_local_rhs(&ctx, &mut local[..nfields * ndofs]);
            scatter_local_vector(
                &mut rhs,
                &local[..nfields * ndofs],
                &field_maps,
                nfields,
                &layout.offsets,
            );
        }
        rhs
    }

    /// Assemble a linear RHS and apply the nonzero Dirichlet correction.
    pub fn assemble_linear_with_dirichlet<B, L>(
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
    pub fn apply_dirichlet_rhs_correction<K: BilinearForm>(
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

    fn apply_tensor_dirichlet_rhs_correction<K: BilinearForm>(
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

    /// Assemble block-diagonal lumped GLL mass for all problem fields.
    pub fn assemble_lumped_mass(&self) -> SparseColMat<usize, f64> {
        let nfields = self.fields.len();
        let layout = self.field_layout();
        assemble_lumped_mass(
            &self.cell_data,
            &self.cell_reduced_dofs,
            &layout.offsets,
            nfields,
        )
    }

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
