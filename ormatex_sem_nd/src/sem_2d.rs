use crate::common::{
    add_dirichlet_rhs_correction, apply_quad_state_boundary_terms_cached,
    apply_quad_state_tensor_boundary_terms_cached, assemble_lumped_mass, assemble_quad_boundaries,
    assemble_quad_state_boundary_jacobian, assemble_quad_state_boundary_residual_cached,
    assemble_quad_state_tensor_boundary_jacobian_cached,
    assemble_quad_state_tensor_boundary_residual_cached, build_quad_state_boundary_cache, cell_ctx,
    interpolate_cell_state, interpolate_tensor_cell_coefficients, interpolate_tensor_cell_state,
    push_local_matrix_triplets, scatter_local_vector, BoundaryContributions, BoundaryFacet,
    CellData, CellState, FieldDofLayout, LocalCtx, QuadStateBoundaryCache, ReducedDofMap,
    StateBoundaryContributions, TensorCtx, TensorProductData, CELL_BATCH_SIZE,
};
use crate::fields::{FieldRegistry, FieldValues};
use crate::jacobian::CompleteResidualOperator;
use crate::kernels::kernel_common::{
    apply_tensor_bilinear_column, apply_tensor_jacobian, assemble_tensor_residual, BilinearForm,
    BoundaryIntegrator, LinearForm, ResidualKernel, StateBoundaryTerms, StateTensorBoundaryTerms,
    TensorResidualKernel,
};
use crate::mesh::MeshMetadata;
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
    /// Eliminate explicitly listed full-space DOFs and prescribe their values.
    /// This is useful when a boundary value changes at a corner.
    DirichletValues { values: Vec<(usize, f64)> },
    /// Apply one reduction policy per scalar field. The number of policies
    /// must match the kernel field count during system assembly.
    FieldSpecific { reductions: Vec<DofReduction2D> },
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
    mesh: M,
    fields: FieldRegistry,
    family: LagrangeElementFamily<f64>,
    p: usize,
    cell_data: CellData,
    cell_reduced_dofs: Vec<Vec<Vec<Option<usize>>>>,
    cell_prescribed_values: Vec<Vec<Vec<Option<f64>>>>,
    dof_map: ReducedDofMap,
    field_dof_maps: Vec<ReducedDofMap>,
    /// (x, y) position of each full GLL nodal DOF.
    dof_xy: Vec<(f64, f64)>,
    metadata: MeshMetadata,
    state_boundary_cache: QuadStateBoundaryCache,
}

pub struct SEM2DResidualOperator<'p, 'k, 'b, M, K>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
{
    problem: &'p SEM2DProblem<M>,
    kernel: &'k K,
    time: f64,
    terms: Option<&'b StateBoundaryTerms>,
}

/// Statically dispatched tensor-product residual operator.
pub struct SEM2DTensorResidualOperator<'p, 'k, 'b, M, K>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
{
    problem: &'p SEM2DProblem<M>,
    kernel: &'k K,
    time: f64,
    terms: Option<&'b StateTensorBoundaryTerms<2>>,
}

impl<'p, 'k, 'b, M, K> SEM2DTensorResidualOperator<'p, 'k, 'b, M, K>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
    K: TensorResidualKernel<2> + Sync,
{
    pub fn system_size(&self) -> usize {
        self.problem.system_size()
    }
    pub fn residual(&self, state: MatRef<f64>) -> Vec<f64> {
        let layout = self.problem.field_layout();
        let mut residual =
            self.problem
                .assemble_tensor_residual_static(self.time, self.kernel, state, &layout);
        if let Some(terms) = self.terms {
            let boundary = self
                .problem
                .assemble_state_tensor_boundary_residual(self.time, state, terms);
            for (volume, boundary) in residual.iter_mut().zip(boundary) {
                *volume += boundary;
            }
        }
        residual
    }
    pub fn assemble_jacobian(&self, state: MatRef<f64>) -> SparseColMat<usize, f64> {
        let layout = self.problem.field_layout();
        let mut jacobian =
            self.problem
                .assemble_tensor_jacobian(self.time, self.kernel, state, &layout);
        if let Some(terms) = self.terms {
            let boundary = self
                .problem
                .assemble_state_tensor_boundary_jacobian(self.time, state, terms);
            jacobian = jacobian.as_ref() + boundary.as_ref();
        }
        jacobian
    }
    pub fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        let mut out = Mat::<f64>::zeros(self.problem.system_size(), direction.ncols());
        let layout = self.problem.field_layout();
        self.problem.apply_tensor_jacobian_into(
            self.time,
            self.kernel,
            state,
            direction,
            &layout,
            out.as_mut(),
        );
        if let Some(terms) = self.terms {
            out += self
                .problem
                .apply_state_tensor_boundary_jacobian(self.time, state, direction, terms);
        }
        out
    }
    pub fn apply_jacobian_into(
        &self,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        mut out: MatMut<'_, f64>,
    ) {
        let layout = self.problem.field_layout();
        self.problem.apply_tensor_jacobian_into(
            self.time,
            self.kernel,
            state,
            direction,
            &layout,
            out.rb_mut(),
        );
        if let Some(terms) = self.terms {
            out += self
                .problem
                .apply_state_tensor_boundary_jacobian(self.time, state, direction, terms);
        }
    }
    pub fn at_time(mut self, time: f64) -> Self {
        self.time = time;
        self
    }

    pub fn with_state_boundary<'terms>(
        self,
        terms: &'terms StateTensorBoundaryTerms<2>,
    ) -> SEM2DTensorResidualOperator<'p, 'k, 'terms, M, K> {
        SEM2DTensorResidualOperator {
            problem: self.problem,
            kernel: self.kernel,
            time: self.time,
            terms: Some(terms),
        }
    }
}

impl<M, K> CompleteResidualOperator for SEM2DTensorResidualOperator<'_, '_, '_, M, K>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
    K: TensorResidualKernel<2> + Sync,
{
    fn system_size(&self) -> usize {
        SEM2DTensorResidualOperator::system_size(self)
    }
    fn residual(&self, state: MatRef<f64>) -> Vec<f64> {
        SEM2DTensorResidualOperator::residual(self, state)
    }
    fn assemble_jacobian(&self, state: MatRef<f64>) -> SparseColMat<usize, f64> {
        SEM2DTensorResidualOperator::assemble_jacobian(self, state)
    }
    fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        SEM2DTensorResidualOperator::apply_jacobian(self, state, direction)
    }
    fn apply_jacobian_into(
        &self,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        out: MatMut<'_, f64>,
    ) {
        SEM2DTensorResidualOperator::apply_jacobian_into(self, state, direction, out);
    }
}

/// Residual operator that combines one statically dispatched tensor kernel
/// with one traditional weak-form kernel.
pub struct SEM2DMixedResidualOperator<'p, 't, 'w, 'bt, 'bw, M, T, W>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
{
    problem: &'p SEM2DProblem<M>,
    tensor_kernel: &'t T,
    weak_kernel: &'w W,
    time: f64,
    tensor_terms: Option<&'bt StateTensorBoundaryTerms<2>>,
    weak_terms: Option<&'bw StateBoundaryTerms>,
}

impl<'p, 't, 'w, 'bt, 'bw, M, T, W> SEM2DMixedResidualOperator<'p, 't, 'w, 'bt, 'bw, M, T, W>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
    T: TensorResidualKernel<2> + Sync,
    W: ResidualKernel + Sync,
{
    pub fn system_size(&self) -> usize {
        self.problem.system_size()
    }

    pub fn residual(&self, state: MatRef<f64>) -> Vec<f64> {
        let layout = self.problem.field_layout();
        let mut residual = self.problem.assemble_tensor_residual_static(
            self.time,
            self.tensor_kernel,
            state,
            &layout,
        );
        let weak = self
            .problem
            .assemble_residual(self.time, self.weak_kernel, state);
        for (tensor, weak) in residual.iter_mut().zip(weak) {
            *tensor += weak;
        }
        if let Some(terms) = self.weak_terms {
            let boundary = self
                .problem
                .assemble_state_boundary_residual(self.time, state, terms);
            for (volume, boundary) in residual.iter_mut().zip(boundary) {
                *volume += boundary;
            }
        }
        if let Some(terms) = self.tensor_terms {
            let boundary = self
                .problem
                .assemble_state_tensor_boundary_residual(self.time, state, terms);
            for (volume, boundary) in residual.iter_mut().zip(boundary) {
                *volume += boundary;
            }
        }
        residual
    }

    pub fn assemble_jacobian(&self, state: MatRef<f64>) -> SparseColMat<usize, f64> {
        let layout = self.problem.field_layout();
        let tensor =
            self.problem
                .assemble_tensor_jacobian(self.time, self.tensor_kernel, state, &layout);
        let weak = self
            .problem
            .assemble_residual_jacobian(self.time, self.weak_kernel, state);
        let mut jacobian = tensor.as_ref() + weak.as_ref();
        if let Some(terms) = self.weak_terms {
            let boundary = self
                .problem
                .assemble_state_boundary_jacobian(self.time, state, terms);
            jacobian = jacobian.as_ref() + boundary.as_ref();
        }
        if let Some(terms) = self.tensor_terms {
            let boundary = self
                .problem
                .assemble_state_tensor_boundary_jacobian(self.time, state, terms);
            jacobian = jacobian.as_ref() + boundary.as_ref();
        }
        jacobian
    }

    pub fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        let mut out = Mat::<f64>::zeros(self.system_size(), direction.ncols());
        let layout = self.problem.field_layout();
        self.problem.apply_tensor_jacobian_into(
            self.time,
            self.tensor_kernel,
            state,
            direction,
            &layout,
            out.as_mut(),
        );
        out += self
            .problem
            .apply_jacobian(self.time, self.weak_kernel, state, direction);
        if let Some(terms) = self.weak_terms {
            out += self
                .problem
                .apply_state_boundary_jacobian(self.time, state, direction, terms);
        }
        if let Some(terms) = self.tensor_terms {
            out += self
                .problem
                .apply_state_tensor_boundary_jacobian(self.time, state, direction, terms);
        }
        out
    }

    pub fn apply_jacobian_into(
        &self,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        mut out: MatMut<'_, f64>,
    ) {
        let layout = self.problem.field_layout();
        self.problem.apply_tensor_jacobian_into(
            self.time,
            self.tensor_kernel,
            state,
            direction,
            &layout,
            out.rb_mut(),
        );
        out += self
            .problem
            .apply_jacobian(self.time, self.weak_kernel, state, direction);
        if let Some(terms) = self.weak_terms {
            out += self
                .problem
                .apply_state_boundary_jacobian(self.time, state, direction, terms);
        }
        if let Some(terms) = self.tensor_terms {
            out += self
                .problem
                .apply_state_tensor_boundary_jacobian(self.time, state, direction, terms);
        }
    }

    pub fn at_time(mut self, time: f64) -> Self {
        self.time = time;
        self
    }

    pub fn with_weak_state_boundary<'terms>(
        self,
        terms: &'terms StateBoundaryTerms,
    ) -> SEM2DMixedResidualOperator<'p, 't, 'w, 'bt, 'terms, M, T, W> {
        SEM2DMixedResidualOperator {
            problem: self.problem,
            tensor_kernel: self.tensor_kernel,
            weak_kernel: self.weak_kernel,
            time: self.time,
            tensor_terms: self.tensor_terms,
            weak_terms: Some(terms),
        }
    }

    pub fn with_tensor_state_boundary<'terms>(
        self,
        terms: &'terms StateTensorBoundaryTerms<2>,
    ) -> SEM2DMixedResidualOperator<'p, 't, 'w, 'terms, 'bw, M, T, W> {
        SEM2DMixedResidualOperator {
            problem: self.problem,
            tensor_kernel: self.tensor_kernel,
            weak_kernel: self.weak_kernel,
            time: self.time,
            tensor_terms: Some(terms),
            weak_terms: self.weak_terms,
        }
    }
}

impl<M, T, W> CompleteResidualOperator for SEM2DMixedResidualOperator<'_, '_, '_, '_, '_, M, T, W>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
    T: TensorResidualKernel<2> + Sync,
    W: ResidualKernel + Sync,
{
    fn system_size(&self) -> usize {
        SEM2DMixedResidualOperator::system_size(self)
    }

    fn residual(&self, state: MatRef<f64>) -> Vec<f64> {
        SEM2DMixedResidualOperator::residual(self, state)
    }

    fn assemble_jacobian(&self, state: MatRef<f64>) -> SparseColMat<usize, f64> {
        SEM2DMixedResidualOperator::assemble_jacobian(self, state)
    }

    fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        SEM2DMixedResidualOperator::apply_jacobian(self, state, direction)
    }

    fn apply_jacobian_into(
        &self,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        out: MatMut<'_, f64>,
    ) {
        SEM2DMixedResidualOperator::apply_jacobian_into(self, state, direction, out);
    }
}

/// Selects a weak, tensor, or mixed residual operator without erasing the
/// concrete kernel types. The match is performed once at the operator
/// boundary; each branch retains static dispatch through the SEM loops.
pub enum SEM2DResidualExecution<'p, 't, 'w, 'bt, 'bw, M, T, W>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
{
    Weak(SEM2DResidualOperator<'p, 'w, 'bw, M, W>),
    Tensor(SEM2DTensorResidualOperator<'p, 't, 'bt, M, T>),
    Mixed(SEM2DMixedResidualOperator<'p, 't, 'w, 'bt, 'bw, M, T, W>),
}

impl<'p, 't, 'w, 'bt, 'bw, M, T, W> SEM2DResidualExecution<'p, 't, 'w, 'bt, 'bw, M, T, W>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
    T: TensorResidualKernel<2> + Sync,
    W: ResidualKernel + Sync,
{
    pub fn system_size(&self) -> usize {
        match self {
            Self::Weak(operator) => operator.system_size(),
            Self::Tensor(operator) => operator.system_size(),
            Self::Mixed(operator) => operator.system_size(),
        }
    }

    pub fn residual(&self, state: MatRef<f64>) -> Vec<f64> {
        match self {
            Self::Weak(operator) => operator.residual(state),
            Self::Tensor(operator) => operator.residual(state),
            Self::Mixed(operator) => operator.residual(state),
        }
    }

    pub fn assemble_jacobian(&self, state: MatRef<f64>) -> SparseColMat<usize, f64> {
        match self {
            Self::Weak(operator) => operator.assemble_jacobian(state),
            Self::Tensor(operator) => operator.assemble_jacobian(state),
            Self::Mixed(operator) => operator.assemble_jacobian(state),
        }
    }

    pub fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        match self {
            Self::Weak(operator) => operator.apply_jacobian(state, direction),
            Self::Tensor(operator) => operator.apply_jacobian(state, direction),
            Self::Mixed(operator) => operator.apply_jacobian(state, direction),
        }
    }

    pub fn apply_jacobian_into(
        &self,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        out: MatMut<'_, f64>,
    ) {
        match self {
            Self::Weak(operator) => {
                CompleteResidualOperator::apply_jacobian_into(operator, state, direction, out)
            }
            Self::Tensor(operator) => operator.apply_jacobian_into(state, direction, out),
            Self::Mixed(operator) => operator.apply_jacobian_into(state, direction, out),
        }
    }

    pub fn at_time(self, time: f64) -> Self {
        match self {
            Self::Weak(operator) => Self::Weak(operator.at_time(time)),
            Self::Tensor(operator) => Self::Tensor(operator.at_time(time)),
            Self::Mixed(operator) => Self::Mixed(operator.at_time(time)),
        }
    }
}

impl<M, T, W> CompleteResidualOperator for SEM2DResidualExecution<'_, '_, '_, '_, '_, M, T, W>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
    T: TensorResidualKernel<2> + Sync,
    W: ResidualKernel + Sync,
{
    fn system_size(&self) -> usize {
        SEM2DResidualExecution::system_size(self)
    }

    fn residual(&self, state: MatRef<f64>) -> Vec<f64> {
        SEM2DResidualExecution::residual(self, state)
    }

    fn assemble_jacobian(&self, state: MatRef<f64>) -> SparseColMat<usize, f64> {
        SEM2DResidualExecution::assemble_jacobian(self, state)
    }

    fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        SEM2DResidualExecution::apply_jacobian(self, state, direction)
    }

    fn apply_jacobian_into(
        &self,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        out: MatMut<'_, f64>,
    ) {
        SEM2DResidualExecution::apply_jacobian_into(self, state, direction, out);
    }
}

impl<'p, 'k, 'b, M, K> SEM2DResidualOperator<'p, 'k, 'b, M, K>
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
    ) -> SEM2DResidualOperator<'p, 'k, 'terms, M, K> {
        SEM2DResidualOperator {
            problem: self.problem,
            kernel: self.kernel,
            time: self.time,
            terms: Some(terms),
        }
    }
}

impl<M, K> CompleteResidualOperator for SEM2DResidualOperator<'_, '_, '_, M, K>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
    K: ResidualKernel + Sync,
{
    fn system_size(&self) -> usize {
        SEM2DResidualOperator::system_size(self)
    }

    fn residual(&self, state: MatRef<f64>) -> Vec<f64> {
        SEM2DResidualOperator::residual(self, state)
    }

    fn assemble_jacobian(&self, state: MatRef<f64>) -> SparseColMat<usize, f64> {
        SEM2DResidualOperator::assemble_jacobian(self, state)
    }

    fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        SEM2DResidualOperator::apply_jacobian(self, state, direction)
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

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> SEM2DProblem<M> {
    pub fn residual_operator<'p, 'k, K: ResidualKernel + Sync>(
        &'p self,
        kernel: &'k K,
    ) -> SEM2DResidualOperator<'p, 'k, 'static, M, K>
    where
        M: Sync,
    {
        self.validate_fields(kernel.nfields(), kernel.field_names(), "residual kernel");
        SEM2DResidualOperator {
            problem: self,
            kernel,
            time: 0.0,
            terms: None,
        }
    }
    pub fn tensor_residual_operator<'p, 'k, K: TensorResidualKernel<2> + Sync>(
        &'p self,
        kernel: &'k K,
    ) -> SEM2DTensorResidualOperator<'p, 'k, 'static, M, K>
    where
        M: Sync,
    {
        self.validate_fields(
            kernel.nfields(),
            kernel.field_names(),
            "tensor residual kernel",
        );
        SEM2DTensorResidualOperator {
            problem: self,
            kernel,
            time: 0.0,
            terms: None,
        }
    }

    pub fn mixed_residual_operator<'p, 't, 'w, T, W>(
        &'p self,
        tensor_kernel: &'t T,
        weak_kernel: &'w W,
    ) -> SEM2DMixedResidualOperator<'p, 't, 'w, 'static, 'static, M, T, W>
    where
        M: Sync,
        T: TensorResidualKernel<2> + Sync,
        W: ResidualKernel + Sync,
    {
        assert_eq!(
            tensor_kernel.nfields(),
            weak_kernel.nfields(),
            "mixed residual kernel field count mismatch"
        );
        self.validate_fields(
            tensor_kernel.nfields(),
            tensor_kernel.field_names(),
            "tensor residual kernel",
        );
        self.validate_fields(
            weak_kernel.nfields(),
            weak_kernel.field_names(),
            "residual kernel",
        );
        if let (Some(tensor_names), Some(weak_names)) =
            (tensor_kernel.field_names(), weak_kernel.field_names())
        {
            assert_eq!(
                tensor_names, weak_names,
                "mixed residual kernel field names/order mismatch"
            );
        }
        SEM2DMixedResidualOperator {
            problem: self,
            tensor_kernel,
            weak_kernel,
            time: 0.0,
            tensor_terms: None,
            weak_terms: None,
        }
    }
    /// Build a GLL quadrilateral spectral-element problem.
    /// Build a problem with named scalar fields in system-vector order.
    pub fn new(mesh: M, p: usize, fields: FieldRegistry, bc: DofReduction2D) -> Self {
        Self::new_with_metadata(mesh, p, fields, bc, MeshMetadata::default())
    }

    /// Build a problem with named scalar fields and mesh metadata.
    pub fn new_with_metadata(
        mesh: M,
        p: usize,
        fields: FieldRegistry,
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

        let state_boundary_cache = build_quad_state_boundary_cache(&mesh, &family, p, &metadata);
        Self {
            mesh,
            fields,
            family,
            p,
            cell_data,
            cell_reduced_dofs,
            cell_prescribed_values,
            dof_map,
            field_dof_maps,
            dof_xy,
            metadata,
            state_boundary_cache,
        }
    }

    /// O(1) full -> Option<reduced> DOF lookup.
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
                        acc += cd.jinv_cache
                            [(cell_index * npts + q) * tdim * gdim + td * gdim + gd]
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

    fn tensor_ctx<'a>(&'a self, time: f64, cell_index: usize) -> TensorCtx<'a> {
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

    /// Assemble the positive weak spatial residual `R(U)` of a state-aware
    /// kernel. Semi-discrete systems apply the lumped inverse mass separately.
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
        let gdim = self.mesh.geometry_dim();
        let cd = &self.cell_data;
        let local_stride = nfields * cd.ndofs;
        let batches: Vec<Vec<f64>> = self.cell_reduced_dofs[0]
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
                        let (field_maps, field_prescribed) =
                            self.cell_field_maps(cell_index, nfields);
                        self.populate_cell_grads(
                            cell_index,
                            ndofs,
                            &mut basis_grads[..ndofs * gdim * cd.npts],
                        );
                        let state_cell = self.prepare_cell_state(
                            nfields,
                            &field_maps,
                            &field_prescribed,
                            &layout.offsets,
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

    fn assemble_tensor_residual_static<K: TensorResidualKernel<2> + Sync>(
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
                        vec![0.0; nfields * cd.ndofs],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; nfields * 2 * cd.npts],
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
                            2,
                            nfields,
                            &field_maps,
                            &field_prescribed,
                            &layout.offsets,
                            state,
                            &mut coefficients[..cell_size],
                            &mut field_values[..nfields * cd.npts],
                            &mut field_grads[..nfields * 2 * cd.npts],
                            cell_index,
                        );
                        let ctx = self.tensor_ctx(time, cell_index);
                        assemble_tensor_residual(
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

    fn assemble_state_boundary_residual(
        &self,
        time: f64,
        state: MatRef<f64>,
        terms: &StateBoundaryTerms,
    ) -> Vec<f64> {
        assemble_quad_state_boundary_residual_cached(
            &self.state_boundary_cache,
            &self.fields,
            time,
            state,
            &self.cell_reduced_dofs,
            &self.cell_prescribed_values,
            &self.field_layout().offsets,
            terms,
        )
    }

    fn assemble_state_tensor_boundary_residual(
        &self,
        time: f64,
        state: MatRef<f64>,
        terms: &StateTensorBoundaryTerms<2>,
    ) -> Vec<f64> {
        assemble_quad_state_tensor_boundary_residual_cached(
            &self.state_boundary_cache,
            &self.fields,
            time,
            state,
            &self.cell_reduced_dofs,
            &self.cell_prescribed_values,
            &self.field_layout().offsets,
            terms,
        )
    }

    fn assemble_state_tensor_boundary_jacobian(
        &self,
        time: f64,
        state: MatRef<f64>,
        terms: &StateTensorBoundaryTerms<2>,
    ) -> SparseColMat<usize, f64> {
        assemble_quad_state_tensor_boundary_jacobian_cached(
            &self.state_boundary_cache,
            &self.fields,
            time,
            state,
            &self.cell_reduced_dofs,
            &self.cell_prescribed_values,
            &self.field_layout().offsets,
            terms,
        )
    }

    fn apply_state_tensor_boundary_jacobian(
        &self,
        time: f64,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        terms: &StateTensorBoundaryTerms<2>,
    ) -> Mat<f64> {
        apply_quad_state_tensor_boundary_terms_cached(
            &self.state_boundary_cache,
            &self.fields,
            time,
            state,
            direction,
            &self.cell_reduced_dofs,
            &self.cell_prescribed_values,
            &self.field_layout().offsets,
            terms,
        )
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

    /// Assemble `dR/du` for a state-aware kernel at `state`.
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
        let gdim = self.mesh.geometry_dim();
        let cd = &self.cell_data;
        let local_size = nfields * cd.ndofs;
        let batches: Vec<Vec<Triplet<usize, usize, f64>>> = self.cell_reduced_dofs[0]
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
                        let (field_maps, field_prescribed) =
                            self.cell_field_maps(cell_index, nfields);
                        self.populate_cell_grads(
                            cell_index,
                            ndofs,
                            &mut basis_grads[..ndofs * gdim * cd.npts],
                        );
                        let state_cell = self.prepare_cell_state(
                            nfields,
                            &field_maps,
                            &field_prescribed,
                            &layout.offsets,
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

    fn assemble_tensor_jacobian<K: TensorResidualKernel<2> + Sync>(
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
                        vec![0.0; nfields * cd.ndofs],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; nfields * 2 * cd.npts],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; nfields * 2 * cd.npts],
                        vec![0.0; local_size * local_size],
                        vec![0.0; local_size],
                    )
                },
                |(
                    state_coefficients,
                    field_values,
                    field_grads,
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
                            2,
                            nfields,
                            &field_maps,
                            &field_prescribed,
                            &layout.offsets,
                            state,
                            &mut state_coefficients[..cell_size],
                            &mut field_values[..nfields * cd.npts],
                            &mut field_grads[..nfields * 2 * cd.npts],
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
                                    2,
                                    nfields,
                                    &local_direction[..cell_size],
                                    &mut direction_values[..nfields * cd.npts],
                                    &mut direction_grads[..nfields * 2 * cd.npts],
                                    cell_index,
                                );
                                apply_tensor_jacobian(
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

    /// Apply `dR/du(state)` to one or more direction columns without building
    /// a global sparse Jacobian. The local action is direct by default.
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
        let gdim = self.mesh.geometry_dim();
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
                        let (field_maps, field_prescribed) =
                            self.cell_field_maps(cell_index, nfields);
                        self.populate_cell_grads(
                            cell_index,
                            ndofs,
                            &mut basis_grads[..ndofs * gdim * cd.npts],
                        );
                        let state_cell = self.prepare_cell_state(
                            nfields,
                            &field_maps,
                            &field_prescribed,
                            &layout.offsets,
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

    fn apply_tensor_jacobian_into<K: TensorResidualKernel<2> + Sync>(
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
                        vec![0.0; nfields * 2 * cd.npts],
                        vec![0.0; nfields * cd.npts],
                        vec![0.0; nfields * 2 * cd.npts],
                        vec![0.0; local_size],
                        vec![0.0; local_size],
                    )
                },
                |(
                    state_coefficients,
                    field_values,
                    field_grads,
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
                            2,
                            nfields,
                            &field_maps,
                            &field_prescribed,
                            &layout.offsets,
                            state,
                            &mut state_coefficients[..cell_size],
                            &mut field_values[..nfields * cd.npts],
                            &mut field_grads[..nfields * 2 * cd.npts],
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
                                2,
                                nfields,
                                &local_direction[..cell_size],
                                &mut direction_values[..nfields * cd.npts],
                                &mut direction_grads[..nfields * 2 * cd.npts],
                                cell_index,
                            );
                            apply_tensor_jacobian(
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
        let mut action = self.apply_jacobian(time, kernel, state, direction);
        let layout = self.field_layout();
        action += apply_quad_state_boundary_terms_cached(
            &self.state_boundary_cache,
            &self.fields,
            time,
            state,
            direction,
            &self.cell_reduced_dofs,
            &self.cell_prescribed_values,
            &layout.offsets,
            terms,
        );
        action
    }

    /// Assemble a reduced sparse matrix using `kernel.assemble_local` per
    /// quadrilateral. Periodic DOFs are combined and eliminated DOFs omitted
    /// while scattering, so no full-size matrix is exposed.
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
        if kernel.supports_tensor_bilinear() && self.cell_data.tensor.is_some() {
            return self.assemble_tensor_bilinear(time, kernel, &layout);
        }
        let gdim = self.mesh.geometry_dim();
        let cd = &self.cell_data;
        let max_ndofs = cd.ndofs;

        let local_size = nfields * max_ndofs;
        let npts = cd.npts;
        let batches: Vec<Vec<Triplet<usize, usize, f64>>> = self.cell_reduced_dofs[0]
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
                        let (field_maps, _) = self.cell_field_maps(cell_index, nfields);
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
                        vec![0.0; nfields * 2 * cd.npts],
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
                                    2,
                                    nfields,
                                    &trial_coefficients[..cell_size],
                                    &mut trial_values[..nfields * cd.npts],
                                    &mut trial_grads[..nfields * 2 * cd.npts],
                                    cell_index,
                                );
                                apply_tensor_bilinear_column(
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

    /// Assemble a reduced volume RHS from a user-defined `LinearForm`.
    pub fn assemble_linear<K: LinearForm>(&self, time: f64, kernel: &K) -> Vec<f64> {
        let nfields = kernel.nfields();
        assert!(nfields > 0, "linear form must contain at least one field");
        self.validate_fields(nfields, kernel.field_names(), "linear form");
        let layout = self.field_layout();
        let gdim = self.mesh.geometry_dim();
        let cd = &self.cell_data;
        let mut grads_buf = vec![0.0_f64; cd.ndofs * gdim * cd.npts];
        let mut local_rhs = vec![0.0_f64; nfields * cd.ndofs];
        let mut rhs = vec![0.0_f64; layout.total_size];

        for c in 0..self.cell_reduced_dofs[0].len() {
            let reduced_dofs = &self.cell_reduced_dofs[0][c];
            let ndofs = reduced_dofs.len();
            let (field_maps, _) = self.cell_field_maps(c, nfields);
            let npts = cd.npts;
            let ctx = self.prepare_cell_ctx(time, c, ndofs, &mut grads_buf[..ndofs * gdim * npts]);
            kernel.assemble_local_rhs(&ctx, &mut local_rhs[..nfields * ndofs]);
            scatter_local_vector(
                &mut rhs,
                &local_rhs[..nfields * ndofs],
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
        if kernel.supports_tensor_bilinear() && self.cell_data.tensor.is_some() {
            self.apply_tensor_dirichlet_rhs_correction(time, kernel, rhs, &layout);
            return;
        }
        let cd = &self.cell_data;
        let gdim = self.mesh.geometry_dim();
        let mut grads = vec![0.0; cd.ndofs * gdim * cd.npts];
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
        let mut trial_grads = vec![0.0; nfields * 2 * cd.npts];
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
                        2,
                        nfields,
                        &trial_coefficients[..local_size],
                        &mut trial_values[..nfields * cd.npts],
                        &mut trial_grads[..nfields * 2 * cd.npts],
                        cell_index,
                    );
                    apply_tensor_bilinear_column(
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

    /// Assemble the diagonal GLL mass matrix for one scalar field.
    /// Assemble selected natural-boundary kernels. The selector receives the
    /// `ndmesh` facet index and midpoint; returning `None` leaves it adiabatic.
    pub fn assemble_boundary<'a, F>(&self, time: f64, select: F) -> BoundaryContributions
    where
        F: FnMut(BoundaryFacet) -> Option<&'a dyn BoundaryIntegrator>,
    {
        assemble_quad_boundaries(
            &self.mesh,
            &self.family,
            self.p,
            &self.metadata,
            &self.fields,
            time,
            |field, full| self.target_field_dof(field, full),
            |field, full| self.prescribed_field_dof(field, full),
            |field| self.field_reduced_size(field),
            select,
        )
    }

    /// Assemble selected nonlinear state-dependent boundary kernels at `state`.
    pub fn assemble_state_boundary(
        &self,
        time: f64,
        state: MatRef<'_, f64>,
        terms: &StateBoundaryTerms,
    ) -> StateBoundaryContributions {
        let layout = self.field_layout();
        assert_eq!(state.nrows(), layout.total_size, "state size mismatch");
        assert_eq!(
            state.ncols(),
            1,
            "boundary assembly requires one state column"
        );
        StateBoundaryContributions {
            residual: assemble_quad_state_boundary_residual_cached(
                &self.state_boundary_cache,
                &self.fields,
                time,
                state,
                &self.cell_reduced_dofs,
                &self.cell_prescribed_values,
                &layout.offsets,
                terms,
            ),
            jacobian: self.assemble_state_boundary_jacobian(time, state, terms),
        }
    }

    pub fn apply_state_boundary_jacobian(
        &self,
        time: f64,
        state: MatRef<'_, f64>,
        direction: MatRef<'_, f64>,
        terms: &StateBoundaryTerms,
    ) -> Mat<f64> {
        let layout = self.field_layout();
        apply_quad_state_boundary_terms_cached(
            &self.state_boundary_cache,
            &self.fields,
            time,
            state,
            direction,
            &self.cell_reduced_dofs,
            &self.cell_prescribed_values,
            &layout.offsets,
            terms,
        )
    }

    fn assemble_state_boundary_jacobian(
        &self,
        time: f64,
        state: MatRef<'_, f64>,
        terms: &StateBoundaryTerms,
    ) -> SparseColMat<usize, f64> {
        let layout = self.field_layout();
        assemble_quad_state_boundary_jacobian(
            &self.mesh,
            &self.family,
            self.p,
            &self.metadata,
            &self.fields,
            time,
            state,
            &self.cell_reduced_dofs,
            &self.cell_prescribed_values,
            &layout.offsets,
            terms,
        )
    }
}
