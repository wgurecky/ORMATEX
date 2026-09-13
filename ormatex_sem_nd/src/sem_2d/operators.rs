//! Weak / tensor / mixed residual operators for 2D problems.
use crate::jacobian::CompleteResidualOperator;
use crate::kernels::kernel_common::{
    ResidualKernel, StateBoundaryTerms, StateTensorBoundaryTerms, TensorResidualKernel,
};
use faer::prelude::*;
use faer::sparse::SparseColMat;

use ndelement::types::ReferenceCellType;
use ndmesh::traits::Mesh;


use super::problem::SEM2DProblem;
use crate::sem_traits::WeakResidualOps;
/// Weak-form residual operator coupling a problem, a `ResidualKernel`, and
/// optional state-dependent facet terms (see [`with_state_boundary`](SEM2DResidualOperator::with_state_boundary)).
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
/// Retained (reduced) system size, including all fields.
    pub fn system_size(&self) -> usize {
        self.problem.system_size()
    }
/// Residual at the operator's time, including boundary terms when set.
    pub fn residual(&self, state: MatRef<f64>) -> Vec<f64> {
        let layout = self.problem.field_layout();
        let mut residual =
            self.problem
                .assemble_tensor_residual_with_layout(self.time, self.kernel, state, &layout);
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
/// Assembled Jacobian at `state`, including boundary terms when set.
    pub fn assemble_jacobian(&self, state: MatRef<f64>) -> SparseColMat<usize, f64> {
        let layout = self.problem.field_layout();
        let mut jacobian =
            self.problem
                .assemble_tensor_jacobian_with_layout(self.time, self.kernel, state, &layout);
        if let Some(terms) = self.terms {
            let boundary = self
                .problem
                .assemble_state_tensor_boundary_jacobian(self.time, state, terms);
            jacobian = jacobian.as_ref() + boundary.as_ref();
        }
        jacobian
    }
/// Matrix-free Jacobian action on one or more direction columns.
    pub fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        let mut out = Mat::<f64>::zeros(self.problem.system_size(), direction.ncols());
        let layout = self.problem.field_layout();
        self.problem.apply_tensor_jacobian_with_layout(
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
/// Matrix-free Jacobian action into caller-provided storage.
    pub fn apply_jacobian_into(
        &self,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        mut out: MatMut<'_, f64>,
    ) {
        let layout = self.problem.field_layout();
        self.problem.apply_tensor_jacobian_with_layout(
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
/// Return this operator at a new time.
    pub fn at_time(mut self, time: f64) -> Self {
        self.time = time;
        self
    }

/// Attach state-dependent natural-boundary terms.
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
/// Retained (reduced) system size, including all fields.
    pub fn system_size(&self) -> usize {
        self.problem.system_size()
    }

/// Residual at the operator's time, including boundary terms when set.
    pub fn residual(&self, state: MatRef<f64>) -> Vec<f64> {
        let layout = self.problem.field_layout();
        let mut residual = self.problem.assemble_tensor_residual_with_layout(
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

/// Assembled Jacobian at `state`, including boundary terms when set.
    pub fn assemble_jacobian(&self, state: MatRef<f64>) -> SparseColMat<usize, f64> {
        let layout = self.problem.field_layout();
        let tensor =
            self.problem
                .assemble_tensor_jacobian_with_layout(self.time, self.tensor_kernel, state, &layout);
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

/// Matrix-free Jacobian action on one or more direction columns.
    pub fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        let mut out = Mat::<f64>::zeros(self.system_size(), direction.ncols());
        let layout = self.problem.field_layout();
        self.problem.apply_tensor_jacobian_with_layout(
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

/// Matrix-free Jacobian action into caller-provided storage.
    pub fn apply_jacobian_into(
        &self,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        mut out: MatMut<'_, f64>,
    ) {
        let layout = self.problem.field_layout();
        self.problem.apply_tensor_jacobian_with_layout(
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

/// Return this operator at a new time.
    pub fn at_time(mut self, time: f64) -> Self {
        self.time = time;
        self
    }

/// Attach weak-form state-dependent boundary terms.
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

/// Attach tensor state-dependent boundary terms.
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
/// Retained (reduced) system size, including all fields.
    pub fn system_size(&self) -> usize {
        match self {
            Self::Weak(operator) => operator.system_size(),
            Self::Tensor(operator) => operator.system_size(),
            Self::Mixed(operator) => operator.system_size(),
        }
    }

/// Residual at the operator's time, including boundary terms when set.
    pub fn residual(&self, state: MatRef<f64>) -> Vec<f64> {
        match self {
            Self::Weak(operator) => operator.residual(state),
            Self::Tensor(operator) => operator.residual(state),
            Self::Mixed(operator) => operator.residual(state),
        }
    }

/// Assembled Jacobian at `state`, including boundary terms when set.
    pub fn assemble_jacobian(&self, state: MatRef<f64>) -> SparseColMat<usize, f64> {
        match self {
            Self::Weak(operator) => operator.assemble_jacobian(state),
            Self::Tensor(operator) => operator.assemble_jacobian(state),
            Self::Mixed(operator) => operator.assemble_jacobian(state),
        }
    }

/// Matrix-free Jacobian action on one or more direction columns.
    pub fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        match self {
            Self::Weak(operator) => operator.apply_jacobian(state, direction),
            Self::Tensor(operator) => operator.apply_jacobian(state, direction),
            Self::Mixed(operator) => operator.apply_jacobian(state, direction),
        }
    }

/// Matrix-free Jacobian action into caller-provided storage.
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

/// Return this operator at a new time.
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
/// Retained (reduced) system size, including all fields.
    pub fn system_size(&self) -> usize {
        self.problem.system_size()
    }

/// Residual at the operator's time, including boundary terms when set.
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

/// Assembled Jacobian at `state`, including boundary terms when set.
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

/// Matrix-free Jacobian action on one or more direction columns.
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

/// Return this operator at a new time.
    pub fn at_time(mut self, time: f64) -> Self {
        self.time = time;
        self
    }

/// Attach state-dependent natural-boundary terms.
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
    /// Build a weak-form residual operator for `kernel`.
    ///
    /// Validates the kernel's field selection against the problem up front.
    /// Use for non-tensor physics; attach facet fluxes later with
    /// [`with_state_boundary`](SEM2DResidualOperator::with_state_boundary).
    /// For sum-factorized kernels see
    /// [`tensor_residual_operator`](Self::tensor_residual_operator).
    pub fn residual_operator<'p, 'k, K: ResidualKernel + Sync>(
        &'p self,
        kernel: &'k K,
    ) -> SEM2DResidualOperator<'p, 'k, 'static, M, K>
    where
        M: Sync,
    {
        self.resolve_residual_selection(kernel, "residual kernel");
        SEM2DResidualOperator {
            problem: self,
            kernel,
            time: 0.0,
            terms: None,
        }
    }
    /// Build a statically dispatched tensor-product residual operator.
    ///
    /// Validates tensor field selection up front; volume assembly uses the
    /// [`tensor`](super::tensor) sum-factorized path. Use when the kernel
    /// implements `TensorResidualKernel<2>`; otherwise use
    /// [`residual_operator`](Self::residual_operator), or combine both with
    /// [`mixed_residual_operator`](Self::mixed_residual_operator).
    pub fn tensor_residual_operator<'p, 'k, K: TensorResidualKernel<2> + Sync>(
        &'p self,
        kernel: &'k K,
    ) -> SEM2DTensorResidualOperator<'p, 'k, 'static, M, K>
    where
        M: Sync,
    {
        self.fields.resolve_selection(
            kernel.input_nfields(),
            kernel.input_field_names(),
            kernel.output_nfields(),
            kernel.output_field_names(),
            "tensor residual kernel",
        );
        SEM2DTensorResidualOperator {
            problem: self,
            kernel,
            time: 0.0,
            terms: None,
        }
    }

    /// Build an operator summing one tensor and one weak-form kernel.
    ///
    /// Both kernels must agree on field count, names, and order. Residuals
    /// and Jacobian actions add volume contributions from each path, plus
    /// each path's own boundary terms when attached; use when only part of
    /// the physics has a tensor implementation.
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
}
