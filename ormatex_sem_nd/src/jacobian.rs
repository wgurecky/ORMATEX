//! Reusable assembled and matrix-free Jacobian linear operators.

use std::fmt;

use faer::dyn_stack::{MemStack, StackReq};
use faer::matrix_free::LinOp;
use faer::prelude::*;
use faer::sparse::{SparseColMat, SparseColMatRef};
use faer::Par;

use crate::kernels::kernel_common::ResidualKernel;
use crate::op::ParCsrJacobian;
use crate::simd;
use crate::{SEM1DProblem, SEM2DProblem};
use ndelement::types::ReferenceCellType;
use ndmesh::traits::Mesh;

/// Complete nonlinear residual operator, including any boundary terms.
pub trait CompleteResidualOperator: Sync {
    fn system_size(&self) -> usize;

    fn residual(&self, state: MatRef<f64>) -> Vec<f64>;

    fn assemble_jacobian(&self, state: MatRef<f64>) -> SparseColMat<usize, f64>;

    fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64>;

    fn apply_jacobian_into(
        &self,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        mut out: MatMut<'_, f64>,
    ) {
        let action = self.apply_jacobian(state, direction);
        out.copy_from(action.as_ref());
    }
}

/// Source of a residual Jacobian action for a matrix-free linear operator.
///
/// A source may combine volume terms with any other residual terms, including
/// state-dependent natural-boundary contributions.
pub trait MatrixFreeJacobianSource: Sync {
    /// Number of rows and columns in the residual system.
    fn system_size(&self) -> usize;

    /// Apply the residual Jacobian at `state` to one or more directions.
    fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64>;

    /// Apply a residual Jacobian into caller-provided storage.
    fn apply_jacobian_into(
        &self,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        mut out: MatMut<'_, f64>,
    ) {
        let action = self.apply_jacobian(state, direction);
        out.copy_from(action.as_ref());
    }
}

impl<O: CompleteResidualOperator> MatrixFreeJacobianSource for O {
    fn system_size(&self) -> usize {
        CompleteResidualOperator::system_size(self)
    }

    fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        CompleteResidualOperator::apply_jacobian(self, state, direction)
    }

    fn apply_jacobian_into(
        &self,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        out: MatMut<'_, f64>,
    ) {
        CompleteResidualOperator::apply_jacobian_into(self, state, direction, out)
    }
}

/// A finite-element problem that can apply a residual Jacobian without
/// assembling a global sparse matrix.
pub trait MatrixFreeJacobianProblem: Sync {
    fn reduced_size(&self) -> usize;

    fn system_size(&self, nfields: usize) -> usize;

    fn validate_kernel_fields<K: ResidualKernel>(&self, _kernel: &K) {}

    fn apply_jacobian<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
    ) -> Mat<f64>;

    /// Apply a residual Jacobian into caller-provided storage.
    fn apply_jacobian_into<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        mut out: MatMut<'_, f64>,
    ) {
        let action = self.apply_jacobian(time, kernel, state, direction);
        out.copy_from(action.as_ref());
    }
}

impl<M> MatrixFreeJacobianProblem for SEM1DProblem<M>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
{
    fn reduced_size(&self) -> usize {
        SEM1DProblem::reduced_size(self)
    }

    fn system_size(&self, nfields: usize) -> usize {
        let _ = nfields;
        SEM1DProblem::system_size(self)
    }

    fn validate_kernel_fields<K: ResidualKernel>(&self, kernel: &K) {
        self.fields().resolve_selection(
            kernel.input_nfields(),
            kernel.input_field_names(),
            kernel.output_nfields(),
            kernel.output_field_names(),
            "residual kernel",
        );
    }

    fn apply_jacobian<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
    ) -> Mat<f64> {
        SEM1DProblem::apply_jacobian(self, time, kernel, state, direction)
    }

    fn apply_jacobian_into<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        out: MatMut<'_, f64>,
    ) {
        SEM1DProblem::apply_jacobian_into(self, time, kernel, state, direction, out)
    }
}

impl<M> MatrixFreeJacobianProblem for SEM2DProblem<M>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
{
    fn reduced_size(&self) -> usize {
        SEM2DProblem::reduced_size(self)
    }

    fn system_size(&self, nfields: usize) -> usize {
        let _ = nfields;
        SEM2DProblem::system_size(self)
    }

    fn validate_kernel_fields<K: ResidualKernel>(&self, kernel: &K) {
        self.fields().resolve_selection(
            kernel.input_nfields(),
            kernel.input_field_names(),
            kernel.output_nfields(),
            kernel.output_field_names(),
            "residual kernel",
        );
    }

    fn apply_jacobian<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
    ) -> Mat<f64> {
        SEM2DProblem::apply_jacobian(self, time, kernel, state, direction)
    }

    fn apply_jacobian_into<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        out: MatMut<'_, f64>,
    ) {
        SEM2DProblem::apply_jacobian_into(self, time, kernel, state, direction, out)
    }
}

/// Applies `-M^-1 J` for an assembled residual Jacobian `J` and diagonal
/// lumped mass matrix `M`.
#[derive(Debug)]
pub struct OwnedMinvJacobian<'a> {
    jacobian: SparseColMat<usize, f64>,
    m_inv: &'a [f64],
}

impl<'a> OwnedMinvJacobian<'a> {
    pub fn new(jacobian: SparseColMat<usize, f64>, m_inv: &'a [f64]) -> Self {
        assert_eq!(
            jacobian.nrows(),
            jacobian.ncols(),
            "Jacobian must be square"
        );
        assert_eq!(jacobian.nrows(), m_inv.len(), "Jacobian/mass size mismatch");
        Self { jacobian, m_inv }
    }
}

impl LinOp<f64> for OwnedMinvJacobian<'_> {
    fn apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::empty()
    }

    fn nrows(&self) -> usize {
        self.jacobian.nrows()
    }

    fn ncols(&self) -> usize {
        self.jacobian.ncols()
    }

    fn apply(
        &self,
        mut out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        _par: Par,
        _stack: &mut MemStack,
    ) {
        let action = self.jacobian.as_ref() * rhs;
        for column in 0..out.ncols() {
            for row in 0..out.nrows() {
                out[(row, column)] = -self.m_inv[row] * action[(row, column)];
            }
        }
    }

    fn conj_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
}

/// Applies `-M^-1 J` using parallel sparse products for an assembled Jacobian.
#[derive(Debug)]
pub struct ParallelOwnedMinvJacobian<'a> {
    jacobian: ParCsrJacobian,
    m_inv: &'a [f64],
}

impl<'a> ParallelOwnedMinvJacobian<'a> {
    pub fn new(jacobian: SparseColMat<usize, f64>, m_inv: &'a [f64]) -> Self {
        assert_eq!(
            jacobian.nrows(),
            jacobian.ncols(),
            "Jacobian must be square"
        );
        assert_eq!(jacobian.nrows(), m_inv.len(), "Jacobian/mass size mismatch");
        let n_threads = faer::get_global_parallelism().degree().max(1);
        Self {
            jacobian: ParCsrJacobian::new(jacobian, n_threads),
            m_inv,
        }
    }
}

impl LinOp<f64> for ParallelOwnedMinvJacobian<'_> {
    fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        self.jacobian.apply_scratch(rhs_ncols, par)
    }

    fn nrows(&self) -> usize {
        self.jacobian.nrows()
    }

    fn ncols(&self) -> usize {
        self.jacobian.ncols()
    }

    fn apply(
        &self,
        mut out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.jacobian.apply(out.rb_mut(), rhs, par, stack);
        for column in 0..out.ncols() {
            for row in 0..out.nrows() {
                out[(row, column)] *= -self.m_inv[row];
            }
        }
    }

    fn conj_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
}

enum FixedJacobian<'a> {
    Borrowed(SparseColMatRef<'a, usize, f64>),
    Owned(SparseColMat<usize, f64>),
}

/// Adapts a finite-element problem and residual kernel into a
/// [`MatrixFreeJacobianSource`]. The problem supplies the volume Jacobian
/// action; an optional fixed sparse matrix supplies additional linear terms
/// outside the residual kernel.
struct FiniteElementJacobianSource<'a, P, K> {
    problem: &'a P,
    kernel: &'a K,
    time: f64,
    fixed_jacobian: Option<FixedJacobian<'a>>,
}

impl<'a, P, K> FiniteElementJacobianSource<'a, P, K>
where
    P: MatrixFreeJacobianProblem + 'a,
    K: ResidualKernel + Sync + 'a,
{
    fn new(
        time: f64,
        problem: &'a P,
        kernel: &'a K,
        fixed_jacobian: Option<FixedJacobian<'a>>,
    ) -> Self {
        problem.validate_kernel_fields(kernel);
        let n = problem.system_size(kernel.nfields());
        if let Some(ref fixed) = fixed_jacobian {
            let (nrows, ncols) = match fixed {
                FixedJacobian::Borrowed(fixed) => (fixed.nrows(), fixed.ncols()),
                FixedJacobian::Owned(fixed) => (fixed.nrows(), fixed.ncols()),
            };
            assert_eq!(nrows, n, "fixed Jacobian/problem row mismatch");
            assert_eq!(ncols, n, "fixed Jacobian/problem column mismatch");
        }
        Self {
            problem,
            kernel,
            time,
            fixed_jacobian,
        }
    }
}

impl<P, K> MatrixFreeJacobianSource for FiniteElementJacobianSource<'_, P, K>
where
    P: MatrixFreeJacobianProblem,
    K: ResidualKernel + Sync,
{
    fn system_size(&self) -> usize {
        self.problem.system_size(self.kernel.nfields())
    }

    fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        let mut action = self
            .problem
            .apply_jacobian(self.time, self.kernel, state, direction);
        match self.fixed_jacobian.as_ref() {
            Some(FixedJacobian::Borrowed(fixed)) => action += *fixed * direction,
            Some(FixedJacobian::Owned(fixed)) => action += fixed.as_ref() * direction,
            None => {}
        }
        action
    }

    fn apply_jacobian_into(
        &self,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        mut out: MatMut<'_, f64>,
    ) {
        self.problem
            .apply_jacobian_into(self.time, self.kernel, state, direction, out.rb_mut());
        match self.fixed_jacobian.as_ref() {
            Some(FixedJacobian::Borrowed(fixed)) => out += *fixed * direction,
            Some(FixedJacobian::Owned(fixed)) => out += fixed.as_ref() * direction,
            None => {}
        }
    }
}

/// Applies `-M^-1 J` for any matrix-free residual Jacobian source.
pub struct MatrixFreeMinvJacobian<'a> {
    source: Box<dyn MatrixFreeJacobianSource + 'a>,
    state: Mat<f64>,
    m_inv: &'a [f64],
}

impl<'a> MatrixFreeMinvJacobian<'a> {
    /// Build an operator from a complete residual Jacobian source.
    pub fn new<S>(source: S, state: Mat<f64>, m_inv: &'a [f64]) -> Self
    where
        S: MatrixFreeJacobianSource + 'a,
    {
        assert_eq!(
            state.nrows(),
            source.system_size(),
            "state/operator size mismatch"
        );
        assert_eq!(
            state.ncols(),
            1,
            "matrix-free Jacobian requires one state column"
        );
        assert_eq!(
            m_inv.len(),
            source.system_size(),
            "mass/operator size mismatch"
        );
        Self {
            source: Box::new(source),
            state,
            m_inv,
        }
    }

    pub fn new_at<P, K>(
        time: f64,
        problem: &'a P,
        kernel: &'a K,
        state: Mat<f64>,
        fixed_jacobian: Option<SparseColMatRef<'a, usize, f64>>,
        m_inv: &'a [f64],
    ) -> Self
    where
        P: MatrixFreeJacobianProblem + 'a,
        K: ResidualKernel + Sync + 'a,
    {
        Self::new(
            FiniteElementJacobianSource::new(
                time,
                problem,
                kernel,
                fixed_jacobian.map(FixedJacobian::Borrowed),
            ),
            state,
            m_inv,
        )
    }

    pub fn new_at_with_owned_fixed_jacobian<P, K>(
        time: f64,
        problem: &'a P,
        kernel: &'a K,
        state: Mat<f64>,
        fixed_jacobian: Option<SparseColMat<usize, f64>>,
        m_inv: &'a [f64],
    ) -> Self
    where
        P: MatrixFreeJacobianProblem + 'a,
        K: ResidualKernel + Sync + 'a,
    {
        Self::new(
            FiniteElementJacobianSource::new(
                time,
                problem,
                kernel,
                fixed_jacobian.map(FixedJacobian::Owned),
            ),
            state,
            m_inv,
        )
    }
}

impl fmt::Debug for MatrixFreeMinvJacobian<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatrixFreeMinvJacobian")
            .field("ndofs", &self.m_inv.len())
            .finish()
    }
}

impl LinOp<f64> for MatrixFreeMinvJacobian<'_> {
    fn apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::empty()
    }

    fn nrows(&self) -> usize {
        self.m_inv.len()
    }

    fn ncols(&self) -> usize {
        self.m_inv.len()
    }

    fn apply(
        &self,
        mut out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        _par: Par,
        _stack: &mut MemStack,
    ) {
        self.source
            .apply_jacobian_into(self.state.as_ref(), rhs, out.rb_mut());
        let ncols = out.ncols();
        if let Some(mut contiguous) = out.rb_mut().try_as_col_major_mut() {
            for column in 0..ncols {
                let input = contiguous.rb_mut().col_mut(column).as_slice_mut();
                simd::scale_negate_in_place(input, self.m_inv);
            }
        } else {
            for column in 0..ncols {
                for row in 0..self.m_inv.len() {
                    out[(row, column)] *= -self.m_inv[row];
                }
            }
        }
    }

    fn conj_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
}
