//! Reusable assembled and matrix-free Jacobian linear operators.

use std::fmt;

use faer::dyn_stack::{MemStack, StackReq};
use faer::matrix_free::LinOp;
use faer::prelude::*;
use faer::sparse::{SparseColMat, SparseColMatRef};
use faer::Par;

use crate::{FiniteElement1DProblem, FiniteElement2DProblem, ResidualKernel};
use ndelement::types::ReferenceCellType;
use ndmesh::traits::Mesh;

/// A finite-element problem that can apply a residual Jacobian without
/// assembling a global sparse matrix.
pub trait MatrixFreeJacobianProblem: Sync {
    fn reduced_size(&self) -> usize;

    fn apply_residual_jacobian_matfree<K: ResidualKernel>(
        &self,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
    ) -> Mat<f64>;
}

impl<M> MatrixFreeJacobianProblem for FiniteElement1DProblem<M>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
{
    fn reduced_size(&self) -> usize {
        FiniteElement1DProblem::reduced_size(self)
    }

    fn apply_residual_jacobian_matfree<K: ResidualKernel>(
        &self,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
    ) -> Mat<f64> {
        FiniteElement1DProblem::apply_jacobian_matfree(self, kernel, state, direction)
    }
}

impl<M> MatrixFreeJacobianProblem for FiniteElement2DProblem<M>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
{
    fn reduced_size(&self) -> usize {
        FiniteElement2DProblem::reduced_size(self)
    }

    fn apply_residual_jacobian_matfree<K: ResidualKernel>(
        &self,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
    ) -> Mat<f64> {
        FiniteElement2DProblem::apply_jacobian_matfree(self, kernel, state, direction)
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

/// Applies `-M^-1 J` by assembling only the local Jacobian action. An
/// optional fixed sparse block supports linear terms outside the volume
/// `ResidualKernel`, such as the Robin matrix in the 2D diffusion example.
pub struct MatrixFreeMinvJacobian<'a, P: MatrixFreeJacobianProblem, K: ResidualKernel> {
    problem: &'a P,
    kernel: &'a K,
    state: Mat<f64>,
    fixed_jacobian: Option<SparseColMatRef<'a, usize, f64>>,
    m_inv: &'a [f64],
}

impl<'a, P: MatrixFreeJacobianProblem, K: ResidualKernel> MatrixFreeMinvJacobian<'a, P, K> {
    pub fn new(
        problem: &'a P,
        kernel: &'a K,
        state: Mat<f64>,
        fixed_jacobian: Option<SparseColMatRef<'a, usize, f64>>,
        m_inv: &'a [f64],
    ) -> Self {
        let n = problem.reduced_size();
        assert_eq!(state.nrows(), n, "state/problem size mismatch");
        assert_eq!(
            state.ncols(),
            1,
            "matrix-free Jacobian requires one state column"
        );
        assert_eq!(m_inv.len(), n, "mass/problem size mismatch");
        if let Some(fixed) = fixed_jacobian {
            assert_eq!(fixed.nrows(), n, "fixed Jacobian/problem row mismatch");
            assert_eq!(fixed.ncols(), n, "fixed Jacobian/problem column mismatch");
        }
        Self {
            problem,
            kernel,
            state,
            fixed_jacobian,
            m_inv,
        }
    }
}

impl<P: MatrixFreeJacobianProblem, K: ResidualKernel> fmt::Debug
    for MatrixFreeMinvJacobian<'_, P, K>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatrixFreeMinvJacobian")
            .field("ndofs", &self.m_inv.len())
            .finish()
    }
}

impl<P: MatrixFreeJacobianProblem, K: ResidualKernel + Sync> LinOp<f64>
    for MatrixFreeMinvJacobian<'_, P, K>
{
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
        let mut action =
            self.problem
                .apply_residual_jacobian_matfree(self.kernel, self.state.as_ref(), rhs);
        if let Some(fixed) = self.fixed_jacobian {
            action += fixed * rhs;
        }
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
