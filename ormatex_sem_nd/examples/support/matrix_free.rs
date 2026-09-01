use faer::matrix_free::LinOp;
use faer::prelude::*;
use faer::sparse::{SparseColMat, SparseColMatRef};
use ndelement::types::ReferenceCellType;
use ndmesh::traits::Mesh;
use ormatex::ode_sys::OdeSys;
use ormatex_sem_nd::{
    KernelAdvDiff2D, MatrixFreeJacobianSource, MatrixFreeMinvJacobian, OwnedMinvJacobian,
    SEM2DProblem, TensorKernelAdvDiff2D,
};

use super::linear_system::{lumped_inverse_mass, sparse_add};

#[derive(Clone, Copy)]
pub enum JacobianBackend {
    Assembled,
    MatrixFree,
}

enum ResidualDiffusionKernel {
    Weak(KernelAdvDiff2D),
    Tensor(TensorKernelAdvDiff2D),
}

struct FixedJacobianSource<'a, S> {
    source: S,
    fixed: SparseColMatRef<'a, usize, f64>,
}

impl<S> MatrixFreeJacobianSource for FixedJacobianSource<'_, S>
where
    S: MatrixFreeJacobianSource,
{
    fn system_size(&self) -> usize {
        self.source.system_size()
    }

    fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        self.source.apply_jacobian(state, direction) + self.fixed * direction
    }
}

pub struct ResidualDiffusionNeumannSys<'a, M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> {
    problem: &'a SEM2DProblem<M>,
    kernel: ResidualDiffusionKernel,
    robin: SparseColMat<usize, f64>,
    source: Vec<f64>,
    m_inv: Vec<f64>,
    backend: JacobianBackend,
}

impl<'a, M> ResidualDiffusionNeumannSys<'a, M>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
{
    fn from_parts(
        problem: &'a SEM2DProblem<M>,
        mass: SparseColMat<usize, f64>,
        kernel: ResidualDiffusionKernel,
        robin: SparseColMat<usize, f64>,
        source: Vec<f64>,
        backend: JacobianBackend,
    ) -> Self {
        let n = problem.reduced_size();
        assert_eq!(mass.nrows(), n);
        assert_eq!(mass.ncols(), n);
        assert_eq!(robin.nrows(), n);
        assert_eq!(robin.ncols(), n);
        assert_eq!(source.len(), n);
        let m_inv = lumped_inverse_mass(mass.as_ref());
        Self {
            problem,
            kernel,
            robin,
            source,
            m_inv,
            backend,
        }
    }

    pub fn new(
        problem: &'a SEM2DProblem<M>,
        mass: SparseColMat<usize, f64>,
        kernel: KernelAdvDiff2D,
        robin: SparseColMat<usize, f64>,
        source: Vec<f64>,
        backend: JacobianBackend,
    ) -> Self {
        Self::from_parts(
            problem,
            mass,
            ResidualDiffusionKernel::Weak(kernel),
            robin,
            source,
            backend,
        )
    }

    pub fn new_tensor(
        problem: &'a SEM2DProblem<M>,
        mass: SparseColMat<usize, f64>,
        kernel: TensorKernelAdvDiff2D,
        robin: SparseColMat<usize, f64>,
        source: Vec<f64>,
        backend: JacobianBackend,
    ) -> Self {
        Self::from_parts(
            problem,
            mass,
            ResidualDiffusionKernel::Tensor(kernel),
            robin,
            source,
            backend,
        )
    }
}

impl<'a, M> OdeSys<'a> for ResidualDiffusionNeumannSys<'a, M>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
{
    fn frhs(&self, t: f64, state: MatRef<f64>) -> Mat<f64> {
        let mut residual = match &self.kernel {
            ResidualDiffusionKernel::Weak(kernel) => {
                self.problem.assemble_residual(t, kernel, state)
            }
            ResidualDiffusionKernel::Tensor(kernel) => self
                .problem
                .tensor_residual_operator(kernel)
                .at_time(t)
                .residual(state),
        };
        let robin_state = self.robin.as_ref() * state;
        for row in 0..residual.len() {
            residual[row] += robin_state[(row, 0)] - self.source[row];
        }
        Mat::from_fn(self.m_inv.len(), 1, |row, _| {
            -self.m_inv[row] * residual[row]
        })
    }

    fn fjac<'b>(&'a self, t: f64, state: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
        match self.backend {
            JacobianBackend::Assembled => {
                let volume = match &self.kernel {
                    ResidualDiffusionKernel::Weak(kernel) => {
                        self.problem.assemble_residual_jacobian(t, kernel, state)
                    }
                    ResidualDiffusionKernel::Tensor(kernel) => self
                        .problem
                        .tensor_residual_operator(kernel)
                        .at_time(t)
                        .assemble_jacobian(state),
                };
                Box::new(OwnedMinvJacobian::new(
                    sparse_add(volume.as_ref(), self.robin.as_ref()),
                    &self.m_inv,
                ))
            }
            JacobianBackend::MatrixFree => match &self.kernel {
                ResidualDiffusionKernel::Weak(kernel) => Box::new(MatrixFreeMinvJacobian::new_at(
                    t,
                    self.problem,
                    kernel,
                    state.to_owned(),
                    Some(self.robin.as_ref()),
                    &self.m_inv,
                )),
                ResidualDiffusionKernel::Tensor(kernel) => {
                    let operator = self.problem.tensor_residual_operator(kernel).at_time(t);
                    let source = FixedJacobianSource {
                        source: operator,
                        fixed: self.robin.as_ref(),
                    };
                    Box::new(MatrixFreeMinvJacobian::new(
                        source,
                        state.to_owned(),
                        &self.m_inv,
                    ))
                }
            },
        }
    }
}
