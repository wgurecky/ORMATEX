use faer::matrix_free::LinOp;
use faer::prelude::*;
use faer::sparse::SparseColMat;
use ndelement::types::ReferenceCellType;
use ndmesh::traits::Mesh;
use ormatex::ode_sys::OdeSys;
use ormatex_sem_nd::{KernelAdvDiff2D, MatrixFreeMinvJacobian, OwnedMinvJacobian, SEM2DProblem};

use super::linear_system::{lumped_inverse_mass, sparse_add};

#[derive(Clone, Copy)]
pub enum JacobianBackend {
    Assembled,
    MatrixFree,
}

pub struct ResidualDiffusionNeumannSys<'a, M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> {
    problem: &'a SEM2DProblem<M>,
    kernel: KernelAdvDiff2D,
    robin: SparseColMat<usize, f64>,
    source: Vec<f64>,
    m_inv: Vec<f64>,
    backend: JacobianBackend,
}

impl<'a, M> ResidualDiffusionNeumannSys<'a, M>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
{
    pub fn new(
        problem: &'a SEM2DProblem<M>,
        mass: SparseColMat<usize, f64>,
        kernel: KernelAdvDiff2D,
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
}

impl<'a, M> OdeSys<'a> for ResidualDiffusionNeumannSys<'a, M>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
{
    fn frhs(&self, t: f64, state: MatRef<f64>) -> Mat<f64> {
        let mut residual = self
            .problem
            .assemble_system_residual_at(t, &self.kernel, state);
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
            JacobianBackend::Assembled => Box::new(OwnedMinvJacobian::new(
                sparse_add(
                    self.problem
                        .assemble_system_residual_jacobian_at(t, &self.kernel, state)
                        .as_ref(),
                    self.robin.as_ref(),
                ),
                &self.m_inv,
            )),
            JacobianBackend::MatrixFree => Box::new(MatrixFreeMinvJacobian::new_at(
                t,
                self.problem,
                &self.kernel,
                state.to_owned(),
                Some(self.robin.as_ref()),
                &self.m_inv,
            )),
        }
    }
}
