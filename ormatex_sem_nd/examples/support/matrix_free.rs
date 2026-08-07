use faer::matrix_free::LinOp;
use faer::prelude::*;
use faer::sparse::{SparseColMat, SparseColMatRef, Triplet};
use ndelement::types::ReferenceCellType;
use ndmesh::traits::Mesh;
use ormatex::ode_sys::OdeSys;
use ormatex_sem_nd::{
    FiniteElement2DProblem, KernelAdvDiff2D, MatrixFreeMinvJacobian, OwnedMinvJacobian,
};

#[derive(Clone, Copy)]
pub enum JacobianBackend {
    Assembled,
    MatrixFree,
}

pub fn sparse_add(
    a: SparseColMatRef<'_, usize, f64>,
    b: SparseColMatRef<'_, usize, f64>,
) -> SparseColMat<usize, f64> {
    let n = a.nrows();
    assert_eq!(a.ncols(), n);
    assert_eq!(b.nrows(), n);
    assert_eq!(b.ncols(), n);
    let mut triplets = Vec::new();
    for mat in [a, b] {
        let (symbolic, values) = mat.parts();
        let columns = symbolic.col_ptr();
        let rows = symbolic.row_idx();
        for column in 0..n {
            for entry in columns[column]..columns[column + 1] {
                triplets.push(Triplet::new(rows[entry], column, values[entry]));
            }
        }
    }
    SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
}

pub struct ResidualDiffusionNeumannSys<'a, M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> {
    problem: &'a FiniteElement2DProblem<M>,
    kernel: KernelAdvDiff2D,
    robin: SparseColMat<usize, f64>,
    source: Vec<f64>,
    m_inv: Vec<f64>,
    backend: JacobianBackend,
}

impl<'a, M> ResidualDiffusionNeumannSys<'a, M>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>,
{
    pub fn new(
        problem: &'a FiniteElement2DProblem<M>,
        mass: SparseColMat<usize, f64>,
        kernel: KernelAdvDiff2D,
        robin: SparseColMat<usize, f64>,
        source: Vec<f64>,
        backend: JacobianBackend,
    ) -> Self {
        let n = problem.reduced_size();
        assert_eq!(mass.nrows(), n);
        assert_eq!(mass.ncols(), n);
        assert_eq!(
            mass.compute_nnz(),
            n,
            "matrix-free path requires lumped mass"
        );
        assert_eq!(robin.nrows(), n);
        assert_eq!(robin.ncols(), n);
        assert_eq!(source.len(), n);
        let m_inv = (0..n)
            .map(|i| {
                let value = mass[(i, i)];
                assert!(value.abs() > 1e-30, "zero mass diagonal at {i}");
                1.0 / value
            })
            .collect();
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
    fn frhs(&self, _t: f64, state: MatRef<f64>) -> Mat<f64> {
        let mut residual = self.problem.assemble_residual(&self.kernel, state);
        let robin_state = self.robin.as_ref() * state;
        for row in 0..residual.len() {
            residual[row] += robin_state[(row, 0)] - self.source[row];
        }
        Mat::from_fn(self.m_inv.len(), 1, |row, _| {
            -self.m_inv[row] * residual[row]
        })
    }

    fn fjac<'b>(&'a self, _t: f64, state: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
        match self.backend {
            JacobianBackend::Assembled => Box::new(OwnedMinvJacobian::new(
                sparse_add(
                    self.problem
                        .assemble_residual_jacobian(&self.kernel, state)
                        .as_ref(),
                    self.robin.as_ref(),
                ),
                &self.m_inv,
            )),
            JacobianBackend::MatrixFree => Box::new(MatrixFreeMinvJacobian::new(
                self.problem,
                &self.kernel,
                state.to_owned(),
                Some(self.robin.as_ref()),
                &self.m_inv,
            )),
        }
    }
}
