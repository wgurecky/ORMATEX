use faer::dyn_stack::{MemStack, StackReq};
use faer::matrix_free::LinOp;
use faer::prelude::*;
use faer::sparse::{SparseColMat, SparseColMatRef};
use faer::Par;
use ormatex::ode_sys::OdeSys;

#[derive(Debug)]
pub struct MinvKLinOp<'a> {
    pub k: SparseColMatRef<'a, usize, f64>,
    pub m_inv: &'a [f64],
}

impl LinOp<f64> for MinvKLinOp<'_> {
    fn apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::empty()
    }

    fn nrows(&self) -> usize {
        self.k.nrows()
    }

    fn ncols(&self) -> usize {
        self.k.ncols()
    }

    fn apply(
        &self,
        mut out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        _par: Par,
        _stack: &mut MemStack,
    ) {
        let kv = self.k * rhs;
        for column in 0..out.ncols() {
            for row in 0..out.nrows() {
                out[(row, column)] = -self.m_inv[row] * kv[(row, column)];
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

pub struct AdvDiffSys {
    pub k: SparseColMat<usize, f64>,
    pub m_inv: Vec<f64>,
}

impl AdvDiffSys {
    pub fn new(mass: SparseColMat<usize, f64>, k: SparseColMat<usize, f64>) -> Self {
        let n = k.nrows();
        assert_eq!(mass.nrows(), n);
        assert_eq!(mass.ncols(), n);
        assert_eq!(
            mass.compute_nnz(),
            n,
            "expected lumped diagonal mass matrix"
        );
        let m_inv = (0..n)
            .map(|i| {
                let value = mass[(i, i)];
                assert!(value.abs() > 1e-30, "zero mass diagonal at {i}");
                1.0 / value
            })
            .collect();
        Self { k, m_inv }
    }

    pub fn apply_minv_k(&self, x: MatRef<f64>) -> Mat<f64> {
        let mut out = self.k.as_ref() * x;
        for column in 0..out.ncols() {
            for row in 0..out.nrows() {
                out[(row, column)] *= self.m_inv[row];
            }
        }
        out
    }
}

impl<'a> OdeSys<'a> for AdvDiffSys {
    fn frhs(&self, _t: f64, x: MatRef<f64>) -> Mat<f64> {
        faer::Scale(-1.0) * self.apply_minv_k(x)
    }

    fn fjac<'b>(&'a self, _t: f64, _x: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
        Box::new(MinvKLinOp {
            k: self.k.as_ref(),
            m_inv: &self.m_inv,
        })
    }
}
