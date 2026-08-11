use faer::dyn_stack::{MemStack, StackReq};
use faer::matrix_free::LinOp;
use faer::prelude::*;
use faer::sparse::{SparseColMat, SparseColMatRef, Triplet};
use faer::Par;
use ormatex::ode_implicit::DirkIntegrator;
use ormatex::ode_sys::{IntegrateSys, OdeSys};
use ormatex::tableau_implicit::ImplicitBT;

pub fn lumped_inverse_mass(mass: SparseColMatRef<'_, usize, f64>) -> Vec<f64> {
    assert_eq!(mass.nrows(), mass.ncols(), "mass matrix must be square");
    assert_eq!(
        mass.compute_nnz(),
        mass.nrows(),
        "expected lumped diagonal mass matrix"
    );
    (0..mass.nrows())
        .map(|i| {
            let value = mass[(i, i)];
            assert!(value.abs() > 1e-30, "zero mass diagonal at {i}");
            1.0 / value
        })
        .collect()
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

pub fn implicit_euler_final_state<'a>(
    system: &'a dyn OdeSys<'a>,
    y0: MatRef<'_, f64>,
    dt: f64,
    nsteps: usize,
    tolerance: f64,
) -> Mat<f64> {
    let mut solver =
        DirkIntegrator::new(0.0, y0, ImplicitBT::implicit_euler(), tolerance, tolerance);
    for _ in 0..nsteps {
        let step = solver.step(system, dt).unwrap();
        solver.accept_step(step);
    }
    solver.state()
}

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

pub struct LinearOdeSys {
    k: SparseColMat<usize, f64>,
    m_inv: Vec<f64>,
    source: Vec<f64>,
}

impl LinearOdeSys {
    pub fn new(
        mass: SparseColMat<usize, f64>,
        k: SparseColMat<usize, f64>,
        source: Vec<f64>,
    ) -> Self {
        let n = k.nrows();
        assert_eq!(k.ncols(), n);
        assert_eq!(mass.nrows(), n);
        assert_eq!(mass.ncols(), n);
        assert_eq!(source.len(), n);
        let m_inv = lumped_inverse_mass(mass.as_ref());
        Self { k, m_inv, source }
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

impl<'a> OdeSys<'a> for LinearOdeSys {
    fn frhs(&self, _t: f64, x: MatRef<f64>) -> Mat<f64> {
        let kx = self.k.as_ref() * x;
        Mat::from_fn(x.nrows(), x.ncols(), |row, column| {
            -self.m_inv[row] * (kx[(row, column)] - self.source[row])
        })
    }

    fn fjac<'b>(&'a self, _t: f64, _x: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
        Box::new(MinvKLinOp {
            k: self.k.as_ref(),
            m_inv: &self.m_inv,
        })
    }
}
