//! Matrix-free Jacobian version of `ex_nd_2d_diffusion_neumann`.
//!
//! The volume residual and its Jacobian action use `ResidualKernel`; the
//! existing linear Neumann/Robin boundary assembly supplies the fixed forcing
//! and Robin matrix.  Backward Euler therefore exercises the same PDE as the
//! assembled reference while `fjac` never builds a global volume Jacobian.

use std::fmt;

use faer::dyn_stack::{MemStack, StackReq};
use faer::matrix_free::LinOp;
use faer::prelude::*;
use faer::sparse::{SparseColMat, SparseColMatRef, Triplet};
use faer::Par;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{shapes::unit_square, traits::Mesh, SingleElementMesh};
use ormatex::ode_implicit::DirkIntegrator;
use ormatex::ode_sys::{IntegrateSys, OdeSys};
use ormatex::tableau_implicit::ImplicitBT;

#[path = "ex_nd_2d.rs"]
mod ex_nd_2d;
use ex_nd_2d::ex_nd_common::*;
use ex_nd_2d::{DofReduction2D, FiniteElement2DProblem};

type QuadMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

#[derive(Clone, Copy)]
enum JacobianBackend {
    Assembled,
    MatrixFree,
}

fn sparse_add(
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

#[derive(Debug)]
struct OwnedMinvJacobian<'a> {
    jacobian: SparseColMat<usize, f64>,
    m_inv: &'a [f64],
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

struct MatrixFreeMinvJacobian<'a, M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> {
    problem: &'a FiniteElement2DProblem<M>,
    kernel: &'a KernelAdvDiff2D,
    state: Mat<f64>,
    robin: SparseColMatRef<'a, usize, f64>,
    m_inv: &'a [f64],
}

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> fmt::Debug
    for MatrixFreeMinvJacobian<'_, M>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatrixFreeMinvJacobian")
            .field("ndofs", &self.m_inv.len())
            .finish()
    }
}

impl<M> LinOp<f64> for MatrixFreeMinvJacobian<'_, M>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
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
        let mut action = self
            .problem
            .apply_jacobian_matfree(self.kernel, self.state.as_ref(), rhs);
        action += self.robin * rhs;
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

struct ResidualDiffusionNeumannSys<'a, M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> {
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
    fn new(
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

    fn residual(&self, state: MatRef<f64>) -> Vec<f64> {
        let mut residual = self.problem.assemble_residual(&self.kernel, state);
        let robin_state = self.robin.as_ref() * state;
        for row in 0..residual.len() {
            residual[row] += robin_state[(row, 0)] - self.source[row];
        }
        residual
    }
}

impl<'a, M> OdeSys<'a> for ResidualDiffusionNeumannSys<'a, M>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
{
    fn frhs(&self, _t: f64, state: MatRef<f64>) -> Mat<f64> {
        let residual = self.residual(state);
        Mat::from_fn(self.m_inv.len(), 1, |row, _| {
            -self.m_inv[row] * residual[row]
        })
    }

    fn fjac<'b>(&'a self, _t: f64, state: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
        match self.backend {
            JacobianBackend::Assembled => Box::new(OwnedMinvJacobian {
                jacobian: sparse_add(
                    self.problem
                        .assemble_residual_jacobian(&self.kernel, state)
                        .as_ref(),
                    self.robin.as_ref(),
                ),
                m_inv: &self.m_inv,
            }),
            JacobianBackend::MatrixFree => Box::new(MatrixFreeMinvJacobian {
                problem: self.problem,
                kernel: &self.kernel,
                state: state.to_owned(),
                robin: self.robin.as_ref(),
                m_inv: &self.m_inv,
            }),
        }
    }
}

fn build_problem() -> (
    FiniteElement2DProblem<QuadMesh>,
    SparseColMat<usize, f64>,
    BoundaryContributions,
) {
    let mesh = unit_square(32, 2, ReferenceCellType::Quadrilateral);
    let problem = FiniteElement2DProblem::new(mesh, 2, DofReduction2D::None);
    let mass = problem.assemble_lumped_mass();
    let neumann = NeumannFlux::new(1.0);
    let robin = RobinConvection::new(0.1, 0.0);
    let boundary = problem.assemble_boundary(|facet| {
        const EPS: f64 = 1e-9;
        if facet.midpoint[0] < EPS {
            Some(&neumann)
        } else if facet.midpoint[0] > 1.0 - EPS {
            Some(&robin)
        } else {
            None
        }
    });
    (problem, mass, boundary)
}

fn run_case(backend: JacobianBackend, nsteps: usize) -> (Mat<f64>, Vec<(f64, f64)>) {
    let (problem, mass, boundary) = build_problem();
    let n = problem.reduced_size();
    let sys = ResidualDiffusionNeumannSys::new(
        &problem,
        mass,
        KernelAdvDiff2D::new(0.1, [0.0, 0.0]),
        boundary.mat,
        boundary.rhs,
        backend,
    );
    let mut y = Mat::<f64>::zeros(n, 1);
    let mut solver =
        DirkIntegrator::new(0.0, y.as_ref(), ImplicitBT::implicit_euler(), 1e-10, 1e-10);
    for _ in 0..nsteps {
        let step = solver.step(&sys, 1.0).unwrap();
        y = step.y.clone();
        solver.accept_step(step);
    }
    let positions = problem.dof_positions();
    assert!((solver.time() - nsteps as f64).abs() < 1e-12);
    (y, positions)
}

fn main() {
    let (y, positions) = run_case(JacobianBackend::MatrixFree, 200);
    let mut max_error = 0.0_f64;
    for row in 0..y.nrows() {
        let expected = -10.0 * positions[row].0 + 20.0;
        max_error = max_error.max((y[(row, 0)] - expected).abs());
    }
    assert!(
        max_error < 5e-3,
        "steady-state error {max_error} exceeds tolerance"
    );
    println!(
        "matrix-free diffusion-neumann complete; max |T| = {:.3e}",
        y.norm_max()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::dyn_stack::MemBuffer;

    #[test]
    fn matrix_free_jacobian_matches_assembled_action() {
        let (problem, mass, boundary) = build_problem();
        let n = problem.reduced_size();
        let kernel = KernelAdvDiff2D::new(0.1, [0.0, 0.0]);
        let state = Mat::from_fn(n, 1, |i, _| 0.25 + i as f64 / n as f64);
        let direction = Mat::from_fn(n, 1, |i, _| (i as f64 * 0.17).sin());
        let sys = ResidualDiffusionNeumannSys::new(
            &problem,
            mass,
            kernel,
            boundary.mat,
            boundary.rhs,
            JacobianBackend::MatrixFree,
        );
        let matrix_free_op = sys.fjac(0.0, state.as_ref());
        let mut matrix_free = Mat::zeros(n, 1);
        let mut scratch = MemBuffer::new(StackReq::empty());
        matrix_free_op.apply(
            matrix_free.as_mut(),
            direction.as_ref(),
            faer::get_global_parallelism(),
            MemStack::new(&mut scratch),
        );
        let assembled = sparse_add(
            problem
                .assemble_residual_jacobian(&sys.kernel, state.as_ref())
                .as_ref(),
            sys.robin.as_ref(),
        );
        let expected = assembled.as_ref() * direction.as_ref();
        for row in 0..n {
            assert!(
                (matrix_free[(row, 0)] + sys.m_inv[row] * expected[(row, 0)]).abs() < 1e-11,
                "matrix-free action mismatch at row {row}"
            );
        }
    }

    #[test]
    fn matrix_free_backward_euler_matches_assembled_backend() {
        // The executable runs all 200 requested steps. Two steps are enough
        // here to verify the Backward-Euler/Jacobian integration path without
        // making debug tests spend minutes in unpreconditioned GMRES.
        let (matrix_free, _) = run_case(JacobianBackend::MatrixFree, 2);
        let (assembled, _) = run_case(JacobianBackend::Assembled, 2);
        for row in 0..matrix_free.nrows() {
            assert!((matrix_free[(row, 0)] - assembled[(row, 0)]).abs() < 1e-8);
        }
    }
}
