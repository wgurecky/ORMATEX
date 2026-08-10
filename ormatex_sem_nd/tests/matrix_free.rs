use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::prelude::*;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{shapes::unit_square, SingleElementMesh};
use ormatex::ode_implicit::DirkIntegrator;
use ormatex::ode_sys::{IntegrateSys, OdeSys};
use ormatex::tableau_implicit::ImplicitBT;
use ormatex_sem_nd::{
    DofReduction2D, SEM2DProblem, KernelAdvDiff2D, NeumannFlux, RobinConvection,
};

#[path = "../examples/support/matrix_free.rs"]
mod matrix_free;
use matrix_free::{sparse_add, JacobianBackend, ResidualDiffusionNeumannSys};

type QuadMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

fn build_problem() -> (
    SEM2DProblem<QuadMesh>,
    faer::sparse::SparseColMat<usize, f64>,
    ormatex_sem_nd::BoundaryContributions,
) {
    let mesh = unit_square(32, 2, ReferenceCellType::Quadrilateral);
    let problem = SEM2DProblem::new(mesh, 2, DofReduction2D::None);
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

fn run_case(backend: JacobianBackend, nsteps: usize) -> Mat<f64> {
    let (problem, mass, boundary) = build_problem();
    let n = problem.reduced_size();
    let system = ResidualDiffusionNeumannSys::new(
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
        let step = solver.step(&system, 1.0).unwrap();
        y = step.y.clone();
        solver.accept_step(step);
    }
    y
}

#[test]
fn matrix_free_jacobian_matches_assembled_action() {
    let (problem, mass, boundary) = build_problem();
    let n = problem.reduced_size();
    let kernel = KernelAdvDiff2D::new(0.1, [0.0, 0.0]);
    let state = Mat::from_fn(n, 1, |i, _| 0.25 + i as f64 / n as f64);
    let direction = Mat::from_fn(n, 1, |i, _| (i as f64 * 0.17).sin());
    let assembled = sparse_add(
        problem
            .assemble_residual_jacobian(&kernel, state.as_ref())
            .as_ref(),
        boundary.mat.as_ref(),
    );
    let system = ResidualDiffusionNeumannSys::new(
        &problem,
        mass,
        kernel,
        boundary.mat,
        boundary.rhs,
        JacobianBackend::MatrixFree,
    );
    let operator = system.fjac(0.0, state.as_ref());
    let mut action = Mat::zeros(n, 1);
    let mut scratch = MemBuffer::new(StackReq::empty());
    operator.apply(
        action.as_mut(),
        direction.as_ref(),
        faer::get_global_parallelism(),
        MemStack::new(&mut scratch),
    );
    let expected = assembled.as_ref() * direction.as_ref();
    let mass_again = problem.assemble_lumped_mass();
    for row in 0..n {
        assert!((action[(row, 0)] + expected[(row, 0)] / mass_again[(row, row)]).abs() < 1e-11);
    }
}

#[test]
fn matrix_free_backward_euler_matches_assembled_backend() {
    let matrix_free = run_case(JacobianBackend::MatrixFree, 2);
    let assembled = run_case(JacobianBackend::Assembled, 2);
    for row in 0..matrix_free.nrows() {
        assert!((matrix_free[(row, 0)] - assembled[(row, 0)]).abs() < 1e-8);
    }
}
