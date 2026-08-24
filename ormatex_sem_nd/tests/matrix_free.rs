use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{shapes::unit_square, SingleElementMesh};
use ormatex::ode_sys::OdeSys;
use ormatex_sem_nd::{
    BoundaryIntegrator, DofReduction2D, FieldRegistry, KernelAdvDiff2D, NeumannFlux,
    RobinConvection, SEM2DProblem,
};
use rayon::ThreadPoolBuilder;
use std::time::Instant;

#[path = "../examples/support/linear_system.rs"]
mod linear_system;
#[path = "../examples/support/matrix_free.rs"]
mod matrix_free;
use linear_system::{implicit_euler_final_state, sparse_add, LinearOdeSys};
use matrix_free::{JacobianBackend, ResidualDiffusionNeumannSys};

type QuadMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

fn build_problem() -> (
    SEM2DProblem<QuadMesh>,
    faer::sparse::SparseColMat<usize, f64>,
    ormatex_sem_nd::BoundaryContributions,
) {
    let mesh = unit_square(32, 2, ReferenceCellType::Quadrilateral, 1);
    let problem = SEM2DProblem::new(
        mesh,
        2,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::None,
    );
    let mass = problem.assemble_lumped_mass();
    let neumann = NeumannFlux::new(1.0);
    let robin = RobinConvection::new(0.1, 0.0);
    let boundary = problem.assemble_boundary(0.0, |facet| -> Option<&dyn BoundaryIntegrator> {
        const EPS: f64 = 1e-9;
        if facet.midpoint[0] < EPS {
            Some(&neumann as &dyn BoundaryIntegrator)
        } else if facet.midpoint[0] > 1.0 - EPS {
            Some(&robin as &dyn BoundaryIntegrator)
        } else {
            None
        }
    });
    (problem, mass, boundary)
}

fn build_large_diffusion_case() -> (SEM2DProblem<QuadMesh>, KernelAdvDiff2D, Mat<f64>, Mat<f64>) {
    let mesh = unit_square(64, 64, ReferenceCellType::Quadrilateral, 1);
    let problem = SEM2DProblem::new(
        mesh,
        2,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::None,
    );
    let n = problem.reduced_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.25 + i as f64 / n as f64);
    let direction = Mat::from_fn(n, 1, |i, _| (i as f64 * 0.17).sin());
    (
        problem,
        KernelAdvDiff2D::new(0.1, [0.0, 0.0]),
        state,
        direction,
    )
}

fn benchmark_threads() -> (usize, usize) {
    let available = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    (1, available.min(8))
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
    let y0 = Mat::<f64>::zeros(n, 1);
    implicit_euler_final_state(&system, y0.as_ref(), 1.0, nsteps, 1e-10)
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
            .assemble_residual_jacobian(0.0, &kernel, state.as_ref())
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

#[test]
fn linear_ode_system_applies_source_with_positive_sign() {
    let mass = SparseColMat::try_new_from_triplets(1, 1, &[Triplet::new(0, 0, 2.0)]).unwrap();
    let operator = SparseColMat::try_new_from_triplets(1, 1, &[Triplet::new(0, 0, 4.0)]).unwrap();
    let system = LinearOdeSys::new(mass, operator, vec![3.0]);
    let state = Mat::from_fn(1, 1, |_, _| 2.0);
    assert_eq!(system.frhs(0.0, state.as_ref())[(0, 0)], -2.5);
}

#[test]
#[ignore = "large runtime benchmark; run with --release -- --ignored --nocapture"]
fn large_2d_diffusion_matrix_free_jacobian_runtime() {
    let (problem, kernel, state, direction) = build_large_diffusion_case();
    let (serial_threads, parallel_threads) = benchmark_threads();
    let serial_pool = ThreadPoolBuilder::new()
        .num_threads(serial_threads)
        .build()
        .unwrap();
    let parallel_pool = ThreadPoolBuilder::new()
        .num_threads(parallel_threads)
        .build()
        .unwrap();

    let start = Instant::now();
    let serial = serial_pool
        .install(|| problem.apply_jacobian(0.0, &kernel, state.as_ref(), direction.as_ref()));
    let serial_time = start.elapsed();

    let start = Instant::now();
    let parallel = parallel_pool
        .install(|| problem.apply_jacobian(0.0, &kernel, state.as_ref(), direction.as_ref()));
    let parallel_time = start.elapsed();

    assert_eq!(serial.nrows(), problem.reduced_size());
    assert_eq!(parallel.nrows(), problem.reduced_size());
    for row in 0..serial.nrows() {
        assert!((serial[(row, 0)] - parallel[(row, 0)]).abs() < 1e-12);
    }
    println!(
        "large 2D matrix-free Jacobian: {serial_threads} thread(s) = {:?}, {parallel_threads} thread(s) = {:?}",
        serial_time, parallel_time
    );
}
