use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::matrix_free::LinOp;
use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{shapes::unit_square, SingleElementMesh};
use ormatex::ode_sys::OdeSys;
use ormatex_sem_nd::{
    BilinearForm, BoundaryIntegrator, DofReduction2D, FieldRegistry, KernelAdvDiff2D, LocalCtx,
    NeumannFlux, ParCsrJacobian, ResidualKernel, RobinConvection, SEM2DProblem,
    TensorKernelAdvDiff2D,
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

struct GenericAdvDiff(KernelAdvDiff2D);

impl ResidualKernel for GenericAdvDiff {
    fn residual_integrand(
        &self,
        ctx: &ormatex_sem_nd::LocalCtx<'_>,
        state: &ormatex_sem_nd::CellState<'_>,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        self.0.residual_integrand(ctx, state, equation, q, test_i)
    }

    fn jacobian_integrand(
        &self,
        ctx: &ormatex_sem_nd::LocalCtx<'_>,
        state: &ormatex_sem_nd::CellState<'_>,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        self.0
            .jacobian_integrand(ctx, state, equation, unknown, q, test_i, trial_i)
    }
}

impl BilinearForm for GenericAdvDiff {
    fn integrand(
        &self,
        ctx: &LocalCtx<'_>,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        self.0.integrand(ctx, equation, unknown, q, test_i, trial_i)
    }
}

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

#[test]
fn tensor_path_matches_generic_residual_jacobian_and_action() {
    let problem = SEM2DProblem::new(
        unit_square(1, 1, ReferenceCellType::Quadrilateral, 1),
        3,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::None,
    );
    let n = problem.reduced_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.03 * i as f64);
    let direction = Mat::from_fn(n, 2, |i, column| {
        (0.17 * (i + 1) as f64 * (column as f64 + 1.0)).sin()
    });
    let tensor_kernel = TensorKernelAdvDiff2D(KernelAdvDiff2D::new(0.13, [0.4, -0.2]));
    let generic_kernel = GenericAdvDiff(KernelAdvDiff2D::new(0.13, [0.4, -0.2]));

    let tensor_operator = problem.tensor_residual_operator(&tensor_kernel);
    let tensor_residual = tensor_operator.residual(state.as_ref());
    let generic_residual = problem.assemble_residual(0.0, &generic_kernel, state.as_ref());
    for (tensor, generic) in tensor_residual.iter().zip(generic_residual) {
        assert!((tensor - generic).abs() < 1e-10);
    }

    let tensor_jacobian = tensor_operator.assemble_jacobian(state.as_ref()).to_dense();
    let generic_jacobian = problem
        .assemble_residual_jacobian(0.0, &generic_kernel, state.as_ref())
        .to_dense();
    for row in 0..n {
        for col in 0..n {
            assert!((tensor_jacobian[(row, col)] - generic_jacobian[(row, col)]).abs() < 1e-10);
        }
    }

    let tensor_action = tensor_operator.apply_jacobian(state.as_ref(), direction.as_ref());
    let generic_action =
        problem.apply_jacobian(0.0, &generic_kernel, state.as_ref(), direction.as_ref());
    for row in 0..n {
        for column in 0..direction.ncols() {
            assert!((tensor_action[(row, column)] - generic_action[(row, column)]).abs() < 1e-10);
        }
    }
}

#[test]
fn in_place_jacobian_action_matches_owned_result() {
    let problem = SEM2DProblem::new(
        unit_square(2, 1, ReferenceCellType::Quadrilateral, 1),
        3,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::None,
    );
    let n = problem.reduced_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.01 * i as f64);
    let direction = Mat::from_fn(n, 2, |i, column| {
        (0.11 * (i + 1) as f64 * (column as f64 + 1.0)).cos()
    });
    let kernel = TensorKernelAdvDiff2D(KernelAdvDiff2D::new(0.13, [0.4, -0.2]));
    let mut actual = Mat::zeros(n, direction.ncols());
    let operator = problem.tensor_residual_operator(&kernel);
    let expected = operator.apply_jacobian(state.as_ref(), direction.as_ref());
    operator.apply_jacobian_into(state.as_ref(), direction.as_ref(), actual.as_mut());
    for row in 0..n {
        for column in 0..direction.ncols() {
            assert_eq!(actual[(row, column)], expected[(row, column)]);
        }
    }
}

#[test]
fn tensor_bilinear_assembly_matches_generic_path() {
    let problem = SEM2DProblem::new(
        unit_square(1, 1, ReferenceCellType::Quadrilateral, 1),
        3,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::None,
    );
    let tensor = KernelAdvDiff2D::new(0.13, [0.4, -0.2]);
    let generic = GenericAdvDiff(KernelAdvDiff2D::new(0.13, [0.4, -0.2]));
    let tensor_matrix = problem.assemble_bilinear(0.0, &tensor).to_dense();
    let generic_matrix = problem.assemble_bilinear(0.0, &generic).to_dense();
    for row in 0..tensor_matrix.nrows() {
        for col in 0..tensor_matrix.ncols() {
            assert!((tensor_matrix[(row, col)] - generic_matrix[(row, col)]).abs() < 1e-10);
        }
    }
}

fn benchmark_p_scaling() {
    for p in [2, 4, 8, 12] {
        let problem = SEM2DProblem::new(
            unit_square(8, 8, ReferenceCellType::Quadrilateral, 1),
            p,
            FieldRegistry::new(["temperature"]),
            DofReduction2D::None,
        );
        let n = problem.reduced_size();
        let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.01 * i as f64);
        let direction = Mat::from_fn(n, 1, |i, _| (0.17 * i as f64).sin());
        let kernel = TensorKernelAdvDiff2D(KernelAdvDiff2D::new(0.1, [0.4, -0.2]));
        let start = Instant::now();
        for _ in 0..10 {
            let operator = problem.tensor_residual_operator(&kernel);
            std::hint::black_box(operator.residual(state.as_ref()));
            std::hint::black_box(operator.apply_jacobian(state.as_ref(), direction.as_ref()));
        }
        println!(
            "tensor p={p}, dofs/cell={}, elapsed={:?}",
            (p + 1) * (p + 1),
            start.elapsed()
        );
    }
}

#[test]
#[ignore = "release performance benchmark; run with --release -- --ignored --nocapture"]
fn tensor_path_compares_with_generic_path() {
    let p = 8;
    let problem = SEM2DProblem::new(
        unit_square(8, 8, ReferenceCellType::Quadrilateral, 1),
        p,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::None,
    );
    let n = problem.reduced_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.01 * i as f64);
    let direction = Mat::from_fn(n, 1, |i, _| (0.17 * i as f64).sin());
    let tensor = TensorKernelAdvDiff2D(KernelAdvDiff2D::new(0.1, [0.4, -0.2]));
    let generic = GenericAdvDiff(KernelAdvDiff2D::new(0.1, [0.4, -0.2]));

    let start = Instant::now();
    for _ in 0..10 {
        let operator = problem.tensor_residual_operator(&tensor);
        std::hint::black_box(operator.residual(state.as_ref()));
        std::hint::black_box(operator.apply_jacobian(state.as_ref(), direction.as_ref()));
    }
    let tensor_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..10 {
        std::hint::black_box(problem.assemble_residual(0.0, &generic, state.as_ref()));
        std::hint::black_box(problem.apply_jacobian(
            0.0,
            &generic,
            state.as_ref(),
            direction.as_ref(),
        ));
    }
    let generic_time = start.elapsed();
    println!("p={p}, tensor={tensor_time:?}, generic={generic_time:?}");
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
    let parallel = run_case(JacobianBackend::ParallelAssembled, 2);
    for row in 0..matrix_free.nrows() {
        assert!((matrix_free[(row, 0)] - assembled[(row, 0)]).abs() < 1e-8);
        assert!((parallel[(row, 0)] - assembled[(row, 0)]).abs() < 1e-8);
    }
}

#[test]
fn parallel_sparse_linop_handles_multiple_rhs_and_replaces_output() {
    let jacobian = SparseColMat::try_new_from_triplets(
        5,
        5,
        &[
            Triplet::new(0, 0, 2.0),
            Triplet::new(3, 0, -1.0),
            Triplet::new(1, 1, 3.0),
            Triplet::new(4, 1, 0.5),
            Triplet::new(2, 2, -2.0),
            Triplet::new(0, 3, 4.0),
            Triplet::new(4, 4, 1.5),
        ],
    )
    .unwrap();
    let rhs = Mat::from_fn(5, 2, |row, column| (row + 2 * column + 1) as f64);
    let expected = jacobian.as_ref() * rhs.as_ref();
    let operator = ParCsrJacobian::new(jacobian, 2);

    for par in [faer::Par::Seq, faer::Par::rayon(2)] {
        let mut actual = Mat::from_fn(5, 2, |_, _| 99.0);
        let mut scratch = MemBuffer::new(StackReq::empty());
        operator.apply(
            actual.as_mut(),
            rhs.as_ref(),
            par,
            MemStack::new(&mut scratch),
        );
        for row in 0..5 {
            for column in 0..2 {
                assert!((actual[(row, column)] - expected[(row, column)]).abs() < 1e-12);
            }
        }
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
    let kernel = TensorKernelAdvDiff2D(kernel);
    let (serial_threads, parallel_threads) = benchmark_threads();
    let serial_pool = ThreadPoolBuilder::new()
        .num_threads(serial_threads)
        .build()
        .unwrap();
    let parallel_pool = ThreadPoolBuilder::new()
        .num_threads(parallel_threads)
        .build()
        .unwrap();
    const REPEATS: usize = 16;
    let serial_operator = problem.tensor_residual_operator(&kernel);
    let parallel_operator = problem.tensor_residual_operator(&kernel);

    let start = Instant::now();
    let serial = serial_pool.install(|| {
        let mut action = Mat::zeros(problem.reduced_size(), direction.ncols());
        for _ in 0..REPEATS {
            action = serial_operator.apply_jacobian(state.as_ref(), direction.as_ref());
            std::hint::black_box(&action);
        }
        action
    });
    let serial_time = start.elapsed() / REPEATS as u32;

    let start = Instant::now();
    let parallel = parallel_pool.install(|| {
        let mut action = Mat::zeros(problem.reduced_size(), direction.ncols());
        for _ in 0..REPEATS {
            action = parallel_operator.apply_jacobian(state.as_ref(), direction.as_ref());
            std::hint::black_box(&action);
        }
        action
    });
    let parallel_time = start.elapsed() / REPEATS as u32;

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

#[test]
#[ignore = "release performance benchmark; run with --release -- --ignored --nocapture"]
fn tensor_p_scaling_runtime() {
    benchmark_p_scaling();
}
