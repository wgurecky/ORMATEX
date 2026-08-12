use faer::prelude::*;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{shapes::unit_square, SingleElementMesh};
use ormatex_sem_nd::{DofReduction2D, KernelAdvDiff2D, SEM2DProblem};
use rayon::ThreadPoolBuilder;
use std::time::Instant;

type QuadMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

fn build_large_diffusion_case() -> (SEM2DProblem<QuadMesh>, KernelAdvDiff2D, Mat<f64>) {
    let mesh = unit_square(64, 64, ReferenceCellType::Quadrilateral, 1);
    let problem = SEM2DProblem::new(mesh, 2, DofReduction2D::None);
    let n = problem.reduced_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.25 + i as f64 / n as f64);
    (problem, KernelAdvDiff2D::new(0.1, [0.0, 0.0]), state)
}

fn benchmark_threads() -> (usize, usize) {
    let available = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    (1, available.min(8))
}

#[test]
#[ignore = "large runtime benchmark; run with --release -- --ignored --nocapture"]
fn large_2d_diffusion_assembled_jacobian_runtime() {
    let (problem, kernel, state) = build_large_diffusion_case();
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
    let serial =
        serial_pool.install(|| problem.assemble_residual_jacobian(&kernel, state.as_ref()));
    let serial_time = start.elapsed();

    let start = Instant::now();
    let parallel =
        parallel_pool.install(|| problem.assemble_residual_jacobian(&kernel, state.as_ref()));
    let parallel_time = start.elapsed();

    assert_eq!(serial.nrows(), problem.reduced_size());
    assert_eq!(parallel.nrows(), problem.reduced_size());
    assert_eq!(serial.compute_nnz(), parallel.compute_nnz());
    let (serial_symbolic, serial_values) = serial.as_ref().parts();
    let (parallel_symbolic, parallel_values) = parallel.as_ref().parts();
    assert_eq!(serial_symbolic.col_ptr(), parallel_symbolic.col_ptr());
    assert_eq!(serial_symbolic.row_idx(), parallel_symbolic.row_idx());
    for (serial_value, parallel_value) in serial_values.iter().zip(parallel_values) {
        assert!((serial_value - parallel_value).abs() < 1e-12);
    }
    println!(
        "large 2D assembled Jacobian: {serial_threads} thread(s) = {:?}, {parallel_threads} thread(s) = {:?}",
        serial_time, parallel_time
    );
}
