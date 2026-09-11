use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::matrix_free::LinOp;
use faer::prelude::*;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{shapes::unit_square, SingleElementMesh};
use ormatex_sem_nd::{
    DofReduction2D, FieldRegistry, KernelAdvDiff2D, ParCsrJacobian, SEM2DProblem,
};
use rayon::ThreadPoolBuilder;
use std::time::Instant;

type QuadMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

fn build_large_diffusion_case() -> (SEM2DProblem<QuadMesh>, KernelAdvDiff2D, Mat<f64>) {
    let mesh = unit_square(64, 64, ReferenceCellType::Quadrilateral, 1);
    let problem = SEM2DProblem::new(
        mesh,
        2,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::None,
    );
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

fn measure_spmv(
    pool: &rayon::ThreadPool,
    operator: &dyn LinOp<f64>,
    par: Par,
    rhs: MatRef<'_, f64>,
    warmup: usize,
    iterations: usize,
) -> f64 {
    let mut out = Mat::zeros(operator.nrows(), rhs.ncols());
    let mut scratch = MemBuffer::new(StackReq::empty());
    pool.install(|| {
        for _ in 0..warmup {
            operator.apply(out.as_mut(), rhs, par, MemStack::new(&mut scratch));
        }
        let start = Instant::now();
        for _ in 0..iterations {
            operator.apply(out.as_mut(), rhs, par, MemStack::new(&mut scratch));
        }
        std::hint::black_box(out[(0, 0)]);
        start.elapsed().as_secs_f64() / iterations as f64
    })
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[test]
#[ignore = "release performance benchmark; run with --release -- --ignored --nocapture"]
fn large_2d_assembled_jacobian_spmv_speedup() {
    benchmark_assembled_jacobian_spmv("25k", 80, 80, 2, 5_000);
}

#[test]
#[ignore = "release performance benchmark; run with --release -- --ignored --nocapture"]
fn ten_thousand_dof_assembled_jacobian_spmv_speedup() {
    benchmark_assembled_jacobian_spmv("10k", 100, 100, 1, 10_000);
}

#[test]
#[ignore = "release performance benchmark; run with --release -- --ignored --nocapture"]
fn one_hundred_thousand_dof_assembled_jacobian_spmv_speedup() {
    benchmark_assembled_jacobian_spmv("100k", 316, 316, 1, 100_000);
}

fn benchmark_assembled_jacobian_spmv(
    label: &str,
    nx: usize,
    ny: usize,
    p: usize,
    minimum_dofs: usize,
) {
    let mesh = unit_square(nx, ny, ReferenceCellType::Quadrilateral, 1);
    let problem = SEM2DProblem::new(
        mesh,
        p,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::None,
    );
    let n = problem.reduced_size();
    assert!(
        n >= minimum_dofs,
        "{label} benchmark needs at least {minimum_dofs} DOFs, got {n}"
    );
    let state = Mat::from_fn(n, 1, |i, _| 0.25 + i as f64 / n as f64);
    let rhs = Mat::from_fn(n, 1, |i, _| (0.17 * i as f64).sin());
    let jacobian = problem.assemble_residual_jacobian(
        0.0,
        &KernelAdvDiff2D::new(0.1, [0.0, 0.0]),
        state.as_ref(),
    );
    let serial_pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let serial = jacobian.as_ref();

    let mut timings = Vec::new();
    let serial_time = median(
        (0..7)
            .map(|_| measure_spmv(&serial_pool, &serial, Par::Seq, rhs.as_ref(), 100, 1_000))
            .collect(),
    );
    println!(
        "assembled Jv benchmark: case={label}, dofs={n}, nnz={}, serial={:.3} us/product",
        jacobian.compute_nnz(),
        serial_time * 1e6,
    );

    for threads in [1, 2, 4] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let parallel = ParCsrJacobian::new(jacobian.clone(), threads);
        let parallel_time = median(
            (0..7)
                .map(|_| {
                    measure_spmv(
                        &pool,
                        &parallel,
                        Par::rayon(threads),
                        rhs.as_ref(),
                        100,
                        1_000,
                    )
                })
                .collect(),
        );
        timings.push((threads, parallel_time));
        println!(
            "assembled Jv benchmark: case={label}, threads={threads}, parallel={:.3} us/product, speedup={:.3}x",
            parallel_time * 1e6,
            serial_time / parallel_time,
        );
    }

    let mut expected = Mat::zeros(n, 1);
    let mut actual = Mat::zeros(n, 1);
    let mut scratch = MemBuffer::new(StackReq::empty());
    serial.apply(
        expected.as_mut(),
        rhs.as_ref(),
        Par::Seq,
        MemStack::new(&mut scratch),
    );
    for threads in [1, 2, 4] {
        let parallel = ParCsrJacobian::new(jacobian.clone(), threads);
        parallel.apply(
            actual.as_mut(),
            rhs.as_ref(),
            Par::rayon(threads),
            MemStack::new(&mut scratch),
        );
        assert!((expected.as_ref() - actual.as_ref()).norm_max() < 1e-12);
    }
    assert_eq!(timings.len(), 3);
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
        serial_pool.install(|| problem.assemble_residual_jacobian(0.0, &kernel, state.as_ref()));
    let serial_time = start.elapsed();

    let start = Instant::now();
    let parallel =
        parallel_pool.install(|| problem.assemble_residual_jacobian(0.0, &kernel, state.as_ref()));
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
