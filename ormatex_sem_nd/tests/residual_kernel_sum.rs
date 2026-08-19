use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{shapes::unit_square, SingleElementMesh};
use ormatex_sem_nd::{
    DofReduction2D, FieldRegistry, KernelAdvDiff2D, KernelAdvection2D, KernelDiffusion2D,
    KernelLinearReaction, ResidualKernelSum, SEM2DProblem,
};
use std::time::Instant;

type QuadMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

fn problem() -> SEM2DProblem<QuadMesh> {
    SEM2DProblem::new(
        unit_square(2, 1, ReferenceCellType::Quadrilateral, 1),
        2,
        FieldRegistry::new(["u"]),
        DofReduction2D::None,
    )
}

fn assert_matrix_close(a: &Mat<f64>, b: &Mat<f64>) {
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.ncols(), b.ncols());
    for row in 0..a.nrows() {
        for col in 0..a.ncols() {
            assert!(
                (a[(row, col)] - b[(row, col)]).abs() < 1e-12,
                "matrix mismatch at ({row}, {col}): {} != {}",
                a[(row, col)],
                b[(row, col)]
            );
        }
    }
}

fn assert_vector_close(a: &Mat<f64>, b: &Mat<f64>) {
    assert_matrix_close(a, b);
}

fn benchmark_duration<T>(mut operation: impl FnMut() -> T) -> std::time::Duration {
    let start = Instant::now();
    for _ in 0..3 {
        std::hint::black_box(operation());
    }
    start.elapsed()
}

#[test]
fn advection_diffusion_sum_matches_fused_and_separate_assembly() {
    let problem = problem();
    let n = problem.reduced_size();
    let state = Mat::from_fn(n, 1, |row, _| 0.2 + 0.07 * row as f64);
    let direction = Mat::from_fn(n, 1, |row, _| (0.31 * row as f64).sin());

    let fused = KernelAdvDiff2D::new(0.13, [0.4, -0.2]);
    let composed = ResidualKernelSum::from_kernel(KernelAdvection2D::new([0.4, -0.2]))
        .with(KernelDiffusion2D::new(0.13));
    let advection = KernelAdvection2D::new([0.4, -0.2]);
    let diffusion = KernelDiffusion2D::new(0.13);

    let fused_residual_values = problem.assemble_residual(&fused, state.as_ref());
    let composed_residual_values = problem.assemble_residual(&composed, state.as_ref());
    let fused_residual = Mat::from_fn(n, 1, |row, _| fused_residual_values[row]);
    let composed_residual = Mat::from_fn(n, 1, |row, _| composed_residual_values[row]);
    let advection_residual = problem.assemble_residual(&advection, state.as_ref());
    let diffusion_residual = problem.assemble_residual(&diffusion, state.as_ref());
    let separate_residual = Mat::from_fn(n, 1, |row, _| {
        advection_residual[row] + diffusion_residual[row]
    });
    assert_vector_close(&fused_residual, &composed_residual);
    assert_vector_close(&fused_residual, &separate_residual);

    let fused_jacobian = problem
        .assemble_residual_jacobian(&fused, state.as_ref())
        .to_dense();
    let composed_jacobian = problem
        .assemble_residual_jacobian(&composed, state.as_ref())
        .to_dense();
    let advection_jacobian = problem
        .assemble_residual_jacobian(&advection, state.as_ref())
        .to_dense();
    let diffusion_jacobian = problem
        .assemble_residual_jacobian(&diffusion, state.as_ref())
        .to_dense();
    let separate_jacobian = Mat::from_fn(n, n, |row, col| {
        advection_jacobian[(row, col)] + diffusion_jacobian[(row, col)]
    });
    assert_matrix_close(&fused_jacobian, &composed_jacobian);
    assert_matrix_close(&fused_jacobian, &separate_jacobian);

    let fused_action = problem.apply_jacobian_matfree(&fused, state.as_ref(), direction.as_ref());
    let composed_action =
        problem.apply_jacobian_matfree(&composed, state.as_ref(), direction.as_ref());
    let advection_action =
        problem.apply_jacobian_matfree(&advection, state.as_ref(), direction.as_ref());
    let diffusion_action =
        problem.apply_jacobian_matfree(&diffusion, state.as_ref(), direction.as_ref());
    let separate_action = Mat::from_fn(n, 1, |row, _| {
        advection_action[(row, 0)] + diffusion_action[(row, 0)]
    });
    assert_vector_close(&fused_action, &composed_action);
    assert_vector_close(&fused_action, &separate_action);
}

#[test]
#[should_panic(expected = "at least one kernel")]
fn residual_kernel_sum_rejects_empty_input() {
    ResidualKernelSum::new(Vec::new());
}

#[test]
#[should_panic(expected = "field count mismatch")]
fn residual_kernel_sum_rejects_mismatched_field_counts() {
    ResidualKernelSum::new(vec![
        Box::new(KernelDiffusion2D::new(0.13)),
        Box::new(ormatex_sem_nd::KernelEdacNavierStokes2D::new(
            1.0, 0.01, 4.0, 0.1,
        )),
    ]);
}

#[test]
#[should_panic(expected = "field names/order mismatch")]
fn residual_kernel_sum_rejects_mismatched_field_names() {
    let rate = |name| {
        KernelLinearReaction::with_field_names(
            SparseColMat::try_new_from_triplets(1, 1, &[Triplet::new(0, 0, 1.0)]).unwrap(),
            [name],
        )
    };
    ResidualKernelSum::new(vec![Box::new(rate("u")), Box::new(rate("v"))]);
}

#[test]
#[ignore = "release performance benchmark; run with --release -- --ignored --nocapture"]
fn residual_kernel_sum_benchmark() {
    let problem = SEM2DProblem::new(
        unit_square(16, 16, ReferenceCellType::Quadrilateral, 2),
        2,
        FieldRegistry::new(["u"]),
        DofReduction2D::None,
    );
    let n = problem.reduced_size();
    let state = Mat::from_fn(n, 1, |row, _| 0.2 + 0.01 * row as f64);
    let direction = Mat::from_fn(n, 1, |row, _| (0.17 * row as f64).sin());
    let fused = KernelAdvDiff2D::new(0.13, [0.4, -0.2]);
    let composed = ResidualKernelSum::from_kernel(KernelAdvection2D::new([0.4, -0.2]))
        .with(KernelDiffusion2D::new(0.13));
    let advection = KernelAdvection2D::new([0.4, -0.2]);
    let diffusion = KernelDiffusion2D::new(0.13);

    let fused_residual = benchmark_duration(|| problem.assemble_residual(&fused, state.as_ref()));
    let composed_residual =
        benchmark_duration(|| problem.assemble_residual(&composed, state.as_ref()));
    let separate_residual = benchmark_duration(|| {
        let _ = problem.assemble_residual(&advection, state.as_ref());
        let _ = problem.assemble_residual(&diffusion, state.as_ref());
    });
    let fused_jacobian = benchmark_duration(|| {
        std::hint::black_box(
            problem
                .assemble_residual_jacobian(&fused, state.as_ref())
                .to_dense(),
        )
    });
    let composed_jacobian = benchmark_duration(|| {
        std::hint::black_box(
            problem
                .assemble_residual_jacobian(&composed, state.as_ref())
                .to_dense(),
        )
    });
    let separate_jacobian = benchmark_duration(|| {
        let _ = problem
            .assemble_residual_jacobian(&advection, state.as_ref())
            .to_dense();
        let _ = problem
            .assemble_residual_jacobian(&diffusion, state.as_ref())
            .to_dense();
    });
    let fused_action = benchmark_duration(|| {
        problem.apply_jacobian_matfree(&fused, state.as_ref(), direction.as_ref())
    });
    let composed_action = benchmark_duration(|| {
        problem.apply_jacobian_matfree(&composed, state.as_ref(), direction.as_ref())
    });
    let separate_action = benchmark_duration(|| {
        let _ = problem.apply_jacobian_matfree(&advection, state.as_ref(), direction.as_ref());
        let _ = problem.apply_jacobian_matfree(&diffusion, state.as_ref(), direction.as_ref());
    });
    println!("residual: fused={fused_residual:?}, composed={composed_residual:?}, separate={separate_residual:?}");
    println!("jacobian: fused={fused_jacobian:?}, composed={composed_jacobian:?}, separate={separate_jacobian:?}");
    println!(
        "jv: fused={fused_action:?}, composed={composed_action:?}, separate={separate_action:?}"
    );
}
