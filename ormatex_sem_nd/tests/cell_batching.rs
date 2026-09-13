//! SIMD-over-element batching: tail widths, Rayon tile boundaries, reductions.
//!
//! Covers `W-1/W/W+1` around `SIMD_CELL_WIDTH=8`, Rayon tile edges
//! (127/128/129 cells), multi-RHS directions, and periodic/Dirichlet maps.

use faer::prelude::*;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{
    shapes::{unit_interval, unit_square},
    SingleElementMesh,
};
use ormatex_sem_nd::{
    CellState, DofReduction1D, DofReduction2D, FieldRegistry, KernelAdvDiff, KernelAdvDiff2D,
    KernelAdvDiffSUPG2D, KernelEnergyAdvectionDiffusion2D, LocalCtx, ResidualKernel, SEM1DProblem,
    SEM2DProblem, TensorKernelAdvDiff, TensorKernelAdvDiff2D, TensorKernelAdvDiffSUPG2D,
    TensorKernelEnergyAdvectionDiffusion2D,
};

type IntervalMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;
type QuadMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;
use ormatex_sem_nd::{WeakResidualOps};

struct GenericAdvDiff1D(KernelAdvDiff);

impl ResidualKernel for GenericAdvDiff1D {
    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        self.0.residual_integrand(ctx, state, equation, q, test_i)
    }

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
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

struct GenericAdvDiff2D(KernelAdvDiff2D);

impl ResidualKernel for GenericAdvDiff2D {
    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        self.0.residual_integrand(ctx, state, equation, q, test_i)
    }

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
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

fn check_1d(nx: usize, p: usize) {
    let problem: SEM1DProblem<IntervalMesh> = SEM1DProblem::new(
        unit_interval(nx, 1),
        p,
        FieldRegistry::new(["temperature"]),
        DofReduction1D::None,
    );
    let n = problem.system_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.03 * i as f64);
    let direction =
        Mat::from_fn(n, 2, |i, column| (0.17 * (i + 1) as f64 * (column as f64 + 1.0)).sin());
    let tensor = TensorKernelAdvDiff(KernelAdvDiff::new(0.13, 0.4));
    let generic = GenericAdvDiff1D(KernelAdvDiff::new(0.13, 0.4));
    let operator = problem.tensor_residual_operator(&tensor);

    for (a, b) in operator
        .residual(state.as_ref())
        .iter()
        .zip(problem.assemble_residual(0.0, &generic, state.as_ref()))
    {
        assert!((a - b).abs() < 1e-10, "1D residual mismatch nx={nx} p={p}");
    }
    let action = operator.apply_jacobian(state.as_ref(), direction.as_ref());
    let expected = problem.apply_jacobian(0.0, &generic, state.as_ref(), direction.as_ref());
    for row in 0..n {
        for column in 0..2 {
            assert!(
                (action[(row, column)] - expected[(row, column)]).abs() < 1e-10,
                "1D Jv mismatch nx={nx} p={p} ({row},{column})"
            );
        }
    }
    // matrix-free action matches assembled Jacobian
    let assembled = operator.assemble_jacobian(state.as_ref()).to_dense();
    let reference = assembled.as_ref() * direction.as_ref();
    for row in 0..n {
        for column in 0..2 {
            assert!(
                (action[(row, column)] - reference[(row, column)]).abs() < 1e-9,
                "1D assembled-Jv mismatch nx={nx} p={p}"
            );
        }
    }
}

#[test]
fn tensor_batching_matches_scalar_across_1d_tail_widths() {
    for nx in [1, 3, 7, 8, 9, 15, 16, 17, 65, 127, 128, 129] {
        check_1d(nx, 2);
    }
    check_1d(9, 8);
}

#[test]
fn tensor_batching_preserves_1d_periodic_and_dirichlet_maps() {
    for reduction in [
        DofReduction1D::Periodic { facets: [0, 9] },
        DofReduction1D::Dirichlet {
            facets: vec![(0, 1.0)],
        },
    ] {
        let problem: SEM1DProblem<IntervalMesh> = SEM1DProblem::new(
            unit_interval(9, 1),
            2,
            FieldRegistry::new(["temperature"]),
            reduction,
        );
        let n = problem.system_size();
        let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.03 * i as f64);
        let direction = Mat::from_fn(n, 1, |i, _| (0.11 * i as f64).sin());
        let tensor = TensorKernelAdvDiff(KernelAdvDiff::new(0.13, 0.4));
        let generic = GenericAdvDiff1D(KernelAdvDiff::new(0.13, 0.4));
        let operator = problem.tensor_residual_operator(&tensor);
        for (a, b) in operator
            .residual(state.as_ref())
            .iter()
            .zip(problem.assemble_residual(0.0, &generic, state.as_ref()))
        {
            assert!((a - b).abs() < 1e-10, "1D BC residual mismatch");
        }
        let action = operator.apply_jacobian(state.as_ref(), direction.as_ref());
        let expected = problem.apply_jacobian(0.0, &generic, state.as_ref(), direction.as_ref());
        for row in 0..n {
            assert!((action[(row, 0)] - expected[(row, 0)]).abs() < 1e-10);
        }
    }
}

fn check_2d(nx: usize, ny: usize, p: usize) {
    let problem: SEM2DProblem<QuadMesh> = SEM2DProblem::new(
        unit_square(nx, ny, ReferenceCellType::Quadrilateral, 1),
        p,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::None,
    );
    let n = problem.reduced_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.01 * i as f64);
    let direction = Mat::from_fn(n, 2, |i, column| {
        (0.17 * (i + 1) as f64 * (column as f64 + 1.0)).sin()
    });
    let tensor = TensorKernelAdvDiff2D(KernelAdvDiff2D::new(0.13, [0.4, -0.2]));
    let generic = GenericAdvDiff2D(KernelAdvDiff2D::new(0.13, [0.4, -0.2]));
    let operator = problem.tensor_residual_operator(&tensor);

    for (a, b) in operator
        .residual(state.as_ref())
        .iter()
        .zip(problem.assemble_residual(0.0, &generic, state.as_ref()))
    {
        assert!((a - b).abs() < 1e-10, "2D residual mismatch {nx}x{ny} p={p}");
    }
    let action = operator.apply_jacobian(state.as_ref(), direction.as_ref());
    let expected = problem.apply_jacobian(0.0, &generic, state.as_ref(), direction.as_ref());
    for row in 0..n {
        for column in 0..2 {
            assert!(
                (action[(row, column)] - expected[(row, column)]).abs() < 1e-10,
                "2D Jv mismatch {nx}x{ny} p={p}"
            );
        }
    }
}

#[test]
fn tensor_batching_matches_scalar_across_2d_tail_widths() {
    for (nx, ny) in [(1, 1), (3, 2), (4, 2), (3, 3), (4, 4), (9, 2), (16, 8)] {
        check_2d(nx, ny, 2);
    }
    check_2d(4, 4, 4);
}

#[test]
fn lane_overrides_match_scalar_for_supg_and_energy() {
    // 3x3 = 9 cells exercises one full lane group plus a scalar tail.
    let problem: SEM2DProblem<QuadMesh> = SEM2DProblem::new(
        unit_square(3, 3, ReferenceCellType::Quadrilateral, 1),
        2,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::None,
    );
    let n = problem.reduced_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.01 * i as f64);
    let direction = Mat::from_fn(n, 1, |i, _| (0.17 * i as f64).sin());
    let tensor = TensorKernelAdvDiffSUPG2D::new(0.1, [0.4, -0.2], 0.05);
    let generic = KernelAdvDiffSUPG2D::new(0.1, [0.4, -0.2], 0.05);
    let operator = problem.tensor_residual_operator(&tensor);
    for (a, b) in operator
        .residual(state.as_ref())
        .iter()
        .zip(problem.assemble_residual(0.0, &generic, state.as_ref()))
    {
        assert!((a - b).abs() < 1e-10, "SUPG residual mismatch");
    }
    let action = operator.apply_jacobian(state.as_ref(), direction.as_ref());
    let expected = problem.apply_jacobian(0.0, &generic, state.as_ref(), direction.as_ref());
    for row in 0..n {
        assert!(
            (action[(row, 0)] - expected[(row, 0)]).abs() < 1e-10,
            "SUPG Jv mismatch at row {row}"
        );
    }

    let coupled: SEM2DProblem<QuadMesh> = SEM2DProblem::new(
        unit_square(3, 3, ReferenceCellType::Quadrilateral, 1),
        2,
        FieldRegistry::new(["u", "v", "T"]),
        DofReduction2D::None,
    );
    let m = coupled.system_size();
    let coupled_state = Mat::from_fn(m, 1, |i, _| 0.2 + 0.01 * i as f64);
    let coupled_direction = Mat::from_fn(m, 1, |i, _| (0.11 * i as f64).cos());
    let tensor = TensorKernelEnergyAdvectionDiffusion2D::new(0.05);
    let generic = KernelEnergyAdvectionDiffusion2D::new(0.05);
    let operator = coupled.tensor_residual_operator(&tensor);
    for (a, b) in operator
        .residual(coupled_state.as_ref())
        .iter()
        .zip(coupled.assemble_residual(0.0, &generic, coupled_state.as_ref()))
    {
        assert!((a - b).abs() < 1e-10, "energy residual mismatch");
    }
    let action = operator.apply_jacobian(coupled_state.as_ref(), coupled_direction.as_ref());
    let expected =
        coupled.apply_jacobian(0.0, &generic, coupled_state.as_ref(), coupled_direction.as_ref());
    for row in 0..m {
        assert!(
            (action[(row, 0)] - expected[(row, 0)]).abs() < 1e-10,
            "energy Jv mismatch at row {row}"
        );
    }
}

#[test]
#[ignore = "release scaling benchmark; run with --release -- --ignored --nocapture"]
fn batched_tensor_jv_thread_scaling() {
    use rayon::ThreadPoolBuilder;
    use std::time::Instant;

    let problem: SEM2DProblem<QuadMesh> = SEM2DProblem::new(
        unit_square(64, 64, ReferenceCellType::Quadrilateral, 1),
        2,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::None,
    );
    let n = problem.reduced_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.25 + i as f64 / n as f64);
    let direction = Mat::from_fn(n, 1, |i, _| (i as f64 * 0.17).sin());
    let kernel = TensorKernelAdvDiff2D(KernelAdvDiff2D::new(0.1, [0.0, 0.0]));
    let operator = problem.tensor_residual_operator(&kernel);
    let available = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(1)
        .min(8);
    const REPEATS: usize = 8;
    let mut baseline = None;
    for threads in [1, available] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            for _ in 0..2 {
                std::hint::black_box(operator.apply_jacobian(state.as_ref(), direction.as_ref()));
            }
            let start = Instant::now();
            for _ in 0..REPEATS {
                std::hint::black_box(operator.apply_jacobian(state.as_ref(), direction.as_ref()));
            }
            let per_call = start.elapsed() / REPEATS as u32;
            if baseline.is_none() {
                baseline = Some(per_call.as_secs_f64());
            }
            let speedup = baseline.unwrap() / per_call.as_secs_f64();
            println!(
                "batched 2D tensor Jv: threads={threads}, per_call={per_call:?}, speedup={speedup:.2}x, efficiency={:.2}",
                speedup / threads as f64,
            );
        });
    }
}
