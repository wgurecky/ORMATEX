//! 1D two-phase tensor pass: weak references and bit-identity.
//!
//! The 1D tensor residual and matrix-free Jacobian action run over the flat
//! natural-order [`TensorBatchPlan`](ormatex_sem_nd::common) through a
//! row-sorted E-vector. These tests assert:
//! (a) the residual matches the independent weak-form assembly and the action
//!     matches the assembled tensor Jacobian times direction (serial per-cell
//!     references that never touch the E-vector), with multi-column
//!     directions and a partial trailing batch;
//! (b) residual and multi-column actions are bit-identical across Rayon
//!     thread counts (1 vs 4), including with the linearization cache;
//! (c) the prepared (cached) path is bit-identical to the unprepared path,
//!     including the fused `-M^{-1}` epilogue.
use faer::prelude::*;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap};
use ndmesh::{shapes::unit_interval, SingleElementMesh};
use ormatex_sem_nd::{
    BilinearOps, CellState, DofReduction1D, FieldRegistry, KernelAdvDiff, LaneState, Lanes,
    LocalCtx, ResidualKernel, RowEpilogue, SEM1DProblem, TensorCtx, TensorKernelAdvDiff,
    TensorResidualKernel, WeakResidualOps, LANES,
};

type IntervalMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

/// Weak-form wrapper around the 1D advection-diffusion kernel for reference.
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

/// Lane-packed 1D diffusion kernel (`f = [0, nu*du/dx, 0]` per lane).
///
/// Writes one lane-packed triple per `(equation, q)` directly from the
/// lane-packed state and direction.
struct LaneDiffusion1D {
    /// Diffusion coefficient.
    nu: f64,
}

impl TensorResidualKernel<1> for LaneDiffusion1D {
    fn tensor_residual(
        &self,
        _ctxs: &[TensorCtx<'_>],
        state: &LaneState<'_>,
        _equation: usize,
        q: usize,
        f0: &mut Lanes,
        f1x: &mut Lanes,
        f1y: &mut Lanes,
    ) {
        let g = state.grad(0, q, 0);
        for lane in 0..LANES {
            f0[lane] = 0.0;
            f1x[lane] = self.nu * g[lane];
            f1y[lane] = 0.0;
        }
    }
    fn tensor_jacobian_action(
        &self,
        _ctxs: &[TensorCtx<'_>],
        _state: &LaneState<'_>,
        direction: &LaneState<'_>,
        _equation: usize,
        q: usize,
        f0: &mut Lanes,
        f1x: &mut Lanes,
        f1y: &mut Lanes,
    ) {
        let g = direction.grad(0, q, 0);
        for lane in 0..LANES {
            f0[lane] = 0.0;
            f1x[lane] = self.nu * g[lane];
            f1y[lane] = 0.0;
        }
    }
}

/// Run `f` on a Rayon pool with `threads` threads.
///
/// # Arguments
/// * `threads` - pool thread count.
/// * `f` - closure to run inside the pool.
///
/// # Returns
/// The closure's return value.
fn run_in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(f)
}

/// Assert two float slices are bit-identical.
///
/// # Arguments
/// * `a` - first slice.
/// * `b` - second slice.
/// * `what` - label for failure messages.
fn assert_bits_eq_slice(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{what}: length {} != {}",
        a.len(),
        b.len()
    );
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}[{i}]: {x:e} != {y:e} (bits {:x} != {:x})",
            x.to_bits(),
            y.to_bits()
        );
    }
}

/// Assert two matrices are bit-identical.
///
/// # Arguments
/// * `a` - first matrix.
/// * `b` - second matrix.
/// * `what` - label for failure messages.
fn assert_bits_eq_mat(a: MatRef<'_, f64>, b: MatRef<'_, f64>, what: &str) {
    assert_eq!(a.nrows(), b.nrows(), "{what}: row count mismatch");
    assert_eq!(a.ncols(), b.ncols(), "{what}: column count mismatch");
    for c in 0..a.ncols() {
        for r in 0..a.nrows() {
            assert_eq!(
                a[(r, c)].to_bits(),
                b[(r, c)].to_bits(),
                "{what}[({r},{c})]",
            );
        }
    }
}

/// 10-cell Dirichlet problem (partial trailing batch: 10 = 8 + 2).
///
/// # Returns
/// Single-field 1D problem with a prescribed left endpoint value.
fn dirichlet_problem(ncells: usize, degree: usize) -> SEM1DProblem<IntervalMesh> {
    SEM1DProblem::new(
        unit_interval(ncells, 1),
        degree,
        FieldRegistry::new(["c"]),
        DofReduction1D::Dirichlet {
            facets: vec![(0, 1.0)],
        },
    )
}

#[test]
fn tensor_1d_residual_and_action_match_weak_reference() {
    // 10 cells exercise a partial trailing batch (W = 8); the prescribed
    // left endpoint exercises eliminated-DOF handling in the packed gather.
    let problem = dirichlet_problem(10, 3);
    let n = problem.system_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.01 * i as f64);
    let direction = Mat::from_fn(n, 2, |i, c| {
        (0.17 * (i + 1) as f64 * (c as f64 + 1.0)).sin()
    });
    let tensor = TensorKernelAdvDiff::new(0.13, 0.4);
    let generic = GenericAdvDiff1D(KernelAdvDiff::new(0.13, 0.4));
    let operator = problem.tensor_residual_operator(&tensor);

    // Residual vs the independent weak-form path.
    let actual = run_in_pool(4, || operator.residual(state.as_ref()));
    let expected = problem.assemble_residual(0.0, &generic, state.as_ref());
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (a - b).abs() < 1e-10,
            "residual mismatch at {i}: {a} != {b}"
        );
    }

    // Multi-column Jacobian action vs the weak-form action.
    let action = run_in_pool(4, || {
        operator.apply_jacobian(state.as_ref(), direction.as_ref())
    });
    let expected_action = problem.apply_jacobian(0.0, &generic, state.as_ref(), direction.as_ref());
    for r in 0..n {
        for c in 0..2 {
            assert!(
                (action[(r, c)] - expected_action[(r, c)]).abs() < 1e-10,
                "Jv mismatch at ({r},{c})"
            );
        }
    }

    // Multi-column action vs the assembled tensor Jacobian (a serial per-cell
    // reference that never touches the E-vector).
    let assembled = operator.assemble_jacobian(state.as_ref()).to_dense();
    let reference = assembled.as_ref() * direction.as_ref();
    for r in 0..n {
        for c in 0..2 {
            assert!(
                (action[(r, c)] - reference[(r, c)]).abs() < 1e-9,
                "assembled-Jv mismatch at ({r},{c})"
            );
        }
    }
}

#[test]
fn tensor_1d_bit_identical_across_thread_counts() {
    let problem = dirichlet_problem(20, 2);
    let n = problem.system_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.03 * i as f64);
    // Three direction columns exercise the multi-column E-vector path.
    let direction = Mat::from_fn(n, 3, |i, c| {
        (0.17 * (i + 1) as f64 * (c as f64 + 1.0)).sin()
    });
    let kernel = TensorKernelAdvDiff::new(0.13, 0.4);
    let operator = problem.tensor_residual_operator(&kernel);

    let res_1 = run_in_pool(1, || operator.residual(state.as_ref()));
    let res_4 = run_in_pool(4, || operator.residual(state.as_ref()));
    assert_bits_eq_slice(&res_1, &res_4, "residual threads");

    let act_1 = run_in_pool(1, || {
        operator.apply_jacobian(state.as_ref(), direction.as_ref())
    });
    let act_4 = run_in_pool(4, || {
        operator.apply_jacobian(state.as_ref(), direction.as_ref())
    });
    assert_bits_eq_mat(act_1.as_ref(), act_4.as_ref(), "jacobian threads");

    // The linearization cache must agree bitwise at any thread count.
    let mut cached = problem.tensor_residual_operator(&kernel);
    run_in_pool(1, || cached.prepare_linearization(state.as_ref()));
    let cached_4 = run_in_pool(4, || {
        cached.apply_jacobian(state.as_ref(), direction.as_ref())
    });
    assert_bits_eq_mat(act_1.as_ref(), cached_4.as_ref(), "cached jacobian");
}

#[test]
fn tensor_1d_prepared_matches_unprepared_bitwise() {
    let problem = dirichlet_problem(20, 2);
    let n = problem.system_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.03 * i as f64);
    let direction = Mat::from_fn(n, 3, |i, c| {
        (0.17 * (i + 1) as f64 * (c as f64 + 1.0)).sin()
    });
    let mass = problem.assemble_lumped_mass();
    let m_inv: Vec<f64> = (0..n).map(|i| 1.0 / mass[(i, i)]).collect();
    let kernel = TensorKernelAdvDiff::new(0.13, 0.4);

    let plain = problem.tensor_residual_operator(&kernel);
    let mut prepared = problem.tensor_residual_operator(&kernel);
    prepared.prepare_linearization(state.as_ref());

    // Cached `apply_jacobian` (state match) vs the fresh operator.
    let expected = plain.apply_jacobian(state.as_ref(), direction.as_ref());
    let actual = prepared.apply_jacobian(state.as_ref(), direction.as_ref());
    assert_bits_eq_mat(expected.as_ref(), actual.as_ref(), "prepared apply");

    // Prepared `into` with no epilogue vs the unprepared `into`.
    let mut expected_into = Mat::zeros(n, direction.ncols());
    plain.apply_jacobian_into(state.as_ref(), direction.as_ref(), expected_into.as_mut());
    let mut actual_into = Mat::zeros(n, direction.ncols());
    assert!(prepared.apply_prepared_jacobian_into(
        direction.as_ref(),
        actual_into.as_mut(),
        RowEpilogue::None,
    ));
    assert_bits_eq_mat(
        expected_into.as_ref(),
        actual_into.as_ref(),
        "prepared into",
    );

    // Fused `-M^{-1}` epilogue vs the manual scaling of the unprepared action.
    let mut fused = Mat::zeros(n, direction.ncols());
    assert!(prepared.apply_prepared_jacobian_into(
        direction.as_ref(),
        fused.as_mut(),
        RowEpilogue::NegScale(&m_inv),
    ));
    let mut manual = expected_into.clone();
    for c in 0..direction.ncols() {
        for r in 0..n {
            manual[(r, c)] = -(manual[(r, c)] * m_inv[r]);
        }
    }
    assert_bits_eq_mat(manual.as_ref(), fused.as_ref(), "prepared epilogue");
}

#[test]
fn tensor_1d_custom_kernel_matches_weak_reference() {
    // A hand-written lane kernel still matches the weak reference (sanity,
    // not bitwise): the lane path is the only path.
    let problem = dirichlet_problem(10, 3);
    let n = problem.system_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.01 * i as f64);
    let lanes = LaneDiffusion1D { nu: 0.13 };

    let lane_op = problem.tensor_residual_operator(&lanes);
    let lane_res = run_in_pool(4, || lane_op.residual(state.as_ref()));

    let generic = GenericAdvDiff1D(KernelAdvDiff::new(0.13, 0.0));
    let expected = problem.assemble_residual(0.0, &generic, state.as_ref());
    for (i, (&a, &b)) in lane_res.iter().zip(&expected).enumerate() {
        assert!(
            (a - b).abs() < 1e-10,
            "lane residual vs weak mismatch at {i}: {a} != {b}"
        );
    }
}
