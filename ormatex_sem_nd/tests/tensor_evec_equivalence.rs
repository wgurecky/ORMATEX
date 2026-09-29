//! Two-phase E-vector scheme: serial references and thread-count bit-identity.
//!
//! The 2D tensor residual and matrix-free Jacobian action reduce lane-packed
//! per-batch (E-vector) actions through a CSR table sorted by cell color.
//! These tests assert:
//! (a) the residual matches the independent weak-form assembly and the
//!     Jacobian action matches the assembled tensor Jacobian times direction
//!     (both serial per-cell paths that never touch the E-vector);
//! (b) residual and multi-column Jacobian actions are bit-identical across
//!     Rayon thread counts (1 vs 4), including with the linearization cache;
//! (c) field-specific reductions and subset (non-all-fields) kernels work and
//!     stay bit-identical across thread counts with multi-column directions.
use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::shapes::unit_square;
use ndmesh::SingleElementMesh;
use ormatex_sem_nd::{
    CellState, DofReduction2D, FieldRegistry, KernelAdvDiff2D, KernelLinearReaction, LocalCtx,
    ResidualKernel, SEM2DProblem, TensorKernelAdvDiff2D, TensorKernelEdacNavierStokes2D,
    TensorKernelLinearReaction, WeakResidualOps,
};

type QuadMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

/// Weak-form wrapper around the 2D advection-diffusion kernel for reference.
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

#[test]
fn evec_matches_weak_and_assembled_serial_reference() {
    // 3x2 = 6 cells exercises a partial trailing batch (W = 8).
    let problem: SEM2DProblem<QuadMesh> = SEM2DProblem::new(
        unit_square(3, 2, ReferenceCellType::Quadrilateral, 1),
        2,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::None,
    );
    let n = problem.system_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.01 * i as f64);
    let direction = Mat::from_fn(n, 2, |i, c| {
        (0.17 * (i + 1) as f64 * (c as f64 + 1.0)).sin()
    });
    let tensor = TensorKernelAdvDiff2D(KernelAdvDiff2D::new(0.13, [0.4, -0.2]));
    let generic = GenericAdvDiff2D(KernelAdvDiff2D::new(0.13, [0.4, -0.2]));
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
fn evec_bit_identical_across_thread_counts() {
    let problem: SEM2DProblem<QuadMesh> = SEM2DProblem::new(
        unit_square(4, 4, ReferenceCellType::Quadrilateral, 1),
        2,
        FieldRegistry::new(["u", "v", "p"]),
        DofReduction2D::None,
    );
    let kernel = TensorKernelEdacNavierStokes2D::new(1.0, 0.1, 10.0, 0.0);
    let n = problem.system_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.03 * i as f64);
    // Three direction columns exercise the multi-column E-vector path.
    let direction = Mat::from_fn(n, 3, |i, c| {
        (0.17 * (i + 1) as f64 * (c as f64 + 1.0)).sin()
    });
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
fn evec_field_specific_and_subset_bit_identical() {
    // Two fields with different Dirichlet reductions (field-specific maps).
    let problem: SEM2DProblem<QuadMesh> = SEM2DProblem::new(
        unit_square(3, 2, ReferenceCellType::Quadrilateral, 1),
        2,
        FieldRegistry::new(["a", "b"]),
        DofReduction2D::FieldSpecific {
            reductions: vec![
                DofReduction2D::Dirichlet {
                    facets: vec![(0, 1.0)],
                },
                DofReduction2D::Dirichlet {
                    facets: vec![(1, 2.0)],
                },
            ],
        },
    );
    let rates = SparseColMat::try_new_from_triplets(
        2,
        2,
        &[Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 3.0)],
    )
    .unwrap();
    let tensor = TensorKernelLinearReaction::new(rates);
    let n = problem.system_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.13 + 0.07 * i as f64);
    let direction = Mat::from_fn(n, 2, |i, c| (0.31 * i as f64 + c as f64).sin());
    let operator = problem.tensor_residual_operator(&tensor);

    let res_1 = run_in_pool(1, || operator.residual(state.as_ref()));
    let res_4 = run_in_pool(4, || operator.residual(state.as_ref()));
    assert_bits_eq_slice(&res_1, &res_4, "field-specific residual");
    assert!(res_1.iter().all(|v| v.is_finite()));

    let act_1 = run_in_pool(1, || {
        operator.apply_jacobian(state.as_ref(), direction.as_ref())
    });
    let act_4 = run_in_pool(4, || {
        operator.apply_jacobian(state.as_ref(), direction.as_ref())
    });
    assert_bits_eq_mat(act_1.as_ref(), act_4.as_ref(), "field-specific jacobian");

    // Weak-form reference for the same field-specific problem.
    let weak_rates = SparseColMat::try_new_from_triplets(
        2,
        2,
        &[Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 3.0)],
    )
    .unwrap();
    let weak = KernelLinearReaction::new(weak_rates);
    let expected = problem.assemble_residual(0.0, &weak, state.as_ref());
    assert_eq!(res_1.len(), expected.len());
    for (i, (&a, &b)) in res_1.iter().zip(&expected).enumerate() {
        assert!(
            (a - b).abs() < 1e-10,
            "field-specific weak residual mismatch at {i}: {a} != {b}"
        );
    }

    // Subset kernel: only ["T", "p"] of a four-field problem, exercising a
    // non-all-fields row-entries selection with multi-column directions.
    let big: SEM2DProblem<QuadMesh> = SEM2DProblem::new(
        unit_square(2, 2, ReferenceCellType::Quadrilateral, 1),
        2,
        FieldRegistry::new(["u", "v", "p", "T"]),
        DofReduction2D::None,
    );
    let sub_rates = SparseColMat::try_new_from_triplets(
        2,
        2,
        &[Triplet::new(0, 0, 2.0), Triplet::new(1, 0, -0.5)],
    )
    .unwrap();
    let subset = TensorKernelLinearReaction::with_field_names(sub_rates, ["T", "p"]);
    let m = big.system_size();
    let big_state = Mat::from_fn(m, 1, |i, _| 0.11 + 0.05 * i as f64);
    let big_direction = Mat::from_fn(m, 3, |i, c| (0.23 * i as f64 + c as f64).cos());
    let subset_op = big.tensor_residual_operator(&subset);
    let sub_1 = run_in_pool(1, || subset_op.residual(big_state.as_ref()));
    let sub_4 = run_in_pool(4, || subset_op.residual(big_state.as_ref()));
    assert_bits_eq_slice(&sub_1, &sub_4, "subset residual");
    let sub_act_1 = run_in_pool(1, || {
        subset_op.apply_jacobian(big_state.as_ref(), big_direction.as_ref())
    });
    let sub_act_4 = run_in_pool(4, || {
        subset_op.apply_jacobian(big_state.as_ref(), big_direction.as_ref())
    });
    assert_bits_eq_mat(sub_act_1.as_ref(), sub_act_4.as_ref(), "subset jacobian");
}
