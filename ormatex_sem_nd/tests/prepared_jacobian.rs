//! Prepared (linearized) tensor Jacobian: bitwise equivalence.
//!
//! The prepared path applies the Jacobian at the `prepare_linearization`
//! state from the cache's owned copy with no per-apply state comparison, and
//! optionally fuses the boundary add plus `-M^{-1}` row epilogue into the
//! row-sorted phase-2 reduction. These tests assert bitwise identity:
//! (a) prepared (`None` epilogue) vs unprepared `apply_jacobian_into`,
//!     without boundary terms, multi-column;
//! (b) same with tensor state-boundary terms attached;
//! (c) prepared `NegScale(m_inv)` vs unprepared action followed by the manual
//!     SIMD-order scaling `-(v * m)` per row, with and without boundary;
//! (d) unprepared cache-miss still matches fresh (kept from the old suite).
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::matrix_free::LinOp;
use faer::prelude::*;
use ndelement::types::ReferenceCellType;
use ndmesh::shapes::unit_square;
use ormatex_sem_nd::{
    DofReduction2D, FieldRegistry, KernelEdacNavierStokes2D, KernelEdacSplitBoundaryFlux2D,
    MatrixFreeMinvJacobian, RowEpilogue, SEM2DProblem, StateBoundaryTerms,
    StateTensorBoundaryTerms, TensorKernelEdacNavierStokes2D, TensorKernelEdacSplitBoundaryFlux2D,
};

type QuadMesh = ndmesh::SingleElementMesh<
    f64,
    ndelement::ciarlet::CiarletElement<f64, ndelement::map::IdentityMap, f64>,
>;

/// Build a small 3-field EDAC tensor problem.
fn problem() -> SEM2DProblem<QuadMesh> {
    SEM2DProblem::new(
        unit_square(4, 4, ReferenceCellType::Quadrilateral, 1),
        2,
        FieldRegistry::new(["u", "v", "p"]),
        DofReduction2D::None,
    )
}

/// EDAC kernel used across the prepared tests.
fn kernel() -> TensorKernelEdacNavierStokes2D {
    TensorKernelEdacNavierStokes2D::new(1.0, 0.1, 10.0, 0.0)
}

/// Weak kernel paired with [`kernel`] in the mixed-operator tests below.
fn weak_kernel() -> KernelEdacNavierStokes2D {
    KernelEdacNavierStokes2D::new(1.0, 0.1, 10.0, 0.0)
}

/// Assert two matrices are bitwise equal.
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
                "{what}[({r},{c})]: {:e} != {:e}",
                a[(r, c)],
                b[(r, c)],
            );
        }
    }
}

#[test]
fn prepared_matches_unprepared_without_boundary_multicolumn() {
    let problem = problem();
    let k = kernel();
    let n = problem.system_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.03 * i as f64);
    // Three columns exercise the multi-column sorted E-vector path.
    let direction = Mat::from_fn(n, 3, |i, c| {
        (0.17 * (i + 1) as f64 * (c as f64 + 1.0)).sin()
    });

    // No cache prepared: must report `false` and leave `out` untouched.
    let plain = problem.tensor_residual_operator(&k);
    let mut untouched = Mat::from_fn(n, 3, |_, _| 7.0);
    let ran = plain.apply_prepared_jacobian_into(
        direction.as_ref(),
        untouched.as_mut(),
        RowEpilogue::None,
    );
    assert!(!ran, "unprepared operator must return false");
    for r in 0..n {
        for c in 0..3 {
            assert_eq!(untouched[(r, c)].to_bits(), 7.0f64.to_bits());
        }
    }

    // Prepared with `None` epilogue matches unprepared bitwise.
    let mut prepared_op = problem.tensor_residual_operator(&k);
    prepared_op.prepare_linearization(state.as_ref());
    let mut expected = Mat::zeros(n, 3);
    plain.apply_jacobian_into(state.as_ref(), direction.as_ref(), expected.as_mut());
    let mut actual = Mat::zeros(n, 3);
    let ran = prepared_op.apply_prepared_jacobian_into(
        direction.as_ref(),
        actual.as_mut(),
        RowEpilogue::None,
    );
    assert!(ran, "prepared operator must return true");
    assert_bits_eq_mat(actual.as_ref(), expected.as_ref(), "prepared no-boundary");

    // `apply_jacobian` (allocating) still matches via the state-checked path.
    let via_checked = prepared_op.apply_jacobian(state.as_ref(), direction.as_ref());
    assert_bits_eq_mat(via_checked.as_ref(), expected.as_ref(), "checked path");
}

#[test]
fn prepared_matches_unprepared_with_tensor_boundary_multicolumn() {
    let problem = problem();
    let k = kernel();
    let terms = StateTensorBoundaryTerms::new().with_default(TensorKernelEdacSplitBoundaryFlux2D);
    let n = problem.system_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.15 + 0.02 * i as f64);
    let direction = Mat::from_fn(n, 3, |i, c| {
        (0.11 * (i + 1) as f64 * (c as f64 + 1.0)).cos()
    });

    let plain = problem
        .tensor_residual_operator(&k)
        .with_state_boundary(&terms);
    let mut prepared_op = problem
        .tensor_residual_operator(&k)
        .with_state_boundary(&terms);
    prepared_op.prepare_linearization(state.as_ref());

    let mut expected = Mat::zeros(n, 3);
    plain.apply_jacobian_into(state.as_ref(), direction.as_ref(), expected.as_mut());
    let mut actual = Mat::zeros(n, 3);
    let ran = prepared_op.apply_prepared_jacobian_into(
        direction.as_ref(),
        actual.as_mut(),
        RowEpilogue::None,
    );
    assert!(ran);
    assert_bits_eq_mat(actual.as_ref(), expected.as_ref(), "prepared boundary");
}

#[test]
fn prepared_negscale_matches_manual_scaling_with_and_without_boundary() {
    let problem = problem();
    let k = kernel();
    let n = problem.system_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.01 * i as f64);
    let direction = Mat::from_fn(n, 2, |i, c| (0.23 * i as f64 + c as f64).cos());
    // Positive inverse-mass diagonal (values bracketing SIMD width 4 + tail).
    let m_inv: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * (i % 17) as f64).collect();

    for with_boundary in [false, true] {
        let terms =
            StateTensorBoundaryTerms::new().with_default(TensorKernelEdacSplitBoundaryFlux2D);
        // Build operators with/without boundary terms for this round.
        let plain = if with_boundary {
            problem
                .tensor_residual_operator(&k)
                .with_state_boundary(&terms)
                .at_time(0.0)
        } else {
            problem.tensor_residual_operator(&k).at_time(0.0)
        };
        // NOTE: `at_time(0.0)` keeps the same time; both operators share it.
        let mut prepared_op = if with_boundary {
            problem
                .tensor_residual_operator(&k)
                .with_state_boundary(&terms)
        } else {
            problem.tensor_residual_operator(&k)
        };
        let _ = &plain;
        prepared_op.prepare_linearization(state.as_ref());

        // Unprepared action, then manual SIMD-order scaling `-(v * m)`.
        let mut manual = Mat::zeros(n, 2);
        // Rebuild the matching plain operator for the manual reference.
        if with_boundary {
            let op = problem
                .tensor_residual_operator(&k)
                .with_state_boundary(&terms);
            op.apply_jacobian_into(state.as_ref(), direction.as_ref(), manual.as_mut());
        } else {
            let op = problem.tensor_residual_operator(&k);
            op.apply_jacobian_into(state.as_ref(), direction.as_ref(), manual.as_mut());
        }
        for c in 0..2 {
            for r in 0..n {
                manual[(r, c)] = -(manual[(r, c)] * m_inv[r]);
            }
        }

        let mut fused = Mat::zeros(n, 2);
        let ran = prepared_op.apply_prepared_jacobian_into(
            direction.as_ref(),
            fused.as_mut(),
            RowEpilogue::NegScale(&m_inv),
        );
        assert!(ran);
        assert_bits_eq_mat(
            fused.as_ref(),
            manual.as_ref(),
            if with_boundary {
                "negscale boundary"
            } else {
                "negscale volume"
            },
        );
    }
}

#[test]
fn minv_jacobian_uses_prepared_path_bit_identically() {
    // `MatrixFreeMinvJacobian::apply` prefers the prepared `NegScale` path and
    // must match the old `apply_jacobian_into` + serial scaling sequence.
    let problem = problem();
    let k = kernel();
    let n = problem.system_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.03 * i as f64);
    let direction = Mat::from_fn(n, 2, |i, c| {
        (0.17 * (i + 1) as f64 * (c as f64 + 1.0)).sin()
    });
    let m_inv: Vec<f64> = (0..n).map(|i| 0.7 + 0.013 * (i % 13) as f64).collect();

    let operator = problem.tensor_residual_operator(&k);
    let minv = MatrixFreeMinvJacobian::new(operator, state.clone(), &m_inv);
    let mut actual = Mat::zeros(n, 2);
    let mut scratch = MemBuffer::new(StackReq::empty());
    minv.apply(
        actual.as_mut(),
        direction.as_ref(),
        faer::Par::Seq,
        MemStack::new(&mut scratch),
    );

    let reference_op = problem.tensor_residual_operator(&k);
    let mut expected = Mat::zeros(n, 2);
    reference_op.apply_jacobian_into(state.as_ref(), direction.as_ref(), expected.as_mut());
    for c in 0..2 {
        for r in 0..n {
            expected[(r, c)] = -(expected[(r, c)] * m_inv[r]);
        }
    }
    assert_bits_eq_mat(actual.as_ref(), expected.as_ref(), "minv prepared");
}

#[test]
fn mixed_prepared_matches_unprepared_with_both_boundaries_multicolumn() {
    // Mixed operator (tensor kernel + weak kernel) with weak state-boundary
    // terms AND tensor state-boundary terms: the prepared path (cached tensor
    // volume, recomputed weak volume and both boundary actions from the owned
    // state copy, tensor boundary overlapped via `rayon::join`) must match the
    // unprepared `apply_jacobian_into` bitwise, multi-column, with
    // `RowEpilogue::None` and `RowEpilogue::NegScale` (the latter against the
    // manual SIMD-order scaling `-(v * m)` per row).
    let problem = problem();
    let tensor = kernel();
    let weak = weak_kernel();
    let weak_terms = StateBoundaryTerms::new().with_default(KernelEdacSplitBoundaryFlux2D);
    let tensor_terms =
        StateTensorBoundaryTerms::new().with_default(TensorKernelEdacSplitBoundaryFlux2D);
    let n = problem.system_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.12 + 0.023 * i as f64);
    // Three columns exercise the multi-column sorted E-vector path.
    let direction = Mat::from_fn(n, 3, |i, c| {
        (0.13 * (i + 1) as f64 * (c as f64 + 1.0)).sin()
    });
    // Positive inverse-mass diagonal (values bracketing SIMD width 4 + tail).
    let m_inv: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * (i % 17) as f64).collect();

    // No cache prepared: must report `false` and leave `out` untouched.
    let plain = || {
        problem
            .mixed_residual_operator(&tensor, &weak)
            .with_weak_state_boundary(&weak_terms)
            .with_tensor_state_boundary(&tensor_terms)
    };
    let mut untouched = Mat::from_fn(n, 3, |_, _| 7.0);
    let ran = plain().apply_prepared_jacobian_into(
        direction.as_ref(),
        untouched.as_mut(),
        RowEpilogue::None,
    );
    assert!(!ran, "unprepared mixed operator must return false");
    for r in 0..n {
        for c in 0..3 {
            assert_eq!(untouched[(r, c)].to_bits(), 7.0f64.to_bits());
        }
    }

    for scaled in [false, true] {
        let mut prepared_op = plain();
        prepared_op.prepare_linearization(state.as_ref());

        // Unprepared reference, then the manual SIMD-order scaling for NegScale.
        let mut expected = Mat::zeros(n, 3);
        plain().apply_jacobian_into(state.as_ref(), direction.as_ref(), expected.as_mut());
        if scaled {
            for c in 0..3 {
                for r in 0..n {
                    expected[(r, c)] = -(expected[(r, c)] * m_inv[r]);
                }
            }
        }

        let mut actual = Mat::zeros(n, 3);
        let epilogue = if scaled {
            RowEpilogue::NegScale(&m_inv)
        } else {
            RowEpilogue::None
        };
        let ran =
            prepared_op.apply_prepared_jacobian_into(direction.as_ref(), actual.as_mut(), epilogue);
        assert!(ran, "prepared mixed operator must return true");
        assert_bits_eq_mat(
            actual.as_ref(),
            expected.as_ref(),
            if scaled {
                "mixed prepared negscale both-boundaries"
            } else {
                "mixed prepared none both-boundaries"
            },
        );
    }
}

#[test]
fn prepared_strided_out_matches_contiguous_bitwise() {
    // Non-contiguous `out` (strided row view, column stride != view rows)
    // reduces through the pooled column-major scratch and must match the
    // contiguous in-place reduction bitwise, with tensor boundary terms and
    // the fused `-M^{-1}` epilogue active.
    let problem = problem();
    let k = kernel();
    let terms = StateTensorBoundaryTerms::new().with_default(TensorKernelEdacSplitBoundaryFlux2D);
    let n = problem.system_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.03 * i as f64);
    let direction = Mat::from_fn(n, 2, |i, c| {
        (0.17 * (i + 1) as f64 * (c as f64 + 1.0)).sin()
    });
    let m_inv: Vec<f64> = (0..n).map(|i| 0.7 + 0.013 * (i % 13) as f64).collect();

    let mut prepared_op = problem
        .tensor_residual_operator(&k)
        .with_state_boundary(&terms);
    prepared_op.prepare_linearization(state.as_ref());

    let mut expected = Mat::zeros(n, 2);
    let ran = prepared_op.apply_prepared_jacobian_into(
        direction.as_ref(),
        expected.as_mut(),
        RowEpilogue::NegScale(&m_inv),
    );
    assert!(ran);

    let mut backing = Mat::zeros(2 * n, 2);
    {
        let view = backing.as_mut().submatrix_mut(0, 0, n, 2);
        let ran = prepared_op.apply_prepared_jacobian_into(
            direction.as_ref(),
            view,
            RowEpilogue::NegScale(&m_inv),
        );
        assert!(ran);
    }
    let actual = backing.as_ref().submatrix(0, 0, n, 2);
    assert_bits_eq_mat(actual, expected.as_ref(), "strided out");
}
