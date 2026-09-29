//! Tensor batch plan, lane path, and linearization cache equivalence.
//!
//! Builds a small 2D EDAC tensor problem and asserts:
//! (a) cached vs uncached Jacobian actions are bit-identical;
//! (b) the Jacobian action matches a finite-difference of the residual.
use faer::prelude::*;
use ndelement::types::ReferenceCellType;
use ndmesh::shapes::unit_square;
use ormatex_sem_nd::{DofReduction2D, FieldRegistry, SEM2DProblem};

type QuadMesh = ndmesh::SingleElementMesh<
    f64,
    ndelement::ciarlet::CiarletElement<f64, ndelement::map::IdentityMap, f64>,
>;

fn problem() -> SEM2DProblem<QuadMesh> {
    SEM2DProblem::new(
        unit_square(4, 4, ReferenceCellType::Quadrilateral, 1),
        2,
        FieldRegistry::new(["u", "v", "p"]),
        DofReduction2D::None,
    )
}

fn kernel() -> ormatex_sem_nd::TensorKernelEdacNavierStokes2D {
    ormatex_sem_nd::TensorKernelEdacNavierStokes2D::new(1.0, 0.1, 10.0, 0.0)
}

fn assert_bits_eq(a: f64, b: f64, what: &str) {
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "{what}: {a:e} != {b:e} (bits {:x} != {:x})",
        a.to_bits(),
        b.to_bits()
    );
}

#[test]
fn cached_vs_uncached_jacobian_action_bit_identical() {
    let problem = problem();
    let k = kernel();
    let n = problem.system_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.03 * i as f64);
    let direction = Mat::from_fn(n, 2, |i, c| {
        (0.17 * (i + 1) as f64 * (c as f64 + 1.0)).sin()
    });

    let op = problem.tensor_residual_operator(&k);
    let expected = op.apply_jacobian(state.as_ref(), direction.as_ref());
    let mut expected_into = Mat::zeros(n, direction.ncols());
    op.apply_jacobian_into(state.as_ref(), direction.as_ref(), expected_into.as_mut());

    let mut cached_op = problem.tensor_residual_operator(&k);
    cached_op.prepare_linearization(state.as_ref());
    let actual = cached_op.apply_jacobian(state.as_ref(), direction.as_ref());
    let mut actual_into = Mat::zeros(n, direction.ncols());
    cached_op.apply_jacobian_into(state.as_ref(), direction.as_ref(), actual_into.as_mut());

    assert_eq!(expected.nrows(), actual.nrows());
    assert_eq!(expected.ncols(), actual.ncols());
    for r in 0..n {
        for c in 0..direction.ncols() {
            assert_bits_eq(
                actual[(r, c)],
                expected[(r, c)],
                &format!("cached apply r={r} c={c}"),
            );
            assert_bits_eq(
                actual_into[(r, c)],
                expected_into[(r, c)],
                &format!("cached apply_into r={r} c={c}"),
            );
        }
    }

    // Cloned state (different allocation, same bits) must also hit the cache
    // via bitwise comparison and stay bit-identical.
    let state_clone = state.clone();
    let actual_clone = cached_op.apply_jacobian(state_clone.as_ref(), direction.as_ref());
    for r in 0..n {
        for c in 0..direction.ncols() {
            assert_bits_eq(
                actual_clone[(r, c)],
                expected[(r, c)],
                &format!("cached clone r={r} c={c}"),
            );
        }
    }

    // A different state must miss the cache and compute fresh (still correct vs FD below).
    let mut other = state.clone();
    other[(0, 0)] += 1e-3;
    let fresh = op.apply_jacobian(other.as_ref(), direction.as_ref());
    let via_cache_op = cached_op.apply_jacobian(other.as_ref(), direction.as_ref());
    for r in 0..n {
        for c in 0..direction.ncols() {
            assert_bits_eq(
                via_cache_op[(r, c)],
                fresh[(r, c)],
                &format!("cache-miss r={r} c={c}"),
            );
        }
    }
}

#[test]
fn cache_misses_after_in_place_mutation() {
    // The state buffer is mutated in place after `prepare_linearization`
    // (same allocation, different bits). The cached operator must observe a
    // miss and recompute, staying bit-identical to a fresh operator at the
    // mutated state; a pointer-equality fast path would serve stale data.
    let problem = problem();
    let k = kernel();
    let n = problem.system_size();
    let mut state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.03 * i as f64);
    let direction = Mat::from_fn(n, 1, |i, _| (0.17 * (i + 1) as f64).sin());

    let mut cached_op = problem.tensor_residual_operator(&k);
    cached_op.prepare_linearization(state.as_ref());
    state[(0, 0)] += 1e-3;
    let actual = cached_op.apply_jacobian(state.as_ref(), direction.as_ref());

    let fresh_op = problem.tensor_residual_operator(&k);
    let expected = fresh_op.apply_jacobian(state.as_ref(), direction.as_ref());
    for r in 0..n {
        assert_bits_eq(
            actual[(r, 0)],
            expected[(r, 0)],
            &format!("in-place mutation r={r}"),
        );
    }
}

#[test]
#[should_panic(expected = "state size mismatch")]
fn tensor_residual_rejects_wrong_size_state() {
    let problem = problem();
    let k = kernel();
    let n = problem.system_size();
    let bad = Mat::from_fn(n + 1, 1, |_, _| 0.0);
    let op = problem.tensor_residual_operator(&k);
    let _ = op.residual(bad.as_ref());
}

#[test]
fn jacobian_action_matches_finite_difference() {
    let problem = problem();
    let k = kernel();
    let op = problem.tensor_residual_operator(&k);
    let n = problem.system_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.01 * i as f64);
    let direction = Mat::from_fn(n, 1, |i, _| (0.13 * i as f64).sin());
    let eps = 1e-7;
    let perturbed = state.as_ref() + faer::Scale(eps) * direction.as_ref();
    let base = op.residual(state.as_ref());
    let pert = op.residual(perturbed.as_ref());
    let action = op.apply_jacobian(state.as_ref(), direction.as_ref());
    for i in 0..n {
        let fd = (pert[i] - base[i]) / eps;
        assert!(
            (action[(i, 0)] - fd).abs() < 1e-6,
            "FD mismatch at {i}: {} != {fd}",
            action[(i, 0)]
        );
    }
}
