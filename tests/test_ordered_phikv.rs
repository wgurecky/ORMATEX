use faer::prelude::*;
use faer::Par;
use ormatex::matexp_cauchy::gen_cram_expm;
use ormatex::matexp_krylov::KrylovExpm;
use ormatex::matexp_leja::{LejaEllipseAdapterArnoldiIOM, LejaPhiEval, LejaPoints};
use ormatex::matexp_pade::PadeExpm;
use ormatex::matexp_traits::{DensePhikvEvaluator, LinOpPhikvEvaluator};
use ormatex::ode_sys::DynRefExtendedLinOp;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

fn matrix() -> Mat<f64> {
    Mat::from_fn(12, 12, |i, j| {
        if i == j { -0.5 - i as f64 / 10.0 }
        else if i + 1 == j { 0.3 }
        else if j + 1 == i { -0.2 }
        else { 0.0 }
    })
}

fn leja(reuse: bool, substeps: usize) -> LejaPhiEval {
    let adapter = LejaEllipseAdapterArnoldiIOM::new(-2.0, 0.0, 1.0, 1e-8, 8, 8, 1.1);
    let mut eval = LejaPhiEval::new(
        LejaPoints::new_from_lib("leja_circle").slice(0, 180),
        160, 1e-13, "clapm", "dd_taylor", reuse, Box::new(adapter),
    );
    eval.set_max_substeps(substeps);
    eval
}

#[test]
fn trait_objects_accept_arrays_and_subslices() {
    let a = matrix();
    let v0 = Mat::<f64>::ones(12, 1);
    let v1 = Mat::from_fn(12, 1, |i, _| (i as f64 / 3.0).sin());
    let v2 = Mat::from_fn(12, 1, |i, _| (i as f64 / 4.0).cos());
    let vectors = [v0.as_ref(), v1.as_ref(), v2.as_ref()];
    let orders = [1, 2, 0];
    let vb = &vectors[1..];
    let ks = &orders[1..];
    let scale = 0.4;
    let pade = PadeExpm::new(12);
    let expected = pade.apply_phi_k(a.as_ref(), scale, v1.as_ref(), 2, Par::Seq)
        + pade.apply_phi_k(a.as_ref(), scale, v2.as_ref(), 0, Par::Seq);

    let mut dense: Box<dyn DensePhikvEvaluator> = Box::new(gen_cram_expm(16));
    dense.apply_prepare(a.as_ref(), scale, vb, ks, Par::Seq);
    let dense_slice = dense.apply_phi_k_v(a.as_ref(), scale, vb, ks, Par::Seq);
    let dense_array = dense.apply_phi_k_v(a.as_ref(), scale, &[v1.as_ref(), v2.as_ref()], &[2, 0], Par::Seq);
    assert!((dense_slice - expected.as_ref()).norm_max() < 1e-10);
    assert!((dense_array - expected.as_ref()).norm_max() < 1e-10);

    let ext = DynRefExtendedLinOp::new(scale, &a, vb, ks);
    let mut evaluators: [Box<dyn LinOpPhikvEvaluator>; 2] = [
        Box::new(KrylovExpm::new(Box::new(PadeExpm::new(12)), 12, 30, 1e-12, Some(12))),
        Box::new(leja(true, 0)),
    ];
    for eval in &mut evaluators {
        eval.apply_prepare(&ext, scale, vb, ks, Par::Seq);
        let actual = eval.apply_phi_k_v(&ext, 1.0, vb, ks, Par::Seq);
        assert!((actual - expected.as_ref()).norm_max() < 1e-8);
    }
}

#[test]
#[should_panic]
fn dense_slices_reject_mismatched_lengths() {
    let a = matrix();
    let v = Mat::<f64>::ones(12, 1);
    PadeExpm::new(12).apply_phi_k_v(a.as_ref(), 0.4, &[v.as_ref()], &[], Par::Seq);
}

#[test]
#[should_panic]
fn cauchy_slices_reject_empty_input() {
    let a = matrix();
    gen_cram_expm(16).apply_phi_k_v(a.as_ref(), 0.4, &[], &[], Par::Seq);
}

#[test]
fn arbitrary_orders_match_dense_reference() {
    let a = matrix();
    let vectors: Vec<_> = (0..4).map(|j| {
        Mat::from_fn(12, 1, |i, _| (((i + 1) * (j + 1)) as f64 / 7.0).sin())
    }).collect();
    let scale = 0.4;
    let pade = PadeExpm::new(12);
    for ks in [vec![2, 0], vec![3], vec![2, 2, 0], vec![0, 0, 1], vec![0], vec![0, 1, 2]] {
        let vb: Vec<_> = vectors[..ks.len()].iter().map(|v| v.as_ref()).collect();
        let ext = DynRefExtendedLinOp::new(scale, &a, &vb, &ks);
        for tau in [1.0_f64, 0.6] {
            let mut reference = Mat::<f64>::zeros(12, 1);
            for (v, &k) in vb.iter().zip(&ks) {
                reference += tau.powi(k as i32) * pade.apply_phi_k(a.as_ref(), tau * scale, *v, k, Par::Seq);
            }
            for substeps in [0, 3] {
                let mut krylov = KrylovExpm::new(Box::new(PadeExpm::new(12)), 12, 30, 1e-12, Some(12));
                krylov.set_max_substeps(substeps);
                let actual = krylov.apply_phi_k_v(&ext, tau, &vb, &ks, Par::Seq);
                assert!((actual - reference.as_ref()).norm_max() < 1e-9, "Krylov ks={ks:?}, tau={tau}, substeps={substeps}");
                for reuse in [false, true] {
                    let mut eval = leja(reuse, substeps);
                    eval.apply_prepare(&ext, scale, &vb, &ks, Par::Seq);
                    let actual = eval.apply_phi_k_v(&ext, tau, &vb, &ks, Par::Seq);
                    assert!((actual - reference.as_ref()).norm_max() < 1e-8, "Leja ks={ks:?}, tau={tau}, substeps={substeps}, reuse={reuse}");
                }
            }
        }
    }
}

#[test]
fn cauchy_handles_reordered_and_repeated_zero_orders() {
    let a = matrix();
    let v0 = Mat::from_fn(12, 1, |i, _| (i as f64 / 3.0).sin());
    let v1 = Mat::from_fn(12, 1, |i, _| (i as f64 / 4.0).cos());
    let vb = vec![v0.as_ref(), v1.as_ref(), v0.as_ref()];
    let ks = vec![2, 0, 0];
    let pade = PadeExpm::new(12);
    let expected = pade.apply_phi_k_v(a.as_ref(), 0.4, &vb, &ks, Par::Seq);
    let mut cram = gen_cram_expm(16);
    let uncached = cram.apply_phi_k_v(a.as_ref(), 0.4, &vb, &ks, Par::Seq);
    cram.apply_prepare(a.as_ref(), 0.4, &vb, &ks, Par::Seq);
    let cached = cram.apply_phi_k_v(a.as_ref(), 0.4, &vb, &ks, Par::Seq);
    assert!((uncached - expected.as_ref()).norm_max() < 1e-10);
    assert!((cached - expected).norm_max() < 1e-10);
}

#[test]
fn leja_reprepares_for_a_different_problem() {
    let a = matrix();
    let a1 = 0.8 * a.as_ref();
    let v0 = Mat::from_fn(12, 1, |i, _| (i as f64 / 3.0).sin());
    let v1 = Mat::from_fn(12, 1, |i, _| (i as f64 / 4.0).cos());
    let vb0 = vec![v0.as_ref()];
    let ks0 = vec![2];
    let ext0 = DynRefExtendedLinOp::new(0.4, &a, &vb0, &ks0);
    let vb1 = vec![v1.as_ref()];
    let ks1 = vec![3];
    let ext1 = DynRefExtendedLinOp::new(0.4, &a1, &vb1, &ks1);
    let mut eval = leja(true, 3);
    eval.apply_prepare(&ext0, 0.4, &vb0, &ks0, Par::Seq);
    let expected0 = PadeExpm::new(12).apply_phi_k(a.as_ref(), 0.4, v0.as_ref(), 2, Par::Seq);
    let first = eval.apply_phi_k_v(&ext0, 1.0, &vb0, &ks0, Par::Seq);
    assert!((first - expected0).norm_max() < 1e-8);
    eval.apply_prepare(&ext1, 0.4, &vb1, &ks1, Par::Seq);
    let expected = PadeExpm::new(12).apply_phi_k(a1.as_ref(), 0.4, v1.as_ref(), 3, Par::Seq);
    let other = eval.apply_phi_k_v(&ext1, 1.0, &vb1, &ks1, Par::Seq);
    let single = eval.apply_phi_k(&a1, 0.4, v1.as_ref(), 3, Par::Seq);
    assert!((other - expected.as_ref()).norm_max() < 1e-8);
    assert!((single - expected).norm_max() < 1e-8);
}

#[test]
fn enabling_leja_reuse_after_preparation_does_not_use_stale_data() {
    let a = matrix();
    let v = Mat::<f64>::ones(12, 1);
    let vb = vec![v.as_ref()];
    let ks = vec![2];
    let ext = DynRefExtendedLinOp::new(0.4, &a, &vb, &ks);
    let mut eval = leja(false, 0);
    eval.apply_prepare(&ext, 0.4, &vb, &ks, Par::Seq);
    eval.set_krylov_reuse(true);
    let actual = eval.apply_phi_k_v(&ext, 1.0, &vb, &ks, Par::Seq);
    let expected = PadeExpm::new(12).apply_phi_k(a.as_ref(), 0.4, v.as_ref(), 2, Par::Seq);
    assert!((actual - expected).norm_max() < 1e-8);
}

#[test]
fn repeated_orders_can_cancel() {
    let a = matrix();
    let v = Mat::<f64>::ones(12, 1);
    let minus_v = -v.as_ref();
    let vb = vec![v.as_ref(), minus_v.as_ref()];
    let ks = vec![2, 2];
    let ext = DynRefExtendedLinOp::new(0.4, &a, &vb, &ks);
    let mut krylov = KrylovExpm::new(Box::new(PadeExpm::new(12)), 12, 30, 1e-12, None);
    assert!(krylov.apply_phi_k_v(&ext, 1.0, &vb, &ks, Par::Seq).norm_max() < 1e-12);
    let mut eval = leja(true, 0);
    eval.apply_prepare(&ext, 0.4, &vb, &ks, Par::Seq);
    assert!(eval.apply_phi_k_v(&ext, 1.0, &vb, &ks, Par::Seq).norm_max() < 1e-12);
}

#[test]
fn leja_zero_taylor_seed_can_have_nonzero_result() {
    let a = -Mat::<f64>::identity(12, 12);
    let v = Mat::<f64>::ones(12, 1);
    let half_v = 0.5 * v.as_ref();
    let vb = vec![v.as_ref(), half_v.as_ref()];
    let ks = vec![1, 2];
    let ext = DynRefExtendedLinOp::new(0.5, &a, &vb, &ks);
    let mut eval = leja(true, 3);
    eval.apply_prepare(&ext, 0.5, &vb, &ks, Par::Seq);
    let actual = eval.apply_phi_k_v(&ext, 1.0, &vb, &ks, Par::Seq);
    assert!((actual - v).norm_max() < 1e-10);
}

#[test]
#[should_panic(expected = "orders differ from the extended operator")]
fn mismatched_orders_are_rejected() {
    let a = matrix();
    let v = Mat::<f64>::ones(12, 1);
    let vb = vec![v.as_ref()];
    let ext = DynRefExtendedLinOp::new(0.4, &a, &vb, &[2]);
    let mut eval = KrylovExpm::new(Box::new(PadeExpm::new(12)), 12, 30, 1e-12, None);
    eval.apply_phi_k_v(&ext, 1.0, &vb, &vec![0], Par::Seq);
}

#[test]
fn krylov_keeps_configured_dense_parallelism() {
    struct RecordingPade(Arc<AtomicBool>);
    impl DensePhikvEvaluator for RecordingPade {
        fn apply_phi_k(&self, a: MatRef<f64>, dt: f64, v: MatRef<f64>, k: usize, par: Par) -> Mat<f64> {
            self.0.store(matches!(par, Par::Seq), Ordering::Relaxed);
            PadeExpm::new(12).apply_phi_k(a, dt, v, k, par)
        }
    }
    let a = matrix();
    let v = Mat::<f64>::ones(12, 1);
    let observed_seq = Arc::new(AtomicBool::new(false));
    let mut eval = KrylovExpm::new(Box::new(RecordingPade(observed_seq.clone())), 12, 30, 1e-12, None);
    eval.set_dense_par(Par::Seq);
    eval.apply_phi_k(&a, 0.4, v.as_ref(), 1, Par::rayon(2));
    assert!(observed_seq.load(Ordering::Relaxed));
}
