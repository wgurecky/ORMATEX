//! Facet-batched boundary driver consistency tests.
//!
//! Through the public tensor operator on a 3x3 unit-square problem (12
//! boundary facets, so batches of 8 + 4 exercise padded lanes), this checks
//! the batched driver of `StateTensorBoundaryTerms<2>`: the assembled
//! Jacobian matches central finite differences of the residual and the
//! matrix-free action matches the assembled matvec, for every owned boundary
//! kernel and one multi-kernel batching.
use faer::prelude::*;
use faer::sparse::SparseColMat;
use ndelement::types::ReferenceCellType;
use ndmesh::shapes::unit_square;
use ormatex_sem_nd::{
    DofReduction2D, FieldRegistry, SEM2DProblem, TensorKernelEdacDirectionalDoNothing2D,
    TensorKernelEdacDongOutflow2D, TensorKernelEdacNavierStokes2D, TensorKernelEdacNoSlipWall2D,
    TensorKernelEdacSlipWall2D, TensorKernelEdacSplitBoundaryFlux2D,
};

type QuadMesh = ndmesh::SingleElementMesh<
    f64,
    ndelement::ciarlet::CiarletElement<f64, ndelement::map::IdentityMap, f64>,
>;

/// Build a small 3-field EDAC tensor problem (12 boundary facets).
fn problem() -> SEM2DProblem<QuadMesh> {
    SEM2DProblem::new(
        unit_square(3, 3, ReferenceCellType::Quadrilateral, 1),
        2,
        FieldRegistry::new(["u", "v", "p"]),
        DofReduction2D::None,
    )
}

/// Conservative EDAC volume kernel paired with every boundary term below.
fn kernel() -> TensorKernelEdacNavierStokes2D {
    TensorKernelEdacNavierStokes2D::new(1.0, 0.1, 10.0, 0.0)
}

/// Smooth state with both signs of velocity (covers inflow and outflow).
fn state(n: usize) -> Mat<f64> {
    Mat::from_fn(n, 1, |i, _| 0.4 + 0.25 * ((0.37 * i as f64).sin()))
}

/// Two-column direction for the action tests.
fn direction(n: usize) -> Mat<f64> {
    Mat::from_fn(n, 2, |i, c| (0.23 * i as f64 + 1.7 * c as f64).cos())
}

/// Dense matvec `y = m * x` for one column.
fn matvec(m: MatRef<'_, f64>, x: &[f64]) -> Vec<f64> {
    assert_eq!(m.ncols(), x.len());
    (0..m.nrows())
        .map(|r| (0..m.ncols()).map(|c| m[(r, c)] * x[c]).sum())
        .collect()
}

/// Max-norm relative difference between two vectors.
fn rel_diff(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f64::max);
    let den: f64 = a
        .iter()
        .chain(b.iter())
        .map(|x| x.abs())
        .fold(0.0, f64::max)
        .max(1e-30);
    num / den
}

/// Check the assembled Jacobian against central finite differences of the
/// residual, and the matrix-free action against the assembled matvec.
fn check_linearization(
    problem: &SEM2DProblem<QuadMesh>,
    k: &TensorKernelEdacNavierStokes2D,
    terms: &ormatex_sem_nd::StateTensorBoundaryTerms<2>,
    what: &str,
) {
    let n = problem.system_size();
    let state = state(n);
    let direction = direction(n);
    let op = problem
        .tensor_residual_operator(k)
        .with_state_boundary(terms);

    let jac: SparseColMat<usize, f64> = op.assemble_jacobian(state.as_ref());
    let dense = jac.to_dense();

    // Action matches the assembled matvec per column.
    let mut action = Mat::zeros(n, 2);
    op.apply_jacobian_into(state.as_ref(), direction.as_ref(), action.as_mut());
    for c in 0..2 {
        let x: Vec<f64> = (0..n).map(|r| direction[(r, c)]).collect();
        let expected = matvec(dense.as_ref(), &x);
        let got: Vec<f64> = (0..n).map(|r| action[(r, c)]).collect();
        assert!(
            rel_diff(&got, &expected) < 1e-9,
            "{what}: action vs assembled column {c}: {}",
            rel_diff(&got, &expected)
        );
    }

    // Assembled Jacobian matches finite differences of the residual.
    let h = 1e-5;
    for c in 0..2 {
        let mut plus = state.clone();
        let mut minus = state.clone();
        for r in 0..n {
            plus[(r, 0)] += h * direction[(r, c)];
            minus[(r, 0)] -= h * direction[(r, c)];
        }
        let r_plus = op.residual(plus.as_ref());
        let r_minus = op.residual(minus.as_ref());
        let fd: Vec<f64> = r_plus
            .iter()
            .zip(&r_minus)
            .map(|(&a, &b)| (a - b) / (2.0 * h))
            .collect();
        let x: Vec<f64> = (0..n).map(|r| direction[(r, c)]).collect();
        let expected = matvec(dense.as_ref(), &x);
        assert!(
            rel_diff(&fd, &expected) < 1e-6,
            "{what}: finite difference vs assembled column {c}: {}",
            rel_diff(&fd, &expected)
        );
    }
}

#[test]
fn single_kernel_terms_linearize() {
    let problem = problem();
    let k = kernel();
    let cases: Vec<(&str, ormatex_sem_nd::StateTensorBoundaryTerms<2>)> = vec![
        (
            "split",
            ormatex_sem_nd::StateTensorBoundaryTerms::new()
                .with_default(TensorKernelEdacSplitBoundaryFlux2D),
        ),
        (
            "dong",
            ormatex_sem_nd::StateTensorBoundaryTerms::new()
                .with_default(TensorKernelEdacDongOutflow2D::new(1.0, 0.1, 1.0)),
        ),
        (
            "dong-split",
            ormatex_sem_nd::StateTensorBoundaryTerms::new()
                .with_default(TensorKernelEdacDongOutflow2D::new(1.0, 0.1, 1.0).with_split_flux()),
        ),
        (
            "directional",
            ormatex_sem_nd::StateTensorBoundaryTerms::new()
                .with_default(TensorKernelEdacDirectionalDoNothing2D::new(1.0)),
        ),
        (
            "directional-split",
            ormatex_sem_nd::StateTensorBoundaryTerms::new()
                .with_default(TensorKernelEdacDirectionalDoNothing2D::new(1.0).with_split_flux()),
        ),
        (
            "no-slip",
            ormatex_sem_nd::StateTensorBoundaryTerms::new()
                .with_default(TensorKernelEdacNoSlipWall2D::new()),
        ),
        (
            "slip",
            ormatex_sem_nd::StateTensorBoundaryTerms::new()
                .with_default(TensorKernelEdacSlipWall2D::new()),
        ),
    ];
    for (what, terms) in &cases {
        check_linearization(&problem, &k, terms, what);
    }
}

#[test]
fn multi_kernel_terms_linearize() {
    let problem = problem();
    let k = kernel();
    // Even facets: Dong outflow; odd facets: directional do-nothing; the
    // default is never hit but keeps the terms well-formed. Entity indices
    // beyond the mesh facet count are ignored.
    let even: Vec<usize> = (0..64).step_by(2).collect();
    let odd: Vec<usize> = (1..64).step_by(2).collect();
    let terms = ormatex_sem_nd::StateTensorBoundaryTerms::new()
        .with_default(TensorKernelEdacSplitBoundaryFlux2D)
        .with_entities(
            even,
            TensorKernelEdacDongOutflow2D::new(1.0, 0.1, 1.0).with_split_flux(),
        )
        .with_entities(
            odd,
            TensorKernelEdacDirectionalDoNothing2D::new(1.0).with_split_flux(),
        );
    check_linearization(&problem, &k, &terms, "multi-kernel");
}
