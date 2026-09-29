//! Assembled-Jacobian determinism regression tests.
//!
//! The shared Jacobian pattern cache must fill values in the same summation
//! order on the first build as on every later reuse: `try_new_from_triplets`
//! sums duplicate triplets in sorted order while the cached path accumulates
//! in triplet order, which differed by 1 ulp on shared-DOF entries (e.g.
//! entry (108, 10) on the 3x3 problem below). These tests pin repeat-assembly
//! bitwise equality, including across the volume/boundary operator sum.
use faer::prelude::*;
use faer::sparse::SparseColMat;
use ndelement::types::ReferenceCellType;
use ndmesh::shapes::unit_square;
use ormatex_sem_nd::{
    DofReduction2D, FieldRegistry, SEM2DProblem, TensorKernelEdacNavierStokes2D,
    TensorKernelEdacSplitBoundaryFlux2D,
};

type QuadMesh = ndmesh::SingleElementMesh<
    f64,
    ndelement::ciarlet::CiarletElement<f64, ndelement::map::IdentityMap, f64>,
>;

fn problem() -> SEM2DProblem<QuadMesh> {
    SEM2DProblem::new(
        unit_square(3, 3, ReferenceCellType::Quadrilateral, 1),
        2,
        FieldRegistry::new(["u", "v", "p"]),
        DofReduction2D::None,
    )
}

fn kernel() -> TensorKernelEdacNavierStokes2D {
    TensorKernelEdacNavierStokes2D::new(1.0, 0.1, 10.0, 0.0)
}

fn state(n: usize) -> Mat<f64> {
    Mat::from_fn(n, 1, |i, _| 0.4 + 0.25 * ((0.37 * i as f64).sin()))
}

fn assert_dense_bits_eq(a: MatRef<'_, f64>, b: MatRef<'_, f64>, what: &str) {
    assert_eq!(
        (a.nrows(), a.ncols()),
        (b.nrows(), b.ncols()),
        "{what}: shape"
    );
    for col in 0..a.ncols() {
        for row in 0..a.nrows() {
            assert_eq!(
                a[(row, col)].to_bits(),
                b[(row, col)].to_bits(),
                "{what}: ({row}, {col}): {:e} != {:e}",
                a[(row, col)],
                b[(row, col)],
            );
        }
    }
}

#[test]
fn volume_repeat_assembly_bitwise() {
    let problem = problem();
    let k = kernel();
    let n = problem.system_size();
    let state = state(n);
    // Volume-only operator: the first assembly builds the shared pattern
    // cache, the second reuses it; both must agree bitwise.
    let op = problem.tensor_residual_operator(&k);
    let first: SparseColMat<usize, f64> = op.assemble_jacobian(state.as_ref());
    let second: SparseColMat<usize, f64> = op.assemble_jacobian(state.as_ref());
    assert_dense_bits_eq(
        first.to_dense().as_ref(),
        second.to_dense().as_ref(),
        "volume repeat assembly",
    );
}

#[test]
fn operator_repeat_assembly_bitwise() {
    let problem = problem();
    let k = kernel();
    let n = problem.system_size();
    let state = state(n);
    let terms = ormatex_sem_nd::StateTensorBoundaryTerms::new()
        .with_default(TensorKernelEdacSplitBoundaryFlux2D);
    let op = problem
        .tensor_residual_operator(&k)
        .with_state_boundary(&terms);
    let first: SparseColMat<usize, f64> = op.assemble_jacobian(state.as_ref());
    let second: SparseColMat<usize, f64> = op.assemble_jacobian(state.as_ref());
    assert_dense_bits_eq(
        first.to_dense().as_ref(),
        second.to_dense().as_ref(),
        "operator repeat assembly",
    );
}
