use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{shapes::unit_square, SingleElementMesh};
use ormatex_sem_nd::{
    DofReduction2D, EdacNavierStokes2DConfig, FieldRegistry, KernelBoussinesq2D,
    KernelEdacMomentumConvection2D, KernelEdacNoSlipWall2D, KernelLinearReaction,
    ResidualKernelSet, SEM2DProblem, StateBoundaryTerms, StateTensorBoundaryTerms,
    TensorKernelBoussinesq2D, TensorKernelEdacMomentumConvection2D, TensorKernelEdacNoSlipWall2D,
    TensorKernelLinearReaction, TensorResidualKernelSet,
};

type QuadMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

fn problem() -> SEM2DProblem<QuadMesh> {
    SEM2DProblem::new(
        unit_square(1, 1, ReferenceCellType::Quadrilateral, 1),
        2,
        FieldRegistry::new(["u", "v", "p", "T"]),
        DofReduction2D::None,
    )
}

fn state(problem: &SEM2DProblem<QuadMesh>) -> Mat<f64> {
    Mat::from_fn(problem.system_size(), 1, |row, _| 0.13 + 0.07 * row as f64)
}

fn edac() -> KernelEdacMomentumConvection2D {
    KernelEdacMomentumConvection2D::new(EdacNavierStokes2DConfig::new(1.0, 0.01, 4.0, 0.1))
}

fn boussinesq() -> KernelBoussinesq2D {
    KernelBoussinesq2D::new(0.5, [0.0, -9.81]).with_reference_temperature(0.25)
}

fn assert_close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (index, (&actual, &expected)) in a.iter().zip(b).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-11,
            "mismatch at {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn weak_subset_terms_preserve_local_field_order_and_global_rows() {
    let problem = problem();
    let state = state(&problem);
    let set = ResidualKernelSet::from_kernel(edac()).with(boussinesq());

    let expected_edac = problem.assemble_residual(0.0, &edac(), state.as_ref());
    let expected_boussinesq = problem.assemble_residual(0.0, &boussinesq(), state.as_ref());
    let actual = problem.assemble_residual(0.0, &set, state.as_ref());
    let expected: Vec<_> = expected_edac
        .iter()
        .zip(expected_boussinesq)
        .map(|(&edac, boussinesq)| edac + boussinesq)
        .collect();
    assert_close(&actual, &expected);

    let direction = Mat::from_fn(problem.system_size(), 1, |row, _| (row as f64 * 0.31).sin());
    let expected_edac = problem.apply_jacobian(0.0, &edac(), state.as_ref(), direction.as_ref());
    let expected_boussinesq =
        problem.apply_jacobian(0.0, &boussinesq(), state.as_ref(), direction.as_ref());
    let actual = problem.apply_jacobian(0.0, &set, state.as_ref(), direction.as_ref());
    for row in 0..problem.system_size() {
        assert!(
            (actual[(row, 0)] - expected_edac[(row, 0)] - expected_boussinesq[(row, 0)]).abs()
                < 1e-11
        );
    }
}

#[test]
fn tensor_subset_terms_match_separate_tensor_assembly() {
    let problem = problem();
    let state = state(&problem);
    let config = EdacNavierStokes2DConfig::new(1.0, 0.01, 4.0, 0.1);
    let set =
        TensorResidualKernelSet::from_kernel(TensorKernelEdacMomentumConvection2D::new(config))
        .with(TensorKernelBoussinesq2D::new(0.5, [0.0, -9.81]));

    let edac_values = problem
        .tensor_residual_operator(&TensorKernelEdacMomentumConvection2D::new(config))
        .residual(state.as_ref());
    let boussinesq_values = problem
        .tensor_residual_operator(&TensorKernelBoussinesq2D::new(0.5, [0.0, -9.81]))
        .residual(state.as_ref());
    let actual = problem
        .tensor_residual_operator(&set)
        .residual(state.as_ref());
    let expected: Vec<_> = edac_values
        .iter()
        .zip(boussinesq_values)
        .map(|(&edac, boussinesq)| edac + boussinesq)
        .collect();
    assert_close(&actual, &expected);
}

#[test]
fn existing_edac_state_boundary_binds_to_its_subset() {
    let problem = problem();
    let state = state(&problem);
    let terms = StateBoundaryTerms::new().with_default(KernelEdacNoSlipWall2D);
    let edac = edac();
    let operator = problem.residual_operator(&edac).with_state_boundary(&terms);
    let residual = operator.residual(state.as_ref());
    let jacobian = operator.assemble_jacobian(state.as_ref());
    assert_eq!(residual.len(), problem.system_size());
    assert_eq!(jacobian.nrows(), problem.system_size());
    assert_eq!(jacobian.ncols(), problem.system_size());
}

#[test]
fn named_reaction_can_target_species_inside_a_larger_problem() {
    let problem = problem();
    let rates = SparseColMat::try_new_from_triplets(
        2,
        2,
        &[Triplet::new(0, 0, 2.0), Triplet::new(1, 0, -0.5)],
    )
    .unwrap();
    let kernel = KernelLinearReaction::with_field_names(rates, ["T", "p"]);
    let matrix = problem.assemble_bilinear(0.0, &kernel).to_dense();
    let t = problem.field_reduced_size(problem.field_id("T").unwrap());
    let p = problem.field_reduced_size(problem.field_id("p").unwrap());
    let t_offset = problem.field_offset(problem.field_id("T").unwrap());
    let p_offset = problem.field_offset(problem.field_id("p").unwrap());
    assert!(matrix[(t_offset, t_offset)] != 0.0);
    assert!(matrix[(p_offset, t_offset)] != 0.0);
    assert_eq!(matrix[(0, t_offset)], 0.0);
    assert_eq!(matrix[(t_offset, 0)], 0.0);
    assert_eq!(t, p);

    let tensor_rates = SparseColMat::try_new_from_triplets(
        2,
        2,
        &[Triplet::new(0, 0, 2.0), Triplet::new(1, 0, -0.5)],
    )
    .unwrap();
    let tensor = TensorKernelLinearReaction::with_field_names(tensor_rates, ["T", "p"]);
    let _ = problem
        .tensor_residual_operator(&tensor)
        .residual(state(&problem).as_ref());
}

#[test]
fn existing_edac_tensor_boundary_binds_to_its_subset() {
    let problem = problem();
    let state = state(&problem);
    let terms = StateTensorBoundaryTerms::<2>::new().with_default(TensorKernelEdacNoSlipWall2D);
    let kernel = TensorKernelEdacMomentumConvection2D::new(EdacNavierStokes2DConfig::new(
        1.0, 0.01, 4.0, 0.1,
    ));
    let operator = problem
        .tensor_residual_operator(&kernel)
        .with_state_boundary(&terms);
    let residual = operator.residual(state.as_ref());
    let jacobian = operator.assemble_jacobian(state.as_ref());
    assert_eq!(residual.len(), problem.system_size());
    assert_eq!(jacobian.nrows(), problem.system_size());
}
