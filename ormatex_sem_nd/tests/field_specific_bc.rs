use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};
use ndelement::types::ReferenceCellType;
use ndmesh::{
    shapes::{unit_interval, unit_square},
    traits::{Entity, Geometry, Mesh, Point},
};
use ormatex_sem_nd::{
    DofReduction1D, DofReduction2D, KernelLinearReaction, SEM1DProblem, SEM2DProblem,
};

#[test]
fn one_dimensional_fields_can_have_different_reductions() {
    let problem = SEM1DProblem::new(
        unit_interval(1),
        1,
        DofReduction1D::FieldSpecific {
            reductions: vec![
                DofReduction1D::Dirichlet {
                    facets: vec![(0, 1.0)],
                },
                DofReduction1D::Dirichlet {
                    facets: vec![(1, 2.0)],
                },
            ],
        },
    );
    assert_eq!(problem.field_reduced_size(0), 1);
    assert_eq!(problem.field_reduced_size(1), 1);
    assert_eq!(problem.system_size(2), 2);
    assert!(problem.target_field_dof(0, 0).is_none());
    assert!(problem.target_field_dof(1, 1).is_none());
}

#[test]
fn two_dimensional_fields_can_have_different_reductions() {
    let mesh = unit_square(1, 1, ReferenceCellType::Quadrilateral, 1);
    let left = mesh
        .entity_iter(ReferenceCellType::Interval)
        .find(|facet| {
            facet.geometry().points().all(|point| {
                let mut xy = [0.0_f64; 2];
                point.coords(&mut xy);
                xy[0].abs() < 1e-12
            })
        })
        .unwrap()
        .local_index();
    let right = mesh
        .entity_iter(ReferenceCellType::Interval)
        .find(|facet| {
            facet.geometry().points().all(|point| {
                let mut xy = [0.0_f64; 2];
                point.coords(&mut xy);
                (xy[0] - 1.0).abs() < 1e-12
            })
        })
        .unwrap()
        .local_index();
    let problem = SEM2DProblem::new(
        mesh,
        1,
        DofReduction2D::FieldSpecific {
            reductions: vec![
                DofReduction2D::Dirichlet {
                    facets: vec![(left, 1.0)],
                },
                DofReduction2D::Dirichlet {
                    facets: vec![(right, 2.0)],
                },
            ],
        },
    );
    assert_eq!(problem.system_size(2), 4);
    assert_eq!(problem.field_reduced_size(0), 2);
    assert_eq!(problem.field_reduced_size(1), 2);
}

#[test]
fn explicit_dirichlet_values_can_override_shared_corners() {
    let problem = SEM2DProblem::new(
        unit_square(1, 1, ReferenceCellType::Quadrilateral, 1),
        1,
        DofReduction2D::DirichletValues {
            values: vec![(0, 1.0), (1, 0.0)],
        },
    );
    assert!(problem.target_dof(0).is_none());
    assert!(problem.target_dof(1).is_none());
    assert_eq!(problem.reduced_size(), 2);
}

#[test]
fn field_specific_offsets_are_used_by_multifield_assembly() {
    let problem = SEM1DProblem::new(
        unit_interval(1),
        1,
        DofReduction1D::FieldSpecific {
            reductions: vec![
                DofReduction1D::Dirichlet {
                    facets: vec![(0, 0.0)],
                },
                DofReduction1D::Dirichlet {
                    facets: vec![(1, 0.0)],
                },
            ],
        },
    );
    let rates = SparseColMat::try_new_from_triplets(
        2,
        2,
        &[
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 2.0),
            Triplet::new(1, 0, 3.0),
            Triplet::new(1, 1, 4.0),
        ],
    )
    .unwrap();
    let kernel = KernelLinearReaction::new(rates);
    let state = Mat::<f64>::zeros(problem.system_size(2), 1);
    let matrix = problem
        .assemble_system_residual_jacobian(&kernel, state.as_ref())
        .to_dense();
    assert_eq!(matrix.nrows(), problem.system_size(2));
    assert!(matrix[(0, 0)] != 0.0);
    assert!(matrix[(1, 1)] != 0.0);
}
