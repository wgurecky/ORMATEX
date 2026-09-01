use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};
use ndmesh::shapes::unit_interval;
use ormatex_sem_nd::{
    DofReduction1D, FieldRegistry, KernelLinearReaction, ResidualKernel, SEM1DProblem,
    TensorKernelLinearReaction,
};

fn rates() -> [[f64; 3]; 3] {
    [[-0.1, 0.0, 0.0], [0.1, -10.0, 0.0], [0.0, 10.0, -0.01]]
}

fn kernel() -> KernelLinearReaction {
    let matrix = rates();
    let triplets = (0..3)
        .flat_map(|row| {
            (0..3).filter_map(move |column| {
                (matrix[row][column] != 0.0).then_some(Triplet::new(
                    row,
                    column,
                    matrix[row][column],
                ))
            })
        })
        .collect::<Vec<_>>();
    KernelLinearReaction::with_field_names(
        SparseColMat::try_new_from_triplets(3, 3, &triplets).unwrap(),
        ["c0", "c1", "c2"],
    )
}

#[test]
fn sparse_linear_reaction_assembles_expected_blocks_and_action() {
    let problem = SEM1DProblem::new(
        unit_interval(1),
        2,
        FieldRegistry::new(["c0", "c1", "c2"]),
        DofReduction1D::None,
    );
    let n = problem.reduced_size();
    let weak_kernel = kernel();
    assert_eq!(ResidualKernel::nfields(&weak_kernel), 3);

    let state = Mat::from_fn(3 * n, 1, |row, _| 0.2 + 0.03 * row as f64);
    let direction = Mat::from_fn(3 * n, 1, |row, _| (0.2 * row as f64).sin());
    let mass = problem.assemble_lumped_mass().to_dense();
    let jacobian = problem
        .assemble_residual_jacobian(0.0, &weak_kernel, state.as_ref())
        .to_dense();
    let matrix = rates();

    for equation in 0..3 {
        for unknown in 0..3 {
            for test in 0..n {
                for trial in 0..n {
                    let expected = -matrix[equation][unknown] * mass[(test, trial)];
                    assert!(
                        (jacobian[(equation * n + test, unknown * n + trial)] - expected).abs()
                            < 1e-12
                    );
                }
            }
        }
    }
    assert_ne!(jacobian[(n, 0)], 0.0);
    assert_ne!(jacobian[(2 * n, n)], 0.0);
    assert_eq!(jacobian[(0, n)], 0.0);

    let residual = problem.assemble_residual(0.0, &weak_kernel, state.as_ref());
    let expected_residual = jacobian.as_ref() * state.as_ref();
    for row in 0..3 * n {
        assert!((residual[row] - expected_residual[(row, 0)]).abs() < 1e-12);
    }

    let action = problem.apply_jacobian(0.0, &weak_kernel, state.as_ref(), direction.as_ref());
    let expected_action = jacobian.as_ref() * direction.as_ref();
    for row in 0..3 * n {
        assert!((action[(row, 0)] - expected_action[(row, 0)]).abs() < 1e-12);
    }

    let tensor_kernel = TensorKernelLinearReaction(kernel());
    let tensor_operator = problem.tensor_residual_operator(&tensor_kernel);
    let tensor_residual = tensor_operator.residual(state.as_ref());
    for row in 0..3 * n {
        assert!((tensor_residual[row] - residual[row]).abs() < 1e-12);
    }
    let tensor_jacobian = tensor_operator.assemble_jacobian(state.as_ref()).to_dense();
    for row in 0..3 * n {
        for col in 0..3 * n {
            assert!((tensor_jacobian[(row, col)] - jacobian[(row, col)]).abs() < 1e-12);
        }
    }
    let tensor_action = tensor_operator.apply_jacobian(state.as_ref(), direction.as_ref());
    for row in 0..3 * n {
        assert!((tensor_action[(row, 0)] - action[(row, 0)]).abs() < 1e-12);
    }
}
