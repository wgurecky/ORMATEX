//! Fused split-form EDAC kernels equal the sum of their part kernels.
//!
//! `KernelEdacNavierStokesSplit2D` / `TensorKernelEdacNavierStokesSplit2D`
//! delegate every integrand to the six `*_split` / conservative part kernels,
//! so this checks residual, assembled Jacobian, and matrix-free action agree
//! on a nontrivial state (zeros would hide the convection terms).

use faer::prelude::*;
use ormatex_sem_nd::{
    EdacNavierStokes2DConfig, KernelEdacMomentumConvectionSplit2D, KernelEdacNavierStokesSplit2D,
    KernelEdacPressureAdvectionSplit2D, KernelEdacPressureDiffusion2D,
    KernelEdacPressureDivergence2D, KernelEdacPressureGradient2D, KernelEdacViscousStress2D,
    ResidualKernelSum, TensorKernelEdacMomentumConvectionSplit2D,
    TensorKernelEdacNavierStokesSplit2D, TensorKernelEdacPressureAdvectionSplit2D,
    TensorKernelEdacPressureDiffusion2D, TensorKernelEdacPressureDivergence2D,
    TensorKernelEdacPressureGradient2D, TensorKernelEdacViscousStress2D,
    TensorResidualKernelSum, WeakResidualOps,
};

#[path = "../examples/support/cavity_setup.rs"]
mod cavity_setup;

fn config() -> EdacNavierStokes2DConfig {
    EdacNavierStokes2DConfig::new(1.0, 0.01, 4.0, 0.1)
}

fn weak_sum() -> ResidualKernelSum<'static> {
    let config = config();
    ResidualKernelSum::from_kernel(KernelEdacMomentumConvectionSplit2D::new(config))
        .with(KernelEdacPressureGradient2D::new(config))
        .with(KernelEdacViscousStress2D::new(config))
        .with(KernelEdacPressureDivergence2D::new(config))
        .with(KernelEdacPressureAdvectionSplit2D::new(config))
        .with(KernelEdacPressureDiffusion2D::new(config))
}

fn tensor_sum() -> impl ormatex_sem_nd::TensorResidualKernel<2> {
    let config = config();
    TensorResidualKernelSum::from_kernel(TensorKernelEdacMomentumConvectionSplit2D::new(config))
        .with(TensorKernelEdacPressureGradient2D::new(config))
        .with(TensorKernelEdacViscousStress2D::new(config))
        .with(TensorKernelEdacPressureDivergence2D::new(config))
        .with(TensorKernelEdacPressureAdvectionSplit2D::new(config))
        .with(TensorKernelEdacPressureDiffusion2D::new(config))
}

/// Deterministic non-trivial state (zeros would hide convection terms).
fn state(problem: &ormatex_sem_nd::SEM2DProblem<ormatex_sem_nd::QuadMesh>) -> Mat<f64> {
    let n = problem.system_size();
    Mat::<f64>::from_fn(n, 1, |row, _| ((row * 37) % 11) as f64 * 0.13 - 0.5)
}

fn direction(problem: &ormatex_sem_nd::SEM2DProblem<ormatex_sem_nd::QuadMesh>) -> Mat<f64> {
    let n = problem.system_size();
    Mat::<f64>::from_fn(n, 1, |row, _| ((row * 53) % 7) as f64 * 0.17 - 0.4)
}

fn max_abs(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |m, &v| m.max(v.abs()))
}

#[test]
fn fused_split_weak_matches_sum_of_parts() {
    let problem = cavity_setup::problem();
    let state = state(&problem);
    let direction = direction(&problem);
    let fused = KernelEdacNavierStokesSplit2D::new(config());
    let parts = weak_sum();

    let fused_residual = problem.assemble_residual(0.0, &fused, state.as_ref());
    let parts_residual = problem.assemble_residual(0.0, &parts, state.as_ref());
    let residual_diff: Vec<f64> = fused_residual
        .iter()
        .zip(&parts_residual)
        .map(|(a, b)| a - b)
        .collect();
    assert!(max_abs(&residual_diff) < 1e-12);

    let fused_jacobian = problem
        .assemble_residual_jacobian(0.0, &fused, state.as_ref())
        .to_dense();
    let parts_jacobian = problem
        .assemble_residual_jacobian(0.0, &parts, state.as_ref())
        .to_dense();
    let mut jacobian_diff = 0.0f64;
    for row in 0..fused_jacobian.nrows() {
        for col in 0..fused_jacobian.ncols() {
            jacobian_diff = jacobian_diff.max((fused_jacobian[(row, col)] - parts_jacobian[(row, col)]).abs());
        }
    }
    assert!(jacobian_diff < 1e-12);

    let fused_action =
        problem.apply_jacobian(0.0, &fused, state.as_ref(), direction.as_ref());
    let parts_action =
        problem.apply_jacobian(0.0, &parts, state.as_ref(), direction.as_ref());
    let mut action_diff = 0.0f64;
    for row in 0..fused_action.nrows() {
        action_diff =
            action_diff.max((fused_action[(row, 0)] - parts_action[(row, 0)]).abs());
    }
    assert!(action_diff < 1e-12);
}

#[test]
fn fused_split_tensor_matches_sum_of_parts() {
    let problem = cavity_setup::problem();
    let state = state(&problem);
    let direction = direction(&problem);
    let fused = TensorKernelEdacNavierStokesSplit2D::new(config());
    let parts = tensor_sum();

    let fused_operator = problem.tensor_residual_operator(&fused);
    let parts_operator = problem.tensor_residual_operator(&parts);

    let fused_residual = fused_operator.residual(state.as_ref());
    let parts_residual = parts_operator.residual(state.as_ref());
    let residual_diff: Vec<f64> = fused_residual
        .iter()
        .zip(&parts_residual)
        .map(|(a, b)| a - b)
        .collect();
    assert!(max_abs(&residual_diff) < 1e-12);

    let fused_jacobian = fused_operator
        .assemble_jacobian(state.as_ref())
        .to_dense();
    let parts_jacobian = parts_operator
        .assemble_jacobian(state.as_ref())
        .to_dense();
    let mut jacobian_diff = 0.0f64;
    for row in 0..fused_jacobian.nrows() {
        for col in 0..fused_jacobian.ncols() {
            jacobian_diff = jacobian_diff.max((fused_jacobian[(row, col)] - parts_jacobian[(row, col)]).abs());
        }
    }
    assert!(jacobian_diff < 1e-12);

    let fused_action = fused_operator.apply_jacobian(state.as_ref(), direction.as_ref());
    let parts_action = parts_operator.apply_jacobian(state.as_ref(), direction.as_ref());
    let mut action_diff = 0.0f64;
    for row in 0..fused_action.nrows() {
        action_diff =
            action_diff.max((fused_action[(row, 0)] - parts_action[(row, 0)]).abs());
    }
    assert!(action_diff < 1e-12);
}
