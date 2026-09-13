//! 1D EDAC split kernels: fused equals the sum of parts, weak equals tensor.
//!
//! Mirrors `edac_fused_split_equivalence.rs` for the two-field `[u, p]` 1D
//! family on a small interval mesh with a deterministic non-trivial state
//! (zeros would hide the convection terms).

use faer::prelude::*;
use ndmesh::shapes::unit_interval;
use ormatex_sem_nd::{
    DofReduction1D, EdacNavierStokes1DConfig, FieldRegistry, KernelEdacMomentumConvectionSplit1D,
    KernelEdacNavierStokesSplit1D, KernelEdacPressureAdvectionSplit1D,
    KernelEdacPressureDiffusion1D, KernelEdacPressureDivergence1D, KernelEdacPressureGradient1D,
    KernelEdacViscousStress1D, KernelEnergyAdvectionDiffusion1D, ResidualKernelSum, SEM1DProblem,
    TensorKernelEdacMomentumConvectionSplit1D, TensorKernelEdacNavierStokesSplit1D,
    TensorKernelEdacPressureAdvectionSplit1D, TensorKernelEdacPressureDiffusion1D,
    TensorKernelEdacPressureDivergence1D, TensorKernelEdacPressureGradient1D,
    TensorKernelEdacViscousStress1D, TensorKernelEnergyAdvectionDiffusion1D,
    TensorResidualKernelSum, WeakResidualOps,
};

type IntervalMesh = ndmesh::SingleElementMesh<
    f64,
    ndelement::ciarlet::CiarletElement<f64, ndelement::map::IdentityMap, f64>,
>;

fn config() -> EdacNavierStokes1DConfig {
    EdacNavierStokes1DConfig::new(1.0, 0.01, 4.0)
}

fn fluid_problem() -> SEM1DProblem<IntervalMesh> {
    SEM1DProblem::new(
        unit_interval(4, 1),
        2,
        FieldRegistry::new(["u", "p"]),
        DofReduction1D::None,
    )
}

fn weak_sum() -> ResidualKernelSum<'static> {
    let config = config();
    ResidualKernelSum::from_kernel(KernelEdacMomentumConvectionSplit1D::new(config))
        .with(KernelEdacPressureGradient1D::new(config))
        .with(KernelEdacViscousStress1D::new(config))
        .with(KernelEdacPressureDivergence1D::new(config))
        .with(KernelEdacPressureAdvectionSplit1D::new(config))
        .with(KernelEdacPressureDiffusion1D::new(config))
}

fn tensor_sum() -> impl ormatex_sem_nd::TensorResidualKernel<1> {
    let config = config();
    TensorResidualKernelSum::from_kernel(TensorKernelEdacMomentumConvectionSplit1D::new(config))
        .with(TensorKernelEdacPressureGradient1D::new(config))
        .with(TensorKernelEdacViscousStress1D::new(config))
        .with(TensorKernelEdacPressureDivergence1D::new(config))
        .with(TensorKernelEdacPressureAdvectionSplit1D::new(config))
        .with(TensorKernelEdacPressureDiffusion1D::new(config))
}

/// Deterministic non-trivial state (zeros would hide convection terms).
fn state(n: usize) -> Mat<f64> {
    Mat::<f64>::from_fn(n, 1, |row, _| ((row * 37) % 11) as f64 * 0.13 - 0.5)
}

fn direction(n: usize) -> Mat<f64> {
    Mat::<f64>::from_fn(n, 1, |row, _| ((row * 53) % 7) as f64 * 0.17 - 0.4)
}

fn max_abs(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |m, &v| m.max(v.abs()))
}

#[test]
fn fused_split_weak_1d_matches_sum_of_parts() {
    let problem = fluid_problem();
    let state = state(problem.system_size());
    let direction = direction(problem.system_size());
    let fused = KernelEdacNavierStokesSplit1D::new(config());
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
            jacobian_diff =
                jacobian_diff.max((fused_jacobian[(row, col)] - parts_jacobian[(row, col)]).abs());
        }
    }
    assert!(jacobian_diff < 1e-12);

    let fused_action = problem.apply_jacobian(0.0, &fused, state.as_ref(), direction.as_ref());
    let parts_action = problem.apply_jacobian(0.0, &parts, state.as_ref(), direction.as_ref());
    let mut action_diff = 0.0f64;
    for row in 0..fused_action.nrows() {
        action_diff = action_diff.max((fused_action[(row, 0)] - parts_action[(row, 0)]).abs());
    }
    assert!(action_diff < 1e-12);
}

#[test]
fn fused_split_tensor_1d_matches_sum_of_parts() {
    let problem = fluid_problem();
    let state = state(problem.system_size());
    let direction = direction(problem.system_size());
    let fused = TensorKernelEdacNavierStokesSplit1D::new(config());
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

    let fused_jacobian = fused_operator.assemble_jacobian(state.as_ref()).to_dense();
    let parts_jacobian = parts_operator.assemble_jacobian(state.as_ref()).to_dense();
    let mut jacobian_diff = 0.0f64;
    for row in 0..fused_jacobian.nrows() {
        for col in 0..fused_jacobian.ncols() {
            jacobian_diff =
                jacobian_diff.max((fused_jacobian[(row, col)] - parts_jacobian[(row, col)]).abs());
        }
    }
    assert!(jacobian_diff < 1e-12);

    let fused_action = fused_operator.apply_jacobian(state.as_ref(), direction.as_ref());
    let parts_action = parts_operator.apply_jacobian(state.as_ref(), direction.as_ref());
    let mut action_diff = 0.0f64;
    for row in 0..fused_action.nrows() {
        action_diff = action_diff.max((fused_action[(row, 0)] - parts_action[(row, 0)]).abs());
    }
    assert!(action_diff < 1e-12);
}

#[test]
fn weak_tensor_1d_agree_on_split_sum() {
    let problem = fluid_problem();
    let state = state(problem.system_size());
    let direction = direction(problem.system_size());
    let weak = weak_sum();
    let tensor = tensor_sum();

    let weak_residual = problem.assemble_residual(0.0, &weak, state.as_ref());
    let tensor_residual = problem
        .tensor_residual_operator(&tensor)
        .residual(state.as_ref());
    let residual_diff: Vec<f64> = weak_residual
        .iter()
        .zip(&tensor_residual)
        .map(|(a, b)| a - b)
        .collect();
    assert!(max_abs(&residual_diff) < 1e-12);

    let weak_jacobian = problem
        .assemble_residual_jacobian(0.0, &weak, state.as_ref())
        .to_dense();
    let tensor_jacobian = problem
        .tensor_residual_operator(&tensor)
        .assemble_jacobian(state.as_ref())
        .to_dense();
    let mut jacobian_diff = 0.0f64;
    for row in 0..weak_jacobian.nrows() {
        for col in 0..weak_jacobian.ncols() {
            jacobian_diff =
                jacobian_diff.max((weak_jacobian[(row, col)] - tensor_jacobian[(row, col)]).abs());
        }
    }
    assert!(jacobian_diff < 1e-12);

    let weak_action = problem.apply_jacobian(0.0, &weak, state.as_ref(), direction.as_ref());
    let tensor_action = problem
        .tensor_residual_operator(&tensor)
        .apply_jacobian(state.as_ref(), direction.as_ref());
    let mut action_diff = 0.0f64;
    for row in 0..weak_action.nrows() {
        action_diff = action_diff.max((weak_action[(row, 0)] - tensor_action[(row, 0)]).abs());
    }
    assert!(action_diff < 1e-12);
}

#[test]
fn energy_1d_weak_matches_tensor() {
    let problem = SEM1DProblem::new(
        unit_interval(4, 1),
        2,
        FieldRegistry::new(["u", "T"]),
        DofReduction1D::None,
    );
    let state = state(problem.system_size());
    let direction = direction(problem.system_size());
    let weak = KernelEnergyAdvectionDiffusion1D::new(0.01);
    let tensor = TensorKernelEnergyAdvectionDiffusion1D::new(0.01);

    let weak_residual = problem.assemble_residual(0.0, &weak, state.as_ref());
    let tensor_residual = problem
        .tensor_residual_operator(&tensor)
        .residual(state.as_ref());
    let residual_diff: Vec<f64> = weak_residual
        .iter()
        .zip(&tensor_residual)
        .map(|(a, b)| a - b)
        .collect();
    assert!(max_abs(&residual_diff) < 1e-12);

    let weak_jacobian = problem
        .assemble_residual_jacobian(0.0, &weak, state.as_ref())
        .to_dense();
    let tensor_jacobian = problem
        .tensor_residual_operator(&tensor)
        .assemble_jacobian(state.as_ref())
        .to_dense();
    let mut jacobian_diff = 0.0f64;
    for row in 0..weak_jacobian.nrows() {
        for col in 0..weak_jacobian.ncols() {
            jacobian_diff =
                jacobian_diff.max((weak_jacobian[(row, col)] - tensor_jacobian[(row, col)]).abs());
        }
    }
    assert!(jacobian_diff < 1e-12);

    let weak_action = problem.apply_jacobian(0.0, &weak, state.as_ref(), direction.as_ref());
    let tensor_action = problem
        .tensor_residual_operator(&tensor)
        .apply_jacobian(state.as_ref(), direction.as_ref());
    let mut action_diff = 0.0f64;
    for row in 0..weak_action.nrows() {
        action_diff = action_diff.max((weak_action[(row, 0)] - tensor_action[(row, 0)]).abs());
    }
    assert!(action_diff < 1e-12);
}
