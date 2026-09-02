use faer::prelude::*;
use ormatex::ode_sys::{IntegrateSys, OdeSys};
use ormatex_sem_nd::{
    EdacNavierStokes2DConfig, KernelEdacMomentumConvection2D, KernelEdacMomentumConvectionSplit2D,
    KernelEdacPressureAdvection2D, KernelEdacPressureAdvectionSplit2D,
    KernelEdacPressureDiffusion2D, KernelEdacPressureDivergence2D, KernelEdacPressureGradient2D,
    KernelEdacViscousStress2D, ResidualKernelSum, TensorKernelEdacMomentumConvection2D,
    TensorKernelEdacMomentumConvectionSplit2D, TensorKernelEdacPressureAdvection2D,
    TensorKernelEdacPressureAdvectionSplit2D, TensorKernelEdacPressureDiffusion2D,
    TensorKernelEdacPressureDivergence2D, TensorKernelEdacPressureGradient2D,
    TensorKernelEdacViscousStress2D, TensorResidualKernel, TensorResidualKernelSum,
};

#[path = "../examples/support/cavity_setup.rs"]
mod cavity_setup;
#[path = "../examples/support/cylinder_setup.rs"]
mod cylinder_setup;
#[path = "../examples/support/edac.rs"]
mod edac;
#[path = "../examples/support/linear_system.rs"]
mod linear_system;

use edac::{
    advance, advance_tensor, FluidSystem, GenericResidual, JacobianBackend, TensorFluidSystem,
};

fn cavity_generic_kernel() -> ResidualKernelSum<'static> {
    let config = EdacNavierStokes2DConfig::new(1.0, 0.1, 10.0, 0.0);
    ResidualKernelSum::from_kernel(KernelEdacMomentumConvection2D::new(config))
        .with(KernelEdacPressureGradient2D::new(config))
        .with(KernelEdacViscousStress2D::new(config))
        .with(KernelEdacPressureDivergence2D::new(config))
        .with(KernelEdacPressureAdvection2D::new(config))
        .with(KernelEdacPressureDiffusion2D::new(config))
}

fn cavity_tensor_kernel() -> impl TensorResidualKernel<2> {
    let config = EdacNavierStokes2DConfig::new(1.0, 0.1, 10.0, 0.0);
    TensorResidualKernelSum::from_kernel(TensorKernelEdacMomentumConvection2D::new(config))
        .with(TensorKernelEdacPressureGradient2D::new(config))
        .with(TensorKernelEdacViscousStress2D::new(config))
        .with(TensorKernelEdacPressureDivergence2D::new(config))
        .with(TensorKernelEdacPressureAdvection2D::new(config))
        .with(TensorKernelEdacPressureDiffusion2D::new(config))
}

fn cylinder_generic_kernel() -> ResidualKernelSum<'static> {
    let config = EdacNavierStokes2DConfig::new(1.0, 1.0 / 200.0, 4.0, 0.1);
    ResidualKernelSum::from_kernel(KernelEdacMomentumConvectionSplit2D::new(config))
        .with(KernelEdacPressureGradient2D::new(config))
        .with(KernelEdacViscousStress2D::new(config))
        .with(KernelEdacPressureDivergence2D::new(config))
        .with(KernelEdacPressureAdvectionSplit2D::new(config))
        .with(KernelEdacPressureDiffusion2D::new(config))
}

fn cylinder_tensor_kernel() -> impl TensorResidualKernel<2> {
    let config = EdacNavierStokes2DConfig::new(1.0, 1.0 / 200.0, 4.0, 0.1);
    TensorResidualKernelSum::from_kernel(TensorKernelEdacMomentumConvectionSplit2D::new(config))
        .with(TensorKernelEdacPressureGradient2D::new(config))
        .with(TensorKernelEdacViscousStress2D::new(config))
        .with(TensorKernelEdacPressureDivergence2D::new(config))
        .with(TensorKernelEdacPressureAdvectionSplit2D::new(config))
        .with(TensorKernelEdacPressureDiffusion2D::new(config))
}

fn max_difference(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> f64 {
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.ncols(), b.ncols());
    (0..a.nrows())
        .flat_map(|row| (0..a.ncols()).map(move |col| (a[(row, col)] - b[(row, col)]).abs()))
        .fold(0.0, f64::max)
}

fn advance_cylinder<'a, S>(
    system: &'a S,
    state0: MatRef<'_, f64>,
    dt: f64,
    nsteps: usize,
) -> Mat<f64>
where
    S: OdeSys<'a>,
{
    let mut integrator = edac::epi3(state0);
    for step in 0..nsteps {
        let result = integrator
            .step(system, dt)
            .unwrap_or_else(|error| panic!("EDAC step {step} failed: {}", error.msg));
        integrator.accept_step(result);
    }
    integrator.state()
}

#[test]
fn cavity_tensor_and_generic_composed_examples_match() {
    let problem = cavity_setup::problem();
    let state0 = Mat::<f64>::zeros(problem.system_size(), 1);
    let generic = FluidSystem::new_with_backend(
        &problem,
        GenericResidual(cavity_generic_kernel()),
        JacobianBackend::MatrixFree,
    );
    let tensor = TensorFluidSystem::new_with_backend(
        &problem,
        cavity_tensor_kernel(),
        JacobianBackend::MatrixFree,
    );
    let generic_state = advance(&generic, state0.as_ref(), 0.01, 2);
    let tensor_state = advance_tensor(&tensor, state0.as_ref(), 0.01, 2);

    assert!(max_difference(generic_state.as_ref(), tensor_state.as_ref()) < 1e-10);
}

#[test]
fn cylinder_tensor_and_generic_examples_match() {
    let generic_case = cylinder_setup::problem(false);
    let tensor_case = cylinder_setup::problem(false);
    let generic_problem = generic_case.problem;
    let tensor_problem = tensor_case.problem;
    let state0 = Mat::<f64>::zeros(generic_problem.system_size(), 1);
    let generic = FluidSystem::new_with_backend(
        &generic_problem,
        GenericResidual(cylinder_generic_kernel()),
        JacobianBackend::MatrixFree,
    )
    .with_wall_boundaries(generic_case.cylinder, generic_case.slip_wall);
    let tensor = TensorFluidSystem::new_with_backend(
        &tensor_problem,
        cylinder_tensor_kernel(),
        JacobianBackend::MatrixFree,
    )
    .with_wall_boundaries(tensor_case.cylinder, tensor_case.slip_wall);
    let generic_state = advance_cylinder(&generic, state0.as_ref(), 0.05, 2);
    let tensor_state = advance_cylinder(&tensor, state0.as_ref(), 0.05, 2);

    assert!(max_difference(generic_state.as_ref(), tensor_state.as_ref()) < 1e-9);
}
