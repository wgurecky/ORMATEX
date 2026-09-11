//! Minimal tensor split-kernel EDAC backward-facing-step example.

use faer::prelude::*;
use ormatex_sem_nd::{
    EdacNavierStokes2DConfig, TensorKernelEdacDirectionalDoNothing2D,
    TensorKernelEdacMomentumConvectionSplit2D, TensorKernelEdacPressureAdvectionSplit2D,
    TensorKernelEdacPressureDiffusion2D, TensorKernelEdacPressureDivergence2D,
    TensorKernelEdacPressureGradient2D, TensorKernelEdacViscousStress2D, TensorResidualKernel,
    TensorResidualKernelSum,
};

#[path = "../support/backward_step_setup.rs"]
mod backward_step_setup;
#[path = "../support/edac.rs"]
mod edac;
#[path = "../support/linear_system.rs"]
mod linear_system;

use backward_step_setup::problem;
use edac::{advance_tensor, write_spatial_csv, TensorFluidSystem};

fn tensor_split_kernel() -> impl TensorResidualKernel<2> {
    let config = EdacNavierStokes2DConfig::new(1.0, 1.0 / 200.0, 4.0, 0.1);
    TensorResidualKernelSum::from_kernel(TensorKernelEdacMomentumConvectionSplit2D::new(config))
        .with(TensorKernelEdacPressureGradient2D::new(config))
        .with(TensorKernelEdacViscousStress2D::new(config))
        .with(TensorKernelEdacPressureDivergence2D::new(config))
        .with(TensorKernelEdacPressureAdvectionSplit2D::new(config))
        .with(TensorKernelEdacPressureDiffusion2D::new(config))
}

fn main() {
    let case = problem();
    let problem = case.problem;
    let state0 = Mat::<f64>::zeros(problem.system_size(), 1);
    let system = TensorFluidSystem::new(&problem, tensor_split_kernel())
        .with_wall_boundaries(case.wall, Vec::new())
        .with_directional_do_nothing_outflow(
            TensorKernelEdacDirectionalDoNothing2D::new(1.0),
            case.outlet,
            true,
        );
    let state = advance_tensor(&system, state0.as_ref(), 0.05, 100);

    std::fs::create_dir_all("target").expect("failed to create output directory");
    write_spatial_csv(
        "target/navier_stokes_backward_step_tensor.csv",
        [
            ("u", problem.field_values("u", state.as_ref()).unwrap()),
            ("v", problem.field_values("v", state.as_ref()).unwrap()),
            ("p", problem.field_values("p", state.as_ref()).unwrap()),
        ],
    );
    println!("tensor backward-step result: target/navier_stokes_backward_step_tensor.csv");
}
