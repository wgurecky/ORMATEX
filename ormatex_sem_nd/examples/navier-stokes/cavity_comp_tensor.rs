//! Tensor composed-kernel lid-driven cavity example.

use faer::prelude::*;
use ormatex_sem_nd::{
    EdacNavierStokes2DConfig, TensorKernelEdacMomentumConvection2D,
    TensorKernelEdacPressureAdvection2D, TensorKernelEdacPressureDiffusion2D,
    TensorKernelEdacPressureDivergence2D, TensorKernelEdacPressureGradient2D,
    TensorKernelEdacViscousStress2D, TensorResidualKernelSum,
};

#[path = "../support/cavity_setup.rs"]
mod cavity_setup;
#[path = "../support/edac.rs"]
mod edac;
#[path = "../support/linear_system.rs"]
mod linear_system;

use edac::{advance_tensor, write_spatial_csv, JacobianBackend, TensorFluidSystem};

const STEPS: usize = 300;

fn tensor_composed_kernel() -> impl ormatex_sem_nd::TensorResidualKernel<2> {
    let config = EdacNavierStokes2DConfig::new(1.0, 0.1, 10.0, 0.0);
    TensorResidualKernelSum::from_kernel(TensorKernelEdacMomentumConvection2D::new(config))
        .with(TensorKernelEdacPressureGradient2D::new(config))
        .with(TensorKernelEdacViscousStress2D::new(config))
        .with(TensorKernelEdacPressureDivergence2D::new(config))
        .with(TensorKernelEdacPressureAdvection2D::new(config))
        .with(TensorKernelEdacPressureDiffusion2D::new(config))
}

fn main() {
    let problem = cavity_setup::problem();
    let state0 = Mat::<f64>::zeros(problem.system_size(), 1);
    let system = TensorFluidSystem::new_with_backend(
        &problem,
        tensor_composed_kernel(),
        JacobianBackend::MatrixFree,
    );
    let state = advance_tensor(&system, state0.as_ref(), 0.01, STEPS);

    std::fs::create_dir_all("target").expect("failed to create output directory");
    write_spatial_csv(
        "target/navier_stokes_lid_driven_cavity_comp_tensor.csv",
        [
            ("u", problem.field_values("u", state.as_ref()).unwrap()),
            ("v", problem.field_values("v", state.as_ref()).unwrap()),
            ("p", problem.field_values("p", state.as_ref()).unwrap()),
        ],
    );
    println!("tensor cavity result: target/navier_stokes_lid_driven_cavity_comp_tensor.csv");
}
