//! Generic composed-kernel lid-driven cavity example.

use faer::prelude::*;
use ormatex_sem_nd::{
    EdacNavierStokes2DConfig, KernelEdacMomentumConvection2D, KernelEdacPressureAdvection2D,
    KernelEdacPressureDiffusion2D, KernelEdacPressureDivergence2D, KernelEdacPressureGradient2D,
    KernelEdacViscousStress2D, ResidualKernelSum,
};

#[path = "../support/cavity_setup.rs"]
mod cavity_setup;
#[path = "../support/edac.rs"]
mod edac;
#[path = "../support/linear_system.rs"]
mod linear_system;

use edac::{advance, write_spatial_csv, FluidSystem, GenericResidual};

const STEPS: usize = 300;

fn composed_kernel() -> ResidualKernelSum<'static> {
    let config = EdacNavierStokes2DConfig::new(1.0, 0.1, 10.0, 0.0);
    ResidualKernelSum::from_kernel(KernelEdacMomentumConvection2D::new(config))
        .with(KernelEdacPressureGradient2D::new(config))
        .with(KernelEdacViscousStress2D::new(config))
        .with(KernelEdacPressureDivergence2D::new(config))
        .with(KernelEdacPressureAdvection2D::new(config))
        .with(KernelEdacPressureDiffusion2D::new(config))
}

fn main() {
    let problem = cavity_setup::problem();
    let state0 = Mat::<f64>::zeros(problem.system_size(), 1);
    let system = FluidSystem::new(&problem, GenericResidual(composed_kernel()));
    let state = advance(&system, state0.as_ref(), 0.01, STEPS);

    std::fs::create_dir_all("target").expect("failed to create output directory");
    write_spatial_csv(
        "target/navier_stokes_lid_driven_cavity_comp_generic.csv",
        [
            ("u", problem.field_values("u", state.as_ref()).unwrap()),
            ("v", problem.field_values("v", state.as_ref()).unwrap()),
            ("p", problem.field_values("p", state.as_ref()).unwrap()),
        ],
    );
    println!("generic cavity result: target/navier_stokes_lid_driven_cavity_comp_generic.csv");
}
