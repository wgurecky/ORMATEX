//! Generic split-kernel EDAC cylinder example on the Gmsh quad mesh.

use faer::prelude::*;
use ormatex::ode_sys::IntegrateSys;
use ormatex_sem_nd::{
    EdacNavierStokes2DConfig, KernelEdacDirectionalDoNothing2D,
    KernelEdacMomentumConvectionSplit2D, KernelEdacPressureAdvectionSplit2D,
    KernelEdacPressureDiffusion2D, KernelEdacPressureDivergence2D, KernelEdacPressureGradient2D,
    KernelEdacViscousStress2D, ResidualKernelSum,
};

#[path = "../support/cylinder_setup.rs"]
mod cylinder_setup;
#[path = "../support/edac.rs"]
mod edac;
#[path = "../support/linear_system.rs"]
mod linear_system;

use cylinder_setup::problem;
use edac::{epi3, write_spatial_csv, FluidSystem, GenericResidual};

fn split_kernel() -> ResidualKernelSum<'static> {
    let config = EdacNavierStokes2DConfig::new(1.0, 1.0 / 200.0, 4.0, 0.1);
    ResidualKernelSum::from_kernel(KernelEdacMomentumConvectionSplit2D::new(config))
        .with(KernelEdacPressureGradient2D::new(config))
        .with(KernelEdacViscousStress2D::new(config))
        .with(KernelEdacPressureDivergence2D::new(config))
        .with(KernelEdacPressureAdvectionSplit2D::new(config))
        .with(KernelEdacPressureDiffusion2D::new(config))
}

fn main() {
    let directional = std::env::args().any(|arg| arg == "--directional");
    let case = problem(directional);
    let problem = case.problem;
    let state0 = Mat::<f64>::zeros(problem.system_size(), 1);

    let system = FluidSystem::new(&problem, GenericResidual(split_kernel()))
        .with_wall_boundaries(case.cylinder, case.slip_wall);
    let system = if directional {
        system.with_directional_do_nothing_outflow(
            KernelEdacDirectionalDoNothing2D::new(1.0),
            case.outlet,
            true,
        )
    } else {
        system
    };

    let mut integrator = epi3(state0.as_ref());
    let dt = 0.05;
    let nsteps = 100;
    for step in 0..nsteps {
        let result = integrator
            .step(&system, dt)
            .unwrap_or_else(|error| panic!("EDAC step {step} failed: {}", error.msg));
        integrator.accept_step(result);
    }
    let state = integrator.state();

    std::fs::create_dir_all("target").expect("failed to create output directory");
    write_spatial_csv(
        "target/navier_stokes_cylinder_generic.csv",
        [
            ("u", problem.field_values("u", state.as_ref()).unwrap()),
            ("v", problem.field_values("v", state.as_ref()).unwrap()),
            ("p", problem.field_values("p", state.as_ref()).unwrap()),
        ],
    );
    println!(
        "generic cylinder result (directional={directional}): target/navier_stokes_cylinder_generic.csv"
    );
}
