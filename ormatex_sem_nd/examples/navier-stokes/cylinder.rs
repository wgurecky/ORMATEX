//! EDAC/Smagorinsky vortex shedding around a cylinder on a Gmsh quad mesh.

use std::fs::File;
use std::io::{BufWriter, Write};

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

use cylinder_setup::{nearest, problem};
use edac::{epi3, write_spatial_csv, FluidSystem};

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

    let system = if directional {
        FluidSystem::new(&problem, split_kernel())
            .with_wall_boundaries(case.cylinder.clone(), case.slip_wall.clone())
            .with_directional_do_nothing_outflow(
                KernelEdacDirectionalDoNothing2D::new(1.0),
                case.outlet,
                true,
            )
    } else {
        FluidSystem::new(&problem, split_kernel())
            .with_wall_boundaries(case.cylinder, case.slip_wall)
    };

    let dt = 0.005;
    let nsteps = 1000;
    let mut integrator = epi3(state0.as_ref());
    std::fs::create_dir_all("target").expect("failed to create output directory");
    let probe_u = nearest(
        &problem
            .field_values("u", state0.as_ref())
            .unwrap()
            .positions,
        (2.0, 0.5),
    );
    let probe_v = nearest(
        &problem
            .field_values("v", state0.as_ref())
            .unwrap()
            .positions,
        (2.0, 0.5),
    );
    let probe_p = nearest(
        &problem
            .field_values("p", state0.as_ref())
            .unwrap()
            .positions,
        (2.0, 0.5),
    );
    let mut probe = BufWriter::new(
        File::create("target/navier_stokes_cylinder_probe.csv")
            .expect("failed to create probe csv"),
    );
    writeln!(probe, "t,u,v,p").unwrap();
    for step in 0..nsteps {
        let result = integrator
            .step(&system, dt)
            .unwrap_or_else(|error| panic!("EDAC step {step} failed: {}", error.msg));
        integrator.accept_step(result);
        let state = integrator.state();
        let u = problem.field_values("u", state.as_ref()).unwrap();
        let v = problem.field_values("v", state.as_ref()).unwrap();
        let p = problem.field_values("p", state.as_ref()).unwrap();
        writeln!(
            probe,
            "{:.8},{:.9e},{:.9e},{:.9e}",
            integrator.time(),
            u.values[probe_u],
            v.values[probe_v],
            p.values[probe_p],
        )
        .unwrap();
    }
    let state = integrator.state();
    let final_time = integrator.time();

    write_spatial_csv(
        "target/navier_stokes_cylinder.csv",
        [
            ("u", problem.field_values("u", state.as_ref()).unwrap()),
            ("v", problem.field_values("v", state.as_ref()).unwrap()),
            ("p", problem.field_values("p", state.as_ref()).unwrap()),
        ],
    );
    println!("cylinder final state (t={final_time:.6}): target/navier_stokes_cylinder.csv");
}
