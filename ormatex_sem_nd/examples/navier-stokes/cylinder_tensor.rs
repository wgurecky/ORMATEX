//! Tensor split-kernel EDAC cylinder example on the Gmsh quad mesh.

use std::fs::File;
use std::io::{BufWriter, Write};

use faer::prelude::*;
use ormatex::ode_sys::IntegrateSys;
use ormatex_sem_nd::{
    EdacNavierStokes2DConfig, TensorKernelEdacDirectionalDoNothing2D,
    TensorKernelEdacMomentumConvectionSplit2D, TensorKernelEdacPressureAdvectionSplit2D,
    TensorKernelEdacPressureDiffusion2D, TensorKernelEdacPressureDivergence2D,
    TensorKernelEdacPressureGradient2D, TensorKernelEdacViscousStress2D, TensorResidualKernel,
    TensorResidualKernelSum,
};

#[path = "../support/cylinder_setup.rs"]
mod cylinder_setup;
#[path = "../support/edac.rs"]
mod edac;
#[path = "../support/linear_system.rs"]
mod linear_system;

use cylinder_setup::{nearest, problem};
use edac::{epi3, write_spatial_csv, JacobianBackend, TensorFluidSystem};

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
    let directional = std::env::args().any(|arg| arg == "--directional");
    let case = problem(directional);
    let problem = case.problem;
    let state0 = Mat::<f64>::zeros(problem.system_size(), 1);

    let system = TensorFluidSystem::new_with_backend(
        &problem,
        tensor_split_kernel(),
        JacobianBackend::MatrixFree,
    )
    .with_wall_boundaries(case.cylinder, case.slip_wall);
    let system = if directional {
        system.with_directional_do_nothing_outflow(
            TensorKernelEdacDirectionalDoNothing2D::new(1.0),
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
        File::create("target/navier_stokes_cylinder_tensor_probe.csv")
            .expect("failed to create tensor cylinder probe csv"),
    );
    writeln!(probe, "u,v,p").unwrap();
    writeln!(
        probe,
        "{:.9e},{:.9e},{:.9e}",
        state[(problem.field_offset(0) + probe_u, 0)],
        state[(problem.field_offset(1) + probe_v, 0)],
        state[(problem.field_offset(2) + probe_p, 0)],
    )
    .unwrap();
    write_spatial_csv(
        "target/navier_stokes_cylinder_tensor.csv",
        [
            ("u", problem.field_values("u", state.as_ref()).unwrap()),
            ("v", problem.field_values("v", state.as_ref()).unwrap()),
            ("p", problem.field_values("p", state.as_ref()).unwrap()),
        ],
    );
    println!(
        "tensor cylinder result (directional={directional}): target/navier_stokes_cylinder_tensor.csv"
    );
}
