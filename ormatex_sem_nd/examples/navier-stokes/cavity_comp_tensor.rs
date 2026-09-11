//! Tensor composed-kernel lid-driven cavity example.

use std::time::Instant;

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

use edac::{advance_tensor, advance_tensor_leja, write_spatial_csv, TensorFluidSystem};

const DT: f64 = 0.0025;
const DEFAULT_STEPS: usize = 1200;
const DEFAULT_RESOLUTION: usize = 32;

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
    let args: Vec<String> = std::env::args().collect();
    let resolution = parse_usize_flag(&args, "--resolution").unwrap_or(DEFAULT_RESOLUTION);
    let steps = parse_usize_flag(&args, "--steps").unwrap_or(DEFAULT_STEPS);
    let threads = parse_usize_flag(&args, "--threads");
    let benchmark = args.iter().any(|arg| arg == "--benchmark");
    if let Some(threads) = threads {
        assert!(threads > 0, "--threads must be positive");
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .expect("failed to configure Rayon thread pool");
        faer::set_global_parallelism(faer::Par::rayon(threads));
    }

    let setup_start = Instant::now();
    let problem = cavity_setup::problem_with_resolution(resolution, resolution);
    let state0 = Mat::<f64>::zeros(problem.system_size(), 1);
    let system = TensorFluidSystem::new(&problem, tensor_composed_kernel());
    let setup_time = setup_start.elapsed();
    let integration_start = Instant::now();
    let state = if std::env::args().any(|arg| arg == "--leja") {
        advance_tensor_leja(&system, state0.as_ref(), DT, steps)
    } else {
        advance_tensor(&system, state0.as_ref(), DT, steps)
    };
    let integration_time = integration_start.elapsed();

    println!(
        "cavity tensor: resolution={resolution} steps={steps} dofs={} setup={setup_time:?} integration={integration_time:?} threads={:?} backend={:?}",
        problem.system_size(),
        threads,
        if args.iter().any(|arg| arg == "--assembled-jacobian") {
            "assembled"
        } else {
            "matrix-free"
        },
    );
    if benchmark {
        return;
    }

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

fn parse_usize_flag(args: &[String], name: &str) -> Option<usize> {
    args.windows(2)
        .find(|window| window[0] == name)
        .map(|window| {
            window[1]
                .parse()
                .expect("invalid numeric command-line flag")
        })
}
