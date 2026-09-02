//! Lid-driven cavity comparison between fused and composed EDAC kernels.

use std::time::Instant;

use faer::prelude::*;
use ormatex_sem_nd::{
    EdacNavierStokes2DConfig, KernelEdacMomentumConvection2D, KernelEdacMomentumConvectionSplit2D,
    KernelEdacNavierStokes2D, KernelEdacPressureAdvection2D, KernelEdacPressureAdvectionSplit2D,
    KernelEdacPressureDiffusion2D, KernelEdacPressureDivergence2D, KernelEdacPressureGradient2D,
    KernelEdacViscousStress2D, QuadMesh, ResidualKernelSum, SEM2DProblem,
};

#[path = "../support/edac.rs"]
mod edac;
#[path = "../support/linear_system.rs"]
mod linear_system;
use edac::{advance, write_spatial_csv, FluidSystem};

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

fn split_composed_kernel() -> ResidualKernelSum<'static> {
    let config = EdacNavierStokes2DConfig::new(1.0, 0.1, 10.0, 0.0);
    ResidualKernelSum::from_kernel(KernelEdacMomentumConvectionSplit2D::new(config))
        .with(KernelEdacPressureGradient2D::new(config))
        .with(KernelEdacViscousStress2D::new(config))
        .with(KernelEdacPressureDivergence2D::new(config))
        .with(KernelEdacPressureAdvectionSplit2D::new(config))
        .with(KernelEdacPressureDiffusion2D::new(config))
}

fn max_state_difference(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> f64 {
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.ncols(), b.ncols());
    (0..a.nrows())
        .flat_map(|row| (0..a.ncols()).map(move |col| (a[(row, col)] - b[(row, col)]).abs()))
        .fold(0.0, f64::max)
}

fn write_state(problem: &SEM2DProblem<QuadMesh>, state: MatRef<'_, f64>) {
    std::fs::create_dir_all("target").expect("failed to create output directory");
    let u = problem.field_values("u", state).unwrap();
    let v = problem.field_values("v", state).unwrap();
    let p = problem.field_values("p", state).unwrap();
    write_spatial_csv(
        "target/navier_stokes_lid_driven_cavity_comp.csv",
        [("u", u), ("v", v), ("p", p)],
    );
}

fn main() {
    let problem = cavity_setup::problem();
    let state0 = Mat::<f64>::zeros(problem.system_size(), 1);
    let split = std::env::args().any(|arg| arg == "--split");

    let fused_system =
        FluidSystem::new(&problem, KernelEdacNavierStokes2D::new(1.0, 0.1, 10.0, 0.0));
    let start = Instant::now();
    let fused_state = advance(&fused_system, state0.as_ref(), 0.01, STEPS);
    let fused_runtime = start.elapsed();

    let mut composed_system = FluidSystem::new(
        &problem,
        if split {
            split_composed_kernel()
        } else {
            composed_kernel()
        },
    );
    if split {
        composed_system = composed_system.with_split_boundary();
    }
    let start = Instant::now();
    let composed_state = advance(&composed_system, state0.as_ref(), 0.01, STEPS);
    let composed_runtime = start.elapsed();

    let max_difference = max_state_difference(fused_state.as_ref(), composed_state.as_ref());
    if !split {
        assert!(
            max_difference < 1e-10,
            "composed EDAC state differs from fused state: {max_difference:e}"
        );
    }
    write_state(&problem, composed_state.as_ref());

    let ratio = composed_runtime.as_secs_f64() / fused_runtime.as_secs_f64();
    assert!(
        ratio < 2.0,
        "composed cavity runtime is more than 2x fused runtime: ratio={ratio:.3}"
    );
    println!(
        "cavity comparison (split={split}): max |fused-composed|={max_difference:.3e}, fused={fused_runtime:?}, composed={composed_runtime:?}, ratio={ratio:.3}"
    );
    println!("composed result: target/navier_stokes_lid_driven_cavity_comp.csv");
}

#[path = "../support/cavity_setup.rs"]
mod cavity_setup;
