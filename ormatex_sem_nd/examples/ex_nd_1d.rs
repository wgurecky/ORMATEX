//! 1D GLL spectral-element advection-diffusion on an interval mesh.

use std::fs::File;
use std::io::Write;

use faer::prelude::*;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap};
use ndmesh::{shapes::unit_interval, SingleElementMesh};
use ormatex::ode_implicit::DirkIntegrator;
use ormatex::ode_sys::IntegrateSys;
use ormatex::tableau_implicit::ImplicitBT;
use ormatex_sem_nd::{
    DofReduction1D, KernelAdvDiff, KernelAdvDiffSUPG, KernelVolumeSource, SEM1DProblem,
};

#[path = "support/linear_system.rs"]
mod linear_system;
use linear_system::AdvDiffSys;

fn main() {
    let nx = 64;
    let p = 2;
    let nu = 0.001;
    let vel = 0.5;
    let sigma = 0.05;
    let x0 = 0.5;
    let dt = 0.01;
    let nsteps = 200;
    let snapshot_every = 20;

    let mesh: SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>> = unit_interval(nx);
    let problem = SEM1DProblem::new(mesh, p, DofReduction1D::Periodic { facets: [0, nx] });
    let mass = problem.assemble_lumped_mass();
    let operator = problem.assemble_bilinear(&KernelAdvDiff::new(nu, vel));
    let n = mass.nrows();
    println!(
        "reduced ndofs = {}, mass nnz = {}, adv_diff nnz = {}",
        n,
        mass.compute_nnz(),
        operator.compute_nnz()
    );

    let plain = operator.to_dense();
    let supg_zero = problem
        .assemble_bilinear(&KernelAdvDiffSUPG::new(nu, vel, 0.0))
        .to_dense();
    for i in 0..n {
        for j in 0..n {
            assert!((plain[(i, j)] - supg_zero[(i, j)]).abs() < 1e-12);
        }
    }
    assert!(
        (problem
            .assemble_linear(&KernelVolumeSource::new(1.0))
            .iter()
            .sum::<f64>()
            - 1.0)
            .abs()
            < 1e-12
    );

    let system = AdvDiffSys::new(mass, operator);
    let ones = Mat::<f64>::from_fn(n, 1, |_, _| 1.0);
    assert!(
        (0..n)
            .map(|i| system.apply_minv_k(ones.as_ref())[(i, 0)].abs())
            .fold(0.0_f64, f64::max)
            < 1e-12
    );

    let xs = problem.dof_positions();
    let mut y0 = Mat::<f64>::zeros(n, 1);
    for i in 0..n {
        y0[(i, 0)] = periodic_gaussian(xs[i], x0, sigma);
    }
    let mut solver =
        DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::implicit_euler(), 1e-12, 1e-12);
    let mut snapshots = vec![(0.0, (0..n).map(|i| y0[(i, 0)]).collect::<Vec<_>>())];
    for step in 1..=nsteps {
        let result = solver.step(&system, dt).unwrap();
        if step % snapshot_every == 0 || step == nsteps {
            snapshots.push((result.t, (0..n).map(|i| result.y[(i, 0)]).collect()));
        }
        solver.accept_step(result);
    }

    let out_path = "target/ex_nd_1d_out.csv";
    let mut output = File::create(out_path).expect("failed to create output csv");
    writeln!(output, "t,x,u").unwrap();
    for (t, profile) in snapshots {
        for (i, u) in profile.iter().enumerate() {
            writeln!(output, "{t:.6},{:.6},{u:.9e}", xs[i]).unwrap();
        }
    }
}

fn periodic_gaussian(x: f64, x0: f64, sigma: f64) -> f64 {
    let mut dx = x - x0;
    if dx > 0.5 {
        dx -= 1.0;
    } else if dx < -0.5 {
        dx += 1.0;
    }
    (-(dx * dx) / (2.0 * sigma * sigma)).exp()
}
