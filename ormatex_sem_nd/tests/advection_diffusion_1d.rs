use faer::prelude::*;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap};
use ndmesh::{shapes::unit_interval, SingleElementMesh};
use ormatex::matexp_krylov::KrylovExpm;
use ormatex::matexp_pade::PadeExpm;
use ormatex::ode_epirk::EpirkIntegrator;
use ormatex::ode_sys::IntegrateSys;
use ormatex_sem_nd::{DofReduction1D, KernelAdvDiff, SEM1DProblem};

#[path = "../examples/support/linear_system.rs"]
mod linear_system;
use linear_system::LinearOdeSys;

fn run_epi3_krylov(system: &LinearOdeSys, y0: MatRef<'_, f64>, dt: f64, nsteps: usize) -> Mat<f64> {
    let expmv = Box::new(PadeExpm::new(12));
    let krylov = KrylovExpm::new(expmv, 30, 100, 1e-12, Some(2));
    let mut solver = EpirkIntegrator::new(0.0, y0, "epi3".to_string(), krylov);
    for _ in 0..nsteps {
        let step = solver.step(system, dt).unwrap();
        solver.accept_step(step);
    }
    solver.state()
}

fn periodic_gaussian(x: f64, center: f64, sigma: f64) -> f64 {
    (-4..=4)
        .map(|image| {
            let dx = x - center - image as f64;
            (-(dx * dx) / (2.0 * sigma * sigma)).exp()
        })
        .sum()
}

#[test]
fn p2_epi3_krylov_advects_and_diffuses_periodic_gaussian() {
    let nx = 64;
    let velocity = 0.5;
    let diffusivity = 0.001;
    let sigma0 = 0.1;
    let x0 = 0.25;
    let final_time = 0.5;
    let dt = 0.01;
    let nsteps = (final_time / dt) as usize;

    let problem = SEM1DProblem::new(
        unit_interval(nx),
        2,
        DofReduction1D::Periodic { facets: [0, nx] },
    );
    let mass = problem.assemble_lumped_mass();
    let operator = problem.assemble_bilinear(&KernelAdvDiff::new(diffusivity, velocity));
    let n = problem.reduced_size();
    let positions = problem.dof_positions();
    let y0 = Mat::from_fn(n, 1, |row, _| periodic_gaussian(positions[row], x0, sigma0));
    let system = LinearOdeSys::new(mass, operator, vec![0.0; n]);
    let state = run_epi3_krylov(&system, y0.as_ref(), dt, nsteps);

    let sigma = (sigma0 * sigma0 + 2.0 * diffusivity * final_time).sqrt();
    let center = x0 + velocity * final_time;
    let amplitude = sigma0 / sigma;
    let max_error = (0..n)
        .map(|row| {
            let expected = amplitude * periodic_gaussian(positions[row], center, sigma);
            (state[(row, 0)] - expected).abs()
        })
        .fold(0.0_f64, f64::max);
    assert!(
        max_error < 2e-4,
        "maximum periodic Gaussian error: {max_error}"
    );
}
