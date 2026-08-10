//! 2D GLL spectral-element advection-diffusion on a periodic quadrilateral mesh.

use std::fs::File;
use std::io::Write;

use faer::prelude::*;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{shapes::unit_square, SingleElementMesh};
use ormatex::ode_implicit::DirkIntegrator;
use ormatex::ode_sys::IntegrateSys;
use ormatex::tableau_implicit::ImplicitBT;
use ormatex_sem_nd::{DofReduction2D, SEM2DProblem, KernelAdvDiff2D, KernelVolumeSource};

#[path = "support/linear_system.rs"]
mod linear_system;
use linear_system::AdvDiffSys;

fn main() {
    let mesh: SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>> =
        unit_square(32, 2, ReferenceCellType::Quadrilateral);
    let problem = SEM2DProblem::new(mesh, 2, DofReduction2D::Periodic);
    let mass = problem.assemble_lumped_mass();
    let operator = problem.assemble_bilinear(&KernelAdvDiff2D::new(0.001, [0.5, 0.1]));
    let n = mass.nrows();
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
    let positions = problem.dof_positions();
    let mut y0 = Mat::<f64>::zeros(n, 1);
    for i in 0..n {
        y0[(i, 0)] = periodic_gaussian(positions[i].0, positions[i].1, 0.5, 0.5, 0.1);
    }
    let mut solver =
        DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::implicit_euler(), 1e-12, 1e-12);
    let mut y = y0;
    for _ in 0..200 {
        let result = solver.step(&system, 0.01).unwrap();
        y = result.y.clone();
        solver.accept_step(result);
    }

    let mut output = File::create("target/ex_nd_2d_quad_out.csv").unwrap();
    writeln!(output, "x,y,u").unwrap();
    for i in 0..n {
        writeln!(
            output,
            "{:.6},{:.6},{:.9e}",
            positions[i].0,
            positions[i].1,
            y[(i, 0)]
        )
        .unwrap();
    }
}

fn periodic_gaussian(x: f64, y: f64, x0: f64, y0: f64, sigma: f64) -> f64 {
    let distance = |value: f64, center: f64| {
        let mut delta = value - center;
        if delta > 0.5 {
            delta -= 1.0;
        } else if delta < -0.5 {
            delta += 1.0;
        }
        delta
    };
    let dx = distance(x, x0);
    let dy = distance(y, y0);
    (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp()
}
