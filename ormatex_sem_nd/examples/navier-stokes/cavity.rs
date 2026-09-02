//! Low-Reynolds-number lid-driven cavity validation for the EDAC solver.

use faer::prelude::*;
use ormatex_sem_nd::KernelEdacNavierStokes2D;

#[path = "../support/edac.rs"]
mod edac;
#[path = "../support/linear_system.rs"]
mod linear_system;
use edac::{advance, write_spatial_csv, FluidSystem};

fn main() {
    let problem = cavity_setup::problem();
    let kernel = KernelEdacNavierStokes2D::new(1.0, 0.1, 10.0, 0.0);
    let system = FluidSystem::new(&problem, kernel);
    let state0 = Mat::<f64>::zeros(problem.system_size(), 1);
    let state = advance(&system, state0.as_ref(), 0.01, 300);
    std::fs::create_dir_all("target").expect("failed to create output directory");

    let u = problem.field_values("u", state.as_ref()).unwrap();
    let v = problem.field_values("v", state.as_ref()).unwrap();
    let p = problem.field_values("p", state.as_ref()).unwrap();
    let mut min_u = f64::INFINITY;
    let mut max_abs_v: f64 = 0.0;
    for (&(x, y), &value) in u.positions.iter().zip(&u.values) {
        assert!(value.is_finite(), "non-finite cavity state");
        if x > 0.1 && x < 0.9 && y > 0.1 && y < 0.9 {
            min_u = min_u.min(value);
        }
    }
    for (&(x, y), &value) in v.positions.iter().zip(&v.values) {
        assert!(value.is_finite(), "non-finite cavity state");
        if x > 0.1 && x < 0.9 && y > 0.1 && y < 0.9 {
            max_abs_v = max_abs_v.max(value.abs());
        }
    }
    for &value in &p.values {
        assert!(value.is_finite(), "non-finite cavity state");
    }
    write_spatial_csv(
        "target/navier_stokes_lid_driven_cavity.csv",
        [("u", u), ("v", v), ("p", p)],
    );
    assert!(
        min_u < -1e-3,
        "cavity has no recirculation: min interior u={min_u}"
    );
    assert!(max_abs_v > 1e-3, "cavity has no transverse motion");
    println!("cavity validation: min interior u={min_u:.3e}, max |v|={max_abs_v:.3e}");
}

#[path = "../support/cavity_setup.rs"]
mod cavity_setup;
