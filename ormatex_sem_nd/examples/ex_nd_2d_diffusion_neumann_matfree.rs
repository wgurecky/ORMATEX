//! Matrix-free Jacobian version of the Neumann/Robin diffusion case.

use faer::prelude::*;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{shapes::unit_square, SingleElementMesh};
#[path = "support/diffusion_neumann.rs"]
mod diffusion_neumann;
#[path = "support/linear_system.rs"]
mod linear_system;
#[path = "support/matrix_free.rs"]
mod matrix_free;
use diffusion_neumann::{diffusion_neumann_problem, unit_square_neumann_robin_boundary};
use linear_system::implicit_euler_final_state;
use matrix_free::{JacobianBackend, ResidualDiffusionNeumannSys};

type QuadMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

fn main() {
    let mesh: QuadMesh = unit_square(32, 2, ReferenceCellType::Quadrilateral);
    let (problem, mass, kernel) =
        diffusion_neumann_problem(mesh, 2, ormatex_sem_nd::MeshMetadata::default());
    let boundary = unit_square_neumann_robin_boundary(&problem);
    let system = ResidualDiffusionNeumannSys::new(
        &problem,
        mass,
        kernel,
        boundary.mat,
        boundary.rhs,
        JacobianBackend::MatrixFree,
    );

    let n = problem.reduced_size();
    let y0 = Mat::<f64>::zeros(n, 1);
    let y = implicit_euler_final_state(&system, y0.as_ref(), 1.0, 200, 1e-10);

    let positions = problem.dof_positions();
    let mut max_error = 0.0_f64;
    for row in 0..n {
        let expected = -10.0 * positions[row].0 + 20.0;
        max_error = max_error.max((y[(row, 0)] - expected).abs());
    }
    assert!(
        max_error < 5e-3,
        "steady-state error {max_error} exceeds tolerance"
    );
    println!(
        "matrix-free diffusion-neumann complete; max |T| = {:.3e}",
        y.norm_max()
    );
}
