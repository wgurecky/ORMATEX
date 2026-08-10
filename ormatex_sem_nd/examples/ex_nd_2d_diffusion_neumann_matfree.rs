//! Matrix-free Jacobian version of the Neumann/Robin diffusion case.

use faer::prelude::*;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{shapes::unit_square, SingleElementMesh};
use ormatex::ode_implicit::DirkIntegrator;
use ormatex::ode_sys::IntegrateSys;
use ormatex::tableau_implicit::ImplicitBT;
use ormatex_sem_nd::{
    DofReduction2D, SEM2DProblem, KernelAdvDiff2D, NeumannFlux, RobinConvection,
};

#[path = "support/matrix_free.rs"]
mod matrix_free;
use matrix_free::{JacobianBackend, ResidualDiffusionNeumannSys};

type QuadMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

fn main() {
    let mesh: QuadMesh = unit_square(32, 2, ReferenceCellType::Quadrilateral);
    let problem = SEM2DProblem::new(mesh, 2, DofReduction2D::None);
    let mass = problem.assemble_lumped_mass();
    let neumann = NeumannFlux::new(1.0);
    let robin = RobinConvection::new(0.1, 0.0);
    let boundary = problem.assemble_boundary(|facet| {
        const EPS: f64 = 1e-9;
        if facet.midpoint[0] < EPS {
            Some(&neumann)
        } else if facet.midpoint[0] > 1.0 - EPS {
            Some(&robin)
        } else {
            None
        }
    });
    let system = ResidualDiffusionNeumannSys::new(
        &problem,
        mass,
        KernelAdvDiff2D::new(0.1, [0.0, 0.0]),
        boundary.mat,
        boundary.rhs,
        JacobianBackend::MatrixFree,
    );

    let n = problem.reduced_size();
    let mut y = Mat::<f64>::zeros(n, 1);
    let mut solver =
        DirkIntegrator::new(0.0, y.as_ref(), ImplicitBT::implicit_euler(), 1e-10, 1e-10);
    for _ in 0..200 {
        let step = solver.step(&system, 1.0).unwrap();
        y = step.y.clone();
        solver.accept_step(step);
    }
    assert!((solver.time() - 200.0).abs() < 1e-12);

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
