use faer::prelude::*;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap};
use ndmesh::shapes::unit_interval;
use ndmesh::SingleElementMesh;
use ormatex_sem_nd::{
    DofReduction1D, FieldRegistry, KernelDiffusion, KernelVolumeSource, NeumannFlux,
    RobinConvection, SEM1DProblem,
};

#[path = "../examples/support/linear_system.rs"]
mod linear_system;
use linear_system::{implicit_euler_final_state, sparse_add, LinearOdeSys};

type IntervalMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

fn run_to_steady_state(
    problem: &SEM1DProblem<IntervalMesh>,
    stiffness: faer::sparse::SparseColMat<usize, f64>,
    source: Vec<f64>,
) -> Mat<f64> {
    let mass = problem.assemble_lumped_mass();
    let system = LinearOdeSys::new(mass, stiffness, source);
    let y0 = Mat::<f64>::zeros(problem.reduced_size(), 1);
    implicit_euler_final_state(&system, y0.as_ref(), 1.0, 200, 1e-10)
}

#[test]
fn p2_transient_diffusion_reaches_dirichlet_linear_profile() {
    let nx = 8;
    let left_temperature = 1.5;
    let right_temperature = 4.5;
    let problem = SEM1DProblem::new(
        unit_interval(nx),
        2,
        FieldRegistry::new(["temperature"]),
        DofReduction1D::Dirichlet {
            facets: vec![(0, left_temperature), (nx, right_temperature)],
        },
    );
    let diffusion = KernelDiffusion::new(0.4);
    let source =
        problem.assemble_linear_with_dirichlet(0.0, &diffusion, &KernelVolumeSource::new(0.0));
    let state = run_to_steady_state(&problem, problem.assemble_bilinear(0.0, &diffusion), source);

    for (row, x) in problem.dof_positions().into_iter().enumerate() {
        let expected = left_temperature + (right_temperature - left_temperature) * x;
        assert!((state[(row, 0)] - expected).abs() < 1e-8);
    }
}

#[test]
fn p2_transient_diffusion_reaches_neumann_robin_linear_profile() {
    let problem = SEM1DProblem::new(
        unit_interval(8),
        2,
        FieldRegistry::new(["temperature"]),
        DofReduction1D::None,
    );
    let diffusivity = 0.4;
    let flux = 1.2;
    let h = 0.8;
    let ambient = 0.5;
    let diffusion = KernelDiffusion::new(diffusivity);
    let neumann = NeumannFlux::new(flux);
    let robin = RobinConvection::new(h, ambient);
    let boundary = problem.assemble_boundary(0.0, |point| {
        if point.coordinate < 1e-12 {
            Some(&neumann)
        } else if point.coordinate > 1.0 - 1e-12 {
            Some(&robin)
        } else {
            None
        }
    });
    let stiffness = sparse_add(
        problem.assemble_bilinear(0.0, &diffusion).as_ref(),
        boundary.mat.as_ref(),
    );
    let state = run_to_steady_state(&problem, stiffness, boundary.rhs);
    let slope = -flux / diffusivity;
    let intercept = ambient + flux / h - slope;

    let mut max_error = 0.0_f64;
    for (row, x) in problem.dof_positions().into_iter().enumerate() {
        let expected = slope * x + intercept;
        max_error = max_error.max((state[(row, 0)] - expected).abs());
    }
    assert!(max_error < 1e-8, "maximum steady-state error: {max_error}");
}
