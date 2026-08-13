use faer::prelude::*;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{
    shapes::unit_square,
    traits::{Entity, Mesh, Topology},
    SingleElementMesh,
};
use ormatex_sem_nd::{
    DofReduction2D, KernelDiffusion2D, KernelVolumeSource, NeumannFlux, RobinConvection,
    SEM2DProblem,
};

#[path = "../examples/support/linear_system.rs"]
mod linear_system;
use linear_system::{implicit_euler_final_state, sparse_add, LinearOdeSys};

type QuadMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

fn boundary_facets(mesh: &QuadMesh) -> Vec<usize> {
    mesh.entity_iter(ReferenceCellType::Interval)
        .filter(|facet| {
            let topology = facet.topology();
            let mut cells = topology.connected_entity_iter(ReferenceCellType::Quadrilateral);
            cells.next().is_some() && cells.next().is_none()
        })
        .map(|facet| facet.local_index())
        .collect()
}

fn run_to_steady_state(
    problem: &SEM2DProblem<QuadMesh>,
    stiffness: faer::sparse::SparseColMat<usize, f64>,
    source: Vec<f64>,
) -> Mat<f64> {
    let mass = problem.assemble_lumped_mass();
    let system = LinearOdeSys::new(mass, stiffness, source);
    let y0 = Mat::<f64>::zeros(problem.reduced_size(), 1);
    implicit_euler_final_state(&system, y0.as_ref(), 1.0, 200, 1e-10)
}

#[test]
fn p2_transient_diffusion_reaches_constant_dirichlet_temperature() {
    let mesh = unit_square(4, 4, ReferenceCellType::Quadrilateral, 1);
    let facets = boundary_facets(&mesh);
    let temperature = 3.25;
    let problem = SEM2DProblem::new(
        mesh,
        2,
        DofReduction2D::Dirichlet {
            facets: facets
                .into_iter()
                .map(|facet| (facet, temperature))
                .collect(),
        },
    );
    let diffusion = KernelDiffusion2D::new(0.4);
    let source = problem.assemble_linear_with_dirichlet(&diffusion, &KernelVolumeSource::new(0.0));
    let state = run_to_steady_state(&problem, problem.assemble_bilinear(&diffusion), source);

    for row in 0..state.nrows() {
        assert!((state[(row, 0)] - temperature).abs() < 1e-8);
    }
}

#[test]
fn p2_transient_diffusion_reaches_neumann_robin_linear_x_profile() {
    let problem = SEM2DProblem::new(
        unit_square(4, 4, ReferenceCellType::Quadrilateral, 1),
        2,
        DofReduction2D::None,
    );
    let diffusivity = 0.4;
    let flux = 1.2;
    let h = 0.8;
    let ambient = 0.5;
    let diffusion = KernelDiffusion2D::new(diffusivity);
    let neumann = NeumannFlux::new(flux);
    let robin = RobinConvection::new(h, ambient);
    let boundary = problem.assemble_boundary(|facet| {
        if facet.midpoint[0] < 1e-12 {
            Some(&neumann)
        } else if facet.midpoint[0] > 1.0 - 1e-12 {
            Some(&robin)
        } else {
            None
        }
    });
    let stiffness = sparse_add(
        problem.assemble_bilinear(&diffusion).as_ref(),
        boundary.mat.as_ref(),
    );
    let state = run_to_steady_state(&problem, stiffness, boundary.rhs);
    let slope = -flux / diffusivity;
    let intercept = ambient + flux / h - slope;

    let mut max_error = 0.0_f64;
    for (row, (x, _y)) in problem.dof_positions().into_iter().enumerate() {
        let expected = slope * x + intercept;
        max_error = max_error.max((state[(row, 0)] - expected).abs());
    }
    assert!(max_error < 1e-8, "maximum steady-state error: {max_error}");
}
