use std::fs::File;
use std::io::Write;

use faer::prelude::*;
use faer::sparse::SparseColMat;
use ndelement::types::ReferenceCellType;
use ndmesh::traits::Mesh;
use ormatex_sem_nd::{
    BoundaryContributions, BoundaryFacet, DofReduction2D, FieldRegistry, KernelAdvDiff2D,
    MeshMetadata, NeumannFlux, RobinConvection, SEM2DProblem,
};

use super::linear_system::{implicit_euler_final_state, sparse_add, LinearOdeSys};

pub fn diffusion_neumann_problem<M>(
    mesh: M,
    p: usize,
    metadata: MeshMetadata,
) -> (SEM2DProblem<M>, SparseColMat<usize, f64>, KernelAdvDiff2D)
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
{
    let problem = SEM2DProblem::new_with_metadata(
        mesh,
        p,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::None,
        metadata,
    );
    let mass = problem.assemble_lumped_mass();
    (problem, mass, KernelAdvDiff2D::new(0.1, [0.0, 0.0]))
}

pub fn unit_square_neumann_robin_boundary<M>(problem: &SEM2DProblem<M>) -> BoundaryContributions
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
{
    let neumann = NeumannFlux::new(1.0);
    let robin = RobinConvection::new(0.1, 0.0);
    problem.assemble_boundary(0.0, |facet: BoundaryFacet| {
        const EPS: f64 = 1e-9;
        if facet.midpoint[0] < EPS {
            Some(&neumann as &dyn ormatex_sem_nd::BoundaryIntegrator)
        } else if facet.midpoint[0] > 1.0 - EPS {
            Some(&robin as &dyn ormatex_sem_nd::BoundaryIntegrator)
        } else {
            None
        }
    })
}

pub fn run_diffusion_neumann<M, F>(
    label: &str,
    mesh: M,
    p: usize,
    assemble_boundary: F,
    out_path: &str,
) where
    M: ndmesh::traits::Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
    F: FnOnce(&SEM2DProblem<M>) -> BoundaryContributions,
{
    run_diffusion_neumann_with_metadata(
        label,
        mesh,
        p,
        MeshMetadata::default(),
        assemble_boundary,
        out_path,
    );
}

pub fn run_diffusion_neumann_with_metadata<M, F>(
    label: &str,
    mesh: M,
    p: usize,
    metadata: MeshMetadata,
    assemble_boundary: F,
    out_path: &str,
) where
    M: ndmesh::traits::Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
    F: FnOnce(&SEM2DProblem<M>) -> BoundaryContributions,
{
    let q_left = 1.0;
    let k = 0.1;
    let h = 0.1;
    let t_amb = 0.0;
    let dt = 1.0;
    let nsteps = 200;
    println!("\n=== {label} (p={p}) ===");

    let (problem, mass, kernel) = diffusion_neumann_problem(mesh, p, metadata);
    let n = problem.reduced_size();
    let k_diff = problem.assemble_bilinear(0.0, &kernel);
    let boundary = assemble_boundary(&problem);
    let b = boundary.rhs;
    let k_eff = sparse_add(k_diff.as_ref(), boundary.mat.as_ref());

    let system = LinearOdeSys::new(mass, k_eff, b);
    let y0 = Mat::<f64>::zeros(n, 1);
    let y = implicit_euler_final_state(&system, y0.as_ref(), dt, nsteps, 1e-10);

    let positions = problem.dof_positions();
    let slope = -q_left / k;
    let intercept = t_amb + q_left / h - slope;
    let mut max_error = 0.0_f64;
    for row in 0..n {
        max_error = max_error.max((y[(row, 0)] - (slope * positions[row].0 + intercept)).abs());
    }
    assert!(
        max_error < 5e-3,
        "steady-state error {max_error} exceeds tolerance"
    );
    let mut output = File::create(out_path).expect("failed to create output csv");
    writeln!(output, "x,y,T").unwrap();
    for row in 0..n {
        writeln!(
            output,
            "{:.6},{:.6},{:.9e}",
            positions[row].0,
            positions[row].1,
            y[(row, 0)]
        )
        .unwrap();
    }
}
