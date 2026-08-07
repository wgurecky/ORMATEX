//! Neumann/Robin diffusion on a Gmsh MSH2 quadrilateral mesh.

use ormatex_sem_nd::{
    gmsh_quad_mesh, FiniteElement2DProblem, NeumannFlux, QuadMesh, RobinConvection,
};

#[path = "support/diffusion_neumann.rs"]
mod diffusion_neumann;
#[path = "support/linear_system.rs"]
mod linear_system;
use diffusion_neumann::run_diffusion_neumann;

fn main() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/ex_nd_2d_diffusion_neumann_gmsh.msh"
    );
    let (mesh, facet_tags) = gmsh_quad_mesh(path).expect("failed to load Gmsh quadrilateral mesh");
    for tag in 1..=4 {
        assert!(facet_tags.iter().any(|facet_tag| *facet_tag == Some(tag)));
    }
    run_diffusion_neumann(
        "diffusion-neumann-robin-gmsh",
        mesh,
        2,
        |problem: &FiniteElement2DProblem<QuadMesh>| {
            let neumann = NeumannFlux::new(1.0);
            let robin = RobinConvection::new(0.1, 0.0);
            problem.assemble_boundary(|facet| match facet_tags[facet.index] {
                Some(1) => Some(&neumann),
                Some(2) => Some(&robin),
                _ => None,
            })
        },
        "target/ex_nd_2d_diffusion_neumann_gmsh_out.csv",
    );
}
