//! Neumann/Robin diffusion on a Gmsh MSH2 quadrilateral mesh.

use std::collections::HashSet;

use ormatex_sem_nd::{
    gmsh_quad_data, BoundaryIntegrator, NeumannFlux, QuadMesh, RobinConvection, SEM2DProblem,
};

#[path = "support/diffusion_neumann.rs"]
mod diffusion_neumann;
#[path = "support/linear_system.rs"]
mod linear_system;
use diffusion_neumann::run_diffusion_neumann_with_metadata;

fn main() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/ex_nd_2d_diffusion_neumann_gmsh.msh"
    );
    let data = gmsh_quad_data(path).expect("failed to load Gmsh quadrilateral mesh");
    let mesh = data.mesh;
    let metadata = data.metadata;
    let neumann_facets: HashSet<_> = metadata.boundary_facets("left").into_iter().collect();
    let robin_facets: HashSet<_> = metadata.boundary_facets("right").into_iter().collect();
    assert!(!neumann_facets.is_empty() && !robin_facets.is_empty());
    run_diffusion_neumann_with_metadata(
        "diffusion-neumann-robin-gmsh",
        mesh,
        2,
        metadata,
        |problem: &SEM2DProblem<QuadMesh>| {
            let neumann = NeumannFlux::new(1.0);
            let robin = RobinConvection::new(0.1, 0.0);
            problem.assemble_boundary(0.0, |facet| -> Option<&dyn BoundaryIntegrator> {
                if neumann_facets.contains(&facet.index) {
                    Some(&neumann as &dyn BoundaryIntegrator)
                } else if robin_facets.contains(&facet.index) {
                    Some(&robin as &dyn BoundaryIntegrator)
                } else {
                    None
                }
            })
        },
        "target/ex_nd_2d_diffusion_neumann_gmsh_out.csv",
    );
}
