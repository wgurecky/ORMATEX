//! Neumann/Robin diffusion on a Gmsh MSH2 quadrilateral mesh.

use ormatex_sem_nd::{gmsh_quad_data, NeumannFlux, QuadMesh, RobinConvection, SEM2DProblem};

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
    for tag in 1..=4 {
        assert!(metadata
            .facet_regions
            .iter()
            .any(|region| { region.map(|region| region.tag) == Some(tag) }));
    }
    run_diffusion_neumann_with_metadata(
        "diffusion-neumann-robin-gmsh",
        mesh,
        2,
        metadata,
        |problem: &SEM2DProblem<QuadMesh>| {
            let neumann = NeumannFlux::new(1.0);
            let robin = RobinConvection::new(0.1, 0.0);
            problem.assemble_boundary(|facet| {
                match facet.physical_region.map(|region| region.tag) {
                    Some(1) => Some(&neumann),
                    Some(2) => Some(&robin),
                    _ => None,
                }
            })
        },
        "target/ex_nd_2d_diffusion_neumann_gmsh_out.csv",
    );
}
