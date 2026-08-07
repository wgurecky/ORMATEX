//! Neumann/Robin diffusion on the unit square.

use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{shapes::unit_square, SingleElementMesh};
use ormatex_sem_nd::{NeumannFlux, RobinConvection};

#[path = "support/diffusion_neumann.rs"]
mod diffusion_neumann;
#[path = "support/linear_system.rs"]
mod linear_system;
use diffusion_neumann::run_diffusion_neumann;

fn main() {
    let mesh: SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>> =
        unit_square(32, 2, ReferenceCellType::Quadrilateral);
    run_diffusion_neumann(
        "diffusion-neumann-robin",
        mesh,
        2,
        |problem| {
            let neumann = NeumannFlux::new(1.0);
            let robin = RobinConvection::new(0.1, 0.0);
            problem.assemble_boundary(|facet| {
                const EPS: f64 = 1e-9;
                if facet.midpoint[0] < EPS {
                    Some(&neumann)
                } else if facet.midpoint[0] > 1.0 - EPS {
                    Some(&robin)
                } else {
                    None
                }
            })
        },
        "target/ex_nd_2d_diffusion_neumann_out.csv",
    );
}
