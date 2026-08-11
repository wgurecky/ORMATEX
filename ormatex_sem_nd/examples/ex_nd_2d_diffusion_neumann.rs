//! Neumann/Robin diffusion on the unit square.

use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{shapes::unit_square, SingleElementMesh};

#[path = "support/diffusion_neumann.rs"]
mod diffusion_neumann;
#[path = "support/linear_system.rs"]
mod linear_system;
use diffusion_neumann::{run_diffusion_neumann, unit_square_neumann_robin_boundary};

fn main() {
    let mesh: SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>> =
        unit_square(32, 2, ReferenceCellType::Quadrilateral);
    run_diffusion_neumann(
        "diffusion-neumann-robin",
        mesh,
        2,
        unit_square_neumann_robin_boundary,
        "target/ex_nd_2d_diffusion_neumann_out.csv",
    );
}
