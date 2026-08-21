//! Shared finite-element contexts and geometry assembly infrastructure.

mod assembly;
mod boundary;
mod cell;
mod contexts;
mod reduction;

pub(crate) use assembly::{
    assemble_lumped_mass, push_local_matrix_triplets, scatter_local_vector,
};
pub(crate) use boundary::{
    add_dirichlet_rhs_correction, apply_quad_state_boundary_terms, assemble_quad_boundaries,
    assemble_quad_state_boundary, assemble_quad_state_boundary_terms,
};
pub use boundary::{BoundaryContributions, BoundaryFacet, StateBoundaryContributions};
pub(crate) use cell::{cell_ctx, interpolate_cell_state, CellData, CELL_BATCH_SIZE};
pub use contexts::{CellState, FacetCtx, LocalCtx, ShapeFn};
pub(crate) use reduction::{FieldDofLayout, ReducedDofMap};
