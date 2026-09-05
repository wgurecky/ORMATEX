//! Shared finite-element contexts and geometry assembly infrastructure.

mod assembly;
mod boundary;
mod cell;
mod contexts;
mod reduction;

pub(crate) use assembly::{assemble_lumped_mass, push_local_matrix_triplets, scatter_local_vector};
pub(crate) use boundary::{
    add_dirichlet_rhs_correction, apply_quad_state_boundary_terms_cached,
    apply_quad_state_tensor_boundary_terms_cached, assemble_quad_boundaries,
    assemble_quad_state_boundary_jacobian, assemble_quad_state_boundary_residual_cached,
    assemble_quad_state_tensor_boundary_jacobian_cached,
    assemble_quad_state_tensor_boundary_residual_cached, build_quad_state_boundary_cache,
    QuadStateBoundaryCache,
};
pub use boundary::{
    dirichlet_values_with_precedence, BoundaryContributions, BoundaryFacet,
    StateBoundaryContributions,
};
pub(crate) use cell::{
    cell_ctx, interpolate_cell_state, interpolate_tensor_cell_coefficients,
    interpolate_tensor_cell_state, CellData, TensorProductData, CELL_BATCH_SIZE,
};
pub use contexts::{CellState, FacetCtx, LocalCtx, ShapeFn, TensorCtx, TensorFacetCtx};
pub(crate) use reduction::{FieldDofLayout, ReducedDofMap};
