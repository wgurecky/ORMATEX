use faer::prelude::MatRef;
use rlst::DynArray;

use crate::material::MeshMetadata;

use super::contexts::{CellState, LocalCtx};

// ponytail: fixed batches reuse scratch without creating a task per cell; tune only after profiling.
pub(crate) const CELL_BATCH_SIZE: usize = 32;

/// Cached per-cell-type quadrature, basis tabulation, and geometry data.
pub(crate) struct CellData {
    pub(crate) wts: Vec<f64>,
    pub(crate) npts: usize,
    pub(crate) ndofs: usize,
    pub(crate) table: DynArray<f64, 4>,
    pub(crate) reference_values: Vec<f64>,
    pub(crate) nodal_quadrature: Vec<usize>,
    pub(crate) jinv_cache: DynArray<f64, 4>,
    pub(crate) jdets_cache: Vec<f64>,
    pub(crate) physical_points_cache: Vec<f64>,
}

/// Build a [`LocalCtx`] from cached data for one cell.
///
/// `cell_index` selects the cell metadata, Jacobian determinants, and physical
/// quadrature points. `basis_grads` contains physical basis gradients in
/// `[local_basis, geometric_direction, quadrature_point]` order for the first
/// `local_dof_count` basis functions. The returned context borrows the cached
/// data and `basis_grads`.
pub(crate) fn cell_ctx<'a>(
    cell_data: &'a CellData,
    metadata: &'a MeshMetadata,
    topological_dimension: usize,
    geometric_dimension: usize,
    time: f64,
    cell_index: usize,
    local_dof_count: usize,
    basis_grads: &'a [f64],
) -> LocalCtx<'a> {
    let quadrature_point_count = cell_data.npts;
    LocalCtx {
        time,
        cell: metadata.cell(cell_index),
        tdim: topological_dimension,
        gdim: geometric_dimension,
        ncomp: 1,
        npts: quadrature_point_count,
        ndofs: local_dof_count,
        wts: &cell_data.wts,
        jdets: &cell_data.jdets_cache
            [cell_index * quadrature_point_count..(cell_index + 1) * quadrature_point_count],
        points: &cell_data.physical_points_cache[cell_index
            * quadrature_point_count
            * geometric_dimension
            ..(cell_index + 1) * quadrature_point_count * geometric_dimension],
        values: &cell_data.reference_values,
        grads: &basis_grads[..local_dof_count * geometric_dimension * quadrature_point_count],
    }
}

/// Interpolate a reduced global state and its physical gradients to cell
/// quadrature points, including prescribed values for eliminated DOFs.
///
/// `reduced_dofs` maps each local basis function to a reduced global degree of
/// freedom. An entry of `None` uses the matching `prescribed_values` entry.
/// The state is field-major with row `field * reduced_dof_count + reduced_dof`
/// and must have one column. The output buffers use
/// `values[field * quadrature_point_count + q]` and
/// `field_grads[(field * geometric_dimension + geometric_direction) *
/// quadrature_point_count + q]` layouts.
pub(crate) fn interpolate_cell_state<'a>(
    cell_data: &CellData,
    geometric_dimension: usize,
    reduced_dof_count: usize,
    field_count: usize,
    reduced_dofs: &[Option<usize>],
    prescribed_values: &[Option<f64>],
    state: MatRef<'_, f64>,
    basis_grads: &[f64],
    values: &'a mut [f64],
    field_grads: &'a mut [f64],
) -> CellState<'a> {
    let quadrature_point_count = cell_data.npts;
    assert_eq!(
        reduced_dofs.len(),
        prescribed_values.len(),
        "cell DOF maps must have matching lengths"
    );
    values[..field_count * quadrature_point_count].fill(0.0);
    field_grads[..field_count * geometric_dimension * quadrature_point_count].fill(0.0);
    for field_index in 0..field_count {
        for (local_basis_index, &reduced_dof) in reduced_dofs.iter().enumerate() {
            let coefficient = reduced_dof.map_or(
                prescribed_values[local_basis_index].unwrap_or(0.0),
                |reduced_dof| state[(field_index * reduced_dof_count + reduced_dof, 0)],
            );
            for quadrature_index in 0..quadrature_point_count {
                values[field_index * quadrature_point_count + quadrature_index] += coefficient
                    * cell_data.reference_values
                        [local_basis_index * quadrature_point_count + quadrature_index];
                for geometric_direction in 0..geometric_dimension {
                    field_grads[(field_index * geometric_dimension + geometric_direction)
                        * quadrature_point_count
                        + quadrature_index] += coefficient
                        * basis_grads[(local_basis_index * geometric_dimension
                            + geometric_direction)
                            * quadrature_point_count
                            + quadrature_index];
                }
            }
        }
    }
    CellState {
        nfields: field_count,
        npts: quadrature_point_count,
        gdim: geometric_dimension,
        values: &values[..field_count * quadrature_point_count],
        grads: &field_grads[..field_count * geometric_dimension * quadrature_point_count],
    }
}
