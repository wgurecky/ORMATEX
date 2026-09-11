use faer::prelude::MatRef;
use rlst::DynArray;

use crate::mesh::MeshMetadata;
use crate::simd;

use super::contexts::{CellState, LocalCtx};

// ponytail: fixed batches reuse scratch without creating a task per cell; tune only after profiling.
pub(crate) const CELL_BATCH_SIZE: usize = 128;

/// One-dimensional data needed by a tensor-product evaluator.
///
/// `q_to_local` maps a tensor-product quadrature index to the corresponding
/// local basis index. The finite-element package uses an entity-based local
/// ordering, so this permutation must not be inferred from `local_index`.
pub(crate) struct TensorProductData {
    pub(crate) n1d: usize,
    pub(crate) differentiation: Vec<f64>,
    pub(crate) q_to_local: Vec<usize>,
}

/// Cached per-cell-type quadrature, basis tabulation, and geometry data.
pub(crate) struct CellData {
    pub(crate) wts: Vec<f64>,
    pub(crate) npts: usize,
    pub(crate) ndofs: usize,
    pub(crate) table: DynArray<f64, 4>,
    pub(crate) reference_values: Vec<f64>,
    pub(crate) nodal_quadrature: Vec<usize>,
    /// Flattened as `[((cell * npts + q) * tdim + td) * gdim + gd]`.
    pub(crate) jinv_cache: Vec<f64>,
    pub(crate) jdets_cache: Vec<f64>,
    /// Weighted physical cell measure, flattened as `[cell, q]`.
    pub(crate) wdet_cache: Vec<f64>,
    pub(crate) cell_sizes: Vec<f64>,
    pub(crate) physical_points_cache: Vec<f64>,
    pub(crate) tensor: Option<TensorProductData>,
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
    field_count: usize,
    field_reduced_dofs: &[&[Option<usize>]],
    field_prescribed_values: &[&[Option<f64>]],
    field_offsets: &[usize],
    state: MatRef<'_, f64>,
    basis_grads: &[f64],
    values: &'a mut [f64],
    field_grads: &'a mut [f64],
) -> CellState<'a> {
    let quadrature_point_count = cell_data.npts;
    assert_eq!(field_reduced_dofs.len(), field_count);
    assert_eq!(field_prescribed_values.len(), field_count);
    assert!(field_offsets.len() >= field_count);
    values[..field_count * quadrature_point_count].fill(0.0);
    field_grads[..field_count * geometric_dimension * quadrature_point_count].fill(0.0);
    for field_index in 0..field_count {
        assert_eq!(
            field_reduced_dofs[field_index].len(),
            field_prescribed_values[field_index].len(),
            "cell DOF maps must have matching lengths"
        );
        for (local_basis_index, &reduced_dof) in field_reduced_dofs[field_index].iter().enumerate()
        {
            let coefficient = reduced_dof.map_or(
                field_prescribed_values[field_index][local_basis_index].unwrap_or(0.0),
                |reduced_dof| state[(field_offsets[field_index] + reduced_dof, 0)],
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
        field_indices: &[],
    }
}

/// Interpolate tensor-product coefficients that are already in local basis
/// order. This is used for Jacobian directions, whose eliminated DOFs are zero.
pub(crate) fn interpolate_tensor_cell_coefficients<'a>(
    cell_data: &CellData,
    geometric_dimension: usize,
    field_count: usize,
    coefficients: &[f64],
    values: &'a mut [f64],
    field_grads: &'a mut [f64],
    cell_index: usize,
) -> CellState<'a> {
    let tensor = cell_data
        .tensor
        .as_ref()
        .expect("tensor-product interpolation requires tensor data");
    let npts = cell_data.npts;
    let n1d = tensor.n1d;
    assert!(matches!(geometric_dimension, 1 | 2));
    assert_eq!(coefficients.len(), field_count * cell_data.ndofs);
    assert_eq!(values.len(), field_count * npts);
    assert_eq!(field_grads.len(), field_count * geometric_dimension * npts);

    for field in 0..field_count {
        let field_coefficients =
            &coefficients[field * cell_data.ndofs..(field + 1) * cell_data.ndofs];
        let field_values = &mut values[field * npts..(field + 1) * npts];
        for q in 0..npts {
            field_values[q] = field_coefficients[tensor.q_to_local[q]];
        }

        let field_grads = &mut field_grads
            [field * geometric_dimension * npts..(field + 1) * geometric_dimension * npts];
        if geometric_dimension == 1 {
            for q in 0..n1d {
                let reference = simd::dot(
                    &tensor.differentiation[q * n1d..(q + 1) * n1d],
                    field_values,
                );
                field_grads[q] = reference * cell_data.jinv_cache[cell_index * npts + q];
            }
        } else {
            let (gradients_x, gradients_y) = field_grads.split_at_mut(npts);
            for j in 0..n1d {
                for i in 0..n1d {
                    gradients_x[j * n1d + i] = simd::dot(
                        &tensor.differentiation[i * n1d..(i + 1) * n1d],
                        &field_values[j * n1d..(j + 1) * n1d],
                    );
                }
                let gradients_y = &mut gradients_y[j * n1d..(j + 1) * n1d];
                let first_factor = tensor.differentiation[j * n1d];
                for (gradient, &value) in gradients_y.iter_mut().zip(&field_values[..n1d]) {
                    *gradient = first_factor * value;
                }
                for a in 1..n1d {
                    simd::axpy(
                        gradients_y,
                        tensor.differentiation[j * n1d + a],
                        &field_values[a * n1d..(a + 1) * n1d],
                    );
                }
            }

            for q in 0..npts {
                let reference_x = gradients_x[q];
                let reference_y = gradients_y[q];
                let jinv = &cell_data.jinv_cache
                    [(cell_index * npts + q) * 4..(cell_index * npts + q + 1) * 4];
                gradients_x[q] = jinv[0] * reference_x + jinv[2] * reference_y;
                gradients_y[q] = jinv[1] * reference_x + jinv[3] * reference_y;
            }
        }
    }

    CellState {
        nfields: field_count,
        npts,
        gdim: geometric_dimension,
        values: &values[..field_count * npts],
        grads: &field_grads[..field_count * geometric_dimension * npts],
        field_indices: &[],
    }
}

/// Gather a reduced state and evaluate it with the tensor-product GLL basis.
pub(crate) fn interpolate_tensor_cell_state<'a>(
    cell_data: &CellData,
    geometric_dimension: usize,
    field_count: usize,
    field_reduced_dofs: &[&[Option<usize>]],
    field_prescribed_values: &[&[Option<f64>]],
    field_offsets: &[usize],
    state: MatRef<'_, f64>,
    coefficients: &mut [f64],
    values: &'a mut [f64],
    field_grads: &'a mut [f64],
    cell_index: usize,
) -> CellState<'a> {
    assert_eq!(field_reduced_dofs.len(), field_count);
    assert_eq!(field_prescribed_values.len(), field_count);
    assert!(field_offsets.len() >= field_count);
    assert_eq!(coefficients.len(), field_count * cell_data.ndofs);
    for field in 0..field_count {
        assert_eq!(
            field_reduced_dofs[field].len(),
            cell_data.ndofs,
            "tensor state interpolation requires complete local DOF maps"
        );
        assert_eq!(
            field_reduced_dofs[field].len(),
            field_prescribed_values[field].len(),
            "cell DOF maps must have matching lengths"
        );
        for (local, &reduced) in field_reduced_dofs[field].iter().enumerate() {
            coefficients[field * cell_data.ndofs + local] = reduced.map_or(
                field_prescribed_values[field][local].unwrap_or(0.0),
                |reduced| state[(field_offsets[field] + reduced, 0)],
            );
        }
    }
    interpolate_tensor_cell_coefficients(
        cell_data,
        geometric_dimension,
        field_count,
        coefficients,
        values,
        field_grads,
        cell_index,
    )
}
