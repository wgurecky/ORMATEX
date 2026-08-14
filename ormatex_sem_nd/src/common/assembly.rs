use faer::sparse::{SparseColMat, Triplet};

use super::cell::CellData;

/// Add a field-major local vector into a reduced global vector.
///
/// `local[field * local_dof_count + local_dof]` is accumulated into
/// `out[field * reduced_dof_count + reduced_dof]`. Entries whose reduced
/// degree of freedom is `None` are omitted. Shared or periodically identified
/// degrees of freedom are combined by repeated additions.
pub(crate) fn scatter_local_vector(
    out: &mut [f64],
    local: &[f64],
    reduced_dofs: &[Option<usize>],
    field_count: usize,
    reduced_dof_count: usize,
) {
    let local_dof_count = reduced_dofs.len();
    for field in 0..field_count {
        for (local_dof, &reduced_dof) in reduced_dofs.iter().enumerate() {
            if let Some(reduced_dof) = reduced_dof {
                out[field * reduced_dof_count + reduced_dof] +=
                    local[field * local_dof_count + local_dof];
            }
        }
    }
}

/// Append selected entries of a field-major local matrix as global triplets.
///
/// The local matrix is row-major with field-major rows and columns. A row is
/// `equation * local_dof_count + test_dof`, and a column is
/// `unknown * local_dof_count + trial_dof`. Reduced degrees of freedom mapped
/// to `None` are omitted. Existing triplets are preserved and may contain
/// duplicate global coordinates from neighboring cells.
pub(crate) fn push_local_matrix_triplets(
    triplets: &mut Vec<Triplet<usize, usize, f64>>,
    local: &[f64],
    reduced_dofs: &[Option<usize>],
    field_count: usize,
    reduced_dof_count: usize,
    keep: impl Fn(f64) -> bool,
) {
    let local_dof_count = reduced_dofs.len();
    let local_size = field_count * local_dof_count;
    for equation in 0..field_count {
        for (test_dof, &reduced_test_dof) in reduced_dofs.iter().enumerate() {
            let Some(reduced_test_dof) = reduced_test_dof else {
                continue;
            };
            for unknown in 0..field_count {
                for (trial_dof, &reduced_trial_dof) in reduced_dofs.iter().enumerate() {
                    let Some(reduced_trial_dof) = reduced_trial_dof else {
                        continue;
                    };
                    let value = local[(equation * local_dof_count + test_dof) * local_size
                        + unknown * local_dof_count
                        + trial_dof];
                    if keep(value) {
                        triplets.push(Triplet::new(
                            equation * reduced_dof_count + reduced_test_dof,
                            unknown * reduced_dof_count + reduced_trial_dof,
                            value,
                        ));
                    }
                }
            }
        }
    }
}

/// Assemble a block-diagonal lumped GLL mass matrix for several scalar fields.
///
/// Each local nodal basis function contributes its quadrature weight multiplied
/// by the cell Jacobian determinant at that node. Eliminated degrees of freedom
/// are skipped, and contributions from cells sharing a reduced degree of
/// freedom are combined. The matrix has one identical scalar diagonal block
/// per field.
pub(crate) fn assemble_lumped_mass(
    cell_data: &CellData,
    cell_reduced_dofs: &[Vec<Option<usize>>],
    reduced_dof_count: usize,
    field_count: usize,
) -> SparseColMat<usize, f64> {
    assert!(field_count > 0, "mass requires at least one field");
    let mut triplets = Vec::with_capacity(cell_reduced_dofs.len() * cell_data.ndofs * field_count);
    for field in 0..field_count {
        for (cell_index, reduced_dofs) in cell_reduced_dofs.iter().enumerate() {
            for (local_dof, &reduced_dof) in reduced_dofs.iter().enumerate() {
                if let Some(reduced_dof) = reduced_dof {
                    let quadrature_index = cell_data.nodal_quadrature[local_dof];
                    let index = field * reduced_dof_count + reduced_dof;
                    triplets.push(Triplet::new(
                        index,
                        index,
                        cell_data.wts[quadrature_index]
                            * cell_data.jdets_cache[cell_index * cell_data.npts + quadrature_index],
                    ));
                }
            }
        }
    }
    let system_size = field_count * reduced_dof_count;
    SparseColMat::try_new_from_triplets(system_size, system_size, &triplets).unwrap()
}
