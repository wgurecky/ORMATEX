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
    field_reduced_dofs: &[&[Option<usize>]],
    field_count: usize,
    field_offsets: &[usize],
) {
    let local_dof_count = field_reduced_dofs[0].len();
    for field in 0..field_count {
        assert_eq!(field_reduced_dofs[field].len(), local_dof_count);
        for (local_dof, &reduced_dof) in field_reduced_dofs[field].iter().enumerate() {
            if let Some(reduced_dof) = reduced_dof {
                out[field_offsets[field] + reduced_dof] +=
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
    field_reduced_dofs: &[&[Option<usize>]],
    field_count: usize,
    field_offsets: &[usize],
    keep: impl Fn(f64) -> bool,
) {
    let local_dof_count = field_reduced_dofs[0].len();
    let local_size = field_count * local_dof_count;
    for equation in 0..field_count {
        for (test_dof, &reduced_test_dof) in field_reduced_dofs[equation].iter().enumerate() {
            let Some(reduced_test_dof) = reduced_test_dof else {
                continue;
            };
            for unknown in 0..field_count {
                for (trial_dof, &reduced_trial_dof) in
                    field_reduced_dofs[unknown].iter().enumerate()
                {
                    let Some(reduced_trial_dof) = reduced_trial_dof else {
                        continue;
                    };
                    let value = local[(equation * local_dof_count + test_dof) * local_size
                        + unknown * local_dof_count
                        + trial_dof];
                    if keep(value) {
                        triplets.push(Triplet::new(
                            field_offsets[equation] + reduced_test_dof,
                            field_offsets[unknown] + reduced_trial_dof,
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
    cell_reduced_dofs: &[Vec<Vec<Option<usize>>>],
    field_offsets: &[usize],
    field_count: usize,
) -> SparseColMat<usize, f64> {
    assert!(field_count > 0, "mass requires at least one field");
    let mut triplets = Vec::with_capacity(cell_reduced_dofs.len() * cell_data.ndofs * field_count);
    for field in 0..field_count {
        let field_cells = if cell_reduced_dofs.len() == 1 {
            &cell_reduced_dofs[0]
        } else {
            &cell_reduced_dofs[field]
        };
        for (cell_index, field_dofs) in field_cells.iter().enumerate() {
            for (local_dof, &reduced_dof) in field_dofs.iter().enumerate() {
                if let Some(reduced_dof) = reduced_dof {
                    let quadrature_index = cell_data.nodal_quadrature[local_dof];
                    let index = field_offsets[field] + reduced_dof;
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
    let system_size = *field_offsets.last().unwrap();
    SparseColMat::try_new_from_triplets(system_size, system_size, &triplets).unwrap()
}
