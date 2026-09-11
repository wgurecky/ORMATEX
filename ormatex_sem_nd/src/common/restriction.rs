use faer::MatRef;
use rayon::prelude::*;

/// Reusable element restriction for field-major SEM vectors.
///
/// `gather` is libCEED's E operation. `transpose_reduce` is E^T: it reduces
/// element-local actions by global output row, avoiding atomics and coloring.
pub(crate) struct ElementRestriction {
    maps: Vec<Vec<Vec<Option<usize>>>>,
    prescribed: Vec<Vec<Vec<Option<f64>>>>,
    cell_size: usize,
    rows: Vec<Vec<(usize, usize, usize)>>,
    total_size: usize,
}

impl ElementRestriction {
    pub(crate) fn new(
        maps: &[Vec<Vec<Option<usize>>>],
        prescribed: &[Vec<Vec<Option<f64>>>],
        field_sizes: &[usize],
    ) -> Self {
        assert!(!maps.is_empty(), "element restriction needs a field map");
        assert_eq!(
            maps.len(),
            prescribed.len(),
            "restriction map count mismatch"
        );
        assert!(
            !field_sizes.is_empty(),
            "element restriction needs a field size"
        );
        assert!(
            maps.len() == 1 || maps.len() == field_sizes.len(),
            "element restriction field map count mismatch"
        );
        let cell_count = maps[0].len();
        assert!(
            cell_count > 0,
            "element restriction needs at least one cell"
        );
        let cell_size = maps[0][0].len();
        for field in 0..maps.len() {
            assert_eq!(maps[field].len(), cell_count, "element count mismatch");
            assert_eq!(
                prescribed[field].len(),
                cell_count,
                "prescribed cell count mismatch"
            );
            for cell in 0..cell_count {
                assert_eq!(
                    maps[field][cell].len(),
                    cell_size,
                    "element local sizes must match"
                );
                assert_eq!(
                    prescribed[field][cell].len(),
                    cell_size,
                    "prescribed local sizes must match"
                );
            }
        }

        let mut offsets = Vec::with_capacity(field_sizes.len());
        let mut total_size = 0;
        for &size in field_sizes {
            offsets.push(total_size);
            total_size += size;
        }
        let mut rows = vec![Vec::new(); total_size];
        for cell in 0..cell_count {
            for field in 0..field_sizes.len() {
                let source_field = if maps.len() == 1 { 0 } else { field };
                for (local, &reduced) in maps[source_field][cell].iter().enumerate() {
                    if let Some(reduced) = reduced {
                        rows[offsets[field] + reduced].push((cell, local, field));
                    }
                }
            }
        }

        Self {
            maps: maps.to_vec(),
            prescribed: prescribed.to_vec(),
            cell_size,
            rows,
            total_size,
        }
    }

    pub(crate) fn maps_for(
        &self,
        cell: usize,
        fields: &[usize],
    ) -> (Vec<&[Option<usize>]>, Vec<&[Option<f64>]>) {
        let mut maps = Vec::with_capacity(fields.len());
        let mut prescribed = Vec::with_capacity(fields.len());
        for &field in fields {
            let source = if self.maps.len() == 1 { 0 } else { field };
            maps.push(self.maps[source][cell].as_slice());
            prescribed.push(self.prescribed[source][cell].as_slice());
        }
        (maps, prescribed)
    }

    #[inline]
    pub(crate) fn gather_state(
        &self,
        cell: usize,
        fields: &[usize],
        offsets: &[usize],
        state: MatRef<'_, f64>,
        out: &mut [f64],
    ) {
        assert_eq!(out.len(), fields.len() * self.cell_size);
        for (field_pos, &field) in fields.iter().enumerate() {
            let source = if self.maps.len() == 1 { 0 } else { field };
            let map = &self.maps[source][cell];
            let prescribed = &self.prescribed[source][cell];
            for (local, &reduced) in map.iter().enumerate() {
                out[field_pos * self.cell_size + local] = reduced
                    .map_or(prescribed[local].unwrap_or(0.0), |reduced| {
                        state[(offsets[field_pos] + reduced, 0)]
                    });
            }
        }
    }

    pub(crate) fn gather_direction_column(
        &self,
        cell: usize,
        fields: &[usize],
        offsets: &[usize],
        direction: MatRef<'_, f64>,
        column: usize,
        out: &mut [f64],
    ) {
        assert_eq!(out.len(), fields.len() * self.cell_size);
        for (field_pos, &field) in fields.iter().enumerate() {
            let source = if self.maps.len() == 1 { 0 } else { field };
            let map = &self.maps[source][cell];
            for (local, &reduced) in map.iter().enumerate() {
                out[field_pos * self.cell_size + local] = reduced.map_or(0.0, |reduced| {
                    direction[(offsets[field_pos] + reduced, column)]
                });
            }
        }
    }

    /// Reduce element-major actions by global row. `actions` is laid out as
    /// `[cell][column][output_field][local_dof]`.
    pub(crate) fn transpose_reduce(
        &self,
        actions: &[f64],
        action_stride: usize,
        output_size: usize,
        ncols: usize,
        output_fields: &[usize],
    ) -> Vec<f64> {
        assert_eq!(output_size, output_fields.len() * self.cell_size);
        let mut field_positions = vec![
            usize::MAX;
            self.maps.len().max(
                output_fields.iter().copied().max().unwrap_or(0) + 1,
            )
        ];
        for (position, &field) in output_fields.iter().enumerate() {
            if field_positions.len() <= field {
                field_positions.resize(field + 1, usize::MAX);
            }
            field_positions[field] = position;
        }

        let mut output = vec![0.0; self.total_size * ncols];
        output
            .par_chunks_mut(ncols)
            .enumerate()
            .for_each(|(row, destination)| {
                for &(cell, local, field) in &self.rows[row] {
                    let field_position = field_positions.get(field).copied().unwrap_or(usize::MAX);
                    if field_position == usize::MAX {
                        continue;
                    }
                    let local_offset = field_position * self.cell_size + local;
                    for column in 0..ncols {
                        destination[column] +=
                            actions[cell * action_stride + column * output_size + local_offset];
                    }
                }
            });
        output
    }
}

#[cfg(test)]
mod tests {
    use super::ElementRestriction;
    use faer::Mat;

    #[test]
    fn restriction_transpose_reduces_shared_dofs() {
        let maps = vec![vec![vec![Some(0), Some(1)], vec![Some(1), Some(2)]]];
        let prescribed = vec![vec![vec![None, None], vec![None, None]]];
        let restriction = ElementRestriction::new(&maps, &prescribed, &[3]);
        let actions = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            restriction.transpose_reduce(&actions, 2, 2, 1, &[0]),
            vec![1.0, 5.0, 4.0]
        );

        let state = Mat::from_fn(3, 1, |row, _| row as f64 + 1.0);
        let mut local = vec![0.0; 2];
        restriction.gather_state(1, &[0], &[0], state.as_ref(), &mut local);
        assert_eq!(local, vec![2.0, 3.0]);
    }
}
