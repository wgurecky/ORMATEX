use faer::MatRef;
use rayon::prelude::*;

/// Greedy coloring of cells so same-color cells share no global row.
///
/// Conflicts come from `rows` (row -> sharing cells); each color class is an
/// independent set, so its scatters are row-disjoint. Runs once at problem
/// construction; cost is linear in cells plus sharing pairs.
fn color_cells(cell_count: usize, rows: &[Vec<(usize, usize, usize)>]) -> Vec<Vec<usize>> {
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); cell_count];
    for entries in rows {
        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                let (a, b) = (entries[i].0, entries[j].0);
                if a != b {
                    neighbors[a].push(b);
                    neighbors[b].push(a);
                }
            }
        }
    }
    let mut color_of = vec![usize::MAX; cell_count];
    let mut colors: Vec<Vec<usize>> = Vec::new();
    let mut forbidden = Vec::new();
    for cell in 0..cell_count {
        forbidden.clear();
        forbidden.resize(colors.len(), false);
        for &other in &neighbors[cell] {
            let used = color_of[other];
            if used != usize::MAX {
                forbidden[used] = true;
            }
        }
        let color = forbidden
            .iter()
            .position(|&used| !used)
            .unwrap_or(colors.len());
        if color == colors.len() {
            colors.push(Vec::new());
        }
        colors[color].push(cell);
        color_of[cell] = color;
    }
    colors
}

/// Verify the coloring invariant: no row is shared within one color.
fn colors_are_disjoint(
    cell_count: usize,
    rows: &[Vec<(usize, usize, usize)>],
    colors: &[Vec<usize>],
) -> bool {
    let mut color_of = vec![usize::MAX; cell_count];
    for (color, cells) in colors.iter().enumerate() {
        for &cell in cells {
            if color_of[cell] != usize::MAX {
                return false;
            }
            color_of[cell] = color;
        }
    }
    if color_of.iter().any(|&color| color == usize::MAX) {
        return false;
    }
    // Distinct cells sharing a row must differ; repeats of one cell in a row
    // (two locals merged to one reduced DOF, e.g. periodic wrap) accumulate
    // within that cell's own action and need no separation.
    rows.iter().all(|entries| {
        entries.iter().enumerate().all(|(i, &(cell, _, _))| {
            entries[..i]
                .iter()
                .all(|&(other, _, _)| other == cell || color_of[other] != color_of[cell])
        })
    })
}

/// Reusable element restriction for field-major SEM vectors.
///
/// `gather` is libCEED's E operation. `transpose_reduce` is E^T: it reduces
/// element-local actions by global output row, avoiding atomics and coloring.
pub(crate) struct ElementRestriction {
    maps: Vec<Vec<Vec<Option<usize>>>>,
    prescribed: Vec<Vec<Vec<Option<f64>>>>,
    cell_size: usize,
    rows: Vec<Vec<(usize, usize, usize)>>,
    /// Global row offset per field, matching `rows` and `field_sizes` order.
    offsets: Vec<usize>,
    /// Greedy cell coloring: same-color cells share no reduced DOF, so one
    /// color's actions scatter into disjoint global rows with no atomics and
    /// no second reduction pass (deal.II `partition_partition` analogue).
    colors: Vec<Vec<usize>>,
    total_size: usize,
}

/// Raw global-output handle for color-disjoint parallel scatter.
///
/// Concurrent scatters within one color touch disjoint rows (see
/// [`ElementRestriction::cell_colors`]). The type cannot enforce this; the
/// construction-time debug check in [`ElementRestriction::new`] does.
#[derive(Clone, Copy)]
pub(crate) struct DisjointOut {
    ptr: *mut f64,
}

// SAFETY: sound only for row-disjoint concurrent use; callers uphold this by
// scattering one color at a time (colors are sequential, cells parallel).
unsafe impl Send for DisjointOut {}
unsafe impl Sync for DisjointOut {}

impl DisjointOut {
    /// Wrap one zeroed global column for disjoint scatter-adds.
    ///
    /// SAFETY: `slice` must outlive all uses, and concurrent uses of handles
    /// derived from overlapping memory must be row-disjoint.
    pub(crate) unsafe fn new(slice: &mut [f64]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
        }
    }

    /// Add `value` into one global row.
    ///
    /// SAFETY: same requirements as the handle itself: concurrent adds must
    /// target disjoint rows.
    pub(crate) unsafe fn add(&self, row: usize, value: f64) {
        *self.ptr.add(row) += value;
    }
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

        let colors = color_cells(cell_count, &rows);
        debug_assert!(
            colors_are_disjoint(cell_count, &rows, &colors),
            "same-color cells must not share a reduced DOF"
        );

        Self {
            maps: maps.to_vec(),
            prescribed: prescribed.to_vec(),
            cell_size,
            rows,
            offsets,
            colors,
            total_size,
        }
    }

    pub(crate) fn maps_for(
        &self,
        cell: usize,
        fields: &[usize],
    ) -> (Vec<&[Option<usize>]>, Vec<&[Option<f64>]>) {        let mut maps = Vec::with_capacity(fields.len());
        let mut prescribed = Vec::with_capacity(fields.len());
        for &field in fields {
            let source = if self.maps.len() == 1 { 0 } else { field };
            maps.push(self.maps[source][cell].as_slice());
            prescribed.push(self.prescribed[source][cell].as_slice());
        }
        (maps, prescribed)
    }

    /// Cells grouped by color; same-color cells share no reduced DOF.
    ///
    /// Process colors sequentially and cells within a color in parallel,
    /// scattering with [`ElementRestriction::scatter_add_column`] directly
    /// into the global vector (no intermediate actions, no second pass).
    pub(crate) fn cell_colors(&self) -> &[Vec<usize>] {
        &self.colors
    }

    /// Add field-major `local` (`[field_pos][local]`) into one global column.
    ///
    /// `output_fields` holds global field ids; `None` (eliminated) map entries
    /// are skipped. Performs no synchronization: concurrent calls must be
    /// row-disjoint (see [`ElementRestriction::cell_colors`]).
    ///
    /// SAFETY: `out` must point to `total_size` writable doubles that remain
    /// valid for the call.
    pub(crate) unsafe fn scatter_add_column(
        &self,
        cell: usize,
        output_fields: &[usize],
        local: &[f64],
        out: DisjointOut,
    ) {
        debug_assert_eq!(local.len(), output_fields.len() * self.cell_size);
        for (position, &field) in output_fields.iter().enumerate() {
            debug_assert!(field < self.offsets.len(), "field index out of range");
            let source = if self.maps.len() == 1 { 0 } else { field };
            let map = &self.maps[source][cell];
            let base = self.offsets[field];
            let row = &local[position * self.cell_size..(position + 1) * self.cell_size];
            for (local_dof, &reduced) in map.iter().enumerate() {
                if let Some(reduced) = reduced {
                    *out.ptr.add(base + reduced) += row[local_dof];
                }
            }
        }
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

    /// Batched lane-contiguous gather for SIMD-over-element tiles.
    ///
    /// `cells.len() <= lane_width`; output is `[(field_pos * cell_size + local) *
    /// lane_width + lane]`. Irregular restriction stays scalar; callers run
    /// dense tensor ops across lanes afterwards.
    pub(crate) fn gather_state_batch(
        &self,
        cells: &[usize],
        fields: &[usize],
        offsets: &[usize],
        state: MatRef<'_, f64>,
        out: &mut [f64],
        lane_width: usize,
    ) {
        assert!(cells.len() <= lane_width);
        assert_eq!(out.len(), fields.len() * self.cell_size * lane_width);
        // ponytail: lane-inner order keeps the W consecutive lane writes contiguous.
        for (field_pos, &field) in fields.iter().enumerate() {
            let source = if self.maps.len() == 1 { 0 } else { field };
            for local in 0..self.cell_size {
                for (lane, &cell) in cells.iter().enumerate() {
                    let map = &self.maps[source][cell];
                    let prescribed = &self.prescribed[source][cell];
                    let reduced = map[local];
                    out[(field_pos * self.cell_size + local) * lane_width + lane] = reduced
                        .map_or(prescribed[local].unwrap_or(0.0), |reduced| {
                            state[(offsets[field_pos] + reduced, 0)]
                        });
                }
            }
        }
    }

    /// Batched lane-contiguous direction gather (eliminated DOFs read as zero).
    pub(crate) fn gather_direction_batch(
        &self,
        cells: &[usize],
        fields: &[usize],
        offsets: &[usize],
        direction: MatRef<'_, f64>,
        column: usize,
        out: &mut [f64],
        lane_width: usize,
    ) {
        assert!(cells.len() <= lane_width);
        assert_eq!(out.len(), fields.len() * self.cell_size * lane_width);
        for (field_pos, &field) in fields.iter().enumerate() {
            let source = if self.maps.len() == 1 { 0 } else { field };
            for local in 0..self.cell_size {
                for (lane, &cell) in cells.iter().enumerate() {
                    let reduced = self.maps[source][cell][local];
                    out[(field_pos * self.cell_size + local) * lane_width + lane] =
                        reduced.map_or(0.0, |reduced| {
                            direction[(offsets[field_pos] + reduced, column)]
                        });
                }
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
    use super::{DisjointOut, ElementRestriction};
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

    #[test]
    fn coloring_scatter_matches_transpose_reduce() {
        // Chain cell0 {0,1} - cell1 {1,2} - cell2 {2,3}: needs two colors,
        // with the disjoint ends sharing one.
        let maps = vec![vec![
            vec![Some(0), Some(1)],
            vec![Some(1), Some(2)],
            vec![Some(2), Some(3)],
        ]];
        let prescribed = vec![vec![vec![None, None], vec![None, None], vec![None, None]]];
        let restriction = ElementRestriction::new(&maps, &prescribed, &[4]);
        assert_eq!(restriction.cell_colors().len(), 2);

        let mut global = vec![0.0; 4];
        {
            let out = unsafe { DisjointOut::new(&mut global) };
            for cells in restriction.cell_colors() {
                for &cell in cells {
                    // SAFETY: test scatters serially, hence trivially disjoint.
                    unsafe {
                        restriction.scatter_add_column(cell, &[0], &[1.0, 1.0], out)
                    };
                }
            }
        }
        assert_eq!(global, vec![1.0, 2.0, 2.0, 1.0]);

        let actions = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(
            restriction.transpose_reduce(&actions, 2, 2, 1, &[0]),
            global
        );
    }
}
