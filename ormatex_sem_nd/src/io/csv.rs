//! Standardized single-snapshot CSV export: `x,(y),field_0,...,field_N`.
//!
//! Rows are sorted by position so output is deterministic across runs.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use super::mesh::ExportMesh;

/// Write `mesh` as CSV to any writer: `x,(y),field_0,...` with one row per point.
pub fn write_csv_to(mut writer: impl Write, mesh: &ExportMesh) {
    assert!(!mesh.field_names.is_empty(), "export mesh has no fields");
    if mesh.dim == 1 {
        write!(writer, "x").unwrap();
    } else {
        write!(writer, "x,y").unwrap();
    }
    for name in &mesh.field_names {
        write!(writer, ",{name}").unwrap();
    }
    writeln!(writer).unwrap();

    let mut order: Vec<usize> = (0..mesh.points.len()).collect();
    order.sort_by(|&a, &b| {
        mesh.points[a]
            .partial_cmp(&mesh.points[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for &i in &order {
        let [x, y, _] = mesh.points[i];
        if mesh.dim == 1 {
            write!(writer, "{x:.9}").unwrap();
        } else {
            write!(writer, "{x:.9},{y:.9}").unwrap();
        }
        for field in &mesh.point_fields {
            write!(writer, ",{:.9e}", field[i]).unwrap();
        }
        writeln!(writer).unwrap();
    }
}

/// Write `mesh` as CSV to `path`.
pub fn write_csv(path: impl AsRef<Path>, mesh: &ExportMesh) {
    let output = BufWriter::new(File::create(path.as_ref()).expect("failed to create csv output"));
    write_csv_to(output, mesh);
}
