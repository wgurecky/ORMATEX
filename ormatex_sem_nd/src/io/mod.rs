//! Solution export: a shared [`ExportMesh`] intermediate plus CSV and VTK writers.
//!
//! `export_1d` / `export_2d` sample every registered field from a solved state
//! into an [`ExportMesh`]; the CSV writer is always available while the VTK
//! (`.vtu`) writer requires the `vtk` crate feature.
//!
//! Cells match the SEM order: linear `Line`/`Quad` for `p = 1`, Lagrange
//! curves/quadrilaterals of matching order for `p >= 2`, so the full GLL
//! accuracy reaches the visualization. (In ParaView, raise "Nonlinear
//! Subdivision Level" above 1 to see the curved interpolation; VTK evaluates
//! Lagrange cells on equispaced parametric nodes, so values are exact at the
//! GLL nodes and geometry is exact for straight-sided elements.)

pub mod csv;
pub mod mesh;
#[cfg(feature = "vtk")]
pub mod vtk;

pub use csv::{write_csv, write_csv_to};
pub use mesh::{ExportCell, ExportMesh, export_1d, export_2d, lagrange_quad_node_order};
#[cfg(feature = "vtk")]
pub use vtk::{write_sem1d_vtu, write_sem2d_vtu, write_vtu};
/// `vtkio` re-export for roundtrip tests and example `--vtk` handling.
#[cfg(feature = "vtk")]
pub use vtkio;

/// Resolve `--vtk [path]` / `--vtk=path` CLI export requests.
/// Bare `--vtk` uses `default_vtu`; absent flag yields `None` (no-op).
pub fn vtk_output_path(args: &[String], default_vtu: &str) -> Option<String> {
    if let Some(path) = args.iter().find_map(|arg| arg.strip_prefix("--vtk=")) {
        return Some(path.to_owned());
    }
    let pos = args.iter().position(|arg| arg == "--vtk")?;
    match args.get(pos + 1) {
        Some(next) if !next.starts_with("--") => Some(next.clone()),
        _ => Some(default_vtu.to_owned()),
    }
}
