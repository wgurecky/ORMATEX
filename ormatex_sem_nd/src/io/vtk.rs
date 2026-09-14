//! VTK `.vtu` (XML UnstructuredGrid) export via `vtkio`.
//!
//! Requires the crate `vtk` feature. Points carry `z = 0`; every registered
//! field is written as point data.

use std::path::Path;

use faer::MatRef;
use ndelement::types::ReferenceCellType;
use ndmesh::traits::Mesh;
use vtkio::model::*;

use super::mesh::{ExportCell, ExportMesh, export_1d, export_2d};
use crate::sem_1d::SEM1DProblem;
use crate::sem_2d::SEM2DProblem;

/// Convert an [`ExportMesh`] into a `vtkio` XML UnstructuredGrid model.
pub fn export_mesh_to_vtk(mesh: &ExportMesh) -> Vtk {
    let mut points = Vec::with_capacity(mesh.points.len() * 3);
    for &[x, y, z] in &mesh.points {
        points.extend_from_slice(&[x, y, z]);
    }
    let mut connectivity = Vec::new();
    let mut offsets = Vec::new();
    let mut types = Vec::with_capacity(mesh.cells.len());
    for cell in &mesh.cells {
        connectivity.extend_from_slice(cell.connectivity());
        offsets.push(connectivity.len() as u64);
        types.push(match cell {
            ExportCell::Line(_) => CellType::Line,
            ExportCell::Quad(_) => CellType::Quad,
            ExportCell::LagrangeCurve(_) => CellType::LagrangeCurve,
            ExportCell::LagrangeQuad(_) => CellType::LagrangeQuadrilateral,
        });
    }
    let mut point_data = Vec::with_capacity(mesh.field_names.len());
    for (name, values) in mesh.field_names.iter().zip(&mesh.point_fields) {
        assert_eq!(values.len(), mesh.points.len(), "field length mismatch");
        point_data.push(Attribute::scalars(name.clone(), 1).with_data(values.clone()));
    }
    Vtk {
        version: Version::Auto,
        title: String::from("ormatex_sem_nd solution"),
        byte_order: ByteOrder::LittleEndian,
        file_path: None,
        data: DataSet::from(UnstructuredGridPiece {
            points: IOBuffer::F64(points),
            cells: Cells {
                cell_verts: VertexNumbers::XML {
                    connectivity,
                    offsets,
                },
                types,
            },
            data: Attributes {
                point: point_data,
                cell: Vec::new(),
            },
        }),
    }
}

/// Write an [`ExportMesh`] to a `.vtu` file (extension selects XML output).
pub fn write_vtu(path: impl AsRef<Path>, mesh: &ExportMesh) {
    export_mesh_to_vtk(mesh)
        .export(path.as_ref())
        .expect("failed to write vtu output");
}

/// Export a 1D solved state (all GLL nodes) directly to `.vtu`.
pub fn write_sem1d_vtu<M>(problem: &SEM1DProblem<M>, state: MatRef<'_, f64>, path: impl AsRef<Path>)
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>,
{
    write_vtu(path, &export_1d(problem, state));
}

/// Export a 2D solved state (all GLL nodes) directly to `.vtu`.
pub fn write_sem2d_vtu<M>(problem: &SEM2DProblem<M>, state: MatRef<'_, f64>, path: impl AsRef<Path>)
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>,
{
    write_vtu(path, &export_2d(problem, state));
}
