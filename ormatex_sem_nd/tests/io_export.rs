//! VTK/CSV solution-export tests: extraction checks plus CSV cross-checks.
//!
//! The VTK roundtrip test requires the `vtk` feature; CSV and extraction run always.

use faer::prelude::*;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{
    shapes::{unit_interval, unit_square},
    SingleElementMesh,
};
use ormatex_sem_nd::{
    io::{ExportCell, export_1d, export_2d, lagrange_quad_node_order, vtk_output_path, write_csv_to},
    DofReduction1D, DofReduction2D, FieldRegistry, SEM1DProblem, SEM2DProblem,
};

type Mesh1D = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;
type Mesh2D = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

fn analytic_1d(x: f64) -> f64 {
    x * x + 1.0
}

fn problem_1d_degree(p: usize) -> (SEM1DProblem<Mesh1D>, Mat<f64>) {
    let problem = SEM1DProblem::new(
        unit_interval(2, 1),
        p,
        FieldRegistry::new(["u"]),
        DofReduction1D::None,
    );
    let xs = problem.dof_positions();
    let mut state = Mat::<f64>::zeros(problem.system_size(), 1);
    for (i, x) in xs.iter().enumerate() {
        state[(i, 0)] = analytic_1d(*x);
    }
    (problem, state)
}

fn problem_1d() -> (SEM1DProblem<Mesh1D>, Mat<f64>) {
    problem_1d_degree(2)
}

#[test]
fn lagrange_quad_node_order_matches_vtk() {
    // Corner-first, then bottom/right/top/left edges, then row-major interior,
    // mirroring vtkHigherOrderQuadrilateral::PointIndexFromIJK.
    assert_eq!(
        lagrange_quad_node_order(1),
        vec![(0, 0), (1, 0), (1, 1), (0, 1)]
    );
    assert_eq!(
        lagrange_quad_node_order(2),
        vec![
            (0, 0),
            (2, 0),
            (2, 2),
            (0, 2),
            (1, 0),
            (2, 1),
            (1, 2),
            (0, 1),
            (1, 1),
        ]
    );
    assert_eq!(
        lagrange_quad_node_order(3),
        vec![
            (0, 0),
            (3, 0),
            (3, 3),
            (0, 3),
            (1, 0),
            (2, 0),
            (3, 1),
            (3, 2),
            (1, 3),
            (2, 3),
            (0, 1),
            (0, 2),
            (1, 1),
            (2, 1),
            (1, 2),
            (2, 2),
        ]
    );
}

#[test]
fn export_1d_linear_for_p1() {
    let (problem, state) = problem_1d_degree(1);
    let mesh = export_1d(&problem, state.as_ref());
    // 2 linear elements share one vertex: 3 points, one `Line` each.
    assert_eq!(mesh.point_count(), 3);
    assert_eq!(mesh.cell_count(), 2);
    for cell in &mesh.cells {
        assert!(matches!(cell, ExportCell::Line(_)));
    }
}

#[test]
fn export_1d_samples_all_gll_nodes() {
    let (problem, state) = problem_1d();
    let mesh = export_1d(&problem, state.as_ref());
    // 2 elements of degree 2 share one vertex: 5 unique nodes, one
    // order-2 Lagrange curve per element.
    assert_eq!(mesh.dim, 1);
    assert_eq!(mesh.field_names, vec!["u".to_string()]);
    assert_eq!(mesh.point_count(), 5);
    assert_eq!(mesh.cell_count(), 2);
    for cell in &mesh.cells {
        match cell {
            ExportCell::LagrangeCurve(points) => assert_eq!(points.len(), 3),
            other => panic!("expected Lagrange curve, got {other:?}"),
        }
    }
    for (i, &[x, y, z]) in mesh.points.iter().enumerate() {
        assert_eq!((y, z), (0.0, 0.0));
        assert!((mesh.point_fields[0][i] - analytic_1d(x)).abs() < 1e-12);
    }
    let mut buf = Vec::new();
    write_csv_to(&mut buf, &mesh);
    let text = String::from_utf8(buf).unwrap();
    assert!(text.starts_with("x,u\n"), "unexpected header: {text}");
    assert_eq!(text.lines().count(), mesh.point_count() + 1);
}

#[test]
fn export_1d_includes_dirichlet_values() {
    let problem = SEM1DProblem::new(
        unit_interval(2, 1),
        2,
        FieldRegistry::new(["u"]),
        DofReduction1D::Dirichlet {
            facets: vec![(0, 3.0)],
        },
    );
    let mut state = Mat::<f64>::zeros(problem.system_size(), 1);
    for (i, x) in problem.dof_positions().iter().enumerate() {
        state[(i, 0)] = analytic_1d(*x);
    }
    let mesh = export_1d(&problem, state.as_ref());
    // The eliminated x=0 endpoint is still exported with its prescribed value.
    assert_eq!(mesh.point_count(), 5);
    let zero = mesh
        .points
        .iter()
        .position(|&[x, _, _]| x == 0.0)
        .expect("missing x=0 export point");
    assert!((mesh.point_fields[0][zero] - 3.0).abs() < 1e-12);
}

fn problem_2d_degree(p: usize) -> (SEM2DProblem<Mesh2D>, Mat<f64>) {
    let problem = SEM2DProblem::new(
        unit_square(2, 1, ReferenceCellType::Quadrilateral, 1),
        p,
        FieldRegistry::new(["u", "v"]),
        DofReduction2D::None,
    );
    let mut state = Mat::<f64>::zeros(problem.system_size(), 1);
    for (i, &(x, y)) in problem.field_dof_positions(0).iter().enumerate() {
        state[(problem.field_offset(0) + i, 0)] = x + 2.0 * y;
    }
    for (i, &(x, y)) in problem.field_dof_positions(1).iter().enumerate() {
        state[(problem.field_offset(1) + i, 0)] = x * y;
    }
    (problem, state)
}

fn problem_2d() -> (SEM2DProblem<Mesh2D>, Mat<f64>) {
    problem_2d_degree(2)
}

#[test]
fn export_2d_linear_for_p1() {
    let (problem, state) = problem_2d_degree(1);
    let mesh = export_2d(&problem, state.as_ref());
    // 2x1 linear elements: one `Quad` each, (2+1) x (1+1) corner points.
    assert_eq!(mesh.cell_count(), 2);
    assert_eq!(mesh.point_count(), 6);
    for cell in &mesh.cells {
        assert!(matches!(cell, ExportCell::Quad(_)));
    }
}

#[test]
fn export_2d_writes_lagrange_quads() {
    let (problem, state) = problem_2d();
    let mesh = export_2d(&problem, state.as_ref());
    // 2x1 elements of degree 2: one order-2 Lagrange quad each over the full
    // (2*2+1) x (1*2+1) GLL grid.
    assert_eq!(mesh.dim, 2);
    assert_eq!(mesh.field_names, vec!["u".to_string(), "v".to_string()]);
    assert_eq!(mesh.cell_count(), 2);
    assert_eq!(mesh.point_count(), 15);
    for cell in &mesh.cells {
        match cell {
            ExportCell::LagrangeQuad(points) => assert_eq!(points.len(), 9),
            other => panic!("expected Lagrange quad, got {other:?}"),
        }
    }
    for (i, &[x, y, z]) in mesh.points.iter().enumerate() {
        assert_eq!(z, 0.0);
        assert!((mesh.point_fields[0][i] - (x + 2.0 * y)).abs() < 1e-12);
        assert!((mesh.point_fields[1][i] - x * y).abs() < 1e-12);
    }
    let mut buf = Vec::new();
    write_csv_to(&mut buf, &mesh);
    let text = String::from_utf8(buf).unwrap();
    assert!(text.starts_with("x,y,u,v\n"), "unexpected header: {text}");
    assert_eq!(text.lines().count(), mesh.point_count() + 1);
}

#[test]
fn export_2d_order3_lagrange_quad() {
    let problem = SEM2DProblem::new(
        unit_square(1, 1, ReferenceCellType::Quadrilateral, 1),
        3,
        FieldRegistry::new(["u"]),
        DofReduction2D::None,
    );
    let mut state = Mat::<f64>::zeros(problem.system_size(), 1);
    for (i, &(x, y)) in problem.field_dof_positions(0).iter().enumerate() {
        state[(i, 0)] = x + 2.0 * y;
    }
    let mesh = export_2d(&problem, state.as_ref());
    assert_eq!(mesh.cell_count(), 1);
    assert_eq!(mesh.point_count(), 16);
    match &mesh.cells[0] {
        ExportCell::LagrangeQuad(points) => assert_eq!(points.len(), 16),
        other => panic!("expected order-3 Lagrange quad, got {other:?}"),
    }
    // Edge midpoints and interior nodes carry exact nodal values, which
    // vertex-only export would have missed.
    for (i, &[x, y, _]) in mesh.points.iter().enumerate() {
        assert!((mesh.point_fields[0][i] - (x + 2.0 * y)).abs() < 1e-12);
    }
}

#[test]
fn append_fields_joins_second_problem() {
    let (problem, state) = problem_2d();
    let mut mesh = export_2d(&problem, state.as_ref());
    let species = SEM2DProblem::new(
        unit_square(2, 1, ReferenceCellType::Quadrilateral, 1),
        2,
        FieldRegistry::new(["c0"]),
        DofReduction2D::None,
    );
    let mut species_state = Mat::<f64>::zeros(species.system_size(), 1);
    for (i, &(x, y)) in species.field_dof_positions(0).iter().enumerate() {
        species_state[(i, 0)] = x - y;
    }
    mesh.append_fields(&export_2d(&species, species_state.as_ref()));
    assert_eq!(mesh.field_names, vec!["u", "v", "c0"]);
    assert_eq!(mesh.point_count(), 15);
    for (i, &[x, y, _]) in mesh.points.iter().enumerate() {
        assert!((mesh.point_fields[2][i] - (x - y)).abs() < 1e-12);
    }
    let mut buf = Vec::new();
    write_csv_to(&mut buf, &mesh);
    assert!(String::from_utf8(buf).unwrap().starts_with("x,y,u,v,c0\n"));
}

#[test]
fn vtk_flag_parsing() {
    let args = |words: &[&str]| words.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    assert_eq!(vtk_output_path(&args(&["prog"]), "d.vtu"), None);
    assert_eq!(
        vtk_output_path(&args(&["prog", "--vtk"]), "d.vtu"),
        Some("d.vtu".to_string())
    );
    assert_eq!(
        vtk_output_path(&args(&["prog", "--vtk", "out.vtu"]), "d.vtu"),
        Some("out.vtu".to_string())
    );
    assert_eq!(
        vtk_output_path(&args(&["prog", "--vtk=out.vtu"]), "d.vtu"),
        Some("out.vtu".to_string())
    );
    // A following flag is not consumed as the path.
    assert_eq!(
        vtk_output_path(&args(&["prog", "--vtk", "--steps", "3"]), "d.vtu"),
        Some("d.vtu".to_string())
    );
}

/// Parse our `x,(y),fields...` CSV back into `(coords, values)` rows.
fn parse_csv(text: &str, dim: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut rows = Vec::new();
    for line in text.lines().skip(1) {
        let nums: Vec<f64> = line.split(',').map(|s| s.parse().unwrap()).collect();
        rows.push((nums[..dim].to_vec(), nums[dim..].to_vec()));
    }
    rows
}

#[cfg(feature = "vtk")]
#[test]
fn vtu_roundtrip_matches_csv() {
    use ormatex_sem_nd::io::{vtkio::model::*, write_vtu, ExportMesh};
    use std::collections::HashMap;

    // (tag, dim, expected cell type, expected points, expected cells)
    let cases: Vec<(&str, usize, CellType, usize, usize, ExportMesh)> = vec![
        {
            let (problem, state) = problem_1d_degree(1);
            ("io_export_1d_p1", 1, CellType::Line, 3, 2, export_1d(&problem, state.as_ref()))
        },
        {
            let (problem, state) = problem_1d();
            ("io_export_1d_p2", 1, CellType::LagrangeCurve, 5, 2, export_1d(&problem, state.as_ref()))
        },
        {
            let (problem, state) = problem_2d_degree(1);
            ("io_export_2d_p1", 2, CellType::Quad, 6, 2, export_2d(&problem, state.as_ref()))
        },
        {
            let (problem, state) = problem_2d();
            ("io_export_2d_p2", 2, CellType::LagrangeQuadrilateral, 15, 2, export_2d(&problem, state.as_ref()))
        },
    ];
    for (tag, dim, expected_type, npoints, ncells, mesh) in &cases {
        assert_eq!(mesh.cell_count(), *ncells);
        assert_eq!(mesh.point_count(), *npoints);

        let path = std::env::temp_dir().join(format!("ormatex-{tag}-{}.vtu", std::process::id()));
        write_vtu(&path, mesh);
        let vtk = Vtk::import(&path).expect("failed to re-import exported vtu");
        std::fs::remove_file(&path).unwrap();
        let piece = UnstructuredGridPiece::try_from(vtk.data).expect("expected unstructured grid");

        // Points agree with the export mesh.
        let coords = match &piece.points {
            IOBuffer::F64(coords) => coords.clone(),
            other => panic!("expected f64 points, got {other:?}"),
        };
        assert_eq!(coords.len(), mesh.points.len() * 3);
        for (i, &[x, y, z]) in mesh.points.iter().enumerate() {
            assert!((coords[3 * i] - x).abs() < 1e-12);
            assert!((coords[3 * i + 1] - y).abs() < 1e-12);
            assert!((coords[3 * i + 2] - z).abs() < 1e-12);
        }
        // Cell types match the SEM order (linear for p=1, Lagrange above).
        assert_eq!(piece.cells.num_cells(), mesh.cell_count());
        for ty in &piece.cells.types {
            assert_eq!(*ty, *expected_type);
        }
        // Point data matches the export mesh field by field.
        assert_eq!(piece.data.point.len(), mesh.field_names.len());
        let mut vtk_fields = HashMap::new();
        for attr in &piece.data.point {
            let array = match attr {
                Attribute::DataArray(array) => array,
                other => panic!("expected data array, got {other:?}"),
            };
            let values = match &array.data {
                IOBuffer::F64(values) => values.clone(),
                other => panic!("expected f64 field data, got {other:?}"),
            };
            assert_eq!(values.len(), mesh.point_count());
            vtk_fields.insert(array.name.clone(), values);
        }
        for (name, values) in mesh.field_names.iter().zip(&mesh.point_fields) {
            assert_eq!(&vtk_fields[name], values);
        }

        // CSV agrees with VTK point data at every coordinate (CSV is 9-digit rounded).
        let mut buf = Vec::new();
        write_csv_to(&mut buf, mesh);
        let rows = parse_csv(&String::from_utf8(buf).unwrap(), *dim);
        assert_eq!(rows.len(), mesh.point_count());
        for (coords, values) in rows {
            let id = mesh
                .points
                .iter()
                .position(|p| {
                    (0..*dim)
                        .all(|d| (p[d] - coords[d]).abs() < 1e-9)
                })
                .expect("csv row without vtk point");
            for (f, &value) in values.iter().enumerate() {
                assert!((value - mesh.point_fields[f][id]).abs() < 1e-6);
            }
        }
    }
}
