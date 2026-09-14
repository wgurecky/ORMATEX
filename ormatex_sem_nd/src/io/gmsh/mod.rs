//! Gmsh MSH2 quadrilateral mesh import (ASCII and binary).
//!
//! [`gmsh_quad_data`] auto-detects the file flavor from its `$MeshFormat`
//! header (`2.x 0 8` ASCII, `2.x 1 8` binary) and returns the ndmesh
//! quadrilateral mesh plus physical-region metadata: Physical Curve (dim 1)
//! tags for boundary-condition assignment and Physical Surface (dim 2) tags
//! for material regions, each addressable by number or by name.

mod ascii;
mod binary;
pub mod model;

pub use model::{GmshQuadData, QuadMesh, RawLine, RawMesh, RawNode, RawQuad};

use std::fs;

/// Load a linear quadrilateral MSH2 mesh and its physical-region metadata.
///
/// Accepts both ASCII (`2.x 0 8`) and binary (`2.x 1 8`) files.
pub fn gmsh_quad_data(path: &str) -> Result<GmshQuadData, String> {
    let bytes = fs::read(path).map_err(|err| format!("read {path}: {err}"))?;
    raw_mesh(&bytes)?.build()
}

/// Parse an MSH2 file (ASCII or binary, auto-detected) into format-agnostic records.
pub fn raw_mesh(bytes: &[u8]) -> Result<RawMesh, String> {
    match mesh_flavor(bytes)? {
        Flavor::Ascii => ascii::parse_ascii(
            std::str::from_utf8(bytes).map_err(|_| "ASCII mesh is not valid UTF-8")?,
        ),
        Flavor::Binary => binary::parse_binary(bytes),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Flavor {
    Ascii,
    Binary,
}

/// Detect the MSH2 flavor from the `$MeshFormat` header lines.
fn mesh_flavor(bytes: &[u8]) -> Result<Flavor, String> {
    let mut lines = bytes.split(|&b| b == b'\n');
    let first = lines.next().ok_or("missing $MeshFormat section")?;
    if first.strip_suffix(b"\r").unwrap_or(first) != b"$MeshFormat" {
        return Err("missing $MeshFormat section".into());
    }
    let header = lines.next().ok_or("missing mesh-format header")?;
    let header =
        std::str::from_utf8(header).map_err(|_| "mesh-format header is not ASCII")?;
    let fields: Vec<_> = header.split_whitespace().collect();
    if fields.len() != 3 || !fields[0].starts_with("2.") || fields[2] != "8" {
        return Err("only MSH2 meshes (2.x, 8-byte) are supported".into());
    }
    match fields[1] {
        "0" => Ok(Flavor::Ascii),
        "1" => Ok(Flavor::Binary),
        _ => Err("only MSH2 meshes (2.x, 8-byte) are supported".into()),
    }
}

/// Compatibility loader returning only boundary numeric tags.
pub fn gmsh_quad_mesh(path: &str) -> Result<(QuadMesh, Vec<Option<usize>>), String> {
    let data = gmsh_quad_data(path)?;
    let tags = data
        .metadata
        .facet_regions
        .iter()
        .map(|region| region.map(|region| region.tag))
        .collect();
    Ok((data.mesh, tags))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regions::PhysicalRegion;
    use ndelement::types::ReferenceCellType;
    use ndmesh::traits::Mesh;

    #[test]
    fn rejects_triangle_elements() {
        let path =
            std::env::temp_dir().join(format!("ormatex_triangle_{}.msh", std::process::id()));
        fs::write(
            &path,
            "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n$Elements\n1\n1 2 2 1 1 1 2 3\n$EndElements\n",
        )
        .unwrap();
        let error = gmsh_quad_mesh(path.to_str().unwrap()).unwrap_err();
        fs::remove_file(path).unwrap();
        assert!(error.contains("triangle element"));
    }

    #[test]
    fn rejects_binary_triangle_elements() {
        let mut bytes = b"$MeshFormat\n2.2 1 8\n".to_vec();
        bytes.extend_from_slice(&1i32.to_le_bytes());
        bytes.extend_from_slice(b"\n$EndMeshFormat\n$Nodes\n1\n");
        bytes.extend_from_slice(&1i32.to_le_bytes());
        bytes.extend_from_slice(&0f64.to_le_bytes());
        bytes.extend_from_slice(&0f64.to_le_bytes());
        bytes.extend_from_slice(&0f64.to_le_bytes());
        bytes.extend_from_slice(b"\n$EndNodes\n$Elements\n1\n");
        // One group: triangle type, one element, two tags.
        for value in [2i32, 1, 2] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [1i32, 1, 1, 1, 2, 3] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(b"\n$EndElements\n");
        let error = raw_mesh(&bytes).unwrap_err();
        assert!(error.contains("triangle element"), "got: {error}");
    }

    #[test]
    fn binary_single_quad_matches_ascii() {
        let ascii = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n2\n1 7 \"wall\"\n2 10 \"domain\"\n$EndPhysicalNames\n$Nodes\n4\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n$EndNodes\n$Elements\n5\n1 1 2 7 8 1 2\n2 1 2 7 8 2 3\n3 1 2 7 8 3 4\n4 1 2 7 8 4 1\n5 3 2 10 11 1 2 3 4\n$EndElements\n";
        let from_ascii = raw_mesh(ascii.as_bytes()).unwrap().build().unwrap();
        let mut bytes = b"$MeshFormat\n2.2 1 8\n".to_vec();
        bytes.extend_from_slice(&1i32.to_le_bytes());
        bytes.extend_from_slice(b"\n$EndMeshFormat\n$PhysicalNames\n2\n1 7 \"wall\"\n2 10 \"domain\"\n$EndPhysicalNames\n$Nodes\n4\n");
        for (id, x, y) in [(1i32, 0.0f64, 0.0f64), (2, 1.0, 0.0), (3, 1.0, 1.0), (4, 0.0, 1.0)] {
            bytes.extend_from_slice(&id.to_le_bytes());
            bytes.extend_from_slice(&x.to_le_bytes());
            bytes.extend_from_slice(&y.to_le_bytes());
            bytes.extend_from_slice(&0f64.to_le_bytes());
        }
        bytes.extend_from_slice(b"\n$EndNodes\n$Elements\n5\n");
        // Line group: type 1, four elements, two tags each.
        for value in [1i32, 4, 2] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for (id, nodes) in [(1i32, [1i32, 2]), (2, [2, 3]), (3, [3, 4]), (4, [4, 1])] {
            bytes.extend_from_slice(&id.to_le_bytes());
            for tag in [7i32, 8] {
                bytes.extend_from_slice(&tag.to_le_bytes());
            }
            for node in nodes {
                bytes.extend_from_slice(&node.to_le_bytes());
            }
        }
        // Quad group: type 3, one element, two tags.
        for value in [3i32, 1, 2] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [5i32, 10, 11, 1, 2, 3, 4] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(b"\n$EndElements\n");
        let from_binary = raw_mesh(&bytes).unwrap().build().unwrap();
        for data in [&from_ascii, &from_binary] {
            assert_eq!(
                data.mesh.entity_count(ReferenceCellType::Quadrilateral),
                1
            );
            assert_eq!(data.metadata.cell_indices("domain").len(), 1);
            assert_eq!(data.metadata.boundary_facets("wall").len(), 4);
        }
    }

    #[test]
    fn preserves_cell_facet_regions_and_names() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/ex_nd_2d_diffusion_neumann_gmsh.msh"
        );
        let data = gmsh_quad_data(path).unwrap();
        assert_eq!(
            data.metadata.cell_regions.len(),
            data.mesh.entity_count(ReferenceCellType::Quadrilateral)
        );
        assert!(data.metadata.cell_regions.iter().all(|region| {
            *region
                == Some(PhysicalRegion {
                    dimension: 2,
                    tag: 10,
                })
        }));
        assert_eq!(
            data.metadata.physical_names[&PhysicalRegion {
                dimension: 2,
                tag: 10
            }],
            "domain"
        );
        assert!(data.metadata.facet_regions.iter().any(|region| {
            *region
                == Some(PhysicalRegion {
                    dimension: 1,
                    tag: 1,
                })
        }));
    }

    #[test]
    fn cylinder_binary_loads_with_named_curves_and_surface() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/navier-stokes/cylinder.msh"
        );
        let data = gmsh_quad_data(path).unwrap();
        assert_eq!(
            data.mesh.entity_count(ReferenceCellType::Quadrilateral),
            1680
        );
        assert_eq!(
            data.metadata.physical_names[&PhysicalRegion {
                dimension: 1,
                tag: 5,
            }],
            "cylinder"
        );
        for name in ["inlet", "outlet", "bottom", "top", "cylinder"] {
            assert!(
                !data.metadata.boundary_facets(name).is_empty(),
                "missing Physical Curve {name}"
            );
        }
        assert!(!data.metadata.cell_indices("domain").is_empty());
    }

    #[test]
    fn backward_step_ascii_loads_with_named_curves_and_surface() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/navier-stokes/backward_step.msh"
        );
        let data = gmsh_quad_data(path).unwrap();
        assert!(
            data.mesh.entity_count(ReferenceCellType::Quadrilateral) > 0
        );
        for name in ["inlet", "outlet", "wall"] {
            assert!(
                !data.metadata.boundary_facets(name).is_empty(),
                "missing Physical Curve {name}"
            );
        }
        assert!(!data.metadata.cell_indices("domain").is_empty());
    }
}
