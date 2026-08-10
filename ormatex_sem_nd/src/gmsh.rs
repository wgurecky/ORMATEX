//! Minimal MSH2 quadrilateral mesh loader for the Neumann/Robin example.

use std::collections::{HashMap, HashSet};
use std::fs;

use crate::material::{MeshMetadata, PhysicalRegion};
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{
    traits::{Builder, Entity, Geometry, Mesh, Point, Topology},
    SingleElementMesh, SingleElementMeshBuilder,
};

pub type QuadMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

fn triangle_element(element_type: usize) -> bool {
    matches!(element_type, 2 | 9 | 20..=25 | 42..=46)
}

pub struct GmshQuadData {
    pub mesh: QuadMesh,
    pub metadata: MeshMetadata,
}

/// Load a linear quadrilateral MSH2 mesh and its physical-region metadata.
pub fn gmsh_quad_data(path: &str) -> Result<GmshQuadData, String> {
    let source = fs::read_to_string(path).map_err(|err| format!("read {path}: {err}"))?;
    let mut nodes = Vec::new();
    let mut lines = Vec::new();
    let mut quads = Vec::new();
    let mut physical_names = HashMap::new();
    let mut input = source.lines();

    while let Some(section) = input.next() {
        match section.trim() {
            "$PhysicalNames" => {
                let count: usize = input
                    .next()
                    .ok_or("missing physical-name count")?
                    .parse()
                    .map_err(|_| "invalid physical-name count")?;
                for _ in 0..count {
                    let line = input.next().ok_or("truncated physical-name list")?;
                    let fields: Vec<_> = line.splitn(3, char::is_whitespace).collect();
                    if fields.len() != 3 {
                        return Err("invalid physical name header".into());
                    }
                    let dimension = fields[0]
                        .parse()
                        .map_err(|_| "invalid physical dimension")?;
                    let tag = fields[1].parse().map_err(|_| "invalid physical tag")?;
                    physical_names.insert(
                        PhysicalRegion { dimension, tag },
                        fields[2].trim().trim_matches('"').to_string(),
                    );
                }
                if input.next().map(str::trim) != Some("$EndPhysicalNames") {
                    return Err("missing $EndPhysicalNames".into());
                }
            }
            "$Nodes" => {
                let count: usize = input
                    .next()
                    .ok_or("missing node count")?
                    .parse()
                    .map_err(|_| "invalid node count")?;
                for _ in 0..count {
                    let fields: Vec<_> = input
                        .next()
                        .ok_or("truncated node list")?
                        .split_whitespace()
                        .collect();
                    if fields.len() != 4 {
                        return Err("MSH2 nodes must contain an id and three coordinates".into());
                    }
                    nodes.push((
                        fields[0].parse().map_err(|_| "invalid node id")?,
                        [
                            fields[1].parse().map_err(|_| "invalid node x coordinate")?,
                            fields[2].parse().map_err(|_| "invalid node y coordinate")?,
                        ],
                    ));
                }
                if input.next().map(str::trim) != Some("$EndNodes") {
                    return Err("missing $EndNodes".into());
                }
            }
            "$Elements" => {
                let count: usize = input
                    .next()
                    .ok_or("missing element count")?
                    .parse()
                    .map_err(|_| "invalid element count")?;
                for _ in 0..count {
                    let fields: Vec<_> = input
                        .next()
                        .ok_or("truncated element list")?
                        .split_whitespace()
                        .collect();
                    if fields.len() < 3 {
                        return Err("invalid MSH2 element".into());
                    }
                    let element_type: usize =
                        fields[1].parse().map_err(|_| "invalid element type")?;
                    if triangle_element(element_type) {
                        return Err(format!(
                            "triangle element type {element_type} found; only quadrilaterals are supported"
                        ));
                    }
                    let tag_count: usize =
                        fields[2].parse().map_err(|_| "invalid element tag count")?;
                    let nodes_start = 3 + tag_count;
                    if fields.len() < nodes_start {
                        return Err("truncated element tags".into());
                    }
                    let physical_tag = fields[3..nodes_start]
                        .first()
                        .map(|tag| tag.parse())
                        .transpose()
                        .map_err(|_| "invalid physical tag")?;
                    let element_nodes: Result<Vec<usize>, _> = fields[nodes_start..]
                        .iter()
                        .map(|node| node.parse())
                        .collect();
                    let element_nodes = element_nodes.map_err(|_| "invalid element node id")?;
                    match element_type {
                        1 => {
                            if element_nodes.len() != 2 {
                                return Err("only linear boundary lines are supported".into());
                            }
                            lines.push((
                                physical_tag.ok_or("boundary line has no physical tag")?,
                                [element_nodes[0], element_nodes[1]],
                            ));
                        }
                        3 => {
                            if element_nodes.len() != 4 {
                                return Err("only linear quadrilaterals are supported".into());
                            }
                            quads.push((
                                physical_tag.map(|tag| PhysicalRegion { dimension: 2, tag }),
                                [
                                    element_nodes[0],
                                    element_nodes[1],
                                    element_nodes[2],
                                    element_nodes[3],
                                ],
                            ));
                        }
                        15 => {}
                        _ => return Err(format!("unsupported Gmsh element type {element_type}")),
                    }
                }
                if input.next().map(str::trim) != Some("$EndElements") {
                    return Err("missing $EndElements".into());
                }
            }
            _ => {}
        }
    }

    if nodes.is_empty() || quads.is_empty() {
        return Err("mesh must contain nodes and quadrilateral cells".into());
    }
    let mut node_indices = HashMap::new();
    let mut builder =
        SingleElementMeshBuilder::<f64>::new(2, (ReferenceCellType::Quadrilateral, 1));
    for (index, (id, xy)) in nodes.iter().enumerate() {
        if node_indices.insert(*id, index).is_some() {
            return Err(format!("duplicate node id {id}"));
        }
        builder.add_point(*id, xy);
    }
    let mut cell_regions = Vec::with_capacity(quads.len());
    for (index, (region, quad)) in quads.iter().enumerate() {
        if !quad.iter().all(|node| node_indices.contains_key(node)) {
            return Err(format!("quadrilateral {index} references an unknown node"));
        }
        // ndmesh orders quad vertices [bottom-left, bottom-right, top-left,
        // top-right], while Gmsh uses [bottom-left, bottom-right, top-right,
        // top-left].
        builder.add_cell(index + 1, &[quad[0], quad[1], quad[3], quad[2]]);
        cell_regions.push(*region);
    }
    let mesh = builder.create_mesh();

    let mut line_tags = HashMap::new();
    for (tag, endpoints) in lines {
        let a = *node_indices
            .get(&endpoints[0])
            .ok_or("boundary line references an unknown node")?;
        let b = *node_indices
            .get(&endpoints[1])
            .ok_or("boundary line references an unknown node")?;
        let edge = if a < b { (a, b) } else { (b, a) };
        if line_tags
            .insert(edge, PhysicalRegion { dimension: 1, tag })
            .is_some()
        {
            return Err("duplicate boundary line".into());
        }
    }
    let mut facet_tags = vec![None; mesh.entity_count(ReferenceCellType::Interval)];
    let mut matched = HashSet::new();
    for facet in mesh.entity_iter(ReferenceCellType::Interval) {
        let facet_index = facet.local_index();
        let geometry = facet.geometry();
        let mut points = geometry.points();
        let a = points
            .next()
            .ok_or("interval has no first endpoint")?
            .index();
        let b = points
            .next()
            .ok_or("interval has no second endpoint")?
            .index();
        if points.next().is_some() {
            return Err("only linear interval facets are supported".into());
        }
        let edge = if a < b { (a, b) } else { (b, a) };
        if let Some(&region) = line_tags.get(&edge) {
            let topology = facet.topology();
            let mut cells = topology.connected_entity_iter(ReferenceCellType::Quadrilateral);
            if cells.next().is_none() || cells.next().is_some() {
                return Err(
                    "a tagged line must match exactly one quadrilateral boundary facet".into(),
                );
            }
            facet_tags[facet_index] = Some(region);
            matched.insert(edge);
        }
    }
    if matched.len() != line_tags.len() {
        return Err("a tagged Gmsh boundary line did not match an ndmesh facet".into());
    }
    Ok(GmshQuadData {
        mesh,
        metadata: MeshMetadata {
            cell_regions,
            facet_regions: facet_tags,
            physical_names,
        },
    })
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
}
