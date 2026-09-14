//! Shared MSH2 import structures: format-agnostic records plus the
//! `SingleElementMesh` builder used by both the ASCII and binary parsers.
//!
//! Gmsh terminology used here: a *Physical Curve* (dimension 1) labels boundary
//! lines for boundary-condition assignment, a *Physical Surface* (dimension 2)
//! labels quadrilateral cells for material-region assignment.

use std::collections::{HashMap, HashSet};

use crate::regions::{MeshMetadata, PhysicalRegion};
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{
    traits::{Builder, Entity, Geometry, Mesh, Point, Topology},
    SingleElementMesh, SingleElementMeshBuilder,
};

pub type QuadMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

pub struct GmshQuadData {
    pub mesh: QuadMesh,
    pub metadata: MeshMetadata,
}

/// One MSH2 node: Gmsh id plus its `x, y` position (`z` must be zero).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawNode {
    pub id: usize,
    pub xy: [f64; 2],
}

/// One MSH2 line element: mandatory Physical Curve tag plus endpoint node ids.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RawLine {
    pub tag: usize,
    pub nodes: [usize; 2],
}

/// One MSH2 quadrilateral element: optional Physical Surface region plus node ids.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RawQuad {
    pub region: Option<PhysicalRegion>,
    pub nodes: [usize; 4],
}

/// Format-agnostic MSH2 content: everything both parsers extract before building.
#[derive(Clone, Debug, Default)]
pub struct RawMesh {
    pub nodes: Vec<RawNode>,
    pub lines: Vec<RawLine>,
    pub quads: Vec<RawQuad>,
    pub physical_names: HashMap<PhysicalRegion, String>,
}

/// Parse one `$PhysicalNames` entry line: `<dim> <tag> "<name>"`.
pub(crate) fn parse_physical_name(line: &str) -> Result<(PhysicalRegion, String), String> {
    let fields: Vec<_> = line.splitn(3, char::is_whitespace).collect();
    if fields.len() != 3 {
        return Err("invalid physical name entry".into());
    }
    let dimension = fields[0]
        .parse()
        .map_err(|_| "invalid physical dimension")?;
    let tag = fields[1].parse().map_err(|_| "invalid physical tag")?;
    Ok((
        PhysicalRegion { dimension, tag },
        fields[2].trim().trim_matches('"').to_string(),
    ))
}

fn triangle_element(element_type: usize) -> bool {
    matches!(element_type, 2 | 9 | 20..=25 | 42..=46)
}

/// Classify one MSH2 element record shared by both parsers.
///
/// `tags[0]` is the physical tag (required for lines, optional for quads).
/// Returns `None` for point elements, which carry no mesh data.
pub(crate) fn classify_element(
    element_type: usize,
    tags: &[usize],
    nodes: &[usize],
) -> Result<Option<RawElement>, String> {
    if triangle_element(element_type) {
        return Err(format!(
            "triangle element type {element_type} found; only quadrilaterals are supported"
        ));
    }
    let physical_tag = tags.first().copied().filter(|tag| *tag != 0);
    match element_type {
        1 => {
            if nodes.len() != 2 {
                return Err("only linear boundary lines are supported".into());
            }
            Ok(Some(RawElement::Line(RawLine {
                tag: physical_tag.ok_or("boundary line has no physical tag")?,
                nodes: [nodes[0], nodes[1]],
            })))
        }
        3 => {
            if nodes.len() != 4 {
                return Err("only linear quadrilaterals are supported".into());
            }
            Ok(Some(RawElement::Quad(RawQuad {
                region: physical_tag.map(|tag| PhysicalRegion { dimension: 2, tag }),
                nodes: [nodes[0], nodes[1], nodes[2], nodes[3]],
            })))
        }
        15 => Ok(None),
        _ => Err(format!("unsupported Gmsh element type {element_type}")),
    }
}

/// Classified MSH2 element record.
pub(crate) enum RawElement {
    Line(RawLine),
    Quad(RawQuad),
}

impl RawMesh {
    /// Push one classified element record.
    pub(crate) fn push(&mut self, element: Option<RawElement>) {
        match element {
            Some(RawElement::Line(line)) => self.lines.push(line),
            Some(RawElement::Quad(quad)) => self.quads.push(quad),
            None => {}
        }
    }

    /// Build the ndmesh quadrilateral mesh plus physical-region metadata.
    pub fn build(self) -> Result<GmshQuadData, String> {
        if self.nodes.is_empty() || self.quads.is_empty() {
            return Err("mesh must contain nodes and quadrilateral cells".into());
        }
        let mut node_indices = HashMap::new();
        let mut builder =
            SingleElementMeshBuilder::<f64>::new(2, (ReferenceCellType::Quadrilateral, 1));
        for (index, node) in self.nodes.iter().enumerate() {
            if node_indices.insert(node.id, index).is_some() {
                return Err(format!("duplicate node id {}", node.id));
            }
            builder.add_point(node.id, &node.xy);
        }
        let mut cell_regions = Vec::with_capacity(self.quads.len());
        for (index, quad) in self.quads.iter().enumerate() {
            if !quad.nodes.iter().all(|node| node_indices.contains_key(node)) {
                return Err(format!("quadrilateral {index} references an unknown node"));
            }
            // ndmesh orders quad vertices [bottom-left, bottom-right, top-left,
            // top-right], while Gmsh uses [bottom-left, bottom-right, top-right,
            // top-left].
            builder.add_cell(
                index + 1,
                &[quad.nodes[0], quad.nodes[1], quad.nodes[3], quad.nodes[2]],
            );
            cell_regions.push(quad.region);
        }
        let mesh = builder.create_mesh();

        let mut line_tags = HashMap::new();
        for line in self.lines {
            let a = *node_indices
                .get(&line.nodes[0])
                .ok_or("boundary line references an unknown node")?;
            let b = *node_indices
                .get(&line.nodes[1])
                .ok_or("boundary line references an unknown node")?;
            let edge = if a < b { (a, b) } else { (b, a) };
            if line_tags
                .insert(edge, PhysicalRegion {
                    dimension: 1,
                    tag: line.tag,
                })
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
                physical_names: self.physical_names,
            },
        })
    }
}
