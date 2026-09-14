//! ASCII MSH2 (`2.x 0 8`) section parser producing a [`RawMesh`](super::model::RawMesh).

use super::model::{classify_element, parse_physical_name, RawMesh, RawNode};

/// Parse one `$Nodes` entry line: `<id> <x> <y> <z>` (`z` is ignored).
fn parse_node(line: &str) -> Result<RawNode, String> {
    let fields: Vec<_> = line.split_whitespace().collect();
    if fields.len() != 4 {
        return Err("MSH2 nodes must contain an id and three coordinates".into());
    }
    Ok(RawNode {
        id: fields[0].parse().map_err(|_| "invalid node id")?,
        xy: [
            fields[1]
                .parse()
                .map_err(|_| "invalid node x coordinate")?,
            fields[2]
                .parse()
                .map_err(|_| "invalid node y coordinate")?,
        ],
    })
}

/// Parse one `$Elements` entry line into node ids via the shared classifier.
fn parse_element(line: &str, mesh: &mut RawMesh) -> Result<(), String> {
    let fields: Vec<_> = line.split_whitespace().collect();
    if fields.len() < 3 {
        return Err("invalid MSH2 element".into());
    }
    let element_type: usize = fields[1].parse().map_err(|_| "invalid element type")?;
    let tag_count: usize = fields[2]
        .parse()
        .map_err(|_| "invalid element tag count")?;
    let nodes_start = 3 + tag_count;
    if fields.len() < nodes_start {
        return Err("truncated element tags".into());
    }
    let tags: Result<Vec<usize>, _> = fields[3..nodes_start].iter().map(|tag| tag.parse()).collect();
    let tags = tags.map_err(|_| "invalid element tag")?;
    let nodes: Result<Vec<usize>, _> = fields[nodes_start..].iter().map(|node| node.parse()).collect();
    let nodes = nodes.map_err(|_| "invalid element node id")?;
    mesh.push(classify_element(element_type, &tags, &nodes)?);
    Ok(())
}

/// Parse a counted `<count> <entries...> $End<name>` section body.
fn parse_section<'a, T>(
    input: &mut impl Iterator<Item = &'a str>,
    section: &str,
    end: &str,
    parse_entry: impl Fn(&str) -> Result<T, String>,
) -> Result<Vec<T>, String> {
    let count: usize = input
        .next()
        .ok_or(format!("missing {section} count"))?
        .parse()
        .map_err(|_| format!("invalid {section} count"))?;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        entries.push(parse_entry(
            input.next().ok_or(format!("truncated {section} list"))?,
        )?);
    }
    if input.next().map(str::trim) != Some(end) {
        return Err(format!("missing {end}"));
    }
    Ok(entries)
}

/// Parse full ASCII MSH2 content into a [`RawMesh`](super::model::RawMesh).
pub(crate) fn parse_ascii(source: &str) -> Result<RawMesh, String> {
    let mut mesh = RawMesh::default();
    let mut mesh_format_seen = false;
    let mut input = source.lines();
    while let Some(section) = input.next() {
        match section.trim() {
            "$MeshFormat" => {
                let fields: Vec<_> = input
                    .next()
                    .ok_or("missing mesh-format header")?
                    .split_whitespace()
                    .collect();
                if fields.len() != 3 || !fields[0].starts_with("2.") || fields[1] != "0" {
                    return Err("only ASCII MSH2 meshes are supported".into());
                }
                if input.next().map(str::trim) != Some("$EndMeshFormat") {
                    return Err("missing $EndMeshFormat".into());
                }
                mesh_format_seen = true;
            }
            "$PhysicalNames" => {
                for (region, name) in
                    parse_section(&mut input, "physical-name", "$EndPhysicalNames", parse_physical_name)?
                {
                    if mesh.physical_names.insert(region, name).is_some() {
                        return Err("duplicate physical name declaration".into());
                    }
                }
            }
            "$Nodes" => {
                mesh.nodes = parse_section(&mut input, "node", "$EndNodes", parse_node)?;
            }
            "$Elements" => {
                let count: usize = input
                    .next()
                    .ok_or("missing element count")?
                    .parse()
                    .map_err(|_| "invalid element count")?;
                for _ in 0..count {
                    parse_element(
                        input.next().ok_or("truncated element list")?,
                        &mut mesh,
                    )?;
                }
                if input.next().map(str::trim) != Some("$EndElements") {
                    return Err("missing $EndElements".into());
                }
            }
            _ => {}
        }
    }
    if !mesh_format_seen {
        return Err("missing $MeshFormat section".into());
    }
    Ok(mesh)
}
