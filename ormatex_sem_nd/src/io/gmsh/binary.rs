//! Binary MSH2 (`2.x 1 8`) section parser producing a [`RawMesh`](super::model::RawMesh).
//!
//! Layout: `$MeshFormat` + endianness marker (`int32` 1), then ASCII-framed
//! sections. `$PhysicalNames` stays ASCII; `$Nodes` holds `int32 id` +
//! three `float64` per node; `$Elements` holds same-`(type, tag-count)`
//! groups, each with an `int32 type, group-size, tag-count` header followed
//! by `int32 id, [tags], [node ids]` rows. All integers and floats are
//! little-endian.

use super::model::{classify_element, parse_physical_name, RawMesh, RawNode};

/// Cursor over the raw file bytes.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn eof(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn read_exact(&mut self, n: usize, what: &str) -> Result<&'a [u8], String> {
        let end = self.pos + n;
        if end > self.bytes.len() {
            return Err(format!("truncated binary mesh ({what})"));
        }
        let chunk = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(chunk)
    }

    fn read_i32(&mut self, what: &str) -> Result<usize, String> {
        let chunk = self.read_exact(4, what)?;
        let value = i32::from_le_bytes(chunk.try_into().unwrap());
        usize::try_from(value).map_err(|_| format!("invalid {what}: negative value"))
    }

    fn read_i32_raw(&mut self, what: &str) -> Result<i32, String> {
        let chunk = self.read_exact(4, what)?;
        Ok(i32::from_le_bytes(chunk.try_into().unwrap()))
    }

    fn read_f64(&mut self, what: &str) -> Result<f64, String> {
        let chunk = self.read_exact(8, what)?;
        Ok(f64::from_le_bytes(chunk.try_into().unwrap()))
    }

    /// Next `\n`-terminated line without the line break (`\r` stripped).
    fn read_line(&mut self) -> Result<&'a str, String> {
        if self.eof() {
            return Err("truncated binary mesh (section line)".into());
        }
        let start = self.pos;
        while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
            self.pos += 1;
        }
        let line = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| "binary mesh section header is not ASCII")?;
        self.pos = (self.pos + 1).min(self.bytes.len());
        Ok(line.strip_suffix('\r').unwrap_or(line))
    }

    /// Skip blank separation between a binary blob and its `$End` marker.
    fn skip_blank(&mut self) {
        while self.pos < self.bytes.len()
            && (self.bytes[self.pos] == b'\n' || self.bytes[self.pos] == b'\r')
        {
            self.pos += 1;
        }
    }
}

/// Node count per supported element type (triangles included for diagnostics).
fn element_node_count(element_type: usize) -> Result<usize, String> {
    match element_type {
        15 => Ok(1),
        1 => Ok(2),
        2 => Ok(3),
        3 => Ok(4),
        9 => Ok(6),
        _ => Err(format!("unsupported Gmsh element type {element_type}")),
    }
}

/// Parse full binary MSH2 content (past the endianness marker) into a [`RawMesh`](super::model::RawMesh).
fn parse_binary_body(reader: &mut Reader, mesh: &mut RawMesh) -> Result<(), String> {
    loop {
        reader.skip_blank();
        if reader.eof() {
            break;
        }
        match reader.read_line()?.trim() {
            "$PhysicalNames" => {
                let count: usize = reader
                    .read_line()?
                    .parse()
                    .map_err(|_| "invalid physical-name count")?;
                for _ in 0..count {
                    let line = reader.read_line()?;
                    let (region, name) = parse_physical_name(line)?;
                    if mesh.physical_names.insert(region, name).is_some() {
                        return Err("duplicate physical name declaration".into());
                    }
                }
                if reader.read_line()?.trim() != "$EndPhysicalNames" {
                    return Err("missing $EndPhysicalNames".into());
                }
            }
            "$Nodes" => {
                let count: usize = reader
                    .read_line()?
                    .parse()
                    .map_err(|_| "invalid node count")?;
                mesh.nodes.reserve(count);
                for _ in 0..count {
                    let id = reader.read_i32("node id")?;
                    let x = reader.read_f64("node x coordinate")?;
                    let y = reader.read_f64("node y coordinate")?;
                    let z = reader.read_f64("node z coordinate")?;
                    if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                        return Err("non-finite node coordinate".into());
                    }
                    mesh.nodes.push(RawNode { id, xy: [x, y] });
                }
                reader.skip_blank();
                if reader.read_line()?.trim() != "$EndNodes" {
                    return Err("missing $EndNodes".into());
                }
            }
            "$Elements" => {
                let total: usize = reader
                    .read_line()?
                    .parse()
                    .map_err(|_| "invalid element count")?;
                let mut read = 0;
                while read < total {
                    let element_type = reader.read_i32("element group type")?;
                    let group_size = reader.read_i32("element group size")?;
                    if group_size == 0 {
                        return Err("empty binary element group".into());
                    }
                    let tag_count = reader.read_i32("element tag count")?;
                    let node_count = element_node_count(element_type)?;
                    for _ in 0..group_size {
                        let _id = reader.read_i32_raw("element id")?;
                        let mut tags = Vec::with_capacity(tag_count);
                        for _ in 0..tag_count {
                            tags.push(reader.read_i32("element tag")?);
                        }
                        let mut nodes = Vec::with_capacity(node_count);
                        for _ in 0..node_count {
                            nodes.push(reader.read_i32("element node id")?);
                        }
                        mesh.push(classify_element(element_type, &tags, &nodes)?);
                    }
                    read += group_size;
                }
                reader.skip_blank();
                if reader.read_line()?.trim() != "$EndElements" {
                    return Err("missing $EndElements".into());
                }
            }
            section => return Err(format!("unsupported binary mesh section {section}")),
        }
    }
    Ok(())
}

/// Parse a complete binary MSH2 file: header, endianness marker, then sections.
pub(crate) fn parse_binary(bytes: &[u8]) -> Result<RawMesh, String> {
    let mut reader = Reader::new(bytes);
    if reader.read_line()? != "$MeshFormat" {
        return Err("missing $MeshFormat section".into());
    }
    let header = reader.read_line()?.to_string();
    let fields: Vec<_> = header.split_whitespace().collect();
    if fields.len() != 3 || !fields[0].starts_with("2.") || fields[1] != "1" || fields[2] != "8" {
        return Err("only binary MSH2 meshes (2.x, 8-byte) are supported".into());
    }
    if reader.read_i32_raw("endianness marker")? != 1 {
        return Err("only little-endian binary MSH2 meshes are supported".into());
    }
    // Gmsh writes one blank line between the marker and `$EndMeshFormat`.
    let mut line = reader.read_line()?.to_string();
    if line.trim().is_empty() {
        line = reader.read_line()?.to_string();
    }
    if line.trim() != "$EndMeshFormat" {
        return Err("missing $EndMeshFormat".into());
    }
    let mut mesh = RawMesh::default();
    parse_binary_body(&mut reader, &mut mesh)?;
    Ok(mesh)
}
