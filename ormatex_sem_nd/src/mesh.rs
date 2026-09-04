//! Mesh metadata and physical-region selection.

use std::collections::HashMap;

/// A Gmsh-style physical region identifier.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PhysicalRegion {
    pub dimension: usize,
    pub tag: usize,
}

/// A numeric or named physical-region selector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PhysicalSelector {
    Tag(usize),
    Name(String),
}

impl From<usize> for PhysicalSelector {
    fn from(tag: usize) -> Self {
        Self::Tag(tag)
    }
}

impl From<&str> for PhysicalSelector {
    fn from(name: &str) -> Self {
        Self::Name(name.to_owned())
    }
}

impl From<String> for PhysicalSelector {
    fn from(name: String) -> Self {
        Self::Name(name)
    }
}

/// Metadata for one volume cell.
#[derive(Clone, Copy, Debug, Default)]
pub struct CellMeta {
    pub local_index: usize,
    pub physical_region: Option<PhysicalRegion>,
}

/// Metadata for one boundary facet.
#[derive(Clone, Copy, Debug, Default)]
pub struct FacetMeta {
    pub local_index: usize,
    pub physical_region: Option<PhysicalRegion>,
}

/// Optional mesh metadata indexed by mesh-local entity index.
///
/// Empty region vectors mean no region metadata. Nonempty vectors must follow
/// the `local_index()` ordering of the corresponding mesh entity type.
#[derive(Clone, Debug, Default)]
pub struct MeshMetadata {
    pub cell_regions: Vec<Option<PhysicalRegion>>,
    pub facet_regions: Vec<Option<PhysicalRegion>>,
    pub physical_names: HashMap<PhysicalRegion, String>,
}

impl MeshMetadata {
    pub fn validate(&self, cell_count: usize, facet_count: usize) {
        assert!(
            self.cell_regions.is_empty() || self.cell_regions.len() == cell_count,
            "cell-region metadata length does not match the mesh"
        );
        assert!(
            self.facet_regions.is_empty() || self.facet_regions.len() == facet_count,
            "facet-region metadata length does not match the mesh"
        );
    }

    /// Return boundary facet indices selected by a physical tag or name.
    ///
    /// Numeric selection is scoped to the facet metadata, while named
    /// selection uses the dimension-aware physical-name table.
    pub fn boundary_facets<S>(&self, selector: S) -> Vec<usize>
    where
        S: Into<PhysicalSelector>,
    {
        self.indices(&self.facet_regions, selector)
    }

    /// Return cell indices selected by a physical tag or name.
    pub fn cell_indices<S>(&self, selector: S) -> Vec<usize>
    where
        S: Into<PhysicalSelector>,
    {
        self.indices(&self.cell_regions, selector)
    }

    fn indices<S>(&self, regions: &[Option<PhysicalRegion>], selector: S) -> Vec<usize>
    where
        S: Into<PhysicalSelector>,
    {
        let selector = selector.into();
        regions
            .iter()
            .enumerate()
            .filter_map(|(index, region)| {
                let region = (*region)?;
                let selected = match &selector {
                    PhysicalSelector::Tag(tag) => region.tag == *tag,
                    PhysicalSelector::Name(name) => self
                        .physical_names
                        .get(&region)
                        .is_some_and(|physical_name| physical_name == name),
                };
                selected.then_some(index)
            })
            .collect()
    }

    pub fn cell(&self, index: usize) -> CellMeta {
        CellMeta {
            local_index: index,
            physical_region: self.cell_regions.get(index).copied().flatten(),
        }
    }

    pub fn facet(&self, index: usize) -> FacetMeta {
        FacetMeta {
            local_index: index,
            physical_region: self.facet_regions.get(index).copied().flatten(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_cells_and_facets_by_tag_or_name() {
        let left = PhysicalRegion {
            dimension: 1,
            tag: 1,
        };
        let domain = PhysicalRegion {
            dimension: 2,
            tag: 10,
        };
        let metadata = MeshMetadata {
            cell_regions: vec![Some(domain)],
            facet_regions: vec![Some(left), None],
            physical_names: HashMap::from([
                (left, "left".to_owned()),
                (domain, "domain".to_owned()),
            ]),
        };

        assert_eq!(metadata.boundary_facets(1), vec![0]);
        assert_eq!(metadata.boundary_facets("left"), vec![0]);
        assert_eq!(metadata.cell_indices(10), vec![0]);
        assert_eq!(metadata.cell_indices("domain"), vec![0]);
        assert!(metadata.boundary_facets("domain").is_empty());
        assert!(metadata.cell_indices("missing").is_empty());
    }
}
