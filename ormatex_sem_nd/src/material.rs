//! External coefficient and material-property evaluation.

use std::collections::HashMap;

use crate::common::CellState;

/// A Gmsh-style physical region identifier.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PhysicalRegion {
    pub dimension: usize,
    pub tag: usize,
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

/// All information available while evaluating a coefficient at one point.
///
/// `state` is absent during state-independent bilinear assembly. Coefficients
/// that require state therefore belong in residual assembly, not a bilinear form.
pub struct MaterialContext<'a> {
    pub time: f64,
    pub point: &'a [f64],
    pub cell: CellMeta,
    pub state: Option<&'a CellState<'a>>,
    pub q: usize,
}

/// A coefficient that may depend on space, time, region, and optionally state.
pub trait Coefficient<T>: Send + Sync {
    fn eval(&self, ctx: &MaterialContext<'_>) -> T;
}

/// A coefficient with derivatives needed by nonlinear residual Jacobians.
///
/// `derivative(ctx, field)` is the derivative with respect to the point value
/// of `state.value(field, ctx.q)`. Returning `None` freezes that coefficient
/// with a zero derivative. Gradient-dependent material laws are not supported.
pub trait MaterialProperty<T>: Coefficient<T> {
    fn derivative(&self, _ctx: &MaterialContext<'_>, _solution_field: usize) -> Option<T> {
        None
    }
}

impl<T, F> Coefficient<T> for F
where
    F: for<'a> Fn(&MaterialContext<'a>) -> T + Send + Sync,
{
    fn eval(&self, ctx: &MaterialContext<'_>) -> T {
        self(ctx)
    }
}

impl<T, F> MaterialProperty<T> for F where F: for<'a> Fn(&MaterialContext<'a>) -> T + Send + Sync {}

/// A constant coefficient/material property.
#[derive(Clone, Copy, Debug)]
pub struct ConstantCoefficient<T>(pub T);

impl<T: Copy + Send + Sync> Coefficient<T> for ConstantCoefficient<T> {
    fn eval(&self, _ctx: &MaterialContext<'_>) -> T {
        self.0
    }
}

impl<T: Copy + Send + Sync> MaterialProperty<T> for ConstantCoefficient<T> {}

/// Select a constant value by physical region, with a fallback value.
#[derive(Clone, Debug)]
pub struct RegionCoefficient<T> {
    pub values: HashMap<PhysicalRegion, T>,
    pub default: T,
}

impl<T> RegionCoefficient<T> {
    pub fn new(values: HashMap<PhysicalRegion, T>, default: T) -> Self {
        Self { values, default }
    }
}

impl<T: Clone + Send + Sync> Coefficient<T> for RegionCoefficient<T> {
    fn eval(&self, ctx: &MaterialContext<'_>) -> T {
        ctx.cell
            .physical_region
            .and_then(|region| self.values.get(&region))
            .cloned()
            .unwrap_or_else(|| self.default.clone())
    }
}

impl<T: Clone + Send + Sync> MaterialProperty<T> for RegionCoefficient<T> {}
