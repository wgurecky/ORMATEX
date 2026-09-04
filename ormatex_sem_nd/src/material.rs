//! External coefficient and material-property evaluation.

use std::collections::HashMap;

use crate::common::CellState;

// Keep the old module paths available while mesh metadata has its own home.
pub use crate::mesh::{CellMeta, FacetMeta, MeshMetadata, PhysicalRegion};

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
