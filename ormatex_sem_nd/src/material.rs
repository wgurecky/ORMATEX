//! External coefficient and material-property evaluation.

use std::collections::HashMap;
use std::sync::Arc;

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

/// A frozen discrete field sampled at quadrature points on the same mesh.
///
/// Layout is `values[cell * npts + q]` where `cell` is
/// `MaterialContext.cell.local_index` and `q` is `MaterialContext.q`.
/// Sampling must use the source problem's quadrature (`npts`); the target
/// problem must share the same cell count and `npts` (same mesh and `P`).
/// `derivative()` stays `None` so the field is frozen w.r.t. the target
/// unknown (zero Jacobian contribution).
#[derive(Clone, Debug)]
pub struct FrozenQuadratureField {
    npts: usize,
    values: Arc<Vec<f64>>,
}

impl FrozenQuadratureField {
    pub fn new(npts: usize, values: Vec<f64>) -> Self {
        assert!(npts > 0, "frozen field npts must be positive");
        assert!(
            values.len() % npts == 0,
            "frozen field values length must be a multiple of npts"
        );
        Self {
            npts,
            values: Arc::new(values),
        }
    }

    pub fn npts(&self) -> usize {
        self.npts
    }

    pub fn cell_count(&self) -> usize {
        self.values.len() / self.npts
    }

    /// Assert this snapshot matches a target `(cell_count, npts)` discretization.
    pub fn assert_compatible(&self, cell_count: usize, npts: usize) {
        assert_eq!(
            self.npts, npts,
            "frozen field npts mismatch: snapshot has {}, target has {npts} (same mesh and P required)",
            self.npts,
        );
        assert_eq!(
            self.cell_count(),
            cell_count,
            "frozen field cell-count mismatch (same mesh required)"
        );
    }
}

impl Coefficient<f64> for FrozenQuadratureField {
    #[inline(always)]
    fn eval(&self, ctx: &MaterialContext<'_>) -> f64 {
        self.values[ctx.cell.local_index * self.npts + ctx.q]
    }
}

impl MaterialProperty<f64> for FrozenQuadratureField {}

/// A frozen 2D velocity pair sharing one sampling layout.
///
/// Convenience wrapper so examples pass one object while kernels keep
/// consuming two scalar `MaterialProperty<f64>` coefficients.
#[derive(Clone, Debug)]
pub struct FrozenVelocity2D {
    pub x: FrozenQuadratureField,
    pub y: FrozenQuadratureField,
}

impl FrozenVelocity2D {
    pub fn new(x: FrozenQuadratureField, y: FrozenQuadratureField) -> Self {
        assert_eq!(
            x.npts(),
            y.npts(),
            "frozen velocity components need equal npts"
        );
        assert_eq!(
            x.cell_count(),
            y.cell_count(),
            "frozen velocity components need equal cell counts"
        );
        Self { x, y }
    }
}

/// A frozen discrete field sampled at boundary-facet quadrature points.
///
/// Layout is `values[facet * npts + q]` where `facet` is the mesh-global
/// facet index (`FacetMeta.local_index`) and `q` is the facet quadrature
/// index. Covers every mesh facet (interior entries are unused zeros) so
/// boundary kernels can index directly. Sampling must use the source
/// problem's facet quadrature; the target problem must share the facet count
/// and facet `npts` (same mesh and `P`). Boundary kernels treat the field as
/// frozen data (no Jacobian contribution from velocity).
#[derive(Clone, Debug)]
pub struct FrozenFacetField {
    npts: usize,
    nfacets: usize,
    values: Arc<Vec<f64>>,
}

impl FrozenFacetField {
    pub fn new(npts: usize, nfacets: usize, values: Vec<f64>) -> Self {
        assert!(npts > 0, "frozen facet field npts must be positive");
        assert_eq!(
            values.len(),
            nfacets * npts,
            "frozen facet field values length must equal nfacets * npts"
        );
        Self {
            npts,
            nfacets,
            values: Arc::new(values),
        }
    }

    pub fn npts(&self) -> usize {
        self.npts
    }

    pub fn facet_count(&self) -> usize {
        self.nfacets
    }

    #[inline(always)]
    pub fn value(&self, facet: usize, q: usize) -> f64 {
        self.values[facet * self.npts + q]
    }

    /// Assert this snapshot matches a target `(facet_count, npts)` discretization.
    pub fn assert_compatible(&self, facet_count: usize, npts: usize) {
        assert_eq!(
            self.npts, npts,
            "frozen facet field npts mismatch: snapshot has {}, target has {npts} (same mesh and P required)",
            self.npts,
        );
        assert_eq!(
            self.nfacets, facet_count,
            "frozen facet field facet-count mismatch (same mesh required)"
        );
    }
}
