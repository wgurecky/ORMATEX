//! Shared solution-export mesh: points, linear/Lagrange cells, per-field data.
//!
//! Both the CSV and VTK writers consume this struct, so the two outputs agree
//! by construction. Coordinates come from the problems' cached physical points
//! (which include Dirichlet-eliminated nodes), and values resolve reduced DOFs
//! with prescribed-value fallback, so periodic, Dirichlet, and field-specific
//! reductions all export correctly.

use std::collections::HashMap;

use faer::MatRef;
use ndelement::types::ReferenceCellType;
use ndmesh::traits::Mesh;

use crate::sem_1d::SEM1DProblem;
use crate::sem_2d::SEM2DProblem;

/// One export cell with global point indices: linear (`Line`, `Quad`) for
/// `p = 1`, Lagrange (`LagrangeCurve`, `LagrangeQuad`) of matching order for
/// `p >= 2`, so the full GLL accuracy reaches the visualization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExportCell {
    Line([u64; 2]),
    Quad([u64; 4]),
    /// Order-`p` curve: `p + 1` points in ascending physical order.
    LagrangeCurve(Vec<u64>),
    /// Order-`p` quadrilateral: `(p + 1)^2` points in VTK Lagrange order
    /// (see [`lagrange_quad_node_order`]).
    LagrangeQuad(Vec<u64>),
}

impl ExportCell {
    /// Global point indices of this cell in VTK winding order.
    pub fn connectivity(&self) -> &[u64] {
        match self {
            ExportCell::Line(pair) => pair,
            ExportCell::Quad(quad) => quad,
            ExportCell::LagrangeCurve(points) | ExportCell::LagrangeQuad(points) => points,
        }
    }

    /// Number of points in this cell.
    pub fn size(&self) -> u64 {
        self.connectivity().len() as u64
    }
}

/// `(i, j)` GLL grid indices of one order-`p` quad in VTK Lagrange order:
/// corners `(0,0), (p,0), (p,p), (0,p)`, then bottom/right/top/left edge
/// interiors in ascending axis order, then row-major face interiors.
///
/// This mirrors `vtkHigherOrderQuadrilateral::PointIndexFromIJK` (VTK
/// `Common/DataModel`); for `p = 2` it reproduces the classic biquadratic
/// corner/edge-midpoint/center ordering. VTK derives the order from the
/// point count (`(p + 1)^2`), so no extra metadata is needed.
pub fn lagrange_quad_node_order(p: usize) -> Vec<(usize, usize)> {
    assert!(p >= 1, "Lagrange quad order must be >= 1");
    let mut order = Vec::with_capacity((p + 1) * (p + 1));
    order.extend_from_slice(&[(0, 0), (p, 0), (p, p), (0, p)]);
    order.extend((1..p).map(|i| (i, 0)));
    order.extend((1..p).map(|j| (p, j)));
    order.extend((1..p).map(|i| (i, p)));
    order.extend((1..p).map(|j| (0, j)));
    for j in 1..p {
        for i in 1..p {
            order.push((i, j));
        }
    }
    order
}

// ponytail: single intermediate for CSV + VTK so both agree by construction.
#[derive(Clone, Debug)]
pub struct ExportMesh {
    /// Spatial dimension: 1 or 2 (`z` is always zero-padded in `points`).
    pub dim: usize,
    /// Deduplicated point coordinates as `[x, y, z]`.
    pub points: Vec<[f64; 3]>,
    /// Linear (`p = 1`) or Lagrange (`p >= 2`) cells referencing `points`.
    pub cells: Vec<ExportCell>,
    /// Field names in system-vector order.
    pub field_names: Vec<String>,
    /// `point_fields[field][point]` values for every registered field.
    pub point_fields: Vec<Vec<f64>>,
}

impl ExportMesh {
    /// Number of deduplicated export points.
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Number of export cells (one per spectral element).
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Append another mesh's fields point-by-point, matching on coordinates.
    ///
    /// For multi-problem outputs on one mesh (e.g. fluid + species states).
    /// Duplicate field names are rejected; points must coincide.
    pub fn append_fields(&mut self, other: &ExportMesh) {
        assert_eq!(self.dim, other.dim, "cannot join meshes of different dimension");
        for name in &other.field_names {
            assert!(
                !self.field_names.contains(name),
                "duplicate joined field: {name}"
            );
        }
        let mut index = HashMap::with_capacity(self.points.len() * 2);
        for (id, &[x, y, _]) in self.points.iter().enumerate() {
            index.insert(PointTable::key(x, y), id);
        }
        let mut joined: Vec<Vec<f64>> = other
            .field_names
            .iter()
            .map(|_| vec![f64::NAN; self.points.len()])
            .collect();
        for (i, &[x, y, _]) in other.points.iter().enumerate() {
            let &id = index
                .get(&PointTable::key(x, y))
                .expect("joined mesh has a point with no match");
            for (dst, src) in joined.iter_mut().zip(&other.point_fields) {
                assert!(dst[id].is_nan(), "duplicate joined point");
                dst[id] = src[i];
            }
        }
        assert!(
            joined.iter().flatten().all(|v| v.is_finite()),
            "joined mesh leaves unmatched points"
        );
        self.field_names.extend(other.field_names.iter().cloned());
        self.point_fields.extend(joined);
    }
}

/// Insert-or-lookup helper keyed on coordinate bits (shared vertices and
/// periodic identifications merge to one point).
struct PointTable {
    dim: usize,
    points: Vec<[f64; 3]>,
    values: Vec<Vec<f64>>,
    index: HashMap<(i128, i128), usize>,
}

impl PointTable {
    fn new(dim: usize, nfields: usize) -> Self {
        Self {
            dim,
            points: Vec::new(),
            values: vec![Vec::new(); nfields],
            index: HashMap::new(),
        }
    }

    /// Quantized coordinate key: per-cell geometry evaluations of one shared
    /// vertex differ by ~1 ulp, so exact-bit hashing would split merged nodes.
    /// A 1e-12 grid is far above that noise and far below any real element.
    fn key(x: f64, y: f64) -> (i128, i128) {
        const SCALE: f64 = 1e12;
        (
            (x * SCALE).round() as i128,
            (y * SCALE).round() as i128,
        )
    }

    /// Return the global index for `(x, y)`, filling per-field values.
    ///
    /// Repeat visits must agree; this catches inconsistent Dirichlet or
    /// periodic mappings instead of silently averaging them.
    fn point(&mut self, x: f64, y: f64, field_values: &[f64]) -> usize {
        assert!(x.is_finite() && y.is_finite(), "non-finite export coordinate");
        assert_eq!(field_values.len(), self.values.len());
        let key = Self::key(x, y);
        if let Some(&id) = self.index.get(&key) {
            for (stored, &value) in self.values.iter_mut().zip(field_values) {
                assert!(value.is_finite(), "non-finite export field value");
                let previous = stored[id];
                assert!(
                    (previous - value).abs() <= 1e-12 * previous.abs().max(value.abs()).max(1.0),
                    "conflicting export values at one point"
                );
            }
            return id;
        }
        let id = self.points.len();
        self.index.insert(key, id);
        self.points.push(if self.dim == 1 { [x, 0.0, 0.0] } else { [x, y, 0.0] });
        for (stored, &value) in self.values.iter_mut().zip(field_values) {
            assert!(value.is_finite(), "non-finite export field value");
            stored.push(value);
        }
        id
    }
}

/// Resolve one cell-local value: reduced state entry or prescribed Dirichlet value.
fn local_value(
    reduced: &[Option<usize>],
    prescribed: &[Option<f64>],
    local: usize,
    offset: usize,
    state: MatRef<'_, f64>,
) -> f64 {
    match reduced[local] {
        Some(r) => state[(offset + r, 0)],
        None => prescribed[local].expect("eliminated DOF without prescribed value"),
    }
}

/// Sample every field of a 1D solved state: all GLL nodes, one cell per
/// spectral element (`Line` for `p = 1`, order-`p` `LagrangeCurve` above).
pub fn export_1d<M>(problem: &SEM1DProblem<M>, state: MatRef<'_, f64>) -> ExportMesh
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>,
{
    let field_names = problem.field_names().to_vec();
    let nfields = field_names.len();
    assert_eq!(state.nrows(), problem.system_size(), "state size mismatch");
    assert_eq!(state.ncols(), 1, "export requires one state column");
    let offsets: Vec<usize> = (0..nfields).map(|f| problem.field_offset(f)).collect();

    let npts = problem.cell_data.npts;
    let nodal = &problem.cell_data.nodal_quadrature;
    let mut table = PointTable::new(1, nfields);
    let mut cells = Vec::new();
    let mut field_values = vec![0.0; nfields];
    let mut order: Vec<usize> = Vec::new();

    for cell in 0..problem.cell_count() {
        let (reduced, prescribed) = problem.cell_field_maps(cell, nfields);
        let ndofs = reduced[0].len();
        // Physical order of the cell's GLL nodes (affine maps keep this sorted).
        order.clear();
        order.extend(0..ndofs);
        order.sort_by(|&a, &b| {
            let xa = problem.cell_data.physical_points_cache[cell * npts + nodal[a]];
            let xb = problem.cell_data.physical_points_cache[cell * npts + nodal[b]];
            xa.partial_cmp(&xb).unwrap()
        });
        let mut ids = Vec::with_capacity(ndofs);
        for &local in &order {
            let x = problem.cell_data.physical_points_cache[cell * npts + nodal[local]];
            for f in 0..nfields {
                field_values[f] =
                    local_value(reduced[f], prescribed[f], local, offsets[f], state);
            }
            ids.push(table.point(x, 0.0, &field_values) as u64);
        }
        cells.push(if ids.len() == 2 {
            ExportCell::Line([ids[0], ids[1]])
        } else {
            ExportCell::LagrangeCurve(ids)
        });
    }
    ExportMesh {
        dim: 1,
        points: table.points,
        cells,
        field_names,
        point_fields: table.values,
    }
}

/// Sample every field of a 2D solved state at all GLL nodes: one cell per
/// spectral element (`Quad` for `p = 1`, order-`p` `LagrangeQuad` above).
pub fn export_2d<M>(problem: &SEM2DProblem<M>, state: MatRef<'_, f64>) -> ExportMesh
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>,
{
    let field_names = problem.field_names().to_vec();
    let nfields = field_names.len();
    assert_eq!(state.nrows(), problem.system_size(), "state size mismatch");
    assert_eq!(state.ncols(), 1, "export requires one state column");
    let offsets: Vec<usize> = (0..nfields).map(|f| problem.field_offset(f)).collect();

    // Reference tensor ordering is q = j * n1d + i; `q_to_local` maps each
    // quadrature node to its entity-based local DOF.
    let npts = problem.cell_data.npts;
    let tensor = problem
        .cell_data
        .tensor
        .as_ref()
        .expect("2D export requires tensor-product cell data");
    let n1d = tensor.n1d;
    let node_order = lagrange_quad_node_order(n1d - 1);
    let local_of: Vec<usize> = node_order
        .iter()
        .map(|&(i, j)| tensor.q_to_local[j * n1d + i])
        .collect();
    let q_of: Vec<usize> = node_order.iter().map(|&(i, j)| j * n1d + i).collect();

    let mut table = PointTable::new(2, nfields);
    let mut cells = Vec::with_capacity(problem.cell_count());
    let mut field_values = vec![0.0; nfields];

    for cell in 0..problem.cell_count() {
        let (reduced, prescribed) = problem.cell_field_maps(cell, nfields);
        let mut ids = Vec::with_capacity(node_order.len());
        for (&local, &q) in local_of.iter().zip(q_of.iter()) {
            let base = (cell * npts + q) * 2;
            let x = problem.cell_data.physical_points_cache[base];
            let y = problem.cell_data.physical_points_cache[base + 1];
            for f in 0..nfields {
                field_values[f] =
                    local_value(reduced[f], prescribed[f], local, offsets[f], state);
            }
            ids.push(table.point(x, y, &field_values) as u64);
        }
        cells.push(if n1d == 2 {
            ExportCell::Quad([ids[0], ids[1], ids[2], ids[3]])
        } else {
            ExportCell::LagrangeQuad(ids)
        });
    }
    ExportMesh {
        dim: 2,
        points: table.points,
        cells,
        field_names,
        point_fields: table.values,
    }
}
