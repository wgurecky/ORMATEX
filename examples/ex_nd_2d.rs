//! 2D GLL spectral-element advection-diffusion example on quadrilateral
//! meshes with periodic boundary conditions.

use std::fs::File;
use std::io::Write;

use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};
use ormatex::ode_implicit::DirkIntegrator;
use ormatex::ode_sys::IntegrateSys;
use ormatex::tableau_implicit::ImplicitBT;

#[path = "ex_nd_common.rs"]
pub mod ex_nd_common;
use ex_nd_common::*;

use ndelement::{
    ciarlet::{CiarletElement, LagrangeElementFamily, LagrangeVariant},
    map::IdentityMap,
    traits::{ElementFamily, FiniteElement, MappedFiniteElement},
    types::{Continuity, ReferenceCellType},
};
use ndfunctionspace::{traits::FunctionSpace, FunctionSpaceImpl};
use ndmesh::{
    shapes::unit_square,
    traits::{Entity, GeometryMap, Mesh, Topology},
    SingleElementMesh,
};
use quadraturerules::{single_integral_quadrature, Domain, QuadratureRule};
use rlst::{rlst_dynamic_array, DynArray};

/// DOF reduction selector for the 2D example.
///
/// * `Periodic` -- identify left<->right and bottom<->top boundary vertices
///   (cyclic), system size = (# unique canonical vertices).
/// * `None` -- retain all dofs.
/// * `Dirichlet { facets_to_eliminate }` -- eliminate every DOF on the listed
///   boundary interval facets.
#[derive(Clone, Debug)]
pub enum DofReduction2D {
    /// Identify left<->right and bottom<->top boundary vertices (cyclic).
    Periodic,
    /// Retain all dofs.
    None,
    /// Eliminate every closure DOF on the listed boundary facets (homogeneous
    /// Dirichlet), including high-order edge DOFs.
    Dirichlet { facets_to_eliminate: Vec<usize> },
}

// `CellData` (per-cell-type quadrature + tabulation + jacobian caches) is
// shared with the 1D example via `ex_nd_common::CellData`.  See there for
// the struct definition + per-field docs.

// =============================================================================
// FiniteElement2DProblem
// =============================================================================

/// 2D GLL spectral-element problem on quadrilateral meshes.
pub struct FiniteElement2DProblem<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> {
    pub mesh: M,
    pub family: LagrangeElementFamily<f64>,
    p: usize,
    cell_data: CellData,
    cell_dofs: Vec<Vec<usize>>,
    cell_reduced_dofs: Vec<Vec<Option<usize>>>,
    bc: DofReduction2D,
    /// Precomputed `full -> Option<reduced>` DOF map (BC reduction LUT).
    /// Built once in `new`; `target_dof` is O(1), `apply_bc` is O(nnz).
    dof_lut: Vec<Option<usize>>,
    /// Number of reduced dofs (count of unique canonical vertices after
    /// periodic identification).  Distinct from `dof_lut.len()` when BCs
    /// identify boundary dofs.
    n_reduced: usize,
    /// (x, y) position of each full GLL nodal DOF, used to build the periodic
    /// identification LUT and to expose `dof_positions()`.
    dof_xy: Vec<(f64, f64)>,
}

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> FiniteElement2DProblem<M> {
    /// Build a GLL quadrilateral spectral-element problem.
    pub fn new(mesh: M, p: usize, bc: DofReduction2D) -> Self {
        assert!(p >= 1, "polynomial degree p must be >= 1");
        let family =
            LagrangeElementFamily::<f64>::new(p, Continuity::Standard, LagrangeVariant::GLL);
        let tdim = mesh.topology_dim();
        let gdim = mesh.geometry_dim();
        assert_eq!(tdim, 2, "FiniteElement2DProblem: mesh tdim must be 2");
        assert_eq!(
            gdim, 2,
            "FiniteElement2DProblem: mesh gdim must be 2 (2D-in-2D)"
        );

        let space = FunctionSpaceImpl::new(&mesh, &family);
        let n = space.process_size();

        assert_eq!(
            mesh.entity_types(tdim),
            &[ReferenceCellType::Quadrilateral],
            "FiniteElement2DProblem supports quadrilateral meshes only"
        );
        let cell_type = ReferenceCellType::Quadrilateral;
        let element = family.element(cell_type);
        let ndofs = element.dim();

        let order = element.lagrange_superdegree().saturating_sub(1);
        let (qx, qw) = single_integral_quadrature(
            QuadratureRule::GaussLobattoLegendre,
            Domain::Interval,
            order,
        )
        .unwrap();
        let n1d = qw.len();
        let xs_1d: Vec<f64> = (0..n1d).map(|i| qx[2 * i + 1]).collect();
        let npts = n1d * n1d;
        let mut pts = rlst_dynamic_array!(f64, [2, npts]);
        let mut wts = vec![0.0_f64; npts];
        for j in 0..n1d {
            for i in 0..n1d {
                let k = j * n1d + i;
                *pts.get_mut([0, k]).unwrap() = xs_1d[i];
                *pts.get_mut([1, k]).unwrap() = xs_1d[j];
                wts[k] = qw[i] * qw[j];
            }
        }

        let mut table = DynArray::<f64, 4>::from_shape(element.tabulate_array_shape(1, npts));
        element.tabulate(&pts, 1, &mut table);

        let ncells = mesh.entity_count(cell_type);
        let gmap = mesh.geometry_map(cell_type, 1, &pts);
        let mut jac_scratch = rlst_dynamic_array!(f64, [gdim, tdim, npts]);
        let mut jinv_scratch = rlst_dynamic_array!(f64, [tdim, gdim, npts]);
        let mut jdet_scratch = vec![0.0_f64; npts];
        let mut jinv_cache = rlst_dynamic_array!(f64, [tdim, gdim, npts, ncells]);
        let mut jdets_cache = vec![0.0_f64; ncells * npts];
        let mut physical_points_cache = vec![0.0_f64; ncells * npts * gdim];
        let mut dof_xy = vec![(f64::NAN, f64::NAN); n];
        let mut physical_pts = rlst_dynamic_array!(f64, [gdim, npts]);
        // The cell's local_index() indexes the quadrilateral Jacobian cache.
        for cell in mesh.entity_iter(cell_type) {
            let c = cell.local_index();
            gmap.jacobians_inverses_dets(c, &mut jac_scratch, &mut jinv_scratch, &mut jdet_scratch);
            for td in 0..tdim {
                for gd in 0..gdim {
                    for q in 0..npts {
                        *jinv_cache.get_mut([td, gd, q, c]).unwrap() =
                            *jinv_scratch.get([td, gd, q]).unwrap();
                    }
                }
            }
            for q in 0..npts {
                jdets_cache[c * npts + q] = jdet_scratch[q];
            }
            gmap.physical_points(c, &mut physical_pts);
            for q in 0..npts {
                for gd in 0..gdim {
                    physical_points_cache[(c * npts + q) * gdim + gd] =
                        *physical_pts.get([gd, q]).unwrap();
                }
            }
            let cell_dofs = space.entity_closure_dofs(cell_type, c).unwrap();
            for (local_dof, &full_dof) in cell_dofs.iter().enumerate() {
                let q = (0..npts)
                    .find(|&q| *table.get([0, q, local_dof, 0]).unwrap() > 1.0 - 1e-12)
                    .expect("GLL basis dof has no nodal quadrature point");
                let xy = (
                    *physical_pts.get([0, q]).unwrap(),
                    *physical_pts.get([1, q]).unwrap(),
                );
                let old = dof_xy[full_dof];
                if old.0.is_nan() {
                    dof_xy[full_dof] = xy;
                } else {
                    assert!(
                        (old.0 - xy.0).abs() < 1e-12 && (old.1 - xy.1).abs() < 1e-12,
                        "inconsistent coordinates for global dof {full_dof}"
                    );
                }
            }
        }
        drop(gmap);

        let mut reference_values = vec![0.0; ndofs * npts];
        let mut nodal_quadrature = vec![0; ndofs];
        for dof in 0..ndofs {
            for q in 0..npts {
                reference_values[dof * npts + q] = *table.get([0, q, dof, 0]).unwrap();
            }
            nodal_quadrature[dof] = (0..npts)
                .find(|&q| reference_values[dof * npts + q] > 1.0 - 1e-12)
                .expect("GLL basis dof has no nodal quadrature point");
        }
        let cell_data = CellData {
            pts,
            wts,
            npts,
            table,
            reference_values,
            nodal_quadrature,
            jinv_cache,
            jdets_cache,
            physical_points_cache,
            ndofs,
        };

        assert!(
            dof_xy.iter().all(|(x, y)| !x.is_nan() && !y.is_nan()),
            "every GLL dof must have a physical coordinate"
        );

        // --- DOF reduction LUT (full dof -> Option<reduced>).  Three cases:
        //   * `Periodic` -- identify every GLL node by snapped XY position
        //     (left<->right, top<->bottom).
        //   * `None` -- retain all dofs.
        //   * `Dirichlet { facets_to_eliminate }` -- eliminate every closure
        //     DOF on those boundary facets (`None` in the LUT), keep the rest
        //     contiguously-indexed.
        use std::collections::HashMap;
        let (dof_lut, n_reduced) = match &bc {
            DofReduction2D::Periodic => {
                const EPS: f64 = 1e-9;
                let snap = |c: f64| -> i64 {
                    let s = if c < EPS || c > 1.0 - EPS { 0.0 } else { c };
                    (s * 1e9).round() as i64
                };
                let mut canonical: HashMap<(i64, i64), usize> = HashMap::new();
                let mut canonical_dof = vec![0; n];
                for full in 0..n {
                    let (x, y) = dof_xy[full];
                    let key = (snap(x), snap(y));
                    canonical_dof[full] = *canonical.entry(key).or_insert(full);
                }
                let mut canonical_set: Vec<usize> =
                    (0..n).filter(|&full| canonical_dof[full] == full).collect();
                canonical_set.sort_unstable();
                let mut reduced_index: HashMap<usize, usize> = HashMap::new();
                for (i, &vi) in canonical_set.iter().enumerate() {
                    reduced_index.insert(vi, i);
                }
                let lut: Vec<Option<usize>> = (0..n)
                    .map(|d| {
                        let canon = canonical_dof[d];
                        Some(reduced_index[&canon])
                    })
                    .collect();
                (lut, canonical_set.len())
            }
            DofReduction2D::None => ((0..n).map(Some).collect(), n),
            DofReduction2D::Dirichlet {
                facets_to_eliminate,
            } => {
                use std::collections::HashSet;
                let mut eliminated_dofs = HashSet::new();
                for &facet_index in facets_to_eliminate {
                    let facet = mesh
                        .entity(ReferenceCellType::Interval, facet_index)
                        .expect("Dirichlet facet index out of range");
                    let topology = facet.topology();
                    let mut cells =
                        topology.connected_entity_iter(ReferenceCellType::Quadrilateral);
                    assert!(
                        cells.next().is_some() && cells.next().is_none(),
                        "Dirichlet facet {facet_index} must be a boundary interval"
                    );
                    for dof in space
                        .entity_closure_dofs(ReferenceCellType::Interval, facet_index)
                        .unwrap()
                    {
                        eliminated_dofs.insert(dof);
                    }
                }
                let mut lut: Vec<Option<usize>> = Vec::with_capacity(n);
                let mut reduced = 0;
                for d in 0..n {
                    if eliminated_dofs.contains(&d) {
                        lut.push(None);
                    } else {
                        lut.push(Some(reduced));
                        reduced += 1;
                    }
                }
                (lut, reduced)
            }
        };
        let cell_dofs: Vec<Vec<usize>> = (0..ncells)
            .map(|cell| {
                space
                    .entity_closure_dofs(ReferenceCellType::Quadrilateral, cell)
                    .unwrap()
                    .to_vec()
            })
            .collect();
        let cell_reduced_dofs = cell_dofs
            .iter()
            .map(|dofs| dofs.iter().map(|&dof| dof_lut[dof]).collect())
            .collect();

        Self {
            mesh,
            family,
            p,
            cell_data,
            cell_dofs,
            cell_reduced_dofs,
            bc,
            dof_lut,
            n_reduced,
            dof_xy,
        }
    }

    /// O(1) full -> Option<reduced> DOF lookup.
    pub fn target_dof(&self, full: usize) -> Option<usize> {
        self.dof_lut[full]
    }

    /// Number of reduced dofs (post-BC) = count of unique canonical
    /// vertices after periodic identification.
    pub fn reduced_size(&self) -> usize {
        self.n_reduced
    }

    /// (x, y) position of each reduced dof, in reduced-index order.
    pub fn dof_positions(&self) -> Vec<(f64, f64)> {
        let nr = self.reduced_size();
        let mut out = vec![(f64::NAN, f64::NAN); nr];
        // Snap x>1-eps and y>1-eps to 0 only under `Periodic`.
        let do_snap = matches!(self.bc, DofReduction2D::Periodic);
        let snap = |c: f64| if do_snap && c > 1.0 - 1e-9 { 0.0 } else { c };
        for full in 0..self.dof_lut.len() {
            if let Some(r) = self.dof_lut[full] {
                let (x, y) = self.dof_xy[full];
                out[r] = (snap(x), snap(y));
            }
        }
        debug_assert!(out.iter().all(|(x, y)| !x.is_nan() && !y.is_nan()));
        out
    }

    fn prepare_cell_ctx<'a>(
        &'a self,
        cell_index: usize,
        ndofs: usize,
        grads: &'a mut [f64],
    ) -> LocalCtx<'a> {
        let cd = &self.cell_data;
        let npts = cd.npts;
        let gdim = self.mesh.geometry_dim();
        let tdim = self.mesh.topology_dim();
        for dof_i in 0..ndofs {
            for q in 0..npts {
                for gd in 0..gdim {
                    let mut acc = 0.0;
                    for td in 0..tdim {
                        acc += *cd.jinv_cache.get([td, gd, q, cell_index]).unwrap()
                            * *cd.table.get([1 + td, q, dof_i, 0]).unwrap();
                    }
                    grads[(dof_i * gdim + gd) * npts + q] = acc;
                }
            }
        }
        LocalCtx {
            tdim,
            gdim,
            ncomp: 1,
            npts,
            ndofs,
            wts: &cd.wts,
            jdets: &cd.jdets_cache[cell_index * npts..(cell_index + 1) * npts],
            points: &cd.physical_points_cache
                [cell_index * npts * gdim..(cell_index + 1) * npts * gdim],
            values: &cd.reference_values,
            grads: &grads[..ndofs * gdim * npts],
        }
    }

    /// Assemble a reduced sparse matrix using `kernel.assemble_local` per
    /// quadrilateral. Periodic DOFs are combined and eliminated DOFs omitted
    /// while scattering, so no full-size matrix is exposed.
    pub fn assemble_bilinear<K: BilinearForm>(&self, kernel: &K) -> SparseColMat<usize, f64> {
        let gdim = self.mesh.geometry_dim();
        let cd = &self.cell_data;
        let max_ndofs = cd.ndofs;

        // Per-call reused scratch (sized once, reused for every cell).
        let mut grads_buf = vec![0.0_f64; max_ndofs * gdim * cd.npts];
        let mut local_mat = vec![0.0_f64; max_ndofs * max_ndofs];
        let mut triplets = Vec::with_capacity(self.cell_dofs.len() * cd.ndofs * cd.ndofs);

        let npts = cd.npts;
        grads_buf.resize(cd.ndofs * gdim * npts, 0.0);
        for (c, (dofs, reduced_dofs)) in self
            .cell_dofs
            .iter()
            .zip(&self.cell_reduced_dofs)
            .enumerate()
        {
            let ndofs = dofs.len();
            debug_assert!(ndofs == cd.ndofs, "ndofs mismatch: cell vs CellData");

            let ctx = self.prepare_cell_ctx(c, ndofs, &mut grads_buf[..ndofs * gdim * npts]);
            {
                let mat_slice = &mut local_mat[..ndofs * ndofs];
                mat_slice.fill(0.0);
                kernel.assemble_local(&ctx, mat_slice);
            }

            // Scatter directly to reduced indices.
            for (ti, &reduced_i) in reduced_dofs.iter().enumerate() {
                let Some(reduced_i) = reduced_i else {
                    continue;
                };
                for (si, &reduced_j) in reduced_dofs.iter().enumerate() {
                    let Some(reduced_j) = reduced_j else {
                        continue;
                    };
                    let entry = local_mat[ti * ndofs + si];
                    if entry.abs() > 1e-12 {
                        triplets.push(Triplet::new(reduced_i, reduced_j, entry));
                    }
                }
            }
        }
        let n = self.reduced_size();
        SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
    }

    /// Assemble a reduced volume RHS from a user-defined `LinearForm`.
    pub fn assemble_linear<K: LinearForm>(&self, kernel: &K) -> Vec<f64> {
        let gdim = self.mesh.geometry_dim();
        let cd = &self.cell_data;
        let mut grads_buf = vec![0.0_f64; cd.ndofs * gdim * cd.npts];
        let mut local_rhs = vec![0.0_f64; cd.ndofs];
        let mut rhs = vec![0.0_f64; self.reduced_size()];

        for (c, (dofs, reduced_dofs)) in self
            .cell_dofs
            .iter()
            .zip(&self.cell_reduced_dofs)
            .enumerate()
        {
            let ndofs = dofs.len();
            let npts = cd.npts;
            let ctx = self.prepare_cell_ctx(c, ndofs, &mut grads_buf[..ndofs * gdim * npts]);
            kernel.assemble_local_rhs(&ctx, &mut local_rhs[..ndofs]);
            for (local_i, &reduced_i) in reduced_dofs.iter().enumerate() {
                if let Some(reduced_i) = reduced_i {
                    rhs[reduced_i] += local_rhs[local_i];
                }
            }
        }
        rhs
    }

    /// Assemble the diagonal GLL mass matrix using collocated nodal quadrature.
    pub fn assemble_lumped_mass(&self) -> SparseColMat<usize, f64> {
        let cd = &self.cell_data;
        let mut triplets = Vec::with_capacity(self.cell_dofs.len() * cd.ndofs);
        for (cell, reduced_dofs) in self.cell_reduced_dofs.iter().enumerate() {
            for (local_dof, &reduced) in reduced_dofs.iter().enumerate() {
                if let Some(reduced) = reduced {
                    let q = cd.nodal_quadrature[local_dof];
                    triplets.push(Triplet::new(
                        reduced,
                        reduced,
                        cd.wts[q] * cd.jdets_cache[cell * cd.npts + q],
                    ));
                }
            }
        }
        let n = self.reduced_size();
        SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
    }

    /// Assemble selected natural-boundary kernels. The selector receives the
    /// `ndmesh` facet index and midpoint; returning `None` leaves it adiabatic.
    pub fn assemble_boundary<'a, F>(&self, select: F) -> BoundaryContributions
    where
        F: FnMut(BoundaryFacet) -> Option<&'a dyn BoundaryIntegrator>,
    {
        assemble_quad_boundaries(
            &self.mesh,
            &self.family,
            self.p,
            self.reduced_size(),
            |full| self.target_dof(full),
            select,
        )
    }
}

// =============================================================================
// Case driver: build problem, run self-checks, integrate, write CSV
// =============================================================================

fn run_case<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>>(
    label: &str,
    mesh: M,
    p: usize,
    nu: f64,
    vel: [f64; 2],
    nsteps: usize,
    dt: f64,
    out_path: &str,
) {
    println!("\n=== {label} (p={p}) ===");

    let problem = FiniteElement2DProblem::new(mesh, p, DofReduction2D::Periodic);
    let m_sparse = problem.assemble_lumped_mass();
    let k_sparse = problem.assemble_bilinear(&KernelAdvDiff2D::new(nu, vel));
    let n = m_sparse.nrows();
    println!(
        "reduced ndofs = {}, mass nnz = {}, adv_diff nnz = {}",
        n,
        m_sparse.compute_nnz(),
        k_sparse.compute_nnz()
    );

    // --- self-check 1: mass matrix symmetry.
    let m_dense = m_sparse.to_dense();
    let mut max_asym = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            max_asym = max_asym.max((m_dense[(i, j)] - m_dense[(j, i)]).abs());
        }
    }
    println!("mass symmetry max |M - M^T|: {:.3e}", max_asym);
    assert!(max_asym < 1e-14, "mass matrix not symmetric");

    let source = problem.assemble_linear(&KernelVolumeSource::new(1.0));
    let total_source: f64 = source.iter().sum();
    assert!(
        (total_source - 1.0).abs() < 1e-12,
        "unit-square source integral must be one, got {total_source}"
    );

    // --- self-check 2: periodic null-space A.1 = M^{-1} K . 1 ~ 0
    //     (constant mode is in the null space of the periodic operator).
    let sys = AdvDiffSys::new(m_sparse, k_sparse);
    let ones = Mat::<f64>::from_fn(n, 1, |_, _| 1.0);
    let a_one = sys.apply_minv_k(ones.as_ref());
    let max_row_sum = (0..n).map(|i| a_one[(i, 0)].abs()).fold(0.0_f64, f64::max);
    println!(
        "A = M^-1 K . ones max (periodic null-space check): {:.3e}",
        max_row_sum
    );
    assert!(max_row_sum < 1e-12, "periodic null-space check failed");

    // --- initial condition: periodic Gaussian bump centred at (0.5, 0.5).
    let xs = problem.dof_positions();
    let sigma = 0.1;
    let (x0, y0) = (0.5, 0.5);
    let mut y0_vec = Mat::<f64>::zeros(n, 1);
    for i in 0..n {
        y0_vec[(i, 0)] = periodic_gaussian_2d(xs[i].0, xs[i].1, x0, y0, sigma);
    }

    // --- integrate with implicit Euler (DIRK + implicit_euler tableau).
    let mut solver = DirkIntegrator::new(
        0.0,
        y0_vec.as_ref(),
        ImplicitBT::implicit_euler(),
        1e-12,
        1e-12,
    );
    let mut y = y0_vec.clone();
    for _ in 1..=nsteps {
        let res = solver.step(&sys, dt).unwrap();
        y = res.y.clone();
        solver.accept_step(res);
    }
    println!(
        "final t = {:.3}, max |u_final|: {:.3e}",
        solver.time(),
        y.col(0).iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
    );

    // --- CSV: write final snapshot (x, y, u).
    let mut f = File::create(out_path).expect("failed to create output csv");
    writeln!(f, "x,y,u").unwrap();
    for i in 0..n {
        writeln!(f, "{:.6},{:.6},{:.9e}", xs[i].0, xs[i].1, y[(i, 0)]).unwrap();
    }
    println!("wrote {} dofs to {}", n, out_path);
}

/// 2D periodic Gaussian evaluated at (x, y) on [0,1)^2 centred at (x0, y0).
fn periodic_gaussian_2d(x: f64, y: f64, x0: f64, y0: f64, sigma: f64) -> f64 {
    let two_s2 = 2.0 * sigma * sigma;
    let mut dx = x - x0;
    if dx > 0.5 {
        dx -= 1.0;
    } else if dx < -0.5 {
        dx += 1.0;
    }
    let mut dy = y - y0;
    if dy > 0.5 {
        dy -= 1.0;
    } else if dy < -0.5 {
        dy += 1.0;
    }
    (-(dx * dx + dy * dy) / two_s2).exp()
}

fn main() {
    let nx = 32;
    let ny = 2;
    let p = 2;
    let nu = 0.001;
    let vel = [0.5, 0.1];
    let dt = 0.01;
    let nsteps = 200;

    let quad_mesh: SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>> =
        unit_square(nx, ny, ReferenceCellType::Quadrilateral);
    run_case(
        "quad",
        quad_mesh,
        p,
        nu,
        vel,
        nsteps,
        dt,
        "target/ex_nd_2d_quad_out.csv",
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndmesh::traits::{Geometry, Point};

    struct XSource;

    impl LinearForm for XSource {
        fn integrand(&self, ctx: &LocalCtx, q: usize, test_i: usize) -> f64 {
            ctx.point(q)[0] * ctx.test(test_i, 0).v(q)
        }
    }

    struct YFlux;

    impl BoundaryIntegrator for YFlux {
        fn integrand_rhs(&self, ctx: &FacetCtx, q: usize, test_i: usize) -> f64 {
            ctx.point(q)[1] * ctx.test(test_i, 0).v(q)
        }
    }

    #[test]
    fn dirichlet_eliminates_every_high_order_facet_dof() {
        let mesh: SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>> =
            unit_square(2, 1, ReferenceCellType::Quadrilateral);
        let left_facet = mesh
            .entity_iter(ReferenceCellType::Interval)
            .find(|facet| {
                facet.geometry().points().all(|point| {
                    let mut xy = [0.0; 2];
                    point.coords(&mut xy);
                    xy[0].abs() < 1e-12
                })
            })
            .unwrap()
            .local_index();
        let problem = FiniteElement2DProblem::new(
            mesh,
            2,
            DofReduction2D::Dirichlet {
                facets_to_eliminate: vec![left_facet],
            },
        );
        let space = FunctionSpaceImpl::new(&problem.mesh, &problem.family);
        let facet_dofs = space
            .entity_closure_dofs(ReferenceCellType::Interval, left_facet)
            .unwrap();
        assert_eq!(
            facet_dofs.len(),
            3,
            "P2 interval has two vertices and one edge DOF"
        );
        assert!(facet_dofs
            .iter()
            .all(|&dof| problem.target_dof(dof).is_none()));
        assert_eq!(
            problem.reduced_size(),
            12,
            "all three left P2 DOFs are eliminated"
        );
    }

    #[test]
    fn volume_kernel_reads_physical_quadrature_points() {
        let mesh: SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>> =
            unit_square(2, 1, ReferenceCellType::Quadrilateral);
        let problem = FiniteElement2DProblem::new(mesh, 2, DofReduction2D::None);
        let rhs = problem.assemble_linear(&XSource);
        let integral: f64 = rhs.iter().sum();
        assert!(
            (integral - 0.5).abs() < 1e-12,
            "integral of x over the unit square must be one half, got {integral}"
        );
    }

    #[test]
    fn lumped_mass_matches_generic_gll_mass() {
        let mesh: SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>> =
            unit_square(2, 1, ReferenceCellType::Quadrilateral);
        let problem = FiniteElement2DProblem::new(mesh, 2, DofReduction2D::Periodic);
        let generic = problem.assemble_bilinear(&KernelMass::new()).to_dense();
        let lumped = problem.assemble_lumped_mass().to_dense();
        for i in 0..generic.nrows() {
            for j in 0..generic.ncols() {
                assert!((generic[(i, j)] - lumped[(i, j)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn boundary_kernel_reads_physical_quadrature_points() {
        let mesh: SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>> =
            unit_square(1, 1, ReferenceCellType::Quadrilateral);
        let problem = FiniteElement2DProblem::new(mesh, 2, DofReduction2D::None);
        let flux = YFlux;
        let boundary =
            problem.assemble_boundary(|facet| (facet.midpoint[0].abs() < 1e-12).then_some(&flux));
        let total_flux: f64 = boundary.rhs.iter().sum();
        assert!(
            (total_flux - 0.5).abs() < 1e-12,
            "integral of y on the left unit-square edge must be one half, got {total_flux}"
        );
    }
}
