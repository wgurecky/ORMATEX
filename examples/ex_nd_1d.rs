//! 1D GLL spectral-element advection-diffusion on `ndmesh` interval meshes.

use std::fs::File;
use std::io::Write;

use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};
use ormatex::ode_implicit::DirkIntegrator;
use ormatex::ode_sys::IntegrateSys;
use ormatex::tableau_implicit::ImplicitBT;

#[path = "ex_nd_common.rs"]
mod ex_nd_common;
use ex_nd_common::*;

use ndelement::{
    ciarlet::{CiarletElement, LagrangeElementFamily, LagrangeVariant},
    map::IdentityMap,
    traits::{ElementFamily, FiniteElement, MappedFiniteElement},
    types::{Continuity, ReferenceCellType},
};
use ndfunctionspace::{traits::FunctionSpace, FunctionSpaceImpl};
use ndmesh::{
    shapes::unit_interval,
    traits::{Entity, Geometry, GeometryMap, Mesh, Point, Topology},
    SingleElementMesh,
};
use quadraturerules::{single_integral_quadrature, Domain, QuadratureRule};
use rlst::{rlst_dynamic_array, DynArray};

/// DOF reduction for a 1D interval mesh. Facets are `Point` entity indices.
#[derive(Clone, Debug)]
pub enum DofReduction1D {
    None,
    /// Identify the second endpoint facet with the first endpoint facet.
    Periodic {
        facets: [usize; 2],
    },
    /// Eliminate every closure DOF on selected endpoint facets.
    Dirichlet {
        facets_to_eliminate: Vec<usize>,
    },
}

/// Geometry used to select a 1D natural-boundary kernel.
#[derive(Clone, Copy, Debug)]
pub struct BoundaryPoint {
    pub index: usize,
    pub coordinate: f64,
    pub normal: f64,
}

/// 1D GLL spectral-element problem on interval meshes.
pub struct FiniteElement1DProblem<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> {
    pub mesh: M,
    pub family: LagrangeElementFamily<f64>,
    cell_data: CellData,
    cell_dofs: Vec<Vec<usize>>,
    cell_reduced_dofs: Vec<Vec<Option<usize>>>,
    dof_lut: Vec<Option<usize>>,
    n_reduced: usize,
    dof_x: Vec<f64>,
}

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> FiniteElement1DProblem<M> {
    pub fn new(mesh: M, p: usize, reduction: DofReduction1D) -> Self {
        assert!(p >= 1, "polynomial degree p must be >= 1");
        assert_eq!(
            mesh.topology_dim(),
            1,
            "FiniteElement1DProblem: mesh tdim must be 1"
        );
        assert_eq!(
            mesh.geometry_dim(),
            1,
            "FiniteElement1DProblem: mesh gdim must be 1"
        );
        assert_eq!(
            mesh.entity_types(1),
            &[ReferenceCellType::Interval],
            "FiniteElement1DProblem supports interval meshes only"
        );

        let family =
            LagrangeElementFamily::<f64>::new(p, Continuity::Standard, LagrangeVariant::GLL);
        let space = FunctionSpaceImpl::new(&mesh, &family);
        let n = space.process_size();
        let element = family.element(ReferenceCellType::Interval);
        let (qpts, wts) = single_integral_quadrature(
            QuadratureRule::GaussLobattoLegendre,
            Domain::Interval,
            element.lagrange_superdegree().saturating_sub(1),
        )
        .unwrap();
        let npts = wts.len();
        let mut pts = rlst_dynamic_array!(f64, [1, npts]);
        for q in 0..npts {
            *pts.get_mut([0, q]).unwrap() = qpts[2 * q + 1];
        }
        let mut table = DynArray::<f64, 4>::from_shape(element.tabulate_array_shape(1, npts));
        element.tabulate(&pts, 1, &mut table);

        let ncells = mesh.entity_count(ReferenceCellType::Interval);
        let gmap = mesh.geometry_map(ReferenceCellType::Interval, 1, &pts);
        let mut jac_scratch = rlst_dynamic_array!(f64, [1, 1, npts]);
        let mut jinv_scratch = rlst_dynamic_array!(f64, [1, 1, npts]);
        let mut jdet_scratch = vec![0.0; npts];
        let mut jinv_cache = rlst_dynamic_array!(f64, [1, 1, npts, ncells]);
        let mut jdets_cache = vec![0.0; ncells * npts];
        let mut physical_points_cache = vec![0.0; ncells * npts];
        let mut dof_x = vec![f64::NAN; n];
        let mut physical_pts = rlst_dynamic_array!(f64, [1, npts]);

        for cell in mesh.entity_iter(ReferenceCellType::Interval) {
            let c = cell.local_index();
            gmap.jacobians_inverses_dets(c, &mut jac_scratch, &mut jinv_scratch, &mut jdet_scratch);
            gmap.physical_points(c, &mut physical_pts);
            for q in 0..npts {
                *jinv_cache.get_mut([0, 0, q, c]).unwrap() = *jinv_scratch.get([0, 0, q]).unwrap();
                jdets_cache[c * npts + q] = jdet_scratch[q];
                physical_points_cache[c * npts + q] = *physical_pts.get([0, q]).unwrap();
            }
            let cell_dofs = space
                .entity_closure_dofs(ReferenceCellType::Interval, c)
                .unwrap();
            for (local_dof, &full_dof) in cell_dofs.iter().enumerate() {
                let q = (0..npts)
                    .find(|&q| *table.get([0, q, local_dof, 0]).unwrap() > 1.0 - 1e-12)
                    .expect("GLL basis dof has no nodal quadrature point");
                let x = *physical_pts.get([0, q]).unwrap();
                if dof_x[full_dof].is_nan() {
                    dof_x[full_dof] = x;
                } else {
                    assert!(
                        (dof_x[full_dof] - x).abs() < 1e-12,
                        "inconsistent coordinate for DOF {full_dof}"
                    );
                }
            }
        }
        drop(gmap);
        assert!(
            dof_x.iter().all(|x| !x.is_nan()),
            "every GLL DOF must have a coordinate"
        );

        let mut reference_values = vec![0.0; element.dim() * npts];
        let mut nodal_quadrature = vec![0; element.dim()];
        for dof in 0..element.dim() {
            for q in 0..npts {
                reference_values[dof * npts + q] = *table.get([0, q, dof, 0]).unwrap();
            }
            nodal_quadrature[dof] = (0..npts)
                .find(|&q| reference_values[dof * npts + q] > 1.0 - 1e-12)
                .expect("GLL basis dof has no nodal quadrature point");
        }
        let cell_data = CellData {
            wts,
            npts,
            ndofs: element.dim(),
            table,
            reference_values,
            nodal_quadrature,
            jinv_cache,
            jdets_cache,
            physical_points_cache,
            pts,
        };

        let boundary_dof = |facet_index: usize| -> usize {
            let facet = mesh
                .entity(ReferenceCellType::Point, facet_index)
                .expect("boundary facet index out of range");
            let topology = facet.topology();
            let mut cells = topology.connected_entity_iter(ReferenceCellType::Interval);
            assert!(
                cells.next().is_some() && cells.next().is_none(),
                "facet {facet_index} must be a boundary point"
            );
            let dofs = space
                .entity_closure_dofs(ReferenceCellType::Point, facet_index)
                .unwrap();
            assert_eq!(dofs.len(), 1, "a scalar endpoint must have one DOF");
            dofs[0]
        };

        let (dof_lut, n_reduced) = match reduction {
            DofReduction1D::None => ((0..n).map(Some).collect(), n),
            DofReduction1D::Dirichlet {
                facets_to_eliminate,
            } => {
                let mut eliminated = std::collections::HashSet::new();
                for facet in facets_to_eliminate {
                    eliminated.insert(boundary_dof(facet));
                }
                let mut reduced = 0;
                let lut: Vec<Option<usize>> = (0..n)
                    .map(|dof| {
                        if eliminated.contains(&dof) {
                            None
                        } else {
                            let out = Some(reduced);
                            reduced += 1;
                            out
                        }
                    })
                    .collect();
                (lut, reduced)
            }
            DofReduction1D::Periodic { facets } => {
                let master = boundary_dof(facets[0]);
                let slave = boundary_dof(facets[1]);
                assert_ne!(master, slave, "periodic facets must be distinct");
                let canonical: Vec<usize> = (0..n)
                    .map(|dof| if dof == slave { master } else { dof })
                    .collect();
                let canonical_set: Vec<usize> =
                    (0..n).filter(|&dof| canonical[dof] == dof).collect();
                let reduced_index: std::collections::HashMap<usize, usize> = canonical_set
                    .iter()
                    .enumerate()
                    .map(|(reduced, &dof)| (dof, reduced))
                    .collect();
                let lut = canonical
                    .iter()
                    .map(|dof| Some(reduced_index[dof]))
                    .collect();
                (lut, canonical_set.len())
            }
        };
        let cell_dofs: Vec<Vec<usize>> = (0..ncells)
            .map(|cell| {
                space
                    .entity_closure_dofs(ReferenceCellType::Interval, cell)
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
            cell_data,
            cell_dofs,
            cell_reduced_dofs,
            dof_lut,
            n_reduced,
            dof_x,
        }
    }

    pub fn target_dof(&self, full: usize) -> Option<usize> {
        self.dof_lut[full]
    }

    pub fn reduced_size(&self) -> usize {
        self.n_reduced
    }

    pub fn dof_positions(&self) -> Vec<f64> {
        let mut out = vec![f64::NAN; self.reduced_size()];
        for (full, &x) in self.dof_x.iter().enumerate() {
            if let Some(reduced) = self.target_dof(full) {
                if out[reduced].is_nan() {
                    out[reduced] = x;
                }
            }
        }
        debug_assert!(out.iter().all(|x| !x.is_nan()));
        out
    }

    fn prepare_cell_ctx<'a>(
        &'a self,
        cell_index: usize,
        ndofs: usize,
        grads: &'a mut [f64],
    ) -> LocalCtx<'a> {
        let cd = &self.cell_data;
        for dof_i in 0..ndofs {
            for q in 0..cd.npts {
                grads[dof_i * cd.npts + q] = *cd.jinv_cache.get([0, 0, q, cell_index]).unwrap()
                    * *cd.table.get([1, q, dof_i, 0]).unwrap();
            }
        }
        LocalCtx {
            tdim: 1,
            gdim: 1,
            ncomp: 1,
            npts: cd.npts,
            ndofs,
            wts: &cd.wts,
            jdets: &cd.jdets_cache[cell_index * cd.npts..(cell_index + 1) * cd.npts],
            points: &cd.physical_points_cache[cell_index * cd.npts..(cell_index + 1) * cd.npts],
            values: &cd.reference_values,
            grads: &grads[..ndofs * cd.npts],
        }
    }

    pub fn assemble_bilinear<K: BilinearForm>(&self, kernel: &K) -> SparseColMat<usize, f64> {
        let cd = &self.cell_data;
        let mut grads = vec![0.0; cd.ndofs * cd.npts];
        let mut local = vec![0.0; cd.ndofs * cd.ndofs];
        let mut triplets = Vec::with_capacity(self.cell_dofs.len() * cd.ndofs * cd.ndofs);

        for (c, (dofs, reduced_dofs)) in self
            .cell_dofs
            .iter()
            .zip(&self.cell_reduced_dofs)
            .enumerate()
        {
            let ndofs = dofs.len();
            let ctx = self.prepare_cell_ctx(c, ndofs, &mut grads[..ndofs * cd.npts]);
            let local = &mut local[..ndofs * ndofs];
            local.fill(0.0);
            kernel.assemble_local(&ctx, local);
            for (ti, &reduced_i) in reduced_dofs.iter().enumerate() {
                let Some(reduced_i) = reduced_i else {
                    continue;
                };
                for (si, &reduced_j) in reduced_dofs.iter().enumerate() {
                    if let Some(reduced_j) = reduced_j {
                        let value = local[ti * ndofs + si];
                        if value.abs() > 1e-12 {
                            triplets.push(Triplet::new(reduced_i, reduced_j, value));
                        }
                    }
                }
            }
        }
        let n = self.reduced_size();
        SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
    }

    pub fn assemble_linear<K: LinearForm>(&self, kernel: &K) -> Vec<f64> {
        let cd = &self.cell_data;
        let mut grads = vec![0.0; cd.ndofs * cd.npts];
        let mut local = vec![0.0; cd.ndofs];
        let mut rhs = vec![0.0; self.reduced_size()];

        for (c, (dofs, reduced_dofs)) in self
            .cell_dofs
            .iter()
            .zip(&self.cell_reduced_dofs)
            .enumerate()
        {
            let ndofs = dofs.len();
            let ctx = self.prepare_cell_ctx(c, ndofs, &mut grads[..ndofs * cd.npts]);
            kernel.assemble_local_rhs(&ctx, &mut local[..ndofs]);
            for (local_i, &reduced_i) in reduced_dofs.iter().enumerate() {
                if let Some(reduced_i) = reduced_i {
                    rhs[reduced_i] += local[local_i];
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

    pub fn assemble_boundary<'a, F>(&self, mut select: F) -> BoundaryContributions
    where
        F: FnMut(BoundaryPoint) -> Option<&'a dyn BoundaryIntegrator>,
    {
        let space = FunctionSpaceImpl::new(&self.mesh, &self.family);
        let element = self.family.element(ReferenceCellType::Interval);
        let mut rhs = vec![0.0; self.reduced_size()];
        let mut triplets = Vec::new();
        let mut coord = [0.0; 1];

        for point in self.mesh.entity_iter(ReferenceCellType::Point) {
            let point_index = point.local_index();
            let topology = point.topology();
            let mut cells = topology.connected_entity_iter(ReferenceCellType::Interval);
            let Some(cell_index) = cells.next() else {
                continue;
            };
            if cells.next().is_some() {
                continue;
            }
            point.geometry().points().next().unwrap().coords(&mut coord);
            let cell = self
                .mesh
                .entity(ReferenceCellType::Interval, cell_index)
                .unwrap();
            let local_point = cell
                .topology()
                .sub_entity_iter(ReferenceCellType::Point)
                .position(|index| index == point_index)
                .expect("boundary point missing from owning interval");
            let normal = if local_point == 0 { -1.0 } else { 1.0 };
            let Some(kernel) = select(BoundaryPoint {
                index: point_index,
                coordinate: coord[0],
                normal,
            }) else {
                continue;
            };
            let mut reference_point = rlst_dynamic_array!(f64, [1, 1]);
            *reference_point.get_mut([0, 0]).unwrap() = local_point as f64;
            let mut table = DynArray::<f64, 4>::from_shape(element.tabulate_array_shape(0, 1));
            element.tabulate(&reference_point, 0, &mut table);
            let cell_dofs = space
                .entity_closure_dofs(ReferenceCellType::Interval, cell_index)
                .unwrap();
            let point_dofs = space
                .entity_closure_dofs(ReferenceCellType::Point, point_index)
                .unwrap();
            assert_eq!(point_dofs.len(), 1, "a scalar boundary point has one DOF");
            let cell_dof = cell_dofs
                .iter()
                .position(|&dof| dof == point_dofs[0])
                .expect("point DOF missing from owning interval");
            let values = [*table.get([0, 0, cell_dof, 0]).unwrap()];
            let points = [coord[0]];
            let normal = [normal];
            let ctx = FacetCtx {
                tdim: 0,
                gdim: 1,
                ncomp: 1,
                npts: 1,
                ndofs: 1,
                wts: &[1.0],
                jfacet_det: &[1.0],
                points: &points,
                normal: &normal,
                values: &values,
                grads: &[],
            };
            let mut local_rhs = [0.0];
            let mut local_mat = [0.0];
            kernel.assemble_facet_rhs(&ctx, &mut local_rhs);
            kernel.assemble_facet_mat(&ctx, &mut local_mat);
            if let Some(reduced) = self.target_dof(point_dofs[0]) {
                rhs[reduced] += local_rhs[0];
                if local_mat[0] != 0.0 {
                    triplets.push(Triplet::new(reduced, reduced, local_mat[0]));
                }
            }
        }
        let n = self.reduced_size();
        BoundaryContributions {
            rhs,
            mat: SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap(),
        }
    }
}

pub fn main() {
    let nx = 64;
    let p = 2;
    let nu = 0.001;
    let vel = 0.5;
    let sigma = 0.05;
    let x0 = 0.5;
    let dt = 0.01;
    let nsteps = 200;
    let snapshot_every = 20;

    let mesh: SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>> = unit_interval(nx);
    let problem =
        FiniteElement1DProblem::new(mesh, p, DofReduction1D::Periodic { facets: [0, nx] });
    let m_sparse = problem.assemble_lumped_mass();
    let k_sparse = problem.assemble_bilinear(&KernelAdvDiff::new(nu, vel));
    let n = m_sparse.nrows();
    println!(
        "reduced ndofs = {}, mass nnz = {}, adv_diff nnz = {}",
        n,
        m_sparse.compute_nnz(),
        k_sparse.compute_nnz()
    );

    let k_plain_dense = k_sparse.to_dense();
    let k_supg0 = problem
        .assemble_bilinear(&KernelAdvDiffSUPG::new(nu, vel, 0.0))
        .to_dense();
    let mut max_diff = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            max_diff = max_diff.max((k_plain_dense[(i, j)] - k_supg0[(i, j)]).abs());
        }
    }
    assert!(
        max_diff < 1e-12,
        "SUPG(tau=0) should equal plain adv-diff kernel"
    );

    let source = problem.assemble_linear(&KernelVolumeSource::new(1.0));
    assert!((source.iter().sum::<f64>() - 1.0).abs() < 1e-12);

    let sys = AdvDiffSys::new(m_sparse, k_sparse);
    let ones = Mat::<f64>::from_fn(n, 1, |_, _| 1.0);
    let a_one = sys.apply_minv_k(ones.as_ref());
    assert!(
        (0..n).map(|i| a_one[(i, 0)].abs()).fold(0.0_f64, f64::max) < 1e-12,
        "periodic constant must be in the null space"
    );

    let xs = problem.dof_positions();
    let mut y0 = Mat::<f64>::zeros(n, 1);
    for i in 0..n {
        y0[(i, 0)] = periodic_gaussian(xs[i], x0, sigma);
    }
    let mut solver =
        DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::implicit_euler(), 1e-12, 1e-12);
    let mut snapshots = vec![(0.0, (0..n).map(|i| y0[(i, 0)]).collect::<Vec<_>>())];
    for step in 1..=nsteps {
        let res = solver.step(&sys, dt).unwrap();
        let y = res.y.clone();
        if step % snapshot_every == 0 || step == nsteps {
            snapshots.push((res.t, (0..n).map(|i| y[(i, 0)]).collect()));
        }
        solver.accept_step(res);
    }

    let out_path = "target/ex_nd_1d_out.csv";
    let mut f = File::create(out_path).expect("failed to create output csv");
    writeln!(f, "t,x,u").unwrap();
    for (t, profile) in &snapshots {
        for (i, u) in profile.iter().enumerate() {
            writeln!(f, "{t:.6},{:.6},{u:.9e}", xs[i]).unwrap();
        }
    }
    println!(
        "wrote {} snapshots x {} dofs to {out_path}",
        snapshots.len(),
        n
    );
}

fn periodic_gaussian(x: f64, x0: f64, sigma: f64) -> f64 {
    let mut dx = x - x0;
    if dx > 0.5 {
        dx -= 1.0;
    } else if dx < -0.5 {
        dx += 1.0;
    }
    (-(dx * dx) / (2.0 * sigma * sigma)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct XSource;

    impl LinearForm for XSource {
        fn integrand(&self, ctx: &LocalCtx, q: usize, test_i: usize) -> f64 {
            ctx.point(q)[0] * ctx.test(test_i, 0).v(q)
        }
    }

    struct EndpointFlux;

    impl BoundaryIntegrator for EndpointFlux {
        fn integrand_rhs(&self, ctx: &FacetCtx, q: usize, test_i: usize) -> f64 {
            ctx.point(q)[0] * ctx.normal[0] * ctx.test(test_i, 0).v(q)
        }
    }

    fn mesh(nx: usize) -> SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>> {
        unit_interval(nx)
    }

    #[test]
    fn dirichlet_eliminates_selected_endpoint() {
        let problem = FiniteElement1DProblem::new(
            mesh(2),
            2,
            DofReduction1D::Dirichlet {
                facets_to_eliminate: vec![0],
            },
        );
        assert_eq!(problem.reduced_size(), 4);
        assert!(problem.target_dof(0).is_none());
    }

    #[test]
    fn periodic_identifies_selected_endpoints() {
        let problem =
            FiniteElement1DProblem::new(mesh(2), 2, DofReduction1D::Periodic { facets: [0, 2] });
        assert_eq!(problem.reduced_size(), 4);
        let space = FunctionSpaceImpl::new(&problem.mesh, &problem.family);
        let left = space
            .entity_closure_dofs(ReferenceCellType::Point, 0)
            .unwrap()[0];
        let right = space
            .entity_closure_dofs(ReferenceCellType::Point, 2)
            .unwrap()[0];
        assert_eq!(problem.target_dof(left), problem.target_dof(right));
    }

    #[test]
    fn volume_kernel_reads_physical_points() {
        let problem = FiniteElement1DProblem::new(mesh(2), 2, DofReduction1D::None);
        assert!((problem.assemble_linear(&XSource).iter().sum::<f64>() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn lumped_mass_matches_generic_gll_mass() {
        let problem =
            FiniteElement1DProblem::new(mesh(2), 2, DofReduction1D::Periodic { facets: [0, 2] });
        let generic = problem.assemble_bilinear(&KernelMass::new()).to_dense();
        let lumped = problem.assemble_lumped_mass().to_dense();
        for i in 0..generic.nrows() {
            for j in 0..generic.ncols() {
                assert!((generic[(i, j)] - lumped[(i, j)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn boundary_kernel_reads_endpoint_coordinates_and_normal() {
        let problem = FiniteElement1DProblem::new(mesh(1), 2, DofReduction1D::None);
        let flux = EndpointFlux;
        let boundary = problem.assemble_boundary(|_point| Some(&flux));
        assert!((boundary.rhs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}
