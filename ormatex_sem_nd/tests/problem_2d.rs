use faer::prelude::*;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndfunctionspace::{traits::FunctionSpace, FunctionSpaceImpl};
use ndmesh::{
    shapes::unit_square,
    traits::{Entity, Geometry, Mesh, Point},
    SingleElementMesh,
};
use ormatex_sem_nd::{
    BoundaryIntegrator, CellState, DofReduction2D, FacetCtx, KernelAdvDiff2D, KernelAdvDiffSUPG2D,
    KernelMass, LinearForm, LocalCtx, ResidualKernel, SEM2DProblem,
};
use ormatex_sem_nd::{ConstantCoefficient, MeshMetadata, PhysicalRegion, RegionCoefficient};

type QuadMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

struct XSource;

impl LinearForm for XSource {
    fn integrand(&self, ctx: &LocalCtx, _equation: usize, q: usize, test_i: usize) -> f64 {
        ctx.point(q)[0] * ctx.test(test_i, 0).v(q)
    }
}

struct YFlux;

impl BoundaryIntegrator for YFlux {
    fn integrand_rhs(&self, ctx: &FacetCtx, _equation: usize, q: usize, test_i: usize) -> f64 {
        ctx.point(q)[1] * ctx.test(test_i, 0).v(q)
    }
}

struct QuadraticReaction;

impl ResidualKernel for QuadraticReaction {
    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        _equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        state.value(0, q).powi(2) * ctx.test(test_i, 0).v(q)
    }

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        _equation: usize,
        _unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        2.0 * state.value(0, q) * ctx.test(test_i, 0).v(q) * ctx.trial(trial_i, 0).v(q)
    }
}

struct CoupledReaction;

impl ResidualKernel for CoupledReaction {
    fn nfields(&self) -> usize {
        2
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        let test = ctx.test(test_i, 0).v(q);
        match equation {
            0 => (state.value(0, q) + 2.0 * state.value(1, q)) * test,
            1 => (3.0 * state.value(0, q) - state.value(1, q)) * test,
            _ => unreachable!(),
        }
    }

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx,
        _state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        let coefficient = match (equation, unknown) {
            (0, 0) => 1.0,
            (0, 1) => 2.0,
            (1, 0) => 3.0,
            (1, 1) => -1.0,
            _ => unreachable!(),
        };
        coefficient * ctx.test(test_i, 0).v(q) * ctx.trial(trial_i, 0).v(q)
    }
}

#[test]
fn dirichlet_eliminates_every_high_order_facet_dof() {
    let mesh: QuadMesh = unit_square(2, 1, ReferenceCellType::Quadrilateral);
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
    let problem = SEM2DProblem::new(
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
    assert_eq!(facet_dofs.len(), 3);
    assert!(facet_dofs
        .iter()
        .all(|&dof| problem.target_dof(dof).is_none()));
    assert_eq!(problem.reduced_size(), 12);
}

#[test]
fn volume_kernel_reads_physical_quadrature_points() {
    let mesh: QuadMesh = unit_square(2, 1, ReferenceCellType::Quadrilateral);
    let problem = SEM2DProblem::new(mesh, 2, DofReduction2D::None);
    assert!((problem.assemble_linear(&XSource).iter().sum::<f64>() - 0.5).abs() < 1e-12);
}

#[test]
fn lumped_mass_matches_generic_gll_mass() {
    let mesh: QuadMesh = unit_square(2, 1, ReferenceCellType::Quadrilateral);
    let problem = SEM2DProblem::new(mesh, 2, DofReduction2D::Periodic);
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
    let mesh: QuadMesh = unit_square(1, 1, ReferenceCellType::Quadrilateral);
    let problem = SEM2DProblem::new(mesh, 2, DofReduction2D::None);
    let flux = YFlux;
    let boundary =
        problem.assemble_boundary(|facet| (facet.midpoint[0].abs() < 1e-12).then_some(&flux));
    assert!((boundary.rhs.iter().sum::<f64>() - 0.5).abs() < 1e-12);
}

#[test]
fn residual_kernel_jacobian_matches_directional_difference() {
    let mesh: QuadMesh = unit_square(1, 1, ReferenceCellType::Quadrilateral);
    let problem = SEM2DProblem::new(mesh, 2, DofReduction2D::None);
    let n = problem.reduced_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.1 * i as f64);
    let direction = Mat::from_fn(n, 1, |i, _| (0.3 * i as f64).sin());
    let kernel = QuadraticReaction;
    let action = problem.apply_jacobian_matfree(&kernel, state.as_ref(), direction.as_ref());
    let eps = 1e-7;
    let perturbed = state.as_ref() + faer::Scale(eps) * direction.as_ref();
    let residual = problem.assemble_residual(&kernel, state.as_ref());
    let perturbed_residual = problem.assemble_residual(&kernel, perturbed.as_ref());
    for i in 0..n {
        assert!((action[(i, 0)] - (perturbed_residual[i] - residual[i]) / eps).abs() < 1e-7);
    }
}

#[test]
fn coupled_2d_system_assembles_cross_field_blocks_and_matrix_free_action() {
    let mesh: QuadMesh = unit_square(1, 1, ReferenceCellType::Quadrilateral);
    let problem = SEM2DProblem::new(mesh, 2, DofReduction2D::None);
    let n = problem.reduced_size();
    let state = Mat::from_fn(2 * n, 1, |i, _| 0.2 + 0.02 * i as f64);
    let direction = Mat::from_fn(2 * n, 1, |i, _| (0.17 * i as f64).sin());
    let kernel = CoupledReaction;
    let assembled = problem
        .assemble_system_residual_jacobian(&kernel, state.as_ref())
        .to_dense();
    assert!(assembled[(0, n)] != 0.0);
    assert!(assembled[(n, 0)] != 0.0);
    let action = problem.apply_system_jacobian_matfree(&kernel, state.as_ref(), direction.as_ref());
    let expected = assembled.as_ref() * direction.as_ref();
    for row in 0..2 * n {
        assert!((action[(row, 0)] - expected[(row, 0)]).abs() < 1e-11);
    }
}

#[test]
fn supg_2d_zero_tau_matches_advection_diffusion() {
    let mesh: QuadMesh = unit_square(1, 1, ReferenceCellType::Quadrilateral);
    let problem = SEM2DProblem::new(mesh, 2, DofReduction2D::None);
    let plain = problem
        .assemble_bilinear(&KernelAdvDiff2D::new(0.1, [0.4, -0.2]))
        .to_dense();
    let supg = problem
        .assemble_bilinear(&KernelAdvDiffSUPG2D::new(0.1, [0.4, -0.2], 0.0))
        .to_dense();
    for i in 0..plain.nrows() {
        for j in 0..plain.ncols() {
            assert!((plain[(i, j)] - supg[(i, j)]).abs() < 1e-12);
        }
    }
}

#[test]
fn region_coefficient_changes_2d_material_operator() {
    let region = PhysicalRegion {
        dimension: 2,
        tag: 7,
    };
    let metadata = MeshMetadata {
        cell_regions: vec![Some(region)],
        ..MeshMetadata::default()
    };
    let mesh: QuadMesh = unit_square(1, 1, ReferenceCellType::Quadrilateral);
    let problem = SEM2DProblem::new_with_metadata(mesh, 2, DofReduction2D::None, metadata);
    let region_diffusion =
        RegionCoefficient::new(std::collections::HashMap::from([(region, 3.0)]), 1.0);
    let region_kernel = KernelAdvDiff2D::with_coefficients(
        region_diffusion,
        [ConstantCoefficient(0.0), ConstantCoefficient(0.0)],
    );
    let selected = problem
        .assemble_system_bilinear_at(0.0, &region_kernel)
        .to_dense();
    let expected = problem
        .assemble_bilinear(&KernelAdvDiff2D::new(3.0, [0.0, 0.0]))
        .to_dense();
    for i in 0..selected.nrows() {
        for j in 0..selected.ncols() {
            assert!((selected[(i, j)] - expected[(i, j)]).abs() < 1e-12);
        }
    }
}
