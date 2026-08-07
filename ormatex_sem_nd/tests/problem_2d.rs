use faer::prelude::*;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndfunctionspace::{traits::FunctionSpace, FunctionSpaceImpl};
use ndmesh::{
    shapes::unit_square,
    traits::{Entity, Geometry, Mesh, Point},
    SingleElementMesh,
};
use ormatex_sem_nd::{
    BoundaryIntegrator, CellField, DofReduction2D, FacetCtx, FiniteElement2DProblem, KernelMass,
    LinearForm, LocalCtx, ResidualKernel,
};

type QuadMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

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

struct QuadraticReaction;

impl ResidualKernel for QuadraticReaction {
    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellField,
        q: usize,
        test_i: usize,
    ) -> f64 {
        state.value(q).powi(2) * ctx.test(test_i, 0).v(q)
    }

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellField,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        2.0 * state.value(q) * ctx.test(test_i, 0).v(q) * ctx.trial(trial_i, 0).v(q)
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
    assert_eq!(facet_dofs.len(), 3);
    assert!(facet_dofs
        .iter()
        .all(|&dof| problem.target_dof(dof).is_none()));
    assert_eq!(problem.reduced_size(), 12);
}

#[test]
fn volume_kernel_reads_physical_quadrature_points() {
    let mesh: QuadMesh = unit_square(2, 1, ReferenceCellType::Quadrilateral);
    let problem = FiniteElement2DProblem::new(mesh, 2, DofReduction2D::None);
    assert!((problem.assemble_linear(&XSource).iter().sum::<f64>() - 0.5).abs() < 1e-12);
}

#[test]
fn lumped_mass_matches_generic_gll_mass() {
    let mesh: QuadMesh = unit_square(2, 1, ReferenceCellType::Quadrilateral);
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
    let mesh: QuadMesh = unit_square(1, 1, ReferenceCellType::Quadrilateral);
    let problem = FiniteElement2DProblem::new(mesh, 2, DofReduction2D::None);
    let flux = YFlux;
    let boundary =
        problem.assemble_boundary(|facet| (facet.midpoint[0].abs() < 1e-12).then_some(&flux));
    assert!((boundary.rhs.iter().sum::<f64>() - 0.5).abs() < 1e-12);
}

#[test]
fn residual_kernel_jacobian_matches_directional_difference() {
    let mesh: QuadMesh = unit_square(1, 1, ReferenceCellType::Quadrilateral);
    let problem = FiniteElement2DProblem::new(mesh, 2, DofReduction2D::None);
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
