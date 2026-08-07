use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::matrix_free::LinOp;
use faer::prelude::*;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndfunctionspace::{traits::FunctionSpace, FunctionSpaceImpl};
use ndmesh::{shapes::unit_interval, SingleElementMesh};
use ormatex_sem_nd::{
    BoundaryIntegrator, DofReduction1D, FacetCtx, FiniteElement1DProblem, KernelMass, LinearForm,
    LocalCtx, MatrixFreeMinvJacobian, ResidualKernel,
};

type IntervalMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

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

struct QuadraticReaction;

impl ResidualKernel for QuadraticReaction {
    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &ormatex_sem_nd::CellField,
        q: usize,
        test_i: usize,
    ) -> f64 {
        state.value(q).powi(2) * ctx.test(test_i, 0).v(q)
    }

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx,
        state: &ormatex_sem_nd::CellField,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        2.0 * state.value(q) * ctx.test(test_i, 0).v(q) * ctx.trial(trial_i, 0).v(q)
    }
}

fn mesh(nx: usize) -> IntervalMesh {
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
    let boundary = problem.assemble_boundary(|_| Some(&flux));
    assert!((boundary.rhs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn matrix_free_minv_jacobian_matches_assembled_1d_action() {
    let problem = FiniteElement1DProblem::new(mesh(2), 2, DofReduction1D::None);
    let n = problem.reduced_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.1 * i as f64);
    let direction = Mat::from_fn(n, 1, |i, _| (0.4 * i as f64).sin());
    let mass = problem.assemble_lumped_mass();
    let m_inv: Vec<f64> = (0..n).map(|i| 1.0 / mass[(i, i)]).collect();
    let kernel = QuadraticReaction;
    let operator = MatrixFreeMinvJacobian::new(&problem, &kernel, state.clone(), None, &m_inv);
    let mut action = Mat::zeros(n, 1);
    let mut scratch = MemBuffer::new(StackReq::empty());
    operator.apply(
        action.as_mut(),
        direction.as_ref(),
        faer::get_global_parallelism(),
        MemStack::new(&mut scratch),
    );
    let expected = problem
        .assemble_residual_jacobian(&kernel, state.as_ref())
        .as_ref()
        * direction.as_ref();
    for row in 0..n {
        assert!((action[(row, 0)] + m_inv[row] * expected[(row, 0)]).abs() < 1e-11);
    }
}
