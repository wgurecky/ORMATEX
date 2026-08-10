use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::matrix_free::LinOp;
use faer::prelude::*;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndfunctionspace::{traits::FunctionSpace, FunctionSpaceImpl};
use ndmesh::{shapes::unit_interval, SingleElementMesh};
use ormatex_sem_nd::material::{
    CellMeta, Coefficient, ConstantCoefficient, MaterialContext, MaterialProperty, PhysicalRegion,
    RegionCoefficient,
};
use ormatex_sem_nd::{
    BoundaryIntegrator, CellState, DofReduction1D, FacetCtx, FluxKernel1D, KernelAdvDiff,
    KernelMass, LinearForm, LocalCtx, MatrixFreeMinvJacobian, ResidualKernel, SEM1DProblem,
};

#[path = "../examples/support/euler_1d.rs"]
mod euler_1d;
#[path = "../examples/support/isothermal_euler.rs"]
mod isothermal_euler;
use euler_1d::Euler1D;
use isothermal_euler::IsothermalEuler1D;

type IntervalMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

struct XSource;

impl LinearForm for XSource {
    fn integrand(&self, ctx: &LocalCtx, _equation: usize, q: usize, test_i: usize) -> f64 {
        ctx.point(q)[0] * ctx.test(test_i, 0).v(q)
    }
}

struct EndpointFlux;

impl BoundaryIntegrator for EndpointFlux {
    fn integrand_rhs(&self, ctx: &FacetCtx, _equation: usize, q: usize, test_i: usize) -> f64 {
        ctx.point(q)[0] * ctx.normal[0] * ctx.test(test_i, 0).v(q)
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

struct TemperatureDiffusion;

impl Coefficient<f64> for TemperatureDiffusion {
    fn eval(&self, ctx: &MaterialContext<'_>) -> f64 {
        1.0 + ctx.state.expect("temperature is required").value(0, ctx.q)
    }
}

impl MaterialProperty<f64> for TemperatureDiffusion {
    fn derivative(&self, _ctx: &MaterialContext<'_>, field: usize) -> Option<f64> {
        (field == 0).then_some(1.0)
    }
}

fn flux_context() -> LocalCtx<'static> {
    LocalCtx {
        time: 0.0,
        cell: CellMeta::default(),
        tdim: 1,
        gdim: 1,
        ncomp: 1,
        npts: 1,
        ndofs: 1,
        wts: &[],
        jdets: &[],
        points: &[],
        values: &[],
        grads: &[],
    }
}

fn mesh(nx: usize) -> IntervalMesh {
    unit_interval(nx)
}

#[test]
fn dirichlet_eliminates_selected_endpoint() {
    let problem = SEM1DProblem::new(
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
    let problem = SEM1DProblem::new(mesh(2), 2, DofReduction1D::Periodic { facets: [0, 2] });
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
    let problem = SEM1DProblem::new(mesh(2), 2, DofReduction1D::None);
    assert!((problem.assemble_linear(&XSource).iter().sum::<f64>() - 0.5).abs() < 1e-12);
}

#[test]
fn lumped_mass_matches_generic_gll_mass() {
    let problem = SEM1DProblem::new(mesh(2), 2, DofReduction1D::Periodic { facets: [0, 2] });
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
    let problem = SEM1DProblem::new(mesh(1), 2, DofReduction1D::None);
    let flux = EndpointFlux;
    let boundary = problem.assemble_boundary(|_| Some(&flux));
    assert!((boundary.rhs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn matrix_free_minv_jacobian_matches_assembled_1d_action() {
    let problem = SEM1DProblem::new(mesh(2), 2, DofReduction1D::None);
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

#[test]
fn coupled_system_assembles_cross_field_blocks_and_matrix_free_action() {
    let problem = SEM1DProblem::new(mesh(2), 2, DofReduction1D::None);
    let n = problem.reduced_size();
    let state = Mat::from_fn(2 * n, 1, |i, _| 0.2 + 0.03 * i as f64);
    let direction = Mat::from_fn(2 * n, 1, |i, _| (0.2 * i as f64).sin());
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
fn coupled_lumped_mass_repeats_scalar_blocks_without_cross_terms() {
    let problem = SEM1DProblem::new(mesh(2), 2, DofReduction1D::None);
    let n = problem.reduced_size();
    let scalar = problem.assemble_lumped_mass().to_dense();
    let block = problem.assemble_system_lumped_mass(2).to_dense();
    assert_eq!(block.nrows(), 2 * n);
    for i in 0..n {
        for j in 0..n {
            assert!((block[(i, j)] - scalar[(i, j)]).abs() < 1e-12);
            assert!((block[(n + i, n + j)] - scalar[(i, j)]).abs() < 1e-12);
            assert_eq!(block[(i, n + j)], 0.0);
            assert_eq!(block[(n + i, j)], 0.0);
        }
    }
}

#[test]
fn coupled_system_jacobian_matches_directional_difference() {
    let problem = SEM1DProblem::new(mesh(1), 2, DofReduction1D::None);
    let n = problem.reduced_size();
    let state = Mat::from_fn(2 * n, 1, |i, _| 0.2 + 0.03 * i as f64);
    let direction = Mat::from_fn(2 * n, 1, |i, _| (0.3 * i as f64).cos());
    let kernel = CoupledReaction;
    let action = problem.apply_system_jacobian_matfree(&kernel, state.as_ref(), direction.as_ref());
    let eps = 1e-7;
    let perturbed = state.as_ref() + faer::Scale(eps) * direction.as_ref();
    let residual = problem.assemble_system_residual(&kernel, state.as_ref());
    let perturbed_residual = problem.assemble_system_residual(&kernel, perturbed.as_ref());
    for i in 0..2 * n {
        assert!((action[(i, 0)] - (perturbed_residual[i] - residual[i]) / eps).abs() < 1e-7);
    }
}

#[test]
fn isothermal_euler_flux_jacobian_matches_finite_difference() {
    let kernel = IsothermalEuler1D::new(1.3);
    let ctx = flux_context();
    let state = CellState {
        nfields: 2,
        npts: 1,
        gdim: 1,
        values: &[0.4, 1.2],
        grads: &[0.0, 0.0],
    };
    let eps = 1e-7;
    for equation in 0..2 {
        for unknown in 0..2 {
            let mut plus = state.values.to_vec();
            plus[unknown] += eps;
            let plus_state = CellState {
                values: &plus,
                ..state
            };
            let derivative = (kernel.flux(&ctx, &plus_state, equation, 0)
                - kernel.flux(&ctx, &state, equation, 0))
                / eps;
            assert!(
                (derivative - kernel.flux_jacobian(&ctx, &state, equation, unknown, 0)).abs()
                    < 1e-7
            );
        }
    }
}

#[test]
fn conservative_euler_flux_jacobian_matches_finite_difference() {
    let kernel = Euler1D::new(1.4);
    let state = CellState {
        nfields: 3,
        npts: 1,
        gdim: 1,
        values: &[1.0, 0.3, 2.6],
        grads: &[0.0, 0.0, 0.0],
    };
    let eps = 1e-7;
    let ctx = flux_context();
    for equation in 0..3 {
        for unknown in 0..3 {
            let mut plus = state.values.to_vec();
            plus[unknown] += eps;
            let plus_state = CellState {
                values: &plus,
                ..state
            };
            let derivative = (kernel.flux(&ctx, &plus_state, equation, 0)
                - kernel.flux(&ctx, &state, equation, 0))
                / eps;
            assert!(
                (derivative - kernel.flux_jacobian(&ctx, &state, equation, unknown, 0)).abs()
                    < 1e-6,
                "Euler flux Jacobian mismatch at ({equation}, {unknown})"
            );
        }
    }
}

#[test]
fn coefficient_receives_explicit_time_and_space() {
    let problem = SEM1DProblem::new(mesh(1), 2, DofReduction1D::None);
    let kernel = KernelAdvDiff::with_coefficients(
        |ctx: &MaterialContext<'_>| 1.0 + ctx.time + ctx.point[0],
        ConstantCoefficient(0.0),
    );
    let at_zero = problem.assemble_system_bilinear_at(0.0, &kernel).to_dense();
    let at_one = problem.assemble_system_bilinear_at(1.0, &kernel).to_dense();
    assert!((0..at_zero.nrows())
        .flat_map(|i| (0..at_zero.ncols()).map(move |j| (i, j)))
        .any(|(i, j)| (at_one[(i, j)] - at_zero[(i, j)]).abs() > 1e-12));
}

#[test]
fn nonlinear_coefficient_derivative_is_included_in_jacobian() {
    let problem = SEM1DProblem::new(mesh(1), 2, DofReduction1D::None);
    let n = problem.reduced_size();
    let kernel = KernelAdvDiff::with_coefficients(TemperatureDiffusion, ConstantCoefficient(0.0));
    let state = Mat::from_fn(n, 1, |i, _| 1.0 + 0.2 * i as f64);
    let direction = Mat::from_fn(n, 1, |i, _| (0.4 * i as f64).sin());
    let action =
        problem.apply_system_jacobian_matfree_at(0.0, &kernel, state.as_ref(), direction.as_ref());
    let eps = 1e-7;
    let perturbed = state.as_ref() + faer::Scale(eps) * direction.as_ref();
    let residual = problem.assemble_system_residual_at(0.0, &kernel, state.as_ref());
    let perturbed_residual = problem.assemble_system_residual_at(0.0, &kernel, perturbed.as_ref());
    for i in 0..n {
        assert!((action[(i, 0)] - (perturbed_residual[i] - residual[i]) / eps).abs() < 1e-7);
    }
}

#[test]
fn matrix_free_jacobian_uses_explicit_material_time() {
    let problem = SEM1DProblem::new(mesh(1), 2, DofReduction1D::None);
    let n = problem.reduced_size();
    let kernel = KernelAdvDiff::with_coefficients(
        |ctx: &MaterialContext<'_>| 1.0 + ctx.time,
        ConstantCoefficient(0.0),
    );
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.1 * i as f64);
    let direction = Mat::from_fn(n, 1, |i, _| (0.4 * i as f64).sin());
    let mass = problem.assemble_lumped_mass();
    let m_inv: Vec<f64> = (0..n).map(|i| 1.0 / mass[(i, i)]).collect();
    let operator =
        MatrixFreeMinvJacobian::new_at(2.0, &problem, &kernel, state.clone(), None, &m_inv);
    let mut action = Mat::zeros(n, 1);
    let mut scratch = MemBuffer::new(StackReq::empty());
    operator.apply(
        action.as_mut(),
        direction.as_ref(),
        faer::get_global_parallelism(),
        MemStack::new(&mut scratch),
    );
    let expected = problem
        .assemble_system_residual_jacobian_at(2.0, &kernel, state.as_ref())
        .as_ref()
        * direction.as_ref();
    for row in 0..n {
        assert!((action[(row, 0)] + m_inv[row] * expected[(row, 0)]).abs() < 1e-11);
    }
}

#[test]
fn region_coefficient_selects_region_and_default() {
    let region = PhysicalRegion {
        dimension: 2,
        tag: 7,
    };
    let mut values = std::collections::HashMap::new();
    values.insert(region, 3.5);
    let coefficient = RegionCoefficient::new(values, 1.25);
    let selected = MaterialContext {
        time: 0.0,
        point: &[],
        cell: CellMeta {
            local_index: 0,
            physical_region: Some(region),
        },
        state: None,
        q: 0,
    };
    let fallback = MaterialContext {
        cell: CellMeta::default(),
        ..selected
    };
    assert_eq!(coefficient.eval(&selected), 3.5);
    assert_eq!(coefficient.eval(&fallback), 1.25);
}
