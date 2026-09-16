use std::path::Path;

use faer::matrix_free::LinOp;
use faer::prelude::*;
use ormatex::matexp_krylov::KrylovExpm;
use ormatex::matexp_leja::{LejaEllipseAdapterArnoldiIOM, LejaPhiEval, LejaPoints};
use ormatex::matexp_pade::PadeExpm;
use ormatex::ode_epirk::EpirkIntegrator;
use ormatex::ode_implicit::DirkIntegrator;
use ormatex::ode_sys::{IntegrateSys, OdeSys};
use ormatex::tableau_implicit::ImplicitBT;
use ormatex_sem_nd::{
    CellState, KernelEdacDirectionalDoNothing2D, KernelEdacDongOutflow2D, KernelEdacNoSlipWall2D,
    KernelEdacSlipWall2D, KernelEdacSplitBoundaryFlux2D, LocalCtx, MatrixFreeMinvJacobian,
    OwnedMinvJacobian, ParallelOwnedMinvJacobian, QuadMesh, ResidualKernel, SEM2DProblem,
    StateBoundaryTerms, StateTensorBoundaryTerms, TensorKernelEdacDirectionalDoNothing2D,
    TensorKernelEdacDongOutflow2D, TensorKernelEdacNoSlipWall2D, TensorKernelEdacSlipWall2D,
    TensorKernelEdacSplitBoundaryFlux2D, TensorResidualKernel,
};

use super::linear_system::lumped_inverse_mass;
use ormatex_sem_nd::BilinearOps;

/// Retains only the traditional weak-form kernel interface for comparison.
/// Example binaries use this as a correctness and performance oracle.
pub struct GenericResidual<K>(pub K);

impl<K: ResidualKernel> ResidualKernel for GenericResidual<K> {
    fn nfields(&self) -> usize {
        self.0.nfields()
    }

    fn field_names(&self) -> Option<Vec<String>> {
        self.0.field_names()
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        self.0.residual_integrand(ctx, state, equation, q, test_i)
    }

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        self.0
            .jacobian_integrand(ctx, state, equation, unknown, q, test_i, trial_i)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum JacobianBackend {
    MatrixFree,
    Assembled,
    ParallelAssembled,
}

impl JacobianBackend {
    fn from_args() -> Self {
        let mut assembled = false;
        let mut matrix_free = false;
        for arg in std::env::args() {
            match arg.as_str() {
                "--assembled-jacobian" => assembled = true,
                "--matrix-free" => matrix_free = true,
                _ => {}
            }
        }
        assert!(
            !(assembled && matrix_free),
            "choose only one of --assembled-jacobian and --matrix-free"
        );
        if assembled {
            Self::ParallelAssembled
        } else {
            Self::MatrixFree
        }
    }
}

/// Write one 2D solved state as standardized `x,y,field_0,...` CSV.
pub fn write_solution_csv(
    problem: &SEM2DProblem<QuadMesh>,
    state: MatRef<'_, f64>,
    path: impl AsRef<Path>,
) {
    ormatex_sem_nd::io::write_csv(path, &ormatex_sem_nd::io::export_2d(problem, state));
}

/// Write the `.vtu` solution when `--vtk` was passed; no-op otherwise.
/// Panics with a build hint when the `vtk` feature is off.
pub fn maybe_write_vtk(
    problem: &SEM2DProblem<QuadMesh>,
    state: MatRef<'_, f64>,
    default_vtu: &str,
) {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = ormatex_sem_nd::io::vtk_output_path(&args, default_vtu) else {
        return;
    };
    #[cfg(feature = "vtk")]
    {
        ormatex_sem_nd::io::write_sem2d_vtu(problem, state, &path);
        println!("vtk solution: {path}");
    }
    #[cfg(not(feature = "vtk"))]
    {
        let _ = (problem, state);
        panic!("--vtk ({path}) requires building with `--features vtk`");
    }
}

/// `--vtk` export for a pre-built export mesh (multi-problem outputs like the
/// frozen-velocity species example); no-op without the flag.
pub fn maybe_write_vtk_mesh(mesh: &ormatex_sem_nd::io::ExportMesh, default_vtu: &str) {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = ormatex_sem_nd::io::vtk_output_path(&args, default_vtu) else {
        return;
    };
    #[cfg(feature = "vtk")]
    {
        ormatex_sem_nd::io::write_vtu(&path, mesh);
        println!("vtk solution: {path}");
    }
    #[cfg(not(feature = "vtk"))]
    {
        let _ = mesh;
        panic!("--vtk ({path}) requires building with `--features vtk`");
    }
}

pub struct FluidSystem<'a, K> {
    pub problem: &'a SEM2DProblem<QuadMesh>,
    pub kernel: K,
    m_inv: Vec<f64>,
    terms: StateBoundaryTerms,
    backend: JacobianBackend,
}

pub struct TensorFluidSystem<'a, K> {
    pub problem: &'a SEM2DProblem<QuadMesh>,
    pub kernel: K,
    m_inv: Vec<f64>,
    terms: StateTensorBoundaryTerms<2>,
    backend: JacobianBackend,
}

impl<'a, K> FluidSystem<'a, K> {
    pub fn new(problem: &'a SEM2DProblem<QuadMesh>, kernel: K) -> Self {
        Self::new_with_backend(problem, kernel, JacobianBackend::from_args())
    }

    pub fn new_with_backend(
        problem: &'a SEM2DProblem<QuadMesh>,
        kernel: K,
        backend: JacobianBackend,
    ) -> Self {
        let mass = problem.assemble_lumped_mass();
        Self {
            problem,
            kernel,
            m_inv: lumped_inverse_mass(mass.as_ref()),
            terms: StateBoundaryTerms::new(),
            backend,
        }
    }

    pub fn with_state_boundary(mut self, terms: StateBoundaryTerms) -> Self {
        self.terms = terms;
        self
    }

    pub fn with_split_boundary(self) -> Self {
        self.with_state_boundary(
            StateBoundaryTerms::new().with_default(KernelEdacSplitBoundaryFlux2D),
        )
    }

    pub fn with_wall_boundaries(self, no_slip_facets: Vec<usize>, slip_facets: Vec<usize>) -> Self {
        let terms = StateBoundaryTerms::new()
            .with_default(KernelEdacSplitBoundaryFlux2D)
            .with_entities(no_slip_facets, KernelEdacNoSlipWall2D)
            .with_entities(slip_facets, KernelEdacSlipWall2D);
        self.with_state_boundary(terms)
    }

    pub fn with_dong_outflow(
        self,
        kernel: KernelEdacDongOutflow2D,
        facets: Vec<usize>,
        split_form: bool,
    ) -> Self {
        assert!(
            !facets.is_empty(),
            "Dong outflow requires at least one facet"
        );
        let terms = if split_form {
            self.terms
                .clone()
                .with_default(KernelEdacSplitBoundaryFlux2D)
                .with_entities(facets, kernel.with_split_flux())
        } else {
            self.terms.clone().with_entities(facets, kernel)
        };
        self.with_state_boundary(terms)
    }

    pub fn with_directional_do_nothing_outflow(
        self,
        kernel: KernelEdacDirectionalDoNothing2D,
        facets: Vec<usize>,
        split_form: bool,
    ) -> Self {
        assert!(
            !facets.is_empty(),
            "directional do-nothing outflow requires at least one facet"
        );
        let terms = if split_form {
            StateBoundaryTerms::new()
                .with_default(KernelEdacSplitBoundaryFlux2D)
                .with_entities(facets, kernel.with_split_flux())
        } else {
            StateBoundaryTerms::new().with_entities(facets, kernel)
        };
        self.with_state_boundary(terms)
    }
}

impl<'a, K> TensorFluidSystem<'a, K> {
    pub fn new(problem: &'a SEM2DProblem<QuadMesh>, kernel: K) -> Self {
        Self::new_with_backend(problem, kernel, JacobianBackend::from_args())
    }

    pub fn new_with_backend(
        problem: &'a SEM2DProblem<QuadMesh>,
        kernel: K,
        backend: JacobianBackend,
    ) -> Self {
        let mass = problem.assemble_lumped_mass();
        Self {
            problem,
            kernel,
            m_inv: lumped_inverse_mass(mass.as_ref()),
            terms: StateTensorBoundaryTerms::new(),
            backend,
        }
    }

    pub fn with_state_boundary(mut self, terms: StateTensorBoundaryTerms<2>) -> Self {
        self.terms = terms;
        self
    }

    pub fn with_split_boundary(self) -> Self {
        self.with_state_boundary(
            StateTensorBoundaryTerms::new().with_default(TensorKernelEdacSplitBoundaryFlux2D),
        )
    }

    pub fn with_wall_boundaries(self, no_slip_facets: Vec<usize>, slip_facets: Vec<usize>) -> Self {
        let terms = StateTensorBoundaryTerms::new()
            .with_default(TensorKernelEdacSplitBoundaryFlux2D)
            .with_entities(no_slip_facets, TensorKernelEdacNoSlipWall2D)
            .with_entities(slip_facets, TensorKernelEdacSlipWall2D);
        self.with_state_boundary(terms)
    }

    pub fn with_dong_outflow(
        self,
        kernel: TensorKernelEdacDongOutflow2D,
        facets: Vec<usize>,
        split_form: bool,
    ) -> Self {
        assert!(
            !facets.is_empty(),
            "Dong outflow requires at least one facet"
        );
        let terms = if split_form {
            self.terms
                .clone()
                .with_default(TensorKernelEdacSplitBoundaryFlux2D)
                .with_entities(facets, kernel.with_split_flux())
        } else {
            self.terms.clone().with_entities(facets, kernel)
        };
        self.with_state_boundary(terms)
    }

    pub fn with_directional_do_nothing_outflow(
        self,
        kernel: TensorKernelEdacDirectionalDoNothing2D,
        facets: Vec<usize>,
        split_form: bool,
    ) -> Self {
        assert!(
            !facets.is_empty(),
            "directional do-nothing outflow requires at least one facet"
        );
        let terms = if split_form {
            self.terms
                .clone()
                .with_default(TensorKernelEdacSplitBoundaryFlux2D)
                .with_entities(facets, kernel.with_split_flux())
        } else {
            self.terms.clone().with_entities(facets, kernel)
        };
        self.with_state_boundary(terms)
    }
}

impl<'a, K> OdeSys<'a> for FluidSystem<'a, K>
where
    K: ResidualKernel + Sync + Send,
{
    fn frhs(&self, t: f64, state: MatRef<f64>) -> Mat<f64> {
        let operator = self
            .problem
            .residual_operator(&self.kernel)
            .at_time(t)
            .with_state_boundary(&self.terms);
        let residual = operator.residual(state);
        Mat::from_fn(self.m_inv.len(), 1, |row, _| {
            -self.m_inv[row] * residual[row]
        })
    }

    fn fjac<'b>(&'a self, t: f64, state: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
        let operator = self
            .problem
            .residual_operator(&self.kernel)
            .at_time(t)
            .with_state_boundary(&self.terms);
        match self.backend {
            // The complete operator assembly includes state-dependent boundary Jacobians.
            JacobianBackend::Assembled => Box::new(OwnedMinvJacobian::new(
                operator.assemble_jacobian(state),
                &self.m_inv,
            )),
            JacobianBackend::ParallelAssembled => Box::new(ParallelOwnedMinvJacobian::new(
                operator.assemble_jacobian(state),
                &self.m_inv,
            )),
            JacobianBackend::MatrixFree => Box::new(MatrixFreeMinvJacobian::new(
                operator,
                state.to_owned(),
                &self.m_inv,
            )),
        }
    }
}

impl<'a, K> OdeSys<'a> for TensorFluidSystem<'a, K>
where
    K: TensorResidualKernel<2> + Sync + Send,
{
    fn frhs(&self, t: f64, state: MatRef<f64>) -> Mat<f64> {
        let residual = self
            .problem
            .tensor_residual_operator(&self.kernel)
            .at_time(t)
            .with_state_boundary(&self.terms)
            .residual(state);
        Mat::from_fn(self.m_inv.len(), 1, |row, _| {
            -self.m_inv[row] * residual[row]
        })
    }

    fn fjac<'b>(&'a self, t: f64, state: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
        let operator = self
            .problem
            .tensor_residual_operator(&self.kernel)
            .at_time(t)
            .with_state_boundary(&self.terms);
        match self.backend {
            JacobianBackend::Assembled => Box::new(OwnedMinvJacobian::new(
                operator.assemble_jacobian(state),
                &self.m_inv,
            )),
            JacobianBackend::ParallelAssembled => Box::new(ParallelOwnedMinvJacobian::new(
                operator.assemble_jacobian(state),
                &self.m_inv,
            )),
            JacobianBackend::MatrixFree => Box::new(MatrixFreeMinvJacobian::new(
                operator,
                state.to_owned(),
                &self.m_inv,
            )),
        }
    }
}

pub fn epi3(state0: MatRef<'_, f64>) -> EpirkIntegrator<KrylovExpm> {
    let expmv = Box::new(PadeExpm::new(12));
    let krylov = KrylovExpm::new(expmv, 30, 100, 1e-12, Some(2));
    EpirkIntegrator::new(0.0, state0, "epi3".to_string(), krylov)
}

/// SDIRK32 (3 stages, order 2, L-stable) implicit stepper for stiff systems.
pub fn sdirk32(state0: MatRef<'_, f64>) -> DirkIntegrator<'_> {
    DirkIntegrator::new(0.0, state0, ImplicitBT::sdirk32(), 1e-10, 1e-10)
}

/// Time-stepper option for the tensor examples: EPI3 exponential by default,
/// SDIRK32 implicit with `--sdirk32`.
pub enum TimeStepper<'a> {
    Epi3(EpirkIntegrator<KrylovExpm>),
    Sdirk32(DirkIntegrator<'a>),
}

impl<'a> TimeStepper<'a> {
    pub fn new(state0: MatRef<'a, f64>) -> Self {
        if std::env::args().any(|arg| arg == "--sdirk32") {
            Self::Sdirk32(sdirk32(state0))
        } else {
            Self::Epi3(epi3(state0))
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Epi3(_) => "epi3",
            Self::Sdirk32(_) => "sdirk32",
        }
    }

    pub fn run_steps<'b>(
        &mut self,
        system: &'b dyn OdeSys<'b>,
        dt: f64,
        nsteps: usize,
        label: &str,
    ) {
        for step in 0..nsteps {
            match self {
                Self::Epi3(integrator) => {
                    let result = integrator.step(system, dt).unwrap_or_else(|error| {
                        panic!("{label} step {step} failed: {}", error.msg)
                    });
                    integrator.accept_step(result);
                }
                Self::Sdirk32(integrator) => {
                    let result = integrator.step(system, dt).unwrap_or_else(|error| {
                        panic!("{label} step {step} failed: {}", error.msg)
                    });
                    integrator.accept_step(result);
                }
            }
        }
    }

    pub fn state(&self) -> Mat<f64> {
        match self {
            Self::Epi3(integrator) => integrator.state(),
            Self::Sdirk32(integrator) => integrator.state(),
        }
    }

    pub fn time(&self) -> f64 {
        match self {
            Self::Epi3(integrator) => integrator.time(),
            Self::Sdirk32(integrator) => integrator.time(),
        }
    }
}

pub fn epi3_leja(state0: MatRef<'_, f64>) -> EpirkIntegrator<LejaPhiEval> {
    let points = LejaPoints::new_from_fn("leja_circle").slice(0, 800);
    let adapter = LejaEllipseAdapterArnoldiIOM::new(-1.0, 0.0, 1.0, 1e-8, 30, 2, 1.0);
    let leja = LejaPhiEval::new(
        points,
        100,
        1e-12,
        "clapm",
        "dd_taylor",
        false,
        Box::new(adapter),
    );
    EpirkIntegrator::new(0.0, state0, "epi3".to_string(), leja)
}

pub fn advance(
    system: &FluidSystem<'_, impl ResidualKernel + Sync + Send>,
    state0: MatRef<'_, f64>,
    dt: f64,
    nsteps: usize,
) -> Mat<f64> {
    let mut integrator = epi3(state0);
    for step in 0..nsteps {
        let result = integrator
            .step(system, dt)
            .unwrap_or_else(|error| panic!("EDAC step {step} failed: {}", error.msg));
        integrator.accept_step(result);
    }
    integrator.state()
}

pub fn advance_tensor(
    system: &TensorFluidSystem<'_, impl TensorResidualKernel<2> + Sync + Send>,
    state0: MatRef<'_, f64>,
    dt: f64,
    nsteps: usize,
) -> Mat<f64> {
    let mut integrator = epi3(state0);
    for step in 0..nsteps {
        let result = integrator
            .step(system, dt)
            .unwrap_or_else(|error| panic!("EDAC step {step} failed: {}", error.msg));
        integrator.accept_step(result);
    }
    integrator.state()
}

pub fn advance_tensor_leja(
    system: &TensorFluidSystem<'_, impl TensorResidualKernel<2> + Sync + Send>,
    state0: MatRef<'_, f64>,
    dt: f64,
    nsteps: usize,
) -> Mat<f64> {
    let mut integrator = epi3_leja(state0);
    for step in 0..nsteps {
        let result = integrator
            .step(system, dt)
            .unwrap_or_else(|error| panic!("EDAC step {step} failed: {}", error.msg));
        integrator.accept_step(result);
    }
    integrator.state()
}
