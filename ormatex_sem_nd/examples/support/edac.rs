use faer::matrix_free::LinOp;
use faer::prelude::*;
use ormatex::matexp_krylov::KrylovExpm;
use ormatex::matexp_pade::PadeExpm;
use ormatex::ode_epirk::EpirkIntegrator;
use ormatex::ode_sys::{IntegrateSys, OdeSys};
use ormatex_sem_nd::{
    KernelEdacDongOutflow2D, KernelEdacSplitBoundaryFlux2D, MatrixFreeMinvCompleteJacobian,
    QuadMesh, ResidualKernel, SEM2DProblem, StateBoundaryTerms,
};

use super::linear_system::lumped_inverse_mass;

pub struct FluidSystem<'a, K> {
    pub problem: &'a SEM2DProblem<QuadMesh>,
    pub kernel: K,
    m_inv: Vec<f64>,
    terms: StateBoundaryTerms,
}

impl<'a, K> FluidSystem<'a, K> {
    pub fn new(problem: &'a SEM2DProblem<QuadMesh>, kernel: K) -> Self {
        let mass = problem.assemble_system_lumped_mass();
        Self {
            problem,
            kernel,
            m_inv: lumped_inverse_mass(mass.as_ref()),
            terms: StateBoundaryTerms::new(),
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
            StateBoundaryTerms::new()
                .with_default(KernelEdacSplitBoundaryFlux2D)
                .with_entities(facets, kernel.with_split_flux())
        } else {
            StateBoundaryTerms::new().with_entities(facets, kernel)
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
            .residual_operator_at(t, &self.kernel)
            .with_state_boundary(self.terms.clone());
        let residual = operator.residual(state);
        Mat::from_fn(self.m_inv.len(), 1, |row, _| {
            -self.m_inv[row] * residual[row]
        })
    }

    fn fjac<'b>(&'a self, t: f64, state: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
        let operator = self
            .problem
            .residual_operator_at(t, &self.kernel)
            .with_state_boundary(self.terms.clone());
        Box::new(MatrixFreeMinvCompleteJacobian::new(
            operator,
            state.to_owned(),
            &self.m_inv,
        ))
    }
}

pub fn epi3(state0: MatRef<'_, f64>) -> EpirkIntegrator<KrylovExpm> {
    let expmv = Box::new(PadeExpm::new(12));
    let krylov = KrylovExpm::new(expmv, 30, 100, 1e-12, Some(2));
    EpirkIntegrator::new(0.0, state0, "epi3".to_string(), krylov)
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
