use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use faer::matrix_free::LinOp;
use faer::prelude::*;
use ormatex::matexp_krylov::KrylovExpm;
use ormatex::matexp_pade::PadeExpm;
use ormatex::ode_epirk::EpirkIntegrator;
use ormatex::ode_sys::{IntegrateSys, OdeSys};
use ormatex_sem_nd::{
    CellState, FieldValues, KernelEdacDongOutflow2D, KernelEdacSplitBoundaryFlux2D, LocalCtx,
    MatrixFreeMinvJacobian, OwnedMinvJacobian, QuadMesh, ResidualKernel, SEM2DProblem,
    StateBoundaryTerms,
};

use super::linear_system::lumped_inverse_mass;

/// Disables tensor hooks while retaining the same pointwise kernel behavior.
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
            Self::Assembled
        } else {
            Self::MatrixFree
        }
    }
}

pub fn write_spatial_csv(path: impl AsRef<Path>, fields: [(&str, FieldValues<(f64, f64)>); 3]) {
    let mut rows = Vec::new();
    let mut row_indices = HashMap::new();
    for (field, field_values) in fields {
        let column = match field {
            "u" => 2,
            "v" => 3,
            "p" => 4,
            _ => panic!("unsupported Navier-Stokes field: {field}"),
        };
        assert_eq!(
            field_values.positions.len(),
            field_values.values.len(),
            "field {field} positions and values have different lengths"
        );
        for (&(x, y), &value) in field_values.positions.iter().zip(&field_values.values) {
            assert!(
                x.is_finite() && y.is_finite() && value.is_finite(),
                "non-finite {field} spatial output"
            );
            let key = (x.to_bits(), y.to_bits());
            let index = if let Some(&index) = row_indices.get(&key) {
                index
            } else {
                let index = rows.len();
                row_indices.insert(key, index);
                rows.push([x, y, f64::NAN, f64::NAN, f64::NAN]);
                index
            };
            assert!(
                rows[index][column].is_nan(),
                "duplicate {field} value at ({x}, {y})"
            );
            rows[index][column] = value;
        }
    }

    let mut output = BufWriter::new(
        File::create(path).expect("failed to create Navier-Stokes spatial output csv"),
    );
    writeln!(output, "x,y,u,v,p").unwrap();
    for [x, y, u, v, p] in rows {
        writeln!(output, "{x:.9},{y:.9},{u:.9e},{v:.9e},{p:.9e}").unwrap();
    }
}

pub struct FluidSystem<'a, K> {
    pub problem: &'a SEM2DProblem<QuadMesh>,
    pub kernel: K,
    m_inv: Vec<f64>,
    terms: StateBoundaryTerms,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spatial_csv_joins_fields_by_position() {
        let path =
            std::env::temp_dir().join(format!("ormatex-navier-stokes-{}.csv", std::process::id()));
        write_spatial_csv(
            &path,
            [
                (
                    "u",
                    FieldValues {
                        positions: vec![(0.0, 0.0), (1.0, 0.0)],
                        values: vec![1.0, 2.0],
                    },
                ),
                (
                    "v",
                    FieldValues {
                        positions: vec![(1.0, 0.0), (0.0, 0.0)],
                        values: vec![3.0, 4.0],
                    },
                ),
                (
                    "p",
                    FieldValues {
                        positions: vec![(0.0, 0.0)],
                        values: vec![5.0],
                    },
                ),
            ],
        );
        let output = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            output,
            "x,y,u,v,p\n0.000000000,0.000000000,1.000000000e0,4.000000000e0,5.000000000e0\n1.000000000,0.000000000,2.000000000e0,3.000000000e0,NaN\n"
        );
        std::fs::remove_file(path).unwrap();
    }
}
