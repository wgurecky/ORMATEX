//! EDAC/Smagorinsky vortex shedding around a cylinder on a Gmsh quad mesh.

use std::fs::File;
use std::io::{BufWriter, Write};

use faer::matrix_free::LinOp;
use faer::prelude::*;
use ormatex::ode_rk::RkIntegrator;
use ormatex::ode_sys::{IntegrateSys, OdeSys};
use ormatex_sem_nd::{
    gmsh_quad_data, DofReduction2D, KernelEdacNavierStokes2D, MatrixFreeMinvJacobian, QuadMesh,
    SEM2DProblem,
};

#[path = "../support/linear_system.rs"]
mod linear_system;
use linear_system::lumped_inverse_mass;

struct FluidSystem<'a> {
    problem: &'a SEM2DProblem<QuadMesh>,
    kernel: KernelEdacNavierStokes2D,
    m_inv: Vec<f64>,
}

impl<'a> FluidSystem<'a> {
    fn new(problem: &'a SEM2DProblem<QuadMesh>, kernel: KernelEdacNavierStokes2D) -> Self {
        let mass = problem.assemble_system_lumped_mass(3);
        Self {
            problem,
            kernel,
            m_inv: lumped_inverse_mass(mass.as_ref()),
        }
    }
}

impl<'a> OdeSys<'a> for FluidSystem<'a> {
    fn frhs(&self, t: f64, state: MatRef<f64>) -> Mat<f64> {
        let residual = self
            .problem
            .assemble_system_residual_at(t, &self.kernel, state);
        Mat::from_fn(self.m_inv.len(), 1, |row, _| {
            -self.m_inv[row] * residual[row]
        })
    }

    fn fjac<'b>(&'a self, t: f64, state: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
        Box::new(MatrixFreeMinvJacobian::new_at(
            t,
            self.problem,
            &self.kernel,
            state.to_owned(),
            None,
            &self.m_inv,
        ))
    }
}

fn boundary_facets(data: &ormatex_sem_nd::MeshMetadata, tag: usize) -> Vec<usize> {
    data.facet_regions
        .iter()
        .enumerate()
        .filter_map(|(index, region)| {
            (region.map(|region| region.tag) == Some(tag)).then_some(index)
        })
        .collect()
}

fn dirichlet(values: &[(usize, f64)]) -> DofReduction2D {
    DofReduction2D::Dirichlet {
        facets: values.to_vec(),
    }
}

fn nearest(positions: &[(f64, f64)], target: (f64, f64)) -> usize {
    positions
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = (a.0 - target.0).hypot(a.1 - target.1);
            let db = (b.0 - target.0).hypot(b.1 - target.1);
            da.partial_cmp(&db).unwrap()
        })
        .map(|(index, _)| index)
        .expect("field has no retained DOFs")
}

fn main() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/navier-stokes/cylinder.msh"
    );
    let data = gmsh_quad_data(path).expect("failed to load all-quad cylinder mesh");
    let mesh = data.mesh;
    let metadata = data.metadata;
    let inlet = boundary_facets(&metadata, 1);
    let outlet = boundary_facets(&metadata, 2);
    let cylinder = boundary_facets(&metadata, 5);
    assert!(!inlet.is_empty() && !outlet.is_empty() && !cylinder.is_empty());

    let u_in = 1.0;
    let inlet_u: Vec<_> = inlet.iter().copied().map(|facet| (facet, u_in)).collect();
    let inlet_v: Vec<_> = inlet.iter().copied().map(|facet| (facet, 0.0)).collect();
    let wall_u: Vec<_> = cylinder.iter().copied().map(|facet| (facet, 0.0)).collect();
    let symmetry: Vec<_> = metadata
        .facet_regions
        .iter()
        .enumerate()
        .filter_map(|(facet, region)| {
            (region.map(|region| region.tag) == Some(3)
                || region.map(|region| region.tag) == Some(4))
            .then_some(facet)
        })
        .collect();
    let wall_v: Vec<_> = cylinder
        .iter()
        .copied()
        .chain(symmetry)
        .map(|facet| (facet, 0.0))
        .collect();
    let outlet_p: Vec<_> = outlet.iter().copied().map(|facet| (facet, 0.0)).collect();
    let problem = SEM2DProblem::new_with_metadata(
        mesh,
        2,
        DofReduction2D::FieldSpecific {
            reductions: vec![
                dirichlet(
                    &inlet_u
                        .iter()
                        .chain(wall_u.iter())
                        .copied()
                        .collect::<Vec<_>>(),
                ),
                dirichlet(
                    &inlet_v
                        .iter()
                        .chain(wall_v.iter())
                        .copied()
                        .collect::<Vec<_>>(),
                ),
                dirichlet(&outlet_p),
            ],
        },
        metadata,
    );

    // Keep this example's user-facing setup compact; the kernel owns the model.
    let kernel = KernelEdacNavierStokes2D::new(1.0, 1.0 / 200.0, 4.0, 0.1);
    let system = FluidSystem::new(&problem, kernel);
    let state0 = Mat::<f64>::zeros(problem.system_size(3), 1);
    let dt = 0.005;
    let nsteps = 2400;
    let mut integrator = RkIntegrator::new(0.0, state0.as_ref(), 4);
    let probe_u = nearest(&problem.field_dof_positions(0), (2.0, 0.5));
    let probe_v = nearest(&problem.field_dof_positions(1), (2.0, 0.5));
    let probe_p = nearest(&problem.field_dof_positions(2), (2.0, 0.5));
    let mut probe = BufWriter::new(
        File::create("target/navier_stokes_cylinder_probe.csv")
            .expect("failed to create probe csv"),
    );
    writeln!(probe, "t,u,v,p").unwrap();
    for step in 0..nsteps {
        let result = integrator
            .step(&system, dt)
            .unwrap_or_else(|error| panic!("EDAC step {step} failed: {}", error.msg));
        integrator.accept_step(result);
        let state = integrator.state();
        writeln!(
            probe,
            "{:.8},{:.9e},{:.9e},{:.9e}",
            integrator.time(),
            state[(problem.field_offset(0, 3) + probe_u, 0)],
            state[(problem.field_offset(1, 3) + probe_v, 0)],
            state[(problem.field_offset(2, 3) + probe_p, 0)],
        )
        .unwrap();
    }
    let state = integrator.state();

    let mut output = BufWriter::new(
        File::create("target/navier_stokes_cylinder.csv").expect("failed to create output csv"),
    );
    writeln!(output, "field,x,y,value").unwrap();
    for field in 0..3 {
        for (local, (x, y)) in problem.field_dof_positions(field).into_iter().enumerate() {
            let value = state[(problem.field_offset(field, 3) + local, 0)];
            writeln!(output, "{field},{x:.8},{y:.8},{value:.9e}").unwrap();
        }
    }
}
