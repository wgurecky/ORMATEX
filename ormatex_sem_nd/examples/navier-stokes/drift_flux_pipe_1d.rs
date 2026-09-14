//! 1D EDAC drift-flux pipe flow (`[u, p, alpha]`).
//!
//! Vertical up-flow pipe: uniform mixture velocity enters with a fixed inlet
//! void fraction (`alpha = 0.2`); the outlet pins the pressure datum with a
//! natural void outflow. The pipe angle `theta(x) = pi/2` is a
//! space-dependent `MaterialProperty` (axial gravity `-g sin(theta)` and axial
//! Ishii-Zuber drift `V_gj sin(theta)`), marched to steady state with
//! implicit Euler. The void profile must stay bounded and physical.

use std::f64::consts::FRAC_PI_2;
use std::fs::File;
use std::io::Write;

use faer::matrix_free::LinOp;
use faer::prelude::*;
use ndmesh::shapes::unit_interval;
use ormatex::ode_implicit::DirkIntegrator;
use ormatex::ode_sys::{IntegrateSys, OdeSys};
use ormatex::tableau_implicit::ImplicitBT;
use ormatex_sem_nd::{
    BilinearOps, DofReduction1D, DriftFlux1DConfig, DriftOutflow1D, FieldRegistry, MaterialContext,
    MatrixFreeMinvJacobian, ParallelOwnedMinvJacobian, SEM1DProblem, StateBoundaryTerms,
    TensorDriftFlux1D, TensorDriftGravity1D, TensorDriftMomentumConvectionSplit1D,
    TensorDriftPressureAdvectionSplit1D, TensorDriftPressureDiffusion1D,
    TensorDriftPressureDivergence1D, TensorDriftPressureGradient1D,
    TensorDriftTurbulentDispersion1D, TensorDriftViscousStress1D, TensorDriftVoidAdvectionSplit1D,
    TensorResidualKernelSet, TensorResidualKernelSum,
};

#[path = "../support/linear_system.rs"]
mod linear_system;
use linear_system::lumped_inverse_mass;

type IntervalMesh = ndmesh::SingleElementMesh<
    f64,
    ndelement::ciarlet::CiarletElement<f64, ndelement::map::IdentityMap, f64>,
>;

fn drift_kernel(config: DriftFlux1DConfig) -> TensorResidualKernelSet<'static, 1> {
    // Space-dependent pipe angle (vertical up-flow here); replace with any
    // `theta(x)` profile for inclined pipes or bends.
    let vertical = |_: &MaterialContext<'_>| FRAC_PI_2;
    TensorResidualKernelSet::from_kernel(
        TensorResidualKernelSum::from_kernel(TensorDriftMomentumConvectionSplit1D::new(config))
            .with(TensorDriftPressureGradient1D::new(config))
            .with(TensorDriftViscousStress1D::new(config))
            .with(TensorDriftPressureDivergence1D::new(config))
            .with(TensorDriftPressureAdvectionSplit1D::new(config))
            .with(TensorDriftPressureDiffusion1D::new(config))
            .with(TensorDriftVoidAdvectionSplit1D::new(config))
            .with(TensorDriftFlux1D::new(config, vertical))
            .with(TensorDriftGravity1D::new(config, vertical))
            .with(TensorDriftTurbulentDispersion1D::new(config, 0.01)),
    )
}

struct DriftPipeSystem<'a> {
    problem: &'a SEM1DProblem<IntervalMesh>,
    kernel: TensorResidualKernelSet<'a, 1>,
    m_inv: Vec<f64>,
    assembled: bool,
    terms: StateBoundaryTerms,
}

impl<'a> OdeSys<'a> for DriftPipeSystem<'a> {
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
        if self.assembled {
            Box::new(ParallelOwnedMinvJacobian::new(
                operator.assemble_jacobian(state),
                &self.m_inv,
            ))
        } else {
            Box::new(MatrixFreeMinvJacobian::new(
                operator,
                state.to_owned(),
                &self.m_inv,
            ))
        }
    }
}

fn main() {
    let assembled = std::env::args().any(|arg| arg == "--assembled-jacobian");
    let nx = 32;
    let p = 2;
    let inlet_velocity = 1.0;
    let inlet_void = 0.2;
    let config = DriftFlux1DConfig::new(1.0, 0.1, 0.01, 0.001, 10.0);
    let dt = 0.05;
    let nsteps = 800;

    // Inlet: fixed velocity and void fraction; outlet: pressure datum with a
    // natural (zero-flux) void outflow.
    let problem = SEM1DProblem::new(
        unit_interval(nx, 1),
        p,
        FieldRegistry::new(["u", "p", "alpha"]),
        DofReduction1D::FieldSpecific {
            reductions: vec![
                DofReduction1D::Dirichlet {
                    facets: vec![(0, inlet_velocity)],
                },
                DofReduction1D::Dirichlet {
                    facets: vec![(nx, 0.0)],
                },
                DofReduction1D::Dirichlet {
                    facets: vec![(0, inlet_void)],
                },
            ],
        },
    );
    // Outflow pairing fluxes (split momentum/pressure/void + axial drift)
    // on the right endpoint; the inlet is fully Dirichlet-prescribed.
    let terms =
        StateBoundaryTerms::new().with_entities([nx], DriftOutflow1D::new(config, FRAC_PI_2.sin()));
    let system = DriftPipeSystem {
        problem: &problem,
        kernel: drift_kernel(config),
        m_inv: lumped_inverse_mass(problem.assemble_lumped_mass().as_ref()),
        assembled,
        terms,
    };

    // Impulsive start from uniform flow at the inlet void and march to
    // steady state (starting velocity from rest shocks Newton's solve).
    let mut y0 = Mat::<f64>::zeros(problem.system_size(), 1);
    let u_field = problem.field_id("u").unwrap();
    let u_offset = problem.field_offset(u_field);
    for row in 0..problem.field_reduced_size(u_field) {
        y0[(u_offset + row, 0)] = inlet_velocity;
    }
    let a_field = problem.field_id("alpha").unwrap();
    let a_offset = problem.field_offset(a_field);
    for row in 0..problem.field_reduced_size(a_field) {
        y0[(a_offset + row, 0)] = inlet_void;
    }
    let mut solver =
        DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::implicit_euler(), 1e-10, 1e-10);
    for _ in 0..nsteps {
        let step = solver.step(&system, dt).unwrap();
        solver.accept_step(step);
    }
    let state = solver.state();

    let u = problem.field_values("u", state.as_ref()).unwrap();
    let alpha = problem.field_values("alpha", state.as_ref()).unwrap();
    let p_field = problem.field_values("p", state.as_ref()).unwrap();

    std::fs::create_dir_all("target").expect("failed to create output directory");
    let mut output =
        File::create("target/drift_flux_pipe_1d.csv").expect("failed to create output csv");
    writeln!(output, "x,u,p,alpha,u_g,u_l").unwrap();
    for (i, x) in u.positions.iter().enumerate() {
        let vapor = config.vapor_velocity_1d(alpha.values[i], u.values[i], FRAC_PI_2);
        let liquid = config.liquid_velocity_1d(alpha.values[i], u.values[i], FRAC_PI_2);
        writeln!(
            output,
            "{x:.6},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}",
            u.values[i], p_field.values[i], alpha.values[i], vapor, liquid,
        )
        .unwrap();
    }
    // Same phase velocities as point data in the `.vtu` snapshot (written
    // only with `--vtk`, mirroring the 2D drift-flux example).
    let mut mesh = ormatex_sem_nd::io::export_1d(&problem, state.as_ref());
    {
        let column = |name: &str| {
            mesh.field_names
                .iter()
                .position(|n| n == name)
                .unwrap_or_else(|| panic!("export mesh has no field {name}"))
        };
        let (iu, ia) = (column("u"), column("alpha"));
        let n = mesh.point_count();
        let mut u_g = Vec::with_capacity(n);
        let mut u_l = Vec::with_capacity(n);
        for i in 0..n {
            u_g.push(config.vapor_velocity_1d(
                mesh.point_fields[ia][i],
                mesh.point_fields[iu][i],
                FRAC_PI_2,
            ));
            u_l.push(config.liquid_velocity_1d(
                mesh.point_fields[ia][i],
                mesh.point_fields[iu][i],
                FRAC_PI_2,
            ));
        }
        mesh.field_names
            .extend(["u_g", "u_l"].into_iter().map(str::to_owned));
        mesh.point_fields.extend([u_g, u_l]);
    }
    let args: Vec<String> = std::env::args().collect();
    if let Some(path) = ormatex_sem_nd::io::vtk_output_path(&args, "target/drift_flux_pipe_1d.vtu")
    {
        #[cfg(feature = "vtk")]
        {
            ormatex_sem_nd::io::write_vtu(&path, &mesh);
            println!("vtk solution: {path}");
        }
        #[cfg(not(feature = "vtk"))]
        {
            let _ = mesh;
            panic!("--vtk ({path}) requires building with `--features vtk`");
        }
    }
    for v in u.values.iter().chain(alpha.values.iter()) {
        assert!(v.is_finite(), "non-finite drift-pipe state: {v}");
    }
    for v in &alpha.values {
        assert!(
            *v > -5e-2 && *v < 1.0 + 5e-2,
            "void fraction must stay bounded and physical: {v}"
        );
    }
    println!(
        "drift-flux pipe: steps={nsteps} u_mean={:.6} alpha_min={:.6} alpha_max={:.6}",
        u.values.iter().sum::<f64>() / u.values.len() as f64,
        alpha.values.iter().fold(f64::INFINITY, |m, &v| m.min(v)),
        alpha
            .values
            .iter()
            .fold(f64::NEG_INFINITY, |m, &v| m.max(v)),
    );
}
