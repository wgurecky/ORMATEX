//! Tensor split-kernel EDAC drift-flux cylinder (`[u, v, p, alpha]`).
//!
//! Extension of `cylinder_tensor.rs` on the same Gmsh quad mesh: uniform
//! mixture inflow with inlet void `alpha = 0.2`, no-slip cylinder, slip
//! top/bottom walls, and a directional-do-nothing outflow on the right. The
//! mixture momentum uses the Smagorinsky-Lilly eddy viscosity; the void
//! spreads via Ishii-Zuber drift plus constant-coefficient turbulent
//! dispersion.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use faer::prelude::*;
use ormatex::ode_sys::IntegrateSys;
use ormatex_sem_nd::{
    DofReduction2D, DriftFlux2DConfig, FieldRegistry, SEM2DProblem, StateTensorBoundaryTerms,
    TensorDriftDirectionalDoNothing2D, TensorDriftFlux2D, TensorDriftGravity2D,
    TensorDriftMomentumConvectionSplit2D, TensorDriftPressureAdvectionSplit2D,
    TensorDriftPressureDiffusion2D, TensorDriftPressureDivergence2D, TensorDriftPressureGradient2D,
    TensorDriftSplitBoundaryFlux2D, TensorDriftTurbulentDispersion2D, TensorDriftViscousStress2D,
    TensorDriftVoidAdvectionSplit2D, TensorKernelEdacNoSlipWall2D, TensorKernelEdacSlipWall2D,
    TensorResidualKernel, TensorResidualKernelSum,
};

#[path = "../support/cylinder_setup.rs"]
mod cylinder_setup;
#[path = "../support/edac.rs"]
mod edac;
#[path = "../support/linear_system.rs"]
mod linear_system;

use cylinder_setup::nearest;
use edac::{epi3, maybe_write_vtk_mesh, TensorFluidSystem};

const INLET_VOID: f64 = 0.05;

fn drift_config() -> DriftFlux2DConfig {
    // ponytail: gentle segregation plus strong-enough dispersion to keep the
    // top-wall accumulation layer resolved under explicit epi3 at dt=0.05.
    // Stronger drift/buoyancy needs smaller dt or implicit stepping.
    DriftFlux2DConfig::new(1.0, 0.95, 1.0 / 200.0, 0.95 / 200.0, 4.0, 0.1).with_gravity_mag(1.0)
}

/// Append per-point vapor (`u_g`, `v_g`) and liquid (`u_l`, `v_l`) phase
/// velocities decoded from the mixture state via the slip closure.
fn append_phase_velocities(mesh: &mut ormatex_sem_nd::io::ExportMesh, config: &DriftFlux2DConfig) {
    let column = |name: &str| {
        mesh.field_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("export mesh has no field {name}"))
    };
    let (iu, iv, ia) = (column("u"), column("v"), column("alpha"));
    let n = mesh.point_count();
    let mut u_g = Vec::with_capacity(n);
    let mut v_g = Vec::with_capacity(n);
    let mut u_l = Vec::with_capacity(n);
    let mut v_l = Vec::with_capacity(n);
    for i in 0..n {
        let mixture = [mesh.point_fields[iu][i], mesh.point_fields[iv][i]];
        let alpha = mesh.point_fields[ia][i];
        let vapor = config.vapor_velocity(alpha, mixture);
        let liquid = config.liquid_velocity(alpha, mixture);
        u_g.push(vapor[0]);
        v_g.push(vapor[1]);
        u_l.push(liquid[0]);
        v_l.push(liquid[1]);
    }
    mesh.field_names
        .extend(["u_g", "v_g", "u_l", "v_l"].into_iter().map(str::to_owned));
    mesh.point_fields.extend([u_g, v_g, u_l, v_l]);
}

fn drift_kernel() -> impl TensorResidualKernel<2> {
    let config = drift_config();
    TensorResidualKernelSum::from_kernel(TensorDriftMomentumConvectionSplit2D::new(config))
        .with(TensorDriftPressureGradient2D::new(config))
        .with(TensorDriftViscousStress2D::new(config))
        .with(TensorDriftPressureDivergence2D::new(config))
        .with(TensorDriftPressureAdvectionSplit2D::new(config))
        .with(TensorDriftPressureDiffusion2D::new(config))
        .with(TensorDriftVoidAdvectionSplit2D::new(config))
        .with(TensorDriftFlux2D::new(config))
        .with(TensorDriftGravity2D::new(config))
        .with(TensorDriftTurbulentDispersion2D::new(config, 5e-2))
}

fn dirichlet(values: &[(usize, f64)]) -> DofReduction2D {
    DofReduction2D::Dirichlet {
        facets: values.to_vec(),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let steps = parse_usize_flag(&args, "--steps").unwrap_or(100);
    let threads = parse_usize_flag(&args, "--threads");
    let benchmark = args.iter().any(|arg| arg == "--benchmark");
    if let Some(threads) = threads {
        assert!(threads > 0, "--threads must be positive");
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .expect("failed to configure Rayon thread pool");
        faer::set_global_parallelism(faer::Par::rayon(threads));
    }

    let setup_start = Instant::now();
    // Same mesh and velocity/pressure BCs as the single-phase cylinder, plus
    // a fixed inlet void fraction.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/navier-stokes/cylinder.msh"
    );
    let data = ormatex_sem_nd::gmsh_quad_data(path).expect("failed to load all-quad cylinder mesh");
    let mesh = data.mesh;
    let metadata = data.metadata;
    let inlet = metadata.boundary_facets("inlet");
    let outlet = metadata.boundary_facets("outlet");
    let cylinder = metadata.boundary_facets("cylinder");
    assert!(!inlet.is_empty() && !outlet.is_empty() && !cylinder.is_empty());

    let inlet_u: Vec<_> = inlet.iter().copied().map(|facet| (facet, 1.0)).collect();
    let inlet_v: Vec<_> = inlet.iter().copied().map(|facet| (facet, 0.0)).collect();
    let inlet_a: Vec<_> = inlet
        .iter()
        .copied()
        .map(|facet| (facet, INLET_VOID))
        .collect();
    let wall_u: Vec<_> = cylinder.iter().copied().map(|facet| (facet, 0.0)).collect();
    let slip_wall: Vec<_> = metadata
        .boundary_facets("bottom")
        .into_iter()
        .chain(metadata.boundary_facets("top"))
        .collect();
    let wall_v: Vec<_> = cylinder
        .iter()
        .copied()
        .chain(slip_wall.iter().copied())
        .map(|facet| (facet, 0.0))
        .collect();

    let problem = SEM2DProblem::new_with_metadata(
        mesh,
        2,
        FieldRegistry::new(["u", "v", "p", "alpha"]),
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
                // Pressure datum via the directional outflow (no pin needed).
                DofReduction2D::None,
                dirichlet(&inlet_a),
            ],
        },
        metadata,
    );
    // Split-flux default for all split volumes (incl. void); base wall
    // closures cover [u, v, p] while alpha keeps the split flux; the outlet
    // uses the drift directional-do-nothing kernel with split flux.
    let terms = StateTensorBoundaryTerms::new()
        .with_default(TensorDriftSplitBoundaryFlux2D)
        .with_entities(cylinder.clone(), TensorKernelEdacNoSlipWall2D::new())
        .with_entities(slip_wall.clone(), TensorKernelEdacSlipWall2D::new())
        .with_entities(
            outlet.clone(),
            TensorDriftDirectionalDoNothing2D::new(1.0).with_split_flux(),
        );
    let system = TensorFluidSystem::new(&problem, drift_kernel()).with_state_boundary(terms);
    let setup_time = setup_start.elapsed();

    // Impulsive start from uniform flow at the inlet void: uniform velocity
    // has zero divergence, so no artificial-compressibility transient pumps
    // the void field at startup (starting from rest fires acoustic waves
    // that slosh the conserved void fraction).
    let mut state0 = Mat::<f64>::zeros(problem.system_size(), 1);
    let u_offset = problem.field_offset(0);
    for row in 0..problem.field_reduced_size(0) {
        state0[(u_offset + row, 0)] = 1.0;
    }
    let a_offset = problem.field_offset(3);
    for row in 0..problem.field_reduced_size(3) {
        state0[(a_offset + row, 0)] = 0.0;
    }
    let integration_start = Instant::now();
    let mut integrator = epi3(state0.as_ref());
    let dt = 0.05;
    for step in 0..steps {
        let result = integrator
            .step(&system, dt)
            .unwrap_or_else(|error| panic!("drift-flux step {step} failed: {}", error.msg));
        integrator.accept_step(result);
    }
    let state = integrator.state();
    let integration_time = integration_start.elapsed();

    println!(
        "drift-flux cylinder tensor: steps={steps} dofs={} setup={setup_time:?} integration={integration_time:?} threads={threads:?}",
        problem.system_size(),
    );
    if benchmark {
        return;
    }

    let alpha = problem.field_values("alpha", state.as_ref()).unwrap();
    for v in &alpha.values {
        assert!(v.is_finite(), "non-finite void fraction: {v}");
    }
    println!(
        "alpha: min={:.6} max={:.6}",
        alpha.values.iter().fold(f64::INFINITY, |m, &v| m.min(v)),
        alpha
            .values
            .iter()
            .fold(f64::NEG_INFINITY, |m, &v| m.max(v)),
    );

    std::fs::create_dir_all("target").expect("failed to create output directory");
    let probe_u = nearest(
        &problem
            .field_values("u", state0.as_ref())
            .unwrap()
            .positions,
        (2.0, 0.5),
    );
    let probe_alpha = nearest(
        &problem
            .field_values("alpha", state0.as_ref())
            .unwrap()
            .positions,
        (2.0, 0.5),
    );
    let config = drift_config();
    let probe_um = [
        state[(problem.field_offset(0) + probe_u, 0)],
        state[(problem.field_offset(1) + probe_u, 0)],
    ];
    let probe_alpha = state[(problem.field_offset(3) + probe_alpha, 0)];
    let probe_vapor = config.vapor_velocity(probe_alpha, probe_um);
    let probe_liquid = config.liquid_velocity(probe_alpha, probe_um);
    let mut probe = BufWriter::new(
        File::create("target/drift_flux_cylinder_tensor_probe.csv")
            .expect("failed to create drift-flux probe csv"),
    );
    writeln!(probe, "u,v,p,alpha,u_g,v_g,u_l,v_l").unwrap();
    writeln!(
        probe,
        "{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}",
        probe_um[0],
        probe_um[1],
        state[(problem.field_offset(2) + probe_u, 0)],
        probe_alpha,
        probe_vapor[0],
        probe_vapor[1],
        probe_liquid[0],
        probe_liquid[1],
    )
    .unwrap();
    let mut mesh = ormatex_sem_nd::io::export_2d(&problem, state.as_ref());
    append_phase_velocities(&mut mesh, &config);
    ormatex_sem_nd::io::write_csv("target/drift_flux_cylinder_tensor.csv", &mesh);
    maybe_write_vtk_mesh(&mesh, "target/drift_flux_cylinder_tensor.vtu");
    println!("drift-flux cylinder result: target/drift_flux_cylinder_tensor.csv");
}

fn parse_usize_flag(args: &[String], name: &str) -> Option<usize> {
    args.windows(2)
        .find(|window| window[0] == name)
        .map(|window| {
            window[1]
                .parse()
                .expect("invalid numeric command-line flag")
        })
}
