//! Tensor split-kernel EDAC drift-flux bubble plume (`[u, v, p, alpha]`).
//!
//! Rising-bubble tank on `bubble_plume.msh` (0.1 m square with a centered
//! 0.01 m square injector hole): true air-water mixture at rest, gas injected
//! downwards (`v = -1.0`, `alpha = 1.0`) through the bottom edge of the inner
//! hole while buoyancy turns the plume upwards. Left/right/bottom plus the
//! three remaining inner edges are no-slip walls; the top is a free surface
//! (`TensorDriftFreeSurface2D`) where void vents out of the domain and the
//! pressure datum is set weakly, so no pressure Dirichlet pin is needed. The
//! mixture momentum uses the Smagorinsky-Lilly eddy viscosity; the void
//! spreads via Ishii-Zuber drift plus constant-coefficient turbulent
//! dispersion.
//!
//! Run with `RAYON_NUM_THREADS=4 cargo run --release --example
//! navier-stokes-drift-flux-bubble-plume-tensor` (`--steps`, total across
//! the 5 ramp stages, default 400; `--dt`, default 5e-4 with 1e-4 startup
//! steps while each ramped jet turns around).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use faer::prelude::*;
use ormatex::ode_implicit::DirkIntegrator;
use ormatex::ode_sys::IntegrateSys;
use ormatex::tableau_implicit::ImplicitBT;
use ormatex_sem_nd::{
    dirichlet_values_with_precedence, DofReduction2D, DriftFlux2DConfig, FieldRegistry, QuadMesh,
    SEM2DProblem, StateTensorBoundaryTerms, TensorDriftFlux2D, TensorDriftFreeSurface2D,
    TensorDriftGravity2D, TensorDriftMomentumConvectionSplit2D,
    TensorDriftPressureAdvectionSplit2D, TensorDriftPressureDiffusion2D,
    TensorDriftPressureDivergence2D, TensorDriftPressureGradient2D, TensorDriftSplitBoundaryFlux2D,
    TensorDriftTurbulentDispersion2D, TensorDriftViscousStress2D, TensorDriftVoidAdvectionSplit2D,
    TensorKernelEdacNoSlipWall2D, TensorResidualKernel, TensorResidualKernelSum,
};

#[path = "../support/edac.rs"]
mod edac;
#[path = "../support/linear_system.rs"]
mod linear_system;

use edac::{maybe_write_vtk_mesh, TensorFluidSystem};

// True air-water properties. NOTE: `acoustic_c0` below is the EDAC
// artificial sound speed, unrelated to the Ishii distribution parameter
// `C0` stored in the same config (see `distribution_parameter()`).
const RHO_LIQUID: f64 = 1000.0;
const RHO_GAS: f64 = 1.2;
const MU_LIQUID: f64 = 1.0e-3;
const MU_GAS: f64 = 1.8e-5;
const ACOUSTIC_C0: f64 = 10.0;
const SMAGORINSKY_CS: f64 = 0.1;
// ponytail: generous turbulent mixing; the only mechanism diluting the
// pure-gas jet core (no interphase drag in this drift-flux phase). Mixture
// buoyancy (~g at alpha = 0.5) diverges toward ~8000 m/s^2 as alpha -> 1,
// so the plume is only computable while dispersion holds the peak alpha
// well below one.
const DISPERSION: f64 = 5.0e-3;
const INLET_VELOCITY: f64 = -1.0;
const INLET_VOID: f64 = 1.0;

fn drift_config() -> DriftFlux2DConfig {
    // ponytail: default gravity [0, -9.81] and sigma 0.0728 already match
    // air-water; Harmathy terminal speed is then ~0.23 m/s.
    DriftFlux2DConfig::new(
        RHO_LIQUID,
        RHO_GAS,
        MU_LIQUID,
        MU_GAS,
        ACOUSTIC_C0,
        SMAGORINSKY_CS,
    )
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
        .with(TensorDriftTurbulentDispersion2D::new(config, DISPERSION))
}

fn dirichlet(values: &[(usize, f64)]) -> DofReduction2D {
    DofReduction2D::Dirichlet {
        facets: values.to_vec(),
    }
}

/// Inlet velocity ramp stages: the impulsive start of a pure-gas downward
/// jet in water stalls Newton's method, so the inlet speed is raised in
/// `NSTAGES` equal fractions of `INLET_VELOCITY`, one case rebuild per
/// stage. Facet sets never change, so every stage shares one reduced-DOF
/// layout and the state vector carries over directly.
const NSTAGES: usize = 5;

/// Build one ramp stage: same mesh, facet sets, and boundary kernels every
/// time, differing only in the prescribed inlet velocity.
fn build_case(inlet_velocity: f64) -> (SEM2DProblem<QuadMesh>, StateTensorBoundaryTerms<2>) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/navier-stokes/bubble_plume.msh"
    );
    let data = ormatex_sem_nd::gmsh_quad_data(path).expect("failed to load bubble-plume mesh");
    let mesh = data.mesh;
    let metadata = data.metadata;
    let inlet = metadata.boundary_facets("injector_inlet");
    let freesurface = metadata.boundary_facets("freesurface");
    let walls: Vec<_> = ["left", "right", "bottom"]
        .into_iter()
        .flat_map(|name| metadata.boundary_facets(name))
        .chain(metadata.boundary_facets("injector_wall_left"))
        .chain(metadata.boundary_facets("injector_wall_right"))
        .chain(metadata.boundary_facets("injector_wall_top"))
        .collect();
    assert!(!inlet.is_empty(), "missing injector_inlet facets");
    assert!(!freesurface.is_empty(), "missing freesurface facets");
    assert!(!walls.is_empty(), "missing wall facets");

    let inlet_u: Vec<_> = inlet.iter().copied().map(|facet| (facet, 0.0)).collect();
    let inlet_v: Vec<_> = inlet
        .iter()
        .copied()
        .map(|facet| (facet, inlet_velocity))
        .collect();
    let inlet_a: Vec<_> = inlet
        .iter()
        .copied()
        .map(|facet| (facet, INLET_VOID))
        .collect();
    let wall_u: Vec<_> = walls.iter().copied().map(|facet| (facet, 0.0)).collect();
    let wall_v: Vec<_> = walls.iter().copied().map(|facet| (facet, 0.0)).collect();

    // The inlet shares its two endpoint DOFs with the injector side walls;
    // walls take precedence there (as in the backward-step case), leaving a
    // jet profile that tapers to zero at the injector lip.
    let u_values = dirichlet_values_with_precedence(&mesh, 2, &wall_u, &inlet_u);
    let v_values = dirichlet_values_with_precedence(&mesh, 2, &wall_v, &inlet_v);

    let problem = SEM2DProblem::new_with_metadata(
        mesh,
        2,
        FieldRegistry::new(["u", "v", "p", "alpha"]),
        DofReduction2D::FieldSpecific {
            reductions: vec![
                DofReduction2D::DirichletValues { values: u_values },
                DofReduction2D::DirichletValues { values: v_values },
                // Pressure datum via the free-surface traction (no pin needed);
                // the free surface also leaves u/v/alpha free so void can vent.
                DofReduction2D::None,
                dirichlet(&inlet_a),
            ],
        },
        metadata,
    );
    // Split-flux default for all split volumes (incl. void and the inlet);
    // base wall closures cover [u, v, p] on no-slip walls while alpha keeps
    // the split flux; the free surface uses the venting drift kernel.
    let terms = StateTensorBoundaryTerms::new()
        .with_default(TensorDriftSplitBoundaryFlux2D)
        .with_entities(walls.clone(), TensorKernelEdacNoSlipWall2D::new())
        .with_entities(
            freesurface.clone(),
            TensorDriftFreeSurface2D::new(drift_config()),
        );
    (problem, terms)
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
    let args: Vec<String> = std::env::args().collect();
    let steps = parse_usize_flag(&args, "--steps").unwrap_or(400);
    let dt = parse_f64_flag(&args, "--dt").unwrap_or(5.0e-4);
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

    // Still tank of pure water; the inlet jet ramps 0 -> full speed over
    // NSTAGES (impulsive pure-gas injection stalls Newton). Stiff true
    // air-water system: L-stable SDIRK32 with 1e-8 tolerances (O(1e3)
    // pressure scales; max-switch Jacobians limit Newton to linear anyway).
    let mut state = Mat::<f64>::zeros(build_case(0.0).0.system_size(), 1);
    let mut t = 0.0;
    let mut total_steps = 0;
    let integration_start = Instant::now();
    for stage in 0..NSTAGES {
        let inlet_velocity = INLET_VELOCITY * (stage + 1) as f64 / NSTAGES as f64;
        let stage_steps = steps / NSTAGES
            + if stage + 1 == NSTAGES {
                steps % NSTAGES
            } else {
                0
            };
        let (problem, terms) = build_case(inlet_velocity);
        assert_eq!(
            state.nrows(),
            problem.system_size() as usize,
            "ramp stages must share one DOF layout"
        );
        let system = TensorFluidSystem::new(&problem, drift_kernel()).with_state_boundary(terms);
        let mut integrator =
            DirkIntegrator::new(t, state.as_ref(), ImplicitBT::sdirk32(), 1e-8, 1e-8);
        for _ in 0..stage_steps {
            // ponytail: small steps while each ramped jet turns around
            // (t < 0.01), full dt once the plume is established.
            let step_dt = if integrator.time() < 0.01 {
                dt.min(1.0e-4)
            } else {
                dt
            };
            let result = integrator.step(&system, step_dt).unwrap_or_else(|error| {
                panic!(
                    "bubble-plume step {total_steps} (stage {stage}) failed: {}",
                    error.msg
                )
            });
            integrator.accept_step(result);
            total_steps += 1;
        }
        state = integrator.state();
        t = integrator.time();
        println!(
            "stage {}/{NSTAGES}: inlet v={inlet_velocity:.2} t={t:.4}",
            stage + 1
        );
    }
    let integration_time = integration_start.elapsed();
    // Rebuild the full-velocity case for export (same DOF layout as every
    // stage, so the carried state applies directly).
    let (problem, _) = build_case(INLET_VELOCITY);

    println!(
        "drift-flux bubble plume tensor: steps={total_steps} dt={dt:e} dofs={} integration={integration_time:?} threads={threads:?}",
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
        &problem.field_values("u", state.as_ref()).unwrap().positions,
        (0.05, 0.08),
    );
    let config = drift_config();
    let probe_um = [
        state[(problem.field_offset(0) + probe_u, 0)],
        state[(problem.field_offset(1) + probe_u, 0)],
    ];
    let probe_alpha = state[(problem.field_offset(3) + probe_u, 0)];
    let probe_vapor = config.vapor_velocity(probe_alpha, probe_um);
    let probe_liquid = config.liquid_velocity(probe_alpha, probe_um);
    let mut probe = BufWriter::new(
        File::create("target/drift_flux_bubble_plume_tensor_probe.csv")
            .expect("failed to create bubble-plume probe csv"),
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
    ormatex_sem_nd::io::write_csv("target/drift_flux_bubble_plume_tensor.csv", &mesh);
    maybe_write_vtk_mesh(&mesh, "target/drift_flux_bubble_plume_tensor.vtu");
    println!("bubble-plume result: target/drift_flux_bubble_plume_tensor.csv");
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

fn parse_f64_flag(args: &[String], name: &str) -> Option<f64> {
    args.windows(2)
        .find(|window| window[0] == name)
        .map(|window| {
            window[1]
                .parse()
                .expect("invalid numeric command-line flag")
        })
}
