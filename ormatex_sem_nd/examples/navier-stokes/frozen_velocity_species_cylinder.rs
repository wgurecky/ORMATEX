//! Sequential EDAC flow + 3-species reaction-advection-diffusion on the cylinder mesh.
//!
//! Two problems share `cylinder.msh` with `P=2`: fluid `[u, v, p]` (tensor
//! split EDAC) and species `[c0, c1, c2]` (frozen-velocity tensor adv-diff +
//! linear reaction, inlet Dirichlet `c0=1, c1=c2=0`). An outer loop
//! alternates fluid substeps, frozen-velocity refresh, and species substeps.
//! Tensor kernels + matrix-free Jacobian by default (see `support/edac.rs`
//! `--assembled-jacobian` / `--matrix-free` flags); EPI3 exponential
//! stepping by default (`--sdirk32` opts into SDIRK32 implicit).

use std::time::Instant;

use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};
use ormatex_sem_nd::{
    ConstantCoefficient, DofReduction2D, EdacNavierStokes2DConfig, FieldRegistry, QuadMesh,
    StateTensorBoundaryTerms, TensorKernelAdvDiff2D, TensorKernelAdvectionOutflow2D,
    TensorKernelEdacDirectionalDoNothing2D, TensorKernelEdacMomentumConvectionSplit2D,
    TensorKernelEdacPressureAdvectionSplit2D, TensorKernelEdacPressureDiffusion2D,
    TensorKernelEdacPressureDivergence2D, TensorKernelEdacPressureGradient2D,
    TensorKernelEdacViscousStress2D, TensorKernelLinearReaction, TensorResidualKernel,
    TensorResidualKernelSet, TensorResidualKernelSum, SEM2DProblem,
};

#[path = "../support/edac.rs"]
mod edac;
#[path = "../support/linear_system.rs"]
mod linear_system;

use edac::{TimeStepper, TensorFluidSystem, maybe_write_vtk_mesh};

const P: usize = 3;

// Same rates as `ex_nd_1d_linear_reaction`.
const RATES: [(usize, usize, f64); 5] = [
    (0, 0, -0.1),
    (1, 0, 0.1),
    (1, 1, -10.0),
    (2, 1, 10.0),
    (2, 2, -0.01),
];

fn reaction_matrix() -> SparseColMat<usize, f64> {
    let triplets: Vec<_> = RATES
        .iter()
        .map(|&(row, column, value)| Triplet::new(row, column, value))
        .collect();
    SparseColMat::try_new_from_triplets(3, 3, &triplets).unwrap()
}

fn fluid_kernel() -> impl TensorResidualKernel<2> {
    let config = EdacNavierStokes2DConfig::new(1.0, 1.0 / 200.0, 4.0, 0.1);
    TensorResidualKernelSum::from_kernel(TensorKernelEdacMomentumConvectionSplit2D::new(config))
        .with(TensorKernelEdacPressureGradient2D::new(config))
        .with(TensorKernelEdacViscousStress2D::new(config))
        .with(TensorKernelEdacPressureDivergence2D::new(config))
        .with(TensorKernelEdacPressureAdvectionSplit2D::new(config))
        .with(TensorKernelEdacPressureDiffusion2D::new(config))
}

fn species_kernel(
    frozen_x: ormatex_sem_nd::FrozenQuadratureField,
    frozen_y: ormatex_sem_nd::FrozenQuadratureField,
    diffusivity: f64,
) -> TensorResidualKernelSet<'static, 2> {
    // ponytail: one frozen snapshot pair shared by 3 species via Arc clones.
    let t0 = TensorKernelAdvDiff2D::with_coefficients(
        ConstantCoefficient(diffusivity),
        [frozen_x.clone(), frozen_y.clone()],
    )
    .with_field_name("c0");
    let t1 = TensorKernelAdvDiff2D::with_coefficients(
        ConstantCoefficient(diffusivity),
        [frozen_x.clone(), frozen_y.clone()],
    )
    .with_field_name("c1");
    let t2 = TensorKernelAdvDiff2D::with_coefficients(
        ConstantCoefficient(diffusivity),
        [frozen_x, frozen_y],
    )
    .with_field_name("c2");
    let reaction =
        TensorKernelLinearReaction::with_field_names(reaction_matrix(), ["c0", "c1", "c2"]);
    TensorResidualKernelSet::from_kernel(t0)
        .with(t1)
        .with(t2)
        .with(reaction)
}

fn dirichlet(facets: &[(usize, f64)]) -> DofReduction2D {
    DofReduction2D::Dirichlet {
        facets: facets.to_vec(),
    }
}

fn parse_usize_flag(args: &[String], name: &str) -> Option<usize> {
    args.windows(2)
        .find(|w| w[0] == name)
        .map(|w| w[1].parse().expect("invalid numeric flag"))
}

fn parse_f64_flag(args: &[String], name: &str) -> Option<f64> {
    args.windows(2)
        .find(|w| w[0] == name)
        .map(|w| w[1].parse().expect("invalid numeric flag"))
}

/// Join fluid `[u, v, p]` and species `[c0, c1, c2]` exports (one shared mesh)
/// into a single standardized `x,y,u,v,p,c0,c1,c2` table.
fn joined_export(
    fluid_problem: &SEM2DProblem<QuadMesh>,
    fluid_state: MatRef<'_, f64>,
    species_problem: &SEM2DProblem<QuadMesh>,
    species_state: MatRef<'_, f64>,
) -> ormatex_sem_nd::io::ExportMesh {
    let mut mesh = ormatex_sem_nd::io::export_2d(fluid_problem, fluid_state);
    mesh.append_fields(&ormatex_sem_nd::io::export_2d(
        species_problem,
        species_state,
    ));
    mesh
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let directional = true;
    let benchmark = args.iter().any(|a| a == "--benchmark");
    let outer = parse_usize_flag(&args, "--outer").unwrap_or(20);
    let fluid_sub = parse_usize_flag(&args, "--fluid-steps").unwrap_or(5);
    let species_sub = parse_usize_flag(&args, "--species-steps").unwrap_or(5);
    let dt = parse_f64_flag(&args, "--dt").unwrap_or(0.02);
    let diffusivity = parse_f64_flag(&args, "--diffusivity").unwrap_or(0.01);
    if let Some(threads) = parse_usize_flag(&args, "--threads") {
        assert!(threads > 0, "--threads must be positive");
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .expect("failed to configure Rayon thread pool");
        faer::set_global_parallelism(faer::Par::rayon(threads));
    }

    let setup_start = Instant::now();
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/navier-stokes/cylinder.msh"
    );
    // ponytail: load twice for two problems; same file+P gives identical
    // cell ordering, asserted below via cell_count/npts compatibility.
    let fluid_data =
        ormatex_sem_nd::gmsh_quad_data(path).expect("failed to load cylinder mesh");
    let species_data =
        ormatex_sem_nd::gmsh_quad_data(path).expect("failed to load cylinder mesh");
    let data = fluid_data;
    let inlet = data.metadata.boundary_facets("inlet");
    let outlet = data.metadata.boundary_facets("outlet");
    let cylinder = data.metadata.boundary_facets("cylinder");
    assert!(!inlet.is_empty() && !outlet.is_empty() && !cylinder.is_empty());
    let slip_wall: Vec<_> = data
        .metadata
        .boundary_facets("bottom")
        .into_iter()
        .chain(data.metadata.boundary_facets("top"))
        .collect();

    let inlet_u: Vec<_> = inlet.iter().copied().map(|f| (f, 1.0)).collect();
    let inlet_v: Vec<_> = inlet.iter().copied().map(|f| (f, 0.0)).collect();
    let wall_u: Vec<_> = cylinder.iter().copied().map(|f| (f, 0.0)).collect();
    let wall_v: Vec<_> = cylinder
        .iter()
        .copied()
        .chain(slip_wall.iter().copied())
        .map(|f| (f, 0.0))
        .collect();
    let outlet_p: Vec<_> = outlet.iter().copied().map(|f| (f, 0.0)).collect();

    let fluid_problem = SEM2DProblem::new_with_metadata(
        data.mesh,
        P,
        FieldRegistry::new(["u", "v", "p"]),
        DofReduction2D::FieldSpecific {
            reductions: vec![
                dirichlet(
                    &inlet_u.iter().chain(wall_u.iter()).copied().collect::<Vec<_>>(),
                ),
                dirichlet(
                    &inlet_v.iter().chain(wall_v.iter()).copied().collect::<Vec<_>>(),
                ),
                if directional {
                    DofReduction2D::None
                } else {
                    dirichlet(&outlet_p)
                },
            ],
        },
        data.metadata.clone(),
    );

    // Species inlet: c0=1, c1=c2=0 on the left (inlet) face.
    let inlet_c0: Vec<_> = inlet.iter().copied().map(|f| (f, 1.0)).collect();
    let inlet_c1: Vec<_> = inlet.iter().copied().map(|f| (f, 0.0)).collect();
    let inlet_c2: Vec<_> = inlet.iter().copied().map(|f| (f, 0.0)).collect();
    let species_problem: SEM2DProblem<QuadMesh> = SEM2DProblem::new_with_metadata(
        species_data.mesh,
        P,
        FieldRegistry::new(["c0", "c1", "c2"]),
        DofReduction2D::FieldSpecific {
            reductions: vec![
                dirichlet(&inlet_c0),
                dirichlet(&inlet_c1),
                dirichlet(&inlet_c2),
            ],
        },
        species_data.metadata,
    );
    assert_eq!(fluid_problem.cell_count(), species_problem.cell_count());
    assert_eq!(
        fluid_problem.quadrature_points_per_cell(),
        species_problem.quadrature_points_per_cell()
    );
    let setup_time = setup_start.elapsed();

    // ponytail: outlet moves into the fluid system below; keep a copy for species outflow.
    let species_outlet = outlet.clone();
    let fluid_system = TensorFluidSystem::new(&fluid_problem, fluid_kernel())
        .with_wall_boundaries(cylinder.clone(), slip_wall);
    let fluid_system = if directional {
        fluid_system.with_directional_do_nothing_outflow(
            TensorKernelEdacDirectionalDoNothing2D::new(1.0),
            outlet,
            true,
        )
    } else {
        fluid_system
    };

    let integration_start = Instant::now();
    let fluid_state0 = Mat::<f64>::zeros(fluid_problem.system_size(), 1);
    let species_state0 = Mat::<f64>::zeros(species_problem.system_size(), 1);
    let mut fluid_integrator = TimeStepper::new(fluid_state0.as_ref());
    let mut species_integrator = TimeStepper::new(species_state0.as_ref());
    for outer_iter in 0..outer {
        fluid_integrator.run_steps(&fluid_system, dt, fluid_sub, "fluid");
        let fluid_state = fluid_integrator.state();
        let frozen_x =
            fluid_problem.sample_quadrature_field(fluid_state.as_ref(), "u");
        let frozen_y =
            fluid_problem.sample_quadrature_field(fluid_state.as_ref(), "v");
        frozen_x.assert_compatible(
            species_problem.cell_count(),
            species_problem.quadrature_points_per_cell(),
        );
        // Directional advective outflow: lets species leave through the
        // outlet, assumes zero concentration for re-entering flow. Without
        // it the dropped volume boundary flux acts as a closed wall and
        // species pile up at the outlet.
        let frozen_fx =
            fluid_problem.sample_facet_quadrature_field(fluid_state.as_ref(), "u");
        let frozen_fy =
            fluid_problem.sample_facet_quadrature_field(fluid_state.as_ref(), "v");
        frozen_fx.assert_compatible(
            species_problem.facet_count(),
            species_problem.facet_quadrature_points(),
        );
        let outflow = TensorKernelAdvectionOutflow2D::with_field_names(
            frozen_fx,
            frozen_fy,
            ["c0", "c1", "c2"],
        );
        let species_system = TensorFluidSystem::new(
            &species_problem,
            species_kernel(frozen_x, frozen_y, diffusivity),
        )
        .with_state_boundary(
            StateTensorBoundaryTerms::new().with_entities(species_outlet.clone(), outflow),
        );
        species_integrator.run_steps(&species_system, dt, species_sub, "species");
        let species_state = species_integrator.state();
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for (i, name) in ["c0", "c1", "c2"].iter().enumerate() {
            let field = species_problem.field_values(name, species_state.as_ref()).unwrap();
            for &v in &field.values {
                lo[i] = lo[i].min(v);
                hi[i] = hi[i].max(v);
            }
        }
        println!(
            "outer {outer_iter}/{outer} t={:.3} c0=[{:.3e},{:.3e}] c1=[{:.3e},{:.3e}] c2=[{:.3e},{:.3e}]",
            species_integrator.time(),
            lo[0],
            hi[0],
            lo[1],
            hi[1],
            lo[2],
            hi[2],
        );
    }
    let fluid_state = fluid_integrator.state();
    let species_state = species_integrator.state();
    let integration_time = integration_start.elapsed();

    println!(
        "frozen cylinder: outer={outer} fluid_dofs={} species_dofs={} stepper={} setup={setup_time:?} integration={integration_time:?} directional={directional}",
        fluid_problem.system_size(),
        species_problem.system_size(),
        fluid_integrator.name(),
    );
    if benchmark {
        return;
    }
    std::fs::create_dir_all("target").expect("failed to create output directory");
    let mesh = joined_export(
        &fluid_problem,
        fluid_state.as_ref(),
        &species_problem,
        species_state.as_ref(),
    );
    ormatex_sem_nd::io::write_csv("target/frozen_velocity_species_cylinder.csv", &mesh);
    maybe_write_vtk_mesh(&mesh, "target/frozen_velocity_species_cylinder.vtu");
}
