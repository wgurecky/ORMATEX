//! Tensor-versus-generic EDAC cylinder comparison on the Gmsh quad mesh.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::{Duration, Instant};

use faer::prelude::*;
use ormatex::ode_sys::{IntegrateSys, OdeSys};
use ormatex_sem_nd::{
    gmsh_quad_data, DofReduction2D, EdacNavierStokes2DConfig, FieldRegistry,
    KernelEdacDirectionalDoNothing2D, KernelEdacMomentumConvectionSplit2D,
    KernelEdacPressureAdvectionSplit2D, KernelEdacPressureDiffusion2D,
    KernelEdacPressureDivergence2D, KernelEdacPressureGradient2D, KernelEdacViscousStress2D,
    MeshMetadata, ResidualKernelSum, SEM2DProblem, TensorKernelEdacDirectionalDoNothing2D,
    TensorKernelEdacMomentumConvectionSplit2D, TensorKernelEdacPressureAdvectionSplit2D,
    TensorKernelEdacPressureDiffusion2D, TensorKernelEdacPressureDivergence2D,
    TensorKernelEdacPressureGradient2D, TensorKernelEdacViscousStress2D, TensorResidualKernel,
    TensorResidualKernelSum,
};

#[path = "../support/edac.rs"]
mod edac;
#[path = "../support/linear_system.rs"]
mod linear_system;
use edac::{
    epi3, write_spatial_csv, FluidSystem, GenericResidual, JacobianBackend, TensorFluidSystem,
};

fn boundary_facets(data: &MeshMetadata, tag: usize) -> Vec<usize> {
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

fn tensor_split_kernel() -> impl TensorResidualKernel<2> {
    let config = EdacNavierStokes2DConfig::new(1.0, 1.0 / 200.0, 4.0, 0.1);
    TensorResidualKernelSum::from_kernel(TensorKernelEdacMomentumConvectionSplit2D::new(config))
        .with(TensorKernelEdacPressureGradient2D::new(config))
        .with(TensorKernelEdacViscousStress2D::new(config))
        .with(TensorKernelEdacPressureDivergence2D::new(config))
        .with(TensorKernelEdacPressureAdvectionSplit2D::new(config))
        .with(TensorKernelEdacPressureDiffusion2D::new(config))
}

fn split_kernel() -> ResidualKernelSum<'static> {
    let config = EdacNavierStokes2DConfig::new(1.0, 1.0 / 200.0, 4.0, 0.1);
    ResidualKernelSum::from_kernel(KernelEdacMomentumConvectionSplit2D::new(config))
        .with(KernelEdacPressureGradient2D::new(config))
        .with(KernelEdacViscousStress2D::new(config))
        .with(KernelEdacPressureDivergence2D::new(config))
        .with(KernelEdacPressureAdvectionSplit2D::new(config))
        .with(KernelEdacPressureDiffusion2D::new(config))
}

fn problem_and_outlet(
    directional: bool,
) -> (
    SEM2DProblem<ormatex_sem_nd::QuadMesh>,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
) {
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

    let inlet_u: Vec<_> = inlet.iter().copied().map(|facet| (facet, 1.0)).collect();
    let inlet_v: Vec<_> = inlet.iter().copied().map(|facet| (facet, 0.0)).collect();
    let wall_u: Vec<_> = cylinder.iter().copied().map(|facet| (facet, 0.0)).collect();
    let slip_wall: Vec<_> = metadata
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
        .chain(slip_wall.iter().copied())
        .map(|facet| (facet, 0.0))
        .collect();
    let outlet_p: Vec<_> = outlet.iter().copied().map(|facet| (facet, 0.0)).collect();
    let problem = SEM2DProblem::new_with_metadata(
        mesh,
        2,
        FieldRegistry::new(["u", "v", "p"]),
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
                if directional {
                    DofReduction2D::None
                } else {
                    dirichlet(&outlet_p)
                },
            ],
        },
        metadata,
    );
    (problem, outlet, cylinder, slip_wall)
}

fn advance_measured<'a, S>(
    system: &'a S,
    state0: MatRef<'_, f64>,
    dt: f64,
    nsteps: usize,
) -> (Mat<f64>, Duration)
where
    S: OdeSys<'a>,
{
    let mut integrator = epi3(state0);
    let start = Instant::now();
    for step in 0..nsteps {
        let result = integrator
            .step(system, dt)
            .unwrap_or_else(|error| panic!("EDAC step {step} failed: {}", error.msg));
        integrator.accept_step(result);
    }
    (integrator.state(), start.elapsed())
}

fn max_state_difference(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> f64 {
    assert_eq!(a.nrows(), b.nrows());
    (0..a.nrows())
        .map(|row| (a[(row, 0)] - b[(row, 0)]).abs())
        .fold(0.0, f64::max)
}

fn main() {
    let directional = std::env::args().any(|arg| arg == "--directional");
    let (problem, outlet, cylinder, slip_wall) = problem_and_outlet(directional);
    let state0 = Mat::<f64>::zeros(problem.system_size(), 1);

    let tensor_system = if directional {
        TensorFluidSystem::new_with_backend(
            &problem,
            tensor_split_kernel(),
            JacobianBackend::MatrixFree,
        )
        .with_wall_boundaries(cylinder.clone(), slip_wall.clone())
        .with_directional_do_nothing_outflow(
            TensorKernelEdacDirectionalDoNothing2D::new(1.0),
            outlet.clone(),
            true,
        )
    } else {
        TensorFluidSystem::new_with_backend(
            &problem,
            tensor_split_kernel(),
            JacobianBackend::MatrixFree,
        )
        .with_wall_boundaries(cylinder.clone(), slip_wall.clone())
    };
    let (tensor_state, tensor_runtime) =
        advance_measured(&tensor_system, state0.as_ref(), 0.05, 100);

    let generic_system = if directional {
        FluidSystem::new_with_backend(
            &problem,
            GenericResidual(split_kernel()),
            JacobianBackend::MatrixFree,
        )
        .with_wall_boundaries(cylinder, slip_wall)
        .with_directional_do_nothing_outflow(
            KernelEdacDirectionalDoNothing2D::new(1.0),
            outlet,
            true,
        )
    } else {
        FluidSystem::new_with_backend(
            &problem,
            GenericResidual(split_kernel()),
            JacobianBackend::MatrixFree,
        )
        .with_wall_boundaries(cylinder, slip_wall)
    };
    let (generic_state, generic_runtime) =
        advance_measured(&generic_system, state0.as_ref(), 0.05, 100);

    let max_difference = max_state_difference(tensor_state.as_ref(), generic_state.as_ref());
    assert!(
        max_difference < 1e-9,
        "tensor cylinder differs from generic cylinder: {max_difference:e}"
    );

    std::fs::create_dir_all("target").expect("failed to create output directory");
    let probe_u = nearest(
        &problem
            .field_values("u", state0.as_ref())
            .unwrap()
            .positions,
        (2.0, 0.5),
    );
    let probe_v = nearest(
        &problem
            .field_values("v", state0.as_ref())
            .unwrap()
            .positions,
        (2.0, 0.5),
    );
    let probe_p = nearest(
        &problem
            .field_values("p", state0.as_ref())
            .unwrap()
            .positions,
        (2.0, 0.5),
    );
    let mut probe = BufWriter::new(
        File::create("target/navier_stokes_cylinder_tensor_probe.csv")
            .expect("failed to create tensor cylinder probe csv"),
    );
    writeln!(probe, "u,v,p").unwrap();
    writeln!(
        probe,
        "{:.9e},{:.9e},{:.9e}",
        tensor_state[(problem.field_offset(0) + probe_u, 0)],
        tensor_state[(problem.field_offset(1) + probe_v, 0)],
        tensor_state[(problem.field_offset(2) + probe_p, 0)],
    )
    .unwrap();
    write_spatial_csv(
        "target/navier_stokes_cylinder_tensor.csv",
        [
            (
                "u",
                problem.field_values("u", tensor_state.as_ref()).unwrap(),
            ),
            (
                "v",
                problem.field_values("v", tensor_state.as_ref()).unwrap(),
            ),
            (
                "p",
                problem.field_values("p", tensor_state.as_ref()).unwrap(),
            ),
        ],
    );
    write_spatial_csv(
        "target/navier_stokes_cylinder_generic.csv",
        [
            (
                "u",
                problem.field_values("u", generic_state.as_ref()).unwrap(),
            ),
            (
                "v",
                problem.field_values("v", generic_state.as_ref()).unwrap(),
            ),
            (
                "p",
                problem.field_values("p", generic_state.as_ref()).unwrap(),
            ),
        ],
    );

    let speedup = generic_runtime.as_secs_f64() / tensor_runtime.as_secs_f64();
    println!(
        "cylinder tensor comparison (directional={directional}): max |tensor-generic|={max_difference:.3e}, tensor={tensor_runtime:?}, generic={generic_runtime:?}, speedup={speedup:.2}x"
    );
    println!(
        "tensor result: target/navier_stokes_cylinder_tensor.csv (compare with cylinder.rs output)"
    );
}
