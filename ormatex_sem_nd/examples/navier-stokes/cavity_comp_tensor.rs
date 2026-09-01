//! Tensor-versus-generic lid-driven cavity comparison.

use std::collections::HashMap;
use std::time::Instant;

use faer::prelude::*;
use ndelement::{
    ciarlet::CiarletElement,
    ciarlet::{LagrangeElementFamily, LagrangeVariant},
    map::IdentityMap,
    types::{Continuity, ReferenceCellType},
};
use ndfunctionspace::{traits::FunctionSpace, FunctionSpaceImpl};
use ndmesh::{
    shapes::unit_square,
    traits::{Entity, Geometry, Mesh, Point, Topology},
};
use ormatex_sem_nd::{
    DofReduction2D, EdacNavierStokes2DConfig, FieldRegistry, KernelEdacMomentumConvection2D,
    KernelEdacPressureAdvection2D, KernelEdacPressureDiffusion2D, KernelEdacPressureDivergence2D,
    KernelEdacPressureGradient2D, KernelEdacViscousStress2D, QuadMesh, ResidualKernelSum,
    SEM2DProblem, TensorKernelEdacMomentumConvection2D, TensorKernelEdacPressureAdvection2D,
    TensorKernelEdacPressureDiffusion2D, TensorKernelEdacPressureDivergence2D,
    TensorKernelEdacPressureGradient2D, TensorKernelEdacViscousStress2D, TensorResidualKernelSum,
};

#[path = "../support/edac.rs"]
mod edac;
#[path = "../support/linear_system.rs"]
mod linear_system;
use edac::{
    advance, advance_tensor, write_spatial_csv, FluidSystem, GenericResidual, JacobianBackend,
    TensorFluidSystem,
};

const EPS: f64 = 1e-12;
const STEPS: usize = 300;
type QuadElement = CiarletElement<f64, IdentityMap, f64>;
type QuadSpace<'a> = FunctionSpaceImpl<'a, f64, f64, ReferenceCellType, QuadMesh, QuadElement>;

fn point_xy<P: Point<T = f64>>(point: P) -> [f64; 2] {
    let mut xy = [0.0; 2];
    point.coords(&mut xy);
    xy
}

fn boundary_facet_values(
    mesh: &QuadMesh,
    space: &QuadSpace<'_>,
) -> (HashMap<usize, f64>, HashMap<usize, f64>, usize) {
    let mut u = HashMap::new();
    let mut v = HashMap::new();
    for facet in mesh.entity_iter(ReferenceCellType::Interval) {
        let topology = facet.topology();
        let mut cells = topology.connected_entity_iter(ReferenceCellType::Quadrilateral);
        if cells.next().is_none() || cells.next().is_some() {
            continue;
        }
        let points: Vec<_> = facet.geometry().points().map(point_xy).collect();
        let midpoint = [
            points.iter().map(|xy| xy[0]).sum::<f64>() / points.len() as f64,
            points.iter().map(|xy| xy[1]).sum::<f64>() / points.len() as f64,
        ];
        let lid = midpoint[1] > 1.0 - EPS;
        for &dof in space
            .entity_closure_dofs(ReferenceCellType::Interval, facet.local_index())
            .unwrap()
        {
            u.insert(dof, if lid { 1.0 } else { 0.0 });
            v.insert(dof, 0.0);
        }
    }

    let mut pressure_dof = None;
    for point in mesh.entity_iter(ReferenceCellType::Point) {
        let xy = point_xy(point.geometry().points().next().unwrap());
        let dof = space
            .entity_closure_dofs(ReferenceCellType::Point, point.local_index())
            .unwrap()[0];
        if (xy[0] < EPS || xy[0] > 1.0 - EPS) && (xy[1] < EPS || xy[1] > 1.0 - EPS) {
            u.insert(dof, 0.0);
            v.insert(dof, 0.0);
        }
        if xy[0] < EPS && xy[1] < EPS {
            pressure_dof = Some(dof);
        }
    }
    (
        u,
        v,
        pressure_dof.expect("unit square has no lower-left point"),
    )
}

fn problem() -> SEM2DProblem<QuadMesh> {
    let mesh = unit_square(8, 8, ReferenceCellType::Quadrilateral, 1);
    let family = LagrangeElementFamily::<f64>::new(2, Continuity::Standard, LagrangeVariant::GLL);
    let space = FunctionSpaceImpl::new(&mesh, &family);
    let (u_values, v_values, pressure_dof) = boundary_facet_values(&mesh, &space);
    SEM2DProblem::new(
        mesh,
        2,
        FieldRegistry::new(["u", "v", "p"]),
        DofReduction2D::FieldSpecific {
            reductions: vec![
                DofReduction2D::DirichletValues {
                    values: u_values.into_iter().collect(),
                },
                DofReduction2D::DirichletValues {
                    values: v_values.into_iter().collect(),
                },
                DofReduction2D::DirichletValues {
                    values: vec![(pressure_dof, 0.0)],
                },
            ],
        },
    )
}

fn tensor_composed_kernel() -> impl ormatex_sem_nd::TensorResidualKernel<2> {
    let config = EdacNavierStokes2DConfig::new(1.0, 0.1, 10.0, 0.0);
    TensorResidualKernelSum::from_kernel(TensorKernelEdacMomentumConvection2D::new(config))
        .with(TensorKernelEdacPressureGradient2D::new(config))
        .with(TensorKernelEdacViscousStress2D::new(config))
        .with(TensorKernelEdacPressureDivergence2D::new(config))
        .with(TensorKernelEdacPressureAdvection2D::new(config))
        .with(TensorKernelEdacPressureDiffusion2D::new(config))
}

fn composed_kernel() -> ResidualKernelSum<'static> {
    let config = EdacNavierStokes2DConfig::new(1.0, 0.1, 10.0, 0.0);
    ResidualKernelSum::from_kernel(KernelEdacMomentumConvection2D::new(config))
        .with(KernelEdacPressureGradient2D::new(config))
        .with(KernelEdacViscousStress2D::new(config))
        .with(KernelEdacPressureDivergence2D::new(config))
        .with(KernelEdacPressureAdvection2D::new(config))
        .with(KernelEdacPressureDiffusion2D::new(config))
}

fn max_state_difference(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> f64 {
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.ncols(), b.ncols());
    (0..a.nrows())
        .flat_map(|row| (0..a.ncols()).map(move |col| (a[(row, col)] - b[(row, col)]).abs()))
        .fold(0.0, f64::max)
}

fn write_state(path: &str, problem: &SEM2DProblem<QuadMesh>, state: MatRef<'_, f64>) {
    std::fs::create_dir_all("target").expect("failed to create output directory");
    write_spatial_csv(
        path,
        [
            ("u", problem.field_values("u", state).unwrap()),
            ("v", problem.field_values("v", state).unwrap()),
            ("p", problem.field_values("p", state).unwrap()),
        ],
    );
}

fn main() {
    let problem = problem();
    let state0 = Mat::<f64>::zeros(problem.system_size(), 1);

    let tensor_system = TensorFluidSystem::new_with_backend(
        &problem,
        tensor_composed_kernel(),
        JacobianBackend::MatrixFree,
    );
    let start = Instant::now();
    let tensor_state = advance_tensor(&tensor_system, state0.as_ref(), 0.01, STEPS);
    let tensor_runtime = start.elapsed();

    let generic_system = FluidSystem::new_with_backend(
        &problem,
        GenericResidual(composed_kernel()),
        JacobianBackend::MatrixFree,
    );
    let start = Instant::now();
    let generic_state = advance(&generic_system, state0.as_ref(), 0.01, STEPS);
    let generic_runtime = start.elapsed();

    let max_difference = max_state_difference(tensor_state.as_ref(), generic_state.as_ref());
    assert!(
        max_difference < 1e-10,
        "tensor cavity differs from generic cavity: {max_difference:e}"
    );
    write_state(
        "target/navier_stokes_lid_driven_cavity_comp_tensor.csv",
        &problem,
        tensor_state.as_ref(),
    );
    write_state(
        "target/navier_stokes_lid_driven_cavity_comp_generic.csv",
        &problem,
        generic_state.as_ref(),
    );

    let speedup = generic_runtime.as_secs_f64() / tensor_runtime.as_secs_f64();
    println!(
        "cavity tensor comparison: max |tensor-generic|={max_difference:.3e}, tensor={tensor_runtime:?}, generic={generic_runtime:?}, speedup={speedup:.2}x"
    );
    println!(
        "tensor result: target/navier_stokes_lid_driven_cavity_comp_tensor.csv (matches cavity_comp.rs composition)"
    );
}
