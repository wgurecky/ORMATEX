//! Lid-driven cavity comparison between fused and composed EDAC kernels.

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
    KernelEdacMomentumConvectionSplit2D, KernelEdacNavierStokes2D, KernelEdacPressureAdvection2D,
    KernelEdacPressureAdvectionSplit2D, KernelEdacPressureDiffusion2D,
    KernelEdacPressureDivergence2D, KernelEdacPressureGradient2D, KernelEdacViscousStress2D,
    QuadMesh, ResidualKernelSum, SEM2DProblem,
};

#[path = "../support/edac.rs"]
mod edac;
#[path = "../support/linear_system.rs"]
mod linear_system;
use edac::{advance, write_spatial_csv, FluidSystem};

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

fn composed_kernel() -> ResidualKernelSum<'static> {
    let config = EdacNavierStokes2DConfig::new(1.0, 0.1, 10.0, 0.0);
    ResidualKernelSum::from_kernel(KernelEdacMomentumConvection2D::new(config))
        .with(KernelEdacPressureGradient2D::new(config))
        .with(KernelEdacViscousStress2D::new(config))
        .with(KernelEdacPressureDivergence2D::new(config))
        .with(KernelEdacPressureAdvection2D::new(config))
        .with(KernelEdacPressureDiffusion2D::new(config))
}

fn split_composed_kernel() -> ResidualKernelSum<'static> {
    let config = EdacNavierStokes2DConfig::new(1.0, 0.1, 10.0, 0.0);
    ResidualKernelSum::from_kernel(KernelEdacMomentumConvectionSplit2D::new(config))
        .with(KernelEdacPressureGradient2D::new(config))
        .with(KernelEdacViscousStress2D::new(config))
        .with(KernelEdacPressureDivergence2D::new(config))
        .with(KernelEdacPressureAdvectionSplit2D::new(config))
        .with(KernelEdacPressureDiffusion2D::new(config))
}

fn max_state_difference(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> f64 {
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.ncols(), b.ncols());
    (0..a.nrows())
        .flat_map(|row| (0..a.ncols()).map(move |col| (a[(row, col)] - b[(row, col)]).abs()))
        .fold(0.0, f64::max)
}

fn write_state(problem: &SEM2DProblem<QuadMesh>, state: MatRef<'_, f64>) {
    std::fs::create_dir_all("target").expect("failed to create output directory");
    let u = problem.field_values("u", state).unwrap();
    let v = problem.field_values("v", state).unwrap();
    let p = problem.field_values("p", state).unwrap();
    write_spatial_csv(
        "target/navier_stokes_lid_driven_cavity_comp.csv",
        [("u", u), ("v", v), ("p", p)],
    );
}

fn main() {
    let problem = problem();
    let state0 = Mat::<f64>::zeros(problem.system_size(), 1);
    let split = std::env::args().any(|arg| arg == "--split");

    let fused_system =
        FluidSystem::new(&problem, KernelEdacNavierStokes2D::new(1.0, 0.1, 10.0, 0.0));
    let start = Instant::now();
    let fused_state = advance(&fused_system, state0.as_ref(), 0.01, STEPS);
    let fused_runtime = start.elapsed();

    let mut composed_system = FluidSystem::new(
        &problem,
        if split {
            split_composed_kernel()
        } else {
            composed_kernel()
        },
    );
    if split {
        composed_system = composed_system.with_split_boundary();
    }
    let start = Instant::now();
    let composed_state = advance(&composed_system, state0.as_ref(), 0.01, STEPS);
    let composed_runtime = start.elapsed();

    let max_difference = max_state_difference(fused_state.as_ref(), composed_state.as_ref());
    if !split {
        assert!(
            max_difference < 1e-10,
            "composed EDAC state differs from fused state: {max_difference:e}"
        );
    }
    write_state(&problem, composed_state.as_ref());

    let ratio = composed_runtime.as_secs_f64() / fused_runtime.as_secs_f64();
    assert!(
        ratio < 2.0,
        "composed cavity runtime is more than 2x fused runtime: ratio={ratio:.3}"
    );
    println!(
        "cavity comparison (split={split}): max |fused-composed|={max_difference:.3e}, fused={fused_runtime:?}, composed={composed_runtime:?}, ratio={ratio:.3}"
    );
    println!("composed result: target/navier_stokes_lid_driven_cavity_comp.csv");
}
