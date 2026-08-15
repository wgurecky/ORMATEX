//! Low-Reynolds-number lid-driven cavity validation for the EDAC solver.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

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
use ormatex_sem_nd::{DofReduction2D, KernelEdacNavierStokes2D, QuadMesh, SEM2DProblem};

#[path = "../support/edac.rs"]
mod edac;
#[path = "../support/linear_system.rs"]
mod linear_system;
use edac::{advance, FluidSystem};

const EPS: f64 = 1e-12;
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

fn main() {
    let mesh = unit_square(8, 8, ReferenceCellType::Quadrilateral, 1);
    let family = LagrangeElementFamily::<f64>::new(2, Continuity::Standard, LagrangeVariant::GLL);
    let space = FunctionSpaceImpl::new(&mesh, &family);
    let (u_values, v_values, pressure_dof) = boundary_facet_values(&mesh, &space);
    let problem = SEM2DProblem::new(
        mesh,
        2,
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
    );
    let kernel = KernelEdacNavierStokes2D::new(1.0, 0.1, 10.0, 0.0);
    let system = FluidSystem::new(&problem, kernel);
    let state0 = Mat::<f64>::zeros(problem.system_size(3), 1);
    let state = advance(&system, state0.as_ref(), 0.01, 300);
    std::fs::create_dir_all("target").expect("failed to create output directory");

    let u_positions = problem.field_dof_positions(0);
    let v_positions = problem.field_dof_positions(1);
    let mut min_u = f64::INFINITY;
    let mut max_abs_v: f64 = 0.0;
    let mut output = BufWriter::new(
        File::create("target/navier_stokes_lid_driven_cavity.csv")
            .expect("failed to create cavity output csv"),
    );
    writeln!(output, "field,x,y,value").unwrap();
    for field in 0..3 {
        let positions = match field {
            0 => &u_positions,
            1 => &v_positions,
            _ => &problem.field_dof_positions(2),
        };
        for (local, &(x, y)) in positions.iter().enumerate() {
            let value = state[(problem.field_offset(field, 3) + local, 0)];
            assert!(value.is_finite(), "non-finite cavity state");
            writeln!(output, "{field},{x:.9},{y:.9},{value:.9e}").unwrap();
            if field == 0 && x > 0.1 && x < 0.9 && y > 0.1 && y < 0.9 {
                min_u = min_u.min(value);
            }
            if field == 1 && x > 0.1 && x < 0.9 && y > 0.1 && y < 0.9 {
                max_abs_v = max_abs_v.max(value.abs());
            }
        }
    }
    assert!(
        min_u < -1e-3,
        "cavity has no recirculation: min interior u={min_u}"
    );
    assert!(max_abs_v > 1e-3, "cavity has no transverse motion");
    println!("cavity validation: min interior u={min_u:.3e}, max |v|={max_abs_v:.3e}");
}
