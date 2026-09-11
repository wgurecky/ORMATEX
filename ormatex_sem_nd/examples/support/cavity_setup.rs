use std::collections::HashMap;

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
use ormatex_sem_nd::{DofReduction2D, FieldRegistry, QuadMesh, SEM2DProblem};

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

pub fn problem() -> SEM2DProblem<QuadMesh> {
    problem_with_resolution(32, 32)
}

pub fn problem_with_resolution(nx: usize, ny: usize) -> SEM2DProblem<QuadMesh> {
    let mesh = unit_square(nx, ny, ReferenceCellType::Quadrilateral, 1);
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
