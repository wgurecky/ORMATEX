//! 2D GLL spectral-element advection-diffusion on a periodic quadrilateral mesh.

use std::fs::File;
use std::io::Write;

use faer::prelude::*;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{
    shapes::unit_square,
    traits::{Entity, Geometry, Mesh, Point, Topology},
    SingleElementMesh,
};
use ormatex_sem_nd::{DofReduction2D, KernelAdvDiff2D, KernelVolumeSource, SEM2DProblem};

#[path = "support/linear_system.rs"]
mod linear_system;
use linear_system::{implicit_euler_final_state, LinearOdeSys};

fn main() {
    let mesh: SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>> =
        unit_square(32, 2, ReferenceCellType::Quadrilateral);
    let facet_pairs = unit_square_periodic_pairs(&mesh);
    let problem = SEM2DProblem::new(
        mesh,
        2,
        DofReduction2D::Periodic {
            facet_pairs,
            tolerance: 1e-12,
        },
    );
    let mass = problem.assemble_lumped_mass();
    let operator = problem.assemble_bilinear(&KernelAdvDiff2D::new(0.001, [0.5, 0.1]));
    let n = mass.nrows();
    assert!(
        (problem
            .assemble_linear(&KernelVolumeSource::new(1.0))
            .iter()
            .sum::<f64>()
            - 1.0)
            .abs()
            < 1e-12
    );

    let system = LinearOdeSys::new(mass, operator, vec![0.0; n]);
    let positions = problem.dof_positions();
    let mut y0 = Mat::<f64>::zeros(n, 1);
    for i in 0..n {
        y0[(i, 0)] = periodic_gaussian(positions[i].0, positions[i].1, 0.5, 0.5, 0.1);
    }
    let y = implicit_euler_final_state(&system, y0.as_ref(), 0.01, 200, 1e-12);

    let mut output = File::create("target/ex_nd_2d_quad_out.csv").unwrap();
    writeln!(output, "x,y,u").unwrap();
    for i in 0..n {
        writeln!(
            output,
            "{:.6},{:.6},{:.9e}",
            positions[i].0,
            positions[i].1,
            y[(i, 0)]
        )
        .unwrap();
    }
}

fn unit_square_periodic_pairs(
    mesh: &SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>,
) -> Vec<[usize; 2]> {
    const EPS: f64 = 1e-12;
    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut bottom = Vec::new();
    let mut top = Vec::new();
    for facet in mesh.entity_iter(ReferenceCellType::Interval) {
        let topology = facet.topology();
        let mut cells = topology.connected_entity_iter(ReferenceCellType::Quadrilateral);
        if cells.next().is_none() || cells.next().is_some() {
            continue;
        }
        let mut midpoint = [0.0_f64; 2];
        for point in facet.geometry().points() {
            let mut xy = [0.0; 2];
            point.coords(&mut xy);
            midpoint[0] += xy[0] / 2.0;
            midpoint[1] += xy[1] / 2.0;
        }
        let entry = (
            if midpoint[0].abs() < EPS || (midpoint[0] - 1.0).abs() < EPS {
                midpoint[1]
            } else {
                midpoint[0]
            },
            facet.local_index(),
        );
        if midpoint[0].abs() < EPS {
            left.push(entry);
        } else if (midpoint[0] - 1.0).abs() < EPS {
            right.push(entry);
        } else if midpoint[1].abs() < EPS {
            bottom.push(entry);
        } else if (midpoint[1] - 1.0).abs() < EPS {
            top.push(entry);
        }
    }
    for facets in [&mut left, &mut right, &mut bottom, &mut top] {
        facets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }
    left.into_iter()
        .zip(right)
        .chain(bottom.into_iter().zip(top))
        .map(|((_, a), (_, b))| [a, b])
        .collect()
}

fn periodic_gaussian(x: f64, y: f64, x0: f64, y0: f64, sigma: f64) -> f64 {
    let distance = |value: f64, center: f64| {
        let mut delta = value - center;
        if delta > 0.5 {
            delta -= 1.0;
        } else if delta < -0.5 {
            delta += 1.0;
        }
        delta
    };
    let dx = distance(x, x0);
    let dy = distance(y, y0);
    (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp()
}
