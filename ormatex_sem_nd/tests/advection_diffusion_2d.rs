use faer::prelude::*;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{
    shapes::unit_square,
    traits::{Entity, Geometry, Mesh, Point, Topology},
    SingleElementMesh,
};
use ormatex::matexp_krylov::KrylovExpm;
use ormatex::matexp_pade::PadeExpm;
use ormatex::ode_epirk::EpirkIntegrator;
use ormatex::ode_sys::IntegrateSys;
use ormatex_sem_nd::{DofReduction2D, FieldRegistry, KernelAdvDiff2D, SEM2DProblem};

#[path = "../examples/support/linear_system.rs"]
mod linear_system;
use linear_system::LinearOdeSys;

type QuadMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

fn run_epi3_krylov(system: &LinearOdeSys, y0: MatRef<'_, f64>, dt: f64, nsteps: usize) -> Mat<f64> {
    let expmv = Box::new(PadeExpm::new(12));
    let krylov = KrylovExpm::new(expmv, 30, 100, 1e-12, Some(2));
    let mut solver = EpirkIntegrator::new(0.0, y0, "epi3".to_string(), krylov);
    for _ in 0..nsteps {
        let step = solver.step(system, dt).unwrap();
        solver.accept_step(step);
    }
    solver.state()
}

fn unit_square_periodic_pairs(mesh: &QuadMesh) -> Vec<[usize; 2]> {
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
        let mut midpoint = [0.0; 2];
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
        .map(|((_, source), (_, target))| [source, target])
        .collect()
}

fn periodic_gaussian(x: f64, y: f64, center_x: f64, center_y: f64, sigma: f64) -> f64 {
    (-4..=4)
        .flat_map(|image_x| {
            (-4..=4).map(move |image_y| {
                let dx = x - center_x - image_x as f64;
                let dy = y - center_y - image_y as f64;
                (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp()
            })
        })
        .sum()
}

#[test]
fn p2_epi3_krylov_advects_and_diffuses_periodic_gaussian() {
    let nx = 64;
    let ny = 32;
    let velocity = [0.5, 0.0];
    let diffusivity = 0.001;
    let sigma0 = 0.1;
    let center0 = [0.25, 0.5];
    let final_time = 0.5;
    let dt = 0.01;
    let nsteps = (final_time / dt) as usize;
    let mesh = unit_square(nx, ny, ReferenceCellType::Quadrilateral, 1);
    let facet_pairs = unit_square_periodic_pairs(&mesh);
    let problem = SEM2DProblem::new(
        mesh,
        2,
        FieldRegistry::new(["u"]),
        DofReduction2D::Periodic {
            facet_pairs,
            tolerance: 1e-12,
        },
    );
    let mass = problem.assemble_lumped_mass();
    let operator = problem.assemble_bilinear(&KernelAdvDiff2D::new(diffusivity, velocity));
    let n = problem.reduced_size();
    let positions = problem.dof_positions();
    let y0 = Mat::from_fn(n, 1, |row, _| {
        periodic_gaussian(
            positions[row].0,
            positions[row].1,
            center0[0],
            center0[1],
            sigma0,
        )
    });
    let system = LinearOdeSys::new(mass, operator, vec![0.0; n]);
    let state = run_epi3_krylov(&system, y0.as_ref(), dt, nsteps);

    let sigma = (sigma0 * sigma0 + 2.0 * diffusivity * final_time).sqrt();
    let center_x = center0[0] + velocity[0] * final_time;
    let center_y = center0[1] + velocity[1] * final_time;
    let amplitude = sigma0 * sigma0 / (sigma * sigma);
    let max_error = (0..n)
        .map(|row| {
            let expected = amplitude
                * periodic_gaussian(
                    positions[row].0,
                    positions[row].1,
                    center_x,
                    center_y,
                    sigma,
                );
            (state[(row, 0)] - expected).abs()
        })
        .fold(0.0_f64, f64::max);
    assert!(
        max_error < 3e-4,
        "maximum periodic Gaussian error: {max_error}"
    );
}
