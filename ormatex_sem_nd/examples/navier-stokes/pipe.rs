//! Pressure-driven 2D channel validation for the EDAC solver.

use faer::prelude::*;
use ndelement::types::ReferenceCellType;
use ndmesh::{
    traits::{Builder, Entity, Geometry, Mesh, Point, Topology},
    SingleElementMeshBuilder,
};
use ormatex_sem_nd::{
    DofReduction2D, FieldRegistry, KernelEdacNavierStokes2D, QuadMesh, SEM2DProblem,
};

#[path = "../support/edac.rs"]
mod edac;
#[path = "../support/linear_system.rs"]
mod linear_system;
use edac::{advance, write_spatial_csv, FluidSystem};

const EPS: f64 = 1e-12;

fn rectangular_mesh(nx: usize, ny: usize, length: f64, height: f64) -> QuadMesh {
    let mut builder = SingleElementMeshBuilder::new_with_capacity(
        2,
        (nx + 1) * (ny + 1),
        nx * ny,
        (ReferenceCellType::Quadrilateral, 1),
    );
    for i in 0..=nx {
        for j in 0..=ny {
            let id = i * (ny + 1) + j;
            builder.add_point(
                id,
                &[length * i as f64 / nx as f64, height * j as f64 / ny as f64],
            );
        }
    }
    for i in 0..nx {
        for j in 0..ny {
            let origin = i * (ny + 1) + j;
            builder.add_cell(
                j * nx + i,
                &[origin, origin + ny + 1, origin + 1, origin + ny + 2],
            );
        }
    }
    builder.create_mesh()
}

fn boundary_facets(mesh: &QuadMesh) -> Vec<(usize, (f64, f64))> {
    mesh.entity_iter(ReferenceCellType::Interval)
        .filter_map(|facet| {
            let topology = facet.topology();
            let mut cells = topology.connected_entity_iter(ReferenceCellType::Quadrilateral);
            (cells.next().is_some() && cells.next().is_none()).then(|| {
                let mut midpoint = (0.0, 0.0);
                let mut count = 0.0;
                for point in facet.geometry().points() {
                    let mut xy = [0.0; 2];
                    point.coords(&mut xy);
                    midpoint.0 += xy[0];
                    midpoint.1 += xy[1];
                    count += 1.0;
                }
                (
                    facet.local_index(),
                    (midpoint.0 / count, midpoint.1 / count),
                )
            })
        })
        .collect()
}

fn main() {
    let length = 4.0;
    let height = 1.0;
    let rho = 1.0;
    let nu = 0.1;
    let delta_p = 0.1;
    let nx = 16;
    let ny = 8;
    let mesh = rectangular_mesh(nx, ny, length, height);
    let facets = boundary_facets(&mesh);
    let walls: Vec<_> = facets
        .iter()
        .filter(|(_, (_, y))| y.abs() < EPS || (*y - height).abs() < EPS)
        .map(|&(facet, _)| (facet, 0.0))
        .collect();
    let inlet: Vec<_> = facets
        .iter()
        .filter(|(_, (x, _))| x.abs() < EPS)
        .map(|&(facet, _)| (facet, delta_p))
        .collect();
    let outlet: Vec<_> = facets
        .iter()
        .filter(|(_, (x, _))| (*x - length).abs() < EPS)
        .map(|&(facet, _)| (facet, 0.0))
        .collect();
    assert!(!walls.is_empty() && !inlet.is_empty() && !outlet.is_empty());

    let problem = SEM2DProblem::new(
        mesh,
        2,
        FieldRegistry::new(["u", "v", "p"]),
        DofReduction2D::FieldSpecific {
            reductions: vec![
                DofReduction2D::Dirichlet {
                    facets: walls.clone(),
                },
                DofReduction2D::Dirichlet { facets: walls },
                DofReduction2D::Dirichlet {
                    facets: inlet.into_iter().chain(outlet).collect(),
                },
            ],
        },
    );
    let kernel = KernelEdacNavierStokes2D::new(rho, nu, 10.0, 0.0);
    let system = FluidSystem::new(&problem, kernel);
    let state0 = Mat::<f64>::zeros(problem.system_size(), 1);
    let state = advance(&system, state0.as_ref(), 0.02, 500);
    std::fs::create_dir_all("target").expect("failed to create output directory");

    let u = problem.field_values("u", state.as_ref()).unwrap();
    let v = problem.field_values("v", state.as_ref()).unwrap();
    let p = problem.field_values("p", state.as_ref()).unwrap();
    let mut max_error: f64 = 0.0;
    let mut max_v: f64 = 0.0;
    let mut profile_count = 0;
    for (&(x, y), &value) in u.positions.iter().zip(&u.values) {
        assert!(value.is_finite(), "non-finite pipe velocity");
        if (x - 0.5 * length).abs() < EPS {
            let analytic = delta_p * y * (height - y) / (2.0 * rho * nu * length);
            max_error = max_error.max((value - analytic).abs());
            profile_count += 1;
        }
    }
    for &value in &v.values {
        assert!(value.is_finite(), "non-finite pipe transverse velocity");
        max_v = max_v.max(value.abs());
    }
    for &value in &p.values {
        assert!(value.is_finite(), "non-finite pipe pressure");
    }
    assert!(profile_count > 0, "no centerline pipe profile DOFs");
    assert!(max_v < 1e-2, "transverse pipe velocity too large: {max_v}");
    assert!(
        max_error < 2e-2,
        "pipe profile error too large: {max_error}"
    );
    write_spatial_csv(
        "target/navier_stokes_pipe.csv",
        [("u", u), ("v", v), ("p", p)],
    );
    println!("pipe validation: max profile error={max_error:.3e}, max |v|={max_v:.3e}");
}
