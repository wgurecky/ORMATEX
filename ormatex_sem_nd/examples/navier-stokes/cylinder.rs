//! EDAC/Smagorinsky vortex shedding around a cylinder on a Gmsh quad mesh.

use std::fs::File;
use std::io::{BufWriter, Write};

use faer::prelude::*;
use ormatex::ode_sys::IntegrateSys;
use ormatex_sem_nd::{
    gmsh_quad_data, DofReduction2D, FieldRegistry, KernelEdacNavierStokes2D, MeshMetadata,
    SEM2DProblem,
};

#[path = "../support/edac.rs"]
mod edac;
#[path = "../support/linear_system.rs"]
mod linear_system;
use edac::{epi3, FluidSystem};

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

fn main() {
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

    // Prescribe the free-stream velocity at the inlet and use a pressure
    // reference at the outlet. The channel edges are slip/symmetry boundaries.
    let u_in = 1.0;
    let inlet_u: Vec<_> = inlet.iter().copied().map(|facet| (facet, u_in)).collect();
    let inlet_v: Vec<_> = inlet.iter().copied().map(|facet| (facet, 0.0)).collect();
    let wall_u: Vec<_> = cylinder.iter().copied().map(|facet| (facet, 0.0)).collect();
    let symmetry: Vec<_> = metadata
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
        .chain(symmetry)
        .map(|facet| (facet, 0.0))
        .collect();
    let outlet_p: Vec<_> = outlet.iter().copied().map(|facet| (facet, 0.0)).collect();
    let problem = SEM2DProblem::new_with_fields_and_metadata(
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
                dirichlet(&outlet_p),
            ],
        },
        metadata,
    );

    // Keep this example's user-facing setup compact; the kernel owns the model.
    let kernel = KernelEdacNavierStokes2D::new(1.0, 1.0 / 200.0, 4.0, 0.1);
    let system = FluidSystem::new(&problem, kernel);
    let state0 = Mat::<f64>::zeros(problem.system_size(), 1);
    let dt = 0.005;
    let nsteps = 1000;
    let mut integrator = epi3(state0.as_ref());
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
        File::create("target/navier_stokes_cylinder_probe.csv")
            .expect("failed to create probe csv"),
    );
    writeln!(probe, "t,u,v,p").unwrap();
    for step in 0..nsteps {
        let result = integrator
            .step(&system, dt)
            .unwrap_or_else(|error| panic!("EDAC step {step} failed: {}", error.msg));
        integrator.accept_step(result);
        let state = integrator.state();
        let u = problem.field_values("u", state.as_ref()).unwrap();
        let v = problem.field_values("v", state.as_ref()).unwrap();
        let p = problem.field_values("p", state.as_ref()).unwrap();
        writeln!(
            probe,
            "{:.8},{:.9e},{:.9e},{:.9e}",
            integrator.time(),
            u.values[probe_u],
            v.values[probe_v],
            p.values[probe_p],
        )
        .unwrap();
    }
    let state = integrator.state();

    let mut output = BufWriter::new(
        File::create("target/navier_stokes_cylinder.csv").expect("failed to create output csv"),
    );
    writeln!(output, "field,x,y,value").unwrap();
    for (field, field_values) in [
        ("u", problem.field_values("u", state.as_ref()).unwrap()),
        ("v", problem.field_values("v", state.as_ref()).unwrap()),
        ("p", problem.field_values("p", state.as_ref()).unwrap()),
    ] {
        for ((x, y), value) in field_values.positions.into_iter().zip(field_values.values) {
            writeln!(output, "{field},{x:.8},{y:.8},{value:.9e}").unwrap();
        }
    }
}
