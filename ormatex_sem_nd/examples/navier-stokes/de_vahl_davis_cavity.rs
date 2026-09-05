//! De Vahl Davis differentially heated square cavity, Ra = 1e3, Pr = 0.71.
//!
//! The hot left wall has T=1, the cold right wall has T=0, horizontal walls
//! are adiabatic, and all walls are no-slip. The tensor EDAC terms are
//! composed with separate Boussinesq and state-coupled energy terms.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

use faer::prelude::*;
use ormatex_sem_nd::{
    dirichlet_values_with_precedence, DofReduction2D, EdacNavierStokes2DConfig, FieldRegistry,
    FieldValues, TensorKernelBoussinesq2D, TensorKernelEdacMomentumConvectionSplit2D,
    TensorKernelEdacPressureAdvectionSplit2D, TensorKernelEdacPressureDiffusion2D,
    TensorKernelEdacPressureDivergence2D, TensorKernelEdacPressureGradient2D,
    TensorKernelEdacViscousStress2D, TensorKernelEnergyAdvectionDiffusion2D, TensorResidualKernel,
    TensorResidualKernelSet, TensorResidualKernelSum,
};

#[path = "../support/edac.rs"]
mod edac;
#[path = "../support/linear_system.rs"]
mod linear_system;

use edac::{advance_tensor, JacobianBackend, TensorFluidSystem};

const P: usize = 2;
const CELLS: usize = 20;
const RA: f64 = 1.0e3;
const PR: f64 = 0.71;
const DT: f64 = 0.005;
const STEPS: usize = 1000;

fn tensor_kernel() -> impl TensorResidualKernel<2> {
    let config = EdacNavierStokes2DConfig::new(1.0, PR, 20.0, 0.0);
    let edac = TensorResidualKernelSum::from_kernel(
        TensorKernelEdacMomentumConvectionSplit2D::new(config),
    )
    .with(TensorKernelEdacPressureGradient2D::new(config))
    .with(TensorKernelEdacViscousStress2D::new(config))
    .with(TensorKernelEdacPressureDivergence2D::new(config))
    .with(TensorKernelEdacPressureAdvectionSplit2D::new(config))
    .with(TensorKernelEdacPressureDiffusion2D::new(config));

    TensorResidualKernelSet::from_kernel(edac)
        // gravity is represented as the positive upward buoyancy coefficient;
        // the residual sign produces upward acceleration for hot fluid.
        .with(TensorKernelBoussinesq2D::new(RA * PR, [0.0, 1.0]))
        .with(TensorKernelEnergyAdvectionDiffusion2D::new(1.0))
}

fn load_problem() -> (
    ormatex_sem_nd::SEM2DProblem<ormatex_sem_nd::QuadMesh>,
    Vec<usize>,
) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/navier-stokes/de_vahl_davis_cavity.msh"
    );
    let data = ormatex_sem_nd::gmsh_quad_data(path).expect("failed to load cavity Gmsh mesh");
    let mesh = data.mesh;
    let metadata = data.metadata;
    let hot = metadata.boundary_facets("hot");
    let cold = metadata.boundary_facets("cold");
    let adiabatic_bottom = metadata.boundary_facets("adiabatic_bottom");
    let adiabatic_top = metadata.boundary_facets("adiabatic_top");
    let all_walls = hot
        .iter()
        .chain(&cold)
        .chain(&adiabatic_bottom)
        .chain(&adiabatic_top)
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(all_walls.len(), 4 * CELLS);

    let zero_walls: Vec<_> = all_walls
        .iter()
        .copied()
        .map(|facet| (facet, 0.0))
        .collect();
    let hot_values: Vec<_> = hot.iter().copied().map(|facet| (facet, 1.0)).collect();
    let cold_values: Vec<_> = cold.iter().copied().map(|facet| (facet, 0.0)).collect();
    let u_values = dirichlet_values_with_precedence(&mesh, P, &zero_walls, &[]);
    let v_values = dirichlet_values_with_precedence(&mesh, P, &zero_walls, &[]);
    let temperature_values = dirichlet_values_with_precedence(&mesh, P, &hot_values, &cold_values);
    let problem = ormatex_sem_nd::SEM2DProblem::new_with_metadata(
        mesh,
        P,
        FieldRegistry::new(["u", "v", "p", "T"]),
        DofReduction2D::FieldSpecific {
            reductions: vec![
                DofReduction2D::DirichletValues { values: u_values },
                DofReduction2D::DirichletValues { values: v_values },
                DofReduction2D::None,
                DofReduction2D::DirichletValues {
                    values: temperature_values,
                },
            ],
        },
        metadata,
    );
    (problem, all_walls)
}

fn initial_state(problem: &ormatex_sem_nd::SEM2DProblem<ormatex_sem_nd::QuadMesh>) -> Mat<f64> {
    let mut state = Mat::<f64>::zeros(problem.system_size(), 1);
    let temperature = problem.field_id("T").unwrap();
    let offset = problem.field_offset(temperature);
    for row in 0..problem.field_reduced_size(temperature) {
        state[(offset + row, 0)] = 0.5;
    }
    state
}

fn field_map(values: FieldValues<(f64, f64)>) -> HashMap<(u64, u64), ((f64, f64), f64)> {
    values
        .positions
        .into_iter()
        .zip(values.values)
        .map(|(position, value)| {
            (
                (position.0.to_bits(), position.1.to_bits()),
                (position, value),
            )
        })
        .collect()
}

fn value_at(map: &HashMap<(u64, u64), ((f64, f64), f64)>, x: f64, y: f64) -> f64 {
    map.iter()
        .find(|((xb, yb), _)| {
            (f64::from_bits(*xb) - x).abs() < 1e-12 && (f64::from_bits(*yb) - y).abs() < 1e-12
        })
        .map(|(_, (_, value))| *value)
        .expect("requested diagnostic point is not a GLL node")
}

fn write_output(path: &str, fields: [(&str, FieldValues<(f64, f64)>); 4]) -> (f64, f64, f64, f64) {
    let maps = fields.clone().map(|(_, values)| field_map(values));
    let mut rows = HashMap::<(u64, u64), [f64; 6]>::new();
    for (field, values) in fields {
        let index = match field {
            "u" => 2,
            "v" => 3,
            "p" => 4,
            "T" => 5,
            _ => unreachable!(),
        };
        for (position, value) in values.positions.into_iter().zip(values.values) {
            rows.entry((position.0.to_bits(), position.1.to_bits()))
                .or_insert([
                    position.0,
                    position.1,
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                ])[index] = value;
        }
    }
    let mut output = BufWriter::new(File::create(path).expect("failed to create cavity output"));
    writeln!(output, "x,y,u,v,p,T").unwrap();
    for [x, y, u, v, p, temperature] in rows.values() {
        writeln!(
            output,
            "{x:.9},{y:.9},{u:.9e},{v:.9e},{p:.9e},{temperature:.9e}"
        )
        .unwrap();
    }

    let t_map = &maps[3];
    let u_map = &maps[0];
    let v_map = &maps[1];
    let mut u_centerline: f64 = 0.0;
    let mut v_centerline: f64 = 0.0;
    for ((_, _), ((x, _), value)) in u_map {
        if (x - 0.5).abs() < 1e-12 {
            u_centerline = u_centerline.max(value.abs());
        }
    }
    for ((_, _), ((_, y), value)) in v_map {
        if (y - 0.5).abs() < 1e-12 {
            v_centerline = v_centerline.max(value.abs());
        }
    }
    let center_temperature = value_at(t_map, 0.5, 0.5);
    let h = 1.0 / (2.0 * CELLS as f64);
    let mut wall_samples = Vec::new();
    for ((_, _), ((_, y), _)) in t_map
        .iter()
        .filter(|((xb, _), _)| (f64::from_bits(*xb) - h).abs() < 1e-12)
    {
        let derivative =
            (-3.0 + 4.0 * value_at(t_map, h, *y) - value_at(t_map, 2.0 * h, *y)) / (2.0 * h);
        wall_samples.push((*y, -derivative));
    }
    wall_samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    let hot_nusselt = wall_samples
        .windows(2)
        .map(|pair| 0.5 * (pair[0].1 + pair[1].1) * (pair[1].0 - pair[0].0))
        .sum();
    (hot_nusselt, center_temperature, u_centerline, v_centerline)
}

fn main() {
    let (problem, walls) = load_problem();
    let state0 = initial_state(&problem);
    let system =
        TensorFluidSystem::new_with_backend(&problem, tensor_kernel(), JacobianBackend::MatrixFree)
            .with_wall_boundaries(walls, Vec::new());
    let state = advance_tensor(&system, state0.as_ref(), DT, STEPS);
    std::fs::create_dir_all("target").expect("failed to create output directory");
    let diagnostics = write_output(
        "target/de_vahl_davis_cavity.csv",
        [
            ("u", problem.field_values("u", state.as_ref()).unwrap()),
            ("v", problem.field_values("v", state.as_ref()).unwrap()),
            ("p", problem.field_values("p", state.as_ref()).unwrap()),
            ("T", problem.field_values("T", state.as_ref()).unwrap()),
        ],
    );
    println!(
        "de Vahl Davis Ra={RA:.0}: Nu_hot={:.6} (reference ~1.118), T_center={:.6}, umax_mid={:.6}, vmax_mid={:.6}",
        diagnostics.0, diagnostics.1, diagnostics.2, diagnostics.3
    );
}
