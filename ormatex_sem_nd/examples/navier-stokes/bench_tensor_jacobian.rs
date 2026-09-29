//! Isolated matrix-free tensor Jacobian benchmark (no time integrator).
//!
//! Measures the 2D tensor (`TensorResidualKernel`) Jacobian pathway on its own
//! so parallel scaling of the Jacobian itself can be separated from
//! Krylov/serial effects. Four metrics are timed with `std::time::Instant`
//! (10 warm-up reps, then min and median over `--reps` reps):
//!
//! 1. `prepared_apply` — `MatrixFreeMinvJacobian::apply` (the prepared fast
//!    path with the fused `-M^{-1}` epilogue; exactly what the Krylov solver
//!    calls ~30 times per step).
//! 2. `unprepared_apply` — `operator.apply_jacobian_into(state, dir, out)`.
//! 3. `residual` — `operator.residual(state)`.
//! 4. `prepare` — `MatrixFreeMinvJacobian::new` (linearization-cache cost).
//!
//! Cases: `--case cavity` (default; lid-driven cavity with the fused 6-kernel
//! non-split EDAC tensor kernel, same as `cavity_comp_tensor.rs`) and
//! `--case cylinder` (split EDAC tensor kernel plus no-slip/slip wall
//! `StateTensorBoundaryTerms` built exactly as `cylinder_tensor.rs` and
//! `with_wall_boundaries` do; the cylinder mesh bakes in degree 2, so
//! `--degree` is ignored there).
//!
//! How to run (default size, 1 and 8 threads):
//! ```sh
//! RAYON_NUM_THREADS=1 cargo run --release --example navier-stokes-bench-tensor-jacobian
//! RAYON_NUM_THREADS=8 cargo run --release --example navier-stokes-bench-tensor-jacobian -- --resolution 96 --degree 4
//! ```

use faer::matrix_free::LinOp;
use faer::prelude::*;
use faer::sparse::SparseColMatRef;
use ormatex_sem_nd::{
    fuse_tensor_kernels, BilinearOps, DofReduction1D, EdacNavierStokes1DConfig,
    EdacNavierStokes2DConfig, FieldRegistry, MatrixFreeMinvJacobian, QuadMesh, SEM1DProblem,
    SEM2DProblem, StateTensorBoundaryTerms, TensorKernelAdvDiff,
    TensorKernelEdacMomentumConvection2D, TensorKernelEdacMomentumConvectionSplit1D,
    TensorKernelEdacMomentumConvectionSplit2D, TensorKernelEdacNoSlipWall2D,
    TensorKernelEdacPressureAdvection2D, TensorKernelEdacPressureAdvectionSplit1D,
    TensorKernelEdacPressureAdvectionSplit2D, TensorKernelEdacPressureDiffusion1D,
    TensorKernelEdacPressureDiffusion2D, TensorKernelEdacPressureDivergence1D,
    TensorKernelEdacPressureDivergence2D, TensorKernelEdacPressureGradient1D,
    TensorKernelEdacPressureGradient2D, TensorKernelEdacSlipWall2D,
    TensorKernelEdacSplitBoundaryFlux2D, TensorKernelEdacViscousStress1D,
    TensorKernelEdacViscousStress2D, TensorKernelEnergyAdvectionDiffusion1D,
    TensorKernelVolumeSource, TensorResidualKernel, TensorResidualKernelSet,
    TensorResidualKernelSum,
};

#[path = "../support/cavity_setup.rs"]
mod cavity_setup;
#[path = "../support/cylinder_setup.rs"]
mod cylinder_setup;

/// Invert a lumped (diagonal) mass matrix into the `m_inv` diagonal.
///
/// # Arguments
/// * `mass` - lumped mass matrix (square, one nonzero per row).
///
/// # Returns
/// Per-row reciprocal of the mass diagonal.
fn lumped_inverse_mass(mass: SparseColMatRef<'_, usize, f64>) -> Vec<f64> {
    assert_eq!(mass.nrows(), mass.ncols(), "mass matrix must be square");
    assert_eq!(
        mass.compute_nnz(),
        mass.nrows(),
        "expected lumped diagonal mass matrix"
    );
    (0..mass.nrows())
        .map(|i| {
            let value = mass[(i, i)];
            assert!(value.abs() > 1e-30, "zero mass diagonal at {i}");
            1.0 / value
        })
        .collect()
}

/// SplitMix64 step for deterministic pseudo-randomness (no extra deps).
///
/// # Arguments
/// * `state` - in-place RNG state.
///
/// # Returns
/// Next `u64` draw.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Smooth nonzero deterministic vector: low-frequency sines plus a small
/// SplitMix64 perturbation, so the linearization state exercises the full
/// nonlinear kernel path without hitting exact zeros.
///
/// # Arguments
/// * `n` - vector length.
/// * `ncols` - column count.
/// * `seed` - RNG seed (state and direction use different seeds).
///
/// # Returns
/// `n × ncols` matrix with entries in roughly `[0.05, 0.55]`.
fn smooth_state(n: usize, ncols: usize, seed: u64) -> Mat<f64> {
    let mut rng = seed;
    Mat::from_fn(n, ncols, |i, c| {
        let draw = (splitmix64(&mut rng) >> 11) as f64 * (1.0 / ((1u64 << 53) as f64)) * 2.0 - 1.0;
        let base = 0.3
            + 0.2 * (0.37 * (i + 1) as f64 + 0.7 * c as f64).sin()
            + 0.1 * (0.11 * (i + 1) as f64).cos();
        base + 0.05 * draw
    })
}

/// Sort a copy of timing samples and return `(min, median)`.
///
/// # Arguments
/// * `samples` - per-rep timings in microseconds.
///
/// # Returns
/// Minimum and median of the samples.
fn min_median_us(samples: &[f64]) -> (f64, f64) {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let min = sorted[0];
    let n = sorted.len();
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    };
    (min, median)
}

/// Print one benchmark line for a metric.
///
/// # Arguments
/// * `metric` - metric name.
/// * `samples` - per-rep timings in microseconds.
/// * `dofs` - system size.
/// * `extra` - trailing `key=value` fields (case, cells, ...).
fn report(metric: &str, samples: &[f64], dofs: usize, extra: &str) {
    let (min, median) = min_median_us(samples);
    println!(
        "metric={metric} dofs={dofs} threads={} min_us={min:.3} median_us={median:.3} {extra}",
        rayon::current_num_threads(),
    );
}

/// Time the four Jacobian/residual metrics for one problem/kernel/terms triple.
///
/// Builds the tensor residual operator with boundary terms (as the fluid
/// systems do), then times the prepared `LinOp::apply`, the unprepared
/// `apply_jacobian_into`, the residual, and the prepare cost.
///
/// # Arguments
/// * `problem` - assembled 2D SEM problem.
/// * `kernel` - tensor residual kernel.
/// * `terms` - tensor state-boundary terms (empty for the cavity).
/// * `m_inv` - inverse lumped-mass diagonal.
/// * `state` - linearization state (`N×1`).
/// * `direction` - Jacobian directions (`N×ncols`).
/// * `reps` - timed repetitions per metric (after 10 warm-ups).
/// * `extra` - trailing `key=value` fields for [`report`].
fn run_bench<K>(
    problem: &SEM2DProblem<QuadMesh>,
    kernel: &K,
    terms: &StateTensorBoundaryTerms<2>,
    m_inv: &[f64],
    state: &Mat<f64>,
    direction: &Mat<f64>,
    reps: usize,
    extra: &str,
) where
    K: TensorResidualKernel<2> + Sync,
{
    let n = problem.system_size();
    let ncols = direction.ncols();
    let build_operator = || {
        problem
            .tensor_residual_operator(kernel)
            .at_time(0.0)
            .with_state_boundary(terms)
    };
    // Prepared operator (as the Krylov solver holds it); the unprepared
    // operator below is a second instance for the direct-action metric.
    let minv = MatrixFreeMinvJacobian::new(build_operator(), state.clone(), m_inv);
    let plain = build_operator();
    let mut out = Mat::<f64>::zeros(n, ncols);
    let mut scratch = faer::dyn_stack::MemBuffer::new(minv.apply_scratch(ncols, faer::Par::Seq));

    // Warm-up: populate the linearization cache, E-vector pool, thread pool.
    for _ in 0..10 {
        minv.apply(
            out.as_mut(),
            direction.as_ref(),
            faer::Par::Seq,
            faer::dyn_stack::MemStack::new(&mut scratch),
        );
        std::hint::black_box(out.as_ref());
        plain.apply_jacobian_into(state.as_ref(), direction.as_ref(), out.as_mut());
        std::hint::black_box(out.as_ref());
        std::hint::black_box(plain.residual(state.as_ref()));
    }

    let mut samples = vec![0.0; reps];
    for i in 0..reps {
        let start = std::time::Instant::now();
        minv.apply(
            out.as_mut(),
            direction.as_ref(),
            faer::Par::Seq,
            faer::dyn_stack::MemStack::new(&mut scratch),
        );
        samples[i] = start.elapsed().as_secs_f64() * 1e6;
        std::hint::black_box(out.as_ref());
    }
    report("prepared_apply", &samples, n, extra);

    for i in 0..reps {
        let start = std::time::Instant::now();
        plain.apply_jacobian_into(state.as_ref(), direction.as_ref(), out.as_mut());
        samples[i] = start.elapsed().as_secs_f64() * 1e6;
        std::hint::black_box(out.as_ref());
    }
    report("unprepared_apply", &samples, n, extra);

    for i in 0..reps {
        let start = std::time::Instant::now();
        let residual = plain.residual(state.as_ref());
        samples[i] = start.elapsed().as_secs_f64() * 1e6;
        std::hint::black_box(residual);
    }
    report("residual", &samples, n, extra);

    for _ in 0..2 {
        std::hint::black_box(MatrixFreeMinvJacobian::new(
            build_operator(),
            state.clone(),
            m_inv,
        ));
    }
    for i in 0..reps {
        let start = std::time::Instant::now();
        std::hint::black_box(MatrixFreeMinvJacobian::new(
            build_operator(),
            state.clone(),
            m_inv,
        ));
        samples[i] = start.elapsed().as_secs_f64() * 1e6;
    }
    report("prepare", &samples, n, extra);
}

/// Fused 6-kernel non-split EDAC tensor kernel (same as `cavity_comp_tensor.rs`).
///
/// # Returns
/// Composed tensor residual kernel with `EdacNavierStokes2DConfig::new(1.0, 0.1, 10.0, 0.0)`.
fn cavity_kernel() -> impl TensorResidualKernel<2> {
    let config = EdacNavierStokes2DConfig::new(1.0, 0.1, 10.0, 0.0);
    fuse_tensor_kernels!(
        TensorKernelEdacMomentumConvection2D::new(config),
        TensorKernelEdacPressureGradient2D::new(config),
        TensorKernelEdacViscousStress2D::new(config),
        TensorKernelEdacPressureDivergence2D::new(config),
        TensorKernelEdacPressureAdvection2D::new(config),
        TensorKernelEdacPressureDiffusion2D::new(config),
    )
}

/// Split-form 6-kernel EDAC tensor kernel (same as `cylinder_tensor.rs`).
///
/// # Returns
/// Composed split tensor residual kernel with `EdacNavierStokes2DConfig::new(1.0, 1.0/200.0, 4.0, 0.1)`.
fn cylinder_kernel() -> impl TensorResidualKernel<2> {
    let config = EdacNavierStokes2DConfig::new(1.0, 1.0 / 200.0, 4.0, 0.1);
    TensorResidualKernelSum::from_kernel(TensorKernelEdacMomentumConvectionSplit2D::new(config))
        .with(TensorKernelEdacPressureGradient2D::new(config))
        .with(TensorKernelEdacViscousStress2D::new(config))
        .with(TensorKernelEdacPressureDivergence2D::new(config))
        .with(TensorKernelEdacPressureAdvectionSplit2D::new(config))
        .with(TensorKernelEdacPressureDiffusion2D::new(config))
}

/// Parse a `--name value` flag.
///
/// # Arguments
/// * `args` - command-line arguments.
/// * `name` - flag name.
///
/// # Returns
/// Parsed value, or `None` when the flag is absent.
fn parse_usize_flag(args: &[String], name: &str) -> Option<usize> {
    args.windows(2)
        .find(|window| window[0] == name)
        .map(|window| {
            window[1]
                .parse()
                .unwrap_or_else(|e| panic!("invalid value for {name}: {:?} ({e})", window[1]))
        })
}

/// Parse a `--name value` string flag.
///
/// # Arguments
/// * `args` - command-line arguments.
/// * `name` - flag name.
///
/// # Returns
/// Flag value, or `None` when the flag is absent.
fn parse_string_flag(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|window| window[0] == name)
        .map(|window| window[1].clone())
}

/// Time the four Jacobian/residual metrics for one 1D problem/kernel pair.
///
/// Same four metrics as [`run_bench`] but for a 1D tensor operator without
/// boundary terms (Dirichlet conditions enter through the DOF reduction).
///
/// # Arguments
/// * `problem` - assembled 1D SEM problem.
/// * `kernel` - 1D tensor residual kernel.
/// * `m_inv` - inverse lumped-mass diagonal.
/// * `state` - linearization state (`N×1`).
/// * `direction` - Jacobian directions (`N×ncols`).
/// * `reps` - timed repetitions per metric (after 10 warm-ups).
/// * `extra` - trailing `key=value` fields for [`report`].
fn run_bench_1d<K, M>(
    problem: &SEM1DProblem<M>,
    kernel: &K,
    m_inv: &[f64],
    state: &Mat<f64>,
    direction: &Mat<f64>,
    reps: usize,
    extra: &str,
) where
    K: TensorResidualKernel<1> + Sync,
    M: ndmesh::traits::Mesh<EntityDescriptor = ndelement::types::ReferenceCellType, T = f64> + Sync,
{
    let n = problem.system_size();
    let ncols = direction.ncols();
    let build_operator = || problem.tensor_residual_operator(kernel).at_time(0.0);
    let minv = MatrixFreeMinvJacobian::new(build_operator(), state.clone(), m_inv);
    let plain = build_operator();
    let mut out = Mat::<f64>::zeros(n, ncols);
    let mut scratch = faer::dyn_stack::MemBuffer::new(minv.apply_scratch(ncols, faer::Par::Seq));

    for _ in 0..10 {
        minv.apply(
            out.as_mut(),
            direction.as_ref(),
            faer::Par::Seq,
            faer::dyn_stack::MemStack::new(&mut scratch),
        );
        std::hint::black_box(out.as_ref());
        plain.apply_jacobian_into(state.as_ref(), direction.as_ref(), out.as_mut());
        std::hint::black_box(out.as_ref());
        std::hint::black_box(plain.residual(state.as_ref()));
    }

    let mut samples = vec![0.0; reps];
    for i in 0..reps {
        let start = std::time::Instant::now();
        minv.apply(
            out.as_mut(),
            direction.as_ref(),
            faer::Par::Seq,
            faer::dyn_stack::MemStack::new(&mut scratch),
        );
        samples[i] = start.elapsed().as_secs_f64() * 1e6;
        std::hint::black_box(out.as_ref());
    }
    report("prepared_apply", &samples, n, extra);

    for i in 0..reps {
        let start = std::time::Instant::now();
        plain.apply_jacobian_into(state.as_ref(), direction.as_ref(), out.as_mut());
        samples[i] = start.elapsed().as_secs_f64() * 1e6;
        std::hint::black_box(out.as_ref());
    }
    report("unprepared_apply", &samples, n, extra);

    for i in 0..reps {
        let start = std::time::Instant::now();
        let residual = plain.residual(state.as_ref());
        samples[i] = start.elapsed().as_secs_f64() * 1e6;
        std::hint::black_box(residual);
    }
    report("residual", &samples, n, extra);

    for _ in 0..2 {
        std::hint::black_box(MatrixFreeMinvJacobian::new(
            build_operator(),
            state.clone(),
            m_inv,
        ));
    }
    for i in 0..reps {
        let start = std::time::Instant::now();
        std::hint::black_box(MatrixFreeMinvJacobian::new(
            build_operator(),
            state.clone(),
            m_inv,
        ));
        samples[i] = start.elapsed().as_secs_f64() * 1e6;
    }
    report("prepare", &samples, n, extra);
}

/// 1D heated-pipe kernel: EDAC split sum plus energy advection-diffusion.
///
/// # Arguments
/// * `config` - 1D EDAC configuration.
/// * `alpha` - thermal diffusivity for the energy kernel.
/// * `heat` - uniform volumetric source for the temperature field.
///
/// # Returns
/// Composed 1D tensor kernel over `[u, p, T]`, as in `ex_nd_1d_edac_heated_pipe`.
fn heated_pipe_kernel_1d(
    config: EdacNavierStokes1DConfig,
    alpha: f64,
    heat: f64,
) -> TensorResidualKernelSet<'static, 1> {
    let edac = TensorResidualKernelSum::from_kernel(
        TensorKernelEdacMomentumConvectionSplit1D::new(config),
    )
    .with(TensorKernelEdacPressureGradient1D::new(config))
    .with(TensorKernelEdacViscousStress1D::new(config))
    .with(TensorKernelEdacPressureDivergence1D::new(config))
    .with(TensorKernelEdacPressureAdvectionSplit1D::new(config))
    .with(TensorKernelEdacPressureDiffusion1D::new(config));
    TensorResidualKernelSet::from_kernel(edac)
        .with(TensorKernelEnergyAdvectionDiffusion1D::new(alpha))
        .with(TensorKernelVolumeSource::with_field_names(heat, ["T"]))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // Warn on unrecognized `--flags` (values of known flags are skipped).
    const KNOWN_FLAGS: &[&str] = &[
        "--resolution",
        "--degree",
        "--case",
        "--reps",
        "--ncols",
        "--dim",
        "--cells",
    ];
    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        if arg.starts_with("--") {
            if KNOWN_FLAGS.contains(&arg.as_str()) {
                // Skip the flag's value on the next iteration.
                i += 1;
            } else {
                eprintln!("warning: unknown flag {arg:?} (ignored)");
            }
        }
        i += 1;
    }
    let dim = parse_usize_flag(&args, "--dim").unwrap_or(2);
    let resolution = parse_usize_flag(&args, "--resolution").unwrap_or(32);
    let degree = parse_usize_flag(&args, "--degree").unwrap_or(2);
    let case = parse_string_flag(&args, "--case").unwrap_or_else(|| "cavity".to_string());
    let reps = parse_usize_flag(&args, "--reps").unwrap_or(200);
    let ncols = parse_usize_flag(&args, "--ncols").unwrap_or(1);
    assert!(resolution > 0, "--resolution must be positive");
    assert!(degree >= 1, "--degree must be >= 1");
    assert!(reps > 0, "--reps must be positive");
    assert!(ncols > 0, "--ncols must be positive");

    if dim == 1 {
        let cells = parse_usize_flag(&args, "--cells").unwrap_or(4096);
        assert!(cells > 0, "--cells must be positive");
        // Default 1D case mirrors the flag default story: heated-pipe.
        let case_1d = if case == "cavity" {
            "heated-pipe".to_string()
        } else {
            case.clone()
        };
        match case_1d.as_str() {
            "heated-pipe" => {
                let problem = SEM1DProblem::new(
                    ndmesh::shapes::unit_interval(cells, 1),
                    degree,
                    FieldRegistry::new(["u", "p", "T"]),
                    DofReduction1D::FieldSpecific {
                        reductions: vec![
                            DofReduction1D::Dirichlet {
                                facets: vec![(0, 0.1)],
                            },
                            DofReduction1D::Dirichlet {
                                facets: vec![(cells, 0.0)],
                            },
                            DofReduction1D::Dirichlet {
                                facets: vec![(0, 0.0)],
                            },
                        ],
                    },
                );
                let config = EdacNavierStokes1DConfig::new(1.0, 0.01, 10.0);
                let kernel = heated_pipe_kernel_1d(config, 0.01, 0.5);
                let mass = problem.assemble_lumped_mass();
                let m_inv = lumped_inverse_mass(mass.as_ref());
                let n = problem.system_size();
                let state = smooth_state(n, 1, 0xC0FFEE);
                let direction = smooth_state(n, ncols, 0xDEC0DE);
                let extra = format!(
                    "case=heated-pipe dim=1 cells={} degree={degree} ncols={ncols} reps={reps}",
                    problem.cell_count()
                );
                run_bench_1d(&problem, &kernel, &m_inv, &state, &direction, reps, &extra);
            }
            "advdiff" => {
                let problem = SEM1DProblem::new(
                    ndmesh::shapes::unit_interval(cells, 1),
                    degree,
                    FieldRegistry::new(["c"]),
                    DofReduction1D::Periodic { facets: [0, cells] },
                );
                let kernel = TensorKernelAdvDiff::new(0.01, 1.0);
                let mass = problem.assemble_lumped_mass();
                let m_inv = lumped_inverse_mass(mass.as_ref());
                let n = problem.system_size();
                let state = smooth_state(n, 1, 0xC0FFEE);
                let direction = smooth_state(n, ncols, 0xDEC0DE);
                let extra = format!(
                    "case=advdiff dim=1 cells={} degree={degree} ncols={ncols} reps={reps}",
                    problem.cell_count()
                );
                run_bench_1d(&problem, &kernel, &m_inv, &state, &direction, reps, &extra);
            }
            other => panic!("unknown --case {other:?} for --dim 1 (expected heated-pipe|advdiff)"),
        }
        return;
    }

    match case.as_str() {
        "cavity" => {
            let problem =
                cavity_setup::problem_with_resolution_and_degree(resolution, resolution, degree);
            let kernel = cavity_kernel();
            // Empty boundary terms, exactly as the cavity fluid system holds.
            let terms = StateTensorBoundaryTerms::new();
            let mass = problem.assemble_lumped_mass();
            let m_inv = lumped_inverse_mass(mass.as_ref());
            let n = problem.system_size();
            let state = smooth_state(n, 1, 0xC0FFEE);
            let direction = smooth_state(n, ncols, 0xDEC0DE);
            let extra = format!(
                "case=cavity cells={} degree={degree} ncols={ncols} reps={reps}",
                problem.cell_count()
            );
            run_bench(
                &problem, &kernel, &terms, &m_inv, &state, &direction, reps, &extra,
            );
        }
        "cylinder" => {
            let case = cylinder_setup::problem(false);
            let problem = case.problem;
            let kernel = cylinder_kernel();
            // Wall boundary terms, exactly as `with_wall_boundaries` builds.
            let terms = StateTensorBoundaryTerms::new()
                .with_default(TensorKernelEdacSplitBoundaryFlux2D)
                .with_entities(case.cylinder, TensorKernelEdacNoSlipWall2D)
                .with_entities(case.slip_wall, TensorKernelEdacSlipWall2D);
            let mass = problem.assemble_lumped_mass();
            let m_inv = lumped_inverse_mass(mass.as_ref());
            let n = problem.system_size();
            let state = smooth_state(n, 1, 0xC0FFEE);
            let direction = smooth_state(n, ncols, 0xDEC0DE);
            let extra = format!(
                "case=cylinder cells={} degree=2 ncols={ncols} reps={reps}",
                problem.cell_count()
            );
            run_bench(
                &problem, &kernel, &terms, &m_inv, &state, &direction, reps, &extra,
            );
        }
        other => panic!("unknown --case {other:?} (expected cavity|cylinder)"),
    }
}
