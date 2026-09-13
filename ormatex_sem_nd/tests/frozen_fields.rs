use faer::matrix_free::LinOp;
use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};
use ndelement::types::ReferenceCellType;
use ndmesh::{
    shapes::{unit_interval, unit_square},
    traits::{Entity, Geometry, Mesh, Point, Topology},
    SingleElementMesh,
};
use ormatex::matexp_krylov::KrylovExpm;
use ormatex::matexp_pade::PadeExpm;
use ormatex::ode_epirk::EpirkIntegrator;
use ormatex::ode_implicit::DirkIntegrator;
use ormatex::ode_sys::{IntegrateSys, OdeSys};
use ormatex::tableau_implicit::ImplicitBT;
use ormatex_sem_nd::{
    BilinearOps, CellMeta, CellState, Coefficient, ConstantCoefficient, DofReduction1D,
    DofReduction2D, FacetCtx, FacetMeta, FieldRegistry, FrozenFacetField, FrozenQuadratureField,
    FrozenVelocity2D, KernelAdvDiff, KernelAdvDiff2D, KernelAdvectionOutflow1D,
    MaterialContext, MaterialProperty, MatrixFreeMinvJacobian, StateBoundaryIntegrator,
    StateBoundaryTerms, StateTensorBoundaryTerms, StateTensorBoundaryIntegrator,
    TensorFacetCtx, TensorKernelAdvDiff, TensorKernelAdvDiff2D,
    TensorKernelAdvectionOutflow2D, TensorKernelLinearReaction, TensorResidualKernelSet,
    SEM1DProblem, SEM2DProblem,
};

fn ctx_at(cell: usize, _q: usize) -> (Vec<f64>, CellMeta) {
    (vec![0.0], CellMeta { local_index: cell, physical_region: None })
}

#[test]
fn frozen_field_evals_cell_major_and_stays_frozen() {
    let frozen = FrozenQuadratureField::new(2, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(frozen.npts(), 2);
    assert_eq!(frozen.cell_count(), 2);
    frozen.assert_compatible(2, 2);
    let (point, cell_meta) = ctx_at(1, 0);
    let ctx = MaterialContext { time: 0.0, point: &point, cell: cell_meta, state: None, q: 0 };
    assert_eq!(frozen.eval(&ctx), 3.0);
    assert!(frozen.derivative(&ctx, 0).is_none());
}

#[test]
fn frozen_velocity_pair_requires_matching_layout() {
    let x = FrozenQuadratureField::new(3, vec![0.0; 6]);
    let y = FrozenQuadratureField::new(3, vec![0.0; 6]);
    let pair = FrozenVelocity2D::new(x.clone(), y.clone());
    assert_eq!(pair.x.npts(), 3);
    let _ = pair.y.cell_count();
}

#[test]
fn sampling_constant_state_is_constant_1d() {
    let problem = SEM1DProblem::new(
        unit_interval(2, 1),
        2,
        FieldRegistry::new(["u"]),
        DofReduction1D::None,
    );
    let state = Mat::from_fn(problem.system_size(), 1, |_, _| 2.0);
    let frozen = problem.sample_quadrature_field(state.as_ref(), "u");
    assert_eq!(frozen.npts(), problem.quadrature_points_per_cell());
    assert_eq!(frozen.cell_count(), problem.cell_count());
    for cell in 0..problem.cell_count() {
        for q in 0..problem.quadrature_points_per_cell() {
            let (point, cell_meta) = ctx_at(cell, q);
            let ctx = MaterialContext {
                time: 0.0,
                point: &point,
                cell: cell_meta,
                state: None,
                q,
            };
            assert!((frozen.eval(&ctx) - 2.0).abs() < 1e-14);
        }
    }
}

#[test]
fn sampling_constant_state_is_constant_2d() {
    let problem = SEM2DProblem::new(
        unit_square(1, 1, ReferenceCellType::Quadrilateral, 1),
        2,
        FieldRegistry::new(["u"]),
        DofReduction2D::None,
    );
    let state = Mat::from_fn(problem.system_size(), 1, |_, _| -1.5);
    let frozen = problem.sample_quadrature_field(state.as_ref(), "u");
    assert_eq!(frozen.npts(), problem.quadrature_points_per_cell());
    for cell in 0..problem.cell_count() {
        for q in 0..problem.quadrature_points_per_cell() {
            let (point, cell_meta) = ctx_at(cell, q);
            let point2 = vec![point[0], 0.0];
            let ctx = MaterialContext {
                time: 0.0,
                point: &point2,
                cell: cell_meta,
                state: None,
                q,
            };
            assert!((frozen.eval(&ctx) + 1.5).abs() < 1e-14);
        }
    }
}

#[test]
fn sampling_includes_prescribed_inlet_value_1d() {
    let problem = SEM1DProblem::new(
        unit_interval(1, 1),
        2,
        FieldRegistry::new(["c0"]),
        DofReduction1D::Dirichlet { facets: vec![(0, 1.0)] },
    );
    let state = Mat::<f64>::zeros(problem.system_size(), 1);
    let frozen = problem.sample_quadrature_field(state.as_ref(), "c0");
    // Collocated GLL: some quadrature node sits on the prescribed endpoint.
    let mut values = Vec::new();
    for q in 0..problem.quadrature_points_per_cell() {
        let (point, cell_meta) = ctx_at(0, q);
        let ctx = MaterialContext {
            time: 0.0,
            point: &point,
            cell: cell_meta,
            state: None,
            q,
        };
        values.push(frozen.eval(&ctx));
    }
    assert!(values.iter().any(|&v| (v - 1.0).abs() < 1e-14));
    assert!(values.iter().any(|&v| v.abs() < 1e-14));
}

#[test]
fn frozen_uniform_matches_constant_bilinear_1d_and_2d() {
    let problem_1d = SEM1DProblem::new(
        unit_interval(2, 1),
        2,
        FieldRegistry::new(["u"]),
        DofReduction1D::None,
    );
    let npts = problem_1d.quadrature_points_per_cell();
    let ncells = problem_1d.cell_count();
    let frozen = FrozenQuadratureField::new(npts, vec![0.5; ncells * npts]);
    let constant = KernelAdvDiff::with_coefficients(
        ConstantCoefficient(0.002),
        ConstantCoefficient(0.5),
    );
    let with_frozen = KernelAdvDiff::with_coefficients(
        ConstantCoefficient(0.002),
        frozen,
    );
    let a = problem_1d.assemble_bilinear(0.0, &constant).to_dense();
    let b = problem_1d.assemble_bilinear(0.0, &with_frozen).to_dense();
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            assert!((a[(i, j)] - b[(i, j)]).abs() < 1e-14);
        }
    }

    let problem_2d = SEM2DProblem::new(
        unit_square(1, 1, ReferenceCellType::Quadrilateral, 1),
        2,
        FieldRegistry::new(["u"]),
        DofReduction2D::None,
    );
    let npts_2d = problem_2d.quadrature_points_per_cell();
    let ncells_2d = problem_2d.cell_count();
    let fx = FrozenQuadratureField::new(npts_2d, vec![0.25; ncells_2d * npts_2d]);
    let fy = FrozenQuadratureField::new(npts_2d, vec![-0.5; ncells_2d * npts_2d]);
    let constant_2d = KernelAdvDiff2D::new(0.002, [0.25, -0.5]);
    let with_frozen_2d =
        KernelAdvDiff2D::with_coefficients(ConstantCoefficient(0.002), [fx, fy]);
    let a2 = problem_2d.assemble_bilinear(0.0, &constant_2d).to_dense();
    let b2 = problem_2d.assemble_bilinear(0.0, &with_frozen_2d).to_dense();
    for i in 0..a2.nrows() {
        for j in 0..a2.ncols() {
            assert!((a2[(i, j)] - b2[(i, j)]).abs() < 1e-14);
        }
    }
}

fn rates() -> SparseColMat<usize, f64> {
    let entries = [
        (0, 0, -0.1),
        (1, 0, 0.1),
        (1, 1, -10.0),
        (2, 1, 10.0),
        (2, 2, -0.01),
    ];
    let triplets: Vec<_> = entries
        .iter()
        .map(|&(r, c, v)| Triplet::new(r, c, v))
        .collect();
    SparseColMat::try_new_from_triplets(3, 3, &triplets).unwrap()
}

#[test]
fn three_species_tensor_set_assembles_and_refreshes_1d() {
    let problem = SEM1DProblem::new(
        unit_interval(2, 1),
        2,
        FieldRegistry::new(["c0", "c1", "c2"]),
        DofReduction1D::FieldSpecific {
            reductions: vec![
                DofReduction1D::Dirichlet { facets: vec![(0, 1.0)] },
                DofReduction1D::Dirichlet { facets: vec![(0, 0.0)] },
                DofReduction1D::Dirichlet { facets: vec![(0, 0.0)] },
            ],
        },
    );
    let npts = problem.quadrature_points_per_cell();
    let ncells = problem.cell_count();
    let slow = FrozenQuadratureField::new(npts, vec![0.1; ncells * npts]);
    let fast = FrozenQuadratureField::new(npts, vec![0.9; ncells * npts]);
    let build = |frozen: FrozenQuadratureField| {
        TensorResidualKernelSet::from_kernel(
            TensorKernelAdvDiff::with_coefficients(
                ConstantCoefficient(0.002),
                frozen.clone(),
            )
            .with_field_name("c0"),
        )
        .with(
            TensorKernelAdvDiff::with_coefficients(
                ConstantCoefficient(0.002),
                frozen.clone(),
            )
            .with_field_name("c1"),
        )
        .with(
            TensorKernelAdvDiff::with_coefficients(ConstantCoefficient(0.002), frozen)
                .with_field_name("c2"),
        )
        .with(TensorKernelLinearReaction::with_field_names(
            rates(),
            ["c0", "c1", "c2"],
        ))
    };
    let state = Mat::<f64>::zeros(problem.system_size(), 1);
    let r_slow = problem
        .tensor_residual_operator(&build(slow))
        .at_time(0.0)
        .residual(state.as_ref());
    let r_fast = problem
        .tensor_residual_operator(&build(fast))
        .at_time(0.0)
        .residual(state.as_ref());
    assert_eq!(r_slow.len(), problem.system_size());
    // Inlet Dirichlet pins c0=1, so advection refresh must move the residual.
    assert!(r_slow
        .iter()
        .zip(r_fast.iter())
        .any(|(a, b)| (a - b).abs() > 1e-12));
    let jac = problem
        .tensor_residual_operator(&build(FrozenQuadratureField::new(
            npts,
            vec![0.1; ncells * npts],
        )))
        .at_time(0.0)
        .assemble_jacobian(state.as_ref());
    assert_eq!(jac.nrows(), problem.system_size());
}

#[test]
fn three_species_tensor_set_assembles_2d() {
    let problem = SEM2DProblem::new(
        unit_square(1, 1, ReferenceCellType::Quadrilateral, 1),
        2,
        FieldRegistry::new(["c0", "c1", "c2"]),
        DofReduction2D::None,
    );
    let npts = problem.quadrature_points_per_cell();
    let ncells = problem.cell_count();
    let fx = FrozenQuadratureField::new(npts, vec![0.5; ncells * npts]);
    let fy = FrozenQuadratureField::new(npts, vec![0.0; ncells * npts]);
    let kernel = TensorResidualKernelSet::from_kernel(
        TensorKernelAdvDiff2D::with_coefficients(
            ConstantCoefficient(0.01),
            [fx.clone(), fy.clone()],
        )
        .with_field_name("c0"),
    )
    .with(
        TensorKernelAdvDiff2D::with_coefficients(
            ConstantCoefficient(0.01),
            [fx.clone(), fy.clone()],
        )
        .with_field_name("c1"),
    )
    .with(
        TensorKernelAdvDiff2D::with_coefficients(
            ConstantCoefficient(0.01),
            [fx, fy],
        )
        .with_field_name("c2"),
    )
    .with(TensorKernelLinearReaction::with_field_names(
        rates(),
        ["c0", "c1", "c2"],
    ));
    let state = Mat::<f64>::zeros(problem.system_size(), 1);
    let residual = problem
        .tensor_residual_operator(&kernel)
        .at_time(0.0)
        .residual(state.as_ref());
    assert_eq!(residual.len(), problem.system_size());
}

type QuadMesh = SingleElementMesh<f64, ndelement::ciarlet::CiarletElement<f64, ndelement::map::IdentityMap, f64>>;

/// Boundary facet indices of `mesh` whose midpoint sits on `x == target`.
fn facets_at_x(mesh: &QuadMesh, target: f64) -> Vec<usize> {
    const EPS: f64 = 1e-12;
    let mut out = Vec::new();
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
        if (midpoint[0] - target).abs() < EPS {
            out.push(facet.local_index());
        }
    }
    out
}

#[test]
fn facet_sampling_constant_state_is_constant() {
    let problem = SEM2DProblem::new(
        unit_square(2, 1, ReferenceCellType::Quadrilateral, 1),
        2,
        FieldRegistry::new(["u"]),
        DofReduction2D::None,
    );
    let state = Mat::from_fn(problem.system_size(), 1, |_, _| 1.25);
    let frozen = problem.sample_facet_quadrature_field(state.as_ref(), "u");
    assert_eq!(frozen.npts(), problem.facet_quadrature_points());
    assert_eq!(frozen.facet_count(), problem.facet_count());
    // Boundary facets read the constant; interior facets stay zero (never queried).
    let mut hi = f64::NEG_INFINITY;
    let mut lo = f64::INFINITY;
    let mut touched = 0;
    for facet in 0..problem.facet_count() {
        for q in 0..problem.facet_quadrature_points() {
            let v = frozen.value(facet, q);
            hi = hi.max(v);
            lo = lo.min(v);
            if v.abs() > 1e-14 {
                assert!((v - 1.25).abs() < 1e-14);
                touched += 1;
            }
        }
    }
    assert!((hi - 1.25).abs() < 1e-14);
    assert!(lo.abs() < 1e-14);
    assert!(touched > 0);
}

#[test]
fn facet_sampling_includes_prescribed_inlet_value() {
    let mesh = unit_square(1, 1, ReferenceCellType::Quadrilateral, 1);
    let inlet = facets_at_x(&mesh, 0.0);
    assert!(!inlet.is_empty());
    let problem = SEM2DProblem::new(
        mesh,
        2,
        FieldRegistry::new(["c0"]),
        DofReduction2D::Dirichlet {
            facets: inlet.into_iter().map(|f| (f, 1.0)).collect(),
        },
    );
    let state = Mat::<f64>::zeros(problem.system_size(), 1);
    let frozen = problem.sample_facet_quadrature_field(state.as_ref(), "c0");
    let mut hi = f64::NEG_INFINITY;
    let mut lo = f64::INFINITY;
    for facet in 0..problem.facet_count() {
        for q in 0..problem.facet_quadrature_points() {
            hi = hi.max(frozen.value(facet, q));
            lo = lo.min(frozen.value(facet, q));
        }
    }
    assert!((hi - 1.0).abs() < 1e-14);
    assert!(lo.abs() < 1e-14);
}

#[test]
fn advective_outflow_passes_outflow_and_blocks_backflow() {
    // Facet 5 sees velocity (2, 0): outflow for normal +x, backflow for -x.
    let npts = 2;
    let nfacets = 8;
    let mut vx = vec![0.0; nfacets * npts];
    vx[5 * npts] = 2.0;
    vx[5 * npts + 1] = 2.0;
    let ux = FrozenFacetField::new(npts, nfacets, vx);
    let uy = FrozenFacetField::new(npts, nfacets, vec![0.0; nfacets * npts]);
    let kernel = TensorKernelAdvectionOutflow2D::with_field_names(ux, uy, ["c0", "c1", "c2"]);
    let wts = vec![1.0; npts];
    let jdet = vec![1.0; npts];
    let points = vec![0.0; 2 * npts];
    let values = vec![0.3, 0.7, 0.0, 0.0, 0.0, 0.0];
    let grads = vec![0.0; 12];
    let state = CellState {
        nfields: 3,
        npts,
        gdim: 2,
        values: &values,
        grads: &grads,
        field_indices: &[],
    };
    let outflow_ctx = TensorFacetCtx {
        time: 0.0,
        facet: FacetMeta { local_index: 5, physical_region: None },
        npts,
        wts: &wts,
        jfacet_det: &jdet,
        points: &points,
        normal: &[1.0, 0.0],
    };
    assert!((kernel.tensor_residual(&outflow_ctx, &state, 0, 0) - 2.0 * 0.3).abs() < 1e-14);
    assert!((kernel.tensor_residual(&outflow_ctx, &state, 0, 1) - 2.0 * 0.7).abs() < 1e-14);
    let direction = CellState {
        nfields: 3,
        npts,
        gdim: 2,
        values: &values,
        grads: &grads,
        field_indices: &[],
    };
    assert!(
        (kernel.tensor_jacobian_action(&outflow_ctx, &state, &direction, 0, 0) - 2.0 * 0.3).abs()
            < 1e-14
    );
    let backflow_ctx = TensorFacetCtx {
        time: 0.0,
        facet: FacetMeta { local_index: 5, physical_region: None },
        npts,
        wts: &wts,
        jfacet_det: &jdet,
        points: &points,
        normal: &[-1.0, 0.0],
    };
    // Zero backflow concentration: re-entering flow contributes nothing.
    assert_eq!(kernel.tensor_residual(&backflow_ctx, &state, 0, 0), 0.0);
    assert_eq!(
        kernel.tensor_jacobian_action(&backflow_ctx, &state, &direction, 0, 0),
        0.0
    );
}

/// Minimal species system mirroring the cylinder example (tensor + matrix-free).
struct TestSpeciesSystem<'a> {
    problem: &'a SEM2DProblem<QuadMesh>,
    kernel: TensorResidualKernelSet<'a, 2>,
    terms: StateTensorBoundaryTerms<2>,
    m_inv: Vec<f64>,
}

impl<'a> OdeSys<'a> for TestSpeciesSystem<'a> {
    fn frhs(&self, t: f64, state: MatRef<f64>) -> Mat<f64> {
        let residual = self
            .problem
            .tensor_residual_operator(&self.kernel)
            .at_time(t)
            .with_state_boundary(&self.terms)
            .residual(state);
        Mat::from_fn(self.m_inv.len(), 1, |row, _| -self.m_inv[row] * residual[row])
    }
    fn fjac<'b>(&'a self, t: f64, state: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
        let operator = self
            .problem
            .tensor_residual_operator(&self.kernel)
            .at_time(t)
            .with_state_boundary(&self.terms);
        Box::new(MatrixFreeMinvJacobian::new(
            operator,
            state.to_owned(),
            &self.m_inv,
        ))
    }
}

fn species_problem(mesh: QuadMesh, inlet: &[usize]) -> SEM2DProblem<QuadMesh> {
    SEM2DProblem::new(
        mesh,
        2,
        FieldRegistry::new(["c0", "c1", "c2"]),
        DofReduction2D::FieldSpecific {
            reductions: vec![
                DofReduction2D::Dirichlet {
                    facets: inlet.iter().copied().map(|f| (f, 1.0)).collect(),
                },
                DofReduction2D::Dirichlet {
                    facets: inlet.iter().copied().map(|f| (f, 0.0)).collect(),
                },
                DofReduction2D::Dirichlet {
                    facets: inlet.iter().copied().map(|f| (f, 0.0)).collect(),
                },
            ],
        },
    )
}

fn uniform_frozen(problem: &SEM2DProblem<QuadMesh>, vx: f64) -> (FrozenQuadratureField, FrozenQuadratureField, FrozenFacetField, FrozenFacetField) {
    let npts = problem.quadrature_points_per_cell();
    let ncells = problem.cell_count();
    let fnpts = problem.facet_quadrature_points();
    let nfacets = problem.facet_count();
    (
        FrozenQuadratureField::new(npts, vec![vx; ncells * npts]),
        FrozenQuadratureField::new(npts, vec![0.0; ncells * npts]),
        FrozenFacetField::new(fnpts, nfacets, vec![vx; nfacets * fnpts]),
        FrozenFacetField::new(fnpts, nfacets, vec![0.0; nfacets * fnpts]),
    )
}

fn species_kernel_for(
    fx: FrozenQuadratureField,
    fy: FrozenQuadratureField,
) -> TensorResidualKernelSet<'static, 2> {
    TensorResidualKernelSet::from_kernel(
        TensorKernelAdvDiff2D::with_coefficients(ConstantCoefficient(0.001), [fx.clone(), fy.clone()])
            .with_field_name("c0"),
    )
    .with(
        TensorKernelAdvDiff2D::with_coefficients(ConstantCoefficient(0.001), [fx.clone(), fy.clone()])
            .with_field_name("c1"),
    )
    .with(
        TensorKernelAdvDiff2D::with_coefficients(ConstantCoefficient(0.001), [fx, fy])
            .with_field_name("c2"),
    )
}

/// Step species with uniform rightward flow to `final_time`; return peak |c|.
fn run_species(outflow: bool) -> (bool, f64) {
    let probe_mesh = unit_square(8, 2, ReferenceCellType::Quadrilateral, 1);
    let inlet = facets_at_x(&probe_mesh, 0.0);
    let outlet = facets_at_x(&probe_mesh, 1.0);
    assert!(!inlet.is_empty() && !outlet.is_empty());
    let problem = species_problem(probe_mesh, &inlet);
    let (fx, fy, ffx, ffy) = uniform_frozen(&problem, 1.0);
    let terms = if outflow {
        let kernel =
            TensorKernelAdvectionOutflow2D::with_field_names(ffx, ffy, ["c0", "c1", "c2"]);
        StateTensorBoundaryTerms::new().with_entities(outlet, kernel)
    } else {
        StateTensorBoundaryTerms::new()
    };
    let mass = problem.assemble_lumped_mass();
    let m_inv: Vec<f64> = (0..mass.nrows()).map(|i| 1.0 / mass[(i, i)]).collect();
    let system = TestSpeciesSystem {
        problem: &problem,
        kernel: species_kernel_for(fx, fy),
        terms,
        m_inv,
    };
    let y0 = Mat::<f64>::zeros(problem.system_size(), 1);
    let expmv = Box::new(PadeExpm::new(12));
    let krylov = KrylovExpm::new(expmv, 30, 100, 1e-12, Some(2));
    let mut solver = EpirkIntegrator::new(0.0, y0.as_ref(), "epi3".to_string(), krylov);
    // Front crosses the unit domain at t ~= 1 with U = 1.
    for _ in 0..40 {
        match solver.step(&system, 0.05) {
            Ok(step) => solver.accept_step(step),
            Err(_) => return (false, f64::INFINITY),
        }
    }
    let state = solver.state();
    let mut peak = 0.0f64;
    for name in ["c0", "c1", "c2"] {
        let field = problem.field_values(name, state.as_ref()).unwrap();
        for &v in &field.values {
            peak = peak.max(v.abs());
        }
    }
    (true, peak)
}

#[test]
fn species_outflow_keeps_advected_front_bounded() {
    let (completed, peak) = run_species(true);
    assert!(completed, "stepper failed with outflow BC");
    assert!(peak <= 1.1, "outflow BC should let the front leave, peak was {peak}");
}

#[test]
fn missing_outflow_accumulates_at_outlet() {
    let (completed, peak) = run_species(false);
    // Without the outflow flux the outlet acts as a closed wall: either the
    // stepper fails or species pile up far beyond physical bounds.
    assert!(!completed || peak > 1.5, "expected blowup without outflow BC, peak was {peak}");
}

/// Weak/tensor boundary equivalence for the advective outflow term.
#[test]
fn advective_outflow_weak_matches_tensor() {
    use ormatex_sem_nd::{
        KernelAdvectionOutflow2D, ResidualKernelSet, StateBoundaryTerms,
    };
    let probe_mesh = unit_square(4, 1, ReferenceCellType::Quadrilateral, 1);
    let inlet = facets_at_x(&probe_mesh, 0.0);
    let outlet = facets_at_x(&probe_mesh, 1.0);
    assert!(!inlet.is_empty() && !outlet.is_empty());
    let problem = species_problem(probe_mesh, &inlet);
    let (fx, fy, ffx, ffy) = uniform_frozen(&problem, 0.7);
    // Same uniform velocity through the weak and tensor snapshots.
    let weak_transport = |name: &str| {
        KernelAdvDiff2D::with_coefficients(
            ConstantCoefficient(0.003),
            [ConstantCoefficient(0.7), ConstantCoefficient(0.0)],
        )
        .with_field_name(name)
    };
    let weak_kernel = ResidualKernelSet::from_kernel(weak_transport("c0"))
        .with(weak_transport("c1"))
        .with(weak_transport("c2"));
    let tensor_kernel = TensorResidualKernelSet::from_kernel(
        TensorKernelAdvDiff2D::with_coefficients(ConstantCoefficient(0.003), [fx.clone(), fy.clone()])
            .with_field_name("c0"),
    )
    .with(
        TensorKernelAdvDiff2D::with_coefficients(ConstantCoefficient(0.003), [fx.clone(), fy.clone()])
            .with_field_name("c1"),
    )
    .with(
        TensorKernelAdvDiff2D::with_coefficients(ConstantCoefficient(0.003), [fx, fy])
            .with_field_name("c2"),
    );
    let weak_terms = StateBoundaryTerms::new().with_entities(
        outlet.clone(),
        KernelAdvectionOutflow2D::with_field_names(ffx.clone(), ffy.clone(), ["c0", "c1", "c2"]),
    );
    let tensor_terms = StateTensorBoundaryTerms::new().with_entities(
        outlet,
        TensorKernelAdvectionOutflow2D::with_field_names(ffx, ffy, ["c0", "c1", "c2"]),
    );
    let n = problem.system_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.05 * ((i % 7) as f64));
    let weak_residual = problem
        .residual_operator(&weak_kernel)
        .at_time(0.0)
        .with_state_boundary(&weak_terms)
        .residual(state.as_ref());
    let tensor_residual = problem
        .tensor_residual_operator(&tensor_kernel)
        .at_time(0.0)
        .with_state_boundary(&tensor_terms)
        .residual(state.as_ref());
    assert_eq!(weak_residual.len(), n);
    for (a, b) in weak_residual.iter().zip(tensor_residual.iter()) {
        assert!((a - b).abs() < 1e-10, "weak/tensor boundary residual mismatch");
    }
    let weak_jac = problem
        .residual_operator(&weak_kernel)
        .at_time(0.0)
        .with_state_boundary(&weak_terms)
        .assemble_jacobian(state.as_ref())
        .to_dense();
    let tensor_jac = problem
        .tensor_residual_operator(&tensor_kernel)
        .at_time(0.0)
        .with_state_boundary(&tensor_terms)
        .assemble_jacobian(state.as_ref())
        .to_dense();
    for i in 0..n {
        for j in 0..n {
            assert!(
                (weak_jac[(i, j)] - tensor_jac[(i, j)]).abs() < 1e-9,
                "weak/tensor boundary jacobian mismatch at ({i}, {j})"
            );
        }
    }
}

#[test]
fn endpoint_sampling_reads_reduced_and_prescribed_dofs() {
    let periodic = SEM1DProblem::new(
        unit_interval(2, 1),
        2,
        FieldRegistry::new(["u"]),
        DofReduction1D::Periodic { facets: [0, 2] },
    );
    let state = Mat::from_fn(periodic.system_size(), 1, |_, _| 3.0);
    assert!((periodic.sample_endpoint_value(state.as_ref(), "u", 0) - 3.0).abs() < 1e-14);
    assert!((periodic.sample_endpoint_value(state.as_ref(), "u", 2) - 3.0).abs() < 1e-14);

    let dirichlet = SEM1DProblem::new(
        unit_interval(1, 1),
        2,
        FieldRegistry::new(["c0"]),
        DofReduction1D::Dirichlet { facets: vec![(1, 2.5)] },
    );
    let state = Mat::<f64>::zeros(dirichlet.system_size(), 1);
    assert!((dirichlet.sample_endpoint_value(state.as_ref(), "c0", 1) - 2.5).abs() < 1e-14);
}

#[test]
fn outflow_1d_passes_outflow_and_blocks_backflow() {
    let kernel = KernelAdvectionOutflow1D::with_field_names(0.5, ["c0", "c1", "c2"]);
    let wts = [1.0];
    let jdet = [1.0];
    let points = [1.0];
    let basis = [1.0];
    let grads = [0.0];
    let values = [0.4, 0.0, 0.0];
    let state = CellState {
        nfields: 3,
        npts: 1,
        gdim: 1,
        values: &values,
        grads: &[0.0; 3],
        field_indices: &[],
    };
    fn ctx_1d<'a>(
        normal: &'a [f64],
        wts: &'a [f64],
        jdet: &'a [f64],
        points: &'a [f64],
        basis: &'a [f64],
        grads: &'a [f64],
    ) -> FacetCtx<'a> {
        FacetCtx {
            time: 0.0,
            facet: FacetMeta { local_index: 8, physical_region: None },
            tdim: 0,
            gdim: 1,
            ncomp: 1,
            npts: 1,
            ndofs: 1,
            wts,
            jfacet_det: jdet,
            points,
            normal,
            values: basis,
            grads,
        }
    }
    let outflow = ctx_1d(&[1.0], &wts, &jdet, &points, &basis, &grads);
    assert!((kernel.residual_integrand(&outflow, &state, 0, 0, 0) - 0.5 * 0.4).abs() < 1e-14);
    assert!((kernel.jacobian_integrand(&outflow, &state, 0, 0, 0, 0, 0) - 0.5).abs() < 1e-14);
    assert_eq!(kernel.jacobian_integrand(&outflow, &state, 0, 1, 0, 0, 0), 0.0);
    // Zero incoming concentration: re-entering flow contributes nothing.
    let backflow = ctx_1d(&[-1.0], &wts, &jdet, &points, &basis, &grads);
    assert_eq!(kernel.residual_integrand(&backflow, &state, 0, 0, 0), 0.0);
    assert_eq!(kernel.jacobian_integrand(&backflow, &state, 0, 0, 0, 0, 0), 0.0);
}

/// Minimal 1D species system mirroring the 1D example (tensor + matrix-free).
struct TestSpeciesSystem1D<'a, M>
where
    M: ndmesh::traits::Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
{
    problem: &'a SEM1DProblem<M>,
    kernel: TensorResidualKernelSet<'a, 1>,
    terms: StateBoundaryTerms,
    m_inv: Vec<f64>,
}

impl<'a, M> OdeSys<'a> for TestSpeciesSystem1D<'a, M>
where
    M: ndmesh::traits::Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
{
    fn frhs(&self, t: f64, state: MatRef<f64>) -> Mat<f64> {
        let residual = self
            .problem
            .tensor_residual_operator(&self.kernel)
            .at_time(t)
            .with_state_boundary(&self.terms)
            .residual(state);
        Mat::from_fn(self.m_inv.len(), 1, |row, _| -self.m_inv[row] * residual[row])
    }
    fn fjac<'b>(&'a self, t: f64, state: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
        let operator = self
            .problem
            .tensor_residual_operator(&self.kernel)
            .at_time(t)
            .with_state_boundary(&self.terms);
        Box::new(MatrixFreeMinvJacobian::new(
            operator,
            state.to_owned(),
            &self.m_inv,
        ))
    }
}

type IntervalMesh = SingleElementMesh<
    f64,
    ndelement::ciarlet::CiarletElement<f64, ndelement::map::IdentityMap, f64>,
>;

/// Step 1D species with uniform rightward flow to t = 3; return peak |c|.
/// The front reaches the outlet (x = 1) at t = 2 with U = 0.5.
fn run_species_1d(outflow: bool) -> (bool, f64) {
    const NX: usize = 8;
    let problem = SEM1DProblem::new(
        unit_interval(NX, 1),
        2,
        FieldRegistry::new(["c0", "c1", "c2"]),
        DofReduction1D::FieldSpecific {
            reductions: vec![
                DofReduction1D::Dirichlet { facets: vec![(0, 1.0)] },
                DofReduction1D::Dirichlet { facets: vec![(0, 0.0)] },
                DofReduction1D::Dirichlet { facets: vec![(0, 0.0)] },
            ],
        },
    );
    let transport = |name: &str| {
        TensorKernelAdvDiff::with_coefficients(ConstantCoefficient(0.001), ConstantCoefficient(0.5))
            .with_field_name(name)
    };
    let kernel = TensorResidualKernelSet::from_kernel(transport("c0"))
        .with(transport("c1"))
        .with(transport("c2"));
    let terms = if outflow {
        StateBoundaryTerms::new().with_entities(
            [NX],
            KernelAdvectionOutflow1D::with_field_names(0.5, ["c0", "c1", "c2"]),
        )
    } else {
        StateBoundaryTerms::new()
    };
    let mass = problem.assemble_lumped_mass();
    let m_inv: Vec<f64> = (0..mass.nrows()).map(|i| 1.0 / mass[(i, i)]).collect();
    let system = TestSpeciesSystem1D {
        problem: &problem,
        kernel,
        terms,
        m_inv,
    };
    let y0 = Mat::<f64>::zeros(problem.system_size(), 1);
    let mut solver = DirkIntegrator::new(
        0.0,
        y0.as_ref(),
        ImplicitBT::implicit_euler(),
        1e-10,
        1e-10,
    );
    for _ in 0..60 {
        match solver.step(&system, 0.05) {
            Ok(step) => solver.accept_step(step),
            Err(_) => return (false, f64::INFINITY),
        }
    }
    let state = solver.state();
    let mut peak = 0.0f64;
    for name in ["c0", "c1", "c2"] {
        let field = problem.field_values(name, state.as_ref()).unwrap();
        for &v in &field.values {
            peak = peak.max(v.abs());
        }
    }
    (true, peak)
}

#[test]
fn species_outflow_1d_keeps_advected_front_bounded() {
    let (completed, peak) = run_species_1d(true);
    assert!(completed, "stepper failed with outflow BC");
    assert!(peak <= 1.1, "outflow BC should let the front leave, peak was {peak}");
}

#[test]
fn missing_outflow_1d_accumulates_at_outlet() {
    let (completed, peak) = run_species_1d(false);
    // Without the outflow flux the outlet acts as a closed wall: either the
    // stepper fails or species pile up far beyond physical bounds.
    assert!(!completed || peak > 1.5, "expected blowup without outflow BC, peak was {peak}");
}
