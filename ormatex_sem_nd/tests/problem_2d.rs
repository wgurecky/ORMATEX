use faer::prelude::*;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndfunctionspace::{traits::FunctionSpace, FunctionSpaceImpl};
use ndmesh::{
    shapes::unit_square,
    traits::{Builder, Entity, Geometry, Mesh, Point, Topology},
    SingleElementMesh, SingleElementMeshBuilder,
};
use ormatex_sem_nd::{
    BoundaryIntegrator, CellState, DofReduction2D, FacetCtx, FieldRegistry, KernelAdvDiff2D,
    KernelAdvDiffSUPG2D, KernelMass, LinearForm, LocalCtx, ResidualKernel, SEM2DProblem,
    StateTensorBoundaryTerms, TensorKernelAdvDiff2D, TensorKernelEdacDongOutflow2D,
    TensorKernelEdacNavierStokes2D, TensorKernelEdacSplitBoundaryFlux2D,
};
use ormatex_sem_nd::{
    ConstantCoefficient, KernelEdacDongOutflow2D, KernelEdacNavierStokes2D,
    KernelEdacSplitBoundaryFlux2D, MeshMetadata, PhysicalRegion, RegionCoefficient,
    RobinConvection, StateBoundaryIntegrator, StateBoundaryTerms,
};

type QuadMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

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
    assert_eq!(left.len(), right.len());
    assert_eq!(bottom.len(), top.len());
    left.into_iter()
        .zip(right)
        .chain(bottom.into_iter().zip(top))
        .map(|((_, a), (_, b))| [a, b])
        .collect()
}

fn translated_periodic_mesh() -> QuadMesh {
    let mut builder = SingleElementMeshBuilder::new(2, (ReferenceCellType::Quadrilateral, 1));
    for (id, xy) in [
        (0, [10.0, -3.0]),
        (1, [14.0, -2.0]),
        (2, [9.0, 3.0]),
        (3, [13.0, 4.0]),
        (4, [18.0, -1.0]),
        (5, [17.0, 5.0]),
    ] {
        builder.add_point(id, &xy);
    }
    builder.add_cell(0, &[0, 1, 2, 3]);
    builder.add_cell(1, &[1, 4, 3, 5]);
    builder.create_mesh()
}

fn facet_with_endpoints(mesh: &QuadMesh, a: [f64; 2], b: [f64; 2]) -> usize {
    mesh.entity_iter(ReferenceCellType::Interval)
        .find(|facet| {
            let endpoints: Vec<_> = facet
                .geometry()
                .points()
                .map(|point| {
                    let mut xy = [0.0; 2];
                    point.coords(&mut xy);
                    xy
                })
                .collect();
            endpoints.len() == 2
                && ((endpoints[0] == a && endpoints[1] == b)
                    || (endpoints[0] == b && endpoints[1] == a))
        })
        .unwrap()
        .local_index()
}

struct XSource;

impl LinearForm for XSource {
    fn integrand(&self, ctx: &LocalCtx, _equation: usize, q: usize, test_i: usize) -> f64 {
        ctx.point(q)[0] * ctx.test(test_i, 0).v(q)
    }
}

struct YFlux;

impl BoundaryIntegrator for YFlux {
    fn integrand_rhs(&self, ctx: &FacetCtx, _equation: usize, q: usize, test_i: usize) -> f64 {
        ctx.point(q)[1] * ctx.test(test_i, 0).v(q)
    }
}

struct GradientFlux;

impl BoundaryIntegrator for GradientFlux {
    fn integrand_rhs(&self, ctx: &FacetCtx, _equation: usize, q: usize, test_i: usize) -> f64 {
        ctx.test(test_i, 0).grad(q, 0)
    }
}

struct StateGradient;

impl StateBoundaryIntegrator for StateGradient {
    fn residual_integrand(
        &self,
        ctx: &FacetCtx,
        state: &CellState,
        _equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        state.grad(0, q, 0) * ctx.test(test_i, 0).v(q)
    }

    fn jacobian_integrand(
        &self,
        _ctx: &FacetCtx,
        _state: &CellState,
        _equation: usize,
        _unknown: usize,
        _q: usize,
        _test_i: usize,
        _trial_i: usize,
    ) -> f64 {
        0.0
    }
}

struct QuadraticReaction;

impl ResidualKernel for QuadraticReaction {
    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        _equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        state.value(0, q).powi(2) * ctx.test(test_i, 0).v(q)
    }

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        _equation: usize,
        _unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        2.0 * state.value(0, q) * ctx.test(test_i, 0).v(q) * ctx.trial(trial_i, 0).v(q)
    }
}

struct CoupledReaction;

struct GenericEdac(KernelEdacNavierStokes2D);

impl ResidualKernel for GenericEdac {
    fn nfields(&self) -> usize {
        self.0.nfields()
    }

    fn field_names(&self) -> Option<Vec<String>> {
        self.0.field_names()
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        self.0.residual_integrand(ctx, state, equation, q, test_i)
    }

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        self.0
            .jacobian_integrand(ctx, state, equation, unknown, q, test_i, trial_i)
    }
}

impl ResidualKernel for CoupledReaction {
    fn nfields(&self) -> usize {
        2
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        let test = ctx.test(test_i, 0).v(q);
        match equation {
            0 => (state.value(0, q) + 2.0 * state.value(1, q)) * test,
            1 => (3.0 * state.value(0, q) - state.value(1, q)) * test,
            _ => unreachable!(),
        }
    }

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx,
        _state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        let coefficient = match (equation, unknown) {
            (0, 0) => 1.0,
            (0, 1) => 2.0,
            (1, 0) => 3.0,
            (1, 1) => -1.0,
            _ => unreachable!(),
        };
        coefficient * ctx.test(test_i, 0).v(q) * ctx.trial(trial_i, 0).v(q)
    }
}

#[test]
fn dirichlet_eliminates_every_high_order_facet_dof() {
    let mesh: QuadMesh = unit_square(2, 1, ReferenceCellType::Quadrilateral, 1);
    let left_facet = mesh
        .entity_iter(ReferenceCellType::Interval)
        .find(|facet| {
            facet.geometry().points().all(|point| {
                let mut xy = [0.0; 2];
                point.coords(&mut xy);
                xy[0].abs() < 1e-12
            })
        })
        .unwrap()
        .local_index();
    let problem = SEM2DProblem::new(
        mesh,
        2,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::Dirichlet {
            facets: vec![(left_facet, 0.0)],
        },
    );
    let space = FunctionSpaceImpl::new(problem.mesh(), problem.family());
    let facet_dofs = space
        .entity_closure_dofs(ReferenceCellType::Interval, left_facet)
        .unwrap();
    assert_eq!(facet_dofs.len(), 3);
    assert!(facet_dofs
        .iter()
        .all(|&dof| problem.target_dof(dof).is_none()));
    assert_eq!(problem.reduced_size(), 12);
}

#[test]
fn nonzero_dirichlet_value_enters_2d_state() {
    let mesh: QuadMesh = unit_square(1, 1, ReferenceCellType::Quadrilateral, 1);
    let left_facet = mesh
        .entity_iter(ReferenceCellType::Interval)
        .find(|facet| {
            facet.geometry().points().all(|point| {
                let mut xy = [0.0; 2];
                point.coords(&mut xy);
                xy[0].abs() < 1e-12
            })
        })
        .unwrap()
        .local_index();
    let problem = SEM2DProblem::new(
        mesh,
        1,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::Dirichlet {
            facets: vec![(left_facet, 3.0)],
        },
    );
    let kernel = KernelAdvDiff2D::new(1.0, [0.0, 0.0]);
    let state = Mat::from_fn(problem.reduced_size(), 1, |_, _| 3.0);
    let residual = problem.assemble_residual(0.0, &kernel, state.as_ref());
    assert!(residual.iter().all(|value| value.abs() < 1e-12));

    let full_problem = SEM2DProblem::new(
        unit_square(1, 1, ReferenceCellType::Quadrilateral, 1),
        1,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::None,
    );
    let full_matrix = full_problem.assemble_bilinear(0.0, &kernel).to_dense();
    let space = FunctionSpaceImpl::new(problem.mesh(), problem.family());
    let boundary_dofs = space
        .entity_closure_dofs(ReferenceCellType::Interval, left_facet)
        .unwrap();
    let mut rhs = vec![0.0; problem.reduced_size()];
    problem.apply_dirichlet_rhs_correction(0.0, &kernel, &mut rhs);
    for full in 0..full_problem.reduced_size() {
        if let Some(reduced) = problem.target_dof(full) {
            let expected: f64 = boundary_dofs
                .iter()
                .map(|&boundary| -3.0 * full_matrix[(full, boundary)])
                .sum();
            assert!((rhs[reduced] - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn state_boundary_assembly_interpolates_and_scatters_fields() {
    let mesh: QuadMesh = unit_square(1, 1, ReferenceCellType::Quadrilateral, 1);
    let problem = SEM2DProblem::new(
        mesh,
        1,
        FieldRegistry::new(["u", "v", "p"]),
        DofReduction2D::None,
    );
    let state = Mat::from_fn(problem.system_size(), 1, |row, _| match row {
        0..=3 => 1.0,
        4..=7 => 0.5,
        _ => 0.3,
    });
    let right_facet = problem
        .mesh()
        .entity_iter(ReferenceCellType::Interval)
        .find(|facet| {
            let mut midpoint = [0.0; 2];
            let mut count = 0.0;
            for point in facet.geometry().points() {
                let mut xy = [0.0; 2];
                point.coords(&mut xy);
                midpoint[0] += xy[0];
                midpoint[1] += xy[1];
                count += 1.0;
            }
            midpoint[0] / count > 1.0 - 1e-12
        })
        .unwrap()
        .local_index();
    let terms =
        StateBoundaryTerms::new().with_entities([right_facet], KernelEdacSplitBoundaryFlux2D);
    let right = problem.assemble_state_boundary(0.0, state.as_ref(), &terms);
    assert!((right.residual[0..4].iter().sum::<f64>() - 0.5).abs() < 1e-12);
    assert!((right.residual[4..8].iter().sum::<f64>() - 0.25).abs() < 1e-12);
    assert!((right.residual[8..12].iter().sum::<f64>() - 0.15).abs() < 1e-12);

    let direction = Mat::from_fn(problem.system_size(), 1, |row, _| 0.1 * (row + 1) as f64);
    let epsilon = 1e-7;
    let perturbed = Mat::from_fn(problem.system_size(), 1, |row, _| {
        state[(row, 0)] + epsilon * direction[(row, 0)]
    });
    let perturbed_right = problem.assemble_state_boundary(0.0, perturbed.as_ref(), &terms);
    let finite_difference = right
        .residual
        .iter()
        .zip(perturbed_right.residual)
        .map(|(base, perturbed)| (perturbed - base) / epsilon)
        .collect::<Vec<_>>();
    let analytic = right.jacobian.as_ref() * direction.as_ref();
    for row in 0..problem.system_size() {
        assert!((finite_difference[row] - analytic[(row, 0)]).abs() < 1e-6);
    }
}

#[test]
fn state_boundary_matrix_free_action_matches_assembled_jacobian() {
    let mesh: QuadMesh = unit_square(1, 1, ReferenceCellType::Quadrilateral, 1);
    let problem = SEM2DProblem::new(
        mesh,
        1,
        FieldRegistry::new(["u", "v", "p"]),
        DofReduction2D::None,
    );
    let state = Mat::from_fn(problem.system_size(), 1, |row, _| 0.2 + 0.03 * row as f64);
    let direction = Mat::from_fn(problem.system_size(), 2, |row, column| {
        0.1 * (row + 1) as f64 * (column as f64 + 1.0)
    });
    let terms = StateBoundaryTerms::new().with_default(KernelEdacSplitBoundaryFlux2D);
    let assembled = problem.assemble_state_boundary(0.0, state.as_ref(), &terms);
    let action =
        problem.apply_state_boundary_jacobian(0.0, state.as_ref(), direction.as_ref(), &terms);
    let expected = assembled.jacobian.as_ref() * direction.as_ref();
    for row in 0..problem.system_size() {
        for column in 0..direction.ncols() {
            assert!((action[(row, column)] - expected[(row, column)]).abs() < 1e-12);
        }
    }
}

#[test]
fn high_order_tensor_state_boundary_action_matches_assembled_jacobian() {
    let problem = SEM2DProblem::new(
        unit_square(1, 1, ReferenceCellType::Quadrilateral, 1),
        4,
        FieldRegistry::new(["u", "v", "p"]),
        DofReduction2D::None,
    );
    let state = Mat::from_fn(problem.system_size(), 1, |row, _| 0.2 + 0.01 * row as f64);
    let direction = Mat::from_fn(problem.system_size(), 2, |row, column| {
        0.03 * (row + 1) as f64 * (column + 1) as f64
    });
    for terms in [
        StateBoundaryTerms::new().with_default(KernelEdacSplitBoundaryFlux2D),
        StateBoundaryTerms::new().with_default(KernelEdacDongOutflow2D::new(1.0, 0.05, 1.0)),
    ] {
        let assembled = problem.assemble_state_boundary(0.0, state.as_ref(), &terms);
        let action =
            problem.apply_state_boundary_jacobian(0.0, state.as_ref(), direction.as_ref(), &terms);
        let expected = assembled.jacobian.as_ref() * direction.as_ref();
        for row in 0..problem.system_size() {
            for column in 0..direction.ncols() {
                assert!((action[(row, column)] - expected[(row, column)]).abs() < 1e-11);
            }
        }
    }
}

#[test]
fn tensor_state_boundary_operator_matches_weak_boundary_path() {
    let problem = SEM2DProblem::new(
        unit_square(1, 1, ReferenceCellType::Quadrilateral, 1),
        3,
        FieldRegistry::new(["u", "v", "p"]),
        DofReduction2D::None,
    );
    let state = Mat::from_fn(problem.system_size(), 1, |row, _| 0.2 + 0.01 * row as f64);
    let direction = Mat::from_fn(problem.system_size(), 1, |row, _| 0.03 * (row + 1) as f64);
    let weak_terms = StateBoundaryTerms::new().with_default(KernelEdacSplitBoundaryFlux2D);
    let tensor_terms =
        StateTensorBoundaryTerms::new().with_default(TensorKernelEdacSplitBoundaryFlux2D);
    let weak_kernel = GenericEdac(KernelEdacNavierStokes2D::new(1.0, 0.01, 4.0, 0.1));
    let tensor_kernel = TensorKernelEdacNavierStokes2D::new(1.0, 0.01, 4.0, 0.1);
    let weak = problem
        .residual_operator(&weak_kernel)
        .with_state_boundary(&weak_terms);
    let tensor = problem
        .tensor_residual_operator(&tensor_kernel)
        .with_state_boundary(&tensor_terms);

    let weak_residual = weak.residual(state.as_ref());
    let tensor_residual = tensor.residual(state.as_ref());
    for (weak, tensor) in weak_residual.iter().zip(tensor_residual) {
        assert!((weak - tensor).abs() < 1e-10);
    }

    let weak_action = weak.apply_jacobian(state.as_ref(), direction.as_ref());
    let tensor_action = tensor.apply_jacobian(state.as_ref(), direction.as_ref());
    for row in 0..problem.system_size() {
        assert!((weak_action[(row, 0)] - tensor_action[(row, 0)]).abs() < 1e-10);
    }

    let weak_dong_terms =
        StateBoundaryTerms::new().with_default(KernelEdacDongOutflow2D::new(1.0, 0.05, 1.0));
    let tensor_dong_terms = StateTensorBoundaryTerms::new()
        .with_default(TensorKernelEdacDongOutflow2D::new(1.0, 0.05, 1.0));
    let weak_dong = problem
        .residual_operator(&weak_kernel)
        .with_state_boundary(&weak_dong_terms);
    let tensor_dong = problem
        .tensor_residual_operator(&tensor_kernel)
        .with_state_boundary(&tensor_dong_terms);
    let weak_dong_residual = weak_dong.residual(state.as_ref());
    let tensor_dong_residual = tensor_dong.residual(state.as_ref());
    for (weak, tensor) in weak_dong_residual.iter().zip(tensor_dong_residual) {
        assert!((weak - tensor).abs() < 1e-10);
    }
    let weak_dong_action = weak_dong.apply_jacobian(state.as_ref(), direction.as_ref());
    let tensor_dong_action = tensor_dong.apply_jacobian(state.as_ref(), direction.as_ref());
    for row in 0..problem.system_size() {
        assert!((weak_dong_action[(row, 0)] - tensor_dong_action[(row, 0)]).abs() < 1e-10);
    }
}

#[test]
fn volume_kernel_reads_physical_quadrature_points() {
    let mesh: QuadMesh = unit_square(2, 1, ReferenceCellType::Quadrilateral, 1);
    let problem = SEM2DProblem::new(mesh, 2, FieldRegistry::new(["x"]), DofReduction2D::None);
    assert!((problem.assemble_linear(0.0, &XSource).iter().sum::<f64>() - 0.5).abs() < 1e-12);
}

#[test]
fn lumped_mass_matches_generic_gll_mass() {
    let mesh: QuadMesh = unit_square(2, 1, ReferenceCellType::Quadrilateral, 1);
    let facet_pairs = unit_square_periodic_pairs(&mesh);
    let problem = SEM2DProblem::new(
        mesh,
        2,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::Periodic {
            facet_pairs,
            tolerance: 1e-12,
        },
    );
    let generic = problem
        .assemble_bilinear(0.0, &KernelMass::new())
        .to_dense();
    let lumped = problem.assemble_lumped_mass().to_dense();
    for i in 0..generic.nrows() {
        for j in 0..generic.ncols() {
            assert!((generic[(i, j)] - lumped[(i, j)]).abs() < 1e-12);
        }
    }
}

#[test]
fn periodic_pairs_identify_translated_reversed_high_order_facets() {
    let mesh = translated_periodic_mesh();
    let source = facet_with_endpoints(&mesh, [10.0, -3.0], [9.0, 3.0]);
    let target = facet_with_endpoints(&mesh, [18.0, -1.0], [17.0, 5.0]);
    let problem = SEM2DProblem::new(
        mesh,
        3,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::Periodic {
            facet_pairs: vec![[source, target]],
            tolerance: 1e-12,
        },
    );
    let space = FunctionSpaceImpl::new(problem.mesh(), problem.family());
    let source_dofs = space
        .entity_closure_dofs(ReferenceCellType::Interval, source)
        .unwrap();
    let target_dofs = space
        .entity_closure_dofs(ReferenceCellType::Interval, target)
        .unwrap();
    assert_eq!(source_dofs.len(), 4);
    assert_eq!(target_dofs.len(), 4);
    assert_eq!(problem.reduced_size(), 24);
    let source_reduced: std::collections::HashSet<_> = source_dofs
        .iter()
        .map(|&dof| problem.target_dof(dof).unwrap())
        .collect();
    let target_reduced: std::collections::HashSet<_> = target_dofs
        .iter()
        .map(|&dof| problem.target_dof(dof).unwrap())
        .collect();
    assert_eq!(source_reduced.len(), 4);
    assert_eq!(source_reduced, target_reduced);
}

#[test]
fn high_order_tensor_action_matches_assembled_on_skew_cells() {
    let problem = SEM2DProblem::new(
        translated_periodic_mesh(),
        4,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::None,
    );
    let n = problem.reduced_size();
    let state = Mat::from_fn(n, 1, |row, _| 0.2 + 0.03 * row as f64);
    let direction = Mat::from_fn(n, 2, |row, column| {
        (0.17 * (row + 1) as f64 * (column as f64 + 1.0)).sin()
    });
    let kernel = TensorKernelAdvDiff2D::new(0.13, [0.4, -0.2]);
    let tensor_operator = problem.tensor_residual_operator(&kernel);
    let assembled = problem
        .tensor_residual_operator(&kernel)
        .assemble_jacobian(state.as_ref())
        .to_dense();
    let action = tensor_operator.apply_jacobian(state.as_ref(), direction.as_ref());
    let expected = assembled.as_ref() * direction.as_ref();
    for row in 0..n {
        for column in 0..direction.ncols() {
            assert!(
                (action[(row, column)] - expected[(row, column)]).abs() < 1e-10,
                "tensor action mismatch at ({row}, {column}): {} != {}",
                action[(row, column)],
                expected[(row, column)]
            );
        }
    }
}

#[test]
fn high_order_edac_tensor_action_matches_assembled() {
    let problem = SEM2DProblem::new(
        unit_square(1, 1, ReferenceCellType::Quadrilateral, 1),
        3,
        FieldRegistry::new(["u", "v", "p"]),
        DofReduction2D::None,
    );
    let n = problem.system_size();
    let state = Mat::from_fn(n, 1, |row, _| 0.2 + 0.01 * row as f64);
    let direction = Mat::from_fn(n, 1, |row, _| (0.13 * row as f64).sin());
    let kernel = TensorKernelEdacNavierStokes2D::new(1.0, 0.01, 4.0, 0.1);
    let generic = GenericEdac(KernelEdacNavierStokes2D::new(1.0, 0.01, 4.0, 0.1));
    let tensor_operator = problem.tensor_residual_operator(&kernel);
    let tensor_residual = tensor_operator.residual(state.as_ref());
    let generic_residual = problem.assemble_residual(0.0, &generic, state.as_ref());
    for (tensor, generic) in tensor_residual.iter().zip(generic_residual) {
        assert!((tensor - generic).abs() < 1e-10);
    }
    let assembled = tensor_operator.assemble_jacobian(state.as_ref()).to_dense();
    let generic_assembled = problem
        .assemble_residual_jacobian(0.0, &generic, state.as_ref())
        .to_dense();
    for row in 0..n {
        for col in 0..n {
            assert!((assembled[(row, col)] - generic_assembled[(row, col)]).abs() < 1e-10);
        }
    }
    let action = tensor_operator.apply_jacobian(state.as_ref(), direction.as_ref());
    let expected = assembled.as_ref() * direction.as_ref();
    for row in 0..n {
        assert!(
            (action[(row, 0)] - expected[(row, 0)]).abs() < 1e-10,
            "EDAC tensor action mismatch at row {row}: {} != {}",
            action[(row, 0)],
            expected[(row, 0)]
        );
    }
}

#[test]
#[should_panic(expected = "not related by a translation")]
fn periodic_pairs_reject_nontranslated_facets() {
    let mesh: QuadMesh = unit_square(2, 1, ReferenceCellType::Quadrilateral, 1);
    let pairs = unit_square_periodic_pairs(&mesh);
    SEM2DProblem::new(
        mesh,
        2,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::Periodic {
            facet_pairs: vec![[pairs[0][0], pairs.last().unwrap()[0]]],
            tolerance: 1e-12,
        },
    );
}

#[test]
#[should_panic(expected = "must be a boundary interval")]
fn periodic_pairs_reject_interior_facets() {
    let mesh: QuadMesh = unit_square(2, 1, ReferenceCellType::Quadrilateral, 1);
    let interior = mesh
        .entity_iter(ReferenceCellType::Interval)
        .find(|facet| {
            let topology = facet.topology();
            topology
                .connected_entity_iter(ReferenceCellType::Quadrilateral)
                .count()
                == 2
        })
        .unwrap()
        .local_index();
    let boundary = unit_square_periodic_pairs(&mesh)[0][0];
    SEM2DProblem::new(
        mesh,
        2,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::Periodic {
            facet_pairs: vec![[interior, boundary]],
            tolerance: 1e-12,
        },
    );
}

#[test]
fn boundary_kernel_reads_physical_quadrature_points() {
    let mesh: QuadMesh = unit_square(1, 1, ReferenceCellType::Quadrilateral, 1);
    let problem = SEM2DProblem::new(mesh, 2, FieldRegistry::new(["x"]), DofReduction2D::None);
    let flux = YFlux;
    let boundary = problem.assemble_boundary(0.0, |facet| {
        (facet.midpoint[0].abs() < 1e-12).then_some(&flux as &dyn BoundaryIntegrator)
    });
    assert!((boundary.rhs.iter().sum::<f64>() - 0.5).abs() < 1e-12);
}

#[test]
fn boundary_kernel_receives_physical_basis_gradients() {
    let mesh: QuadMesh = unit_square(1, 1, ReferenceCellType::Quadrilateral, 1);
    let problem = SEM2DProblem::new(mesh, 2, FieldRegistry::new(["x"]), DofReduction2D::None);
    let flux = GradientFlux;
    let boundary = problem.assemble_boundary(0.0, |facet| {
        (facet.midpoint[0] > 1.0 - 1e-12).then_some(&flux as &dyn BoundaryIntegrator)
    });
    assert!(boundary.rhs.iter().any(|value| value.abs() > 1e-12));
}

#[test]
fn state_boundary_gradient_uses_the_complete_physical_cell_state() {
    let problem = SEM2DProblem::new(
        unit_square(1, 1, ReferenceCellType::Quadrilateral, 1),
        2,
        FieldRegistry::new(["u"]),
        DofReduction2D::None,
    );
    let positions = problem.dof_positions();
    let state = Mat::from_fn(problem.system_size(), 1, |row, _| positions[row].0);
    let right = problem
        .mesh()
        .entity_iter(ReferenceCellType::Interval)
        .find(|facet| {
            let mut midpoint = [0.0; 2];
            let mut count = 0.0;
            for point in facet.geometry().points() {
                let mut xy = [0.0; 2];
                point.coords(&mut xy);
                midpoint[0] += xy[0];
                midpoint[1] += xy[1];
                count += 1.0;
            }
            midpoint[0] / count > 1.0 - 1e-12
        })
        .unwrap()
        .local_index();
    let terms = StateBoundaryTerms::new().with_entities([right], StateGradient);
    let boundary = problem.assemble_state_boundary(0.0, state.as_ref(), &terms);
    assert!((boundary.residual.iter().sum::<f64>() - 1.0).abs() < 1e-10);
}

#[test]
fn natural_boundary_includes_nonzero_dirichlet_column_correction() {
    let left_and_bottom = |mesh: &QuadMesh, x: f64, y: f64| {
        mesh.entity_iter(ReferenceCellType::Interval)
            .find(|facet| {
                let mut midpoint = [0.0; 2];
                let mut count = 0.0;
                for point in facet.geometry().points() {
                    let mut xy = [0.0; 2];
                    point.coords(&mut xy);
                    midpoint[0] += xy[0];
                    midpoint[1] += xy[1];
                    count += 1.0;
                }
                (midpoint[0] / count - x).abs() < 1e-12 && (midpoint[1] / count - y).abs() < 1e-12
            })
            .unwrap()
            .local_index()
    };
    let reduced_mesh: QuadMesh = unit_square(1, 1, ReferenceCellType::Quadrilateral, 1);
    let left = left_and_bottom(&reduced_mesh, 0.0, 0.5);
    let reduced = SEM2DProblem::new(
        reduced_mesh,
        1,
        FieldRegistry::new(["u"]),
        DofReduction2D::Dirichlet {
            facets: vec![(left, 3.0)],
        },
    );
    let full = SEM2DProblem::new(
        unit_square(1, 1, ReferenceCellType::Quadrilateral, 1),
        1,
        FieldRegistry::new(["u"]),
        DofReduction2D::None,
    );
    let robin = RobinConvection::new(2.0, 0.0);
    let reduced_boundary = reduced.assemble_boundary(0.0, |facet| {
        (facet.midpoint[1].abs() < 1e-12).then_some(&robin as &dyn BoundaryIntegrator)
    });
    let full_boundary = full.assemble_boundary(0.0, |facet| {
        (facet.midpoint[1].abs() < 1e-12).then_some(&robin as &dyn BoundaryIntegrator)
    });
    let full_matrix = full_boundary.mat.to_dense();
    let space = FunctionSpaceImpl::new(full.mesh(), full.family());
    let left_dofs = space
        .entity_closure_dofs(ReferenceCellType::Interval, left)
        .unwrap();
    for full_row in 0..full.reduced_size() {
        let Some(reduced_row) = reduced.target_dof(full_row) else {
            continue;
        };
        let expected = left_dofs
            .iter()
            .map(|&full_col| -3.0 * full_matrix[(full_row, full_col)])
            .sum::<f64>();
        assert!((reduced_boundary.rhs[reduced_row] - expected).abs() < 1e-12);
    }
}

#[test]
fn residual_kernel_jacobian_matches_directional_difference() {
    let mesh: QuadMesh = unit_square(1, 1, ReferenceCellType::Quadrilateral, 1);
    let problem = SEM2DProblem::new(
        mesh,
        2,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::None,
    );
    let n = problem.reduced_size();
    let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.1 * i as f64);
    let direction = Mat::from_fn(n, 1, |i, _| (0.3 * i as f64).sin());
    let kernel = QuadraticReaction;
    let action = problem.apply_jacobian(0.0, &kernel, state.as_ref(), direction.as_ref());
    let eps = 1e-7;
    let perturbed = state.as_ref() + faer::Scale(eps) * direction.as_ref();
    let residual = problem.assemble_residual(0.0, &kernel, state.as_ref());
    let perturbed_residual = problem.assemble_residual(0.0, &kernel, perturbed.as_ref());
    for i in 0..n {
        assert!((action[(i, 0)] - (perturbed_residual[i] - residual[i]) / eps).abs() < 1e-7);
    }
}

#[test]
fn coupled_2d_system_assembles_cross_field_blocks_and_matrix_free_action() {
    let mesh: QuadMesh = unit_square(1, 1, ReferenceCellType::Quadrilateral, 1);
    let problem = SEM2DProblem::new(
        mesh,
        2,
        FieldRegistry::new(["a", "b"]),
        DofReduction2D::None,
    );
    let n = problem.reduced_size();
    let state = Mat::from_fn(2 * n, 1, |i, _| 0.2 + 0.02 * i as f64);
    let direction = Mat::from_fn(2 * n, 1, |i, _| (0.17 * i as f64).sin());
    let kernel = CoupledReaction;
    let assembled = problem
        .assemble_residual_jacobian(0.0, &kernel, state.as_ref())
        .to_dense();
    assert!(assembled[(0, n)] != 0.0);
    assert!(assembled[(n, 0)] != 0.0);
    let action = problem.apply_jacobian(0.0, &kernel, state.as_ref(), direction.as_ref());
    let expected = assembled.as_ref() * direction.as_ref();
    for row in 0..2 * n {
        assert!((action[(row, 0)] - expected[(row, 0)]).abs() < 1e-11);
    }
}

#[test]
fn supg_2d_zero_tau_matches_advection_diffusion() {
    let mesh: QuadMesh = unit_square(1, 1, ReferenceCellType::Quadrilateral, 1);
    let problem = SEM2DProblem::new(
        mesh,
        2,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::None,
    );
    let plain = problem
        .assemble_bilinear(0.0, &KernelAdvDiff2D::new(0.1, [0.4, -0.2]))
        .to_dense();
    let supg = problem
        .assemble_bilinear(0.0, &KernelAdvDiffSUPG2D::new(0.1, [0.4, -0.2], 0.0))
        .to_dense();
    for i in 0..plain.nrows() {
        for j in 0..plain.ncols() {
            assert!((plain[(i, j)] - supg[(i, j)]).abs() < 1e-12);
        }
    }
}

#[test]
fn region_coefficient_changes_2d_material_operator() {
    let region = PhysicalRegion {
        dimension: 2,
        tag: 7,
    };
    let metadata = MeshMetadata {
        cell_regions: vec![Some(region)],
        ..MeshMetadata::default()
    };
    let mesh: QuadMesh = unit_square(1, 1, ReferenceCellType::Quadrilateral, 1);
    let problem = SEM2DProblem::new_with_metadata(
        mesh,
        2,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::None,
        metadata,
    );
    let region_diffusion =
        RegionCoefficient::new(std::collections::HashMap::from([(region, 3.0)]), 1.0);
    let region_kernel = KernelAdvDiff2D::with_coefficients(
        region_diffusion,
        [ConstantCoefficient(0.0), ConstantCoefficient(0.0)],
    );
    let selected = problem.assemble_bilinear(0.0, &region_kernel).to_dense();
    let expected = problem
        .assemble_bilinear(0.0, &KernelAdvDiff2D::new(3.0, [0.0, 0.0]))
        .to_dense();
    for i in 0..selected.nrows() {
        for j in 0..selected.ncols() {
            assert!((selected[(i, j)] - expected[(i, j)]).abs() < 1e-12);
        }
    }
}
