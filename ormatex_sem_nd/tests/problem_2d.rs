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
};
use ormatex_sem_nd::{
    ConstantCoefficient, KernelEdacSplitBoundaryFlux2D, MeshMetadata, PhysicalRegion,
    RegionCoefficient, StateBoundaryTerms,
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
    let residual = problem.assemble_residual(&kernel, state.as_ref());
    assert!(residual.iter().all(|value| value.abs() < 1e-12));

    let full_problem = SEM2DProblem::new(
        unit_square(1, 1, ReferenceCellType::Quadrilateral, 1),
        1,
        FieldRegistry::new(["temperature"]),
        DofReduction2D::None,
    );
    let full_matrix = full_problem.assemble_bilinear(&kernel).to_dense();
    let space = FunctionSpaceImpl::new(problem.mesh(), problem.family());
    let boundary_dofs = space
        .entity_closure_dofs(ReferenceCellType::Interval, left_facet)
        .unwrap();
    let mut rhs = vec![0.0; problem.reduced_size()];
    problem.apply_dirichlet_rhs_correction(&kernel, &mut rhs);
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
    let split_flux = KernelEdacSplitBoundaryFlux2D;
    let right = problem.assemble_state_boundary(state.as_ref(), |facet| {
        ((facet.midpoint[0] - 1.0).abs() < 1e-12)
            .then_some(&split_flux as &dyn ormatex_sem_nd::StateBoundaryIntegrator)
    });
    assert!((right.residual[0..4].iter().sum::<f64>() - 0.5).abs() < 1e-12);
    assert!((right.residual[4..8].iter().sum::<f64>() - 0.25).abs() < 1e-12);
    assert!((right.residual[8..12].iter().sum::<f64>() - 0.15).abs() < 1e-12);

    let direction = Mat::from_fn(problem.system_size(), 1, |row, _| 0.1 * (row + 1) as f64);
    let epsilon = 1e-7;
    let perturbed = Mat::from_fn(problem.system_size(), 1, |row, _| {
        state[(row, 0)] + epsilon * direction[(row, 0)]
    });
    let perturbed_right = problem.assemble_state_boundary(perturbed.as_ref(), |facet| {
        ((facet.midpoint[0] - 1.0).abs() < 1e-12)
            .then_some(&split_flux as &dyn ormatex_sem_nd::StateBoundaryIntegrator)
    });
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
    let split_flux = KernelEdacSplitBoundaryFlux2D;
    let terms = StateBoundaryTerms::new().with_default(KernelEdacSplitBoundaryFlux2D);
    let assembled = problem.assemble_state_boundary(state.as_ref(), |_facet| {
        Some(&split_flux as &dyn ormatex_sem_nd::StateBoundaryIntegrator)
    });
    let action =
        problem.apply_state_boundary_jacobian_matfree(state.as_ref(), direction.as_ref(), &terms);
    let expected = assembled.jacobian.as_ref() * direction.as_ref();
    for row in 0..problem.system_size() {
        for column in 0..direction.ncols() {
            assert!((action[(row, column)] - expected[(row, column)]).abs() < 1e-12);
        }
    }
}

#[test]
fn volume_kernel_reads_physical_quadrature_points() {
    let mesh: QuadMesh = unit_square(2, 1, ReferenceCellType::Quadrilateral, 1);
    let problem = SEM2DProblem::new(mesh, 2, FieldRegistry::new(["x"]), DofReduction2D::None);
    assert!((problem.assemble_linear(&XSource).iter().sum::<f64>() - 0.5).abs() < 1e-12);
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
    let generic = problem.assemble_bilinear(&KernelMass::new()).to_dense();
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
    let boundary =
        problem.assemble_boundary(|facet| (facet.midpoint[0].abs() < 1e-12).then_some(&flux));
    assert!((boundary.rhs.iter().sum::<f64>() - 0.5).abs() < 1e-12);
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
    let action = problem.apply_jacobian_matfree(&kernel, state.as_ref(), direction.as_ref());
    let eps = 1e-7;
    let perturbed = state.as_ref() + faer::Scale(eps) * direction.as_ref();
    let residual = problem.assemble_residual(&kernel, state.as_ref());
    let perturbed_residual = problem.assemble_residual(&kernel, perturbed.as_ref());
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
        .assemble_system_residual_jacobian(&kernel, state.as_ref())
        .to_dense();
    assert!(assembled[(0, n)] != 0.0);
    assert!(assembled[(n, 0)] != 0.0);
    let action = problem.apply_system_jacobian_matfree(&kernel, state.as_ref(), direction.as_ref());
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
        .assemble_bilinear(&KernelAdvDiff2D::new(0.1, [0.4, -0.2]))
        .to_dense();
    let supg = problem
        .assemble_bilinear(&KernelAdvDiffSUPG2D::new(0.1, [0.4, -0.2], 0.0))
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
    let selected = problem
        .assemble_system_bilinear_at(0.0, &region_kernel)
        .to_dense();
    let expected = problem
        .assemble_bilinear(&KernelAdvDiff2D::new(3.0, [0.0, 0.0]))
        .to_dense();
    for i in 0..selected.nrows() {
        for j in 0..selected.ncols() {
            assert!((selected[(i, j)] - expected[(i, j)]).abs() < 1e-12);
        }
    }
}
