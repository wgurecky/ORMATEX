use ormatex_sem_nd::{
    dirichlet_values_with_precedence, DofReduction2D, FieldRegistry, QuadMesh, SEM2DProblem,
};

const P: usize = 2;

pub struct BackwardStepCase {
    pub problem: SEM2DProblem<QuadMesh>,
    pub outlet: Vec<usize>,
    pub wall: Vec<usize>,
}

pub fn problem() -> BackwardStepCase {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/navier-stokes/backward_step.msh"
    );
    let data =
        ormatex_sem_nd::gmsh_quad_data(path).expect("failed to load all-quad backward-step mesh");
    let mesh = data.mesh;
    let metadata = data.metadata;
    let inlet = metadata.boundary_facets("inlet");
    let outlet = metadata.boundary_facets("outlet");
    let wall = metadata.boundary_facets("wall");
    assert!(!inlet.is_empty() && !outlet.is_empty() && !wall.is_empty());

    let wall_u: Vec<_> = wall.iter().copied().map(|facet| (facet, 0.0)).collect();
    let inlet_u: Vec<_> = inlet.iter().copied().map(|facet| (facet, 1.0)).collect();
    let inlet_v: Vec<_> = inlet.iter().copied().map(|facet| (facet, 0.0)).collect();
    let u_values = dirichlet_values_with_precedence(&mesh, P, &wall_u, &inlet_u);
    let v_values = dirichlet_values_with_precedence(&mesh, P, &wall_u, &inlet_v);

    let problem = SEM2DProblem::new_with_metadata(
        mesh,
        P,
        FieldRegistry::new(["u", "v", "p"]),
        DofReduction2D::FieldSpecific {
            reductions: vec![
                DofReduction2D::DirichletValues { values: u_values },
                DofReduction2D::DirichletValues { values: v_values },
                DofReduction2D::None,
            ],
        },
        metadata,
    );

    BackwardStepCase {
        problem,
        outlet,
        wall,
    }
}
