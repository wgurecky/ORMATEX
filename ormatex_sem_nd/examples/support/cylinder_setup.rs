use ormatex_sem_nd::{DofReduction2D, FieldRegistry, QuadMesh, SEM2DProblem};

pub struct CylinderCase {
    pub problem: SEM2DProblem<QuadMesh>,
    pub outlet: Vec<usize>,
    pub cylinder: Vec<usize>,
    pub slip_wall: Vec<usize>,
}

fn dirichlet(values: &[(usize, f64)]) -> DofReduction2D {
    DofReduction2D::Dirichlet {
        facets: values.to_vec(),
    }
}

pub fn nearest(positions: &[(f64, f64)], target: (f64, f64)) -> usize {
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

pub fn problem(directional: bool) -> CylinderCase {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/navier-stokes/cylinder.msh"
    );
    let data = ormatex_sem_nd::gmsh_quad_data(path).expect("failed to load all-quad cylinder mesh");
    let mesh = data.mesh;
    let metadata = data.metadata;
    let inlet = metadata.boundary_facets("inlet");
    let outlet = metadata.boundary_facets("outlet");
    let cylinder = metadata.boundary_facets("cylinder");
    assert!(!inlet.is_empty() && !outlet.is_empty() && !cylinder.is_empty());

    let inlet_u: Vec<_> = inlet.iter().copied().map(|facet| (facet, 1.0)).collect();
    let inlet_v: Vec<_> = inlet.iter().copied().map(|facet| (facet, 0.0)).collect();
    let wall_u: Vec<_> = cylinder.iter().copied().map(|facet| (facet, 0.0)).collect();
    let slip_wall: Vec<_> = metadata
        .boundary_facets("bottom")
        .into_iter()
        .chain(metadata.boundary_facets("top"))
        .collect();
    let wall_v: Vec<_> = cylinder
        .iter()
        .copied()
        .chain(slip_wall.iter().copied())
        .map(|facet| (facet, 0.0))
        .collect();
    let outlet_p: Vec<_> = outlet.iter().copied().map(|facet| (facet, 0.0)).collect();

    let problem = SEM2DProblem::new_with_metadata(
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
                if directional {
                    DofReduction2D::None
                } else {
                    dirichlet(&outlet_p)
                },
            ],
        },
        metadata,
    );

    CylinderCase {
        problem,
        outlet,
        cylinder,
        slip_wall,
    }
}
