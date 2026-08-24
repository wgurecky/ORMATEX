use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::matrix_free::LinOp;
use faer::prelude::*;
use ndelement::types::ReferenceCellType;
use ndmesh::shapes::unit_square;
use ormatex::ode_sys::OdeSys;
use ormatex_sem_nd::{
    DofReduction2D, FieldRegistry, KernelEdacDongOutflow2D, KernelEdacNavierStokes2D,
    KernelEdacSplitBoundaryFlux2D, SEM2DProblem, StateBoundaryTerms,
};

#[path = "../examples/support/edac.rs"]
mod edac;
#[path = "../examples/support/linear_system.rs"]
mod linear_system;

use edac::{FluidSystem, JacobianBackend};

fn apply(operator: &dyn LinOp<f64>, direction: MatRef<'_, f64>) -> Mat<f64> {
    let mut action = Mat::zeros(operator.nrows(), direction.ncols());
    let mut scratch = MemBuffer::new(StackReq::empty());
    operator.apply(
        action.as_mut(),
        direction,
        faer::get_global_parallelism(),
        MemStack::new(&mut scratch),
    );
    action
}

#[test]
fn assembled_edac_jacobian_includes_state_boundary_terms() {
    let problem = SEM2DProblem::new(
        unit_square(1, 1, ReferenceCellType::Quadrilateral, 1),
        1,
        FieldRegistry::new(["u", "v", "p"]),
        DofReduction2D::None,
    );
    let n = problem.system_size();
    let state = Mat::from_fn(n, 1, |row, _| 0.2 + 0.03 * row as f64);
    let direction = Mat::from_fn(n, 1, |row, _| 0.1 * (row + 1) as f64);
    let terms = StateBoundaryTerms::new()
        .with_default(KernelEdacSplitBoundaryFlux2D)
        .with_entities(
            [0],
            KernelEdacDongOutflow2D::new(1.0, 0.1, 1.0).with_split_flux(),
        );

    let matrix_free = FluidSystem::new_with_backend(
        &problem,
        KernelEdacNavierStokes2D::new(1.0, 0.1, 10.0, 0.0),
        JacobianBackend::MatrixFree,
    )
    .with_state_boundary(terms.clone());
    let assembled = FluidSystem::new_with_backend(
        &problem,
        KernelEdacNavierStokes2D::new(1.0, 0.1, 10.0, 0.0),
        JacobianBackend::Assembled,
    )
    .with_state_boundary(terms);

    let matrix_free_action = apply(
        matrix_free.fjac(0.0, state.as_ref()).as_ref(),
        direction.as_ref(),
    );
    let assembled_action = apply(
        assembled.fjac(0.0, state.as_ref()).as_ref(),
        direction.as_ref(),
    );

    for row in 0..n {
        assert!(
            (matrix_free_action[(row, 0)] - assembled_action[(row, 0)]).abs() < 1e-11,
            "EDAC Jacobian action mismatch at row {row}: {} != {}",
            matrix_free_action[(row, 0)],
            assembled_action[(row, 0)]
        );
    }
}
