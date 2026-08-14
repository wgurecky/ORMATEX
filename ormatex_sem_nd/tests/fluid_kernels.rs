use ormatex_sem_nd::{
    CellState, KernelEdacNavierStokes2D, LocalCtx, ResidualKernel, SmagorinskyLilly2D,
};

fn context() -> LocalCtx<'static> {
    let weights = Box::leak(vec![1.0].into_boxed_slice());
    let jdets = Box::leak(vec![2.0].into_boxed_slice());
    let points = Box::leak(vec![0.0, 0.0].into_boxed_slice());
    let values = Box::leak(vec![1.0].into_boxed_slice());
    let grads = Box::leak(vec![0.3, -0.2].into_boxed_slice());
    LocalCtx {
        time: 0.0,
        cell: Default::default(),
        tdim: 2,
        gdim: 2,
        ncomp: 1,
        npts: 1,
        ndofs: 1,
        wts: weights,
        jdets,
        points,
        values,
        grads,
    }
}

fn state(values: [f64; 3], grads: [f64; 6]) -> CellState<'static> {
    CellState {
        nfields: 3,
        npts: 1,
        gdim: 2,
        values: Box::leak(Box::new(values)),
        grads: Box::leak(Box::new(grads)),
    }
}

#[test]
fn smagorinsky_filter_width_is_local_cell_spacing() {
    let ctx = context();
    let model = SmagorinskyLilly2D::new(0.1);
    assert!((model.filter_width(&ctx) - 2.0_f64.sqrt()).abs() < 1e-12);
}

#[test]
fn edac_jacobian_matches_directional_difference() {
    let ctx = context();
    let kernel = KernelEdacNavierStokes2D::new(1.0, 0.01, 4.0, 0.1);
    let base_values = [1.1, -0.2, 0.3];
    let base_grads = [0.4, 0.2, -0.3, 0.5, 0.7, -0.1];
    let direction = [0.2, -0.4, 0.5];
    let basis_grad = [0.3, -0.2];
    let eps = 1e-7;
    let perturbed_values = [
        base_values[0] + eps * direction[0],
        base_values[1] + eps * direction[1],
        base_values[2] + eps * direction[2],
    ];
    let mut perturbed_grads = base_grads;
    for field in 0..3 {
        perturbed_grads[2 * field] += eps * direction[field] * basis_grad[0];
        perturbed_grads[2 * field + 1] += eps * direction[field] * basis_grad[1];
    }
    let base = state(base_values, base_grads);
    let perturbed = state(perturbed_values, perturbed_grads);
    let finite_difference = (kernel.residual_integrand(&ctx, &perturbed, 0, 0, 0)
        - kernel.residual_integrand(&ctx, &base, 0, 0, 0))
        / eps;
    let analytic = (0..3)
        .map(|unknown| {
            direction[unknown] * kernel.jacobian_integrand(&ctx, &base, 0, unknown, 0, 0, 0)
        })
        .sum::<f64>();
    assert!((finite_difference - analytic).abs() < 1e-6);
}
