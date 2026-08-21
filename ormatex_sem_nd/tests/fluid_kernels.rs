use ormatex_sem_nd::{
    CellState, EdacNavierStokes2DConfig, FacetCtx, KernelEdacDongOutflow2D,
    KernelEdacMomentumConvection2D, KernelEdacMomentumConvectionSplit2D, KernelEdacNavierStokes2D,
    KernelEdacPressureAdvection2D, KernelEdacPressureAdvectionSplit2D,
    KernelEdacPressureDiffusion2D, KernelEdacPressureDivergence2D, KernelEdacPressureGradient2D,
    KernelEdacViscousStress2D, LocalCtx, ResidualKernel, ResidualKernelSum, SmagorinskyLilly2D,
    StateBoundaryIntegrator,
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

fn facet_context() -> FacetCtx<'static> {
    let weights = Box::leak(vec![1.0].into_boxed_slice());
    let jdet = Box::leak(vec![1.0].into_boxed_slice());
    let points = Box::leak(vec![0.0, 0.0].into_boxed_slice());
    let values = Box::leak(vec![1.0].into_boxed_slice());
    let normal = Box::leak(vec![1.0, 0.0].into_boxed_slice());
    FacetCtx {
        time: 0.0,
        facet: Default::default(),
        tdim: 1,
        gdim: 2,
        ncomp: 1,
        npts: 1,
        ndofs: 1,
        wts: weights,
        jfacet_det: jdet,
        points,
        normal,
        values,
        grads: &[],
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

#[test]
fn decomposed_edac_matches_fused_at_every_local_block() {
    let ctx = context();
    let state = state([1.1, -0.2, 0.3], [0.4, 0.2, -0.3, 0.5, 0.7, -0.1]);
    let fused = KernelEdacNavierStokes2D::new(1.0, 0.01, 4.0, 0.1);
    let config = EdacNavierStokes2DConfig::new(1.0, 0.01, 4.0, 0.1);
    let composed = ResidualKernelSum::from_kernel(KernelEdacMomentumConvection2D::new(config))
        .with(KernelEdacPressureGradient2D::new(config))
        .with(KernelEdacViscousStress2D::new(config))
        .with(KernelEdacPressureDivergence2D::new(config))
        .with(KernelEdacPressureAdvection2D::new(config))
        .with(KernelEdacPressureDiffusion2D::new(config));

    for equation in 0..3 {
        for unknown in 0..3 {
            let fused_jacobian = fused.jacobian_integrand(&ctx, &state, equation, unknown, 0, 0, 0);
            let composed_jacobian =
                composed.jacobian_integrand(&ctx, &state, equation, unknown, 0, 0, 0);
            assert!((fused_jacobian - composed_jacobian).abs() < 1e-12);
        }
        let fused_residual = fused.residual_integrand(&ctx, &state, equation, 0, 0);
        let composed_residual = composed.residual_integrand(&ctx, &state, equation, 0, 0);
        assert!((fused_residual - composed_residual).abs() < 1e-12);
    }
}

#[test]
fn split_edac_jacobians_match_directional_difference() {
    let ctx = context();
    let config = EdacNavierStokes2DConfig::new(1.0, 0.01, 4.0, 0.1);
    let momentum = KernelEdacMomentumConvectionSplit2D::new(config);
    let pressure = KernelEdacPressureAdvectionSplit2D::new(config);
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
    for kernel in [&momentum as &dyn ResidualKernel, &pressure] {
        for equation in 0..3 {
            let finite_difference = (kernel.residual_integrand(&ctx, &perturbed, equation, 0, 0)
                - kernel.residual_integrand(&ctx, &base, equation, 0, 0))
                / eps;
            let analytic = (0..3)
                .map(|unknown| {
                    direction[unknown]
                        * kernel.jacobian_integrand(&ctx, &base, equation, unknown, 0, 0, 0)
                })
                .sum::<f64>();
            assert!(
                (finite_difference - analytic).abs() < 1e-6,
                "split Jacobian mismatch for equation {equation}: {finite_difference} != {analytic}"
            );
        }
    }
}

#[test]
fn dong_outflow_jacobian_matches_directional_difference() {
    let ctx = facet_context();
    let kernel = KernelEdacDongOutflow2D::new(1.0, 0.1, 1.0);
    let base = state([0.4, 0.2, 0.3], [0.0; 6]);
    let direction = [0.2, -0.3, 0.5];
    let eps = 1e-7;
    let perturbed = state(
        [
            0.4 + eps * direction[0],
            0.2 + eps * direction[1],
            0.3 + eps * direction[2],
        ],
        [0.0; 6],
    );
    for equation in 0..3 {
        let finite_difference = (kernel.residual_integrand(&ctx, &perturbed, equation, 0, 0)
            - kernel.residual_integrand(&ctx, &base, equation, 0, 0))
            / eps;
        let analytic = (0..3)
            .map(|unknown| {
                direction[unknown]
                    * kernel.jacobian_integrand(&ctx, &base, equation, unknown, 0, 0, 0)
            })
            .sum::<f64>();
        assert!(
            (finite_difference - analytic).abs() < 1e-6,
            "Dong Jacobian mismatch for equation {equation}: {finite_difference} != {analytic}"
        );
    }
}
