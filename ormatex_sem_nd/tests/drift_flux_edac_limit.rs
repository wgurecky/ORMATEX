//! Drift-flux EDAC regression: `alpha -> 0` recovers single-phase EDAC.
//!
//! * Kernel level (2D + 1D): with matched phases (`rho_g = rho_l`,
//!   `mu_g = mu_l`) the mixture kernels equal the base EDAC kernels exactly;
//!   with distinct phases and `alpha = 1e-5` they agree to `O(alpha)` and the
//!   buoyant gravity vanishes.
//! * Time-marched 1D pipe: drift-flux mixture velocity with tiny inlet void
//!   matches the base EDAC solution, and stays bounded for `alpha_in = 0.2`.

use faer::matrix_free::LinOp;
use faer::prelude::*;
use ndelement::types::ReferenceCellType;
use ndmesh::shapes::{unit_interval, unit_square};
use ormatex::ode_implicit::DirkIntegrator;
use ormatex::ode_sys::{IntegrateSys, OdeSys};
use ormatex::tableau_implicit::ImplicitBT;
use ormatex_sem_nd::{
    BilinearOps, CellState, ConstantCoefficient, DofReduction1D, DofReduction2D, DriftFlux1DConfig,
    DriftFlux2DConfig, DriftOutflow1D, EdacNavierStokes1DConfig, EdacNavierStokes2DConfig,
    FieldRegistry, ParallelOwnedMinvJacobian, SEM1DProblem, SEM2DProblem, StateBoundaryIntegrator,
    StateTensorBoundaryIntegrator, TensorDriftDirectionalDoNothing2D, TensorDriftFlux1D,
    TensorDriftFlux2D, TensorDriftGravity1D, TensorDriftGravity2D,
    TensorDriftMomentumConvectionSplit1D, TensorDriftMomentumConvectionSplit2D,
    TensorDriftPressureAdvectionSplit1D, TensorDriftPressureAdvectionSplit2D,
    TensorDriftPressureDiffusion1D, TensorDriftPressureDiffusion2D,
    TensorDriftPressureDivergence1D, TensorDriftPressureDivergence2D,
    TensorDriftPressureGradient1D, TensorDriftPressureGradient2D, TensorDriftTurbulentDispersion2D,
    TensorDriftViscousStress1D, TensorDriftViscousStress2D, TensorDriftVoidAdvectionSplit1D,
    TensorDriftVoidAdvectionSplit2D, TensorKernelEdacDirectionalDoNothing2D,
    TensorKernelEdacMomentumConvectionSplit1D, TensorKernelEdacMomentumConvectionSplit2D,
    TensorKernelEdacPressureAdvectionSplit1D, TensorKernelEdacPressureAdvectionSplit2D,
    TensorKernelEdacPressureDiffusion1D, TensorKernelEdacPressureDiffusion2D,
    TensorKernelEdacPressureDivergence1D, TensorKernelEdacPressureDivergence2D,
    TensorKernelEdacPressureGradient1D, TensorKernelEdacPressureGradient2D,
    TensorKernelEdacViscousStress1D, TensorKernelEdacViscousStress2D, TensorResidualKernel,
    TensorResidualKernelSet, TensorResidualKernelSum,
};

#[path = "../examples/support/linear_system.rs"]
mod linear_system;
use linear_system::lumped_inverse_mass;

// ---------------------------------------------------------------------------
// Pointwise fixtures (single quadrature point, leaked for 'static).

fn drift_state_2d(u: f64, v: f64, p: f64, a: f64) -> CellState<'static> {
    CellState {
        nfields: 4,
        npts: 1,
        gdim: 2,
        values: Box::leak(Box::new([u, v, p, a])),
        grads: Box::leak(Box::new([0.4, -0.2, 0.3, 0.5, -0.1, 0.2, 0.05, -0.05])),
        field_indices: &[],
    }
}

fn base_state_2d(u: f64, v: f64, p: f64) -> CellState<'static> {
    CellState {
        nfields: 3,
        npts: 1,
        gdim: 2,
        values: Box::leak(Box::new([u, v, p])),
        grads: Box::leak(Box::new([0.4, -0.2, 0.3, 0.5, -0.1, 0.2])),
        field_indices: &[],
    }
}

fn tensor_ctx_2d() -> ormatex_sem_nd::TensorCtx<'static> {
    ormatex_sem_nd::TensorCtx {
        time: 0.0,
        cell: Default::default(),
        n1d: 2,
        npts: 1,
        wts: Box::leak(vec![1.0].into_boxed_slice()),
        jdets: Box::leak(vec![0.25].into_boxed_slice()),
        wdet: Box::leak(vec![0.25].into_boxed_slice()),
        points: Box::leak(vec![0.5, 0.5].into_boxed_slice()),
        differentiation: Box::leak(vec![-0.5, 0.5, -0.5, 0.5].into_boxed_slice()),
        q_to_local: Box::leak(vec![0].into_boxed_slice()),
        jinv: Box::leak(vec![2.0, 0.0, 0.0, 2.0].into_boxed_slice()),
        cell_size: 0.5,
    }
}

fn drift_state_1d(u: f64, p: f64, a: f64) -> CellState<'static> {
    CellState {
        nfields: 3,
        npts: 1,
        gdim: 1,
        values: Box::leak(Box::new([u, p, a])),
        grads: Box::leak(Box::new([0.4, -0.2, 0.05])),
        field_indices: &[],
    }
}

fn base_state_1d(u: f64, p: f64) -> CellState<'static> {
    CellState {
        nfields: 2,
        npts: 1,
        gdim: 1,
        values: Box::leak(Box::new([u, p])),
        grads: Box::leak(Box::new([0.4, -0.2])),
        field_indices: &[],
    }
}

fn tensor_ctx_1d() -> ormatex_sem_nd::TensorCtx<'static> {
    ormatex_sem_nd::TensorCtx {
        time: 0.0,
        cell: Default::default(),
        n1d: 2,
        npts: 1,
        wts: Box::leak(vec![1.0].into_boxed_slice()),
        jdets: Box::leak(vec![0.5].into_boxed_slice()),
        wdet: Box::leak(vec![0.5].into_boxed_slice()),
        points: Box::leak(vec![0.5].into_boxed_slice()),
        differentiation: Box::leak(vec![-1.0, 1.0, -1.0, 1.0].into_boxed_slice()),
        q_to_local: Box::leak(vec![0].into_boxed_slice()),
        jinv: Box::leak(vec![2.0].into_boxed_slice()),
        cell_size: 0.5,
    }
}

// Matched-phase configs: mixture == single phase exactly.
fn drift_2d_matched() -> DriftFlux2DConfig {
    DriftFlux2DConfig::new(1.0, 1.0, 0.01, 0.01, 4.0, 0.0)
}

fn base_2d() -> EdacNavierStokes2DConfig {
    EdacNavierStokes2DConfig::new(1.0, 0.01, 4.0, 0.0)
}

fn drift_1d_matched() -> DriftFlux1DConfig {
    DriftFlux1DConfig::new(1.0, 1.0, 0.01, 0.01, 10.0)
}

fn base_1d() -> EdacNavierStokes1DConfig {
    EdacNavierStokes1DConfig::new(1.0, 0.01, 10.0)
}

fn assert_triple_close(a: [f64; 3], b: [f64; 3], tol: f64, what: &str) {
    for i in 0..3 {
        assert!(
            (a[i] - b[i]).abs() < tol,
            "{what} slot {i} mismatch: {} != {}",
            a[i],
            b[i]
        );
    }
}

// ---------------------------------------------------------------------------
// 2D kernel-level limit.

#[test]
fn drift_2d_matches_edac_for_matched_phases() {
    let ctx = tensor_ctx_2d();
    let drift = drift_state_2d(0.7, -0.3, 0.5, 0.2);
    let base = base_state_2d(0.7, -0.3, 0.5);
    let dd = drift_state_2d(0.1, 0.2, -0.1, 0.05);
    let db = base_state_2d(0.1, 0.2, -0.1);
    let dc = drift_2d_matched();
    let bc = base_2d();

    for eq in 0..2 {
        assert_triple_close(
            TensorDriftMomentumConvectionSplit2D::new(dc).tensor_residual(&ctx, &drift, eq, 0),
            TensorKernelEdacMomentumConvectionSplit2D::new(bc).tensor_residual(&ctx, &base, eq, 0),
            1e-12,
            "momentum convection residual",
        );
        assert_triple_close(
            TensorDriftMomentumConvectionSplit2D::new(dc)
                .tensor_jacobian_action(&ctx, &drift, &dd, eq, 0),
            TensorKernelEdacMomentumConvectionSplit2D::new(bc)
                .tensor_jacobian_action(&ctx, &base, &db, eq, 0),
            1e-12,
            "momentum convection action",
        );
        assert_triple_close(
            TensorDriftPressureGradient2D::new(dc).tensor_residual(&ctx, &drift, eq, 0),
            TensorKernelEdacPressureGradient2D::new(bc).tensor_residual(&ctx, &base, eq, 0),
            1e-12,
            "pressure gradient residual",
        );
        assert_triple_close(
            TensorDriftViscousStress2D::new(dc).tensor_residual(&ctx, &drift, eq, 0),
            TensorKernelEdacViscousStress2D::new(bc).tensor_residual(&ctx, &base, eq, 0),
            1e-12,
            "viscous stress residual",
        );
        assert_triple_close(
            TensorDriftViscousStress2D::new(dc).tensor_jacobian_action(&ctx, &drift, &dd, eq, 0),
            TensorKernelEdacViscousStress2D::new(bc)
                .tensor_jacobian_action(&ctx, &base, &db, eq, 0),
            1e-12,
            "viscous stress action",
        );
        // Buoyant gravity is exactly zero for matched phases.
        assert_triple_close(
            TensorDriftGravity2D::new(dc).tensor_residual(&ctx, &drift, eq, 0),
            [0.0; 3],
            1e-14,
            "gravity residual",
        );
    }

    assert_triple_close(
        TensorDriftPressureAdvectionSplit2D::new(dc).tensor_residual(&ctx, &drift, 2, 0),
        TensorKernelEdacPressureAdvectionSplit2D::new(bc).tensor_residual(&ctx, &base, 2, 0),
        1e-12,
        "pressure advection residual",
    );
    assert_triple_close(
        TensorDriftPressureDivergence2D::new(dc).tensor_residual(&ctx, &drift, 2, 0),
        TensorKernelEdacPressureDivergence2D::new(bc).tensor_residual(&ctx, &base, 2, 0),
        1e-12,
        "pressure divergence residual",
    );
    assert_triple_close(
        TensorDriftPressureDivergence2D::new(dc).tensor_jacobian_action(&ctx, &drift, &dd, 2, 0),
        TensorKernelEdacPressureDivergence2D::new(bc)
            .tensor_jacobian_action(&ctx, &base, &db, 2, 0),
        1e-12,
        "pressure divergence action",
    );
    assert_triple_close(
        TensorDriftPressureDiffusion2D::new(dc).tensor_residual(&ctx, &drift, 2, 0),
        TensorKernelEdacPressureDiffusion2D::new(bc).tensor_residual(&ctx, &base, 2, 0),
        1e-12,
        "pressure diffusion residual",
    );

    // Directional outflow on momentum/pressure matches the base kernel.
    let fctx = ormatex_sem_nd::TensorFacetCtx {
        time: 0.0,
        facet: Default::default(),
        npts: 1,
        wts: Box::leak(vec![1.0].into_boxed_slice()),
        jfacet_det: Box::leak(vec![1.0].into_boxed_slice()),
        points: Box::leak(vec![1.0, 0.5].into_boxed_slice()),
        normal: Box::leak(vec![1.0, 0.0].into_boxed_slice()),
    };
    for eq in 0..3 {
        let got = TensorDriftDirectionalDoNothing2D::new(1.0)
            .with_split_flux()
            .tensor_residual(&fctx, &drift, eq, 0);
        let want = TensorKernelEdacDirectionalDoNothing2D::new(1.0)
            .with_split_flux()
            .tensor_residual(&fctx, &base, eq, 0);
        assert!(
            (got - want).abs() < 1e-12,
            "directional outflow residual eq {eq}: {got} != {want}"
        );
    }
}

#[test]
fn drift_2d_corrections_scale_with_alpha() {
    // Distinct phases, tiny void: mixture terms differ from base by O(alpha).
    let ctx = tensor_ctx_2d();
    let dc = DriftFlux2DConfig::new(1.0, 0.1, 0.01, 0.001, 4.0, 0.0);
    let bc = base_2d();
    let drift = drift_state_2d(0.7, -0.3, 0.5, 1e-5);
    let base = base_state_2d(0.7, -0.3, 0.5);
    for eq in 0..2 {
        let got = TensorDriftPressureGradient2D::new(dc).tensor_residual(&ctx, &drift, eq, 0);
        let want = TensorKernelEdacPressureGradient2D::new(bc).tensor_residual(&ctx, &base, eq, 0);
        assert!(
            (got[0] - want[0]).abs() < 1e-4,
            "pressure gradient O(alpha) violated: {} != {}",
            got[0],
            want[0]
        );
        let g = TensorDriftGravity2D::new(dc).tensor_residual(&ctx, &drift, eq, 0);
        assert!(
            g[0].abs() < 0.2,
            "gravity should be small at low void: {}",
            g[0]
        );
    }
    let got = TensorDriftPressureDivergence2D::new(dc).tensor_residual(&ctx, &drift, 2, 0);
    let want = TensorKernelEdacPressureDivergence2D::new(bc).tensor_residual(&ctx, &base, 2, 0);
    assert!(
        (got[0] - want[0]).abs() / want[0].abs().max(1e-12) < 1e-3,
        "pressure divergence O(alpha) violated"
    );
    // Void advection + drift + dispersion are finite and void-owned only.
    let kernels: Vec<Box<dyn TensorResidualKernel<2>>> = vec![
        Box::new(TensorDriftVoidAdvectionSplit2D::new(dc)),
        Box::new(TensorDriftFlux2D::new(dc)),
        Box::new(TensorDriftTurbulentDispersion2D::new(dc, 1e-3)),
    ];
    for k in &kernels {
        assert!(!k.owns_equation(0) && k.owns_equation(3));
        for v in k.tensor_residual(&ctx, &drift, 3, 0) {
            assert!(v.is_finite());
        }
    }
}

#[test]
fn drift_flux_action_matches_finite_difference() {
    // The hindered flux is non-monotone, so the exact (sign-correct)
    // derivative is load-bearing: check it against finite differences.
    let eps = 1e-7;
    let dc = DriftFlux2DConfig::new(1.0, 0.1, 0.01, 0.001, 4.0, 0.0);
    let kernel = TensorDriftFlux2D::new(dc);
    let ctx = tensor_ctx_2d();
    for alpha in [0.05, 0.2, 0.5, 0.9] {
        let base = drift_state_2d(0.7, -0.3, 0.5, alpha);
        let perturbed = drift_state_2d(0.7, -0.3, 0.5, alpha + eps);
        let direction = drift_state_2d(0.0, 0.0, 0.0, 1.0);
        let fd = (kernel.tensor_residual(&ctx, &perturbed, 3, 0)[1]
            - kernel.tensor_residual(&ctx, &base, 3, 0)[1])
            / eps;
        let analytic = kernel.tensor_jacobian_action(&ctx, &base, &direction, 3, 0)[1];
        assert!(
            (fd - analytic).abs() < 1e-6 * analytic.abs().max(1.0),
            "2D drift action mismatch at alpha={alpha}: {analytic} != {fd}"
        );
    }

    let dc1 = DriftFlux1DConfig::new(1.0, 0.1, 0.01, 0.001, 10.0);
    let kernel1 = TensorDriftFlux1D::new(dc1, ConstantCoefficient(std::f64::consts::FRAC_PI_2));
    let ctx1 = tensor_ctx_1d();
    for alpha in [0.05, 0.2, 0.5, 0.9] {
        let base = drift_state_1d(0.7, 0.5, alpha);
        let perturbed = drift_state_1d(0.7, 0.5, alpha + eps);
        let direction = drift_state_1d(0.0, 0.0, 1.0);
        let fd = (kernel1.tensor_residual(&ctx1, &perturbed, 2, 0)[1]
            - kernel1.tensor_residual(&ctx1, &base, 2, 0)[1])
            / eps;
        let analytic = kernel1.tensor_jacobian_action(&ctx1, &base, &direction, 2, 0)[1];
        assert!(
            (fd - analytic).abs() < 1e-6 * analytic.abs().max(1.0),
            "1D drift action mismatch at alpha={alpha}: {analytic} != {fd}"
        );
    }
}

// ---------------------------------------------------------------------------
// 1D kernel-level limit.

#[test]
fn drift_1d_matches_edac_for_matched_phases() {
    let ctx = tensor_ctx_1d();
    let drift = drift_state_1d(0.7, 0.5, 0.2);
    let base = base_state_1d(0.7, 0.5);
    let dd = drift_state_1d(0.1, -0.1, 0.05);
    let db = base_state_1d(0.1, -0.1);
    let dc = drift_1d_matched();
    let bc = base_1d();

    assert_triple_close(
        TensorDriftMomentumConvectionSplit1D::new(dc).tensor_residual(&ctx, &drift, 0, 0),
        TensorKernelEdacMomentumConvectionSplit1D::new(bc).tensor_residual(&ctx, &base, 0, 0),
        1e-12,
        "1d momentum convection",
    );
    assert_triple_close(
        TensorDriftPressureGradient1D::new(dc).tensor_residual(&ctx, &drift, 0, 0),
        TensorKernelEdacPressureGradient1D::new(bc).tensor_residual(&ctx, &base, 0, 0),
        1e-12,
        "1d pressure gradient",
    );
    assert_triple_close(
        TensorDriftViscousStress1D::new(dc).tensor_residual(&ctx, &drift, 0, 0),
        TensorKernelEdacViscousStress1D::new(bc).tensor_residual(&ctx, &base, 0, 0),
        1e-12,
        "1d viscous stress",
    );
    assert_triple_close(
        TensorDriftViscousStress1D::new(dc).tensor_jacobian_action(&ctx, &drift, &dd, 0, 0),
        TensorKernelEdacViscousStress1D::new(bc).tensor_jacobian_action(&ctx, &base, &db, 0, 0),
        1e-12,
        "1d viscous action",
    );
    assert_triple_close(
        TensorDriftPressureAdvectionSplit1D::new(dc).tensor_residual(&ctx, &drift, 1, 0),
        TensorKernelEdacPressureAdvectionSplit1D::new(bc).tensor_residual(&ctx, &base, 1, 0),
        1e-12,
        "1d pressure advection",
    );
    assert_triple_close(
        TensorDriftPressureDivergence1D::new(dc).tensor_residual(&ctx, &drift, 1, 0),
        TensorKernelEdacPressureDivergence1D::new(bc).tensor_residual(&ctx, &base, 1, 0),
        1e-12,
        "1d pressure divergence",
    );
    assert_triple_close(
        TensorDriftPressureDiffusion1D::new(dc).tensor_residual(&ctx, &drift, 1, 0),
        TensorKernelEdacPressureDiffusion1D::new(bc).tensor_residual(&ctx, &base, 1, 0),
        1e-12,
        "1d pressure diffusion",
    );
    assert_triple_close(
        TensorDriftGravity1D::horizontal(dc).tensor_residual(&ctx, &drift, 0, 0),
        [0.0; 3],
        1e-14,
        "1d horizontal gravity",
    );
    // Horizontal drift flux vanishes; tilted pipe gives finite up-pipe flux.
    assert_triple_close(
        TensorDriftFlux1D::horizontal(dc).tensor_residual(&ctx, &drift, 2, 0),
        [0.0; 3],
        1e-14,
        "1d horizontal drift",
    );
    // Tilted drift needs distinct phases (matched phases have zero buoyancy).
    let buoyant = DriftFlux1DConfig::new(1.0, 0.1, 0.01, 0.001, 10.0);
    let tilted = TensorDriftFlux1D::new(buoyant, ConstantCoefficient(std::f64::consts::FRAC_PI_2));
    let f = tilted.tensor_residual(&ctx, &drift, 2, 0);
    assert!(f[1] < 0.0 && f[1].is_finite(), "tilted drift flux: {f:?}");
    let g = TensorDriftGravity1D::new(buoyant, ConstantCoefficient(std::f64::consts::FRAC_PI_2))
        .tensor_residual(&ctx, &drift_state_1d(0.7, 0.5, 0.3), 0, 0);
    assert!(g[0].is_finite());
}

fn outflow_facet() -> ormatex_sem_nd::FacetCtx<'static> {
    ormatex_sem_nd::FacetCtx {
        time: 0.0,
        facet: Default::default(),
        tdim: 0,
        gdim: 1,
        ncomp: 1,
        npts: 1,
        ndofs: 1,
        wts: Box::leak(vec![1.0].into_boxed_slice()),
        jfacet_det: Box::leak(vec![1.0].into_boxed_slice()),
        points: Box::leak(vec![1.0].into_boxed_slice()),
        normal: Box::leak(vec![1.0].into_boxed_slice()),
        values: Box::leak(vec![1.0].into_boxed_slice()),
        grads: &[],
    }
}

#[test]
fn drift_outflow_matches_split_plus_drift_flux() {
    // Right-endpoint pairing: split fluxes plus the axial drift flux, with an
    // exact Jacobian (finite-difference checked, including the hindered-flux
    // sign past alpha = 4/11).
    let config = DriftFlux1DConfig::new(1.0, 0.1, 0.01, 0.001, 10.0);
    let kernel = DriftOutflow1D::new(config, 1.0);
    let ctx = outflow_facet();
    let (u, p, a) = (0.8, 0.4, 0.25);
    let state = drift_state_1d(u, p, a);
    let c0 = config.distribution_parameter();
    let f = config
        .ishii_zuber
        .drift_flux(a, config.rho_l, config.rho_g, config.gravity);
    assert_eq!(kernel.nfields(), 3);
    for (eq, want) in [
        (0, 0.5 * u * u),
        (1, 0.5 * u * p),
        (2, 0.5 * c0 * u * a + f),
    ] {
        let got = kernel.residual_integrand(&ctx, &state, eq, 0, 0);
        assert!(
            (got - want).abs() < 1e-12,
            "outflow residual eq {eq}: {got} != {want}"
        );
    }
    let eps = 1e-7;
    for eq in 0..3 {
        for unknown in 0..3 {
            let mut pert = [u, p, a];
            pert[unknown] += eps;
            let perturbed = drift_state_1d(pert[0], pert[1], pert[2]);
            let fd = (kernel.residual_integrand(&ctx, &perturbed, eq, 0, 0)
                - kernel.residual_integrand(&ctx, &state, eq, 0, 0))
                / eps;
            // Unit trial/test basis, so the integrand is the derivative itself.
            let action: f64 = kernel.jacobian_integrand(&ctx, &state, eq, unknown, 0, 0, 0);
            assert!(
                (fd - action).abs() < 1e-6 * action.abs().max(1.0),
                "outflow Jacobian mismatch eq {eq} unknown {unknown}: {action} != {fd}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Time-marched 1D pipe regression.

type IntervalMesh = ndmesh::SingleElementMesh<
    f64,
    ndelement::ciarlet::CiarletElement<f64, ndelement::map::IdentityMap, f64>,
>;

fn drift_kernel_1d(config: DriftFlux1DConfig) -> TensorResidualKernelSet<'static, 1> {
    TensorResidualKernelSet::from_kernel(
        TensorResidualKernelSum::from_kernel(TensorDriftMomentumConvectionSplit1D::new(config))
            .with(TensorDriftPressureGradient1D::new(config))
            .with(TensorDriftViscousStress1D::new(config))
            .with(TensorDriftPressureDivergence1D::new(config))
            .with(TensorDriftPressureAdvectionSplit1D::new(config))
            .with(TensorDriftPressureDiffusion1D::new(config))
            .with(TensorDriftVoidAdvectionSplit1D::new(config))
            .with(TensorDriftFlux1D::horizontal(config))
            .with(TensorDriftGravity1D::horizontal(config)),
    )
}

fn base_kernel_1d(config: EdacNavierStokes1DConfig) -> TensorResidualKernelSet<'static, 1> {
    TensorResidualKernelSet::from_kernel(
        TensorResidualKernelSum::from_kernel(TensorKernelEdacMomentumConvectionSplit1D::new(
            config,
        ))
        .with(TensorKernelEdacPressureGradient1D::new(config))
        .with(TensorKernelEdacViscousStress1D::new(config))
        .with(TensorKernelEdacPressureDivergence1D::new(config))
        .with(TensorKernelEdacPressureAdvectionSplit1D::new(config))
        .with(TensorKernelEdacPressureDiffusion1D::new(config)),
    )
}

struct PipeSystem<'a> {
    problem: &'a SEM1DProblem<IntervalMesh>,
    kernel: TensorResidualKernelSet<'a, 1>,
    m_inv: Vec<f64>,
}

impl<'a> OdeSys<'a> for PipeSystem<'a> {
    fn frhs(&self, t: f64, state: MatRef<f64>) -> Mat<f64> {
        let residual = self
            .problem
            .tensor_residual_operator(&self.kernel)
            .at_time(t)
            .residual(state);
        Mat::from_fn(self.m_inv.len(), 1, |row, _| {
            -self.m_inv[row] * residual[row]
        })
    }

    fn fjac<'b>(&'a self, t: f64, state: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
        // ponytail: assembled direct solve; the coarse test mesh makes this
        // far cheaper than matrix-free Krylov.
        Box::new(ParallelOwnedMinvJacobian::new(
            self.problem
                .tensor_residual_operator(&self.kernel)
                .at_time(t)
                .assemble_jacobian(state),
            &self.m_inv,
        ))
    }
}

fn march(system: &PipeSystem<'_>, steps: usize, dt: f64, y0: MatRef<'_, f64>) -> Mat<f64> {
    let mut solver = DirkIntegrator::new(0.0, y0, ImplicitBT::implicit_euler(), 1e-10, 1e-10);
    for _ in 0..steps {
        let step = solver.step(system, dt).unwrap();
        solver.accept_step(step);
    }
    solver.state()
}

#[test]
fn drift_pipe_low_void_matches_edac_velocity() {
    let nx = 16;
    let inlet_velocity = 0.1;
    let tiny_void = 1e-5;
    // Matched phases + horizontal pipe: mixture == single phase up to O(alpha).
    let drift_config = DriftFlux1DConfig::new(1.0, 1.0, 0.01, 0.01, 10.0).with_gravity(0.0);
    let base_config = EdacNavierStokes1DConfig::new(1.0, 0.01, 10.0);

    let drift_problem = SEM1DProblem::new(
        unit_interval(nx, 1),
        2,
        FieldRegistry::new(["u", "p", "alpha"]),
        DofReduction1D::FieldSpecific {
            reductions: vec![
                DofReduction1D::Dirichlet {
                    facets: vec![(0, inlet_velocity)],
                },
                DofReduction1D::Dirichlet {
                    facets: vec![(nx, 0.0)],
                },
                DofReduction1D::Dirichlet {
                    facets: vec![(0, tiny_void)],
                },
            ],
        },
    );
    let drift_system = PipeSystem {
        problem: &drift_problem,
        kernel: drift_kernel_1d(drift_config),
        m_inv: lumped_inverse_mass(drift_problem.assemble_lumped_mass().as_ref()),
    };
    let mut y0 = Mat::<f64>::zeros(drift_problem.system_size(), 1);
    let a_field = drift_problem.field_id("alpha").unwrap();
    let a_offset = drift_problem.field_offset(a_field);
    for row in 0..drift_problem.field_reduced_size(a_field) {
        y0[(a_offset + row, 0)] = tiny_void;
    }
    let drift_state = march(&drift_system, 200, 0.2, y0.as_ref());

    let base_problem = SEM1DProblem::new(
        unit_interval(nx, 1),
        2,
        FieldRegistry::new(["u", "p"]),
        DofReduction1D::FieldSpecific {
            reductions: vec![
                DofReduction1D::Dirichlet {
                    facets: vec![(0, inlet_velocity)],
                },
                DofReduction1D::Dirichlet {
                    facets: vec![(nx, 0.0)],
                },
            ],
        },
    );
    let base_system = PipeSystem {
        problem: &base_problem,
        kernel: base_kernel_1d(base_config),
        m_inv: lumped_inverse_mass(base_problem.assemble_lumped_mass().as_ref()),
    };
    let y0_base = Mat::<f64>::zeros(base_problem.system_size(), 1);
    let base_state = march(&base_system, 200, 0.2, y0_base.as_ref());

    let u_drift = drift_problem
        .field_values("u", drift_state.as_ref())
        .unwrap();
    let u_base = base_problem.field_values("u", base_state.as_ref()).unwrap();
    assert_eq!(u_drift.values.len(), u_base.values.len());
    let mut max_diff = 0.0f64;
    for (a, b) in u_drift.values.iter().zip(&u_base.values) {
        assert!(a.is_finite() && b.is_finite());
        max_diff = max_diff.max((a - b).abs());
    }
    assert!(
        max_diff < 5e-3,
        "low-void drift velocity must match EDAC: {max_diff}"
    );
    let alpha = drift_problem
        .field_values("alpha", drift_state.as_ref())
        .unwrap();
    for v in &alpha.values {
        // No bound-preserving limiter: allow small high-order undershoots.
        assert!(
            v.is_finite() && *v > -1e-4 && *v < 1e-3,
            "low-void alpha must stay tiny and bounded: {v}"
        );
    }
}

#[test]
fn drift_pipe_void_stays_bounded() {
    let nx = 16;
    let inlet_velocity = 0.5;
    let inlet_void = 0.2;
    let config = DriftFlux1DConfig::new(1.0, 0.1, 0.01, 0.001, 10.0).with_gravity(0.0);
    let problem = SEM1DProblem::new(
        unit_interval(nx, 1),
        2,
        FieldRegistry::new(["u", "p", "alpha"]),
        DofReduction1D::FieldSpecific {
            reductions: vec![
                DofReduction1D::Dirichlet {
                    facets: vec![(0, inlet_velocity)],
                },
                DofReduction1D::Dirichlet {
                    facets: vec![(nx, 0.0)],
                },
                DofReduction1D::Dirichlet {
                    facets: vec![(0, inlet_void)],
                },
            ],
        },
    );
    let system = PipeSystem {
        problem: &problem,
        kernel: drift_kernel_1d(config),
        m_inv: lumped_inverse_mass(problem.assemble_lumped_mass().as_ref()),
    };
    let mut y0 = Mat::<f64>::zeros(problem.system_size(), 1);
    let a_field = problem.field_id("alpha").unwrap();
    let a_offset = problem.field_offset(a_field);
    for row in 0..problem.field_reduced_size(a_field) {
        y0[(a_offset + row, 0)] = inlet_void;
    }
    let state = march(&system, 200, 0.1, y0.as_ref());
    let alpha = problem.field_values("alpha", state.as_ref()).unwrap();
    let u = problem.field_values("u", state.as_ref()).unwrap();
    for v in alpha.values.iter().chain(u.values.iter()) {
        assert!(v.is_finite(), "non-finite drift-pipe state: {v}");
    }
    for v in &alpha.values {
        assert!(
            *v > -5e-2 && *v < 1.0 + 5e-2,
            "void fraction must stay bounded and physical: {v}"
        );
    }
    // Void reaches the interior (advected from the inlet, not stuck at zero).
    let max_alpha = alpha.values.iter().fold(0.0f64, |m, &v| m.max(v));
    assert!(
        max_alpha > 0.05,
        "inlet void should advect into the domain, max alpha = {max_alpha}"
    );
}

// ---------------------------------------------------------------------------
// 2D assembled limit on a small mesh.

#[test]
fn drift_2d_assembled_matches_edac_for_matched_phases() {
    use ormatex_sem_nd::TensorResidualOps;
    let problem = ormatex_sem_nd::SEM2DProblem::new(
        unit_square(1, 1, ReferenceCellType::Quadrilateral, 1),
        1,
        FieldRegistry::new(["u", "v", "p", "alpha"]),
        DofReduction2D::None,
    );
    let n = problem.system_size();
    // Nontrivial state with small void; grads exercise convection/viscous.
    let state = Mat::from_fn(n, 1, |row, _| 0.2 + 0.03 * row as f64);
    let dc = drift_2d_matched();
    let drift_sum =
        TensorResidualKernelSum::from_kernel(TensorDriftMomentumConvectionSplit2D::new(dc))
            .with(TensorDriftPressureGradient2D::new(dc))
            .with(TensorDriftViscousStress2D::new(dc))
            .with(TensorDriftPressureDivergence2D::new(dc))
            .with(TensorDriftPressureAdvectionSplit2D::new(dc))
            .with(TensorDriftPressureDiffusion2D::new(dc));
    let bc = base_2d();
    let base_sum =
        TensorResidualKernelSum::from_kernel(TensorKernelEdacMomentumConvectionSplit2D::new(bc))
            .with(TensorKernelEdacPressureGradient2D::new(bc))
            .with(TensorKernelEdacViscousStress2D::new(bc))
            .with(TensorKernelEdacPressureDivergence2D::new(bc))
            .with(TensorKernelEdacPressureAdvectionSplit2D::new(bc))
            .with(TensorKernelEdacPressureDiffusion2D::new(bc));
    let drift_res = problem.assemble_tensor_residual(0.0, &drift_sum, state.as_ref());
    let base_res = problem.assemble_tensor_residual(0.0, &base_sum, state.as_ref());
    // Layout differs (4 vs 3 fields); compare per-field blocks.
    for (name, dfield, bfield) in [("u", 0, 0), ("v", 1, 1), ("p", 2, 2)] {
        let dn = problem.field_reduced_size(dfield);
        let bn = problem.field_reduced_size(bfield);
        assert_eq!(dn, bn, "{name} block size mismatch");
        let doff = problem.field_offset(dfield);
        let boff = problem.field_offset(bfield);
        for i in 0..dn {
            assert!(
                (drift_res[doff + i] - base_res[boff + i]).abs() < 1e-9,
                "{name}[{i}] mismatch: {} != {}",
                drift_res[doff + i],
                base_res[boff + i]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Phase-velocity post-processing helpers.

#[test]
fn phase_velocities_satisfy_slip_and_mixture_identities() {
    let config = DriftFlux2DConfig::new(1.0, 0.1, 0.01, 0.001, 4.0, 0.0);
    // Slip identity: no drift (zero gravity) and C0 = 1 gives vapor == mixture.
    let still = DriftFlux2DConfig::new(1.0, 0.1, 0.01, 0.001, 4.0, 0.0).with_gravity_mag(0.0);
    for (alpha, um) in [
        (0.0, [1.0, 0.5]),
        (0.2, [1.0, -0.3]),
        (0.7, [-0.4, 0.9]),
        (1.0, [0.3, 0.2]),
    ] {
        let vapor = still.vapor_velocity(alpha, um);
        assert!(
            (vapor[0] - um[0]).abs() < 1e-12 && (vapor[1] - um[1]).abs() < 1e-12,
            "still vapor must equal mixture at alpha={alpha}"
        );
        // Mixture-consistency identity at full buoyancy.
        let vapor = config.vapor_velocity(alpha, um);
        let liquid = config.liquid_velocity(alpha, um);
        let rho_m = config.mixture_density(alpha);
        for d in 0..2 {
            let recombined =
                alpha * config.rho_g * vapor[d] + (1.0 - alpha) * config.rho_l * liquid[d];
            assert!(
                (recombined - rho_m * um[d]).abs() < 1e-9,
                "mixture identity violated at alpha={alpha}, dir={d}"
            );
            assert!(vapor[d].is_finite() && liquid[d].is_finite());
        }
    }
    // Pure-gas limit: liquid falls back to the mixture velocity, finite.
    let liquid = config.liquid_velocity(1.0, [2.0, -1.0]);
    assert_eq!(liquid, [2.0, -1.0]);
}

#[test]
fn phase_velocities_1d_follow_pipe_angle() {
    use std::f64::consts::FRAC_PI_2;
    let config = DriftFlux1DConfig::new(1.0, 0.1, 0.01, 0.001, 10.0);
    // Horizontal pipe: no axial drift, so vapor == C0 * u.
    let vapor = config.vapor_velocity_1d(0.3, 1.0, 0.0);
    assert!(
        (vapor - config.distribution_parameter()).abs() < 1e-12,
        "horizontal vapor must equal C0*u"
    );
    // Vertical pipe: drift adds a positive up-pipe component.
    let rising = config.vapor_velocity_1d(0.3, 1.0, FRAC_PI_2);
    assert!(
        rising > vapor && rising.is_finite(),
        "vertical vapor must exceed horizontal"
    );
    // Mixture-consistency identity.
    for alpha in [0.0, 0.2, 0.7, 1.0] {
        let u = 0.8;
        let vapor = config.vapor_velocity_1d(alpha, u, FRAC_PI_2);
        let liquid = config.liquid_velocity_1d(alpha, u, FRAC_PI_2);
        let rho_m = config.mixture_density(alpha);
        let recombined = alpha * config.rho_g * vapor + (1.0 - alpha) * config.rho_l * liquid;
        assert!(
            (recombined - rho_m * u).abs() < 1e-9,
            "1D mixture identity violated at alpha={alpha}"
        );
    }
}
