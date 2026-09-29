//! Free-surface drift-flux boundary regression.
//!
//! * Momentum/pressure rows delegate to the directional-do-nothing outflow
//!   with the liquid reference density, plus a normal penalty
//!   `gamma*(u_l.n)*n` enforcing liquid no-penetration on rows 0..1.
//! * The void row vents `1/2 C0 max(u.n, 0) a + F_vent(a) max(e.n, 0)` with
//!   unhindered degassing flux `F_vent = a*V0` (exact linear Jacobian).
//! * No re-entry: incoming mixture flow contributes zero advective void flux
//!   while drift still vents where the rise vector points out.

mod common;
use common::lanes;

use ormatex_sem_nd::{
    CellState, DriftFlux2DConfig, TensorDriftDirectionalDoNothing2D, TensorDriftFreeSurface2D,
    TensorKernelEdacDirectionalDoNothing2D,
};

fn drift_config() -> DriftFlux2DConfig {
    DriftFlux2DConfig::new(1000.0, 1.2, 1.0e-3, 1.8e-5, 10.0, 0.1)
}

fn drift_state(u: f64, v: f64, p: f64, a: f64) -> CellState<'static> {
    CellState {
        nfields: 4,
        npts: 1,
        gdim: 2,
        values: Box::leak(Box::new([u, v, p, a])),
        grads: Box::leak(Box::new([0.0; 8])),
        field_indices: &[],
    }
}

fn base_state(u: f64, v: f64, p: f64) -> CellState<'static> {
    CellState {
        nfields: 3,
        npts: 1,
        gdim: 2,
        values: Box::leak(Box::new([u, v, p])),
        grads: Box::leak(Box::new([0.0; 6])),
        field_indices: &[],
    }
}

fn facet_ctx(normal: [f64; 2]) -> ormatex_sem_nd::TensorFacetCtx<'static> {
    ormatex_sem_nd::TensorFacetCtx {
        time: 0.0,
        facet: Default::default(),
        npts: 1,
        wts: Box::leak(vec![1.0].into_boxed_slice()),
        jfacet_det: Box::leak(vec![1.0].into_boxed_slice()),
        points: Box::leak(vec![0.05, 0.1].into_boxed_slice()),
        normal: Box::leak(vec![normal[0], normal[1]].into_boxed_slice()),
    }
}

#[test]
fn free_surface_momentum_matches_directional_outflow() {
    let config = drift_config();
    // Zero penalty recovers the directional outflow exactly.
    let kernel = TensorDriftFreeSurface2D::new(config).with_penalty(0.0);
    let ctx = facet_ctx([0.0, 1.0]);
    let drift = drift_state(0.1, 0.2, 5.0, 0.3);
    let base = base_state(0.1, 0.2, 5.0);
    let reference = TensorKernelEdacDirectionalDoNothing2D::new(1000.0).with_split_flux();
    for eq in 0..3 {
        let got = lanes::boundary_residual(&kernel, &ctx, &drift, eq, 0);
        let want = lanes::boundary_residual(&reference, &ctx, &base, eq, 0);
        assert!(
            (got - want).abs() < 1e-12,
            "momentum/pressure row {eq} must delegate: {got} != {want}"
        );
    }
    // Void row on the top facet: split-consistent advection + unhindered degassing.
    let un = 0.2;
    let v0 =
        config
            .ishii_zuber
            .drift_speed(0.0, config.rho_l, config.rho_g, config.gravity_magnitude());
    let want = 0.5 * config.distribution_parameter() * un * 0.3 + 0.3 * v0;
    let got = lanes::boundary_residual(&kernel, &ctx, &drift, 3, 0);
    assert!(
        (got - want).abs() < 1e-12,
        "void vent flux mismatch: {got} != {want}"
    );
    // Same delegation for the drift 4-field directional kernel rows 0..2.
    let drift_out = TensorDriftDirectionalDoNothing2D::new(1000.0).with_split_flux();
    for eq in 0..3 {
        let got = lanes::boundary_residual(&kernel, &ctx, &drift, eq, 0);
        let want = lanes::boundary_residual(&drift_out, &ctx, &drift, eq, 0);
        assert!(
            (got - want).abs() < 1e-9 * want.abs().max(1.0),
            "free-surface row {eq} must match drift outflow: {got} != {want}"
        );
    }
}

#[test]
fn free_surface_liquid_penalty_matches_closure() {
    let config = drift_config();
    let penalty = 1.0e5;
    let kernel = TensorDriftFreeSurface2D::new(config).with_penalty(penalty);
    let zero = TensorDriftFreeSurface2D::new(config).with_penalty(0.0);
    let ctx = facet_ctx([0.0, 1.0]);
    // Tangential (x) row sees no penalty at the top facet (n_x = 0).
    let drift = drift_state(0.1, 0.2, 5.0, 0.3);
    assert!(
        (lanes::boundary_residual(&kernel, &ctx, &drift, 0, 0)
            - lanes::boundary_residual(&zero, &ctx, &drift, 0, 0))
        .abs()
            < 1e-12,
        "tangential row must stay free"
    );
    // Pressure row carries no penalty either.
    assert!(
        (lanes::boundary_residual(&kernel, &ctx, &drift, 2, 0)
            - lanes::boundary_residual(&zero, &ctx, &drift, 2, 0))
        .abs()
            < 1e-12,
        "pressure row must carry no penalty"
    );
    // Normal (y) row adds gamma*(u_l.n): compare against closure inversion.
    for (u, v, a) in [(0.1, 0.2, 0.3), (0.0, -0.5, 0.4), (0.3, 0.0, 0.05)] {
        let state = drift_state(u, v, 1.0, a);
        let liquid = config.liquid_velocity(a, [u, v]);
        let uln = liquid[1];
        let got = lanes::boundary_residual(&kernel, &ctx, &state, 1, 0);
        let base = lanes::boundary_residual(&zero, &ctx, &state, 1, 0);
        assert!(
            (got - base - penalty * uln).abs() < 1e-9 * penalty * uln.abs().max(1.0),
            "normal penalty mismatch at u=({u},{v}) a={a}: {got} != {base}+pen*{uln}"
        );
    }
}

#[test]
fn free_surface_penalty_action_matches_finite_difference() {
    let eps = 1e-7;
    let config = drift_config();
    let kernel = TensorDriftFreeSurface2D::new(config).with_penalty(1.0e5);
    let ctx = facet_ctx([0.0, 1.0]);
    for alpha in [0.05, 0.3, 0.5, 0.9] {
        for (u, v) in [(0.4, 0.3), (-0.2, -0.5), (0.1, 0.2)] {
            let state = drift_state(u, v, 1.0, alpha);
            for eq in 0..2 {
                // Mixture normal direction.
                let plus_v = drift_state(u, v + eps, 1.0, alpha);
                let fd = (lanes::boundary_residual(&kernel, &ctx, &plus_v, eq, 0)
                    - lanes::boundary_residual(&kernel, &ctx, &state, eq, 0))
                    / eps;
                let analytic = lanes::boundary_action(
                    &kernel,
                    &ctx,
                    &state,
                    &drift_state(0.0, 1.0, 0.0, 0.0),
                    eq,
                    0,
                );
                assert!(
                    (fd - analytic).abs() < 1e-5 * analytic.abs().max(1.0),
                    "penalty velocity action mismatch eq={eq} a={alpha} u=({u},{v}): {analytic} != {fd}"
                );
                // Void-fraction direction (nontrivial via A(a), B(a)).
                let plus_a = drift_state(u, v, 1.0, alpha + eps);
                let fd_a = (lanes::boundary_residual(&kernel, &ctx, &plus_a, eq, 0)
                    - lanes::boundary_residual(&kernel, &ctx, &state, eq, 0))
                    / eps;
                let analytic_a = lanes::boundary_action(
                    &kernel,
                    &ctx,
                    &state,
                    &drift_state(0.0, 0.0, 0.0, 1.0),
                    eq,
                    0,
                );
                assert!(
                    (fd_a - analytic_a).abs() < 1e-5 * analytic_a.abs().max(1.0),
                    "penalty alpha action mismatch eq={eq} a={alpha} u=({u},{v}): {analytic_a} != {fd_a}"
                );
            }
        }
    }
}

#[test]
fn free_surface_void_action_matches_finite_difference() {
    let eps = 1e-7;
    let config = drift_config();
    let kernel = TensorDriftFreeSurface2D::new(config);
    let ctx = facet_ctx([0.0, 1.0]);
    for alpha in [0.05, 0.3, 0.5, 0.9] {
        for (u, v) in [(0.4, 0.3), (-0.2, -0.5), (0.0, 0.0)] {
            let state = drift_state(u, v, 1.0, alpha);
            // Alpha direction.
            let plus = drift_state(u, v, 1.0, alpha + eps);
            let fd = (lanes::boundary_residual(&kernel, &ctx, &plus, 3, 0)
                - lanes::boundary_residual(&kernel, &ctx, &state, 3, 0))
                / eps;
            let analytic = lanes::boundary_action(
                &kernel,
                &ctx,
                &state,
                &drift_state(0.0, 0.0, 0.0, 1.0),
                3,
                0,
            );
            assert!(
                (fd - analytic).abs() < 1e-6 * analytic.abs().max(1.0),
                "void action mismatch at alpha={alpha} u=({u},{v}): {analytic} != {fd}"
            );
            // Mixture-velocity direction (normal component only at this facet).
            let plus_v = drift_state(u, v + eps, 1.0, alpha);
            let fd_v = (lanes::boundary_residual(&kernel, &ctx, &plus_v, 3, 0)
                - lanes::boundary_residual(&kernel, &ctx, &state, 3, 0))
                / eps;
            let analytic_v = lanes::boundary_action(
                &kernel,
                &ctx,
                &state,
                &drift_state(0.0, 1.0, 0.0, 0.0),
                3,
                0,
            );
            assert!(
                (fd_v - analytic_v).abs() < 1e-6 * analytic_v.abs().max(1.0),
                "void velocity action mismatch at alpha={alpha} u=({u},{v}): {analytic_v} != {fd_v}"
            );
        }
    }
}

#[test]
fn free_surface_blocks_void_reentry_but_vents_drift() {
    let config = drift_config();
    let kernel = TensorDriftFreeSurface2D::new(config);
    let top = facet_ctx([0.0, 1.0]);
    // Incoming mixture flow: advective part is clipped, drift still vents.
    let inflow = drift_state(0.0, -0.5, 0.0, 0.4);
    let v0 =
        config
            .ishii_zuber
            .drift_speed(0.0, config.rho_l, config.rho_g, config.gravity_magnitude());
    let got = lanes::boundary_residual(&kernel, &top, &inflow, 3, 0);
    assert!(
        (got - 0.4 * v0).abs() < 1e-12,
        "inflow must leave drift-only venting: {got}"
    );
    // Zero gravity: no drift, inflow gives exactly zero void flux.
    let still = TensorDriftFreeSurface2D::new(
        DriftFlux2DConfig::new(1000.0, 1.2, 1.0e-3, 1.8e-5, 10.0, 0.1).with_gravity_mag(0.0),
    );
    let got = lanes::boundary_residual(&still, &top, &inflow, 3, 0);
    assert!(
        got.abs() < 1e-14,
        "still inflow must give zero void flux: {got}"
    );
    // Bottom-facing facet: rise points inward, so no drift contribution.
    let bottom = facet_ctx([0.0, -1.0]);
    let outflow = drift_state(0.0, -0.5, 0.0, 0.4);
    let got = lanes::boundary_residual(&kernel, &bottom, &outflow, 3, 0);
    let want = 0.5 * config.distribution_parameter() * 0.5 * 0.4;
    assert!(
        (got - want).abs() < 1e-12,
        "bottom facet must carry advection only: {got} != {want}"
    );
}

#[test]
fn bubble_plume_mesh_loads_with_named_boundaries() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/navier-stokes/bubble_plume.msh"
    );
    let data = ormatex_sem_nd::gmsh_quad_data(path).expect("failed to load bubble-plume mesh");
    for name in [
        "left",
        "right",
        "bottom",
        "freesurface",
        "injector_inlet",
        "injector_wall_left",
        "injector_wall_right",
        "injector_wall_top",
    ] {
        assert!(
            !data.metadata.boundary_facets(name).is_empty(),
            "missing Physical Curve {name}"
        );
    }
    assert!(!data.metadata.cell_indices("domain").is_empty());
}
