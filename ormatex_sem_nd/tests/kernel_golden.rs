//! Golden bitwise hashes pinning the lane-packed tensor kernel math.
//!
//! Every volume and boundary tensor kernel (plus the example composites) is
//! evaluated through the lane-packed entry points on deterministic
//! pseudo-random inputs (distinct per-lane ctx/cells/facets, all equations,
//! all q; residual and Jacobian action) and all output bits are folded into
//! an FNV-1a u64. Since lanes == scalar bitwise today, these hashes pin the
//! scalar-equivalent math forever across the `_lanes` -> renamed migration.
//!
//! Regeneration: `PRINT_GOLDEN=1 cargo test --release -p ormatex_sem_nd
//! --test kernel_golden -- --nocapture` prints fresh hashes; paste them into
//! the `EXPECTED_*` constants below.
//!
//! Finite-difference checks validate `tensor_jacobian_action` against
//! central differences of `tensor_residual` on smooth interior states.
//!
//! Portability: volume hashes use only `+ - * / sqrt/powi` (no FMA
//! contraction) and are portable; the boundary group includes libm `tanh`
//! and may need regeneration on non-glibc platforms.

mod common;
use common::lanes;

use faer::sparse::{SparseColMat, Triplet};
use ormatex_sem_nd::{
    Coefficient, ConstantCoefficient, DistributionParameter, DriftFlux1DConfig, DriftFlux2DConfig,
    EdacNavierStokes1DConfig, EdacNavierStokes2DConfig, FluxKernel1D, FrozenQuadratureField,
    LaneState, MaterialContext, MaterialProperty, StateTensorBoundaryIntegrator, StateView,
    TensorCtx, TensorFacetCtx, TensorResidualKernel, TensorResidualKernelSet,
    TensorResidualKernelSum, LANES,
};
use ormatex_sem_nd::{FrozenFacetField, ALPHA_1D, ALPHA_2D};
use ormatex_sem_nd::{
    TensorDriftDirectionalDoNothing2D, TensorDriftFlux1D, TensorDriftFlux2D,
    TensorDriftFreeSurface2D, TensorDriftGravity1D, TensorDriftGravity2D,
    TensorDriftMomentumConvectionSplit1D, TensorDriftMomentumConvectionSplit2D,
    TensorDriftPressureAdvectionSplit1D, TensorDriftPressureAdvectionSplit2D,
    TensorDriftPressureDiffusion1D, TensorDriftPressureDiffusion2D,
    TensorDriftPressureDivergence1D, TensorDriftPressureDivergence2D,
    TensorDriftPressureGradient1D, TensorDriftPressureGradient2D, TensorDriftSplitBoundaryFlux2D,
    TensorDriftTurbulentDispersion1D, TensorDriftTurbulentDispersion2D, TensorDriftViscousStress1D,
    TensorDriftViscousStress2D, TensorDriftVoidAdvectionSplit1D, TensorDriftVoidAdvectionSplit2D,
    TensorKernelAdvDiff, TensorKernelAdvDiff2D, TensorKernelAdvDiffSUPG, TensorKernelAdvDiffSUPG2D,
    TensorKernelAdvection2D, TensorKernelAdvectionOutflow2D, TensorKernelBoussinesq2D,
    TensorKernelConservationLaw1D, TensorKernelDiffusion, TensorKernelDiffusion2D,
    TensorKernelEdacDirectionalDoNothing2D, TensorKernelEdacDongOutflow2D,
    TensorKernelEdacMomentumConvection2D, TensorKernelEdacMomentumConvectionSplit1D,
    TensorKernelEdacMomentumConvectionSplit2D, TensorKernelEdacNavierStokes2D,
    TensorKernelEdacNavierStokesSplit1D, TensorKernelEdacNavierStokesSplit2D,
    TensorKernelEdacNoSlipWall2D, TensorKernelEdacPressureAdvection2D,
    TensorKernelEdacPressureAdvectionSplit1D,
    TensorKernelEdacPressureAdvectionSplit2D, TensorKernelEdacPressureDiffusion1D,
    TensorKernelEdacPressureDiffusion2D, TensorKernelEdacPressureDivergence1D,
    TensorKernelEdacPressureDivergence2D, TensorKernelEdacPressureGradient1D,
    TensorKernelEdacPressureGradient2D, TensorKernelEdacSlipWall2D,
    TensorKernelEdacSplitBoundaryFlux2D, TensorKernelEdacViscousStress1D,
    TensorKernelEdacViscousStress2D, TensorKernelEnergyAdvectionDiffusion1D,
    TensorKernelEnergyAdvectionDiffusion2D, TensorKernelLinearReaction, TensorKernelMass,
    TensorKernelVolumeSource,
};

// ---------------------------------------------------------------------------
// FNV-1a folding
// ---------------------------------------------------------------------------

/// Fold one f64's bits into the FNV-1a accumulator.
///
/// # Arguments
/// * `h` - running hash state, updated in place.
/// * `v` - value whose bits are mixed in.
fn fold_f64(h: &mut u64, v: f64) {
    *h ^= v.to_bits();
    *h = h.wrapping_mul(1099511628211u64);
}

/// Fold a lane vector into the accumulator.
///
/// # Arguments
/// * `h` - running hash state.
/// * `v` - lane vector to fold lane by lane.
fn fold_lanes(h: &mut u64, v: &[f64; LANES]) {
    for x in v.iter() {
        fold_f64(h, *x);
    }
}

// ---------------------------------------------------------------------------
// Lane evaluation + hash + FD helpers
// ---------------------------------------------------------------------------

/// Evaluate residual lanes for all equations/q and fold into the hash.
///
/// Generic over `GDIM` (1D and 2D share one body); the call sequence is
/// identical to the old `hash_volume_1d`/`hash_volume_2d` pair, so hashes
/// are unchanged.
///
/// # Arguments
/// * `h` - running hash state.
/// * `kernel` - volume kernel under test.
/// * `ctxs` - one context per lane.
/// * `state`/`direction` - lane-packed states.
fn hash_volume<const GDIM: usize, K: TensorResidualKernel<GDIM> + ?Sized>(
    h: &mut u64,
    kernel: &K,
    ctxs: &[TensorCtx<'_>],
    state: &LaneState<'_>,
    direction: &LaneState<'_>,
) {
    let no = kernel.output_nfields();
    // Residual: all owned equations.
    for eq in 0..no {
        for q in 0..state.npts {
            let mut f0 = [0.0; LANES];
            let mut f1x = [0.0; LANES];
            let mut f1y = [0.0; LANES];
            kernel.tensor_residual(ctxs, state, eq, q, &mut f0, &mut f1x, &mut f1y);
            fold_lanes(h, &f0);
            fold_lanes(h, &f1x);
            fold_lanes(h, &f1y);
            let mut j0 = [0.0; LANES];
            let mut j1x = [0.0; LANES];
            let mut j1y = [0.0; LANES];
            kernel
                .tensor_jacobian_action(ctxs, state, direction, eq, q, &mut j0, &mut j1x, &mut j1y);
            fold_lanes(h, &j0);
            fold_lanes(h, &j1x);
            fold_lanes(h, &j1y);
        }
    }
}

/// Fold boundary lane outputs for all equations/q.
///
/// # Arguments
/// * `h` - running hash state.
/// * `kernel` - boundary kernel under test.
/// * `ctxs` - one facet context per lane.
/// * `state`/`direction` - lane-packed facet states.
fn hash_boundary<K: StateTensorBoundaryIntegrator<2> + ?Sized>(
    h: &mut u64,
    kernel: &K,
    ctxs: &[TensorFacetCtx<'_>],
    state: &LaneState<'_>,
    direction: &LaneState<'_>,
) {
    let no = kernel.output_nfields();
    for eq in 0..no {
        for q in 0..state.npts {
            let mut out = [0.0; LANES];
            kernel.tensor_residual(ctxs, state, eq, q, &mut out);
            fold_lanes(h, &out);
            let mut jout = [0.0; LANES];
            kernel.tensor_jacobian_action(ctxs, state, direction, eq, q, &mut jout);
            fold_lanes(h, &jout);
        }
    }
}

/// Central-difference consistency of the lane Jacobian action.
///
/// Generic over `GDIM`; compares `tensor_jacobian_action` against
/// `(R(s+e*d) - R(s-e*d)) / (2e)` lane by lane on the given states.
/// Skipped (with reason) for kernels with non-differentiable branches at the
/// sampled points; all kernels below use smooth interior states so no skip.
/// The 1D check covers `f0`/`f1x` only (the 1D assembler ignores `f1y`);
/// 2D additionally covers `f1y`.
///
/// # Arguments
/// * `kernel` - volume kernel under test.
/// * `ctxs` - one context per lane.
/// * `values`/`grads`/`dir_values`/`dir_grads` - lane-packed buffers.
fn fd_check<const GDIM: usize, K: TensorResidualKernel<GDIM> + ?Sized>(
    kernel: &K,
    ctxs: &[TensorCtx<'_>],
    nfields: usize,
    npts: usize,
    values: &[f64],
    grads: &[f64],
    dir_values: &[f64],
    dir_grads: &[f64],
) {
    let eps = 1e-6;
    let gdim = GDIM;
    let mut plus_v = values.to_vec();
    let mut minus_v = values.to_vec();
    let mut plus_g = grads.to_vec();
    let mut minus_g = grads.to_vec();
    for i in 0..plus_v.len() {
        plus_v[i] += eps * dir_values[i];
        minus_v[i] -= eps * dir_values[i];
    }
    for i in 0..plus_g.len() {
        plus_g[i] += eps * dir_grads[i];
        minus_g[i] -= eps * dir_grads[i];
    }
    let state = LaneState {
        nfields,
        npts,
        gdim,
        values,
        grads,
        field_indices: &[],
    };
    let dir = LaneState {
        nfields,
        npts,
        gdim,
        values: dir_values,
        grads: dir_grads,
        field_indices: &[],
    };
    let sp = LaneState {
        nfields,
        npts,
        gdim,
        values: &plus_v,
        grads: &plus_g,
        field_indices: &[],
    };
    let sm = LaneState {
        nfields,
        npts,
        gdim,
        values: &minus_v,
        grads: &minus_g,
        field_indices: &[],
    };
    let no = kernel.output_nfields();
    for eq in 0..no {
        for q in 0..npts {
            let mut r0p = [0.0; LANES];
            let mut r1p = [0.0; LANES];
            let mut ryp = [0.0; LANES];
            let mut r0m = [0.0; LANES];
            let mut r1m = [0.0; LANES];
            let mut rym = [0.0; LANES];
            let mut j0 = [0.0; LANES];
            let mut j1x = [0.0; LANES];
            let mut jy = [0.0; LANES];
            kernel.tensor_residual(ctxs, &sp, eq, q, &mut r0p, &mut r1p, &mut ryp);
            kernel.tensor_residual(ctxs, &sm, eq, q, &mut r0m, &mut r1m, &mut rym);
            kernel.tensor_jacobian_action(ctxs, &state, &dir, eq, q, &mut j0, &mut j1x, &mut jy);
            for l in 0..LANES {
                let pairs = [
                    ((r0p[l] - r0m[l]) / (2.0 * eps), j0[l]),
                    ((r1p[l] - r1m[l]) / (2.0 * eps), j1x[l]),
                    ((ryp[l] - rym[l]) / (2.0 * eps), jy[l]),
                ];
                // 1D ignores the y-flux slot; keep the kernel call (it still
                // writes `f1y`) but only check the slots the assembler reads.
                let npairs = if GDIM == 1 { 2 } else { 3 };
                for (fd_n, fd_d) in &pairs[..npairs] {
                    let (fd_n, fd_d) = (*fd_n, *fd_d);
                    let scale = fd_d.abs().max(fd_n.abs()).max(1.0);
                    let rel = (fd_n - fd_d).abs() / scale;
                    assert!(
                        rel < 2e-6,
                        "FD mismatch eq={eq} q={q} lane={l}: fd={fd_n:e} jac={fd_d:e} rel={rel:e}"
                    );
                }
            }
        }
    }
}

/// Central-difference consistency of a lane boundary kernel.
///
/// Same contract as [`fd_check`] for the scalar trace flux. Dong outflow
/// and directional do-nothing have a kink at `u.n == 0` (outflow vs inflow
/// branch); callers keep sampled states away from the kink (normal velocity
/// shifted off zero), so no lane skip is needed and the check stays strict.
fn fd_check_boundary<K: StateTensorBoundaryIntegrator<2> + ?Sized>(
    kernel: &K,
    ctxs: &[TensorFacetCtx<'_>],
    nfields: usize,
    npts: usize,
    values: &[f64],
    dir_values: &[f64],
) {
    let eps = 1e-6;
    // All current boundary kernels report `requires_gradients == false`;
    // FD perturbs values only and grads stay empty.
    let empty: Vec<f64> = Vec::new();
    let mut plus_v = values.to_vec();
    let mut minus_v = values.to_vec();
    for i in 0..plus_v.len() {
        plus_v[i] += eps * dir_values[i];
        minus_v[i] -= eps * dir_values[i];
    }
    let state = LaneState {
        nfields,
        npts,
        gdim: 2,
        values,
        grads: &empty,
        field_indices: &[],
    };
    let dir = LaneState {
        nfields,
        npts,
        gdim: 2,
        values: dir_values,
        grads: &empty,
        field_indices: &[],
    };
    let sp = LaneState {
        nfields,
        npts,
        gdim: 2,
        values: &plus_v,
        grads: &empty,
        field_indices: &[],
    };
    let sm = LaneState {
        nfields,
        npts,
        gdim: 2,
        values: &minus_v,
        grads: &empty,
        field_indices: &[],
    };
    let no = kernel.output_nfields();
    for eq in 0..no {
        for q in 0..npts {
            let mut rp = [0.0; LANES];
            let mut rm = [0.0; LANES];
            let mut j = [0.0; LANES];
            kernel.tensor_residual(ctxs, &sp, eq, q, &mut rp);
            kernel.tensor_residual(ctxs, &sm, eq, q, &mut rm);
            kernel.tensor_jacobian_action(ctxs, &state, &dir, eq, q, &mut j);
            for l in 0..LANES {
                let fd = (rp[l] - rm[l]) / (2.0 * eps);
                let scale = j[l].abs().max(fd.abs()).max(1.0);
                let rel = (fd - j[l]).abs() / scale;
                // Inputs avoid the `u.n == 0` kink by construction (see
                // callers); a failure here is a real math mismatch.
                assert!(
                    rel < 2e-6,
                    "FD mismatch (boundary) eq={eq} q={q} lane={l}: fd={fd:e} jac={} rel={rel:e}",
                    j[l]
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

/// Build lane-packed volume states with fixed seed.
///
/// # Arguments
/// * `nfields` - field count.
/// * `storage` - geometry backing.
/// * `seed` - deterministic seed.
///
/// # Returns
/// `(ctxs, packed_values, packed_grads, packed_dir_values, packed_dir_grads)`.
fn volume_fixture(
    nfields: usize,
    storage: &lanes::VolumeStorage,
    seed: u64,
) -> (Vec<TensorCtx<'_>>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let npts = storage.npts();
    let gdim = storage.gdim();
    let mut s = seed;
    let (values_l, grads_l, dir_values_l, dir_grads_l) =
        lanes::random_lane_buffers(nfields, npts, gdim, &mut s);
    let (packed_values, packed_grads) = lanes::pack_lanes(nfields, npts, gdim, &values_l, &grads_l);
    let (packed_dir_values, packed_dir_grads) =
        lanes::pack_lanes(nfields, npts, gdim, &dir_values_l, &dir_grads_l);
    let ctxs: Vec<TensorCtx<'_>> = (0..LANES).map(|l| storage.ctx(l)).collect();
    (
        ctxs,
        packed_values,
        packed_grads,
        packed_dir_values,
        packed_dir_grads,
    )
}

#[derive(Clone, Copy)]
struct LinCoeff {
    a: f64,
    b: f64,
    c: f64,
}
impl Coefficient<f64> for LinCoeff {
    #[inline]
    fn eval(&self, ctx: &MaterialContext<'_>) -> f64 {
        let u = ctx.state.map(|v| v.value(0, ctx.q)).unwrap_or(0.0);
        self.a * u + self.b * ctx.point[0] + self.c
    }
}
impl MaterialProperty<f64> for LinCoeff {
    #[inline]
    fn derivative(&self, _ctx: &MaterialContext<'_>, solution_field: usize) -> Option<f64> {
        if solution_field == 0 {
            Some(self.a)
        } else {
            Some(0.0)
        }
    }
}

fn reaction_matrix_3() -> SparseColMat<usize, f64> {
    let rates = [
        (0usize, 0usize, -0.1),
        (1, 0, 0.1),
        (1, 1, -10.0),
        (2, 1, 10.0),
        (2, 2, -0.01),
    ];
    let triplets: Vec<_> = rates
        .iter()
        .map(|&(r, c, v)| Triplet::new(r, c, v))
        .collect();
    SparseColMat::try_new_from_triplets(3, 3, &triplets).unwrap()
}

fn distinct_frozen_field(npts: usize, seed: &mut u64) -> FrozenQuadratureField {
    let values: Vec<f64> = (0..LANES * npts)
        .map(|_| lanes::prng_range(seed, -2.0, 2.0))
        .collect();
    FrozenQuadratureField::new(npts, values)
}

#[derive(Clone, Copy)]
struct LinearFlux1D;
impl FluxKernel1D for LinearFlux1D {
    fn nfields(&self) -> usize {
        2
    }
    fn flux(&self, _ctx: &TensorCtx<'_>, state: StateView<'_>, equation: usize, q: usize) -> f64 {
        match equation {
            0 => state.value(0, q) + 0.5 * state.value(1, q),
            1 => 0.25 * state.value(0, q) + 2.0 * state.value(1, q),
            _ => panic!("eq OOR"),
        }
    }
    fn flux_jacobian(
        &self,
        _ctx: &TensorCtx<'_>,
        _state: StateView<'_>,
        equation: usize,
        unknown: usize,
        _q: usize,
    ) -> f64 {
        match (equation, unknown) {
            (0, 0) => 1.0,
            (0, 1) => 0.5,
            (1, 0) => 0.25,
            (1, 1) => 2.0,
            _ => panic!("block OOR"),
        }
    }
}

#[derive(Clone, Copy)]
struct LogFlux1D {
    cs: f64,
}
impl FluxKernel1D for LogFlux1D {
    fn nfields(&self) -> usize {
        2
    }
    fn flux(&self, _ctx: &TensorCtx<'_>, state: StateView<'_>, equation: usize, q: usize) -> f64 {
        let u = state.value(0, q);
        let rho = state.value(1, q);
        let cs2 = self.cs * self.cs;
        match equation {
            0 => 0.5 * u * u + cs2 * rho.ln(),
            1 => rho * u,
            _ => panic!("eq OOR"),
        }
    }
    fn flux_jacobian(
        &self,
        _ctx: &TensorCtx<'_>,
        state: StateView<'_>,
        equation: usize,
        unknown: usize,
        q: usize,
    ) -> f64 {
        let u = state.value(0, q);
        let rho = state.value(1, q);
        let cs2 = self.cs * self.cs;
        match (equation, unknown) {
            (0, 0) => u,
            (0, 1) => cs2 / rho,
            (1, 0) => rho,
            (1, 1) => u,
            _ => panic!("block OOR"),
        }
    }
}

const ALPHA_PALETTE: [f64; 8] = [-0.5, 0.0, 1e-13, 0.1, 0.5, 1.0 - 5e-13, 1.0, 1.4];

/// Overwrite void-fraction field with branch-covering palette.
///
/// # Arguments
/// * `values_lanes` - per-lane value buffers, modified in place.
/// * `npts` - quadrature count.
/// * `alpha_field` - void-fraction field index.
fn craft_alpha(values_lanes: &mut [Vec<f64>], npts: usize, alpha_field: usize) {
    for (l, values) in values_lanes.iter_mut().enumerate() {
        for q in 0..npts {
            values[alpha_field * npts + q] = ALPHA_PALETTE[(q + l) % ALPHA_PALETTE.len()];
        }
    }
}

fn drift_config_2d() -> DriftFlux2DConfig {
    DriftFlux2DConfig::new(1000.0, 1.2, 1.0e-3, 1.8e-5, 10.0, 0.1)
}
fn drift_config_1d() -> DriftFlux1DConfig {
    DriftFlux1DConfig::new(1000.0, 1.2, 1.0e-3, 1.8e-5, 10.0)
}
fn edac_2d() -> EdacNavierStokes2DConfig {
    EdacNavierStokes2DConfig::new(1.0, 1.0 / 200.0, 4.0, 0.1)
}
fn edac_1d() -> EdacNavierStokes1DConfig {
    EdacNavierStokes1DConfig::new(1.0, 0.01, 10.0)
}

// ---------------------------------------------------------------------------
// Golden hash computation
// ---------------------------------------------------------------------------

/// Compute golden hashes for all kernel groups.
///
/// # Returns
/// Map from group name to FNV-1a hash in deterministic evaluation order.
fn compute_hashes() -> Vec<(&'static str, u64)> {
    let mut out = Vec::new();

    // --- group: basic 1D (diffusion, adv-diff, supg, mass, source, reaction)
    {
        let mut h = 0xcbf29ce484222325u64;
        let storage = lanes::VolumeStorage::new_1d(4);
        let npts = storage.npts();
        let mut s = 0xB451_1000u64;
        let fz = distinct_frozen_field(npts, &mut s);
        let k = TensorKernelDiffusion::new(0.7);
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(1, &storage, 0xB451_1001);
        let st = LaneState {
            nfields: 1,
            npts,
            gdim: 1,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 1,
            npts,
            gdim: 1,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<1, _>(&mut h, &k, &ctxs, &st, &di);
        let k2 = TensorKernelDiffusion::with_coefficient(LinCoeff {
            a: 0.4,
            b: 0.1,
            c: 0.5,
        });
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(1, &storage, 0xB451_1004);
        let st = LaneState {
            nfields: 1,
            npts,
            gdim: 1,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 1,
            npts,
            gdim: 1,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<1, _>(&mut h, &k2, &ctxs, &st, &di);
        let k3 = TensorKernelDiffusion::with_coefficient(fz);
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(1, &storage, 0xB451_1005);
        let st = LaneState {
            nfields: 1,
            npts,
            gdim: 1,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 1,
            npts,
            gdim: 1,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<1, _>(&mut h, &k3, &ctxs, &st, &di);
        let k4 = TensorKernelAdvDiff::new(0.05, -1.5);
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(1, &storage, 0xB453_3001);
        let st = LaneState {
            nfields: 1,
            npts,
            gdim: 1,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 1,
            npts,
            gdim: 1,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<1, _>(&mut h, &k4, &ctxs, &st, &di);
        let k5 = TensorKernelAdvDiffSUPG::new(0.05, -2.0, 0.1);
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(1, &storage, 0xB456_6001);
        let st = LaneState {
            nfields: 1,
            npts,
            gdim: 1,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 1,
            npts,
            gdim: 1,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<1, _>(&mut h, &k5, &ctxs, &st, &di);
        let k6 = TensorKernelMass::new();
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(1, &storage, 0xB458_8001);
        let st = LaneState {
            nfields: 1,
            npts,
            gdim: 1,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 1,
            npts,
            gdim: 1,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<1, _>(&mut h, &k6, &ctxs, &st, &di);
        let k7 = TensorKernelVolumeSource::new(0.5);
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(1, &storage, 0xB458_8004);
        let st = LaneState {
            nfields: 1,
            npts,
            gdim: 1,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 1,
            npts,
            gdim: 1,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<1, _>(&mut h, &k7, &ctxs, &st, &di);
        let k8 =
            TensorKernelLinearReaction::with_field_names(reaction_matrix_3(), ["c0", "c1", "c2"]);
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(3, &storage, 0xB459_9001);
        let st = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<1, _>(&mut h, &k8, &ctxs, &st, &di);
        // Frozen species Set exactly like the 1D example.
        let mut s2 = 0xB45b_b000u64;
        let frozen = distinct_frozen_field(npts, &mut s2);
        let kset = TensorResidualKernelSet::from_kernel(
            TensorKernelAdvDiff::with_coefficients(ConstantCoefficient(0.002), frozen.clone())
                .with_field_name("c0"),
        )
        .with(
            TensorKernelAdvDiff::with_coefficients(ConstantCoefficient(0.002), frozen.clone())
                .with_field_name("c1"),
        )
        .with(
            TensorKernelAdvDiff::with_coefficients(ConstantCoefficient(0.002), frozen)
                .with_field_name("c2"),
        )
        .with(TensorKernelLinearReaction::with_field_names(
            reaction_matrix_3(),
            ["c0", "c1", "c2"],
        ));
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(3, &storage, 0xB45b_b001);
        let st = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<1, _>(&mut h, &kset, &ctxs, &st, &di);
        out.push(("basic_1d", h));
    }

    // --- group: basic 2D
    {
        let mut h = 0xcbf29ce484222325u64;
        let storage = lanes::VolumeStorage::new_2d(3);
        let npts = storage.npts();
        let mut s = 0xB452_2000u64;
        let fz = distinct_frozen_field(npts, &mut s);
        let k = TensorKernelDiffusion2D::new(0.7);
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(1, &storage, 0xB452_2001);
        let st = LaneState {
            nfields: 1,
            npts,
            gdim: 2,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 1,
            npts,
            gdim: 2,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<2, _>(&mut h, &k, &ctxs, &st, &di);
        let k2 = TensorKernelDiffusion2D::with_coefficient(fz);
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(1, &storage, 0xB452_2005);
        let st = LaneState {
            nfields: 1,
            npts,
            gdim: 2,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 1,
            npts,
            gdim: 2,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<2, _>(&mut h, &k2, &ctxs, &st, &di);
        let k3 = TensorKernelAdvDiff2D::new(0.05, [-1.5, 0.5]);
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(1, &storage, 0xB454_4001);
        let st = LaneState {
            nfields: 1,
            npts,
            gdim: 2,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 1,
            npts,
            gdim: 2,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<2, _>(&mut h, &k3, &ctxs, &st, &di);
        let k4 = TensorKernelAdvection2D::new([-2.0, 1.0]);
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(1, &storage, 0xB455_5001);
        let st = LaneState {
            nfields: 1,
            npts,
            gdim: 2,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 1,
            npts,
            gdim: 2,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<2, _>(&mut h, &k4, &ctxs, &st, &di);
        let k5 = TensorKernelAdvDiffSUPG2D::new(0.05, [-2.0, 1.0], 0.1);
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(1, &storage, 0xB457_7001);
        let st = LaneState {
            nfields: 1,
            npts,
            gdim: 2,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 1,
            npts,
            gdim: 2,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<2, _>(&mut h, &k5, &ctxs, &st, &di);
        let k6 = TensorKernelMass::new();
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(1, &storage, 0xB458_8003);
        let st = LaneState {
            nfields: 1,
            npts,
            gdim: 2,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 1,
            npts,
            gdim: 2,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<2, _>(&mut h, &k6, &ctxs, &st, &di);
        let k7 = TensorKernelVolumeSource::new(-0.25);
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(1, &storage, 0xB458_8005);
        let st = LaneState {
            nfields: 1,
            npts,
            gdim: 2,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 1,
            npts,
            gdim: 2,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<2, _>(&mut h, &k7, &ctxs, &st, &di);
        let k8 = TensorKernelBoussinesq2D::new(710.0, [0.0, 1.0]);
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(1, &storage, 0xB005_51E0);
        let st = LaneState {
            nfields: 1,
            npts,
            gdim: 2,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 1,
            npts,
            gdim: 2,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<2, _>(&mut h, &k8, &ctxs, &st, &di);
        let k9 = TensorKernelEnergyAdvectionDiffusion2D::new(0.02);
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(3, &storage, 0xB45a_a001);
        let st = LaneState {
            nfields: 3,
            npts,
            gdim: 2,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 3,
            npts,
            gdim: 2,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<2, _>(&mut h, &k9, &ctxs, &st, &di);
        let k10 =
            TensorKernelLinearReaction::with_field_names(reaction_matrix_3(), ["c0", "c1", "c2"]);
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(3, &storage, 0xB459_9003);
        let st = LaneState {
            nfields: 3,
            npts,
            gdim: 2,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 3,
            npts,
            gdim: 2,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<2, _>(&mut h, &k10, &ctxs, &st, &di);
        out.push(("basic_2d", h));
    }

    // --- group: conservation law 1D (isothermal Euler)
    {
        let mut h = 0xcbf29ce484222325u64;
        let storage = lanes::VolumeStorage::new_1d(4);
        let npts = storage.npts();
        for (seed, flux) in [(
            0x11aa_0091u64,
            TensorKernelConservationLaw1D::new(LinearFlux1D),
        )] {
            let (ctxs, pv, pg, pdv, pdg) = volume_fixture(2, &storage, seed);
            let st = LaneState {
                nfields: 2,
                npts,
                gdim: 1,
                values: &pv,
                grads: &pg,
                field_indices: &[],
            };
            let di = LaneState {
                nfields: 2,
                npts,
                gdim: 1,
                values: &pdv,
                grads: &pdg,
                field_indices: &[],
            };
            hash_volume::<1, _>(&mut h, &flux, &ctxs, &st, &di);
        }
        // Log flux needs positive rho.
        {
            let mut s = 0x11aa_00a1u64;
            let (mut vl, gl, mut dvl, dgl) = lanes::random_lane_buffers(2, npts, 1, &mut s);
            for v in vl.iter_mut().chain(dvl.iter_mut()) {
                for q in 0..npts {
                    v[1 * npts + q] = v[1 * npts + q].abs() + 0.5;
                }
            }
            let (pv, pg) = lanes::pack_lanes(2, npts, 1, &vl, &gl);
            let (pdv, pdg) = lanes::pack_lanes(2, npts, 1, &dvl, &dgl);
            let ctxs: Vec<TensorCtx<'_>> = (0..LANES).map(|l| storage.ctx(l)).collect();
            let st = LaneState {
                nfields: 2,
                npts,
                gdim: 1,
                values: &pv,
                grads: &pg,
                field_indices: &[],
            };
            let di = LaneState {
                nfields: 2,
                npts,
                gdim: 1,
                values: &pdv,
                grads: &pdg,
                field_indices: &[],
            };
            let k = TensorKernelConservationLaw1D::new(LogFlux1D { cs: 1.0 });
            hash_volume::<1, _>(&mut h, &k, &ctxs, &st, &di);
        }
        out.push(("conservation_1d", h));
    }

    // --- group: EDAC 1D
    {
        let mut h = 0xcbf29ce484222325u64;
        let storage = lanes::VolumeStorage::new_1d(4);
        let cfg = edac_1d();
        let kernels: Vec<Box<dyn TensorResidualKernel<1>>> = vec![
            Box::new(TensorKernelEdacMomentumConvectionSplit1D::new(cfg)),
            Box::new(TensorKernelEdacPressureGradient1D::new(cfg)),
            Box::new(TensorKernelEdacViscousStress1D::new(cfg)),
            Box::new(TensorKernelEdacPressureDivergence1D::new(cfg)),
            Box::new(TensorKernelEdacPressureAdvectionSplit1D::new(cfg)),
            Box::new(TensorKernelEdacPressureDiffusion1D::new(cfg)),
            Box::new(TensorKernelEdacNavierStokesSplit1D::new(cfg)),
            Box::new(TensorKernelEnergyAdvectionDiffusion1D::new(0.01)),
        ];
        let seeds = [
            0x11aa_0001u64,
            0x11aa_0011,
            0x11aa_0021,
            0x11aa_0031,
            0x11aa_0041,
            0x11aa_0051,
            0x11aa_0061,
            0x11aa_0071,
        ];
        for (k, seed) in kernels.iter().zip(seeds) {
            let nf = k.input_nfields();
            let (ctxs, pv, pg, pdv, pdg) = volume_fixture(nf, &storage, seed);
            let npts = storage.npts();
            let st = LaneState {
                nfields: nf,
                npts,
                gdim: 1,
                values: &pv,
                grads: &pg,
                field_indices: &[],
            };
            let di = LaneState {
                nfields: nf,
                npts,
                gdim: 1,
                values: &pdv,
                grads: &pdg,
                field_indices: &[],
            };
            hash_volume::<1, _>(&mut h, k.as_ref(), &ctxs, &st, &di);
        }
        // Heated-pipe 1D Set composite.
        {
            let edac = TensorResidualKernelSum::from_kernel(
                TensorKernelEdacMomentumConvectionSplit1D::new(cfg),
            )
            .with(TensorKernelEdacPressureGradient1D::new(cfg))
            .with(TensorKernelEdacViscousStress1D::new(cfg))
            .with(TensorKernelEdacPressureDivergence1D::new(cfg))
            .with(TensorKernelEdacPressureAdvectionSplit1D::new(cfg))
            .with(TensorKernelEdacPressureDiffusion1D::new(cfg));
            let set = TensorResidualKernelSet::from_kernel(edac)
                .with(TensorKernelEnergyAdvectionDiffusion1D::new(0.01))
                .with(TensorKernelVolumeSource::with_field_names(0.5, ["T"]));
            let npts = storage.npts();
            let (ctxs, pv, pg, pdv, pdg) = volume_fixture(3, &storage, 0x11aa_0081);
            let st = LaneState {
                nfields: 3,
                npts,
                gdim: 1,
                values: &pv,
                grads: &pg,
                field_indices: &[],
            };
            let di = LaneState {
                nfields: 3,
                npts,
                gdim: 1,
                values: &pdv,
                grads: &pdg,
                field_indices: &[],
            };
            hash_volume::<1, _>(&mut h, &set, &ctxs, &st, &di);
        }
        out.push(("edac_1d", h));
    }

    // --- group: EDAC 2D + composites
    {
        let mut h = 0xcbf29ce484222325u64;
        let storage = lanes::VolumeStorage::new_2d(3);
        let cfg = edac_2d();
        let kernels: Vec<Box<dyn TensorResidualKernel<2>>> = vec![
            Box::new(TensorKernelEdacMomentumConvection2D::new(cfg)),
            Box::new(TensorKernelEdacMomentumConvectionSplit2D::new(cfg)),
            Box::new(TensorKernelEdacPressureGradient2D::new(cfg)),
            Box::new(TensorKernelEdacViscousStress2D::new(cfg)),
            Box::new(TensorKernelEdacPressureDivergence2D::new(cfg)),
            Box::new(TensorKernelEdacPressureAdvection2D::new(cfg)),
            Box::new(TensorKernelEdacPressureAdvectionSplit2D::new(cfg)),
            Box::new(TensorKernelEdacPressureDiffusion2D::new(cfg)),
            Box::new(TensorKernelEdacNavierStokes2D::new(
                1.0,
                1.0 / 200.0,
                4.0,
                0.1,
            )),
            Box::new(TensorKernelEdacNavierStokesSplit2D::new(cfg)),
        ];
        let seeds = [
            0xC001u64, 0xC002, 0xC003, 0xC004, 0xC005, 0xC006, 0xC007, 0xC008, 0xC009, 0xC00A,
        ];
        for (k, seed) in kernels.iter().zip(seeds) {
            let nf = k.input_nfields();
            let (ctxs, pv, pg, pdv, pdg) = volume_fixture(nf, &storage, seed);
            let npts = storage.npts();
            let st = LaneState {
                nfields: nf,
                npts,
                gdim: 2,
                values: &pv,
                grads: &pg,
                field_indices: &[],
            };
            let di = LaneState {
                nfields: nf,
                npts,
                gdim: 2,
                values: &pdv,
                grads: &pdg,
                field_indices: &[],
            };
            hash_volume::<2, _>(&mut h, k.as_ref(), &ctxs, &st, &di);
        }
        // Cylinder split Sum, cavity fused Sum, de Vahl Davis Set.
        {
            let npts = storage.npts();
            let cyl = TensorResidualKernelSum::from_kernel(
                TensorKernelEdacMomentumConvectionSplit2D::new(cfg),
            )
            .with(TensorKernelEdacPressureGradient2D::new(cfg))
            .with(TensorKernelEdacViscousStress2D::new(cfg))
            .with(TensorKernelEdacPressureDivergence2D::new(cfg))
            .with(TensorKernelEdacPressureAdvectionSplit2D::new(cfg))
            .with(TensorKernelEdacPressureDiffusion2D::new(cfg));
            let (ctxs, pv, pg, pdv, pdg) = volume_fixture(3, &storage, 0xC111_10E2);
            let st = LaneState {
                nfields: 3,
                npts,
                gdim: 2,
                values: &pv,
                grads: &pg,
                field_indices: &[],
            };
            let di = LaneState {
                nfields: 3,
                npts,
                gdim: 2,
                values: &pdv,
                grads: &pdg,
                field_indices: &[],
            };
            hash_volume::<2, _>(&mut h, &cyl, &ctxs, &st, &di);
            let cav = ormatex_sem_nd::fuse_tensor_kernels!(
                TensorKernelEdacMomentumConvection2D::new(cfg),
                TensorKernelEdacPressureGradient2D::new(cfg),
                TensorKernelEdacViscousStress2D::new(cfg),
                TensorKernelEdacPressureDivergence2D::new(cfg),
                TensorKernelEdacPressureAdvection2D::new(cfg),
                TensorKernelEdacPressureDiffusion2D::new(cfg),
            );
            let (ctxs, pv, pg, pdv, pdg) = volume_fixture(3, &storage, 0xC111_10E3);
            let st = LaneState {
                nfields: 3,
                npts,
                gdim: 2,
                values: &pv,
                grads: &pg,
                field_indices: &[],
            };
            let di = LaneState {
                nfields: 3,
                npts,
                gdim: 2,
                values: &pdv,
                grads: &pdg,
                field_indices: &[],
            };
            hash_volume::<2, _>(&mut h, &cav, &ctxs, &st, &di);
            let edac = TensorResidualKernelSum::from_kernel(
                TensorKernelEdacMomentumConvectionSplit2D::new(cfg),
            )
            .with(TensorKernelEdacPressureGradient2D::new(cfg))
            .with(TensorKernelEdacViscousStress2D::new(cfg))
            .with(TensorKernelEdacPressureDivergence2D::new(cfg))
            .with(TensorKernelEdacPressureAdvectionSplit2D::new(cfg))
            .with(TensorKernelEdacPressureDiffusion2D::new(cfg));
            let dv = TensorResidualKernelSet::from_kernel(edac)
                .with(TensorKernelBoussinesq2D::new(1000.0 * 0.71, [0.0, 1.0]))
                .with(TensorKernelEnergyAdvectionDiffusion2D::new(1.0));
            let (ctxs, pv, pg, pdv, pdg) = volume_fixture(4, &storage, 0xDE0A_1D15);
            let st = LaneState {
                nfields: 4,
                npts,
                gdim: 2,
                values: &pv,
                grads: &pg,
                field_indices: &[],
            };
            let di = LaneState {
                nfields: 4,
                npts,
                gdim: 2,
                values: &pdv,
                grads: &pdg,
                field_indices: &[],
            };
            hash_volume::<2, _>(&mut h, &dv, &ctxs, &st, &di);
        }
        out.push(("edac_2d", h));
    }

    // --- group: drift 1D
    {
        let mut h = 0xcbf29ce484222325u64;
        let storage = lanes::VolumeStorage::new_1d(4);
        let npts = storage.npts();
        let cfg = drift_config_1d();
        let vertical = |_: &MaterialContext<'_>| std::f64::consts::FRAC_PI_2;
        // Crafted palette states for hashing (branch coverage).
        let mut mk = |seed: u64| {
            let mut s = seed;
            let (mut vl, gl, dvl, dgl) = lanes::random_lane_buffers(3, npts, 1, &mut s);
            craft_alpha(&mut vl, npts, ALPHA_1D);
            let (pv, pg) = lanes::pack_lanes(3, npts, 1, &vl, &gl);
            let (pdv, pdg) = lanes::pack_lanes(3, npts, 1, &dvl, &dgl);
            let ctxs: Vec<TensorCtx<'_>> = (0..LANES).map(|l| storage.ctx(l)).collect();
            (ctxs, pv, pg, pdv, pdg)
        };
        let k = TensorDriftMomentumConvectionSplit1D::new(cfg);
        let (ctxs, pv, pg, pdv, pdg) = mk(0xD1F7_001);
        let st = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<1, _>(&mut h, &k, &ctxs, &st, &di);
        let k = TensorDriftPressureAdvectionSplit1D::new(cfg);
        let (ctxs, pv, pg, pdv, pdg) = mk(0xD1F7_002);
        let st = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<1, _>(&mut h, &k, &ctxs, &st, &di);
        let k = TensorDriftPressureDiffusion1D::new(cfg);
        let (ctxs, pv, pg, pdv, pdg) = mk(0xD1F7_003);
        let st = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<1, _>(&mut h, &k, &ctxs, &st, &di);
        let k = TensorDriftPressureDivergence1D::new(cfg);
        let (ctxs, pv, pg, pdv, pdg) = mk(0xD1F7_004);
        let st = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<1, _>(&mut h, &k, &ctxs, &st, &di);
        let k = TensorDriftPressureGradient1D::new(cfg);
        let (ctxs, pv, pg, pdv, pdg) = mk(0xD1F7_005);
        let st = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<1, _>(&mut h, &k, &ctxs, &st, &di);
        let k = TensorDriftViscousStress1D::new(cfg);
        let (ctxs, pv, pg, pdv, pdg) = mk(0xD1F7_006);
        let st = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<1, _>(&mut h, &k, &ctxs, &st, &di);
        let k = TensorDriftVoidAdvectionSplit1D::new(cfg);
        let (ctxs, pv, pg, pdv, pdg) = mk(0xD1F7_007);
        let st = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<1, _>(&mut h, &k, &ctxs, &st, &di);
        let k = TensorDriftTurbulentDispersion1D::new(cfg, 0.01);
        let (ctxs, pv, pg, pdv, pdg) = mk(0xD1F7_008);
        let st = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<1, _>(&mut h, &k, &ctxs, &st, &di);
        let k = TensorDriftFlux1D::new(cfg, vertical);
        let (ctxs, pv, pg, pdv, pdg) = mk(0xD1F7_009);
        let st = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<1, _>(&mut h, &k, &ctxs, &st, &di);
        let k = TensorDriftGravity1D::new(cfg, vertical);
        let (ctxs, pv, pg, pdv, pdg) = mk(0xD1F7_00A);
        let st = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 3,
            npts,
            gdim: 1,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<1, _>(&mut h, &k, &ctxs, &st, &di);
        // Pipe composite like drift_flux_pipe_1d.rs.
        {
            let pipe = TensorResidualKernelSet::from_kernel(
                TensorResidualKernelSum::from_kernel(TensorDriftMomentumConvectionSplit1D::new(
                    cfg,
                ))
                .with(TensorDriftPressureGradient1D::new(cfg))
                .with(TensorDriftViscousStress1D::new(cfg))
                .with(TensorDriftPressureDivergence1D::new(cfg))
                .with(TensorDriftPressureAdvectionSplit1D::new(cfg))
                .with(TensorDriftPressureDiffusion1D::new(cfg))
                .with(TensorDriftVoidAdvectionSplit1D::new(cfg))
                .with(TensorDriftFlux1D::new(cfg, vertical))
                .with(TensorDriftGravity1D::new(cfg, vertical))
                .with(TensorDriftTurbulentDispersion1D::new(cfg, 0.01)),
            );
            let (ctxs, pv, pg, pdv, pdg) = mk(0xCC1D_001);
            let st = LaneState {
                nfields: 3,
                npts,
                gdim: 1,
                values: &pv,
                grads: &pg,
                field_indices: &[],
            };
            let di = LaneState {
                nfields: 3,
                npts,
                gdim: 1,
                values: &pdv,
                grads: &pdg,
                field_indices: &[],
            };
            hash_volume::<1, _>(&mut h, &pipe, &ctxs, &st, &di);
        }
        out.push(("drift_1d", h));
    }

    // --- group: drift 2D
    {
        let mut h = 0xcbf29ce484222325u64;
        let storage = lanes::VolumeStorage::new_2d(3);
        let npts = storage.npts();
        let cfg = drift_config_2d();
        let mut mk = |seed: u64| {
            let mut s = seed;
            let (mut vl, gl, dvl, dgl) = lanes::random_lane_buffers(4, npts, 2, &mut s);
            craft_alpha(&mut vl, npts, ALPHA_2D);
            let (pv, pg) = lanes::pack_lanes(4, npts, 2, &vl, &gl);
            let (pdv, pdg) = lanes::pack_lanes(4, npts, 2, &dvl, &dgl);
            let ctxs: Vec<TensorCtx<'_>> = (0..LANES).map(|l| storage.ctx(l)).collect();
            (ctxs, pv, pg, pdv, pdg)
        };
        let k = TensorDriftMomentumConvectionSplit2D::new(cfg);
        let (ctxs, pv, pg, pdv, pdg) = mk(0xD2F7_001);
        let st = LaneState {
            nfields: 4,
            npts,
            gdim: 2,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 4,
            npts,
            gdim: 2,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<2, _>(&mut h, &k, &ctxs, &st, &di);
        let k = TensorDriftPressureAdvectionSplit2D::new(cfg);
        let (ctxs, pv, pg, pdv, pdg) = mk(0xD2F7_002);
        let st = LaneState {
            nfields: 4,
            npts,
            gdim: 2,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 4,
            npts,
            gdim: 2,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<2, _>(&mut h, &k, &ctxs, &st, &di);
        let k = TensorDriftPressureDiffusion2D::new(cfg);
        let (ctxs, pv, pg, pdv, pdg) = mk(0xD2F7_003);
        let st = LaneState {
            nfields: 4,
            npts,
            gdim: 2,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 4,
            npts,
            gdim: 2,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<2, _>(&mut h, &k, &ctxs, &st, &di);
        let k = TensorDriftPressureDivergence2D::new(cfg);
        let (ctxs, pv, pg, pdv, pdg) = mk(0xD2F7_004);
        let st = LaneState {
            nfields: 4,
            npts,
            gdim: 2,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 4,
            npts,
            gdim: 2,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<2, _>(&mut h, &k, &ctxs, &st, &di);
        let k = TensorDriftPressureGradient2D::new(cfg);
        let (ctxs, pv, pg, pdv, pdg) = mk(0xD2F7_005);
        let st = LaneState {
            nfields: 4,
            npts,
            gdim: 2,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 4,
            npts,
            gdim: 2,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<2, _>(&mut h, &k, &ctxs, &st, &di);
        let k = TensorDriftViscousStress2D::new(cfg);
        let (ctxs, pv, pg, pdv, pdg) = mk(0xD2F7_006);
        let st = LaneState {
            nfields: 4,
            npts,
            gdim: 2,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 4,
            npts,
            gdim: 2,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<2, _>(&mut h, &k, &ctxs, &st, &di);
        let k = TensorDriftVoidAdvectionSplit2D::new(cfg);
        let (ctxs, pv, pg, pdv, pdg) = mk(0xD2F7_007);
        let st = LaneState {
            nfields: 4,
            npts,
            gdim: 2,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 4,
            npts,
            gdim: 2,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<2, _>(&mut h, &k, &ctxs, &st, &di);
        let k = TensorDriftFlux2D::new(cfg);
        let (ctxs, pv, pg, pdv, pdg) = mk(0xD2F7_008);
        let st = LaneState {
            nfields: 4,
            npts,
            gdim: 2,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 4,
            npts,
            gdim: 2,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<2, _>(&mut h, &k, &ctxs, &st, &di);
        let k = TensorDriftGravity2D::new(cfg);
        let (ctxs, pv, pg, pdv, pdg) = mk(0xD2F7_009);
        let st = LaneState {
            nfields: 4,
            npts,
            gdim: 2,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 4,
            npts,
            gdim: 2,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<2, _>(&mut h, &k, &ctxs, &st, &di);
        let k = TensorDriftTurbulentDispersion2D::new(cfg, 5e-2);
        let (ctxs, pv, pg, pdv, pdg) = mk(0xD2F7_00A);
        let st = LaneState {
            nfields: 4,
            npts,
            gdim: 2,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 4,
            npts,
            gdim: 2,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<2, _>(&mut h, &k, &ctxs, &st, &di);
        // Dix variant + cylinder composite like drift_flux_cylinder_tensor.rs.
        let dix = cfg.with_distribution(DistributionParameter::Dix);
        let k = TensorDriftVoidAdvectionSplit2D::new(dix);
        let (ctxs, pv, pg, pdv, pdg) = mk(0xD2F7_00E);
        let st = LaneState {
            nfields: 4,
            npts,
            gdim: 2,
            values: &pv,
            grads: &pg,
            field_indices: &[],
        };
        let di = LaneState {
            nfields: 4,
            npts,
            gdim: 2,
            values: &pdv,
            grads: &pdg,
            field_indices: &[],
        };
        hash_volume::<2, _>(&mut h, &k, &ctxs, &st, &di);
        {
            let cyl = TensorResidualKernelSum::from_kernel(
                TensorDriftMomentumConvectionSplit2D::new(cfg),
            )
            .with(TensorDriftPressureGradient2D::new(cfg))
            .with(TensorDriftViscousStress2D::new(cfg))
            .with(TensorDriftPressureDivergence2D::new(cfg))
            .with(TensorDriftPressureAdvectionSplit2D::new(cfg))
            .with(TensorDriftPressureDiffusion2D::new(cfg))
            .with(TensorDriftVoidAdvectionSplit2D::new(cfg))
            .with(TensorDriftFlux2D::new(cfg))
            .with(TensorDriftGravity2D::new(cfg))
            .with(TensorDriftTurbulentDispersion2D::new(cfg, 5e-2));
            let (ctxs, pv, pg, pdv, pdg) = mk(0xCC1D_002);
            let st = LaneState {
                nfields: 4,
                npts,
                gdim: 2,
                values: &pv,
                grads: &pg,
                field_indices: &[],
            };
            let di = LaneState {
                nfields: 4,
                npts,
                gdim: 2,
                values: &pdv,
                grads: &pdg,
                field_indices: &[],
            };
            hash_volume::<2, _>(&mut h, &cyl, &ctxs, &st, &di);
        }
        out.push(("drift_2d", h));
    }

    // --- group: boundary (EDAC + outflow + drift)
    {
        let mut h = 0xcbf29ce484222325u64;
        let npts = 4;
        // Simpler: build each kernel's facet states inline below.
        // Helper to build controlled (u,v,p) branch table states.
        let build_branch = |seed: u64| {
            let table: [[f64; 3]; 5] = [
                [2.0, 0.5, 1.0],
                [-1.5, -0.3, 0.5],
                [0.0, 0.0, 2.0],
                [1e-9, 0.2, -0.7],
                [-1e-9, -0.2, 0.9],
            ];
            let mut s = seed;
            let (mut vl, _, mut dvl, _) = lanes::random_lane_buffers(3, npts, 2, &mut s);
            for (lane, &[u, v, p]) in table.iter().enumerate() {
                for q in 0..npts {
                    vl[lane][0 * npts + q] = u;
                    vl[lane][1 * npts + q] = v;
                    vl[lane][2 * npts + q] = p;
                }
            }
            let (_, gl, _, dgl) = lanes::random_lane_buffers(3, npts, 2, &mut s);
            let (pv, _) = lanes::pack_lanes(3, npts, 2, &vl, &gl);
            let (pdv, _) = lanes::pack_lanes(3, npts, 2, &dvl, &dgl);
            (vl, dvl, pv, pdv)
        };
        // Boxed boundary kernels need explicit trait objects; evaluate each.
        {
            let storage = lanes::FacetStorage::new(npts);
            let ctxs: Vec<TensorFacetCtx<'_>> = (0..LANES).map(|l| storage.ctx(l)).collect();
            let empty: Vec<f64> = Vec::new();
            let (vl, dvl, pv, pdv) = build_branch(0xA11CE01);
            let st = LaneState {
                nfields: 3,
                npts,
                gdim: 2,
                values: &pv,
                grads: &empty,
                field_indices: &[],
            };
            let di = LaneState {
                nfields: 3,
                npts,
                gdim: 2,
                values: &pdv,
                grads: &empty,
                field_indices: &[],
            };
            let _ = (&vl, &dvl);
            let k = TensorKernelEdacNoSlipWall2D::new();
            hash_boundary(&mut h, &k, &ctxs, &st, &di);
            let k = TensorKernelEdacSlipWall2D::new();
            hash_boundary(&mut h, &k, &ctxs, &st, &di);
            let k = TensorKernelEdacSplitBoundaryFlux2D;
            hash_boundary(&mut h, &k, &ctxs, &st, &di);
            let k = TensorKernelEdacDongOutflow2D::new(1.0, 0.1, 1.0);
            hash_boundary(&mut h, &k, &ctxs, &st, &di);
            let k = TensorKernelEdacDongOutflow2D::new(1.0, 0.1, 1.0).with_split_flux();
            hash_boundary(&mut h, &k, &ctxs, &st, &di);
            let k = TensorKernelEdacDirectionalDoNothing2D::new(1.0);
            hash_boundary(&mut h, &k, &ctxs, &st, &di);
            let k = TensorKernelEdacDirectionalDoNothing2D::new(1.0).with_split_flux();
            hash_boundary(&mut h, &k, &ctxs, &st, &di);
            // Frozen-velocity advective outflow.
            {
                let ux_table = [1.5, -2.0, 0.0, 1e-12, -1e-12, 0.7, -0.4, 3.0];
                let mut ux = Vec::with_capacity(LANES * npts);
                let mut uy = Vec::with_capacity(LANES * npts);
                for lane in 0..LANES {
                    for _ in 0..npts {
                        ux.push(ux_table[lane]);
                        uy.push(0.0);
                    }
                }
                let k = TensorKernelAdvectionOutflow2D::new(
                    FrozenFacetField::new(npts, LANES, ux),
                    FrozenFacetField::new(npts, LANES, uy),
                );
                hash_boundary(&mut h, &k, &ctxs, &st, &di);
            }
            // Drift boundary kernels with crafted alpha.
            {
                let mut s = 0xBD1F_001u64;
                let (mut vl4, _, mut dvl4, _) = lanes::random_lane_buffers(4, npts, 2, &mut s);
                craft_alpha(&mut vl4, npts, ALPHA_2D);
                let dummy = vec![vec![0.0; 8 * npts]; LANES];
                let (pv4, _) = lanes::pack_lanes(4, npts, 2, &vl4, &dummy);
                let (pdv4, _) = lanes::pack_lanes(4, npts, 2, &dvl4, &dummy);
                let st4 = LaneState {
                    nfields: 4,
                    npts,
                    gdim: 2,
                    values: &pv4,
                    grads: &empty,
                    field_indices: &[],
                };
                let di4 = LaneState {
                    nfields: 4,
                    npts,
                    gdim: 2,
                    values: &pdv4,
                    grads: &empty,
                    field_indices: &[],
                };
                let k = TensorDriftSplitBoundaryFlux2D;
                hash_boundary(&mut h, &k, &ctxs, &st4, &di4);
                let k = TensorDriftDirectionalDoNothing2D::new(1000.0);
                hash_boundary(&mut h, &k, &ctxs, &st4, &di4);
                let k = TensorDriftDirectionalDoNothing2D::new(1000.0).with_split_flux();
                hash_boundary(&mut h, &k, &ctxs, &st4, &di4);
                let k = TensorDriftFreeSurface2D::new(drift_config_2d());
                hash_boundary(&mut h, &k, &ctxs, &st4, &di4);
                let k = TensorDriftFreeSurface2D::new(drift_config_2d()).with_penalty(1.0e3);
                hash_boundary(&mut h, &k, &ctxs, &st4, &di4);
            }
        }
        out.push(("boundary", h));
    }

    out
}

// Expected hashes (filled by PRINT_GOLDEN=1 run).
const EXPECTED_BASIC_1D: u64 = 0x0e3e407a73c9ef48;
const EXPECTED_BASIC_2D: u64 = 0x32d95a2173487ae1;
const EXPECTED_CONSERVATION_1D: u64 = 0x5cbf315fc2b565b1;
const EXPECTED_EDAC_1D: u64 = 0x6cfab6472f1eb9ae;
const EXPECTED_EDAC_2D: u64 = 0x65223b1294417aee;
const EXPECTED_DRIFT_1D: u64 = 0xafc7ca205702e52c;
const EXPECTED_DRIFT_2D: u64 = 0xc6dc694a62004506;
const EXPECTED_BOUNDARY: u64 = 0xd4994fed0ae3f7fa;

#[test]
fn golden_hashes_match() {
    let hashes = compute_hashes();
    if std::env::var("PRINT_GOLDEN").is_ok() {
        for (name, h) in &hashes {
            println!("GOLDEN {name} = {h:#018x} ({h})");
        }
        return;
    }
    let get = |name: &str| hashes.iter().find(|(n, _)| *n == name).unwrap().1;
    assert_eq!(get("basic_1d"), EXPECTED_BASIC_1D, "basic_1d hash mismatch");
    assert_eq!(get("basic_2d"), EXPECTED_BASIC_2D, "basic_2d hash mismatch");
    assert_eq!(
        get("conservation_1d"),
        EXPECTED_CONSERVATION_1D,
        "conservation_1d hash mismatch"
    );
    assert_eq!(get("edac_1d"), EXPECTED_EDAC_1D, "edac_1d hash mismatch");
    assert_eq!(get("edac_2d"), EXPECTED_EDAC_2D, "edac_2d hash mismatch");
    assert_eq!(get("drift_1d"), EXPECTED_DRIFT_1D, "drift_1d hash mismatch");
    assert_eq!(get("drift_2d"), EXPECTED_DRIFT_2D, "drift_2d hash mismatch");
    assert_eq!(get("boundary"), EXPECTED_BOUNDARY, "boundary hash mismatch");
}

#[test]
fn jacobian_action_matches_central_difference() {
    // 1D smooth representatives (interior states; no clamping kinks).
    {
        let storage = lanes::VolumeStorage::new_1d(4);
        let npts = storage.npts();
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(1, &storage, 0xFD_101);
        fd_check::<1, _>(
            &TensorKernelDiffusion::new(0.7),
            &ctxs,
            1,
            npts,
            &pv,
            &pg,
            &pdv,
            &pdg,
        );
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(1, &storage, 0xFD_102);
        fd_check::<1, _>(
            &TensorKernelAdvDiff::new(0.05, 1.5),
            &ctxs,
            1,
            npts,
            &pv,
            &pg,
            &pdv,
            &pdg,
        );
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(2, &storage, 0xFD_103);
        fd_check::<1, _>(
            &TensorKernelConservationLaw1D::new(LinearFlux1D),
            &ctxs,
            2,
            npts,
            &pv,
            &pg,
            &pdv,
            &pdg,
        );
        let cfg = edac_1d();
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(2, &storage, 0xFD_104);
        fd_check::<1, _>(
            &TensorKernelEdacNavierStokesSplit1D::new(cfg),
            &ctxs,
            2,
            npts,
            &pv,
            &pg,
            &pdv,
            &pdg,
        );
        // Drift 1D on interior alpha: remap palette-clamped values into
        // (0.05, 0.95) so closure clamps/min/max branches are not crossed by
        // the eps perturbation (documented; the palette hash above covers
        // the branches themselves).
        {
            let mut s = 0xFD_105u64;
            let (mut vl, gl, mut dvl, dgl) = lanes::random_lane_buffers(3, npts, 1, &mut s);
            for v in vl.iter_mut() {
                for q in 0..npts {
                    let a = v[ALPHA_1D * npts + q];
                    v[ALPHA_1D * npts + q] = 0.5 + 0.4 * (a / 2.0);
                }
            }
            for v in dvl.iter_mut() {
                for x in v.iter_mut() {
                    *x *= 0.1;
                }
            }
            let (pv, pg) = lanes::pack_lanes(3, npts, 1, &vl, &gl);
            let (pdv, pdg) = lanes::pack_lanes(3, npts, 1, &dvl, &dgl);
            let ctxs: Vec<TensorCtx<'_>> = (0..LANES).map(|l| storage.ctx(l)).collect();
            let cfg = drift_config_1d();
            let vertical = |_: &MaterialContext<'_>| std::f64::consts::FRAC_PI_2;
            fd_check::<1, _>(
                &TensorDriftFlux1D::new(cfg, vertical),
                &ctxs,
                3,
                npts,
                &pv,
                &pg,
                &pdv,
                &pdg,
            );
            fd_check::<1, _>(
                &TensorDriftGravity1D::new(cfg, vertical),
                &ctxs,
                3,
                npts,
                &pv,
                &pg,
                &pdv,
                &pdg,
            );
            fd_check::<1, _>(
                &TensorDriftVoidAdvectionSplit1D::new(cfg),
                &ctxs,
                3,
                npts,
                &pv,
                &pg,
                &pdv,
                &pdg,
            );
        }
    }
    // 2D smooth representatives.
    {
        let storage = lanes::VolumeStorage::new_2d(3);
        let npts = storage.npts();
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(1, &storage, 0xFD_201);
        fd_check::<2, _>(
            &TensorKernelDiffusion2D::new(0.7),
            &ctxs,
            1,
            npts,
            &pv,
            &pg,
            &pdv,
            &pdg,
        );
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(1, &storage, 0xFD_202);
        fd_check::<2, _>(
            &TensorKernelAdvection2D::new([1.0, -0.5]),
            &ctxs,
            1,
            npts,
            &pv,
            &pg,
            &pdv,
            &pdg,
        );
        let cfg = edac_2d();
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(3, &storage, 0xFD_203);
        fd_check::<2, _>(
            &TensorKernelEdacNavierStokesSplit2D::new(cfg),
            &ctxs,
            3,
            npts,
            &pv,
            &pg,
            &pdv,
            &pdg,
        );
        let (ctxs, pv, pg, pdv, pdg) = volume_fixture(3, &storage, 0xFD_204);
        fd_check::<2, _>(
            &TensorKernelEnergyAdvectionDiffusion2D::new(0.02),
            &ctxs,
            3,
            npts,
            &pv,
            &pg,
            &pdv,
            &pdg,
        );
        // Drift 2D interior alpha (same clamp-avoidance remap as 1D).
        {
            let mut s = 0xFD_205u64;
            let (mut vl, gl, mut dvl, dgl) = lanes::random_lane_buffers(4, npts, 2, &mut s);
            for v in vl.iter_mut() {
                for q in 0..npts {
                    let a = v[ALPHA_2D * npts + q];
                    v[ALPHA_2D * npts + q] = 0.5 + 0.4 * (a / 2.0);
                }
            }
            for v in dvl.iter_mut() {
                for x in v.iter_mut() {
                    *x *= 0.1;
                }
            }
            let (pv, pg) = lanes::pack_lanes(4, npts, 2, &vl, &gl);
            let (pdv, pdg) = lanes::pack_lanes(4, npts, 2, &dvl, &dgl);
            let ctxs: Vec<TensorCtx<'_>> = (0..LANES).map(|l| storage.ctx(l)).collect();
            let cfg = drift_config_2d();
            fd_check::<2, _>(
                &TensorDriftFlux2D::new(cfg),
                &ctxs,
                4,
                npts,
                &pv,
                &pg,
                &pdv,
                &pdg,
            );
            fd_check::<2, _>(
                &TensorDriftGravity2D::new(cfg),
                &ctxs,
                4,
                npts,
                &pv,
                &pg,
                &pdv,
                &pdg,
            );
        }
    }
    // Boundary smooth states (avoid un == 0 kink: shift lane velocities).
    {
        let npts = 4;
        let storage = lanes::FacetStorage::new(npts);
        let ctxs: Vec<TensorFacetCtx<'_>> = (0..LANES).map(|l| storage.ctx(l)).collect();
        let mut s = 0xFD_301u64;
        let (mut vl, gl, mut dvl, dgl) = lanes::random_lane_buffers(3, npts, 2, &mut s);
        for v in vl.iter_mut() {
            for q in 0..npts {
                v[0 * npts + q] += 1.5; // keep normal velocity away from 0
            }
        }
        for v in dvl.iter_mut() {
            for x in v.iter_mut() {
                *x *= 0.1;
            }
        }
        let (pv, _) = lanes::pack_lanes(3, npts, 2, &vl, &gl);
        let (pdv, _) = lanes::pack_lanes(3, npts, 2, &dvl, &dgl);
        fd_check_boundary(
            &TensorKernelEdacDongOutflow2D::new(1.0, 0.1, 1.0),
            &ctxs,
            3,
            npts,
            &pv,
            &pdv,
        );
        fd_check_boundary(
            &TensorKernelEdacDirectionalDoNothing2D::new(1.0),
            &ctxs,
            3,
            npts,
            &pv,
            &pdv,
        );
        fd_check_boundary(
            &TensorKernelEdacNoSlipWall2D::new(),
            &ctxs,
            3,
            npts,
            &pv,
            &pdv,
        );
    }
}
