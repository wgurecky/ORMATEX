//! Shared helpers for lane-packed (`LANES` = 8) tensor kernel tests.
//!
//! Generic packing/context builders for the single lane-packed kernel
//! interface: deterministic pseudo-random per-lane buffers
//! ([`random_lane_buffers`]), packing into [`LaneState`] layout
//! ([`pack_lanes`]), owned volume/facet [`TensorCtx`]/[`TensorFacetCtx`]
//! backing storage ([`VolumeStorage`], [`FacetStorage`]), and scalar PRNG
//! utilities. Bitwise math is pinned by `tests/kernel_golden.rs`.
//!
//! Typical use for a volume kernel:
//!
//! ```ignore
//! mod common;
//! use common::lanes as lanes;
//!
//! // 1. Per-lane scalar buffers in `CellState` layout.
//! let mut seed = 0x1234_5678u64;
//! let (values_l, grads_l, dir_values_l, dir_grads_l) =
//!     lanes::random_lane_buffers(nfields, npts, gdim, &mut seed);
//! // 2. Pack them into lane-packed buffers.
//! let (packed_values, packed_grads) =
//!     lanes::pack_lanes(nfields, npts, gdim, &values_l, &grads_l);
//! // 3. Owned context storage, then one `TensorCtx` per lane.
//! let storage = lanes::VolumeStorage::new_2d(3);
//! let ctxs: Vec<ormatex_sem_nd::TensorCtx<'_>> =
//!     (0..ormatex_sem_nd::LANES).map(|l| storage.ctx(l)).collect();
//! ```
//!
//! Lane independence is covered implicitly: every lane gets distinct random
//! data, so cross-lane contamination changes lane outputs.

use ormatex_sem_nd::{
    CellMeta, CellState, FacetMeta, LaneState, Lanes, StateTensorBoundaryIntegrator, TensorCtx,
    TensorFacetCtx, TensorResidualKernel, LANES,
};

/// Deterministic pseudo-random generator (no external crates).
///
/// # Arguments
/// * `seed` - generator state, updated in place.
///
/// # Returns
/// A deterministic `f64` in `[0, 1)`.
pub fn prng(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = (*seed >> 11) | 0x3FF0_0000_0000_0000;
    f64::from_bits(bits) - 1.0
}

/// Deterministic pseudo-random value in `[lo, hi)`.
///
/// # Arguments
/// * `seed` - generator state, updated in place.
/// * `lo` - inclusive lower bound.
/// * `hi` - exclusive upper bound.
///
/// # Returns
/// `lo + (hi - lo) * prng(seed)`.
pub fn prng_range(seed: &mut u64, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * prng(seed)
}

/// Build per-lane scalar buffers in [`CellState`] layout with random data.
///
/// Every lane gets distinct values and gradients, so a lane implementation
/// that mixes lanes fails the per-lane comparison.
///
/// # Arguments
/// * `nfields` - scalar field count.
/// * `npts` - quadrature-point count.
/// * `gdim` - geometric dimension.
/// * `seed` - generator state, updated in place.
///
/// # Returns
/// `(values, grads, dir_values, dir_grads)`, each a `Vec` of length `LANES`
/// holding one `CellState`-layout buffer per lane (`values[l][f * npts + q]`,
/// `grads[l][(f * gdim + d) * npts + q]`).
#[allow(clippy::type_complexity)]
pub fn random_lane_buffers(
    nfields: usize,
    npts: usize,
    gdim: usize,
    seed: &mut u64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut values = Vec::with_capacity(LANES);
    let mut grads = Vec::with_capacity(LANES);
    let mut dir_values = Vec::with_capacity(LANES);
    let mut dir_grads = Vec::with_capacity(LANES);
    for _ in 0..LANES {
        values.push(
            (0..nfields * npts)
                .map(|_| prng_range(seed, -2.0, 2.0))
                .collect(),
        );
        grads.push(
            (0..nfields * gdim * npts)
                .map(|_| prng_range(seed, -3.0, 3.0))
                .collect(),
        );
        dir_values.push(
            (0..nfields * npts)
                .map(|_| prng_range(seed, -1.5, 1.5))
                .collect(),
        );
        dir_grads.push(
            (0..nfields * gdim * npts)
                .map(|_| prng_range(seed, -2.5, 2.5))
                .collect(),
        );
    }
    (values, grads, dir_values, dir_grads)
}

/// Pack per-lane scalar buffers into lane-packed [`LaneState`] buffers.
///
/// # Arguments
/// * `nfields` - scalar field count.
/// * `npts` - quadrature-point count.
/// * `gdim` - geometric dimension.
/// * `values_lanes` - `LANES` scalar value buffers (`values[l][f * npts + q]`).
/// * `grads_lanes` - `LANES` scalar gradient buffers
///   (`grads[l][(f * gdim + d) * npts + q]`).
///
/// # Returns
/// `(packed_values, packed_grads)` with
/// `packed_values[(f * npts + q) * LANES + lane]` and
/// `packed_grads[((f * gdim + d) * npts + q) * LANES + lane]`.
pub fn pack_lanes(
    nfields: usize,
    npts: usize,
    gdim: usize,
    values_lanes: &[Vec<f64>],
    grads_lanes: &[Vec<f64>],
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(values_lanes.len(), LANES, "need one value buffer per lane");
    assert_eq!(
        grads_lanes.len(),
        LANES,
        "need one gradient buffer per lane"
    );
    let mut packed_values = vec![0.0; nfields * npts * LANES];
    let mut packed_grads = vec![0.0; nfields * gdim * npts * LANES];
    for lane in 0..LANES {
        for f in 0..nfields {
            for q in 0..npts {
                packed_values[(f * npts + q) * LANES + lane] = values_lanes[lane][f * npts + q];
                for d in 0..gdim {
                    packed_grads[((f * gdim + d) * npts + q) * LANES + lane] =
                        grads_lanes[lane][(f * gdim + d) * npts + q];
                }
            }
        }
    }
    (packed_values, packed_grads)
}

/// Owned backing storage for plausible volume [`TensorCtx`] values.
///
/// All lanes share the same geometry slices (exactly the padded-lane case);
/// per-lane identity comes from `time`, `cell.local_index`, and `cell_size`.
/// The storage owns every borrowed slice; build one [`TensorCtx`] per lane
/// with [`ctx`](Self::ctx) while the storage is alive.
pub struct VolumeStorage {
    wts: Vec<f64>,
    jdets: Vec<f64>,
    wdet: Vec<f64>,
    points: Vec<f64>,
    diff: Vec<f64>,
    q2l: Vec<usize>,
    jinv: Vec<f64>,
    n1d: usize,
    npts: usize,
    gdim: usize,
}

impl VolumeStorage {
    /// Backing storage for a 1D interval with `n1d` GLL nodes.
    ///
    /// # Arguments
    /// * `n1d` - one-dimensional node count; also the quadrature count.
    ///
    /// # Returns
    /// Storage with `jinv[q]` scalar inverse Jacobians (1D layout).
    pub fn new_1d(n1d: usize) -> Self {
        let npts = n1d;
        let gdim = 1;
        Self::build(n1d, npts, gdim)
    }

    /// Backing storage for a 2D quadrilateral with `n1d` GLL nodes per direction.
    ///
    /// # Arguments
    /// * `n1d` - one-dimensional node count; quadrature count is `n1d * n1d`.
    ///
    /// # Returns
    /// Storage with `jinv[q * 4..q * 4 + 4]` inverse Jacobians (2D layout).
    pub fn new_2d(n1d: usize) -> Self {
        let npts = n1d * n1d;
        let gdim = 2;
        Self::build(n1d, npts, gdim)
    }

    fn build(n1d: usize, npts: usize, gdim: usize) -> Self {
        let wts = vec![0.5; npts];
        let jdets: Vec<f64> = (0..npts).map(|q| 0.2 + 0.01 * q as f64).collect();
        let wdet: Vec<f64> = wts.iter().zip(&jdets).map(|(w, j)| w * j).collect();
        let points: Vec<f64> = (0..npts * gdim).map(|i| 0.1 * i as f64 - 0.3).collect();
        // Nontrivial differentiation matrix (row-major, `n1d x n1d`).
        let diff: Vec<f64> = (0..n1d * n1d)
            .map(|i| 0.5 * (i as f64) - 0.25 * (i % n1d) as f64 - 1.0)
            .collect();
        let q2l: Vec<usize> = (0..npts).collect();
        let jinv: Vec<f64> = if gdim == 1 {
            (0..npts).map(|q| 1.0 + 0.05 * q as f64).collect()
        } else {
            (0..npts)
                .flat_map(|q| {
                    let s = 1.0 + 0.05 * q as f64;
                    [s, 0.01 * q as f64, -0.02 * q as f64, 1.0 / s]
                })
                .collect()
        };
        Self {
            wts,
            jdets,
            wdet,
            points,
            diff,
            q2l,
            jinv,
            n1d,
            npts,
            gdim,
        }
    }

    /// Number of quadrature points (`n1d` in 1D, `n1d * n1d` in 2D).
    pub fn npts(&self) -> usize {
        self.npts
    }

    /// Geometric dimension (1 or 2).
    pub fn gdim(&self) -> usize {
        self.gdim
    }

    /// Build the lane-`lane` context (distinct time, cell index, cell size).
    ///
    /// # Arguments
    /// * `lane` - lane index in `0..LANES`; selects `time`, `cell.local_index`,
    ///   and `cell_size` (`0.4 + 0.1 * lane`).
    ///
    /// # Returns
    /// [`TensorCtx`] borrowing this storage.
    pub fn ctx(&self, lane: usize) -> TensorCtx<'_> {
        assert!(lane < LANES, "lane index out of range");
        TensorCtx {
            time: 0.1 * lane as f64,
            cell: CellMeta {
                local_index: lane,
                physical_region: None,
            },
            n1d: self.n1d,
            npts: self.npts,
            wts: &self.wts,
            jdets: &self.jdets,
            wdet: &self.wdet,
            points: &self.points,
            differentiation: &self.diff,
            q_to_local: &self.q2l,
            jinv: &self.jinv,
            cell_size: 0.4 + 0.1 * lane as f64,
        }
    }
}

/// Owned backing storage for plausible facet [`TensorFacetCtx`] values.
///
/// All lanes share the same facet slices (exactly the padded-lane case);
/// per-lane identity comes from `time` and `facet.local_index`.
pub struct FacetStorage {
    wts: Vec<f64>,
    jdet: Vec<f64>,
    points: Vec<f64>,
    normal: Vec<f64>,
    npts: usize,
}

impl FacetStorage {
    /// Backing storage for a facet with `npts` quadrature points.
    ///
    /// # Arguments
    /// * `npts` - facet quadrature-point count.
    ///
    /// # Returns
    /// Storage with 2D physical points and a unit x-normal.
    pub fn new(npts: usize) -> Self {
        let wts = vec![0.5; npts];
        let jdet: Vec<f64> = (0..npts).map(|q| 0.3 + 0.02 * q as f64).collect();
        let points: Vec<f64> = (0..npts * 2).map(|i| 0.15 * i as f64 - 0.2).collect();
        let normal = vec![1.0, 0.0];
        Self {
            wts,
            jdet,
            points,
            normal,
            npts,
        }
    }

    /// Facet quadrature-point count.
    pub fn npts(&self) -> usize {
        self.npts
    }

    /// Build the lane-`lane` facet context.
    ///
    /// # Arguments
    /// * `lane` - lane index in `0..LANES`; selects `time` and
    ///   `facet.local_index`.
    ///
    /// # Returns
    /// [`TensorFacetCtx`] borrowing this storage.
    pub fn ctx(&self, lane: usize) -> TensorFacetCtx<'_> {
        assert!(lane < LANES, "lane index out of range");
        TensorFacetCtx {
            time: 0.05 * lane as f64,
            facet: FacetMeta {
                local_index: lane,
                physical_region: None,
            },
            npts: self.npts,
            wts: &self.wts,
            jfacet_det: &self.jdet,
            points: &self.points,
            normal: &self.normal,
        }
    }
}

/// Replicate one volume `TensorCtx` across all lanes (shared borrows).
///
/// # Arguments
/// * `ctx` - source context; every lane borrows the same slices.
///
/// # Returns
/// Array of `LANES` contexts borrowing `ctx`'s slices with its time/cell.
pub fn broadcast_ctxs<'a>(ctx: &TensorCtx<'a>) -> [TensorCtx<'a>; LANES] {
    std::array::from_fn(|_| TensorCtx {
        time: ctx.time,
        cell: ctx.cell,
        n1d: ctx.n1d,
        npts: ctx.npts,
        wts: ctx.wts,
        jdets: ctx.jdets,
        wdet: ctx.wdet,
        points: ctx.points,
        differentiation: ctx.differentiation,
        q_to_local: ctx.q_to_local,
        jinv: ctx.jinv,
        cell_size: ctx.cell_size,
    })
}

/// Replicate one facet `TensorFacetCtx` across all lanes (shared borrows).
///
/// # Arguments
/// * `ctx` - source facet context; every lane borrows the same slices.
///
/// # Returns
/// Array of `LANES` facet contexts borrowing `ctx`'s slices.
pub fn broadcast_facet_ctxs<'a>(ctx: &TensorFacetCtx<'a>) -> [TensorFacetCtx<'a>; LANES] {
    std::array::from_fn(|_| TensorFacetCtx {
        time: ctx.time,
        facet: ctx.facet,
        npts: ctx.npts,
        wts: ctx.wts,
        jfacet_det: ctx.jfacet_det,
        points: ctx.points,
        normal: ctx.normal,
    })
}

/// Replicate one scalar `CellState` across all lanes.
///
/// # Arguments
/// * `state` - scalar state to broadcast.
///
/// # Returns
/// `(packed_values, packed_grads)` with every lane equal to `state`.
pub fn broadcast_state(state: &CellState<'_>) -> (Vec<f64>, Vec<f64>) {
    let mut packed_values = vec![0.0; state.nfields * state.npts * LANES];
    let mut packed_grads = vec![0.0; state.nfields * state.gdim * state.npts * LANES];
    for f in 0..state.nfields {
        for q in 0..state.npts {
            for lane in 0..LANES {
                packed_values[(f * state.npts + q) * LANES + lane] = state.value(f, q);
                for d in 0..state.gdim {
                    packed_grads[((f * state.gdim + d) * state.npts + q) * LANES + lane] =
                        state.grad(f, q, d);
                }
            }
        }
    }
    (packed_values, packed_grads)
}

/// Evaluate a 1D volume kernel on a broadcast state and return lane 0.
///
/// Broadcasts `state` to all lanes, calls the lane-packed residual, asserts
/// every lane is bitwise equal (lane independence), and returns lane 0's
/// triple, so scalar-era expected values transfer unchanged.
///
/// # Arguments
/// * `kernel` - kernel under test.
/// * `ctx` - source context, replicated across lanes.
/// * `state` - scalar state, broadcast across lanes.
/// * `equation` - output equation index.
/// * `q` - quadrature-point index.
///
/// # Returns
/// Lane 0's `[f0, f1_x, f1_y]` triple.
pub fn volume_residual_1d<K: TensorResidualKernel<1> + ?Sized>(
    kernel: &K,
    ctx: &TensorCtx<'_>,
    state: &CellState<'_>,
    equation: usize,
    q: usize,
) -> [f64; 3] {
    let ctxs = broadcast_ctxs(ctx);
    let (packed_values, packed_grads) = broadcast_state(state);
    let lane_state = LaneState {
        nfields: state.nfields,
        npts: state.npts,
        gdim: state.gdim,
        values: &packed_values,
        grads: &packed_grads,
        field_indices: state.field_indices,
    };
    let mut f0 = [0.0; LANES];
    let mut f1x = [0.0; LANES];
    let mut f1y = [0.0; LANES];
    kernel.tensor_residual(&ctxs, &lane_state, equation, q, &mut f0, &mut f1x, &mut f1y);
    for lane in 1..LANES {
        assert_eq!(
            f0[lane].to_bits(),
            f0[0].to_bits(),
            "broadcast lane {lane} f0"
        );
        assert_eq!(
            f1x[lane].to_bits(),
            f1x[0].to_bits(),
            "broadcast lane {lane} f1x"
        );
        assert_eq!(
            f1y[lane].to_bits(),
            f1y[0].to_bits(),
            "broadcast lane {lane} f1y"
        );
    }
    [f0[0], f1x[0], f1y[0]]
}

/// Evaluate a 1D volume kernel action on broadcast states and return lane 0.
///
/// Same broadcast contract as [`volume_residual_1d`] for the Gateaux direction.
///
/// # Arguments
/// * `kernel` - kernel under test.
/// * `ctx` - source context, replicated across lanes.
/// * `state` - scalar linearization point, broadcast across lanes.
/// * `direction` - scalar Gateaux direction, broadcast across lanes.
/// * `equation` - output equation index.
/// * `q` - quadrature-point index.
///
/// # Returns
/// Lane 0's `[df0, df1_x, df1_y]` triple.
pub fn volume_action_1d<K: TensorResidualKernel<1> + ?Sized>(
    kernel: &K,
    ctx: &TensorCtx<'_>,
    state: &CellState<'_>,
    direction: &CellState<'_>,
    equation: usize,
    q: usize,
) -> [f64; 3] {
    let ctxs = broadcast_ctxs(ctx);
    let (packed_values, packed_grads) = broadcast_state(state);
    let (packed_dir_values, packed_dir_grads) = broadcast_state(direction);
    let lane_state = LaneState {
        nfields: state.nfields,
        npts: state.npts,
        gdim: state.gdim,
        values: &packed_values,
        grads: &packed_grads,
        field_indices: state.field_indices,
    };
    let lane_dir = LaneState {
        nfields: direction.nfields,
        npts: direction.npts,
        gdim: direction.gdim,
        values: &packed_dir_values,
        grads: &packed_dir_grads,
        field_indices: direction.field_indices,
    };
    let mut f0 = [0.0; LANES];
    let mut f1x = [0.0; LANES];
    let mut f1y = [0.0; LANES];
    kernel.tensor_jacobian_action(
        &ctxs,
        &lane_state,
        &lane_dir,
        equation,
        q,
        &mut f0,
        &mut f1x,
        &mut f1y,
    );
    for lane in 1..LANES {
        assert_eq!(
            f0[lane].to_bits(),
            f0[0].to_bits(),
            "broadcast lane {lane} j0"
        );
        assert_eq!(
            f1x[lane].to_bits(),
            f1x[0].to_bits(),
            "broadcast lane {lane} j1x"
        );
        assert_eq!(
            f1y[lane].to_bits(),
            f1y[0].to_bits(),
            "broadcast lane {lane} j1y"
        );
    }
    [f0[0], f1x[0], f1y[0]]
}

/// Evaluate a 2D volume kernel on a broadcast state and return lane 0.
///
/// Same broadcast contract as [`volume_residual_1d`] for
/// `TensorResidualKernel<2>` kernels.
///
/// # Arguments
/// * `kernel` - kernel under test.
/// * `ctx` - source context, replicated across lanes.
/// * `state` - scalar state, broadcast across lanes.
/// * `equation` - output equation index.
/// * `q` - quadrature-point index.
///
/// # Returns
/// Lane 0's `[f0, f1_x, f1_y]` triple.
pub fn volume_residual_2d<K: TensorResidualKernel<2> + ?Sized>(
    kernel: &K,
    ctx: &TensorCtx<'_>,
    state: &CellState<'_>,
    equation: usize,
    q: usize,
) -> [f64; 3] {
    let ctxs = broadcast_ctxs(ctx);
    let (packed_values, packed_grads) = broadcast_state(state);
    let lane_state = LaneState {
        nfields: state.nfields,
        npts: state.npts,
        gdim: state.gdim,
        values: &packed_values,
        grads: &packed_grads,
        field_indices: state.field_indices,
    };
    let mut f0 = [0.0; LANES];
    let mut f1x = [0.0; LANES];
    let mut f1y = [0.0; LANES];
    kernel.tensor_residual(&ctxs, &lane_state, equation, q, &mut f0, &mut f1x, &mut f1y);
    for lane in 1..LANES {
        assert_eq!(
            f0[lane].to_bits(),
            f0[0].to_bits(),
            "broadcast lane {lane} f0"
        );
        assert_eq!(
            f1x[lane].to_bits(),
            f1x[0].to_bits(),
            "broadcast lane {lane} f1x"
        );
        assert_eq!(
            f1y[lane].to_bits(),
            f1y[0].to_bits(),
            "broadcast lane {lane} f1y"
        );
    }
    [f0[0], f1x[0], f1y[0]]
}

/// Evaluate a 2D volume kernel action on broadcast states and return lane 0.
///
/// Same broadcast contract as [`volume_action_1d`] for
/// `TensorResidualKernel<2>` kernels.
///
/// # Arguments
/// * `kernel` - kernel under test.
/// * `ctx` - source context, replicated across lanes.
/// * `state` - scalar linearization point, broadcast across lanes.
/// * `direction` - scalar Gateaux direction, broadcast across lanes.
/// * `equation` - output equation index.
/// * `q` - quadrature-point index.
///
/// # Returns
/// Lane 0's `[df0, df1_x, df1_y]` triple.
pub fn volume_action_2d<K: TensorResidualKernel<2> + ?Sized>(
    kernel: &K,
    ctx: &TensorCtx<'_>,
    state: &CellState<'_>,
    direction: &CellState<'_>,
    equation: usize,
    q: usize,
) -> [f64; 3] {
    let ctxs = broadcast_ctxs(ctx);
    let (packed_values, packed_grads) = broadcast_state(state);
    let (packed_dir_values, packed_dir_grads) = broadcast_state(direction);
    let lane_state = LaneState {
        nfields: state.nfields,
        npts: state.npts,
        gdim: state.gdim,
        values: &packed_values,
        grads: &packed_grads,
        field_indices: state.field_indices,
    };
    let lane_dir = LaneState {
        nfields: direction.nfields,
        npts: direction.npts,
        gdim: direction.gdim,
        values: &packed_dir_values,
        grads: &packed_dir_grads,
        field_indices: direction.field_indices,
    };
    let mut f0 = [0.0; LANES];
    let mut f1x = [0.0; LANES];
    let mut f1y = [0.0; LANES];
    kernel.tensor_jacobian_action(
        &ctxs,
        &lane_state,
        &lane_dir,
        equation,
        q,
        &mut f0,
        &mut f1x,
        &mut f1y,
    );
    for lane in 1..LANES {
        assert_eq!(
            f0[lane].to_bits(),
            f0[0].to_bits(),
            "broadcast lane {lane} j0"
        );
        assert_eq!(
            f1x[lane].to_bits(),
            f1x[0].to_bits(),
            "broadcast lane {lane} j1x"
        );
        assert_eq!(
            f1y[lane].to_bits(),
            f1y[0].to_bits(),
            "broadcast lane {lane} j1y"
        );
    }
    [f0[0], f1x[0], f1y[0]]
}

/// Evaluate a boundary kernel on a broadcast facet state and return lane 0.
///
/// Broadcasts `state` to all lanes, calls the lane-packed trace residual,
/// asserts every lane is bitwise equal, and returns lane 0's flux, so
/// scalar-era expected values transfer unchanged.
///
/// # Arguments
/// * `kernel` - boundary kernel under test.
/// * `ctx` - source facet context, replicated across lanes.
/// * `state` - scalar facet state, broadcast across lanes.
/// * `equation` - output equation index.
/// * `q` - facet quadrature-point index.
///
/// # Returns
/// Lane 0's scalar trace flux.
pub fn boundary_residual<K: StateTensorBoundaryIntegrator<2> + ?Sized>(
    kernel: &K,
    ctx: &TensorFacetCtx<'_>,
    state: &CellState<'_>,
    equation: usize,
    q: usize,
) -> f64 {
    let ctxs = broadcast_facet_ctxs(ctx);
    let (packed_values, packed_grads) = broadcast_state(state);
    let lane_state = LaneState {
        nfields: state.nfields,
        npts: state.npts,
        gdim: state.gdim,
        values: &packed_values,
        grads: &packed_grads,
        field_indices: state.field_indices,
    };
    let mut out: Lanes = [0.0; LANES];
    kernel.tensor_residual(&ctxs, &lane_state, equation, q, &mut out);
    for lane in 1..LANES {
        assert_eq!(
            out[lane].to_bits(),
            out[0].to_bits(),
            "broadcast lane {lane}"
        );
    }
    out[0]
}

/// Evaluate a boundary kernel action on broadcast facet states and return lane 0.
///
/// Same broadcast contract as [`boundary_residual`] for the Gateaux direction.
///
/// # Arguments
/// * `kernel` - boundary kernel under test.
/// * `ctx` - source facet context, replicated across lanes.
/// * `state` - scalar linearization point, broadcast across lanes.
/// * `direction` - scalar Gateaux direction, broadcast across lanes.
/// * `equation` - output equation index.
/// * `q` - facet quadrature-point index.
///
/// # Returns
/// Lane 0's scalar linearized trace flux.
pub fn boundary_action<K: StateTensorBoundaryIntegrator<2> + ?Sized>(
    kernel: &K,
    ctx: &TensorFacetCtx<'_>,
    state: &CellState<'_>,
    direction: &CellState<'_>,
    equation: usize,
    q: usize,
) -> f64 {
    let ctxs = broadcast_facet_ctxs(ctx);
    let (packed_values, packed_grads) = broadcast_state(state);
    let (packed_dir_values, packed_dir_grads) = broadcast_state(direction);
    let lane_state = LaneState {
        nfields: state.nfields,
        npts: state.npts,
        gdim: state.gdim,
        values: &packed_values,
        grads: &packed_grads,
        field_indices: state.field_indices,
    };
    let lane_dir = LaneState {
        nfields: direction.nfields,
        npts: direction.npts,
        gdim: direction.gdim,
        values: &packed_dir_values,
        grads: &packed_dir_grads,
        field_indices: direction.field_indices,
    };
    let mut out: Lanes = [0.0; LANES];
    kernel.tensor_jacobian_action(&ctxs, &lane_state, &lane_dir, equation, q, &mut out);
    for lane in 1..LANES {
        assert_eq!(
            out[lane].to_bits(),
            out[0].to_bits(),
            "broadcast lane {lane}"
        );
    }
    out[0]
}
