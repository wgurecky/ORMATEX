//! Dedicated interfaces separating weak, tensor-product, and linear/bilinear assembly paths.
//!
//! `sem_1d.rs` / `sem_2d.rs` previously exposed every pathway as inherent
//! `SEM*Problem` methods (`assemble_*` / `apply_*`), which cluttered the API:
//! weak-form quadrature, sum-factorized tensor, and state-independent
//! bilinear/linear forms were indistinguishable at the call site.
//!
//! These three traits give each family its own interface with its own docs:
//! - [`WeakResidualOps`]: traditional `ResidualKernel` quadrature
//!   (`assemble_residual`, `assemble_residual_jacobian`, `apply_jacobian[_into]`).
//! - [`TensorResidualOps`]: sum-factorized `TensorResidualKernel<D>` paths
//!   (`assemble_tensor_residual`, `assemble_tensor_jacobian`,
//!   `apply_tensor_jacobian_into`). Replaces the old private
//!   `assemble_tensor_residual_static` name.
//! - [`BilinearOps`]: state-independent `BilinearForm` / `LinearForm` paths
//!   (`assemble_bilinear`, `assemble_linear`, Dirichlet correction,
//!   lumped mass, natural boundaries).
//!
//! Complete (volume + state-dependent boundary) combinations and the
//! `Weak` / `Tensor` / `Mixed` residual operators live in
//! `sem_1d::operators` / `sem_2d::operators` and compose these traits.

use faer::prelude::*;
use faer::sparse::SparseColMat;

use crate::kernels::{BilinearForm, LinearForm, ResidualKernel, TensorResidualKernel};

// ponytail: traits keep static dispatch (generic <K>), no dyn in hot loops.
/// Traditional weak-form residual assembly (`ResidualKernel`).
///
/// Mathematics: for state `U`, `R(U) = Σ_e E_eᵀ r_e(U_e)` with
/// `r_e,i = Σ_q w_q·J_q·residual_integrand(ctx, U, eq, q, i)`,
/// `J(U) = dR/du`, `(Jv)_e = J_e·v_e` scattered with `E_eᵀ`.
///
/// When applicable: any `K: ResidualKernel`. Use when the kernel does not
/// implement the tensor fast path, for non-affine / non-collocated physics,
/// or when exact weak quadrature is required. For sum-factorized GLL
/// collocation see [`TensorResidualOps`]; for state-independent stiffness see
/// [`BilinearOps`].
pub trait WeakResidualOps {
    /// Assemble the positive weak spatial residual `R(U)`.
    ///
    /// Inputs: `time`, `kernel`, `state` (`system_size × 1`).
    /// Output: global residual `Vec` of length `system_size`.
    fn assemble_residual<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
    ) -> Vec<f64>;

    /// Assemble `dR/du` for a state-aware kernel at `state`.
    ///
    /// Inputs: `time`, `kernel`, `state` (`system_size × 1`).
    /// Output: square sparse Jacobian (`system_size × system_size`).
    /// Performance: full `ndofs²` triplets per cell; prefer matrix-free
    /// `apply_jacobian` inside nonlinear solves.
    fn assemble_residual_jacobian<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
    ) -> SparseColMat<usize, f64>;

    /// Apply `dR/du(state)` to one or more direction columns without building
    /// a global sparse Jacobian.
    ///
    /// Inputs: `state` (`N×1`), `direction` (`N×ncols`).
    /// Output: `N×ncols` action. Prefer `*_into` to reuse storage.
    fn apply_jacobian<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
    ) -> Mat<f64>;

    /// Apply `dR/du(state)` into caller-provided storage.
    ///
    /// Inputs/outputs mirror [`WeakResidualOps::apply_jacobian`]; `out` must
    /// be `N×ncols` and is fully overwritten.
    fn apply_jacobian_into<K: ResidualKernel + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        out: MatMut<'_, f64>,
    );
}

/// Sum-factorized tensor-product residual assembly (`TensorResidualKernel<D>`).
///
/// Mathematics: same `R(U) = Σ_e E_eᵀ r_e` as weak, but cell interpolation,
/// differentiation, and integration exploit the 1D tensor structure
/// (`O(p^{d+1})` instead of `O(p^{2d})`) with GLL collocation and SIMD-over-
/// element batching (`SIMD_CELL_WIDTH` lanes, color-disjoint scatters).
///
/// When applicable: only when `K: TensorResidualKernel<D>` for the problem
/// dimension `D` (1 or 2). Falls back to scalar tail for non-uniform /
/// boundary cells. Use for performance-critical volume physics; otherwise
/// use [`WeakResidualOps`].
pub trait TensorResidualOps<const D: usize> {
    /// Assemble the tensor-product residual `R(U)`.
    ///
    /// Inputs: `time`, `kernel`, `state` (`N×1`). Output: `N` residual.
    /// Replaces the old `assemble_tensor_residual_static` name.
    fn assemble_tensor_residual<K: TensorResidualKernel<D> + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
    ) -> Vec<f64>;

    /// Assemble the tensor-product Jacobian `dR/du` at `state` via
    /// unit-impulse columns through the tensor action.
    ///
    /// Inputs/outputs mirror the weak Jacobian. Performance: same sparsity,
    /// cheaper cell formation via sum factorization.
    fn assemble_tensor_jacobian<K: TensorResidualKernel<D> + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
    ) -> SparseColMat<usize, f64>;

    /// Matrix-free tensor Jacobian action `(Jv)` into caller storage.
    ///
    /// Inputs: `state` (`N×1`), `direction` (`N×ncols`), `out` (`N×ncols`,
    /// fully overwritten). Batched over SIMD lanes; no global matrix.
    fn apply_tensor_jacobian_into<K: TensorResidualKernel<D> + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        out: MatMut<'_, f64>,
    );
}

/// State-independent linear/bilinear assembly (`BilinearForm` / `LinearForm`).
///
/// Mathematics: `A_ij = Σ_e Σ_q w_q·J_q·integrand(eq,unk,q,i,j)`,
/// `b_i = Σ_e Σ_q w_q·J_q·integrand(eq,q,i)`, Dirichlet correction
/// `b -= A_fb·u_b`, lumped GLL mass `M_ii = Σ_q w_q·J_q·φ_i(q)`.
///
/// When applicable: any `K: BilinearForm` / `LinearForm`. If the form reports
/// `supports_tensor_bilinear[_1d]()` the assembly transparently uses the
/// tensor fast path (`assemble_tensor_bilinear`); otherwise full quadrature.
/// Use for stiffness/mass/RHS, not for state-dependent residuals (see
/// [`WeakResidualOps`] / [`TensorResidualOps`]).
pub trait BilinearOps {
    /// Assemble a reduced sparse matrix via `kernel.assemble_local` per cell.
    ///
    /// Inputs: `time`, `kernel`. Output: `N×N` sparse. Periodic DOFs combined,
    /// eliminated DOFs omitted at scatter.
    fn assemble_bilinear<K: BilinearForm + Sync>(
        &self,
        time: f64,
        kernel: &K,
    ) -> SparseColMat<usize, f64>;

    /// Assemble a reduced volume RHS from a `LinearForm`.
    ///
    /// Inputs: `time`, `kernel`. Output: `N` RHS.
    fn assemble_linear<K: LinearForm>(&self, time: f64, kernel: &K) -> Vec<f64>;

    /// Assemble a linear RHS and apply the nonzero Dirichlet correction.
    fn assemble_linear_with_dirichlet<B, L>(&self, time: f64, bilinear: &B, linear: &L) -> Vec<f64>
    where
        B: BilinearForm,
        L: LinearForm;

    /// Apply the prescribed-DOF contribution `-A_fb·u_b` to a reduced RHS.
    ///
    /// Call after assembling a source RHS, before solving with nonzero
    /// Dirichlet values. Homogeneous / no-Dirichlet cases are no-ops.
    fn apply_dirichlet_rhs_correction<K: BilinearForm>(
        &self,
        time: f64,
        kernel: &K,
        rhs: &mut [f64],
    );

    /// Assemble block-diagonal lumped GLL mass for all problem fields.
    fn assemble_lumped_mass(&self) -> SparseColMat<usize, f64>;
}
