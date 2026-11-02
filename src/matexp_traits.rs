/*
 * Copyright(c) 2025 UT-Battelle, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
//! The phi-function evaluator traits.
//!
//! This module defines the interfaces used by the exponential integrators to
//! evaluate products of phi-functions with vectors,
//! $\varphi_k(t A) v$, where $\varphi_k$ is defined by
//!
//! $$ \varphi_0(z) = e^z, \qquad
//!    \varphi_k(z) = \int_0^1 e^{(1-\theta) z} \frac{\theta^{k-1}}{(k-1)!} \thinspace d\theta $$
//!
//! [`DensePhikvEvaluator`] is implemented by methods acting on a dense matrix
//! `A` (Pade, Taylor, Cauchy contour integral, ...), while
//! [`LinOpPhikvEvaluator`] is implemented by matrix-free methods acting on a
//! linear operator (Krylov, Leja, ...). [`PhikvStatus`] describes the
//! convergence status of an iterative evaluation.
//!
//! # References
//!
//! * M. Hochbruck, A. Ostermann, "Exponential integrators", Acta Numerica 19
//!   (2010) 209-286, doi:10.1017/S0962492910000048.
//! * M. Caliari, F. Cassini, F. Zivcovich, "BAMPHI: Chebyshev and rational
//!   approximations of phi-functions applied to vectors", J. Comput. Appl.
//!   Math. 423 (2023) 114973.
use crate::ode_sys::DynRefExtendedLinOp;
use faer::matrix_free::LinOp;
use faer::prelude::*;
use faer::Par;
use faer_traits::ComplexField;

/// Status report of an iterative phi-function vector product evaluation.
///
/// The fields are private and currently unused.
pub struct PhikvStatus {
    /// Whether the iteration converged
    _conv: bool,
    /// Number of internal iterations required
    _iter: usize,
    /// Error estimate at termination
    _err: f64,
}

/// Trait for implementors of a $\varphi_k(A \thinspace dt) v$ method for dense `A`.
///
/// The scalar type `T` defaults to `f64`. The time step `dt` is always a real
/// `f64`.
pub trait DensePhikvEvaluator<T: ComplexField = f64> {
    /// Evaluates the phi-function vector product $\varphi_k(dt \thinspace A) v_0$.
    ///
    /// # Arguments
    ///
    /// * `a` - dense square matrix $A$
    /// * `dt` - time step scale factor
    /// * `v0` - the vector (or matrix of column vectors) to which $\varphi_k$ is applied
    /// * `k` - the phi-function order
    /// * `par` - faer parallelism for the dense linear algebra
    ///
    /// # Returns
    ///
    /// The product $\varphi_k(dt \thinspace A) v_0$, same size as `v0`.
    fn apply_phi_k(&self, a: MatRef<T>, dt: f64, v0: MatRef<T>, k: usize, par: Par) -> Mat<T>;

    /// Evaluates a linear combination of phi-function vector products.
    ///
    /// Computes
    ///
    /// $$ \sum_j \varphi_{k_j}(dt \thinspace A) \thinspace v_j $$
    ///
    /// where $k_j$ is `ks[j]` and $v_j$ is `vb[j]`. The default implementation
    /// calls [`DensePhikvEvaluator::apply_phi_k`] once per term in serial and
    /// sums the results. Implementors may override it with a more efficient
    /// evaluation.
    ///
    /// # Arguments
    ///
    /// * `a` - dense square matrix $A$
    /// * `dt` - time step scale factor
    /// * `vb` - slice of vectors $v_j$ to which each phi-function is applied
    /// * `ks` - slice of phi-function orders $k_j$, one per entry of `vb`
    /// * `par` - faer parallelism for the dense linear algebra
    ///
    /// # Returns
    ///
    /// The summed product, with `a.nrows()` rows and `vb[0].ncols()` columns.
    ///
    /// # Panics
    ///
    /// Panics if `vb` is empty or if `vb` and `ks` differ in length.
    fn apply_phi_k_v(
        &self,
        a: MatRef<T>,
        dt: f64,
        vb: &[MatRef<'_, T>],
        ks: &[usize],
        par: Par,
    ) -> Mat<T> {
        assert!(!vb.is_empty());
        assert!(vb.len() == ks.len());
        let mut out = Mat::zeros(a.nrows(), vb[0].ncols());
        // default loops over each phi_k function in serial
        for (v, k) in vb.iter().zip(ks) {
            // if v.norm_l2() >= 0.0 {
            out += self.apply_phi_k(a, dt, v.as_ref(), *k, par);
        }
        out
    }

    /// Prepare for a subsequent `apply_*` call.
    ///
    /// Implementors may use this hook to precompute and cache data that
    /// depends on `a`, `dt`, `vb` and `ks`. The default implementation does
    /// nothing.
    ///
    /// # Arguments
    ///
    /// * `_a` - dense square matrix $A$
    /// * `_dt` - time step scale factor
    /// * `_vb` - slice of rhs vectors to which the phi-function will be applied
    /// * `_ks` - slice of phi-function orders, one per entry of `_vb`
    /// * `_par` - faer parallelism for the dense linear algebra
    fn apply_prepare(
        &mut self,
        _a: MatRef<T>,
        _dt: f64,
        _vb: &[MatRef<'_, T>],
        _ks: &[usize],
        _par: Par,
    ) {
        // default is null-op
    }
}

/// Trait for implementors of a $\varphi_k(A \thinspace dt) v$ method for sparse or
/// matrix-free linear operator `A`.
///
/// The scalar type `T` defaults to `f64`.
/// Krylov and Leja retain faer's global parallelism for their existing operator
/// and basis operations. Krylov uses its configured `dense_par` for the
/// projected matrix; these implementations do not currently use the per-call
/// `par` argument.
pub trait LinOpPhikvEvaluator<T: ComplexField = f64> {
    /// Evaluates a linear combination of phi-function vector products.
    ///
    /// Computes
    ///
    /// $$ \sum_j dt^{k_j} \varphi_{k_j}(dt \thinspace t A) \thinspace v_j $$
    ///
    /// where $k_j$ is `ks[j]`, $v_j$ is `vb[j]`, and $t$ is the scale
    /// used to construct `a_lo`. Orders can be repeated or omitted. When
    /// `dt` is 1, the result is the sum of $\varphi_{k_j}(t A) v_j$.
    ///
    /// # Arguments
    ///
    /// * `a_lo` - extended operator built using the same `vb` and `ks`; the
    ///   vector values must not change between construction and evaluation
    /// * `dt` - additional scale applied to the entire extended operator
    /// * `vb` - slice of single-column vectors $v_j$ used to construct `a_lo`
    /// * `ks` - slice of phi-function orders $k_j$, one per entry of `vb`
    /// * `par` - parallelism for internal dense linear algebra; implementations
    ///   may use their own configured parallelism for projected matrices
    ///
    /// # Returns
    ///
    /// The summed product, a matrix with as many rows as `vb[0]`.
    ///
    /// # Panics
    ///
    /// Panics if the vector shapes or orders do not match the operator.
    fn apply_phi_k_v(
        &mut self,
        ext_a_lo: &DynRefExtendedLinOp,
        dt: f64,
        vb: &[MatRef<'_, T>],
        ks: &[usize],
        par: Par,
    ) -> Mat<T>;

    /// Evaluates the phi-function vector product $\varphi_k(dt \thinspace A) v$.
    ///
    /// Takes `&mut self` because implementations may update internal state,
    /// for example the adaptive Krylov dimension and work storage.
    ///
    /// # Arguments
    ///
    /// * `a_lo` - linear operator $A$
    /// * `dt` - time step scale factor
    /// * `v` - the vector to which $\varphi_k$ is applied
    /// * `k` - the phi-function order
    /// * `par` - parallelism for internal dense linear algebra
    ///
    /// # Returns
    ///
    /// The product $\varphi_k(dt \thinspace A) v$, same size as `v`.
    fn apply_phi_k(&mut self, a_lo: &dyn LinOp<T>, dt: f64, v: MatRef<T>, k: usize, par: Par) -> Mat<T>;

    /// Prepare for a subsequent `apply_*` call. Call this again whenever the
    /// operator or vectors change; cached data is not checked for changes.
    ///
    /// # Arguments
    ///
    /// * `_a_lo` - extended linear operator $\tilde A$
    /// * `_dt` - scale applied to the inner operator for spectrum preparation;
    ///   normally the scale used to construct `_a_lo`
    /// * `_vb` - slice of vectors used to construct `_a_lo`
    /// * `_ks` - slice of phi-function orders, one per entry of `_vb`
    /// * `_par` - parallelism for internal dense linear algebra
    fn apply_prepare(
        &mut self,
        _a_lo: &DynRefExtendedLinOp,
        _dt: f64,
        _vb: &[MatRef<'_, T>],
        _ks: &[usize],
        _par: Par,
    ) {
        // default is null-op
    }
}
