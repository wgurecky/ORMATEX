//! Parallel sparse matrix multiplication for Faer linear operators.

use std::fmt;

use faer::dyn_stack::{MemStack, StackReq};
use faer::matrix_free::LinOp;
use faer::sparse::{SparseRowMat, SparseRowMatRef};
use faer::{MatMut, MatRef, Par};
use rayon::prelude::*;

/// An owned CSR matrix with parallel sparse-dense multiplication.
///
/// The public constructor accepts CSC because SEM assembly produces CSC. The
/// matrix is converted to row-major storage once so each worker can process
/// only the nonzeros in its assigned rows.
pub struct ParCsrJacobian {
    csc: faer::sparse::SparseColMat<usize, f64>,
    lhs: SparseRowMat<usize, f64>,
    n_threads: usize,
}

impl ParCsrJacobian {
    /// Construct an operator using `n_threads` row chunks for Rayon calls.
    pub fn new(lhs: faer::sparse::SparseColMat<usize, f64>, n_threads: usize) -> Self {
        assert!(
            n_threads > 0,
            "parallel sparse multiplication needs a thread count"
        );
        let n_threads = n_threads.min(lhs.nrows().max(1));
        let row_major = lhs
            .as_ref()
            .to_row_major()
            .expect("failed to convert CSC matrix to CSR storage");
        Self {
            csc: lhs,
            lhs: row_major,
            n_threads,
        }
    }

    #[inline]
    fn apply_inner(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        assert_eq!(out.nrows(), self.lhs.nrows(), "output row count mismatch");
        assert_eq!(out.ncols(), rhs.ncols(), "output/RHS column count mismatch");
        assert_eq!(rhs.nrows(), self.lhs.ncols(), "RHS row count mismatch");

        let n_threads = par.degree().max(1).min(self.lhs.nrows().max(1));
        if matches!(par, Par::Seq) || n_threads == 1 {
            self.csc.as_ref().apply(out, rhs, Par::Seq, stack);
        } else {
            sparse_dense_csr_matmul_par(out, self.lhs.as_ref(), rhs, n_threads);
        }
    }
}

impl fmt::Debug for ParCsrJacobian {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParCsrJacobian")
            .field("nrows", &self.lhs.nrows())
            .field("ncols", &self.lhs.ncols())
            .field("n_threads", &self.n_threads)
            .finish()
    }
}

impl LinOp<f64> for ParCsrJacobian {
    #[inline]
    fn apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }

    #[inline]
    fn nrows(&self) -> usize {
        self.lhs.nrows()
    }

    #[inline]
    fn ncols(&self) -> usize {
        self.lhs.ncols()
    }

    #[track_caller]
    fn apply(&self, out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, par: Par, stack: &mut MemStack) {
        self.apply_inner(out, rhs, par, stack);
    }

    #[track_caller]
    fn conj_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.apply_inner(out, rhs, par, stack);
    }
}

fn sparse_dense_csr_matmul_par(
    dst: MatMut<'_, f64>,
    lhs: SparseRowMatRef<'_, usize, f64>,
    rhs: MatRef<'_, f64>,
    n_threads: usize,
) {
    let chunk_size = lhs.nrows().max(1).div_ceil(n_threads);
    dst.par_row_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk, mut dst_chunk)| {
            let row_start = chunk * chunk_size;
            for local_row in 0..dst_chunk.nrows() {
                let row = row_start + local_row;
                let columns = lhs.col_idx_of_row_raw(row);
                let values = lhs.val_of_row(row);
                for output_column in 0..rhs.ncols() {
                    let mut value = 0.0;
                    for (&column, &coefficient) in columns.iter().zip(values) {
                        value += coefficient * rhs[(column, output_column)];
                    }
                    dst_chunk[(local_row, output_column)] = value;
                }
            }
        });
}
