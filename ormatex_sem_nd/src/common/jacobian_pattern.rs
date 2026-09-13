//! Cached CSC symbolic pattern for residual Jacobian assembly.
//!
//! Volume assembly builds `~ncells * (noutputs*ndofs) * (ninputs*ndofs)`
//! triplets in a deterministic order, then `try_new_from_triplets` sorts them
//! every `fjac`. The sort dominates EPI steps. The sparsity pattern depends
//! only on mesh + field selection, so build it once and only update values.
//!
//! ponytail: unfiltered triplets (keep explicit zeros) keep the pattern stable
//! across states; explicit zeros cost a little SpMV but avoid rebuilds.
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use faer::sparse::{SparseColMat, SymbolicSparseColMat, Triplet};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub(crate) struct JacobianPatternKey {
    inputs: Vec<usize>,
    outputs: Vec<usize>,
}

struct CachedPattern {
    symbolic: SymbolicSparseColMat<usize>,
    /// `triplet ordinal -> final CSC value position` (sums duplicates).
    triplet_to_nnz: Vec<usize>,
}

/// Per-problem cache of assembled Jacobian patterns.
pub(crate) struct JacobianPatternCache {
    inner: RwLock<HashMap<JacobianPatternKey, Arc<CachedPattern>>>,
}

impl JacobianPatternCache {
    pub(crate) fn new() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
        }
    }

    /// Assemble from deterministic-order triplets, reusing the cached pattern.
    pub(crate) fn assemble(
        &self,
        system_size: usize,
        inputs: &[usize],
        outputs: &[usize],
        triplets: Vec<Triplet<usize, usize, f64>>,
    ) -> SparseColMat<usize, f64> {
        let key = JacobianPatternKey {
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
        };
        if let Some(cached) = self.inner.read().unwrap().get(&key).cloned() {
            if cached.triplet_to_nnz.len() == triplets.len() {
                let nnz = cached.symbolic.row_idx().len();
                let mut vals = vec![0.0; nnz];
                for (i, t) in triplets.iter().enumerate() {
                    vals[cached.triplet_to_nnz[i]] += t.val;
                }
                return SparseColMat::new(cached.symbolic.clone(), vals);
            }
            // Length mismatch: fall through and rebuild (stale pattern).
        }
        let matrix =
            SparseColMat::try_new_from_triplets(system_size, system_size, &triplets).unwrap();
        let symbolic = matrix.symbolic().to_owned().unwrap();
        let col_ptr = symbolic.col_ptr();
        let row_idx = symbolic.row_idx();
        let mut triplet_to_nnz = Vec::with_capacity(triplets.len());
        for t in &triplets {
            let start = col_ptr[t.col];
            let end = col_ptr[t.col + 1];
            let offset = row_idx[start..end]
                .binary_search(&t.row)
                .expect("triplet missing from symbolic pattern");
            triplet_to_nnz.push(start + offset);
        }
        self.inner.write().unwrap().insert(
            key,
            Arc::new(CachedPattern {
                symbolic: symbolic.clone(),
                triplet_to_nnz,
            }),
        );
        matrix
    }
}
