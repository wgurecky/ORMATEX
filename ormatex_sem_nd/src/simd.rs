//! Small portable SIMD helpers used by tensor-product kernels.

use pulp::{Arch, Simd, WithSimd};

struct ScaleNegateInPlace<'a> {
    values: &'a mut [f64],
    factors: &'a [f64],
}

struct Dot<'a> {
    left: &'a [f64],
    right: &'a [f64],
}

struct Axpy<'a> {
    target: &'a mut [f64],
    source: &'a [f64],
    factor: f64,
}

impl WithSimd for Axpy<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let (target, target_tail) = S::as_mut_simd_f64s(self.target);
        let (source, source_tail) = S::as_simd_f64s(self.source);
        let factor = simd.splat_f64s(self.factor);
        for (target, source) in target.iter_mut().zip(source) {
            *target = simd.mul_add_f64s(factor, *source, *target);
        }
        for (target, source) in target_tail.iter_mut().zip(source_tail) {
            *target += self.factor * *source;
        }
    }
}

impl WithSimd for Dot<'_> {
    type Output = f64;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> f64 {
        let (left, left_tail) = S::as_simd_f64s(self.left);
        let (right, right_tail) = S::as_simd_f64s(self.right);
        let mut sum = simd.splat_f64s(0.0);
        for (left, right) in left.iter().zip(right) {
            sum = simd.mul_add_f64s(*left, *right, sum);
        }
        simd.reduce_sum_f64s(sum)
            + left_tail
                .iter()
                .zip(right_tail)
                .map(|(&left, &right)| left * right)
                .sum::<f64>()
    }
}

impl WithSimd for ScaleNegateInPlace<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let (values, values_tail) = S::as_mut_simd_f64s(self.values);
        let (factors, factors_tail) = S::as_simd_f64s(self.factors);
        for (value, factor) in values.iter_mut().zip(factors) {
            *value = simd.neg_f64s(simd.mul_f64s(*value, *factor));
        }
        for (value, factor) in values_tail.iter_mut().zip(factors_tail) {
            *value = -*value * *factor;
        }
    }
}

/// Scale and negate a vector in place using runtime CPU SIMD dispatch.
#[inline]
pub(crate) fn scale_negate_in_place(values: &mut [f64], factors: &[f64]) {
    assert_eq!(values.len(), factors.len());
    Arch::new().dispatch(ScaleNegateInPlace { values, factors });
}

/// Compute a dot product with runtime CPU SIMD dispatch.
#[inline]
pub(crate) fn dot(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());
    if left.len() < 4 {
        return left
            .iter()
            .zip(right)
            .map(|(&left, &right)| left * right)
            .sum();
    }
    Arch::new().dispatch(Dot { left, right })
}

/// Add `factor * source` to a contiguous target slice with runtime SIMD dispatch.
#[inline]
pub(crate) fn axpy(target: &mut [f64], factor: f64, source: &[f64]) {
    assert_eq!(target.len(), source.len());
    Arch::new().dispatch(Axpy {
        target,
        source,
        factor,
    });
}

#[cfg(test)]
mod tests {
    use super::scale_negate_in_place;

    #[test]
    fn scale_negate_handles_simd_and_tail_values() {
        let mut values: Vec<_> = (1..18).map(|value| value as f64).collect();
        let factors: Vec<_> = (1..18).map(|value| 0.5 * value as f64).collect();
        scale_negate_in_place(&mut values, &factors);
        for (index, (value, factor)) in values.iter().zip(factors).enumerate() {
            assert_eq!(*value, -((index + 1) as f64) * factor);
        }
    }
}
