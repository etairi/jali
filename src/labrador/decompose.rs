//! Centred base-$`b`$ decompositions (BS23 §5.2 and §5.3): $`x=\sum_{k<t}x_kb^k`$ for the
//! centred representative $`x`$, with $`x_k\in[-\lfloor b/2\rfloor,\lceil b/2\rceil-1]`$ for
//! $`k<t-1`$ and the top digit $`x_{t-1}`$ the rest. The verifier bounds the lower digits by
//! $`b/2`$ in the infinity norm (BS23 Fig. 3, lines 10 to 13) and all digits together by the
//! Euclidean norm of the next level.
use super::ring::Rq;

/// The centred digits of `x` in base `b`, `out.len()` of them.
pub(crate) fn digits(mut x: i64, b: u64, out: &mut [i64]) {
    debug_assert!(b >= 2 && !out.is_empty());
    let b = b as i64;
    let half = (b + 1) / 2;
    let last = out.len() - 1;
    for digit in &mut out[..last] {
        let mut r = x.rem_euclid(b);
        if r >= half {
            r -= b;
        }
        *digit = r;
        x = (x - r) / b;
    }
    out[last] = x;
}

/// The `t` digit vectors of the canonical residues `a`: `out[k]` holds digit $`k`$ of every
/// coefficient, canonical modulo $`q`$.
pub(crate) fn decompose(ring: &Rq, a: &[u64], b: u64, t: usize) -> Vec<Vec<u64>> {
    let mut out = vec![vec![0u64; a.len()]; t];
    let mut buf = vec![0i64; t];
    for (j, x) in a.iter().enumerate() {
        digits(ring.center(*x), b, &mut buf);
        for (k, v) in buf.iter().enumerate() {
            out[k][j] = ring.lift(*v);
        }
    }
    out
}

/// $`\sum_kb^k\cdot`$`parts[k]` modulo $`q`$.
#[cfg(test)]
pub(crate) fn recompose(ring: &Rq, parts: &[Vec<u64>], b: u64) -> Vec<u64> {
    let mut out = vec![0u64; parts.first().map_or(0, Vec::len)];
    let mut scale = 1u64;
    for part in parts {
        ring.add_scaled_assign(&mut out, part, scale);
        scale = ring.mul(scale, b % ring.modulus());
    }
    out
}

/// $`b^k\bmod q`$ for $`k<t`$.
pub(crate) fn powers(ring: &Rq, b: u64, t: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(t);
    let mut x = 1u64;
    for _ in 0..t {
        out.push(x);
        x = ring.mul(x, b % ring.modulus());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::labrador::test_rng::TestRng;

    #[test]
    fn digits_recompose_and_respect_the_lower_bound() {
        let mut rng = TestRng::new(5);
        for b in [2u64, 3, 7, 16, 255, 1 << 12] {
            for t in 1..6 {
                for _ in 0..200 {
                    let x = rng.small(1 << 40);
                    let mut out = vec![0i64; t];
                    digits(x, b, &mut out);
                    let mut back = 0i128;
                    for d in out.iter().rev() {
                        back = back * i128::from(b) + i128::from(*d);
                    }
                    assert_eq!(back, i128::from(x));
                    for d in &out[..t - 1] {
                        assert!(d.unsigned_abs() <= b / 2, "{b} {d}");
                        assert!(*d >= -((b / 2) as i64) && *d < b.div_ceil(2) as i64);
                    }
                }
            }
        }
    }

    #[test]
    fn decomposition_of_residues_recomposes_modulo_q() {
        let ring = Rq::new(4294967197, 64).unwrap();
        let mut rng = TestRng::new(6);
        let a: Vec<u64> = (0..640).map(|_| rng.below(ring.modulus())).collect();
        for (b, t) in [(2u64, 33usize), (1 << 11, 3), (65536, 2), (1 << 16, 1)] {
            let parts = decompose(&ring, &a, b, t);
            assert_eq!(recompose(&ring, &parts, b), a);
            assert_eq!(
                powers(&ring, b, t)[t - 1],
                ring.lift_i128(i128::from(b).pow(t as u32 - 1))
            );
        }
    }
}
