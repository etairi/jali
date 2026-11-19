//! Kernels of `math::rns` on AVX-512 registers of eight 64-bit lanes: the lazy 128-bit sums of
//! products of residues, and the residues of canonical coefficients below $`2^{128}`$ or
//! $`2^{63}`$, centred. 128-bit values are pairs of words in memory (low word first), which a
//! pair of registers splits into low and high words with two permutations. Each lane computes
//! the scalar formula of the code it replaces, and the last values of a slice whose length is
//! not a multiple of eight take the scalar formula itself.
use super::lanes::{csub, load, mulhi, splat, store};
use crate::math::ntt::{add_mod, shoup_mul, sub_mod};
use core::arch::x86_64::*;

/// The low words of eight 128-bit values in two registers: lanes 0, 2, 4, ... of the pair.
const EVEN: [i64; 8] = [0, 2, 4, 6, 8, 10, 12, 14];
/// The high words: lanes 1, 3, 5, ...
const ODD: [i64; 8] = [1, 3, 5, 7, 9, 11, 13, 15];
/// Low and high words interleaved again, values 0 to 3 and 4 to 7 (index 8 and up selects the
/// high words).
const JOIN_FIRST: [i64; 8] = [0, 8, 1, 9, 2, 10, 3, 11];
const JOIN_SECOND: [i64; 8] = [4, 12, 5, 13, 6, 14, 7, 15];

/// The low and high words of eight 128-bit values.
#[inline]
#[target_feature(enable = "avx512f")]
fn split(values: &[u128; 8]) -> (__m512i, __m512i) {
    let words = values.as_ptr().cast::<u64>();
    // SAFETY: the array holds sixteen words, eight read from `words` and eight from
    // `words + 8`, and the loads need no alignment.
    let (a, b) = unsafe {
        (
            _mm512_loadu_si512(words.cast()),
            _mm512_loadu_si512(words.add(8).cast()),
        )
    };
    let (even, odd) = (super::lanes::lanes(EVEN), super::lanes::lanes(ODD));
    (
        _mm512_permutex2var_epi64(a, even, b),
        _mm512_permutex2var_epi64(a, odd, b),
    )
}

/// Eight 128-bit values from their low and high words.
#[inline]
#[target_feature(enable = "avx512f")]
fn join(values: &mut [u128; 8], low: __m512i, high: __m512i) {
    let a = _mm512_permutex2var_epi64(low, super::lanes::lanes(JOIN_FIRST), high);
    let b = _mm512_permutex2var_epi64(low, super::lanes::lanes(JOIN_SECOND), high);
    let words = values.as_mut_ptr().cast::<u64>();
    // SAFETY: the array holds sixteen words, eight written at `words` and eight at
    // `words + 8`, and the stores need no alignment.
    unsafe {
        _mm512_storeu_si512(words.cast(), a);
        _mm512_storeu_si512(words.add(8).cast(), b);
    }
}

/// `sums[i] += x[i] * y[i]` as 128-bit integers, for slices of equal length whose sums do not
/// overflow (the caller's bound).
#[target_feature(enable = "avx512f,avx512dq")]
pub(super) fn lazy_mac(sums: &mut [u128], x: &[u64], y: &[u64]) {
    debug_assert!(sums.len() == x.len() && x.len() == y.len());
    let (head, tail) = sums.as_chunks_mut::<8>();
    let (xh, xt) = x.as_chunks::<8>();
    let (yh, yt) = y.as_chunks::<8>();
    let one = splat(1);
    for ((s, a), b) in head.iter_mut().zip(xh).zip(yh) {
        let (low, high) = split(s);
        let (a, b) = (load(a), load(b));
        let product_low = _mm512_mullo_epi64(a, b);
        let low = _mm512_add_epi64(low, product_low);
        let carry = _mm512_cmplt_epu64_mask(low, product_low);
        let high = _mm512_add_epi64(high, mulhi(a, b));
        join(s, low, _mm512_mask_add_epi64(high, carry, high, one));
    }
    for ((s, a), b) in tail.iter_mut().zip(xt).zip(yt) {
        *s += u128::from(*a) * u128::from(*b);
    }
}

/// The constants of the residues modulo one prime $`p<2^{62}`$: $`2^{64}\bmod p`$ with its
/// Shoup constant, the Shoup constant of one, and $`q\bmod p`$.
pub(crate) struct ResidueConstants {
    pub(crate) p: u64,
    pub(crate) c: u64,
    pub(crate) c_shoup: u64,
    pub(crate) one_shoup: u64,
    pub(crate) q_mod_p: u64,
}

/// `out[i]` the residue modulo $`p`$ of the centred representative of `values[i]`, canonical
/// below $`q<2^{128}`$ with $`\lfloor q/2\rfloor=`$ `half`: `WordReducer::reduce` of the value
/// (two Shoup products, $`x_1(2^{64}\bmod p)+x_0`$), less $`q\bmod p`$ above `half`.
#[target_feature(enable = "avx512f,avx512dq")]
pub(super) fn centred_wide(values: &[u128], half: u128, k: &ResidueConstants, out: &mut [u64]) {
    debug_assert_eq!(values.len(), out.len());
    let (p, c, cs, os, qp) = (
        splat(k.p),
        splat(k.c),
        splat(k.c_shoup),
        splat(k.one_shoup),
        splat(k.q_mod_p),
    );
    let (half_low, half_high) = (splat(half as u64), splat((half >> 64) as u64));
    let (head, tail) = values.as_chunks::<8>();
    let (oh, ot) = out.as_chunks_mut::<8>();
    for (v, o) in head.iter().zip(oh) {
        let (low, high) = split(v);
        // shoup_mul(high, c) and shoup_mul(low, 1), each reduced, then added.
        let kh = mulhi(high, cs);
        let rh = csub(
            _mm512_sub_epi64(_mm512_mullo_epi64(high, c), _mm512_mullo_epi64(kh, p)),
            p,
        );
        let kl = mulhi(low, os);
        let rl = csub(_mm512_sub_epi64(low, _mm512_mullo_epi64(kl, p)), p);
        let r = csub(_mm512_add_epi64(rh, rl), p);
        // v > half, as 128-bit integers.
        let above = _mm512_cmpgt_epu64_mask(high, half_high)
            | (_mm512_cmpeq_epu64_mask(high, half_high) & _mm512_cmpgt_epu64_mask(low, half_low));
        let d = _mm512_sub_epi64(r, qp);
        let less = _mm512_min_epu64(d, _mm512_add_epi64(d, p));
        store(o, _mm512_mask_blend_epi64(above, r, less));
    }
    for (v, o) in tail.iter().zip(ot) {
        let high = shoup_mul((*v >> 64) as u64, k.c, k.c_shoup, k.p);
        let low = shoup_mul(*v as u64, 1, k.one_shoup, k.p);
        let r = add_mod(high, low, k.p);
        *o = if *v > half {
            sub_mod(r, k.q_mod_p, k.p)
        } else {
            r
        };
    }
}

/// `out[i]` the residue modulo $`p`$ of the centred representative of `values[i]`, canonical
/// below $`q<2^{63}`$ with $`\lfloor q/2\rfloor=`$ `half`: the value less $`p`$ while at least
/// $`p`$ (at most three times, as $`p>2^{61}`$), less $`q\bmod p`$ above `half`.
#[target_feature(enable = "avx512f")]
pub(super) fn centred_word(values: &[u64], half: u64, p: u64, q_mod_p: u64, out: &mut [u64]) {
    debug_assert_eq!(values.len(), out.len());
    let (pv, two_p, qp, hv) = (splat(p), splat(2 * p), splat(q_mod_p), splat(half));
    let (head, tail) = values.as_chunks::<8>();
    let (oh, ot) = out.as_chunks_mut::<8>();
    for (v, o) in head.iter().zip(oh) {
        let v = load(v);
        // Below 2^63 < 4p: x mod 2p, then mod p, by unsigned minima.
        let r = csub(csub(v, two_p), pv);
        let above = _mm512_cmpgt_epu64_mask(v, hv);
        let d = _mm512_sub_epi64(r, qp);
        let less = _mm512_min_epu64(d, _mm512_add_epi64(d, pv));
        store(o, _mm512_mask_blend_epi64(above, r, less));
    }
    for (v, o) in tail.iter().zip(ot) {
        let mut r = *v;
        while r >= p {
            r -= p;
        }
        *o = if *v > half { sub_mod(r, q_mod_p, p) } else { r };
    }
}
