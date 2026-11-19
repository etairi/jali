//! Elementwise arithmetic modulo a prime $`p<2^{62}`$ on slices of residues in $`[0,p)`$, eight
//! lanes at a time with the tail under a mask: the formulas of `math::ntt` (`add_mod`,
//! `sub_mod`, `shoup_mul`, `NttPlan::mont_mul`) in each lane, so every output is the scalar
//! one. A sum $`s=a+b<2p`$ is reduced by `csub`, a difference $`a-b`$ by the unsigned minimum
//! of it and $`a-b+p`$ (one of the two wraps above $`2^{63}`$ exactly when it is not the
//! residue).
use super::lanes::{csub, load, load_part, mulhi, splat, store, store_part};
use core::arch::x86_64::*;

/// `sub_mod` in each lane: $`a-b`$ or $`a-b+p`$, whichever is below $`p`$.
#[inline]
#[target_feature(enable = "avx512f")]
fn sub(a: __m512i, b: __m512i, p: __m512i) -> __m512i {
    let d = _mm512_sub_epi64(a, b);
    _mm512_min_epu64(d, _mm512_add_epi64(d, p))
}
/// `shoup_mul` in each lane: $`xw-\lfloor xw'/2^{64}\rfloor p`$ reduced into $`[0,p)`$.
#[inline]
#[target_feature(enable = "avx512f,avx512dq")]
fn shoup(x: __m512i, w: __m512i, ws: __m512i, p: __m512i) -> __m512i {
    let k = mulhi(x, ws);
    csub(
        _mm512_sub_epi64(_mm512_mullo_epi64(x, w), _mm512_mullo_epi64(k, p)),
        p,
    )
}
/// `NttPlan::mont_mul` in each lane: with $`t=xy`$ and $`m=t\cdot(-p^{-1})\bmod2^{64}`$, the low
/// word of $`t+mp`$ is zero, so $`(t+mp)/2^{64}`$ is the sum of the high words plus one carry,
/// which occurs exactly when the low word of $`t`$ is nonzero. Below $`2p`$, then reduced.
#[inline]
#[target_feature(enable = "avx512f,avx512dq")]
fn mont(x: __m512i, y: __m512i, p: __m512i, neg_inv: __m512i) -> __m512i {
    let low = _mm512_mullo_epi64(x, y);
    let m = _mm512_mullo_epi64(low, neg_inv);
    let sum = _mm512_add_epi64(mulhi(x, y), mulhi(m, p));
    let carry = _mm512_test_epi64_mask(low, low);
    csub(_mm512_mask_add_epi64(sum, carry, sum, splat(1)), p)
}

/// The kernel of each operation: `$body` maps the lanes `x` of the first slice (and `y` of the
/// second) to the new lanes of the first, over the full registers and then the tail.
macro_rules! zip_kernel {
    ($(#[$doc:meta])* $name:ident($a:ident, $b:ident $(, $arg:ident: $ty:ty)*) |$x:ident, $y:ident| $body:expr) => {
        $(#[$doc])*
        #[target_feature(enable = "avx512f,avx512dq")]
        pub(super) fn $name($a: &mut [u64], $b: &[u64] $(, $arg: $ty)*) {
            debug_assert_eq!($a.len(), $b.len());
            let (head, tail) = $a.as_chunks_mut::<8>();
            let (bhead, btail) = $b.as_chunks::<8>();
            for (xs, ys) in head.iter_mut().zip(bhead) {
                let ($x, $y) = (load(xs), load(ys));
                store(xs, $body);
            }
            if !tail.is_empty() {
                let ($x, $y) = (load_part(tail), load_part(btail));
                store_part(tail, $body);
            }
        }
    };
}

zip_kernel!(
    /// `a[i] = add_mod(a[i], b[i], p)`.
    add_mod(a, b, p: u64) |x, y| csub(_mm512_add_epi64(x, y), splat(p))
);
zip_kernel!(
    /// `a[i] = sub_mod(a[i], b[i], p)`.
    sub_mod(a, b, p: u64) |x, y| sub(x, y, splat(p))
);
zip_kernel!(
    /// `a[i] = add_mod(a[i], shoup_mul(b[i], w, w_shoup, p), p)`.
    add_scaled(a, b, w: u64, w_shoup: u64, p: u64)
        |x, y| csub(_mm512_add_epi64(x, shoup(y, splat(w), splat(w_shoup), splat(p))), splat(p))
);
zip_kernel!(
    /// `a[i] = sub_mod(a[i], shoup_mul(b[i], w, w_shoup, p), p)`.
    sub_scaled(a, b, w: u64, w_shoup: u64, p: u64)
        |x, y| sub(x, shoup(y, splat(w), splat(w_shoup), splat(p)), splat(p))
);

/// `a[i] = shoup_mul(a[i], w, w_shoup, p)`.
#[target_feature(enable = "avx512f,avx512dq")]
pub(super) fn scale(a: &mut [u64], w: u64, w_shoup: u64, p: u64) {
    let (wv, wsv, pv) = (splat(w), splat(w_shoup), splat(p));
    let (head, tail) = a.as_chunks_mut::<8>();
    for xs in head {
        store(xs, shoup(load(xs), wv, wsv, pv));
    }
    if !tail.is_empty() {
        store_part(tail, shoup(load_part(tail), wv, wsv, pv));
    }
}

/// `acc[i] = add_mod(acc[i], mont_mul(a[i], b[i]), p)` for slices of equal length.
#[target_feature(enable = "avx512f,avx512dq")]
pub(super) fn mont_mac(acc: &mut [u64], a: &[u64], b: &[u64], p: u64, neg_inv: u64) {
    debug_assert!(acc.len() == a.len() && a.len() == b.len());
    let (pv, nv) = (splat(p), splat(neg_inv));
    let (head, tail) = acc.as_chunks_mut::<8>();
    let (ahead, atail) = a.as_chunks::<8>();
    let (bhead, btail) = b.as_chunks::<8>();
    for ((s, x), y) in head.iter_mut().zip(ahead).zip(bhead) {
        let t = mont(load(x), load(y), pv, nv);
        store(s, csub(_mm512_add_epi64(load(s), t), pv));
    }
    if !tail.is_empty() {
        let t = mont(load_part(atail), load_part(btail), pv, nv);
        store_part(tail, csub(_mm512_add_epi64(load_part(tail), t), pv));
    }
}

/// `out[i]` the residue modulo $`p`$ of the centred representative of `a[i]` modulo
/// $`q<2p`$: `a[i]` up to $`\lfloor q/2\rfloor`$, else $`a_i-q+p`$ (in $`(0,p)`$ as
/// $`q-a_i<q/2<p`$).
#[target_feature(enable = "avx512f")]
pub(super) fn centred(a: &[u64], q: u64, p: u64, out: &mut [u64]) {
    debug_assert_eq!(a.len(), out.len());
    let (half, shift) = (splat(q / 2), splat(p.wrapping_sub(q)));
    let (head, tail) = out.as_chunks_mut::<8>();
    let (ahead, atail) = a.as_chunks::<8>();
    for (o, x) in head.iter_mut().zip(ahead) {
        store(o, centre(load(x), half, shift));
    }
    if !tail.is_empty() {
        store_part(tail, centre(load_part(atail), half, shift));
    }
}
/// The lanes above `half` plus `shift` (modulo $`2^{64}`$), the others as they are.
#[inline]
#[target_feature(enable = "avx512f")]
fn centre(x: __m512i, half: __m512i, shift: __m512i) -> __m512i {
    let negative = _mm512_cmpgt_epu64_mask(x, half);
    _mm512_mask_add_epi64(x, negative, x, shift)
}

/// The constants of `SmallCrt` for `K` primes: Garner's (the primes, $`M_t\bmod p_i`$ for
/// $`t<i`$ and $`M_i^{-1}\bmod p_i`$ with Shoup constants, the digits of $`\lfloor P/2\rfloor`$),
/// then $`M_i\bmod q`$ with Shoup constants, $`P\bmod q`$ and $`q<2^{63}`$.
pub(crate) struct CrtConstants<const K: usize> {
    pub(crate) primes: [u64; K],
    pub(crate) prefixes: [[(u64, u64); K]; K],
    pub(crate) inverses: [(u64, u64); K],
    pub(crate) half: [u64; K],
    pub(crate) prefixes_q: [(u64, u64); K],
    pub(crate) product_q: u64,
    pub(crate) q: u64,
}

/// `SmallCrt::reconstruct_fixed` of the first coefficients, eight at a time: Garner's digits
/// in each lane, the comparison of the digit vector with that of $`\lfloor P/2\rfloor`$ from
/// the most significant digit down, and $`\sum_iv_i(M_i\bmod q)`$ less $`P\bmod q`$ for a
/// centred value below zero. Returns how many coefficients it wrote, a multiple of eight.
#[target_feature(enable = "avx512f,avx512dq")]
pub(super) fn small_crt<const K: usize>(
    residues: &[&[u64]; K],
    c: &CrtConstants<K>,
    out: &mut [u64],
) -> usize {
    let (head, _) = out.as_chunks_mut::<8>();
    let q = splat(c.q);
    for (chunk, o) in head.iter_mut().enumerate() {
        let j = 8 * chunk;
        let mut digits = [splat(0); K];
        for i in 0..K {
            let p = splat(c.primes[i]);
            let mut sum = splat(0);
            for (t, d) in digits[..i].iter().enumerate() {
                let (w, ws) = c.prefixes[i][t];
                sum = csub(_mm512_add_epi64(sum, shoup(*d, splat(w), splat(ws), p)), p);
            }
            let (w, ws) = c.inverses[i];
            let r: &[u64; 8] = residues[i][j..j + 8].try_into().expect("eight residues");
            digits[i] = shoup(sub(load(r), sum, p), splat(w), splat(ws), p);
        }
        let (mut negative, mut equal): (__mmask8, __mmask8) = (0, 0xff);
        for i in (0..K).rev() {
            let h = splat(c.half[i]);
            negative |= equal & _mm512_cmpgt_epu64_mask(digits[i], h);
            equal &= _mm512_cmpeq_epu64_mask(digits[i], h);
        }
        let mut x = splat(0);
        for (d, (w, ws)) in digits.iter().zip(&c.prefixes_q) {
            x = csub(_mm512_add_epi64(x, shoup(*d, splat(*w), splat(*ws), q)), q);
        }
        let less = sub(x, splat(c.product_q), q);
        store(o, _mm512_mask_blend_epi64(negative, x, less));
    }
    8 * head.len()
}

/// The column sums of LaBRADOR's projection table for eight columns at once: `out[c][k]` the
/// sum over the 64 groups $`g`$ of `table[(256g + columns[c][g])λ + k]` for $`k<\lambda\le8`$, each
/// column's λ sums in one register, the groups in order, as `level::column_sums` adds them.
#[target_feature(enable = "avx512f")]
pub(super) fn table_sums8(
    table: &[i64],
    lambda: usize,
    columns: &[[u8; 64]; 8],
    out: &mut [[i64; 8]; 8],
) {
    let mut acc = [_mm512_setzero_si512(); 8];
    for g in 0..64 {
        let base = 256 * g;
        for (a, column) in acc.iter_mut().zip(columns) {
            let at = (base + usize::from(column[g])) * lambda;
            let entry = &table[at..at + lambda];
            // The table's sums are i64, the lanes are u64: the same bits, added modulo 2^64.
            let words: &[u64] = bytemuck_i64(entry);
            *a = _mm512_add_epi64(*a, load_part(words));
        }
    }
    for (o, a) in out.iter_mut().zip(acc) {
        let mut words = [0u64; 8];
        store(&mut words, a);
        for (x, w) in o.iter_mut().zip(words) {
            *x = w as i64;
        }
    }
}

/// A slice of `i64` read as `u64`, the same bits.
#[inline]
fn bytemuck_i64(a: &[i64]) -> &[u64] {
    // SAFETY: i64 and u64 have the same size, alignment and valid bit patterns.
    unsafe { core::slice::from_raw_parts(a.as_ptr().cast(), a.len()) }
}
