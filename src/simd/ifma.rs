//! LaBRADOR's transform-domain arithmetic with AVX-512 IFMA: the merged negacyclic NTTs of
//! `NttPlan::forward_mont` and `NttPlan::inverse_mont` and the multiply-accumulate of transforms,
//! over primes below $`2^{51}`$ with the 52-bit products of `vpmadd52luq` and `vpmadd52huq`, where
//! the 64-bit kernels assemble each high half from four 32-bit products.
//!
//! Transforms are in Montgomery form with $`R=2^{52}`$ instead of $`2^{64}`$, so a ring on these
//! primes takes every transform, product and inverse here (`labrador::ring::Rq`), and its exact
//! CRT returns the same residues as on the primes of `NTT_PRIMES`. The stages, twiddles and lane
//! regroupings are those of `ntt_avx512`. The Shoup products are in base $`2^{52}`$:
//! $`yw-\lfloor yw'/2^{52}\rfloor p\in[0,2p)`$ with $`w'=\lfloor w2^{52}/p\rfloor`$, exact for
//! $`y<2^{52}`$, and below $`2^{52}`$ since $`p<2^{51}`$, so the remainder is taken modulo
//! $`2^{52}`$ from the low halves. Every value that is multiplied is first reduced below $`2p`$,
//! one conditional subtraction more per butterfly than the 64-bit kernels, so all products stay
//! within the 52 bits that IFMA multiplies.
//!
//! The scalar code of the same arithmetic ([`forward_scalar`], [`inverse_scalar`],
//! [`mac_scalar`]) converts between the two Montgomery forms around the 64-bit transforms, with
//! the same outputs: the reference of the kernels and their fallback.
use super::CrtConstants;
use super::lanes::{csub, lanes, load, splat, store};
use super::ntt_avx512::{BY1, BY2, BY4, small_twiddles};
use crate::math::ntt::{NttPlan, mul_mod, pow_mod};
use core::arch::x86_64::*;

/// Primes below $`2^{51}`$, congruent to 1 modulo 2048, with primitive 2048-th roots, in the
/// convention of `NTT_PRIMES` (found by Miller-Rabin with the first twelve prime bases, which is
/// deterministic below $`2^{64}`$, and the roots checked to have order exactly 2048). Two of them
/// make $`P>2^{101}`$.
pub(crate) const IFMA_PRIMES: [(u64, u64); 4] = [
    (2251799813640193, 1214779003152539),
    (2251799813632001, 300463272824019),
    (2251799813613569, 1460959258732152),
    (2251799813560321, 480106155032164),
];

const MASK52: u64 = (1 << 52) - 1;

/// The tables of one prime: the twiddles of an `NttPlan` with their Shoup constants in base
/// $`2^{52}`$, the constants of the first and last stages for $`R=2^{52}`$, the Montgomery
/// constant $`-p^{-1}\bmod2^{52}`$, and the factors between the two Montgomery forms.
#[derive(Debug, Clone)]
pub(crate) struct IfmaPlan {
    pub(crate) plan: NttPlan,
    p: u64,
    d: usize,
    zetas: Vec<u64>,
    zetas_shoup: Vec<u64>,
    inv_zetas: Vec<u64>,
    inv_zetas_shoup: Vec<u64>,
    r: (u64, u64),
    first: (u64, u64),
    scale: (u64, u64),
    last: (u64, u64),
    n_prime: u64,
    /// $`2^{52}2^{-64}\bmod p`$ and its inverse: the factors from and to the 64-bit form.
    from64: u64,
    to64: u64,
}

impl IfmaPlan {
    /// The tables of `plan`, whose prime is below $`2^{51}`$.
    pub(crate) fn new(plan: NttPlan) -> Self {
        let t = plan.simd_tables();
        let (p, d) = (t.p, t.d);
        assert!(p < 1 << 51, "an IFMA prime is below 2^51");
        let (zetas, inv_zetas) = (t.zetas.to_vec(), t.inv_zetas.to_vec());
        let shoup52 = |w: u64| ((u128::from(w) << 52) / u128::from(p)) as u64;
        let pair = |w: u64| (w, shoup52(w));
        let inv = |a: u64| pow_mod(a, p - 2, p);
        let r52 = ((1u128 << 52) % u128::from(p)) as u64;
        let r64 = ((1u128 << 64) % u128::from(p)) as u64;
        let scale = mul_mod(inv(d as u64 % p), inv(r52), p);
        Self {
            p,
            d,
            zetas_shoup: zetas.iter().map(|w| shoup52(*w)).collect(),
            inv_zetas_shoup: inv_zetas.iter().map(|w| shoup52(*w)).collect(),
            r: pair(r52),
            first: pair(mul_mod(zetas[1], r52, p)),
            scale: pair(scale),
            last: pair(mul_mod(inv_zetas[1], scale, p)),
            n_prime: plan.neg_inv() & MASK52,
            from64: mul_mod(r52, inv(r64), p),
            to64: mul_mod(r64, inv(r52), p),
            zetas,
            inv_zetas,
            plan,
        }
    }
}

/// `forward` of [`IfmaPlan`] in scalar code: the 64-bit transform, then the change of
/// Montgomery form.
pub(crate) fn forward_scalar(t: &IfmaPlan, a: &mut [u64]) {
    t.plan.forward_mont(a);
    for x in a.iter_mut() {
        *x = mul_mod(*x, t.from64, t.p);
    }
}

/// `inverse` of [`IfmaPlan`] in scalar code: the change of Montgomery form, then the 64-bit
/// inverse.
pub(crate) fn inverse_scalar(t: &IfmaPlan, a: &mut [u64]) {
    for x in a.iter_mut() {
        *x = mul_mod(*x, t.to64, t.p);
    }
    t.plan.inverse_mont(a);
}

/// `acc += a ⊙ b` in Montgomery form with $`R=2^{52}`$, in scalar code: $`ab2^{-52}\bmod p`$ for
/// each pair, added modulo $`p`$.
pub(crate) fn mac_scalar(t: &IfmaPlan, acc: &mut [u64], a: &[u64], b: &[u64]) {
    let p = t.p;
    let unscale = pow_mod(((1u128 << 52) % u128::from(p)) as u64, p - 2, p);
    for ((s, x), y) in acc.iter_mut().zip(a).zip(b) {
        let product = mul_mod(mul_mod(*x, *y, p), unscale, p);
        *s = (*s + product) % p;
    }
}

/// $`yw-\lfloor yw'/2^{52}\rfloor p`$ in each lane, in $`[0,2p)`$, for $`y<2^{52}`$: the
/// quotient from the high half of $`yw'`$, the remainder modulo $`2^{52}`$ as the low half of
/// $`yw`$ plus the low half of $`(2^{52}-p)\lfloor yw'/2^{52}\rfloor`$.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn shoup52(y: __m512i, w: __m512i, ws: __m512i, neg_p: __m512i, mask: __m512i) -> __m512i {
    let zero = _mm512_setzero_si512();
    let q = _mm512_madd52hi_epu64(zero, y, ws);
    let low = _mm512_madd52lo_epu64(zero, y, w);
    _mm512_and_si512(_mm512_madd52lo_epu64(low, q, neg_p), mask)
}

/// The constants of one transform in registers.
struct Consts {
    p: __m512i,
    two_p: __m512i,
    neg_p: __m512i,
    mask: __m512i,
}

/// The Cooley-Tukey butterfly with inputs below $`4p`$: both reduced below $`2p`$, the second
/// multiplied lazily, outputs $`(u+t,u+2p-t)`$ below $`4p`$.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn forward_butterfly(
    x: __m512i,
    y: __m512i,
    w: __m512i,
    ws: __m512i,
    c: &Consts,
) -> (__m512i, __m512i) {
    let u = csub(x, c.two_p);
    let t = shoup52(csub(y, c.two_p), w, ws, c.neg_p, c.mask);
    (
        _mm512_add_epi64(u, t),
        _mm512_sub_epi64(_mm512_add_epi64(u, c.two_p), t),
    )
}

/// The Gentleman-Sande butterfly with inputs below $`2p`$: $`((x+y)\bmod2p,(x-y)w)`$, the
/// difference reduced below $`2p`$ before the product, outputs below $`2p`$.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn inverse_butterfly(
    x: __m512i,
    y: __m512i,
    w: __m512i,
    ws: __m512i,
    c: &Consts,
) -> (__m512i, __m512i) {
    let s = csub(_mm512_add_epi64(x, y), c.two_p);
    let diff = csub(_mm512_sub_epi64(_mm512_add_epi64(x, c.two_p), y), c.two_p);
    (s, shoup52(diff, w, ws, c.neg_p, c.mask))
}

/// The forward transform of residues in $`[0,p)`$, returned in $`[0,p)`$ in Montgomery form
/// with $`R=2^{52}`$, for $`d\ge16`$.
#[target_feature(enable = "avx512f,avx512ifma")]
pub(super) fn forward(t: &IfmaPlan, a: &mut [u64]) {
    let d = t.d;
    assert!(d >= 16 && d.is_power_of_two() && a.len() == d);
    assert!(t.zetas.len() >= d && t.zetas_shoup.len() >= d);
    let c = Consts {
        p: splat(t.p),
        two_p: splat(2 * t.p),
        neg_p: splat((1 << 52) - t.p),
        mask: splat(MASK52),
    };
    let (chunks, _) = a.as_chunks_mut::<8>();
    // The first stage, at distance d/2: multiplies by R and zeta_1 R (inputs below p).
    let half = d / 16;
    let (r, rs) = (splat(t.r.0), splat(t.r.1));
    let (z, zs) = (splat(t.first.0), splat(t.first.1));
    for j in 0..half {
        let (x, y) = (load(&chunks[j]), load(&chunks[j + half]));
        let u = shoup52(x, r, rs, c.neg_p, c.mask);
        let v = shoup52(y, z, zs, c.neg_p, c.mask);
        store(&mut chunks[j], _mm512_add_epi64(u, v));
        store(
            &mut chunks[j + half],
            _mm512_sub_epi64(_mm512_add_epi64(u, c.two_p), v),
        );
    }
    // Distances 8 and above: group g of the stage at distance len has twiddle d/(2 len) + g.
    let mut len = d / 4;
    while len >= 8 {
        let (span, k0) = (len / 8, d / (2 * len));
        for g in 0..k0 {
            let (w, ws) = (splat(t.zetas[k0 + g]), splat(t.zetas_shoup[k0 + g]));
            for j in 2 * g * span..(2 * g + 1) * span {
                let (x, y) = (load(&chunks[j]), load(&chunks[j + span]));
                let (x, y) = forward_butterfly(x, y, w, ws, &c);
                store(&mut chunks[j], x);
                store(&mut chunks[j + span], y);
            }
        }
        len /= 2;
    }
    // Distances 4, 2 and 1 on pairs of registers, the last one also reducing into [0, p).
    for (len, regroup) in [(4, &BY4), (2, &BY2), (1, &BY1)] {
        let k0 = d / (2 * len);
        let (sx, sy) = (lanes(regroup.x), lanes(regroup.y));
        let (ja, jb) = (lanes(regroup.a), lanes(regroup.b));
        for i in 0..d / 16 {
            let (a0, b0) = (load(&chunks[2 * i]), load(&chunks[2 * i + 1]));
            let x = _mm512_permutex2var_epi64(a0, sx, b0);
            let y = _mm512_permutex2var_epi64(a0, sy, b0);
            let w = small_twiddles(&t.zetas, k0, len, i);
            let ws = small_twiddles(&t.zetas_shoup, k0, len, i);
            let (mut x, mut y) = forward_butterfly(x, y, w, ws, &c);
            if len == 1 {
                x = csub(csub(x, c.two_p), c.p);
                y = csub(csub(y, c.two_p), c.p);
            }
            store(&mut chunks[2 * i], _mm512_permutex2var_epi64(x, ja, y));
            store(&mut chunks[2 * i + 1], _mm512_permutex2var_epi64(x, jb, y));
        }
    }
}

/// The inverse of [`forward`] for values in $`[0,p)`$ in Montgomery form with $`R=2^{52}`$,
/// returned as residues in $`[0,p)`$.
#[target_feature(enable = "avx512f,avx512ifma")]
pub(super) fn inverse(t: &IfmaPlan, a: &mut [u64]) {
    let d = t.d;
    assert!(d >= 16 && d.is_power_of_two() && a.len() == d);
    assert!(t.inv_zetas.len() >= d && t.inv_zetas_shoup.len() >= d);
    let c = Consts {
        p: splat(t.p),
        two_p: splat(2 * t.p),
        neg_p: splat((1 << 52) - t.p),
        mask: splat(MASK52),
    };
    let (chunks, _) = a.as_chunks_mut::<8>();
    // Distances 1, 2 and 4 on pairs of registers.
    for (len, regroup) in [(1, &BY1), (2, &BY2), (4, &BY4)] {
        let k0 = d / (2 * len);
        let (sx, sy) = (lanes(regroup.x), lanes(regroup.y));
        let (ja, jb) = (lanes(regroup.a), lanes(regroup.b));
        for i in 0..d / 16 {
            let (a0, b0) = (load(&chunks[2 * i]), load(&chunks[2 * i + 1]));
            let x = _mm512_permutex2var_epi64(a0, sx, b0);
            let y = _mm512_permutex2var_epi64(a0, sy, b0);
            let w = small_twiddles(&t.inv_zetas, k0, len, i);
            let ws = small_twiddles(&t.inv_zetas_shoup, k0, len, i);
            let (x, y) = inverse_butterfly(x, y, w, ws, &c);
            store(&mut chunks[2 * i], _mm512_permutex2var_epi64(x, ja, y));
            store(&mut chunks[2 * i + 1], _mm512_permutex2var_epi64(x, jb, y));
        }
    }
    // Distances 8 up to d/4.
    let mut len = 8;
    while len < d / 2 {
        let (span, k0) = (len / 8, d / (2 * len));
        for g in 0..k0 {
            let (w, ws) = (splat(t.inv_zetas[k0 + g]), splat(t.inv_zetas_shoup[k0 + g]));
            for j in 2 * g * span..(2 * g + 1) * span {
                let (x, y) = (load(&chunks[j]), load(&chunks[j + span]));
                let (x, y) = inverse_butterfly(x, y, w, ws, &c);
                store(&mut chunks[j], x);
                store(&mut chunks[j + span], y);
            }
        }
        len *= 2;
    }
    // The last stage, at distance d/2: multiplies by d^-1 R^-1 and zeta_1^-1 d^-1 R^-1, and
    // reduces into [0, p).
    let half = d / 16;
    let (s, ss) = (splat(t.scale.0), splat(t.scale.1));
    let (z, zs) = (splat(t.last.0), splat(t.last.1));
    for j in 0..half {
        let (x, y) = (load(&chunks[j]), load(&chunks[j + half]));
        let sum = csub(_mm512_add_epi64(x, y), c.two_p);
        let diff = csub(_mm512_sub_epi64(_mm512_add_epi64(x, c.two_p), y), c.two_p);
        let u = shoup52(sum, s, ss, c.neg_p, c.mask);
        let v = shoup52(diff, z, zs, c.neg_p, c.mask);
        store(&mut chunks[j], csub(u, c.p));
        store(&mut chunks[j + half], csub(v, c.p));
    }
}

/// `acc += a ⊙ b` for transforms in $`[0,p)`$ in Montgomery form with $`R=2^{52}`$ (lengths
/// multiples of 8, the caller's): each product $`ab2^{-52}\bmod p`$ by Montgomery's reduction
/// in 52-bit halves. With $`m=(ab\bmod2^{52})(-p^{-1})\bmod2^{52}`$ the low halves of $`ab`$ and
/// $`mp`$ sum to 0 or $`2^{52}`$, so $`(ab+mp)/2^{52}`$ is the sum of the high halves plus that
/// carry, below $`2p`$.
#[target_feature(enable = "avx512f,avx512ifma")]
pub(super) fn mac(t: &IfmaPlan, acc: &mut [u64], a: &[u64], b: &[u64]) {
    assert!(acc.len() == a.len() && a.len() == b.len() && acc.len().is_multiple_of(8));
    let zero = _mm512_setzero_si512();
    let (p, n_prime, mask) = (splat(t.p), splat(t.n_prime), splat(MASK52));
    let (sums, _) = acc.as_chunks_mut::<8>();
    let (xs, _) = a.as_chunks::<8>();
    let (ys, _) = b.as_chunks::<8>();
    for ((s, x), y) in sums.iter_mut().zip(xs).zip(ys) {
        let (x, y) = (load(x), load(y));
        let low = _mm512_madd52lo_epu64(zero, x, y);
        let high = _mm512_madd52hi_epu64(zero, x, y);
        let m = _mm512_and_si512(_mm512_madd52lo_epu64(zero, low, n_prime), mask);
        let carry = _mm512_srli_epi64::<52>(_mm512_madd52lo_epu64(low, m, p));
        let u = csub(
            _mm512_add_epi64(_mm512_madd52hi_epu64(high, m, p), carry),
            p,
        );
        store(s, csub(_mm512_add_epi64(load(s), u), p));
    }
}

/// `sub_mod` in each lane, for residues below $`p<2^{63}`$: $`a-b`$ or $`a-b+p`$, whichever is
/// below $`p`$ (the other wraps above $`2^{63}`$).
#[inline]
#[target_feature(enable = "avx512f")]
fn sub_mod(a: __m512i, b: __m512i, p: __m512i) -> __m512i {
    let d = _mm512_sub_epi64(a, b);
    _mm512_min_epu64(d, _mm512_add_epi64(d, p))
}

/// The CRT of `modular_avx512::small_crt` with IFMA products, for primes and $`q`$ below
/// $`2^{51}`$: each Shoup product by [`shoup52`] with the constant
/// $`\lfloor w2^{52}/p\rfloor`$, which is the 64-bit one $`\lfloor w2^{64}/p\rfloor`$ shifted
/// right by 12 bits, then reduced into $`[0,p)`$, so every step gives the residue that the
/// 64-bit kernel gives. [`shoup52`] forms its result below $`2p`$ modulo $`2^{52}`$, exact for
/// $`p\le2^{51}`$, and every multiplied value is a residue below $`2^{51}`$. Returns how many
/// coefficients it wrote, a multiple of eight.
#[target_feature(enable = "avx512f,avx512ifma")]
pub(super) fn small_crt<const K: usize>(
    residues: &[&[u64]; K],
    c: &CrtConstants<K>,
    out: &mut [u64],
) -> usize {
    debug_assert!(c.q < 1 << 51 && c.primes.iter().all(|p| *p < 1 << 51));
    let mask = splat(MASK52);
    let product = |d: __m512i, (w, ws): (u64, u64), p: u64| {
        csub(
            shoup52(d, splat(w), splat(ws >> 12), splat((1 << 52) - p), mask),
            splat(p),
        )
    };
    let q = splat(c.q);
    let (head, _) = out.as_chunks_mut::<8>();
    for (chunk, o) in head.iter_mut().enumerate() {
        let j = 8 * chunk;
        let mut digits = [splat(0); K];
        for i in 0..K {
            let p = splat(c.primes[i]);
            let mut sum = splat(0);
            for (t, d) in digits[..i].iter().enumerate() {
                sum = csub(
                    _mm512_add_epi64(sum, product(*d, c.prefixes[i][t], c.primes[i])),
                    p,
                );
            }
            let r: &[u64; 8] = residues[i][j..j + 8].try_into().expect("eight residues");
            digits[i] = product(sub_mod(load(r), sum, p), c.inverses[i], c.primes[i]);
        }
        let (mut negative, mut equal): (__mmask8, __mmask8) = (0, 0xff);
        for i in (0..K).rev() {
            let h = splat(c.half[i]);
            negative |= equal & _mm512_cmpgt_epu64_mask(digits[i], h);
            equal &= _mm512_cmpeq_epu64_mask(digits[i], h);
        }
        let mut x = splat(0);
        for (d, w) in digits.iter().zip(&c.prefixes_q) {
            x = csub(_mm512_add_epi64(x, product(*d, *w, c.q)), q);
        }
        let less = sub_mod(x, splat(c.product_q), q);
        store(o, _mm512_mask_blend_epi64(negative, x, less));
    }
    8 * head.len()
}

/// The residues modulo the prime $`p`$ of `t` of the centred representatives of `a`, canonical
/// below $`q<2^{63}`$ with $`\lfloor q/2\rfloor=`$ `half`, for $`q\ge2p`$, where centred values
/// exceed $`p`$: each value $`a=h2^{52}+l`$ as $`l+h(2^{52}\bmod p)`$, below $`3p`$ since
/// $`h<2^{11}`$ and $`2^{52}\bmod p=2(2^{51}-p)<2^{18}`$ for the primes of [`IFMA_PRIMES`],
/// reduced by two conditional subtractions, then less $`q\bmod p`$ (`q_mod_p`) above `half`.
#[target_feature(enable = "avx512f")]
pub(super) fn centred_large(t: &IfmaPlan, a: &[u64], half: u64, q_mod_p: u64, out: &mut [u64]) {
    debug_assert!(a.len() == out.len() && t.r.0 < 1 << 18);
    let (p, two_p, c52) = (splat(t.p), splat(2 * t.p), splat(t.r.0));
    let (mask, hv, qp) = (splat(MASK52), splat(half), splat(q_mod_p));
    let (head, tail) = a.as_chunks::<8>();
    let (oh, ot) = out.as_chunks_mut::<8>();
    for (x, o) in head.iter().zip(oh) {
        let v = load(x);
        let s = _mm512_add_epi64(
            _mm512_and_si512(v, mask),
            _mm512_mul_epu32(_mm512_srli_epi64::<52>(v), c52),
        );
        let r = csub(csub(s, two_p), p);
        let above = _mm512_cmpgt_epu64_mask(v, hv);
        store(o, _mm512_mask_blend_epi64(above, r, sub_mod(r, qp, p)));
    }
    for (x, o) in tail.iter().zip(ot) {
        let r = x % t.p;
        *o = if *x > half {
            crate::math::ntt::sub_mod(r, q_mod_p, t.p)
        } else {
            r
        };
    }
}

/// The CRT of [`small_crt`] for IFMA primes and $`2^{51}\le q<2^{63}`$: Garner's digits with
/// IFMA products modulo the primes, as there, and their sum $`\sum_iv_i(M_i\bmod q)`$ with the
/// 64-bit Shoup products of `modular_avx512`, which $`q`$ needs. Each step gives the residue
/// that the 64-bit kernel gives. Returns how many coefficients it wrote, a multiple of eight.
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub(super) fn small_crt_mixed<const K: usize>(
    residues: &[&[u64]; K],
    c: &CrtConstants<K>,
    out: &mut [u64],
) -> usize {
    debug_assert!(c.q < 1 << 63 && c.primes.iter().all(|p| *p < 1 << 51));
    let mask = splat(MASK52);
    let product = |d: __m512i, (w, ws): (u64, u64), p: u64| {
        csub(
            shoup52(d, splat(w), splat(ws >> 12), splat((1 << 52) - p), mask),
            splat(p),
        )
    };
    let q = splat(c.q);
    let (head, _) = out.as_chunks_mut::<8>();
    for (chunk, o) in head.iter_mut().enumerate() {
        let j = 8 * chunk;
        let mut digits = [splat(0); K];
        for i in 0..K {
            let p = splat(c.primes[i]);
            let mut sum = splat(0);
            for (t, d) in digits[..i].iter().enumerate() {
                sum = csub(
                    _mm512_add_epi64(sum, product(*d, c.prefixes[i][t], c.primes[i])),
                    p,
                );
            }
            let r: &[u64; 8] = residues[i][j..j + 8].try_into().expect("eight residues");
            digits[i] = product(sub_mod(load(r), sum, p), c.inverses[i], c.primes[i]);
        }
        let (mut negative, mut equal): (__mmask8, __mmask8) = (0, 0xff);
        for i in (0..K).rev() {
            let h = splat(c.half[i]);
            negative |= equal & _mm512_cmpgt_epu64_mask(digits[i], h);
            equal &= _mm512_cmpeq_epu64_mask(digits[i], h);
        }
        let mut x = splat(0);
        for (d, (w, ws)) in digits.iter().zip(&c.prefixes_q) {
            let t = super::modular_avx512::shoup(*d, splat(*w), splat(*ws), q);
            x = csub(_mm512_add_epi64(x, t), q);
        }
        let less = sub_mod(x, splat(c.product_q), q);
        store(o, _mm512_mask_blend_epi64(negative, x, less));
    }
    8 * head.len()
}
