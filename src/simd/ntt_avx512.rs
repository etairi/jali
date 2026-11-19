//! The merged negacyclic NTTs of `math::ntt` on AVX-512 registers of 8 residues: the
//! butterflies of the radix-2 stages of `NttPlan::forward_mont_radix2` and
//! `NttPlan::inverse_mont_radix2`, with the same Shoup products in 64-bit lanes (the high half
//! of each 64-bit product from four 32-bit products, the low halves from `vpmullq`) and the
//! same bounds ($`4p`$ forward, $`2p`$ inverse, below $`2^{64}`$ for $`p<2^{62}`$). The outputs
//! are reduced into $`[0,p)`$ like the scalar ones, so they are the same values. Stages at
//! distance 8 or more broadcast one twiddle per group; the three below regroup the lanes of two
//! registers, so that each butterfly has its two inputs in the same lane of two registers.
use super::NttTables;
use core::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx512f")]
fn splat(x: u64) -> __m512i {
    _mm512_set1_epi64(x as i64)
}
/// The vector of eight lane indices, lane 0 first.
#[inline]
#[target_feature(enable = "avx512f")]
fn lanes(i: [i64; 8]) -> __m512i {
    _mm512_set_epi64(i[7], i[6], i[5], i[4], i[3], i[2], i[1], i[0])
}
#[inline]
#[target_feature(enable = "avx512f")]
fn load(a: &[u64; 8]) -> __m512i {
    // SAFETY: eight readable values, and the load needs no alignment.
    unsafe { _mm512_loadu_si512(a.as_ptr().cast()) }
}
/// Four values into lanes 0 to 3, the others zero.
#[inline]
#[target_feature(enable = "avx512f")]
fn load4(a: &[u64; 4]) -> __m512i {
    // SAFETY: the mask reads only the four values, without alignment.
    unsafe { _mm512_maskz_loadu_epi64(0x0f, a.as_ptr().cast()) }
}
#[inline]
#[target_feature(enable = "avx512f")]
fn store(a: &mut [u64; 8], v: __m512i) {
    // SAFETY: eight writable values, and the store needs no alignment.
    unsafe { _mm512_storeu_si512(a.as_mut_ptr().cast(), v) }
}
/// The high 64 bits of each lane's 128-bit product: with $`a=a_12^{32}+a_0`$ and
/// $`b=b_12^{32}+b_0`$, $`m=a_1b_0+\lfloor a_0b_0/2^{32}\rfloor`$ and
/// $`m'=a_0b_1+(m\bmod2^{32})`$ stay below $`2^{64}`$, and the high half is
/// $`a_1b_1+\lfloor m/2^{32}\rfloor+\lfloor m'/2^{32}\rfloor`$.
#[inline]
#[target_feature(enable = "avx512f")]
fn mulhi(a: __m512i, b: __m512i) -> __m512i {
    let (a1, b1) = (_mm512_srli_epi64::<32>(a), _mm512_srli_epi64::<32>(b));
    let low = _mm512_mul_epu32(a, b);
    let mid = _mm512_add_epi64(_mm512_mul_epu32(a1, b), _mm512_srli_epi64::<32>(low));
    let mid2 = _mm512_add_epi64(
        _mm512_mul_epu32(a, b1),
        _mm512_and_si512(mid, splat(0xffff_ffff)),
    );
    _mm512_add_epi64(
        _mm512_add_epi64(_mm512_mul_epu32(a1, b1), _mm512_srli_epi64::<32>(mid)),
        _mm512_srli_epi64::<32>(mid2),
    )
}
/// `ntt::shoup_lazy` in each lane: $`xw-\lfloor xw'/2^{64}\rfloor p`$, in $`[0,2p)`$.
#[inline]
#[target_feature(enable = "avx512f,avx512dq")]
fn shoup_lazy(x: __m512i, w: __m512i, ws: __m512i, p: __m512i) -> __m512i {
    let k = mulhi(x, ws);
    _mm512_sub_epi64(_mm512_mullo_epi64(x, w), _mm512_mullo_epi64(k, p))
}
/// $`x`$ or $`x-m`$, whichever is below $`m`$, for $`x<2m`$: below $`m`$ the difference wraps
/// above $`x`$, so the unsigned minimum picks the right one.
#[inline]
#[target_feature(enable = "avx512f")]
fn csub(x: __m512i, m: __m512i) -> __m512i {
    _mm512_min_epu64(x, _mm512_sub_epi64(x, m))
}
/// The Cooley-Tukey butterfly of the forward stages: $`u=x\bmod2p`$, $`t=yw\bmod p`$ lazily,
/// $`(u+t,u+2p-t)`$, both below $`4p`$.
#[inline]
#[target_feature(enable = "avx512f,avx512dq")]
fn forward_butterfly(
    x: __m512i,
    y: __m512i,
    w: __m512i,
    ws: __m512i,
    p: __m512i,
    two_p: __m512i,
) -> (__m512i, __m512i) {
    let u = csub(x, two_p);
    let t = shoup_lazy(y, w, ws, p);
    (
        _mm512_add_epi64(u, t),
        _mm512_sub_epi64(_mm512_add_epi64(u, two_p), t),
    )
}
/// The Gentleman-Sande butterfly of the inverse stages: $`((x+y)\bmod2p,(x-y)w)`$, both below
/// $`2p`$ for inputs below $`2p`$.
#[inline]
#[target_feature(enable = "avx512f,avx512dq")]
fn inverse_butterfly(
    x: __m512i,
    y: __m512i,
    w: __m512i,
    ws: __m512i,
    p: __m512i,
    two_p: __m512i,
) -> (__m512i, __m512i) {
    let s = csub(_mm512_add_epi64(x, y), two_p);
    let t = shoup_lazy(_mm512_sub_epi64(_mm512_add_epi64(x, two_p), y), w, ws, p);
    (s, t)
}

/// How the lanes of a pair of registers $`A,B`$ (16 consecutive values) regroup for a stage
/// at distance 4, 2 or 1: `split` gathers the first inputs $`X`$ and the second inputs $`Y`$ of
/// the eight butterflies, `join` puts the outputs back (indices from 8 on select from the second
/// register).
struct Regroup {
    x: [i64; 8],
    y: [i64; 8],
    a: [i64; 8],
    b: [i64; 8],
}
const BY4: Regroup = Regroup {
    x: [0, 1, 2, 3, 8, 9, 10, 11],
    y: [4, 5, 6, 7, 12, 13, 14, 15],
    a: [0, 1, 2, 3, 8, 9, 10, 11],
    b: [4, 5, 6, 7, 12, 13, 14, 15],
};
const BY2: Regroup = Regroup {
    x: [0, 1, 4, 5, 8, 9, 12, 13],
    y: [2, 3, 6, 7, 10, 11, 14, 15],
    a: [0, 1, 8, 9, 2, 3, 10, 11],
    b: [4, 5, 12, 13, 6, 7, 14, 15],
};
const BY1: Regroup = Regroup {
    x: [0, 2, 4, 6, 8, 10, 12, 14],
    y: [1, 3, 5, 7, 9, 11, 13, 15],
    a: [0, 8, 1, 9, 2, 10, 3, 11],
    b: [4, 12, 5, 13, 6, 14, 7, 15],
};

/// The twiddles of the eight butterflies of pair `i` in the stage at distance `len` (4, 2 or
/// 1), whose groups start at index `k0`: group $`k_0+(16/2\mathrm{len})i+j`$ for the lanes of
/// its butterflies, in the lane order of [`Regroup`]'s `x`.
#[inline]
#[target_feature(enable = "avx512f")]
fn small_twiddles(table: &[u64], k0: usize, len: usize, i: usize) -> __m512i {
    match len {
        4 => {
            let g = k0 + 2 * i;
            _mm512_mask_blend_epi64(0xf0, splat(table[g]), splat(table[g + 1]))
        }
        2 => {
            let g = k0 + 4 * i;
            let four: &[u64; 4] = table[g..g + 4].try_into().expect("four twiddles");
            _mm512_permutexvar_epi64(lanes([0, 0, 1, 1, 2, 2, 3, 3]), load4(four))
        }
        _ => {
            let g = k0 + 8 * i;
            load(table[g..g + 8].try_into().expect("eight twiddles"))
        }
    }
}

/// `NttPlan::forward_mont` for $`d\ge16`$.
#[target_feature(enable = "avx512f,avx512dq")]
pub(super) fn forward_mont(t: &NttTables<'_>, a: &mut [u64]) {
    let d = t.d;
    assert!(d >= 16 && d.is_power_of_two() && a.len() == d);
    assert!(t.zetas.len() >= d && t.zetas_shoup.len() >= d);
    let (p, two_p) = (splat(t.p), splat(2 * t.p));
    let (chunks, _) = a.as_chunks_mut::<8>();
    // The first stage, at distance d/2: multiplies by R and zeta_1 R.
    let half = d / 16;
    let (r, rs) = (splat(t.r.0), splat(t.r.1));
    let (z, zs) = (splat(t.first.0), splat(t.first.1));
    for j in 0..half {
        let (x, y) = (load(&chunks[j]), load(&chunks[j + half]));
        let u = shoup_lazy(x, r, rs, p);
        let v = shoup_lazy(y, z, zs, p);
        store(&mut chunks[j], _mm512_add_epi64(u, v));
        store(
            &mut chunks[j + half],
            _mm512_sub_epi64(_mm512_add_epi64(u, two_p), v),
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
                let (x, y) = forward_butterfly(x, y, w, ws, p, two_p);
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
            let w = small_twiddles(t.zetas, k0, len, i);
            let ws = small_twiddles(t.zetas_shoup, k0, len, i);
            let (mut x, mut y) = forward_butterfly(x, y, w, ws, p, two_p);
            if len == 1 {
                x = csub(csub(x, two_p), p);
                y = csub(csub(y, two_p), p);
            }
            store(&mut chunks[2 * i], _mm512_permutex2var_epi64(x, ja, y));
            store(&mut chunks[2 * i + 1], _mm512_permutex2var_epi64(x, jb, y));
        }
    }
}

/// `NttPlan::inverse_mont` for $`d\ge16`$.
#[target_feature(enable = "avx512f,avx512dq")]
pub(super) fn inverse_mont(t: &NttTables<'_>, a: &mut [u64]) {
    let d = t.d;
    assert!(d >= 16 && d.is_power_of_two() && a.len() == d);
    assert!(t.inv_zetas.len() >= d && t.inv_zetas_shoup.len() >= d);
    let (p, two_p) = (splat(t.p), splat(2 * t.p));
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
            let w = small_twiddles(t.inv_zetas, k0, len, i);
            let ws = small_twiddles(t.inv_zetas_shoup, k0, len, i);
            let (x, y) = inverse_butterfly(x, y, w, ws, p, two_p);
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
                let (x, y) = inverse_butterfly(x, y, w, ws, p, two_p);
                store(&mut chunks[j], x);
                store(&mut chunks[j + span], y);
            }
        }
        len *= 2;
    }
    // The last stage, at distance d/2: multiplies by c = d^-1 R^-1 and zeta_1^-1 c, and
    // reduces into [0, p).
    let half = d / 16;
    let (c, cs) = (splat(t.scale.0), splat(t.scale.1));
    let (z, zs) = (splat(t.last.0), splat(t.last.1));
    for j in 0..half {
        let (x, y) = (load(&chunks[j]), load(&chunks[j + half]));
        let s = shoup_lazy(_mm512_add_epi64(x, y), c, cs, p);
        let v = shoup_lazy(_mm512_sub_epi64(_mm512_add_epi64(x, two_p), y), z, zs, p);
        store(&mut chunks[j], csub(s, p));
        store(&mut chunks[j + half], csub(v, p));
    }
}
