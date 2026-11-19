//! Helpers shared by the AVX-512 kernels: broadcasts, loads and stores of eight 64-bit lanes
//! (whole arrays, or the first lanes of a shorter slice under a mask), the high half of 64-bit
//! products, and the conditional subtraction of a modulus.
use core::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx512f")]
pub(super) fn splat(x: u64) -> __m512i {
    _mm512_set1_epi64(x as i64)
}
/// The vector of eight lane indices, lane 0 first.
#[inline]
#[target_feature(enable = "avx512f")]
pub(super) fn lanes(i: [i64; 8]) -> __m512i {
    _mm512_set_epi64(i[7], i[6], i[5], i[4], i[3], i[2], i[1], i[0])
}
#[inline]
#[target_feature(enable = "avx512f")]
pub(super) fn load(a: &[u64; 8]) -> __m512i {
    // SAFETY: eight readable values, and the load needs no alignment.
    unsafe { _mm512_loadu_si512(a.as_ptr().cast()) }
}
/// Four values into lanes 0 to 3, the others zero.
#[inline]
#[target_feature(enable = "avx512f")]
pub(super) fn load4(a: &[u64; 4]) -> __m512i {
    // SAFETY: the mask reads only the four values, without alignment.
    unsafe { _mm512_maskz_loadu_epi64(0x0f, a.as_ptr().cast()) }
}
#[inline]
#[target_feature(enable = "avx512f")]
pub(super) fn store(a: &mut [u64; 8], v: __m512i) {
    // SAFETY: eight writable values, and the store needs no alignment.
    unsafe { _mm512_storeu_si512(a.as_mut_ptr().cast(), v) }
}
/// The high 64 bits of each lane's 128-bit product: with $`a=a_12^{32}+a_0`$ and
/// $`b=b_12^{32}+b_0`$, $`m=a_1b_0+\lfloor a_0b_0/2^{32}\rfloor`$ and
/// $`m'=a_0b_1+(m\bmod2^{32})`$ stay below $`2^{64}`$, and the high half is
/// $`a_1b_1+\lfloor m/2^{32}\rfloor+\lfloor m'/2^{32}\rfloor`$.
#[inline]
#[target_feature(enable = "avx512f")]
pub(super) fn mulhi(a: __m512i, b: __m512i) -> __m512i {
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
/// The first `a.len()` lanes from `a`, which holds at most eight values, the others zero.
#[inline]
#[target_feature(enable = "avx512f")]
pub(super) fn load_part(a: &[u64]) -> __m512i {
    debug_assert!(a.len() <= 8);
    let mask = ((1u16 << a.len()) - 1) as u8;
    // SAFETY: the mask reads only the `a.len()` values of the slice, and a lane outside the
    // mask is never accessed, so an empty slice reads nothing.
    unsafe { _mm512_maskz_loadu_epi64(mask, a.as_ptr().cast()) }
}
/// The first `a.len()` lanes of `v` into `a`, which holds at most eight values.
#[inline]
#[target_feature(enable = "avx512f")]
pub(super) fn store_part(a: &mut [u64], v: __m512i) {
    debug_assert!(a.len() <= 8);
    let mask = ((1u16 << a.len()) - 1) as u8;
    // SAFETY: the mask writes only the `a.len()` values of the slice.
    unsafe { _mm512_mask_storeu_epi64(a.as_mut_ptr().cast(), mask, v) }
}
/// $`x`$ or $`x-m`$, whichever is below $`m`$, for $`x<2m`$: below $`m`$ the difference wraps
/// above $`x`$, so the unsigned minimum picks the right one.
#[inline]
#[target_feature(enable = "avx512f")]
pub(super) fn csub(x: __m512i, m: __m512i) -> __m512i {
    _mm512_min_epu64(x, _mm512_sub_epi64(x, m))
}
