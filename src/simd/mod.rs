//! The crate's only `unsafe` code, behind the `simd` feature on x86-64: kernels for
//! instruction sets beyond the baseline (AVX-512, BMI1 and BMI2) that the scalar code calls
//! through the safe functions here, chosen at run time on processors that have the
//! instructions. Each kernel returns exactly what the scalar code that it replaces returns (the
//! tests compare them), so proofs do not depend on the processor or the feature. Setting the
//! environment variable `JALI_SCALAR`, to any value, keeps the scalar code everywhere.
#![allow(unsafe_code)]

mod keccak_bmi;
mod ntt_avx512;
#[cfg(test)]
mod tests;

/// The constants of an NTT plan that the kernels read: the twiddles $`\zeta_k`$ and
/// $`\zeta_k^{-1}`$ of the merged transforms with their Shoup constants, each in its own array
/// (index 0 unused), and the factors of the first forward and the last inverse stage, as
/// `(w, shoup(w))` pairs.
pub(crate) struct NttTables<'a> {
    pub(crate) p: u64,
    pub(crate) d: usize,
    pub(crate) zetas: &'a [u64],
    pub(crate) zetas_shoup: &'a [u64],
    pub(crate) inv_zetas: &'a [u64],
    pub(crate) inv_zetas_shoup: &'a [u64],
    pub(crate) r: (u64, u64),
    pub(crate) first: (u64, u64),
    pub(crate) scale: (u64, u64),
    pub(crate) last: (u64, u64),
}

use std::sync::OnceLock;

/// Whether `JALI_SCALAR` is set, read once per process.
fn scalar_forced() -> bool {
    static SET: OnceLock<bool> = OnceLock::new();
    *SET.get_or_init(|| std::env::var_os("JALI_SCALAR").is_some())
}

/// Whether the AVX-512 kernels run: the processor has AVX-512F and AVX-512DQ, and
/// `JALI_SCALAR` is unset. Decided once per process.
pub(crate) fn avx512() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !scalar_forced()
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
    })
}

/// Whether the BMI1 and BMI2 kernels run: the processor has both, and `JALI_SCALAR` is unset.
/// Decided once per process.
pub(crate) fn bmi() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !scalar_forced()
            && std::is_x86_feature_detected!("bmi1")
            && std::is_x86_feature_detected!("bmi2")
    })
}

/// Keccak-f[1600] on `state` when the BMI kernel runs: `true` if `state` is permuted, `false`
/// (and `state` untouched) otherwise.
#[inline]
pub(crate) fn keccak_f1600(state: &mut [u64; 25]) -> bool {
    if bmi() {
        // SAFETY: the processor has BMI1 and BMI2 (`bmi`), all the kernel needs.
        unsafe { keccak_bmi::f1600(state) };
        return true;
    }
    false
}

/// `NttPlan::forward_mont` of `a` when a kernel runs for its degree: `true` if `a` is
/// transformed, `false` (and `a` untouched) otherwise.
#[inline]
pub(crate) fn forward_mont(t: &NttTables<'_>, a: &mut [u64]) -> bool {
    if t.d >= 16 && a.len() == t.d && avx512() {
        // SAFETY: the processor has AVX-512F and AVX-512DQ (`avx512`), and the kernel reads
        // and writes `a` and the tables within their lengths, which it checks.
        unsafe { ntt_avx512::forward_mont(t, a) };
        return true;
    }
    false
}

/// `NttPlan::inverse_mont` of `a` when a kernel runs for its degree, as [`forward_mont`].
#[inline]
pub(crate) fn inverse_mont(t: &NttTables<'_>, a: &mut [u64]) -> bool {
    if t.d >= 16 && a.len() == t.d && avx512() {
        // SAFETY: as in `forward_mont`.
        unsafe { ntt_avx512::inverse_mont(t, a) };
        return true;
    }
    false
}
