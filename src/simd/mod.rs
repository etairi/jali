//! The crate's only `unsafe` code, behind the `simd` feature on x86-64: kernels for
//! instruction sets beyond the baseline (AVX-512, BMI1 and BMI2) that the scalar code calls
//! through the safe functions here, chosen at run time on processors that have the
//! instructions. Each kernel returns exactly what the scalar code that it replaces returns (the
//! tests compare them), so proofs do not depend on the processor or the feature. Setting the
//! environment variable `JALI_SCALAR`, to any value, keeps the scalar code everywhere.
#![allow(unsafe_code)]

mod bytes_avx512;
mod ifma;
mod keccak_bmi;
mod lanes;
mod modular_avx512;
mod ntt_avx512;
mod rns_avx512;

pub(crate) use ifma::{IFMA_PRIMES, IfmaPlan};
pub(crate) use modular_avx512::CrtConstants;
pub(crate) use rns_avx512::ResidueConstants;
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

/// Whether the AVX-512 byte kernels run: [`avx512`] and AVX-512BW. Decided once per process.
pub(crate) fn avx512bw() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| avx512() && std::is_x86_feature_detected!("avx512bw"))
}

/// Whether the IFMA kernels run: [`avx512`] and AVX-512 IFMA. Decided once per process.
pub(crate) fn ifma() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| avx512() && std::is_x86_feature_detected!("avx512ifma"))
}

/// The forward transform of an [`IfmaPlan`] (residues in $`[0,p)`$ to Montgomery form with
/// $`R=2^{52}`$): the IFMA kernel where it runs, the scalar code of the same arithmetic
/// otherwise.
#[inline]
pub(crate) fn ifma_forward(t: &IfmaPlan, a: &mut [u64]) {
    if ifma() && a.len() >= 16 {
        // SAFETY: the processor has AVX-512F and AVX-512 IFMA (`ifma`), and the kernel reads and
        // writes only the slice's elements, whose length it checks against the plan.
        unsafe { ifma::forward(t, a) };
    } else {
        ifma::forward_scalar(t, a);
    }
}

/// The inverse of [`ifma_forward`], as it.
#[inline]
pub(crate) fn ifma_inverse(t: &IfmaPlan, a: &mut [u64]) {
    if ifma() && a.len() >= 16 {
        // SAFETY: as in `ifma_forward`.
        unsafe { ifma::inverse(t, a) };
    } else {
        ifma::inverse_scalar(t, a);
    }
}

/// `acc += a ⊙ b` for transforms of an [`IfmaPlan`], as [`ifma_forward`].
#[inline]
pub(crate) fn ifma_mac(t: &IfmaPlan, acc: &mut [u64], a: &[u64], b: &[u64]) {
    if ifma() && acc.len() == a.len() && a.len() == b.len() && acc.len().is_multiple_of(8) {
        // SAFETY: as in `ifma_forward`, with the lengths checked here.
        unsafe { ifma::mac(t, acc, a, b) };
    } else {
        ifma::mac_scalar(t, acc, a, b);
    }
}

/// Whether the AVX-512 kernels that need AVX-512VL run: [`avx512`] and AVX-512VL. Decided once
/// per process.
pub(crate) fn avx512vl() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| avx512() && std::is_x86_feature_detected!("avx512vl"))
}

/// Whether the AVX-512 VBMI kernels run: [`avx512vl`], [`avx512bw`] and AVX-512 VBMI. Decided
/// once per process.
pub(crate) fn vbmi() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| avx512vl() && avx512bw() && std::is_x86_feature_detected!("avx512vbmi"))
}

/// The raw column sums of LaBRADOR's projection table of sign patterns (32-bit sums, `map` from
/// a byte to its pattern) for eight columns, as `level::ProjectionTable::sums` adds them before
/// reducing, when the AVX-512 kernels run with AVX-512VL and $`\lambda\le8`$: `true` if `out` is
/// filled.
#[inline]
pub(crate) fn pattern_sums8(
    table: &[i32],
    lambda: usize,
    map: &[u8; 256],
    columns: &[[u8; 64]; 8],
    out: &mut [[i64; 8]; 8],
) -> bool {
    if !(1..=8).contains(&lambda) || table.len() < 64 * 81 * lambda || !avx512vl() {
        return false;
    }
    if vbmi() {
        // SAFETY: the processor has AVX-512F, VL, BW and VBMI (`vbmi`), and the kernel reads only
        // the table's entries, by slices whose bounds it checks (patterns are below 81).
        unsafe { modular_avx512::pattern_sums8_vbmi(table, lambda, map, columns, out) };
    } else {
        // SAFETY: as above, with AVX-512F and VL (`avx512vl`).
        unsafe { modular_avx512::pattern_sums8(table, lambda, map, columns, out) };
    }
    true
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

/// The elementwise kernels: each applies its operation to the slices when the AVX-512 kernels
/// run and the lengths agree, and returns `true`, or returns `false` with nothing changed.
macro_rules! modular {
    ($(#[$doc:meta])* $name:ident($($arg:ident: $ty:ty),*) where $len:expr) => {
        $(#[$doc])*
        #[inline]
        pub(crate) fn $name($($arg: $ty),*) -> bool {
            if $len && avx512() {
                // SAFETY: the processor has AVX-512F and AVX-512DQ (`avx512`), and the kernel
                // reads and writes only the elements of its slices.
                unsafe { modular_avx512::$name($($arg),*) };
                return true;
            }
            false
        }
    };
}
modular!(
    /// `a[i] = add_mod(a[i], b[i], p)`, for residues below $`p\le2^{63}`$.
    add_mod(a: &mut [u64], b: &[u64], p: u64) where a.len() == b.len()
);
modular!(
    /// `a[i] = sub_mod(a[i], b[i], p)`, for residues below $`p\le2^{63}`$.
    sub_mod(a: &mut [u64], b: &[u64], p: u64) where a.len() == b.len()
);
modular!(
    /// `a[i] = add_mod(a[i], shoup_mul(b[i], w, w_shoup, p), p)`, for $`w<p<2^{63}`$.
    add_scaled(a: &mut [u64], b: &[u64], w: u64, w_shoup: u64, p: u64) where a.len() == b.len()
);
modular!(
    /// `a[i] = sub_mod(a[i], shoup_mul(b[i], w, w_shoup, p), p)`, for $`w<p<2^{63}`$.
    sub_scaled(a: &mut [u64], b: &[u64], w: u64, w_shoup: u64, p: u64) where a.len() == b.len()
);
modular!(
    /// `a[i] = shoup_mul(a[i], w, w_shoup, p)`, for $`w<p<2^{63}`$.
    scale(a: &mut [u64], w: u64, w_shoup: u64, p: u64) where true
);
modular!(
    /// `acc[i] = add_mod(acc[i], mont_mul(a[i], b[i]), p)` for an NTT prime $`p<2^{62}`$ with
    /// `neg_inv` $`=-p^{-1}\bmod2^{64}`$.
    mont_mac(acc: &mut [u64], a: &[u64], b: &[u64], p: u64, neg_inv: u64)
        where acc.len() == a.len() && a.len() == b.len()
);
modular!(
    /// `out[i]` the residue modulo $`p`$ of the centred representative of `a[i]` modulo
    /// $`q<2p`$.
    centred(a: &[u64], q: u64, p: u64, out: &mut [u64]) where a.len() == out.len()
);

/// The residues modulo the prime of `t` of the centred representatives of `a`, canonical below
/// $`q<2^{63}`$ with $`\lfloor q/2\rfloor=`$ `half`, for $`q\ge2p`$ (`ifma::centred_large`),
/// with the scalar code of the same rule where AVX-512 does not run (an IFMA ring exists only
/// where it does).
#[inline]
pub(crate) fn ifma_centred(t: &IfmaPlan, a: &[u64], half: u64, q_mod_p: u64, out: &mut [u64]) {
    if a.len() == out.len() && avx512() {
        // SAFETY: the processor has AVX-512F (`avx512`), and the kernel reads `a` and writes
        // `out` within their equal lengths.
        unsafe { ifma::centred_large(t, a, half, q_mod_p, out) };
        return;
    }
    let p = t.plan.modulus();
    for (o, x) in out.iter_mut().zip(a) {
        let r = x % p;
        *o = if *x > half {
            crate::math::ntt::sub_mod(r, q_mod_p, p)
        } else {
            r
        };
    }
}

/// `SmallCrt::reconstruct_fixed` of the first coefficients of `out` when the AVX-512 kernels
/// run: how many it wrote (a multiple of eight, all but the last `out.len() % 8`), zero
/// otherwise. Each residue slice must hold at least `out.len()` values.
#[inline]
pub(crate) fn small_crt<const K: usize>(
    residues: &[&[u64]; K],
    c: &CrtConstants<K>,
    out: &mut [u64],
) -> usize {
    if residues.iter().all(|r| r.len() >= out.len()) && avx512() {
        if c.primes.iter().all(|p| *p < 1 << 51) && ifma() {
            // SAFETY: the processor has AVX-512F, DQ and IFMA (`ifma` within `avx512`), and
            // the kernels read the residues and write `out` within their lengths.
            return unsafe {
                if c.q < 1 << 51 {
                    ifma::small_crt(residues, c, out)
                } else {
                    ifma::small_crt_mixed(residues, c, out)
                }
            };
        }
        // SAFETY: as in `lazy_mac`.
        return unsafe { modular_avx512::small_crt(residues, c, out) };
    }
    0
}

/// The columns `columns` with their coefficients `values` added into the 8-bit lanes of
/// `level::project_bytes` (lane $`g`$ of `lanes[e]` is row $`4g+e`$) when the AVX-512 byte
/// kernels run: `true` if they did.
#[inline]
pub(crate) fn project_bytes(
    lanes: &mut [[i8; 64]; 4],
    columns: &[[u8; 64]],
    values: &[i8],
) -> bool {
    if columns.len() == values.len() && avx512bw() {
        // SAFETY: the processor has AVX-512F and AVX-512BW (`avx512bw`), and the kernel reads
        // and writes only the arrays and slices it is given.
        unsafe { bytes_avx512::project_bytes(lanes, columns, values) };
        return true;
    }
    false
}

/// The raw column sums of LaBRADOR's projection table for eight columns, as
/// `level::column_sums` adds them before reducing, when the AVX-512 kernels run and
/// $`\lambda\le8`$: `true` if `out` is filled.
#[inline]
pub(crate) fn table_sums8(
    table: &[i64],
    lambda: usize,
    columns: &[[u8; 64]; 8],
    out: &mut [[i64; 8]; 8],
) -> bool {
    if (1..=8).contains(&lambda) && table.len() >= 64 * 256 * lambda && avx512() {
        // SAFETY: the processor has AVX-512F (`avx512`), and the kernel reads only the table's
        // entries, by slices whose bounds it checks.
        unsafe { modular_avx512::table_sums8(table, lambda, columns, out) };
        return true;
    }
    false
}

/// `sums[i] += x[i] * y[i]` as 128-bit integers when the AVX-512 kernels run and the lengths
/// agree: `true` if done, `false` (and nothing changed) otherwise. The caller bounds the sums.
#[inline]
pub(crate) fn lazy_mac(sums: &mut [u128], x: &[u64], y: &[u64]) -> bool {
    if sums.len() == x.len() && x.len() == y.len() && avx512() {
        // SAFETY: the processor has AVX-512F and AVX-512DQ (`avx512`), and the kernel reads
        // and writes only the elements of its slices.
        unsafe { rns_avx512::lazy_mac(sums, x, y) };
        return true;
    }
    false
}

/// The centred residues of canonical values below $`q<2^{128}`$ modulo one prime, as
/// `centred_residues` computes them, when the AVX-512 kernels run (the vector is allocated
/// only then, so that the scalar code keeps collecting its own).
#[inline]
pub(crate) fn centred_wide(values: &[u128], half: u128, k: &ResidueConstants) -> Option<Vec<u64>> {
    if !avx512() {
        return None;
    }
    let mut out = vec![0; values.len()];
    // SAFETY: as in `lazy_mac`.
    unsafe { rns_avx512::centred_wide(values, half, k, &mut out) };
    Some(out)
}

/// The centred residues of canonical values below $`q<2^{63}`$ modulo one prime when the
/// AVX-512 kernels run, as in [`centred_wide`].
#[inline]
pub(crate) fn centred_word(values: &[u64], half: u64, p: u64, q_mod_p: u64) -> Option<Vec<u64>> {
    if !avx512() {
        return None;
    }
    let mut out = vec![0; values.len()];
    // SAFETY: as in `lazy_mac`.
    unsafe { rns_avx512::centred_word(values, half, p, q_mod_p, &mut out) };
    Some(out)
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
