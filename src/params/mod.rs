//! Public parameter validation and derivation.
pub mod moduli;

/// Bimodal rejection constant $`M=\exp(337\alpha^2/(2\sigma^2))`$ of a range proof, rounded up
/// to an integer, for the width $`\sigma=1.55\cdot2^t`$ and a Euclidean bound
/// $`\|e\|\le\alpha`$ on the projected vector (LNP22 Fig. 10 and §6.1:
/// $`s=\gamma\sqrt{337}\alpha`$, $`M=\exp(1/(2\gamma^2))`$). The prover and the parameter check
/// both use this function, so an accepted parameter set can always run its rejection step.
/// Constants of $`2^{16}`$ or more are rejected, and so are exponents above [`MAX_LOG_SIGMA`].
pub fn range_rejection_constant(log_sigma: u32, alpha_squared: u128) -> Result<u64, Error> {
    if log_sigma > MAX_LOG_SIGMA {
        return Err(Error::Parameter("Gaussian width capacity"));
    }
    let sigma = 1.55 * (2.0f64).powi(log_sigma as i32);
    let m = ((337.0 * alpha_squared as f64 / (2.0 * sigma * sigma)).exp() + 1e-10).ceil();
    if !m.is_finite() || m >= 65536.0 {
        return Err(Error::Parameter("range rejection M"));
    }
    Ok(m as u64)
}

use crate::{
    Error,
    dcompress::Compression,
    math::{Ring, U256, int},
    rand::MAX_LOG_SIGMA,
};

/// Parameters supplied by the offline lattice-estimation/search tool.
///
/// The checker validates the protocol inequalities. An estimator report is evidence supplied
/// by the caller, not a cryptographic certificate of hardness.
///
/// Serde formats that are not human-readable, such as postcard, take each prime factor as 32
/// little-endian bytes and the other integers in the format's own form; see `prime_factors`
/// for JSON.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct TboxParams {
    /// Human-readable identifier, bound into transcripts.
    pub id: String,
    /// One or two ascending prime factors, each congruent to 5 modulo 8, whose product is below
    /// $`2^{256}`$. In JSON, a number below $`2^{64}`$ and a decimal string otherwise.
    #[cfg_attr(feature = "serde", serde(with = "crate::json::u256_list"))]
    pub prime_factors: Vec<U256>,
    /// Proof-ring degree (64 or 128).
    pub degree: usize,
    /// Original Ajtai witness length, before slack polynomials.
    pub m1: usize,
    /// Randomness length.
    pub m2: usize,
    /// BDLOP message length.
    pub l: usize,
    /// MSIS rank.
    pub n_msis: usize,
    /// Bound squared on the original Ajtai witness.
    #[cfg_attr(feature = "serde", serde(with = "crate::json::u64_field"))]
    pub alpha_squared: u64,
    /// Number of binary-block rows.
    pub n_bin: usize,
    /// Number of rows for each exact norm block.
    pub l2_rows: Vec<usize>,
    /// Squared norm bounds of those blocks.
    #[cfg_attr(feature = "serde", serde(with = "crate::json::u64_list"))]
    pub l2_bounds_squared: Vec<u64>,
    /// Approximate range block rows, zero when absent.
    pub n_prime: usize,
    /// Approximate range bound, zero exactly when absent.
    #[cfg_attr(feature = "serde", serde(with = "crate::json::u64_field"))]
    pub linf_bound: u64,
    /// Gaussian exponents: widths equal $`1.55\cdot2^t`$.
    pub log_sigma: [u32; 4],
    /// Compression divisor.
    #[cfg_attr(feature = "serde", serde(with = "crate::json::u64_field"))]
    pub gamma: u64,
    /// Rounding exponent.
    pub d_bits: u32,
    /// MLWE rank supplied by the parameter estimator.
    pub mlwe_rank: usize,
    /// Root Hermite factor of the estimate behind `mlwe_rank`; `check` requires at most 1.0044.
    pub mlwe_delta: f64,
    /// Estimator version/commit and model provenance. Kept in the parameter fingerprint.
    pub estimator: String,
}

/// Derived values after checks; no unchecked cached quantities are accepted from JSON.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedParams {
    /// Product of the prime factors.
    pub q: U256,
    /// Number of independent evaluation challenges (even).
    pub lambda: usize,
    /// Challenge coefficient bound.
    pub omega: i128,
    /// Challenge operator-norm bound $`\eta`$ of LNP22 §2.7, used in the parameter analysis
    /// and met by every challenge: the challenge set holds only challenges within it
    /// (`rand::challenge`, `rand::within_eta`).
    pub eta: u64,
    /// Length of the exact-range vector, including binary and slack coordinates.
    pub n_ex: usize,
    /// Total number of extension slots.
    pub l_ext: usize,
    /// Bound on the joint compressed randomness response.
    pub b_squared: u128,
    /// Squared bound on the first response.
    pub z1_bound_squared: u128,
    /// Squared bound on the exact-range response.
    pub z3_bound_squared: u128,
    /// Infinity bound on the approximate-range response.
    pub z4_bound: u128,
    /// Squared Euclidean bound on the exact-range vector: binary rows, exact-norm rows and slack.
    pub exact_alpha_squared: u128,
    /// Squared Euclidean bound on the approximate-range vector,
    /// $`n'd\cdot\beta_\infty^2`$ for $`n'd`$ coefficients of absolute value at most
    /// `linf_bound`. The range proof's Gaussian must be sized from this bound, not from
    /// `linf_bound` itself (LNP22 Fig. 10: $`\|e^{(d)}\|\le\alpha^{(d)}`$).
    pub approx_alpha_squared: u128,
    /// Infinity bound that extraction guarantees for the approximate-range vector:
    /// $`2\cdot`$`z4_bound`, by LNP22 Lemma 2.7 applied to the verifier's bound on the response.
    pub approx_extraction_bound: u128,
    /// Paper's exact-range extraction bound (t=1.64).
    pub arp_bound: f64,
    /// MSIS root Hermite factor from the specified closed form.
    pub msis_delta: f64,
    /// Estimated proof bytes; not a decoder allocation limit. The full-size polynomials are
    /// counted exactly, a Gaussian coefficient as $`\lceil\log_2\sigma+2.5\rceil`$ bits, and a
    /// hint coefficient as $`\max(2.25,1.6\sigma_2/\gamma+0.7)`$ bits, an upper bound on its
    /// expected length in a model of the hints. It is not a proven bound on a proof's length.
    pub estimated_proof_bytes: usize,
}

/// The transcript encoding of the prime factors.
///
/// Narrow form, when every factor is below $`2^{64}`$: $`\mathrm{LE64}(n)`$ and each factor as
/// $`\mathrm{LE64}`$. Wide form otherwise:
/// $`\mathrm{LE64}(n+2^{63})`$, then each factor $`p`$ as $`\mathrm{LE64}(k)`$ and its $`k`$
/// low-order bytes, $`k=\lceil\mathrm{bits}(p)/8\rceil`$. Narrow counts are one or two, so
/// the top bit tells the forms apart, and each wide factor is length-prefixed.
fn prime_factor_bytes(factors: &[U256], out: &mut Vec<u8>) {
    let small: Option<Vec<u64>> = factors.iter().map(int::to_u64).collect();
    match small {
        Some(small) => {
            out.extend_from_slice(&(small.len() as u64).to_le_bytes());
            for p in small {
                out.extend_from_slice(&p.to_le_bytes());
            }
        }
        None => {
            out.extend_from_slice(&((factors.len() as u64) | (1 << 63)).to_le_bytes());
            for p in factors {
                let k = int::bits(p).div_ceil(8) as usize;
                out.extend_from_slice(&(k as u64).to_le_bytes());
                out.extend_from_slice(&p.to_le_bytes()[..k]);
            }
        }
    }
}

/// Bits per hint coefficient in `estimated_proof_bytes`: $`\max(2.25,\,1.6s+0.7)`$ for
/// $`s=\sigma_2/\gamma`$, so 2.25 up to $`s=0.96875`$, the average that LNP22 §6.1 takes for
/// hints in $`\{-1,0,1\}`$.
///
/// A hint is about $`\mathrm{round}(z_{22}/\gamma)`$, and its code (`BitWriter::hint`) takes
/// 2 bits for $`|h|\le1`$ and $`2|h|-1`$ or $`2|h|`$ beyond. Model $`z_{22}/\gamma`$ as $`X+U`$,
/// $`X`$ normal of deviation $`s`$ and $`U`$ the rounding error, uniform on $`(-1/2,1/2]`$,
/// leaving out the challenge terms. Then $`P(h=k)=E[\max(0,1-|X-k|)]`$, and the expected
/// length is $`H(s)=2-3Q(1/s)-2Q(2/s)+3s\varphi(1/s)+s\varphi(2/s)`$, with $`\varphi`$ the
/// standard normal density and $`Q`$ its tail. $`H`$ is nondecreasing and convex, with slope at
/// most $`2\sqrt{2/\pi}<1.6`$, and $`H(s)=2.25`$ at $`s\approx0.989`$ (computed), so the
/// allowance bounds $`H`$ everywhere. It bounds the expectation, not each proof: at
/// $`s=12.45`$, where $`H=19.48`$ and the allowance is 20.62, ten measured proofs had 18.5 to
/// 19.7 bits per hint (`docs/parameters.md`). Only correctly rounded operations, so the
/// parameter tool's mirror computes the same double.
fn hint_bits(sigma2: f64, gamma: u64) -> f64 {
    (1.6 * (sigma2 / gamma as f64) + 0.7).max(2.25)
}

/// $`\lfloor(a\cdot2^{s})\cdot b/c\rfloor`$ with checked 256-bit arithmetic, as `u128`.
fn response_bound(a: u128, shift: u32, b: u128, c: u128) -> Result<u128, Error> {
    let capacity = Error::Parameter("response bound capacity");
    let shifted = U256::from_u128(a)
        .overflowing_shl_vartime(shift)
        .filter(|x| x.shr_vartime(shift) == U256::from_u128(a))
        .ok_or(capacity.clone())?;
    let product =
        Option::<U256>::from(shifted.checked_mul(&U256::from_u128(b))).ok_or(capacity.clone())?;
    let (quotient, _) =
        product.div_rem_vartime(&crypto_bigint::NonZero::new(U256::from_u128(c)).expect("c"));
    int::to_u128(&quotient).ok_or(capacity)
}

impl TboxParams {
    pub(crate) fn transcript_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for text in [&self.id, &self.estimator] {
            out.extend_from_slice(&(text.len() as u64).to_le_bytes());
            out.extend_from_slice(text.as_bytes());
        }
        prime_factor_bytes(&self.prime_factors, &mut out);
        out.extend_from_slice(&(self.l2_bounds_squared.len() as u64).to_le_bytes());
        for value in &self.l2_bounds_squared {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&(self.l2_rows.len() as u64).to_le_bytes());
        for value in &self.l2_rows {
            out.extend_from_slice(&(*value as u64).to_le_bytes());
        }
        for value in [
            self.degree,
            self.m1,
            self.m2,
            self.l,
            self.n_msis,
            self.n_bin,
            self.n_prime,
            self.mlwe_rank,
        ] {
            out.extend_from_slice(&(value as u64).to_le_bytes());
        }
        for value in [
            self.alpha_squared,
            self.linf_bound,
            self.gamma,
            self.mlwe_delta.to_bits(),
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for value in self
            .log_sigma
            .into_iter()
            .chain(std::iter::once(self.d_bits))
        {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }
    /// Recompute dimensions and bounds and reject each violation.
    pub fn check(&self) -> Result<CheckedParams, Error> {
        self.check_with(true)
    }
    /// Test only: every check except the prime-modulus refusal for range blocks, so that the
    /// regression tests can show what that refusal prevents.
    #[cfg(test)]
    pub(crate) fn check_allowing_composite_range_blocks(&self) -> Result<CheckedParams, Error> {
        self.check_with(false)
    }
    fn check_with(&self, refuse_composite_range_blocks: bool) -> Result<CheckedParams, Error> {
        if self.id.is_empty()
            || self.id.len() > 256
            || self.estimator.is_empty()
            || self.estimator.len() > 4096
        {
            return Err(Error::Parameter("parameter provenance"));
        }
        // Factors of 2^64 and more are checked by Baillie-PSW, a probable-prime test.
        if !matches!(self.prime_factors.len(), 1 | 2)
            || self
                .prime_factors
                .iter()
                .any(|p| p.as_words()[0] % 8 != 5 || !int::is_prime_u256(p))
            || self.prime_factors.windows(2).any(|p| p[0] >= p[1])
        {
            return Err(Error::Parameter("proof modulus factors"));
        }
        let q = self.prime_factors.iter().try_fold(U256::ONE, |q, p| {
            Option::<U256>::from(q.checked_mul(p)).ok_or(Error::Overflow)
        })?;
        if !matches!(self.degree, 64 | 128) {
            return Err(Error::Parameter("proof degree"));
        }
        Ring::with_modulus(q, self.degree)?;
        let compression = Compression::with_modulus(q, U256::from_u64(self.gamma), self.d_bits)?;
        let q_minus_one = q.wrapping_sub(&U256::ONE);
        let q_bits = int::bits(&q_minus_one);
        if int::bits(&compression.power2round(&q_minus_one).0) > q_bits - self.d_bits {
            return Err(Error::Parameter("compressed commitment bit width"));
        }
        let z = self.l2_rows.len();
        if z != self.l2_bounds_squared.len()
            || z > 1024
            || self.l2_rows.contains(&0)
            || self.l2_bounds_squared.contains(&0)
        {
            return Err(Error::Parameter("exact norm blocks"));
        }
        if [
            self.m1,
            self.m2,
            self.l,
            self.n_msis,
            self.n_bin,
            self.n_prime,
            self.mlwe_rank,
        ]
        .into_iter()
        .chain(self.l2_rows.iter().copied())
        .any(|x| x > 65535)
            || self.n_msis == 0
            || self.m2 <= self.n_msis
            || self.alpha_squared == 0
        {
            return Err(Error::Parameter("dimensions"));
        }
        if (self.n_prime == 0) != (self.linf_bound == 0) {
            return Err(Error::Parameter("approximate block"));
        }
        if self.log_sigma.iter().any(|t| *t > MAX_LOG_SIGMA) {
            return Err(Error::Parameter("Gaussian width capacity"));
        }
        let q_float = int::to_f64(&q);
        let mut lambda = 2usize;
        let log_q1 = int::to_f64(&self.prime_factors[0]).log2();
        while lambda as f64 * log_q1 < 128.0 {
            lambda += 2;
        }
        let (omega, eta, challenge_bits) = if self.degree == 64 {
            (8, 140, 5)
        } else {
            (2, 59, 3)
        };
        let n_ex = self.n_bin + self.l2_rows.iter().sum::<usize>() + z;
        // The range proofs conclude that each committed sign is +1 or -1 from b^2 = 1 in Z_q,
        // which needs Z_q to be a field (LNP22, proof of Prop. 5.1). Modulo q = q1*q2 the CRT
        // value (1 mod q1, -1 mod q2) also squares to one, and with that sign a long message
        // component b*e' projects like the short e'. Range and norm blocks therefore require
        // a prime modulus.
        if refuse_composite_range_blocks
            && self.prime_factors.len() > 1
            && (n_ex > 0 || self.n_prime > 0)
        {
            return Err(Error::Parameter(
                "binary, exact-norm and range blocks require a prime modulus",
            ));
        }
        let exact_slots = if n_ex > 0 { 256 / self.degree } else { 0 };
        let approx_slots = if self.n_prime > 0 {
            256 / self.degree
        } else {
            0
        };
        let l_ext = exact_slots + approx_slots + 1 + lambda / 2 + 1;
        if (self.m1 + z) * self.degree < 640 || self.m2 * self.degree < 640 {
            return Err(Error::Parameter("completeness dimensions"));
        }
        let reserved = self.n_msis + self.l + l_ext;
        if self.m2.checked_sub(reserved) != Some(self.mlwe_rank) || self.mlwe_rank == 0 {
            return Err(Error::Parameter("simulatability rank"));
        }
        if !self.mlwe_delta.is_finite() || !(1.0..=1.0044).contains(&self.mlwe_delta) {
            return Err(Error::Parameter("MLWE estimate"));
        }
        let [s1, s2, s3, s4] = self.log_sigma.map(|t| 1.55 * (2.0f64).powi(t as i32));
        let nd = (self.n_msis * self.degree) as f64;
        let b = s2 * ((2 * self.m2 * self.degree) as f64).sqrt();
        let b = b
            + eta as f64 * (2.0f64).powi(self.d_bits as i32 - 1) * nd.sqrt()
            + self.gamma as f64 * nd.sqrt() / 2.0;
        if !b.is_finite() || b.powi(2) >= (2.0f64).powi(128) {
            return Err(Error::Parameter("compressed response bound capacity"));
        }
        let b1 = 2.0 * s1 * ((2 * (self.m1 + z) * self.degree) as f64).sqrt();
        let bound = 4.0 * eta as f64 * (b1 * b1 + 4.0 * b * b).sqrt();
        let msis_delta = (2.0f64).powf(bound.log2().powi(2) / (4.0 * nd * q_float.log2()));
        if !msis_delta.is_finite() || msis_delta >= 1.0044 || bound >= q_float {
            return Err(Error::Parameter("MSIS estimate"));
        }
        if (2.0f64).powi(self.d_bits as i32 - 1) * omega as f64 * self.degree as f64
            >= self.gamma as f64
        {
            return Err(Error::Parameter("rounding bound"));
        }
        let exact_alpha_squared = self
            .l2_bounds_squared
            .iter()
            .try_fold(((self.n_bin + z) * self.degree) as u128, |sum, b| {
                sum.checked_add(u128::from(*b))
            })
            .ok_or(Error::Overflow)?;
        let approx_alpha_squared = u128::from(self.linf_bound)
            .checked_mul(u128::from(self.linf_bound))
            .and_then(|x| x.checked_mul((self.n_prime * self.degree) as u128))
            .ok_or(Error::Overflow)?;
        if n_ex > 0 {
            range_rejection_constant(self.log_sigma[2], exact_alpha_squared)?;
        }
        if self.n_prime > 0 {
            range_rejection_constant(self.log_sigma[3], approx_alpha_squared)?;
        }
        let z4_bound = response_bound(124, self.log_sigma[3], 1, 5)?;
        let arp_bound = 2.0 * (256.0f64 / 26.0).sqrt() * 1.64 * s3;
        if n_ex > 0 {
            if q_float < 41.0 * (n_ex * self.degree) as f64 * arp_bound {
                return Err(Error::Parameter("ARP modulus bound"));
            }
            if q_float <= arp_bound.powi(2) + arp_bound * ((self.n_bin * self.degree) as f64).sqrt()
            {
                return Err(Error::Parameter("binary lifting bound"));
            }
            if q_float <= arp_bound.powi(2) + arp_bound * (self.degree as f64).sqrt() {
                return Err(Error::Parameter("slack lifting bound"));
            }
            if self
                .l2_bounds_squared
                .iter()
                .any(|b| q_float <= 3.0 * (*b as f64) + arp_bound.powi(2))
            {
                return Err(Error::Parameter("exact norm lifting bound"));
            }
        }
        let log_q = q_bits;
        let gaussian_bits = |s: f64| (s.log2() + 2.5).ceil();
        let bits = nd * (log_q - self.d_bits) as f64
            + ((self.l + l_ext + lambda / 2) * self.degree) as f64 * log_q as f64
            + (challenge_bits * self.degree) as f64
            + ((self.m1 + z) * self.degree) as f64 * gaussian_bits(s1)
            + ((self.m2 - self.n_msis) * self.degree) as f64 * gaussian_bits(s2)
            + (exact_slots * self.degree) as f64 * gaussian_bits(s3)
            + (approx_slots * self.degree) as f64 * gaussian_bits(s4)
            + hint_bits(s2, self.gamma) * nd;
        Ok(CheckedParams {
            q,
            lambda,
            omega,
            eta,
            n_ex,
            l_ext,
            b_squared: b.powi(2).floor() as u128,
            z1_bound_squared: response_bound(
                961,
                2 * self.log_sigma[0],
                (2 * (self.m1 + z) * self.degree) as u128,
                400,
            )?,
            z3_bound_squared: response_bound(656 * 656 * 961, 2 * self.log_sigma[2], 1, 250000)?,
            z4_bound,
            exact_alpha_squared,
            approx_alpha_squared,
            approx_extraction_bound: z4_bound
                .checked_mul(2)
                .ok_or(Error::Parameter("response bound capacity"))?,
            arp_bound,
            msis_delta,
            estimated_proof_bytes: (bits / 8.0).ceil() as usize,
        })
    }
    /// Strict JSON decoding followed by the complete parameter check.
    #[cfg(feature = "serde")]
    pub fn from_json(json: &str) -> Result<Self, Error> {
        let params: Self = serde_json::from_str(json).map_err(|_| Error::Encoding)?;
        params.check()?;
        Ok(params)
    }
}

/// A small set for tests and examples: a toy statement shape at proof degree 64 with every
/// kind of block (2 binary rows, exact-norm blocks of 2 and 1 rows with squared bounds 128 and
/// 64, an approximate range block of 2 rows with $`\ell_\infty`$ bound 4) and 2 BDLOP messages,
/// over $`q=2^{40}+141`$, the smallest prime from $`2^{40}`$ on that is 5 modulo 8.
///
/// The parameter tool derives it (`tools/params/lnp_params.py`, positional form) from
/// `tools/params/toy-d64.request.json` and the MLWE report `tools/params/toy-d64.report.json`,
/// which `lnp_params.py rank --mlwe-report` writes: rank 26, the smallest that the tool's
/// `delta` policy accepts at this $`q`$ (uSVP block size 361). The MSIS rank keeps the
/// closed-form root Hermite factor below 1.0044. The same set as `src/params/sets/toy-d64.json`.
pub fn toy_d64() -> TboxParams {
    TboxParams {
        id: "jali-toy-d64".into(),
        prime_factors: vec![U256::from_u64(1099511627917)],
        degree: 64,
        m1: 10,
        m2: 56,
        l: 2,
        n_msis: 16,
        alpha_squared: 5760,
        n_bin: 2,
        l2_rows: vec![2, 1],
        l2_bounds_squared: vec![128, 64],
        n_prime: 2,
        linf_bound: 4,
        log_sigma: [16, 12, 10, 11],
        gamma: 65202,
        d_bits: 7,
        mlwe_rank: 26,
        mlwe_delta: 1.0042736680871425,
        estimator: concat!(
            "jali-params 3 rank; policy delta; ",
            "MLWE uSVP (ADPS16) beta 361 delta 1.0042736680871425"
        )
        .into(),
    }
}

/// The Kyber-1024 key relation at proof degree 64, with parameters from the parameter tool.
///
/// The relation $`Aw+t=0`$ over $`\mathbb Z_{3329}[X]/(X^{256}+1)`$, with $`A`$ of size
/// $`4\times8`$ and $`\|w\|^2\le2950=\lceil(1.2\sqrt{2048})^2\rceil`$, compiled by
/// [`crate::lin::compile`] with one block of 8 polynomials in the Ajtai part. The tool sized the
/// set for the worst case over public data, so it compiles every instance of that shape. Its
/// hardness criterion is the tool's `delta` policy (root Hermite factors of at most 1.0044 for
/// MLWE and below 1.0044 for MSIS), about 101 bits under core-SVP, not 128;
/// `docs/parameters.md` gives the estimates. The same set as
/// `src/params/sets/kyber1024-d64.json`, which `tools/params/lnp_params.py regenerate` writes
/// from `tools/params/sets/kyber1024-d64.request.json` next to the tool's own copy.
pub fn kyber1024_d64() -> TboxParams {
    TboxParams {
        id: "jali-kyber1024-d64".into(),
        prime_factors: vec![U256::from_u64(2423991946973)],
        degree: 64,
        m1: 32,
        m2: 55,
        l: 0,
        n_msis: 17,
        alpha_squared: 2950,
        n_bin: 0,
        l2_rows: vec![32],
        l2_bounds_squared: vec![2950],
        n_prime: 16,
        linf_bound: 3476,
        log_sigma: [16, 12, 12, 23],
        gamma: 262142,
        d_bits: 9,
        mlwe_rank: 26,
        mlwe_delta: 1.004381952541052,
        estimator: concat!(
            "jali-params 3; ",
            "request sha256 bb726fe1272df917; ",
            "policy delta; ",
            "MLWE uSVP (ADPS16) beta 348 delta 1.004381952541052; ",
            "MSIS closed form beta 360 delta 1.004282058",
        )
        .into(),
    }
}

/// The Kyber-1024 key relation at proof degree 128, with parameters from the parameter tool.
///
/// The relation of [`kyber1024_d64`], with the statement ring's degree 256 lowered to 128
/// instead of 64 (challenges with $`\omega=2`$, $`\eta=59`$). The same hardness criterion and
/// provenance: `src/params/sets/kyber1024-d128.json`, from
/// `tools/params/sets/kyber1024-d128.request.json`.
pub fn kyber1024_d128() -> TboxParams {
    TboxParams {
        id: "jali-kyber1024-d128".into(),
        prime_factors: vec![U256::from_u64(2423989135261)],
        degree: 128,
        m1: 16,
        m2: 29,
        l: 0,
        n_msis: 8,
        alpha_squared: 2950,
        n_bin: 0,
        l2_rows: vec![16],
        l2_bounds_squared: vec![2950],
        n_prime: 8,
        linf_bound: 3476,
        log_sigma: [15, 11, 12, 23],
        gamma: 524286,
        d_bits: 11,
        mlwe_rank: 13,
        mlwe_delta: 1.004381952541052,
        estimator: concat!(
            "jali-params 3; ",
            "request sha256 11362de993f93130; ",
            "policy delta; ",
            "MLWE uSVP (ADPS16) beta 348 delta 1.004381952541052; ",
            "MSIS closed form beta 359 delta 1.004297526",
        )
        .into(),
    }
}

/// A Module-LWE relation at proof degree 64, with parameters from the parameter tool.
///
/// The relation $`As+t=0`$ over $`\mathbb Z_p[X]/(X^{256}+1)`$ with $`p=2^{32}-4607`$, the
/// largest prime below $`2^{32}`$ that is 1 modulo 512, $`A`$ of size $`4\times8`$ and
/// $`\|s\|^2\le2048`$, compiled by [`crate::lin::compile`] with one block of 8 polynomials in
/// the Ajtai part, sized for the worst case over public data. Its request keeps
/// $`q<2^{64}`$. The same hardness criterion as [`kyber1024_d64`] and the same provenance:
/// `src/params/sets/demo.json`, from `tools/params/sets/demo.request.json`.
pub fn demo_d64() -> TboxParams {
    TboxParams {
        id: "jali-demo-d64".into(),
        prime_factors: vec![U256::from_u64(1563674726761101613)],
        degree: 64,
        m1: 32,
        m2: 61,
        l: 0,
        n_msis: 11,
        alpha_squared: 2048,
        n_bin: 0,
        l2_rows: vec![32],
        l2_bounds_squared: vec![2048],
        n_prime: 16,
        linf_bound: 2897,
        log_sigma: [16, 12, 11, 22],
        gamma: 65534,
        d_bits: 7,
        mlwe_rank: 38,
        mlwe_delta: 1.004373402767646,
        estimator: concat!(
            "jali-params 3; ",
            "request sha256 9c3d22b74cce1089; ",
            "policy delta; ",
            "MLWE uSVP (ADPS16) beta 349 delta 1.004373402767646; ",
            "MSIS closed form beta 347 delta 1.004398692",
        )
        .into(),
    }
}
