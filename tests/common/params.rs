//! A test-only Rust port of `derive` in `tools/params/lnp_params.py`, so that a test can fit
//! checked parameters to any statement shape in-process, also without the `serde` feature. It
//! computes in `f64` where the tool uses 300-bit `mpmath`; `TboxParams::check` revalidates
//! every result, and `tests/params_crosscheck.rs` compares it with the tool's checked-in
//! outputs. Like the tool, it fails when $`q-1`$ has no divisor in the compression window; it
//! does not fall back to another divisor. The MLWE rank and delta it sets are synthetic
//! metadata, not an estimate.
use jali::{
    math::{U256, int::is_prime},
    params::TboxParams,
    statement::{LinfLifting, Requirements},
};

/// Provenance of every fitted set.
pub const SYNTHETIC: &str = "test-only synthetic MLWE metadata; not an estimate";
/// The MLWE rank of `toy_d64`, reused as synthetic metadata.
pub const RANK: usize = 26;

/// The tool's `rounded`: the exponent $`t`$ whose width $`1.55\cdot2^t`$ is nearest `value`.
fn rounded(value: f64) -> u32 {
    if value == 0.0 {
        return 0;
    }
    let t = (value / 1.55).log2().floor().max(0.0) as i32;
    let below = (1.55 * 2f64.powi(t) - value).abs();
    let above = (1.55 * 2f64.powi(t + 1) - value).abs();
    if below <= above {
        t as u32
    } else {
        t as u32 + 1
    }
}

/// `log_sigma[3]` as the tool derives it: from the Euclidean bound
/// $`\sqrt{n'd}\cdot`$`linf_bound` of the approximate-range vector.
pub fn range_width(req: &Requirements, degree: usize) -> u32 {
    rounded(5.0 * 337f64.sqrt() * ((req.n_prime * degree) as f64).sqrt() * req.linf_bound as f64)
}

/// `log_sigma[3]` by the tool's per-slot rule when the requirements carry
/// `approx_alpha_squared` $`=\alpha_{slot}^2`$: the larger of
/// `rounded`$`(5\sqrt{337}\,\alpha_{slot})`$ and the guard, the smallest $`t`$ whose rejection
/// constant for the default bound $`n'd\beta_\infty^2`$ is 2 (`params::range_rejection_constant`,
/// as the prover computes it). Without the field, `range_width`.
pub fn per_slot_width(req: &Requirements, degree: usize) -> u32 {
    let Some(alpha_squared) = req.approx_alpha_squared else {
        return range_width(req, degree);
    };
    per_slot_rule(req, degree, alpha_squared).0
}

/// (`log_sigma[3]`, the per-slot width, the guard) of the per-slot rule for
/// $`\alpha_{slot}^2`$ = `alpha_squared`.
pub fn per_slot_rule(req: &Requirements, degree: usize, alpha_squared: u128) -> (u32, u32, u32) {
    let slot = rounded(5.0 * 337f64.sqrt() * (alpha_squared as f64).sqrt());
    let default = req.linf_bound * req.linf_bound * (req.n_prime * degree) as u128;
    let guard = (0..=jali::rand::MAX_LOG_SIGMA)
        .find(|t| jali::params::range_rejection_constant(*t, default) == Ok(2))
        .expect("a width with rejection constant 2");
    (slot.max(guard), slot, guard)
}

/// `fit` with `log_sigma[3]` from `per_slot_width`. The width enters no other choice of `fit`
/// (MSIS, $`\gamma`$ and $`D`$ depend on `log_sigma[0..2]` only), so this is `fit` with that
/// one field replaced, checked again.
pub fn fit_per_slot(
    req: &Requirements,
    id: &str,
    factors: Vec<u64>,
    degree: usize,
    mlwe_rank: usize,
) -> Result<TboxParams, String> {
    let mut params = fit(req, id, factors, degree, mlwe_rank)?;
    params.log_sigma[3] = per_slot_width(req, degree);
    params
        .check()
        .map_err(|e| format!("fitted set fails the check: {e}"))?;
    Ok(params)
}

/// All divisors of `n`, by trial division.
fn divisors(n: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    let mut m = n;
    let mut p = 2u64;
    while p * p <= m {
        while m.is_multiple_of(p) {
            factors.push(p);
            m /= p;
        }
        p += if p == 2 { 1 } else { 2 };
    }
    if m > 1 {
        factors.push(m);
    }
    let mut out = vec![1u64];
    let mut i = 0;
    while i < factors.len() {
        let p = factors[i];
        let mut e = 0;
        while i < factors.len() && factors[i] == p {
            e += 1;
            i += 1;
        }
        let current = out.clone();
        let mut power = 1u64;
        for _ in 0..e {
            power *= p;
            out.extend(current.iter().map(|x| x * power));
        }
    }
    out
}

/// Port of `derive(request, report)` with a report of rank `mlwe_rank` and delta 1.0044 for
/// this ring. Errors carry the tool's messages.
pub fn fit(
    req: &Requirements,
    id: &str,
    factors: Vec<u64>,
    degree: usize,
    mlwe_rank: usize,
) -> Result<TboxParams, String> {
    let d = degree;
    if !matches!(d, 64 | 128)
        || !matches!(factors.len(), 1 | 2)
        || factors.iter().any(|p| p % 8 != 5 || !is_prime(*p))
    {
        return Err("invalid proof ring".into());
    }
    if factors.windows(2).any(|p| p[0] >= p[1]) {
        return Err("prime factors must be strictly ascending".into());
    }
    let q_int: u128 = factors.iter().map(|p| u128::from(*p)).product();
    let q = q_int as f64;
    let (omega, eta) = if d == 64 { (8.0, 140.0) } else { (2.0, 59.0) };
    let mut lambda = 2usize;
    while lambda as f64 * (factors[0] as f64).log2() < 128.0 {
        lambda += 2;
    }
    let z = req.l2_rows.len();
    if req.l2_bounds_squared.len() != z {
        return Err("mismatched exact blocks".into());
    }
    let n_ex = req.l2_rows.iter().sum::<usize>() + req.n_bin + z;
    if factors.len() > 1 && (n_ex > 0 || req.n_prime > 0) {
        return Err("binary, exact-norm and range blocks require a prime modulus".into());
    }
    let e_slots = if n_ex > 0 { 256 / d } else { 0 };
    let d_slots = if req.n_prime > 0 { 256 / d } else { 0 };
    let l_ext = e_slots + d_slots + 1 + lambda / 2 + 1;
    let l2_sum: f64 = req.l2_bounds_squared.iter().map(|x| *x as f64).sum();
    let mut logs = [
        rounded(14.0 * eta * (req.alpha_squared as f64 + (z * d) as f64).sqrt()),
        0,
        rounded(5.0 * 337f64.sqrt() * (l2_sum + ((req.n_bin + z) * d) as f64).sqrt()),
        range_width(req, d),
    ];
    let bound = |logs: &[u32; 4], n: usize, m2: usize, d_bits: i32, gamma: f64| {
        let s1 = 1.55 * 2f64.powi(logs[0] as i32);
        let s2 = 1.55 * 2f64.powi(logs[1] as i32);
        let nd = (n * d) as f64;
        let b = s2 * ((2 * m2 * d) as f64).sqrt()
            + eta * 2f64.powi(d_bits - 1) * nd.sqrt()
            + gamma * nd.sqrt() / 2.0;
        let b1 = 2.0 * s1 * ((2 * (req.m1 + z) * d) as f64).sqrt();
        4.0 * eta * (b1 * b1 + 4.0 * b * b).sqrt()
    };
    let hard = |logs: &[u32; 4], n: usize, m2: usize, d_bits: i32, gamma: f64| {
        let b = bound(logs, n, m2, d_bits, gamma);
        let delta = 2f64.powf(b.log2().powi(2) / (4.0 * (n * d) as f64 * q.log2()));
        delta < 1.0044 && b < q
    };
    let mut n = 1;
    let m2 = loop {
        let m2 = mlwe_rank + n + req.l + l_ext;
        logs[1] = rounded(eta * ((m2 * d) as f64).sqrt());
        if hard(&logs, n, m2, 0, 0.0) {
            break m2;
        }
        n += 1;
        if n > 65535 {
            return Err("MSIS search exhausted".into());
        }
    };
    let bits = 128 - q_int.leading_zeros();
    let mut gamma = 1u128 << (bits - 1);
    while !hard(&logs, n, m2, 0, gamma as f64) {
        gamma /= 2;
    }
    let q_minus_one = u64::try_from(q_int - 1).map_err(|_| "q - 1 exceeds 64 bits")?;
    let gamma = divisors(q_minus_one)
        .into_iter()
        .map(u128::from)
        .filter(|x| x % 2 == 0 && 4 * gamma < 5 * x && 5 * x <= 5 * gamma)
        .max()
        .ok_or("no suitable divisor; choose another prime and rerun the estimator")?;
    let mut d_bits = bits as i32 - 1;
    while !(hard(&logs, n, m2, d_bits, gamma as f64)
        && 2f64.powi(d_bits - 1) * omega * (d as f64) < gamma as f64)
    {
        d_bits -= 1;
        if d_bits < 0 {
            return Err("no compression exponent".into());
        }
    }
    let params = TboxParams {
        id: id.into(),
        prime_factors: factors.iter().map(|p| U256::from_u64(*p)).collect(),
        degree,
        m1: req.m1,
        m2,
        l: req.l,
        n_msis: n,
        alpha_squared: u64::try_from(req.alpha_squared).map_err(|_| "alpha_squared")?,
        n_bin: req.n_bin,
        l2_rows: req.l2_rows.clone(),
        l2_bounds_squared: req.l2_bounds_squared.clone(),
        n_prime: req.n_prime,
        linf_bound: u64::try_from(req.linf_bound).map_err(|_| "linf_bound")?,
        log_sigma: logs,
        gamma: gamma as u64,
        d_bits: d_bits as u32,
        mlwe_rank,
        mlwe_delta: 1.0044,
        estimator: SYNTHETIC.into(),
    };
    params
        .check()
        .map_err(|e| format!("fitted set fails the check: {e}"))?;
    Ok(params)
}

/// The smallest prime $`p\ge`$ `lower` with $`p\equiv5\pmod 8`$.
pub fn next_prime_5_mod_8(lower: u64) -> u64 {
    let mut p = lower + (13 - lower % 8) % 8;
    while !is_prime(p) {
        p += 8;
    }
    p
}

/// The largest prime $`p\le`$ `upper` with $`p\equiv5\pmod 8`$.
pub fn prev_prime_5_mod_8(upper: u64) -> u64 {
    let mut p = upper - (upper % 8 + 3) % 8;
    while !is_prime(p) {
        p -= 8;
    }
    p
}

/// Fit at the first prime $`q\equiv5\pmod 8`$ at or above `lower` whose $`q-1`$ has a
/// divisor in the compression window: a simple stand-in for the prime search of the tool's
/// `derive` subcommand, which the positional form ported here lacks.
pub fn fit_upward(req: &Requirements, id: &str, degree: usize, lower: u64) -> (u64, TboxParams) {
    let mut q = next_prime_5_mod_8(lower);
    for _ in 0..200 {
        match fit(req, id, vec![q], degree, RANK) {
            Ok(params) => return (q, params),
            Err(_) => q = next_prime_5_mod_8(q + 1),
        }
    }
    panic!("no prime above {lower} fits {id}");
}

/// Fit at the last prime $`q\equiv5\pmod 8`$ at or below `upper` that fits, as `fit_upward`.
pub fn fit_downward(req: &Requirements, id: &str, degree: usize, upper: u64) -> (u64, TboxParams) {
    let mut q = prev_prime_5_mod_8(upper);
    for _ in 0..200 {
        match fit(req, id, vec![q], degree, RANK) {
            Ok(params) => return (q, params),
            Err(_) => q = prev_prime_5_mod_8(q - 1),
        }
    }
    panic!("no prime below {upper} fits {id}");
}

/// The value that `statement::compile` requires the proof modulus to exceed, recomputed
/// independently from the requirements, the statement modulus $`p`$ and `log_sigma[3]`:
/// $`2(F+p\,\beta_\infty\lceil 28\cdot1.55\cdot2^{t_4}/\beta_\infty\rceil)`$, or with
/// per-constraint moduli the largest $`2(F_j+p_j\,\beta_\infty\lceil\dots\rceil)`$ over
/// `lifted_moduli`, where the statement modulus plays no part. With $`\ell_\infty`$ variables,
/// also each constraint of `linf.lifting` with $`F_j(E)`$ at the extraction bound $`E`$ of
/// `log_sigma[3]`. Panics on a value of $`2^{128}`$ or more.
pub fn lifting_threshold(req: &Requirements, statement_modulus: u128, log_sigma3: u32) -> u128 {
    let psi = (28.0 * 1.55 * 2f64.powi(log_sigma3 as i32) / req.linf_bound as f64).ceil() as u128;
    let threshold = |f: u128, p: u128| {
        p.checked_mul(req.linf_bound * psi)
            .and_then(|x| x.checked_add(f))
            .and_then(|x| x.checked_mul(2))
            .expect("lifting threshold below 2^128")
    };
    let honest = if req.lifted_moduli.is_empty() {
        threshold(
            super::narrow(&req.max_integer_coefficient),
            statement_modulus,
        )
    } else {
        req.lifted_moduli
            .iter()
            .map(|m| {
                threshold(
                    super::narrow(&m.max_integer_coefficient),
                    super::narrow(&m.modulus),
                )
            })
            .max()
            .unwrap()
    };
    let Some(linf) = &req.linf else {
        return honest;
    };
    let e = extraction_bound(log_sigma3);
    linf.lifting
        .iter()
        .map(|entry| threshold(linf_f(entry, e), super::narrow(&entry.modulus)))
        .fold(honest, u128::max)
}

/// `approx_extraction_bound` for `log_sigma[3]` $`=t`$: $`2\lfloor124\cdot2^t/5\rfloor`$.
pub fn extraction_bound(t: u32) -> u128 {
    2 * ((124u128 << t) / 5)
}

/// $`F_j(E)=`$`exact`$`+`$`linear`$`\cdot E+`$`quadratic`$`\cdot E^2`$; panics from
/// $`2^{128}`$ on.
pub fn linf_f(entry: &LinfLifting, e: u128) -> u128 {
    let [exact, linear, quadratic] =
        [&entry.exact, &entry.linear, &entry.quadratic].map(super::narrow);
    quadratic
        .checked_mul(e)
        .and_then(|x| x.checked_mul(e))
        .and_then(|x| x.checked_add(linear.checked_mul(e)?))
        .and_then(|x| x.checked_add(exact))
        .expect("F_j(E) below 2^128")
}

/// Factorizations of $`q-1`$ from Sage, for $`q=2^{240}+325`$ and $`q=2^{128}+165`$, the
/// smallest primes $`\equiv5\pmod8`$ from $`2^{240}`$ and $`2^{128}`$ on, and $`q=2^{256}-435`$,
/// the largest such prime below $`2^{256}`$. `even_divisors` checks each by multiplication.
pub const FACTORS_240: [(&str, u32); 9] = [
    ("2", 2),
    ("5", 2),
    ("13", 1),
    ("2393", 1),
    ("2593", 1),
    ("116257153", 1),
    ("20599429185833", 1),
    ("399968445554189", 1),
    ("228670316016583664287966193", 1),
];
pub const FACTORS_128: [(&str, u32); 7] = [
    ("2", 2),
    ("3", 1),
    ("5", 1),
    ("7", 1),
    ("79", 1),
    ("2273", 1),
    ("4511943239662745109643046187983", 1),
];
pub const FACTORS_256: [(&str, u32); 6] = [
    ("2", 2),
    ("3", 1),
    ("5", 3),
    ("7", 1),
    ("31", 1),
    (
        "355736065245211045848144347184909087106820229387528614560545572988980429",
        1,
    ),
];

/// $`2^k+c`$.
pub fn power_plus(k: u32, c: u64) -> U256 {
    U256::ONE.shl_vartime(k).wrapping_add(&U256::from_u64(c))
}
/// $`2^k-c`$.
pub fn power_minus(k: u32, c: u64) -> U256 {
    match k {
        256 => U256::MAX.wrapping_sub(&U256::from_u64(c - 1)),
        _ => U256::ONE.shl_vartime(k).wrapping_sub(&U256::from_u64(c)),
    }
}

/// The even divisors below $`2^{64}`$ of $`q-1`$, from its factorization (primes and exponents,
/// from Sage), after checking that the factorization multiplies to $`q-1`$.
pub fn even_divisors(q: &U256, factors: &[(&str, u32)]) -> Vec<u64> {
    let parse = |t: &str| {
        let mut x = U256::ZERO;
        for b in t.bytes() {
            x = x
                .wrapping_mul(&U256::from_u8(10))
                .wrapping_add(&U256::from_u8(b - b'0'));
        }
        x
    };
    let product = factors.iter().fold(U256::ONE, |acc, (p, e)| {
        (0..*e).fold(acc, |acc, _| acc.wrapping_mul(&parse(p)))
    });
    assert_eq!(
        product,
        q.wrapping_sub(&U256::ONE),
        "factorization of q - 1"
    );
    let mut divisors = vec![1u128];
    for (p, e) in factors {
        let Ok(p) = p.parse::<u128>() else { continue };
        let current = divisors.clone();
        let mut power = 1u128;
        for _ in 0..*e {
            power = power.saturating_mul(p);
            divisors.extend(current.iter().map(|d| d.saturating_mul(power)));
        }
    }
    let mut out: Vec<u64> = divisors
        .into_iter()
        .filter(|d| d % 2 == 0 && *d < 1 << 64)
        .map(|d| d as u64)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// The value as a float, for parameter fitting only: 64-bit words, most significant first.
pub fn float(x: &U256) -> f64 {
    let bytes = x.to_le_bytes();
    let (words, _) = bytes.as_chunks::<8>();
    words.iter().rev().fold(0.0, |acc, w| {
        acc * 2f64.powi(64) + u64::from_le_bytes(*w) as f64
    })
}

/// `fit` for a prime modulus of any size up to $`2^{256}`$, with its even divisors of $`q-1`$
/// supplied (found with Sage; the port cannot factor $`q-1`$ at these sizes). It follows
/// `fit`, but keeps $`b^2<2^{128}`$ while lowering the compression target, takes the largest
/// supplied divisor at or below the target, not one in the tool's window, and lowers $`D`$
/// while the check refuses the compressed commitment bit width. For test-only parameters;
/// `TboxParams::check` revalidates the result.
pub fn fit_wide(
    req: &Requirements,
    id: &str,
    q: U256,
    divisors: &[u64],
    degree: usize,
) -> Result<TboxParams, String> {
    let d = degree;
    let qf = float(&q);
    let (omega, eta) = if d == 64 { (8.0, 140.0) } else { (2.0, 59.0) };
    let mut lambda = 2usize;
    while lambda as f64 * qf.log2() < 128.0 {
        lambda += 2;
    }
    let z = req.l2_rows.len();
    let n_ex = req.l2_rows.iter().sum::<usize>() + req.n_bin + z;
    let e_slots = if n_ex > 0 { 256 / d } else { 0 };
    let d_slots = if req.n_prime > 0 { 256 / d } else { 0 };
    let l_ext = e_slots + d_slots + 1 + lambda / 2 + 1;
    let l2_sum: f64 = req.l2_bounds_squared.iter().map(|x| *x as f64).sum();
    let mut logs = [
        rounded(14.0 * eta * (req.alpha_squared as f64 + (z * d) as f64).sqrt()),
        0,
        rounded(5.0 * 337f64.sqrt() * (l2_sum + ((req.n_bin + z) * d) as f64).sqrt()),
        range_width(req, d),
    ];
    let b = |logs: &[u32; 4], n: usize, m2: usize, d_bits: i32, gamma: f64| {
        let s2 = 1.55 * 2f64.powi(logs[1] as i32);
        let nd = (n * d) as f64;
        s2 * ((2 * m2 * d) as f64).sqrt()
            + eta * 2f64.powi(d_bits - 1) * nd.sqrt()
            + gamma * nd.sqrt() / 2.0
    };
    let hard = |logs: &[u32; 4], n: usize, m2: usize, d_bits: i32, gamma: f64| {
        let b = b(logs, n, m2, d_bits, gamma);
        let s1 = 1.55 * 2f64.powi(logs[0] as i32);
        let b1 = 2.0 * s1 * ((2 * (req.m1 + z) * d) as f64).sqrt();
        let bound = 4.0 * eta * (b1 * b1 + 4.0 * b * b).sqrt();
        let delta = 2f64.powf(bound.log2().powi(2) / (4.0 * (n * d) as f64 * qf.log2()));
        delta < 1.0044 && bound < qf && b * b < 2f64.powi(128)
    };
    let mut n = 1;
    let m2 = loop {
        let m2 = RANK + n + req.l + l_ext;
        logs[1] = rounded(eta * ((m2 * d) as f64).sqrt());
        if hard(&logs, n, m2, 0, 0.0) {
            break m2;
        }
        n += 1;
        if n > 65535 {
            return Err("MSIS search exhausted".into());
        }
    };
    let mut target = 2f64.powi(63);
    while !hard(&logs, n, m2, 0, target) {
        target /= 2.0;
    }
    let gamma = divisors
        .iter()
        .copied()
        .filter(|x| x % 2 == 0 && (*x as f64) <= target)
        .max()
        .ok_or("no divisor at or below the target")?;
    let mut d_bits = 62i32;
    while !(hard(&logs, n, m2, d_bits, gamma as f64)
        && 2f64.powi(d_bits - 1) * omega * (d as f64) < gamma as f64)
    {
        d_bits -= 1;
        if d_bits < 0 {
            return Err("no compression exponent".into());
        }
    }
    let mut params = TboxParams {
        id: id.into(),
        prime_factors: vec![q],
        degree,
        m1: req.m1,
        m2,
        l: req.l,
        n_msis: n,
        alpha_squared: u64::try_from(req.alpha_squared).map_err(|_| "alpha_squared")?,
        n_bin: req.n_bin,
        l2_rows: req.l2_rows.clone(),
        l2_bounds_squared: req.l2_bounds_squared.clone(),
        n_prime: req.n_prime,
        linf_bound: u64::try_from(req.linf_bound).map_err(|_| "linf_bound")?,
        log_sigma: logs,
        gamma,
        d_bits: d_bits as u32,
        mlwe_rank: RANK,
        mlwe_delta: 1.0044,
        estimator: SYNTHETIC.into(),
    };
    // Just below a power of two, the high part of q - 1 rounds up to one more bit than
    // bits(q - 1) - D allows. A smaller D fits it, and only relaxes the conditions above.
    let width = Err(jali::Error::Parameter("compressed commitment bit width"));
    while params.check() == width && params.d_bits > 0 {
        params.d_bits -= 1;
    }
    params
        .check()
        .map_err(|e| format!("fitted set fails the check: {e}"))?;
    Ok(params)
}
