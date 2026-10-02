//! Checked integer operations for norms and modular arithmetic.
use crate::Error;
use crypto_bigint::{CheckedAdd, Limb, NonZero, U64, U128, U256, Uint};

/// Reduce to $`(-q/2,q/2]`$. The modulus must be positive.
pub fn center(x: i128, q: i128) -> Result<i128, Error> {
    if q <= 0 {
        return Err(Error::Parameter("modulus"));
    }
    let r = x.rem_euclid(q);
    Ok(if r > q / 2 { r - q } else { r })
}

/// Exact sum of squared integer coefficients, rejecting overflow.
pub fn squared_norm(values: &[i128]) -> Result<U256, Error> {
    values.iter().try_fold(U256::ZERO, |acc, x| {
        let x = U256::from(x.unsigned_abs());
        let square = x.wrapping_mul(&x);
        Option::from(acc.checked_add(&square)).ok_or(Error::Overflow)
    })
}

/// Product modulo a modulus $`q\ge2`$, centred in $`(-q/2,q/2]`$. The 256-bit product is exact
/// for every `i128` modulus.
pub fn mul_mod(a: i128, b: i128, q: i128) -> Result<i128, Error> {
    if q < 2 {
        return Err(Error::Parameter("modulus"));
    }
    let a = U256::from(a.rem_euclid(q) as u128);
    let b = U256::from(b.rem_euclid(q) as u128);
    let modulus = NonZero::new(U256::from(q as u128)).expect("positive modulus");
    center(low_u128(&a.mul_mod_vartime(&b, &modulus)) as i128, q)
}

/// Deterministic Miller–Rabin primality test for all 64-bit integers.
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n.is_multiple_of(p) {
            return n == p;
        }
    }
    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;
    'witness: for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022] {
        if a % n == 0 {
            continue;
        }
        let mut x = super::ntt::pow_mod(a % n, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = super::ntt::mul_mod(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// Primality below $`2^{256}`$: [`is_prime`] below $`2^{64}`$, where it is a proof, and the
/// Baillie–PSW test from $`2^{64}`$ on.
///
/// From $`2^{64}`$ on this runs `crypto_primes::is_prime`: a Miller–Rabin test to base 2 and
/// the strengthened Lucas test of Baillie, Fiori and Wagstaff (Math. Comp. 90, 2021). That is
/// a probable-prime test. No composite is known to pass it, but it is not a proof of
/// primality.
pub fn is_prime_u256(n: &U256) -> bool {
    match to_u64(n) {
        Some(small) => is_prime(small),
        None => crypto_primes::is_prime(crypto_primes::Flavor::Any, n),
    }
}

// A limb of `crypto-bigint` has 64 bits on 64-bit targets, wasm32 and ARMv7, and 32 bits on
// other 32-bit targets such as i686 (its `cpubits!` rule). The helpers below hold for both.

/// The low 64 bits.
pub(crate) fn low_u64<const L: usize>(x: &Uint<L>) -> u64 {
    u64::from(x.resize::<{ U64::LIMBS }>())
}

/// The low 128 bits.
pub(crate) fn low_u128<const L: usize>(x: &Uint<L>) -> u128 {
    u128::from(x.resize::<{ U128::LIMBS }>())
}

/// The value as `u64`, if it fits.
pub(crate) fn to_u64(x: &U256) -> Option<u64> {
    let high = &x.as_words()[U64::LIMBS..];
    high.iter().all(|w| *w == 0).then(|| low_u64(x))
}

/// The value as `u128`, if it fits.
pub(crate) fn to_u128(x: &U256) -> Option<u128> {
    let high = &x.as_words()[U128::LIMBS..];
    high.iter().all(|w| *w == 0).then(|| low_u128(x))
}

/// The value as one limb, if it fits.
pub(crate) fn to_limb(x: &U256) -> Option<Limb> {
    let limbs = x.as_limbs();
    limbs[1..].iter().all(|l| l.0 == 0).then_some(limbs[0])
}

/// Remainders modulo a nonzero `u64`: by a one-limb reciprocal where limbs have 64 bits, and
/// by `crypto-bigint`'s division by a two-limb divisor where they have 32.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Divisor64 {
    value: u64,
    inner: DivisorInner,
}
crypto_bigint::cpubits! {
    32 => {
        type DivisorInner = NonZero<U64>;
        fn divisor_inner(value: u64) -> DivisorInner {
            NonZero::new(U64::from_u64(value)).expect("nonzero divisor")
        }
        fn rem_inner<const L: usize>(x: &Uint<L>, divisor: &DivisorInner) -> u64 {
            u64::from(x.rem_vartime(divisor))
        }
    }
    64 => {
        type DivisorInner = crypto_bigint::Reciprocal;
        fn divisor_inner(value: u64) -> DivisorInner {
            DivisorInner::new(NonZero::new(Limb(value)).expect("nonzero divisor"))
        }
        fn rem_inner<const L: usize>(x: &Uint<L>, divisor: &DivisorInner) -> u64 {
            x.rem_limb_with_reciprocal(divisor).0
        }
    }
}
impl Divisor64 {
    /// A divisor. Panics on zero.
    pub(crate) fn new(value: u64) -> Self {
        Self {
            value,
            inner: divisor_inner(value),
        }
    }
    /// The divisor.
    pub(crate) fn value(&self) -> u64 {
        self.value
    }
    /// $`x\bmod d`$.
    pub(crate) fn rem<const L: usize>(&self, x: &Uint<L>) -> u64 {
        rem_inner(x, &self.inner)
    }
}

/// Number of significant bits; zero for zero.
pub(crate) fn bits(x: &U256) -> u32 {
    x.bits_vartime()
}

/// $`2^k`$, or `None` from $`k=256`$ on.
pub(crate) fn power_of_two(k: u32) -> Option<U256> {
    U256::ONE.overflowing_shl_vartime(k)
}

/// The value as the nearest `f64`, rounded as an integer-to-float cast rounds.
pub(crate) fn to_f64(x: &U256) -> f64 {
    let n = bits(x);
    if n <= 64 {
        return low_u64(x) as f64;
    }
    // The top 64 bits, with a sticky bit for everything below them, round exactly as the full
    // value does: 64 bits leave 11 guard bits beyond the 53 of the mantissa.
    let shift = n - 64;
    let top = low_u64(&x.shr_vartime(shift));
    let sticky = u64::from(x.shl_vartime(256 - shift) != U256::ZERO);
    (top | sticky) as f64 * (2.0f64).powi(shift as i32)
}
