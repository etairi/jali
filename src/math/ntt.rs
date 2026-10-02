//! Portable radix-two negacyclic NTT over primes below $`2^{62}`$.
//!
//! A plan precomputes every twiddle of every stage, the twists and the bit-reversal
//! permutation, each multiplier with its Shoup constant $`\lfloor w2^{64}/p\rfloor`$, so that a
//! transform needs no division: products by constants are Shoup products and sums reduce by one
//! conditional subtraction. [`NttPlan::forward`] and [`NttPlan::inverse`] work on standard
//! residues and return exactly what a transform with remainders returns. The crate's RNS
//! products use the Montgomery form instead: values times $`R=2^{64}`$, put in by the twist of
//! the forward transform, kept by the Montgomery products of pointwise multiplication and by
//! sums, and taken out by the untwist of the inverse transform.
use crate::{Error, math::int::is_prime};

pub(crate) fn mul_mod(a: u64, b: u64, p: u64) -> u64 {
    (u128::from(a) * u128::from(b) % u128::from(p)) as u64
}
pub(crate) fn pow_mod(mut a: u64, mut e: u64, p: u64) -> u64 {
    let mut r = 1;
    while e != 0 {
        if e & 1 != 0 {
            r = mul_mod(r, a, p);
        }
        a = mul_mod(a, a, p);
        e >>= 1;
    }
    r
}

/// The Shoup constant $`\lfloor w2^{64}/p\rfloor`$ of a multiplier $`w<p`$.
pub(crate) fn shoup(w: u64, p: u64) -> u64 {
    ((u128::from(w) << 64) / u128::from(p)) as u64
}

/// $`xw\bmod p`$ in $`[0,p)`$ for any $`x<2^{64}`$, given $`w<p<2^{63}`$ and `w_shoup` $`=`$
/// [`shoup`]`(w, p)`.
///
/// Write $`w'=\lfloor w2^{64}/p\rfloor=(w2^{64}-e)/p`$ with $`0\le e<p`$. The quotient estimate
/// $`k=\lfloor xw'/2^{64}\rfloor`$ satisfies $`0\le xw/p-k<xe/(p2^{64})+1<2`$, so
/// $`xw-kp\in[0,2p)`$. As $`2p<2^{64}`$, the wrapping computation below is exact, and one
/// conditional subtraction reduces it.
#[inline(always)]
pub(crate) fn shoup_mul(x: u64, w: u64, w_shoup: u64, p: u64) -> u64 {
    let k = ((u128::from(x) * u128::from(w_shoup)) >> 64) as u64;
    let r = x.wrapping_mul(w).wrapping_sub(k.wrapping_mul(p));
    if r >= p { r - p } else { r }
}

/// $`(a+b)\bmod p`$ for $`a,b<p<2^{63}`$.
#[inline(always)]
pub(crate) fn add_mod(a: u64, b: u64, p: u64) -> u64 {
    let s = a + b;
    if s >= p { s - p } else { s }
}

/// $`(a-b)\bmod p`$ for $`a,b<p`$.
#[inline(always)]
pub(crate) fn sub_mod(a: u64, b: u64, p: u64) -> u64 {
    if a >= b { a - b } else { a + (p - b) }
}

/// A validated negacyclic transform plan. Values are in standard residue form.
#[derive(Clone, Debug)]
pub struct NttPlan {
    p: u64,
    d: usize,
    /// $`-p^{-1}\bmod2^{64}`$, for Montgomery reduction.
    neg_inv: u64,
    /// The twiddles of the cyclic transforms with their Shoup constants, stage after stage: the
    /// stage of length `len` holds $`\omega^{jd/len}`$ for $`j<len/2`$, with $`\omega=\psi^2`$
    /// forward and $`\omega^{-1}`$ inverse.
    forward_twiddles: Vec<(u64, u64)>,
    inverse_twiddles: Vec<(u64, u64)>,
    /// $`\psi^i`$ and $`\psi^{-i}d^{-1}`$: the twist and untwist of the public transforms.
    twist: Vec<(u64, u64)>,
    untwist: Vec<(u64, u64)>,
    /// $`\psi^iR`$ and $`\psi^{-i}d^{-1}R^{-1}`$ with $`R=2^{64}`$: the Montgomery-form twists.
    twist_mont: Vec<(u64, u64)>,
    untwist_mont: Vec<(u64, u64)>,
    /// The bit-reversal permutation of $`[0,d)`$.
    bit_reverse: Vec<u32>,
}

impl NttPlan {
    /// Construct a plan from a prime and a primitive $`2d`$-th root.
    pub fn new(p: u64, d: usize, psi: u64) -> Result<Self, Error> {
        if !(2..=1024).contains(&d)
            || !d.is_power_of_two()
            || p >= (1 << 62)
            || !is_prime(p)
            || !(p - 1).is_multiple_of(2 * d as u64)
            || psi >= p
            || pow_mod(psi, d as u64, p) != p - 1
        {
            return Err(Error::Parameter("NTT prime, degree or root"));
        }
        let root = mul_mod(psi, psi, p);
        let inv_psi = pow_mod(psi, p - 2, p);
        let inv_d = pow_mod(d as u64, p - 2, p);
        // R = 2^64 mod p and its inverse.
        let r = ((1u128 << 64) % u128::from(p)) as u64;
        let r_inv = pow_mod(r, p - 2, p);
        // p^-1 mod 2^64 by Newton's iteration x <- x (2 - p x), which doubles the number of
        // correct low bits; x = 1 is correct modulo 2 since p is odd.
        let mut inv = 1u64;
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(p.wrapping_mul(inv)));
        }
        debug_assert_eq!(p.wrapping_mul(inv), 1);
        let pair = |w: u64| (w, shoup(w, p));
        let stages = |root: u64| {
            let mut out = Vec::with_capacity(d);
            let mut len = 2;
            while len <= d {
                let step = pow_mod(root, (d / len) as u64, p);
                let mut w = 1;
                for _ in 0..len / 2 {
                    out.push(pair(w));
                    w = mul_mod(w, step, p);
                }
                len *= 2;
            }
            out
        };
        let mut twist = Vec::with_capacity(d);
        let mut untwist = Vec::with_capacity(d);
        let mut twist_mont = Vec::with_capacity(d);
        let mut untwist_mont = Vec::with_capacity(d);
        let (mut t, mut u) = (1u64, inv_d);
        for _ in 0..d {
            twist.push(pair(t));
            untwist.push(pair(u));
            twist_mont.push(pair(mul_mod(t, r, p)));
            untwist_mont.push(pair(mul_mod(u, r_inv, p)));
            t = mul_mod(t, psi, p);
            u = mul_mod(u, inv_psi, p);
        }
        let bits = d.trailing_zeros();
        Ok(Self {
            p,
            d,
            neg_inv: inv.wrapping_neg(),
            forward_twiddles: stages(root),
            inverse_twiddles: stages(pow_mod(root, p - 2, p)),
            twist,
            untwist,
            twist_mont,
            untwist_mont,
            bit_reverse: (0..d as u32)
                .map(|i| i.reverse_bits() >> (32 - bits))
                .collect(),
        })
    }
    /// The transform modulus.
    pub fn modulus(&self) -> u64 {
        self.p
    }
    /// The transform degree.
    pub fn degree(&self) -> usize {
        self.d
    }
    fn check(&self, values: &[u64]) -> Result<(), Error> {
        if values.len() != self.d {
            return Err(Error::Dimension);
        }
        if values.iter().any(|x| *x >= self.p) {
            return Err(Error::Parameter("NTT residue"));
        }
        Ok(())
    }
    /// The cyclic transform with the stage twiddles `table`, on residues in $`[0,p)`$.
    fn cyclic(&self, a: &mut [u64], table: &[(u64, u64)]) {
        let p = self.p;
        for (i, j) in self.bit_reverse.iter().enumerate() {
            let j = *j as usize;
            if i < j {
                a.swap(i, j);
            }
        }
        let mut len = 2;
        let mut offset = 0;
        while len <= self.d {
            let half = len / 2;
            let twiddles = &table[offset..offset + half];
            for block in a.chunks_exact_mut(len) {
                let (low, high) = block.split_at_mut(half);
                for ((u, v), (w, w_shoup)) in low.iter_mut().zip(high.iter_mut()).zip(twiddles) {
                    let x = *u;
                    let y = shoup_mul(*v, *w, *w_shoup, p);
                    *u = add_mod(x, y, p);
                    *v = sub_mod(x, y, p);
                }
            }
            offset += half;
            len *= 2;
        }
    }
    /// Multiply each residue by its factor.
    fn scale(&self, a: &mut [u64], factors: &[(u64, u64)]) {
        for (x, (w, w_shoup)) in a.iter_mut().zip(factors) {
            *x = shoup_mul(*x, *w, *w_shoup, self.p);
        }
    }
    /// Transform coefficients to evaluations at odd powers of the primitive root.
    pub fn forward(&self, a: &mut [u64]) -> Result<(), Error> {
        self.check(a)?;
        self.scale(a, &self.twist);
        self.cyclic(a, &self.forward_twiddles);
        Ok(())
    }
    /// Invert a transform, including degree normalization and untwisting.
    pub fn inverse(&self, a: &mut [u64]) -> Result<(), Error> {
        self.check(a)?;
        self.cyclic(a, &self.inverse_twiddles);
        self.scale(a, &self.untwist);
        Ok(())
    }
    /// [`NttPlan::forward`] times $`R=2^{64}`$ (the Montgomery form), for `d` residues in
    /// $`[0,p)`$.
    pub(crate) fn forward_mont(&self, a: &mut [u64]) {
        debug_assert!(a.len() == self.d && a.iter().all(|x| *x < self.p));
        self.scale(a, &self.twist_mont);
        self.cyclic(a, &self.forward_twiddles);
    }
    /// [`NttPlan::inverse`] of values in Montgomery form, which removes the factor $`R`$, for
    /// `d` residues in $`[0,p)`$.
    pub(crate) fn inverse_mont(&self, a: &mut [u64]) {
        debug_assert!(a.len() == self.d && a.iter().all(|x| *x < self.p));
        self.cyclic(a, &self.inverse_twiddles);
        self.scale(a, &self.untwist_mont);
    }
    /// The Montgomery product $`abR^{-1}\bmod p`$ of $`a,b<p`$, in $`[0,p)`$.
    ///
    /// With $`t=ab<p^2`$ and $`m=t(-p^{-1})\bmod R`$, $`t+mp`$ is divisible by $`R`$ and below
    /// $`p^2+Rp<2^{127}`$, and $`u=(t+mp)/R<2p`$, so one conditional subtraction reduces it.
    #[inline(always)]
    pub(crate) fn mont_mul(&self, a: u64, b: u64) -> u64 {
        let t = u128::from(a) * u128::from(b);
        let m = (t as u64).wrapping_mul(self.neg_inv);
        let u = ((t + u128::from(m) * u128::from(self.p)) >> 64) as u64;
        if u >= self.p { u - self.p } else { u }
    }
}

#[cfg(test)]
mod oracle_tests;
