//! Arithmetic in $`\mathcal R_q=\mathbb Z_q[X]/(X^d+1)`$ for a prime $`q\equiv5\pmod8`$ below
//! $`2^{48}`$, the single-precision setting of LaBRADOR (BS23 §2, §6.1).
//!
//! Coefficients are canonical `u64` residues in $`[0,q)`$. Products go through the negacyclic
//! NTTs of [`crate::math::ntt`] over the first $`k`$ primes of [`NTT_PRIMES`], in Montgomery
//! form, and back through the exact CRT of the crate ([`SmallCrt`]), applied to centred
//! operands: each product coefficient of two centred polynomials is a sum of $`d`$ terms of
//! magnitude at most $`((q-1)/2)^2`$. The ring takes the least $`k`$ for which the product
//! $`P`$ of its primes satisfies $`P/2>2^{32}d((q-1)/2)^2`$, so that up to $`2^{32}`$ products
//! accumulated in the transform domain reconstruct exactly: two primes for $`q<2^{40}`$ at
//! $`d\le256`$, three up to $`2^{48}`$. This arithmetic is separate from the 256-bit
//! arithmetic of the LNP22 proofs, which it does not change.
use crate::{
    Error,
    math::{
        SmallCrt,
        int::is_prime,
        ntt::{NttPlan, add_mod, mul_mod, pow_mod, sub_mod},
    },
    params::moduli::NTT_PRIMES,
};
use std::sync::Arc;

/// Moduli must exceed $`2^{16}`$.
pub const MIN_MODULUS: u64 = 1 << 16;
/// Moduli must be below $`2^{48}`$.
pub const MAX_MODULUS_BITS: u32 = 48;

/// The ring $`\mathbb Z_q[X]/(X^d+1)`$ with its transform plans and CRT constants.
#[derive(Debug)]
pub struct Rq {
    q: u64,
    d: usize,
    plans: Vec<NttPlan>,
    crt: SmallCrt,
}

/// A sparse polynomial with small integer coefficients, such as a challenge: the nonzero
/// coefficients as (position, value).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Sparse {
    pub(crate) terms: Vec<(usize, i64)>,
}

impl Rq {
    /// The ring of degree `d` (64, 128 or 256) modulo `q`, a prime with $`q\equiv5\pmod8`$ and
    /// $`2^{16}<q<2^{48}`$. Then $`X^d+1`$ splits into two irreducible factors modulo $`q`$,
    /// the setting of BS23 §2.
    pub fn new(q: u64, d: usize) -> Result<Arc<Self>, Error> {
        if !matches!(d, 64 | 128 | 256) {
            return Err(Error::Parameter("LaBRADOR ring degree"));
        }
        if q <= MIN_MODULUS || q >= 1 << MAX_MODULUS_BITS || q % 8 != 5 || !is_prime(q) {
            return Err(Error::Parameter("LaBRADOR modulus"));
        }
        let q_bits = 64 - q.leading_zeros();
        // The largest accumulated magnitude is below 2^(32 + log2 d + 2 q_bits - 2). Every
        // prime of NTT_PRIMES exceeds 2^61, so k primes give P/2 > 2^(61k - 1).
        let need = 30 + d.trailing_zeros() + 2 * q_bits;
        let k = (need + 1).div_ceil(61) as usize;
        debug_assert!(61 * k as u32 > need && k <= NTT_PRIMES.len());
        let mut plans = Vec::with_capacity(k);
        for (p, root) in NTT_PRIMES.iter().take(k) {
            plans.push(NttPlan::new(*p, d, pow_mod(*root, (1024 / d) as u64, *p))?);
        }
        let primes: Vec<u64> = plans.iter().map(NttPlan::modulus).collect();
        Ok(Arc::new(Self {
            q,
            d,
            crt: SmallCrt::new(q, &primes),
            plans,
        }))
    }

    /// The modulus $`q`$.
    pub fn modulus(&self) -> u64 {
        self.q
    }

    /// The degree $`d`$.
    pub fn degree(&self) -> usize {
        self.d
    }

    /// The number of transform primes.
    #[cfg(test)]
    pub(crate) fn primes(&self) -> usize {
        self.plans.len()
    }

    /// The length of a polynomial in the transform domain, $`kd`$.
    pub(crate) fn hat_len(&self) -> usize {
        self.plans.len() * self.d
    }

    /// The centred representative in $`[-(q-1)/2,(q-1)/2]`$ of a canonical residue.
    #[inline(always)]
    pub(crate) fn center(&self, x: u64) -> i64 {
        if x > self.q / 2 {
            x as i64 - self.q as i64
        } else {
            x as i64
        }
    }

    /// The canonical residue of any integer.
    #[inline(always)]
    pub(crate) fn lift(&self, x: i64) -> u64 {
        x.rem_euclid(self.q as i64) as u64
    }

    /// The canonical residue of any `i128`.
    #[inline(always)]
    pub(crate) fn lift_i128(&self, x: i128) -> u64 {
        x.rem_euclid(i128::from(self.q)) as u64
    }

    /// $`a+b\bmod q`$.
    #[inline(always)]
    pub(crate) fn add(&self, a: u64, b: u64) -> u64 {
        add_mod(a, b, self.q)
    }

    /// $`ab\bmod q`$.
    #[inline(always)]
    pub(crate) fn mul(&self, a: u64, b: u64) -> u64 {
        mul_mod(a, b, self.q)
    }

    /// $`-a\bmod q`$.
    #[inline(always)]
    pub(crate) fn neg(&self, a: u64) -> u64 {
        if a == 0 { 0 } else { self.q - a }
    }

    /// `a += b` coefficientwise, for slices of equal length.
    pub(crate) fn add_assign(&self, a: &mut [u64], b: &[u64]) {
        debug_assert_eq!(a.len(), b.len());
        for (x, y) in a.iter_mut().zip(b) {
            *x = add_mod(*x, *y, self.q);
        }
    }

    /// `a -= b` coefficientwise, for slices of equal length.
    pub(crate) fn sub_assign(&self, a: &mut [u64], b: &[u64]) {
        debug_assert_eq!(a.len(), b.len());
        for (x, y) in a.iter_mut().zip(b) {
            *x = sub_mod(*x, *y, self.q);
        }
    }

    /// `a += s * b` coefficientwise for a scalar $`s\in[0,q)`$.
    pub(crate) fn add_scaled_assign(&self, a: &mut [u64], b: &[u64], s: u64) {
        debug_assert_eq!(a.len(), b.len());
        if s == 0 {
            return;
        }
        for (x, y) in a.iter_mut().zip(b) {
            *x = add_mod(*x, mul_mod(*y, s, self.q), self.q);
        }
    }

    /// `a *= s` coefficientwise for a scalar $`s\in[0,q)`$.
    pub(crate) fn scale_assign(&self, a: &mut [u64], s: u64) {
        for x in a.iter_mut() {
            *x = mul_mod(*x, s, self.q);
        }
    }

    /// $`\sigma_{-1}(a)`$, $`X\mapsto X^{-1}`$: coefficients $`(a_0,-a_{d-1},\dots,-a_1)`$. Then
    /// $`\mathrm{ct}(\sigma_{-1}(a)b)=\sum_ia_ib_i`$ (BS23 §2).
    pub(crate) fn sigma_minus_one(&self, a: &[u64], out: &mut [u64]) {
        debug_assert!(a.len() == self.d && out.len() == self.d);
        out[0] = a[0];
        for i in 1..self.d {
            out[i] = self.neg(a[self.d - i]);
        }
    }

    /// The transform of one polynomial into `out` ($`kd`$ values, prime after prime, in
    /// Montgomery form).
    pub(crate) fn ntt_into(&self, a: &[u64], out: &mut [u64]) {
        debug_assert!(a.len() == self.d && out.len() == self.hat_len());
        for (plan, block) in self.plans.iter().zip(out.chunks_exact_mut(self.d)) {
            let p = plan.modulus();
            for (y, x) in block.iter_mut().zip(a) {
                let c = self.center(*x);
                *y = if c >= 0 {
                    c as u64
                } else {
                    p - c.unsigned_abs()
                };
            }
            plan.forward_mont(block);
        }
    }

    /// The transform of one polynomial.
    pub(crate) fn ntt(&self, a: &[u64]) -> Vec<u64> {
        let mut out = vec![0; self.hat_len()];
        self.ntt_into(a, &mut out);
        out
    }

    /// The transforms of consecutive polynomials ($`|a|/d`$ of them), concatenated.
    pub(crate) fn ntt_many(&self, a: &[u64]) -> Vec<u64> {
        debug_assert_eq!(a.len() % self.d, 0);
        let mut out = vec![0; a.len() / self.d * self.hat_len()];
        for (x, y) in a
            .chunks_exact(self.d)
            .zip(out.chunks_exact_mut(self.hat_len()))
        {
            self.ntt_into(x, y);
        }
        out
    }

    /// The canonical coefficients of a transform-domain value, which `h` held (it is
    /// overwritten). Exact while the value is the sum of at most $`2^{32}`$ products of
    /// transformed canonical polynomials, or of their sums and differences within that count.
    pub(crate) fn intt_into(&self, h: &mut [u64], out: &mut [u64]) {
        debug_assert!(h.len() == self.hat_len() && out.len() == self.d);
        let mut residues = Vec::with_capacity(self.plans.len());
        for (plan, block) in self.plans.iter().zip(h.chunks_exact_mut(self.d)) {
            plan.inverse_mont(block);
            residues.push(block.to_vec());
        }
        for (j, x) in out.iter_mut().enumerate() {
            *x = self.crt.reconstruct(&residues, j);
        }
    }

    /// [`Rq::intt_into`] into a new vector.
    pub(crate) fn intt(&self, mut h: Vec<u64>) -> Vec<u64> {
        let mut out = vec![0; self.d];
        self.intt_into(&mut h, &mut out);
        out
    }

    /// `acc += a ⊙ b` in the transform domain (one or more polynomials, equal lengths).
    #[inline]
    pub(crate) fn mac(&self, acc: &mut [u64], a: &[u64], b: &[u64]) {
        debug_assert!(acc.len() == a.len() && a.len() == b.len());
        debug_assert_eq!(acc.len() % self.hat_len(), 0);
        for poly in 0..acc.len() / self.hat_len() {
            for (i, plan) in self.plans.iter().enumerate() {
                let p = plan.modulus();
                let start = poly * self.hat_len() + i * self.d;
                let range = start..start + self.d;
                for ((x, y), z) in acc[range.clone()]
                    .iter_mut()
                    .zip(&a[range.clone()])
                    .zip(&b[range])
                {
                    *x = add_mod(*x, plan.mont_mul(*y, *z), p);
                }
            }
        }
    }

    /// `acc += a` in the transform domain.
    #[cfg(test)]
    pub(crate) fn hat_add_assign(&self, acc: &mut [u64], a: &[u64]) {
        debug_assert!(acc.len() == a.len() && acc.len().is_multiple_of(self.hat_len()));
        for (block, (x, y)) in acc
            .chunks_exact_mut(self.d)
            .zip(a.chunks_exact(self.d))
            .enumerate()
        {
            let p = self.plans[block % self.plans.len()].modulus();
            for (u, v) in x.iter_mut().zip(y) {
                *u = add_mod(*u, *v, p);
            }
        }
    }

    /// `acc -= a` in the transform domain.
    #[cfg(test)]
    pub(crate) fn hat_sub_assign(&self, acc: &mut [u64], a: &[u64]) {
        debug_assert!(acc.len() == a.len() && acc.len().is_multiple_of(self.hat_len()));
        for (block, (x, y)) in acc
            .chunks_exact_mut(self.d)
            .zip(a.chunks_exact(self.d))
            .enumerate()
        {
            let p = self.plans[block % self.plans.len()].modulus();
            for (u, v) in x.iter_mut().zip(y) {
                *u = sub_mod(*u, *v, p);
            }
        }
    }

    /// $`ab`$ in $`\mathcal R_q`$.
    pub(crate) fn poly_mul(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let (x, y) = (self.ntt(a), self.ntt(b));
        let mut acc = vec![0; self.hat_len()];
        self.mac(&mut acc, &x, &y);
        self.intt(acc)
    }

    /// $`\sum_ja_jb_j`$ over transformed polynomial vectors of equal length.
    pub(crate) fn inner_hat(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        debug_assert_eq!(a.len(), b.len());
        let k = self.hat_len();
        let mut acc = vec![0; k];
        for (x, y) in a.chunks_exact(k).zip(b.chunks_exact(k)) {
            self.mac(&mut acc, x, y);
        }
        self.intt(acc)
    }

    /// `out += c·a` for a sparse polynomial `c` with small coefficients.
    pub(crate) fn mul_sparse_add(&self, c: &Sparse, a: &[u64], out: &mut [u64]) {
        debug_assert!(a.len() == self.d && out.len() == self.d);
        let d = self.d;
        for &(pos, value) in &c.terms {
            let v = self.lift(value);
            // X^pos a: coefficient j moves to j + pos, with a sign change past X^d.
            for (j, x) in a.iter().enumerate() {
                let t = mul_mod(*x, v, self.q);
                let idx = j + pos;
                if idx < d {
                    out[idx] = add_mod(out[idx], t, self.q);
                } else {
                    out[idx - d] = sub_mod(out[idx - d], t, self.q);
                }
            }
        }
    }

    /// The squared Euclidean norm of the centred coefficients.
    pub(crate) fn norm_sq(&self, a: &[u64]) -> u128 {
        a.iter()
            .map(|x| {
                let c = i128::from(self.center(*x));
                (c * c) as u128
            })
            .sum()
    }

    /// The largest centred coefficient magnitude.
    pub(crate) fn linf(&self, a: &[u64]) -> u64 {
        a.iter()
            .map(|x| self.center(*x).unsigned_abs())
            .max()
            .unwrap_or(0)
    }
}

impl Sparse {
    /// The canonical dense coefficients modulo $`q`$.
    pub(crate) fn dense(&self, ring: &Rq) -> Vec<u64> {
        let mut out = vec![0; ring.degree()];
        for &(pos, value) in &self.terms {
            out[pos] = ring.add(out[pos], ring.lift(value));
        }
        out
    }
}

#[cfg(test)]
mod tests;
