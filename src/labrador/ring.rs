//! Arithmetic in $`\mathcal R_q=\mathbb Z_q[X]/(X^d+1)`$ for a prime $`q\equiv5\pmod8`$ below
//! $`2^{62}`$, the single-precision setting of LaBRADOR (BS23 §2, §6.1).
//!
//! Coefficients are canonical `u64` residues in $`[0,q)`$. Products go through the negacyclic
//! NTTs of [`crate::math::ntt`] over the first $`k`$ primes of [`NTT_PRIMES`], in Montgomery
//! form, and back through the exact CRT of the crate ([`SmallCrt`]), applied to centred
//! operands: each product coefficient of two centred polynomials is a sum of $`d`$ terms of
//! magnitude at most $`((q-1)/2)^2`$. The ring takes the least $`k`$ for which the product
//! $`P`$ of its primes satisfies $`P/2>2^{32}d((q-1)/2)^2`$, so that up to $`2^{32}`$ products
//! accumulated in the transform domain reconstruct exactly: two primes for $`q<2^{40}`$ at
//! $`d\le256`$, three up to $`2^{62}`$. Where the IFMA kernels run, the ring takes instead the
//! fewest of the four 51-bit primes of `IFMA_PRIMES` that meet the same bound (two up to
//! $`2^{33}`$, four near $`2^{62}`$): the CRT gives the same coefficients. This arithmetic is
//! separate from the 256-bit arithmetic of the LNP22 proofs, which it does not change.
use crate::{
    Error,
    math::{
        SmallCrt,
        int::is_prime,
        ntt::{NttPlan, add_mod, mul_mod, pow_mod, shoup, shoup_mul, sub_mod},
    },
    params::moduli::NTT_PRIMES,
};
use std::sync::Arc;

/// Moduli must exceed $`2^{16}`$.
pub const MIN_MODULUS: u64 = 1 << 16;
/// Moduli must be below $`2^{62}`$: centred residues and their differences stay within `i64`,
/// Shoup products within `u64`, and three transform primes reconstruct $`2^{32}`$ accumulated
/// products exactly (the rule of [`Rq::new`]).
pub const MAX_MODULUS_BITS: u32 = 62;

/// The ring $`\mathbb Z_q[X]/(X^d+1)`$ with its transform plans and CRT constants.
#[derive(Debug)]
pub struct Rq {
    q: u64,
    d: usize,
    plans: Vec<NttPlan>,
    /// The IFMA tables of the plans, when the ring takes the primes of `IFMA_PRIMES`: where the
    /// IFMA kernels run, as many of those primes as reconstruct exactly (two for $`q<2^{32}`$,
    /// four near $`2^{62}`$). Every transform, product and inverse then goes through them
    /// (Montgomery form with $`R=2^{52}`$). Empty otherwise.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    ifma: Vec<crate::simd::IfmaPlan>,
    /// $`q\bmod p`$ for each IFMA prime $`p`$, for the centred residues when $`q\ge2p`$.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    ifma_q: Vec<u64>,
    crt: SmallCrt,
    /// The Shoup constant of the multiplier one: remainders of one word without a division.
    one_shoup: u64,
    /// $`2^{64}\bmod q`$ with its Shoup constant: remainders of two words without a division.
    wrap: (u64, u64),
    /// The residues of $`2`$ and $`-2`$ with their Shoup constants, for the challenge terms
    /// beyond $`\pm1`$.
    twos: [(u64, u64); 2],
}

/// A sparse polynomial with small integer coefficients, such as a challenge: the nonzero
/// coefficients as (position, value).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Sparse {
    pub(crate) terms: Vec<(usize, i64)>,
}

impl Rq {
    /// The ring of degree `d` (64, 128 or 256) modulo `q`, a prime with $`q\equiv5\pmod8`$ and
    /// $`2^{16}<q<2^{62}`$. Then $`X^d+1`$ splits into two irreducible factors modulo $`q`$,
    /// the setting of BS23 §2.
    pub fn new(q: u64, d: usize) -> Result<Arc<Self>, Error> {
        Self::with_primes(q, d, false)
    }

    /// Test only: the ring of [`Rq::new`] on the primes of `NTT_PRIMES` even where the IFMA
    /// kernels run, the reference of the IFMA transforms.
    #[cfg(all(test, feature = "simd", target_arch = "x86_64"))]
    pub(crate) fn new_word(q: u64, d: usize) -> Result<Arc<Self>, Error> {
        Self::with_primes(q, d, true)
    }

    #[cfg_attr(
        not(all(feature = "simd", target_arch = "x86_64")),
        allow(unused_variables)
    )]
    fn with_primes(q: u64, d: usize, word_only: bool) -> Result<Arc<Self>, Error> {
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
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        let ifma = if word_only {
            Vec::new()
        } else {
            ifma_plans(d, need)?
        };
        let mut plans = Vec::with_capacity(k);
        for (p, root) in NTT_PRIMES.iter().take(k) {
            plans.push(NttPlan::new(*p, d, pow_mod(*root, (1024 / d) as u64, *p))?);
        }
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if !ifma.is_empty() {
            plans = ifma.iter().map(|t| t.plan.clone()).collect();
        }
        let primes: Vec<u64> = plans.iter().map(NttPlan::modulus).collect();
        let wrap = ((1u128 << 64) % u128::from(q)) as u64;
        Ok(Arc::new(Self {
            q,
            d,
            crt: SmallCrt::new(q, &primes),
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            ifma_q: ifma.iter().map(|t| q % t.plan.modulus()).collect(),
            plans,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            ifma,
            one_shoup: shoup(1, q),
            wrap: (wrap, shoup(wrap, q)),
            twos: [(2, shoup(2, q)), (q - 2, shoup(q - 2, q))],
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

    /// The canonical residue of any integer: one addition for $`|x|<q`$, the case of every
    /// digit and short coefficient, a division otherwise.
    #[inline(always)]
    pub(crate) fn lift(&self, x: i64) -> u64 {
        let q = self.q as i64;
        if (0..q).contains(&x) {
            x as u64
        } else if (-q..0).contains(&x) {
            (x + q) as u64
        } else {
            x.rem_euclid(q) as u64
        }
    }

    /// The canonical residue of any `i128`, as [`Rq::lift`].
    #[inline(always)]
    pub(crate) fn lift_i128(&self, x: i128) -> u64 {
        let q = i128::from(self.q);
        if (0..q).contains(&x) {
            x as u64
        } else if (-q..0).contains(&x) {
            (x + q) as u64
        } else {
            x.rem_euclid(q) as u64
        }
    }

    /// $`x\bmod q`$ for any $`x<2^{64}`$, by one Shoup product.
    #[inline(always)]
    pub(crate) fn reduce_word(&self, x: u64) -> u64 {
        shoup_mul(x, 1, self.one_shoup, self.q)
    }

    /// $`x\bmod q`$ for any $`x<2^{128}`$: $`x_12^{64}+x_0\equiv x_1(2^{64}\bmod q)+x_0`$, by
    /// two Shoup products.
    #[inline(always)]
    pub(crate) fn reduce_wide(&self, x: u128) -> u64 {
        let (wrap, wrap_shoup) = self.wrap;
        let high = shoup_mul((x >> 64) as u64, wrap, wrap_shoup, self.q);
        add_mod(high, self.reduce_word(x as u64), self.q)
    }

    /// $`a+b\bmod q`$.
    #[inline(always)]
    pub(crate) fn add(&self, a: u64, b: u64) -> u64 {
        add_mod(a, b, self.q)
    }

    /// $`a-b\bmod q`$.
    #[inline(always)]
    pub(crate) fn sub(&self, a: u64, b: u64) -> u64 {
        sub_mod(a, b, self.q)
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
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if crate::simd::add_mod(a, b, self.q) {
            return;
        }
        for (x, y) in a.iter_mut().zip(b) {
            *x = add_mod(*x, *y, self.q);
        }
    }

    /// `a -= b` coefficientwise, for slices of equal length.
    pub(crate) fn sub_assign(&self, a: &mut [u64], b: &[u64]) {
        debug_assert_eq!(a.len(), b.len());
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if crate::simd::sub_mod(a, b, self.q) {
            return;
        }
        for (x, y) in a.iter_mut().zip(b) {
            *x = sub_mod(*x, *y, self.q);
        }
    }

    /// `a += s * b` coefficientwise for a scalar $`s\in[0,q)`$, with one Shoup constant for
    /// all the products.
    pub(crate) fn add_scaled_assign(&self, a: &mut [u64], b: &[u64], s: u64) {
        debug_assert_eq!(a.len(), b.len());
        if s == 0 {
            return;
        }
        let s_shoup = shoup(s, self.q);
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if crate::simd::add_scaled(a, b, s, s_shoup, self.q) {
            return;
        }
        for (x, y) in a.iter_mut().zip(b) {
            *x = add_mod(*x, shoup_mul(*y, s, s_shoup, self.q), self.q);
        }
    }

    /// `a *= s` coefficientwise for a scalar $`s\in[0,q)`$, with one Shoup constant.
    pub(crate) fn scale_assign(&self, a: &mut [u64], s: u64) {
        let s_shoup = shoup(s, self.q);
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if crate::simd::scale(a, s, s_shoup, self.q) {
            return;
        }
        for x in a.iter_mut() {
            *x = shoup_mul(*x, s, s_shoup, self.q);
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
        for (i, (plan, block)) in self
            .plans
            .iter()
            .zip(out.chunks_exact_mut(self.d))
            .enumerate()
        {
            let p = plan.modulus();
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            if let Some(t) = self.ifma.get(i) {
                // Centred values reach q/2, beyond p from q = 2p on.
                if self.q / 2 >= p {
                    crate::simd::ifma_centred(t, a, self.q / 2, self.ifma_q[i], block);
                } else if !crate::simd::centred(a, self.q, p, block) {
                    self.centre_into(a, p, block);
                }
                crate::simd::ifma_forward(t, block);
                continue;
            }
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            if crate::simd::centred(a, self.q, p, block) {
                plan.forward_mont(block);
                continue;
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            let _ = i;
            self.centre_into(a, p, block);
            plan.forward_mont(block);
        }
    }

    /// The residues modulo $`p`$ of the centred representatives of `a`, into `block`: their
    /// magnitudes are below $`p`$ for $`q<2p`$ and reduced otherwise.
    fn centre_into(&self, a: &[u64], p: u64, block: &mut [u64]) {
        if self.q / 2 < p {
            for (y, x) in block.iter_mut().zip(a) {
                let c = self.center(*x);
                *y = if c >= 0 {
                    c as u64
                } else {
                    p - c.unsigned_abs()
                };
            }
            return;
        }
        for (y, x) in block.iter_mut().zip(a) {
            let c = self.center(*x);
            let r = c.unsigned_abs() % p;
            *y = if c >= 0 || r == 0 { r } else { p - r };
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
        for (i, (plan, block)) in self
            .plans
            .iter()
            .zip(h.chunks_exact_mut(self.d))
            .enumerate()
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            if let Some(t) = self.ifma.get(i) {
                crate::simd::ifma_inverse(t, block);
                continue;
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            let _ = i;
            plan.inverse_mont(block);
        }
        // At most four primes (three of `NTT_PRIMES`, four of `IFMA_PRIMES` near 2^62): the
        // blocks themselves are the CRT's residues.
        let mut residues: [&[u64]; 4] = [&[]; 4];
        for (slot, block) in residues.iter_mut().zip(h.chunks_exact(self.d)) {
            *slot = block;
        }
        self.crt
            .reconstruct_into(&residues[..self.plans.len()], out);
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
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                if let Some(t) = self.ifma.get(i) {
                    crate::simd::ifma_mac(t, &mut acc[range.clone()], &a[range.clone()], &b[range]);
                    continue;
                }
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                if crate::simd::mont_mac(
                    &mut acc[range.clone()],
                    &a[range.clone()],
                    &b[range.clone()],
                    p,
                    plan.neg_inv(),
                ) {
                    continue;
                }
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
    pub(crate) fn hat_add_assign(&self, acc: &mut [u64], a: &[u64]) {
        debug_assert!(acc.len() == a.len() && acc.len().is_multiple_of(self.hat_len()));
        for (block, (x, y)) in acc
            .chunks_exact_mut(self.d)
            .zip(a.chunks_exact(self.d))
            .enumerate()
        {
            let p = self.plans[block % self.plans.len()].modulus();
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            if crate::simd::add_mod(x, y, p) {
                continue;
            }
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
        let (d, q) = (self.d, self.q);
        for &(pos, value) in &c.terms {
            debug_assert!(pos < d);
            // X^pos a: coefficient j moves to j + pos, with a sign change past X^d. The first
            // d - pos coefficients land on out[pos..], the others, negated, on out[..pos].
            let (head, tail) = a.split_at(d - pos);
            let (low, high) = out.split_at_mut(pos);
            match value {
                1 => {
                    self.add_assign(high, head);
                    self.sub_assign(low, tail);
                }
                -1 => {
                    self.sub_assign(high, head);
                    self.add_assign(low, tail);
                }
                _ => {
                    // One Shoup constant per term, so no product divides. Challenges have
                    // coefficients up to 2 in magnitude, whose constants the ring keeps.
                    let (v, v_shoup) = match value {
                        2 => self.twos[0],
                        -2 => self.twos[1],
                        _ => {
                            let v = self.lift(value);
                            (v, shoup(v, q))
                        }
                    };
                    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                    let (high_done, low_done) = (
                        crate::simd::add_scaled(high, head, v, v_shoup, q),
                        crate::simd::sub_scaled(low, tail, v, v_shoup, q),
                    );
                    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                    let (high_done, low_done) = (false, false);
                    if !high_done {
                        for (o, x) in high.iter_mut().zip(head) {
                            *o = add_mod(*o, shoup_mul(*x, v, v_shoup, q), q);
                        }
                    }
                    if !low_done {
                        for (o, x) in low.iter_mut().zip(tail) {
                            *o = sub_mod(*o, shoup_mul(*x, v, v_shoup, q), q);
                        }
                    }
                }
            }
        }
    }

    /// The squared Euclidean norm of the centred coefficients, saturating at `u128::MAX`: a
    /// square is below $`2^{122}`$ for $`q<2^{62}`$, so sums of more than 64 of them could wrap,
    /// and a saturated norm is above every bound.
    pub(crate) fn norm_sq(&self, a: &[u64]) -> u128 {
        a.iter().fold(0u128, |acc, x| {
            let c = u128::from(self.center(*x).unsigned_abs());
            acc.saturating_add(c * c)
        })
    }

    /// The largest centred coefficient magnitude.
    pub(crate) fn linf(&self, a: &[u64]) -> u64 {
        a.iter()
            .map(|x| self.center(*x).unsigned_abs())
            .max()
            .unwrap_or(0)
    }
}

/// The IFMA plans for degree `d`, where the IFMA kernels run: the fewest primes of
/// `IFMA_PRIMES` whose product $`P`$ exceeds $`2^{\mathrm{need}+1}`$, the exact rule of
/// [`Rq::new`], or none if all four fall short.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn ifma_plans(d: usize, need: u32) -> Result<Vec<crate::simd::IfmaPlan>, Error> {
    use crate::math::U256;
    use crate::simd::{IFMA_PRIMES, IfmaPlan};
    if !crate::simd::ifma() {
        return Ok(Vec::new());
    }
    let mut product = U256::ONE;
    for (k, (p, _)) in IFMA_PRIMES.iter().enumerate() {
        product = product.wrapping_mul(&U256::from_u64(*p));
        // P is odd, so P > 2^(need + 1) exactly when P >> (need + 1) is nonzero.
        if product.shr_vartime(need + 1) != U256::ZERO {
            return IFMA_PRIMES[..=k]
                .iter()
                .map(|(p, root)| {
                    Ok(IfmaPlan::new(NttPlan::new(
                        *p,
                        d,
                        pow_mod(*root, (1024 / d) as u64, *p),
                    )?))
                })
                .collect();
        }
    }
    Ok(Vec::new())
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
