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
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

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

/// $`v\bmod p`$ for $`v<4p`$, by two conditional subtractions.
#[inline(always)]
fn reduce_four_p(v: u64, p: u64) -> u64 {
    let v = if v >= 2 * p { v - 2 * p } else { v };
    if v >= p { v - p } else { v }
}

/// [`shoup_mul`] without its final subtraction: a value in $`[0,2p)`$ congruent to $`xw`$, for
/// any $`x<2^{64}`$ (the bound of `shoup_mul`'s proof before the subtraction).
#[inline(always)]
pub(crate) fn shoup_lazy(x: u64, w: u64, w_shoup: u64, p: u64) -> u64 {
    let k = ((u128::from(x) * u128::from(w_shoup)) >> 64) as u64;
    x.wrapping_mul(w).wrapping_sub(k.wrapping_mul(p))
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
    /// $`\lfloor(2^{128}-1)/p\rfloor`$, for Barrett reductions of 128-bit sums.
    barrett: u128,
    /// The twiddles of the cyclic transforms with their Shoup constants, stage after stage: the
    /// stage of length `len` holds $`\omega^{jd/len}`$ for $`j<len/2`$, with $`\omega=\psi^2`$
    /// forward and $`\omega^{-1}`$ inverse.
    forward_twiddles: Vec<(u64, u64)>,
    inverse_twiddles: Vec<(u64, u64)>,
    /// $`\psi^i`$ and $`\psi^{-i}d^{-1}`$: the twist and untwist of the public transforms.
    twist: Vec<(u64, u64)>,
    untwist: Vec<(u64, u64)>,
    /// The merged negacyclic transforms of the Montgomery path, with Shoup constants:
    /// $`\zeta_k=\psi^{\mathrm{brv}(k)}`$ for $`1\le k<d`$ (index 0 unused), the twiddle of the
    /// $`k`$-th butterfly group in Cooley-Tukey order, and $`\zeta_k^{-1}`$.
    zetas: Vec<(u64, u64)>,
    inv_zetas: Vec<(u64, u64)>,
    /// $`R=2^{64}\bmod p`$ and $`\zeta_1R`$: the first forward stage puts the factor $`R`$ in.
    r: (u64, u64),
    first: (u64, u64),
    /// $`c=d^{-1}R^{-1}`$ and $`\zeta_1^{-1}c`$: the last inverse stage takes $`dR`$ out.
    scale: (u64, u64),
    last: (u64, u64),
    /// The bit-reversal permutation of $`[0,d)`$.
    bit_reverse: Vec<u32>,
    /// The twiddles of `zetas` and `inv_zetas` and their Shoup constants, each in its own
    /// array, as the SIMD kernels load them.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    simd: [Vec<u64>; 4],
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
        let (mut t, mut u) = (1u64, inv_d);
        for _ in 0..d {
            twist.push(pair(t));
            untwist.push(pair(u));
            t = mul_mod(t, psi, p);
            u = mul_mod(u, inv_psi, p);
        }
        let bits = d.trailing_zeros();
        let bit_reverse: Vec<u32> = (0..d as u32)
            .map(|i| i.reverse_bits() >> (32 - bits))
            .collect();
        let mut zetas = vec![(0, 0); d];
        let mut inv_zetas = vec![(0, 0); d];
        for k in 1..d {
            let z = pow_mod(psi, u64::from(bit_reverse[k]), p);
            zetas[k] = pair(z);
            inv_zetas[k] = pair(pow_mod(z, p - 2, p));
        }
        let c = mul_mod(inv_d, r_inv, p);
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        let simd = [
            zetas.iter().map(|w| w.0).collect(),
            zetas.iter().map(|w| w.1).collect(),
            inv_zetas.iter().map(|w| w.0).collect(),
            inv_zetas.iter().map(|w| w.1).collect(),
        ];
        Ok(Self {
            p,
            d,
            neg_inv: inv.wrapping_neg(),
            barrett: u128::MAX / u128::from(p),
            forward_twiddles: stages(root),
            inverse_twiddles: stages(pow_mod(root, p - 2, p)),
            twist,
            untwist,
            r: pair(r),
            first: pair(mul_mod(zetas[1].0, r, p)),
            scale: pair(c),
            last: pair(mul_mod(inv_zetas[1].0, c, p)),
            zetas,
            inv_zetas,
            bit_reverse,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            simd,
        })
    }
    /// The constants that the SIMD kernels read.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    pub(crate) fn simd_tables(&self) -> crate::simd::NttTables<'_> {
        crate::simd::NttTables {
            p: self.p,
            d: self.d,
            zetas: &self.simd[0],
            zetas_shoup: &self.simd[1],
            inv_zetas: &self.simd[2],
            inv_zetas_shoup: &self.simd[3],
            r: self.r,
            first: self.first,
            scale: self.scale,
            last: self.last,
        }
    }
    /// The plan of [`NttPlan::new`], built once per process for each $`(p,d,\psi)`$ and shared.
    ///
    /// Rings build one plan per prime, and the statement compiler builds rings for every form
    /// it lowers, so the tables of the same few plans were computed again and again (a tenth of
    /// the time of an end-to-end lattice test, measured 2026-11-16). Only rings call this, with
    /// the primes of `NTT_PRIMES` at the five degrees, so the memo holds at most 45 plans, about
    /// 2 MB at the most. Plans are deterministic, so a shared one equals a new one. Rings keep
    /// copies.
    pub(crate) fn shared(p: u64, d: usize, psi: u64) -> Result<Arc<Self>, Error> {
        type Memo = Mutex<HashMap<(u64, usize, u64), Arc<NttPlan>>>;
        static PLANS: OnceLock<Memo> = OnceLock::new();
        let memo = PLANS.get_or_init(Default::default);
        if let Some(plan) = memo.lock().expect("plan memo").get(&(p, d, psi)) {
            return Ok(plan.clone());
        }
        // Built outside the lock: two threads may build the same plan, and the first one in
        // is kept.
        let plan = Arc::new(Self::new(p, d, psi)?);
        Ok(memo
            .lock()
            .expect("plan memo")
            .entry((p, d, psi))
            .or_insert(plan)
            .clone())
    }
    /// The transform modulus.
    #[inline]
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
    /// The forward transform of the crate's RNS products: evaluations at the roots of
    /// $`X^d+1`$, times $`R=2^{64}`$ (the Montgomery form), in the order of the merged
    /// negacyclic Cooley-Tukey transform (bit-reversed). Only [`NttPlan::inverse_mont`] and
    /// pointwise operations read the values, so their order is internal. For `d` residues in
    /// $`[0,p)`$, returned in $`[0,p)`$.
    ///
    /// The stage of half-length `len` splits each factor $`X^{2len}-\zeta^2`$ into
    /// $`(X^{len}-\zeta)(X^{len}+\zeta)`$ by the butterflies $`(x,y)\mapsto(x+\zeta y,x-\zeta y)`$,
    /// with $`\zeta=\zeta_k`$ for the $`k`$-th group, starting from $`X^d+1=X^d-\psi^d`$. The
    /// first stage multiplies both inputs, by $`R`$ and $`\zeta_1R`$, so every output carries
    /// $`R`$ by linearity. Values stay below $`4p<2^{64}`$ between stages (Harvey's lazy
    /// butterflies: the upper input is reduced below $`2p`$, lazy Shoup products are below $`2p`$).
    pub(crate) fn forward_mont(&self, a: &mut [u64]) {
        debug_assert!(a.len() == self.d && a.iter().all(|x| *x < self.p));
        // The same values from AVX-512 registers where they run (`crate::simd`).
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if crate::simd::forward_mont(&self.simd_tables(), a) {
            return;
        }
        self.forward_mont_scalar(a);
    }
    /// [`NttPlan::forward_mont`] in scalar code, the reference of the SIMD kernels: two stages per
    /// pass on aarch64, one elsewhere, with the same outputs. The pairs of stages took less time
    /// on an Apple M4, also in whole proofs. On an Intel Xeon 8581C they ran faster in a loop of
    /// transforms but made whole proofs take 4% to 6% more time (one binary switching between
    /// the two, measured 2026-11-18), so x86 and the other targets keep the stage-by-stage code.
    pub(crate) fn forward_mont_scalar(&self, a: &mut [u64]) {
        if cfg!(target_arch = "aarch64") {
            self.forward_mont_radix4(a);
        } else {
            self.forward_mont_radix2(a);
        }
    }
    /// [`NttPlan::forward_mont`] two stages per pass (radix 4).
    pub(crate) fn forward_mont_radix4(&self, a: &mut [u64]) {
        let (p, d) = (self.p, self.d);
        let two_p = 2 * p;
        let half = d / 2;
        // The stages two at a time (radix 4): in a block of 4l, the butterflies of the stage at
        // distance 2l with the block's twiddle, then those at distance l with the twiddles of
        // its halves, on the same 4 values. These are the butterflies of the radix-2 stages on
        // the same values ([`NttPlan::forward_mont_radix2`]), so the outputs are equal, and a
        // pass over the vector serves two stages. With an even number of stages the first one
        // (which puts R in) pairs with the second, with an odd number it runs alone. The last
        // pass also reduces into [0, p), which a pass of its own did before.
        let (mut len, mut k) = if d >= 4 && d.trailing_zeros().is_multiple_of(2) {
            if d == 4 {
                self.forward_first_pair::<true>(a);
            } else {
                self.forward_first_pair::<false>(a);
            }
            (half / 4, 4)
        } else {
            let (r, r_shoup) = self.r;
            let (z, z_shoup) = self.first;
            let (low, high) = a.split_at_mut(half);
            for (x, y) in low.iter_mut().zip(high.iter_mut()) {
                let u = shoup_lazy(*x, r, r_shoup, p);
                let t = shoup_lazy(*y, z, z_shoup, p);
                *x = u + t;
                *y = u + two_p - t;
            }
            (half / 2, 2)
        };
        while len >= 2 {
            let l = len / 2;
            if l == 1 {
                self.forward_pair::<true>(a, k, l);
            } else {
                self.forward_pair::<false>(a, k, l);
            }
            k *= 4;
            len /= 4;
        }
        // The pairing leaves no stage alone after the first: an odd number of stages pairs
        // all but the first, an even number all of them. Degree 2 has its one stage only.
        debug_assert_eq!(len, 0);
        if d == 2 {
            for x in a.iter_mut() {
                *x = reduce_four_p(*x, p);
            }
        }
    }
    /// The first two stages of [`NttPlan::forward_mont`] in one pass: the first, which
    /// multiplies its inputs by $`R`$ and $`\zeta_1R`$, and the second, with $`\zeta_2`$ and
    /// $`\zeta_3`$. With `LAST` (degree 4) the outputs are reduced into $`[0,p)`$.
    #[inline(always)]
    fn forward_first_pair<const LAST: bool>(&self, a: &mut [u64]) {
        let p = self.p;
        let two_p = 2 * p;
        let l = self.d / 4;
        let (r, r_shoup) = self.r;
        let (z, z_shoup) = self.first;
        let (w0, w0_shoup) = self.zetas[2];
        let (w1, w1_shoup) = self.zetas[3];
        let (b0, rest) = a.split_at_mut(l);
        let (b1, rest) = rest.split_at_mut(l);
        let (b2, b3) = rest.split_at_mut(l);
        for (((x0, x1), x2), x3) in b0.iter_mut().zip(b1).zip(b2).zip(b3) {
            let u0 = shoup_lazy(*x0, r, r_shoup, p);
            let t0 = shoup_lazy(*x2, z, z_shoup, p);
            let u1 = shoup_lazy(*x1, r, r_shoup, p);
            let t1 = shoup_lazy(*x3, z, z_shoup, p);
            let (y0, y2) = (u0 + t0, u0 + two_p - t0);
            let (y1, y3) = (u1 + t1, u1 + two_p - t1);
            let v0 = if y0 >= two_p { y0 - two_p } else { y0 };
            let s0 = shoup_lazy(y1, w0, w0_shoup, p);
            let v2 = if y2 >= two_p { y2 - two_p } else { y2 };
            let s2 = shoup_lazy(y3, w1, w1_shoup, p);
            if LAST {
                *x0 = reduce_four_p(v0 + s0, p);
                *x1 = reduce_four_p(v0 + two_p - s0, p);
                *x2 = reduce_four_p(v2 + s2, p);
                *x3 = reduce_four_p(v2 + two_p - s2, p);
            } else {
                *x0 = v0 + s0;
                *x1 = v0 + two_p - s0;
                *x2 = v2 + s2;
                *x3 = v2 + two_p - s2;
            }
        }
    }
    /// Two stages of [`NttPlan::forward_mont`] in one pass: blocks of $`4l`$ whose first
    /// twiddle index is `k`. With `LAST` the outputs are reduced into $`[0,p)`$.
    #[inline(always)]
    fn forward_pair<const LAST: bool>(&self, a: &mut [u64], k: usize, l: usize) {
        let p = self.p;
        let two_p = 2 * p;
        for (j, block) in a.chunks_exact_mut(4 * l).enumerate() {
            let (w, w_shoup) = self.zetas[k + j];
            let (w0, w0_shoup) = self.zetas[2 * (k + j)];
            let (w1, w1_shoup) = self.zetas[2 * (k + j) + 1];
            let (b0, rest) = block.split_at_mut(l);
            let (b1, rest) = rest.split_at_mut(l);
            let (b2, b3) = rest.split_at_mut(l);
            for (((x0, x1), x2), x3) in b0.iter_mut().zip(b1).zip(b2).zip(b3) {
                let u0 = if *x0 >= two_p { *x0 - two_p } else { *x0 };
                let t0 = shoup_lazy(*x2, w, w_shoup, p);
                let u1 = if *x1 >= two_p { *x1 - two_p } else { *x1 };
                let t1 = shoup_lazy(*x3, w, w_shoup, p);
                let (y0, y2) = (u0 + t0, u0 + two_p - t0);
                let (y1, y3) = (u1 + t1, u1 + two_p - t1);
                let v0 = if y0 >= two_p { y0 - two_p } else { y0 };
                let s0 = shoup_lazy(y1, w0, w0_shoup, p);
                let v2 = if y2 >= two_p { y2 - two_p } else { y2 };
                let s2 = shoup_lazy(y3, w1, w1_shoup, p);
                if LAST {
                    *x0 = reduce_four_p(v0 + s0, p);
                    *x1 = reduce_four_p(v0 + two_p - s0, p);
                    *x2 = reduce_four_p(v2 + s2, p);
                    *x3 = reduce_four_p(v2 + two_p - s2, p);
                } else {
                    *x0 = v0 + s0;
                    *x1 = v0 + two_p - s0;
                    *x2 = v2 + s2;
                    *x3 = v2 + two_p - s2;
                }
            }
        }
    }
    /// [`NttPlan::forward_mont`] one stage at a time (radix 2).
    pub(crate) fn forward_mont_radix2(&self, a: &mut [u64]) {
        debug_assert!(a.len() == self.d && a.iter().all(|x| *x < self.p));
        let (p, d) = (self.p, self.d);
        let two_p = 2 * p;
        let half = d / 2;
        let (r, r_shoup) = self.r;
        let (z, z_shoup) = self.first;
        let (low, high) = a.split_at_mut(half);
        for (x, y) in low.iter_mut().zip(high.iter_mut()) {
            let u = shoup_lazy(*x, r, r_shoup, p);
            let t = shoup_lazy(*y, z, z_shoup, p);
            *x = u + t;
            *y = u + two_p - t;
        }
        let mut len = half / 2;
        let mut k = 2;
        while len >= 1 {
            for block in a.chunks_exact_mut(2 * len) {
                let (w, w_shoup) = self.zetas[k];
                k += 1;
                let (low, high) = block.split_at_mut(len);
                for (x, y) in low.iter_mut().zip(high.iter_mut()) {
                    let u = if *x >= two_p { *x - two_p } else { *x };
                    let t = shoup_lazy(*y, w, w_shoup, p);
                    *x = u + t;
                    *y = u + two_p - t;
                }
            }
            len /= 2;
        }
        for x in a.iter_mut() {
            *x = reduce_four_p(*x, p);
        }
    }
    /// The inverse of [`NttPlan::forward_mont`] that also removes the factor $`R`$: the
    /// Gentleman-Sande butterflies $`(x,y)\mapsto(x+y,(x-y)\zeta_k^{-1})`$ undo the stages in
    /// reverse order, each group with the inverse of its forward twiddle, which leaves the factor
    /// $`d`$, and the last stage multiplies by $`c=d^{-1}R^{-1}`$. For `d` residues in $`[0,p)`$,
    /// returned in $`[0,p)`$. Values stay below $`2p`$ between stages.
    pub(crate) fn inverse_mont(&self, a: &mut [u64]) {
        debug_assert!(a.len() == self.d && a.iter().all(|x| *x < self.p));
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if crate::simd::inverse_mont(&self.simd_tables(), a) {
            return;
        }
        self.inverse_mont_scalar(a);
    }
    /// [`NttPlan::inverse_mont`] in scalar code, the reference of the SIMD kernels: two stages per
    /// pass on aarch64, one elsewhere, as [`NttPlan::forward_mont_scalar`].
    pub(crate) fn inverse_mont_scalar(&self, a: &mut [u64]) {
        if cfg!(target_arch = "aarch64") {
            self.inverse_mont_radix4(a);
        } else {
            self.inverse_mont_radix2(a);
        }
    }
    /// [`NttPlan::inverse_mont`] two stages per pass (radix 4).
    pub(crate) fn inverse_mont_radix4(&self, a: &mut [u64]) {
        let (p, d) = (self.p, self.d);
        let two_p = 2 * p;
        let half = d / 2;
        // The stages below the last two at a time (radix 4): in a block of 4l, the butterflies
        // at distance l with the twiddles of its halves, then those at distance 2l with the
        // block's twiddle, on the same 4 values, the butterflies of the two radix-2 stages
        // ([`NttPlan::inverse_mont_radix2`]).
        let mut len = 1;
        while 2 * len < half {
            let k0 = d / (2 * len);
            let k1 = d / (4 * len);
            for (j, block) in a.chunks_exact_mut(4 * len).enumerate() {
                let (w0, w0_shoup) = self.inv_zetas[k0 + 2 * j];
                let (w1, w1_shoup) = self.inv_zetas[k0 + 2 * j + 1];
                let (w, w_shoup) = self.inv_zetas[k1 + j];
                let (b0, rest) = block.split_at_mut(len);
                let (b1, rest) = rest.split_at_mut(len);
                let (b2, b3) = rest.split_at_mut(len);
                for (((x0, x1), x2), x3) in b0.iter_mut().zip(b1).zip(b2).zip(b3) {
                    let s0 = *x0 + *x1;
                    let y0 = if s0 >= two_p { s0 - two_p } else { s0 };
                    let y1 = shoup_lazy(*x0 + two_p - *x1, w0, w0_shoup, p);
                    let s2 = *x2 + *x3;
                    let y2 = if s2 >= two_p { s2 - two_p } else { s2 };
                    let y3 = shoup_lazy(*x2 + two_p - *x3, w1, w1_shoup, p);
                    let t0 = y0 + y2;
                    let t1 = y1 + y3;
                    *x0 = if t0 >= two_p { t0 - two_p } else { t0 };
                    *x1 = if t1 >= two_p { t1 - two_p } else { t1 };
                    *x2 = shoup_lazy(y0 + two_p - y2, w, w_shoup, p);
                    *x3 = shoup_lazy(y1 + two_p - y3, w, w_shoup, p);
                }
            }
            len *= 4;
        }
        // An odd number of stages below the last leaves one, at distance d/4: with the last stage
        // in one pass.
        if len < half {
            self.inverse_last_pair(a);
        } else {
            self.inverse_last_stage(a);
        }
    }
    /// The stage of [`NttPlan::inverse_mont`] at distance $`d/4`$ (twiddles $`\zeta_2^{-1}`$
    /// and $`\zeta_3^{-1}`$) and its last stage, which multiplies by $`c`$ and
    /// $`\zeta_1^{-1}c`$, in one pass.
    fn inverse_last_pair(&self, a: &mut [u64]) {
        let p = self.p;
        let two_p = 2 * p;
        let l = self.d / 4;
        let (w0, w0_shoup) = self.inv_zetas[2];
        let (w1, w1_shoup) = self.inv_zetas[3];
        let (c, c_shoup) = self.scale;
        let (z, z_shoup) = self.last;
        let (b0, rest) = a.split_at_mut(l);
        let (b1, rest) = rest.split_at_mut(l);
        let (b2, b3) = rest.split_at_mut(l);
        for (((x0, x1), x2), x3) in b0.iter_mut().zip(b1).zip(b2).zip(b3) {
            let s0 = *x0 + *x1;
            let y0 = if s0 >= two_p { s0 - two_p } else { s0 };
            let y1 = shoup_lazy(*x0 + two_p - *x1, w0, w0_shoup, p);
            let s2 = *x2 + *x3;
            let y2 = if s2 >= two_p { s2 - two_p } else { s2 };
            let y3 = shoup_lazy(*x2 + two_p - *x3, w1, w1_shoup, p);
            *x0 = shoup_mul(y0 + y2, c, c_shoup, p);
            *x2 = shoup_mul(y0 + two_p - y2, z, z_shoup, p);
            *x1 = shoup_mul(y1 + y3, c, c_shoup, p);
            *x3 = shoup_mul(y1 + two_p - y3, z, z_shoup, p);
        }
    }
    /// [`NttPlan::inverse_mont`] one stage at a time (radix 2).
    pub(crate) fn inverse_mont_radix2(&self, a: &mut [u64]) {
        debug_assert!(a.len() == self.d && a.iter().all(|x| *x < self.p));
        let (p, d) = (self.p, self.d);
        let two_p = 2 * p;
        let half = d / 2;
        let mut len = 1;
        while len < half {
            let k0 = d / (2 * len);
            for (g, block) in a.chunks_exact_mut(2 * len).enumerate() {
                let (w, w_shoup) = self.inv_zetas[k0 + g];
                let (low, high) = block.split_at_mut(len);
                for (x, y) in low.iter_mut().zip(high.iter_mut()) {
                    let (u, v) = (*x, *y);
                    let s = u + v;
                    *x = if s >= two_p { s - two_p } else { s };
                    *y = shoup_lazy(u + two_p - v, w, w_shoup, p);
                }
            }
            len *= 2;
        }
        self.inverse_last_stage(a);
    }
    /// The last stage of [`NttPlan::inverse_mont`], which multiplies by $`c=d^{-1}R^{-1}`$.
    fn inverse_last_stage(&self, a: &mut [u64]) {
        let (p, half) = (self.p, self.d / 2);
        let two_p = 2 * p;
        let (c, c_shoup) = self.scale;
        let (z, z_shoup) = self.last;
        let (low, high) = a.split_at_mut(half);
        for (x, y) in low.iter_mut().zip(high.iter_mut()) {
            let (u, v) = (*x, *y);
            *x = shoup_mul(u + v, c, c_shoup, p);
            *y = shoup_mul(u + two_p - v, z, z_shoup, p);
        }
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
    /// $`x\bmod p`$ in $`[0,p)`$ for any $`x<2^{128}`$, by Barrett's method: with
    /// $`\mu=\lfloor(2^{128}-1)/p\rfloor`$ the quotient estimate $`\lfloor x\mu/2^{128}\rfloor`$ is
    /// at most two below $`\lfloor x/p\rfloor`$ and never above it, so the remainder is below
    /// $`3p<2^{64}`$ before two conditional subtractions.
    #[inline(always)]
    pub(crate) fn reduce_wide(&self, x: u128) -> u64 {
        let p = u128::from(self.p);
        let estimate = super::mod128::mul_wide(x, self.barrett).1;
        let mut r = x.wrapping_sub(estimate.wrapping_mul(p));
        if r >= p {
            r -= p;
        }
        if r >= p {
            r -= p;
        }
        r as u64
    }
}

#[cfg(test)]
mod oracle_tests;
