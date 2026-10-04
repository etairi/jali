//! The Bernoulli trial of the base Gaussian sampler, decided from a 64-bit word when an integer
//! enclosure of the cutoff allows.
//!
//! **The trial.** A candidate of the sampler at width exponent $`t\ge1`$ is accepted when a
//! uniform 192-bit integer $`V`$ is below the cutoff $`C`$ that `reject::exp_negative` computes
//! for $`x=n/d`$, $`n=400(a^2-v^2)`$, $`d=2\cdot961\cdot4^t`$, that is
//! $`x=200(a-v)(a+v)/(961\cdot4^t)`$; the sampler's operands $`a`$ and $`v`$ are multiples of
//! its scale $`2^{t-1}`$ up to the offset, and $`t`$ is the width exponent itself. The argument
//! below needs only (A) $`|C-2^{192}e^{-x}|<2^{128}`$, which follows from how $`C`$ is
//! computed, for $`x<16`$ (the sampler's inputs have $`x\le5850/961`$). As
//! $`x_0=\lfloor2^{192}x\rfloor<2^{196}`$, it is halved $`s\le7`$ times, with floors, to
//! $`X\le2^{189}`$, so $`2^{192}y-1<X\le2^{192}y`$ for $`y=x/2^s`$. At most 80 Taylor terms at
//! $`X/2^{192}\le1/8`$ are summed, each floored term below the exact one by less than $`8/7`$
//! (the recurrence of step 4 below), and the series beyond them is below one unit; with one
//! unit for $`X`$ ($`e^{-y}`$ is 1-Lipschitz), the sum is within 94 units of
//! $`2^{192}e^{-y}`$. Each of the $`s`$ floored squarings at most doubles the error and adds 2,
//! so $`|C-2^{192}e^{-x}|\le2^7\cdot96-2=12286`$. (The function's own documentation states
//! $`2^{22}`$.)
//!
//! **The decision.** Write $`V=h2^{128}+l`$ with $`h<2^{64}`$ (bytes 16 to 23 of the uniform
//! value) and $`l<2^{128}`$. [`exp_bounds`] returns integers $`L\le2^{63}e^{-x}\le U`$. Then
//! - if $`h+2\le2L`$: $`V<(h+1)2^{128}\le L2^{129}-2^{128}\le2^{192}e^{-x}-2^{128}<C`$ by (A),
//!   so the exact test accepts;
//! - if $`h\ge2U+1`$: $`V\ge h2^{128}\ge U2^{129}+2^{128}\ge2^{192}e^{-x}+2^{128}>C`$ by (A),
//!   so the exact test rejects;
//! - otherwise the caller runs the exact test.
//!
//! The decisions therefore equal the exact test's whenever (A) holds. For $`h=\lfloor
//! C/2^{128}\rfloor`$ no decision is made: (A) and $`L2^{129}\le2^{192}e^{-x}`$ give
//! $`2L<C/2^{128}+1<h+2`$, and $`U2^{129}\ge2^{192}e^{-x}>C-2^{128}`$ gives $`2U\ge h`$.
//! Nothing is read from the stream here: the 24 uniform bytes are always read, so outputs
//! and consumption are those of the exact sampler.
//!
//! **The enclosure, by construction.** Every step below rounds towards the side of the bound it
//! computes and is monotone, so no error analysis of the result is needed:
//! 1. $`Z^\pm`$, the floor and ceiling of $`(a-v)(a+v)2^{60}/4^t`$, from the exact product.
//! 2. $`X^-=\lfloor Z^-K^-/2^{60}\rfloor\le x2^{64}\le\lceil Z^+K^+/2^{60}\rceil=X^+`$ with
//!    $`K^-=\lfloor200\cdot2^{64}/961\rfloor`$ and $`K^+=K^-+1`$ (961 does not divide
//!    $`200\cdot2^{64}`$).
//! 3. The least $`s\ge0`$ with $`X^+<2^{61+s}`$, and $`Y^-=\lfloor X^-/2^s\rfloor`$,
//!    $`Y^+=\lceil X^+/2^s\rceil`$: $`y=x/2^s\in[y^-,y^+]=[Y^-,Y^+]/2^{64}`$ with
//!    $`y^+\le1/8`$, and $`e^{-x}=(e^{-y})^{2^s}`$.
//! 4. Terms $`D_0=2^{63}`$, $`D_k=\lfloor D_{k-1}Y^-/(2^{64}k)\rfloor`$ for $`k\le13`$. With
//!    $`t_k=(y^-)^k/k!`$, $`\varepsilon_k=2^{63}t_k-D_k`$ satisfies $`\varepsilon_0=0`$ and
//!    $`\varepsilon_k=\varepsilon_{k-1}y^-/k+\mathrm{frac}(D_{k-1}y^-/k)\in[0,
//!    \varepsilon_{k-1}/8+1)`$, so $`0\le\varepsilon_k<8/7<2`$. Taylor's theorem with the
//!    Lagrange remainder gives $`S_{13}(y)\le e^{-y}\le S_{12}(y)`$ for $`y\ge0`$, where
//!    $`S_N(y)=\sum_{k\le N}(-1)^ky^k/k!`$. As $`2^{63}S_N(y^-)=\sum_{k\le N}(-1)^k(D_k+
//!    \varepsilon_k)`$ and there are seven even and seven odd $`k\le13`$, each
//!    $`\varepsilon_k<2`$:
//!    $`U_y=\sum_{k\le12}(-1)^kD_k+14\ge2^{63}e^{-y^-}`$ and
//!    $`\sum_{k\le13}(-1)^kD_k-14\le2^{63}e^{-y^-}`$. As $`e^{-y}`$ is 1-Lipschitz on
//!    $`y\ge0`$, $`L_y=\sum_{k\le13}(-1)^kD_k-14-\lceil(Y^+-Y^-)/2\rceil\le2^{63}e^{-y^+}`$.
//! 5. $`s`$ squarings, $`L\gets\lfloor L^2/2^{63}\rfloor`$ and
//!    $`U\gets\lceil U^2/2^{63}\rceil`$, keep $`L\le2^{63}(e^{-y})^{2^s}\le U`$.
//!
//! The sampler's inputs have $`x=50(1-c)(2K+1-c)/961`$ for sign 1 and $`x=50c(2K+c)/961`$ for
//! sign 0, with base sample $`K\le58`$ and $`c=u/2^{t-1}\in[0,1)`$, so $`x\le5850/961<6.09`$
//! (equality for base sample 58, sign 1 and offset 0) and $`s\le6`$; inputs outside the ranges
//! that the integer types cover give `None`, never a decision. The width $`U-L`$ only decides
//! how often the exact test runs (for $`2(U-L)+2`$ of the $`2^{64}`$ values of $`h`$), not
//! whether a decision is right.

/// $`\lfloor200\cdot2^{64}/961\rfloor`$; 961 does not divide $`200\cdot2^{64}`$.
const K_LOW: u128 = (200 << 64) / 961;
const K_HIGH: u128 = K_LOW + 1;
/// The last Taylor term, odd so that the partial sums bracket the exponential.
const TERMS: u64 = 13;
/// $`2\cdot\lceil(\mathrm{TERMS}+1)/2\rceil`$: the rounding slack of the partial sums.
const SLACK: u128 = 14;

/// `Some(V < C)` when the top word `high` of the uniform value decides the exact test, `None`
/// otherwise; for `a >= v`, the operands of the sampler. See the module documentation.
pub(crate) fn decide(a: u128, v: u128, t: u32, high: u64) -> Option<bool> {
    let [lower, upper] = exp_bounds(a, v, t)?;
    let h = u128::from(high);
    if h + 2 <= 2 * lower {
        Some(true)
    } else if h > 2 * upper {
        Some(false)
    } else {
        None
    }
}

/// $`[L,U]`$ with $`L\le2^{63}\exp(-200(a-v)(a+v)/(961\cdot4^t))\le U`$, or `None` when
/// $`Z^+\ge2^{66}`$ (an exponent of more than about 13; the sampler's stay below 6.09) or an
/// intermediate would not fit. Then $`X^+<2^{68}`$, so at most seven squarings.
pub(crate) fn exp_bounds(a: u128, v: u128, t: u32) -> Option<[u128; 2]> {
    let (x_low, x_high) = x_bounds(a, v, t)?;
    let s = (128 - x_high.leading_zeros()).saturating_sub(61);
    let y_low = x_low >> s;
    let y_high = x_high.div_ceil(1 << s);
    debug_assert!(y_low <= y_high && y_high <= 1 << 61);
    // Taylor terms at y_low; odd terms never exceed the even term before them.
    let y = y_low as u64;
    let mut term: u64 = 1 << 63;
    let (mut even, mut odd) = (u128::from(term), 0u128);
    for k in 1..=TERMS {
        term = ((u128::from(term) * u128::from(y)) >> 64) as u64 / k;
        if k % 2 == 0 {
            even += u128::from(term);
        } else {
            odd += u128::from(term);
        }
    }
    let odd_sum = even - odd;
    let mut upper = odd_sum + u128::from(term) + SLACK;
    let mut lower = odd_sum.checked_sub(SLACK + (y_high - y_low).div_ceil(2))?;
    for _ in 0..s {
        lower = (lower * lower) >> 63;
        upper = upper.checked_mul(upper)?.div_ceil(1 << 63);
    }
    Some([lower, upper])
}

/// Steps 1 and 2: $`X^-\le x2^{64}\le X^+`$, or `None` when $`Z^+\ge2^{66}`$ or an intermediate
/// would not fit.
#[inline]
fn x_bounds(a: u128, v: u128, t: u32) -> Option<(u128, u128)> {
    let (z_low, z_high) = scaled_square_difference(a.checked_sub(v)?, a.checked_add(v)?, t)?;
    if z_high >= 1 << 66 {
        return None;
    }
    let x_low = (z_low * K_LOW) >> 60;
    let x_high = (z_high * K_HIGH).div_ceil(1 << 60);
    Some((x_low, x_high))
}

/// $`\lfloor ms2^{60}/4^t\rfloor`$ and $`\lceil ms2^{60}/4^t\rceil`$, or `None` above
/// $`2^{127}`$.
fn scaled_square_difference(m: u128, s: u128, t: u32) -> Option<(u128, u128)> {
    let twice = 2 * t;
    match m.checked_mul(s) {
        Some(w) if twice <= 60 => {
            let z = w.checked_mul(1 << (60 - twice))?;
            Some((z, z))
        }
        Some(w) => {
            let shift = twice - 60;
            if shift >= 128 {
                return Some((0, u128::from(w != 0)));
            }
            let z = w >> shift;
            Some((z, z + u128::from(z << shift != w)))
        }
        None => {
            // The product needs more than 128 bits; t is then large (above 61 for sampler
            // inputs), so the quotient is a right shift.
            let w =
                crypto_bigint::U256::from_u128(m).wrapping_mul(&crypto_bigint::U256::from_u128(s));
            let shift = twice.checked_sub(60)?;
            let z = w.shr_vartime(shift);
            let low = crate::math::int::to_u128(&z).filter(|z| *z < 1 << 127)?;
            Some((low, low + u128::from(z.shl_vartime(shift) != w)))
        }
    }
}

#[cfg(test)]
mod step_tests;
