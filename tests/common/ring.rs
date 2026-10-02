//! Ring helpers over plain integer coefficient vectors, independent of the library's
//! arithmetic where a test uses them as an oracle.
use jali::{
    math::{Poly, Ring},
    quad::QuadEq,
};
use std::sync::Arc;

pub fn poly(ring: &Arc<Ring>, coefficients: Vec<i128>) -> Poly {
    Poly::new(ring.clone(), coefficients).unwrap()
}

/// The representative of $`x`$ modulo $`m`$ in $`(-m/2,m/2]`$.
pub fn centred(x: i128, m: i128) -> i128 {
    let r = x.rem_euclid(m);
    if r > m / 2 { r - m } else { r }
}

/// Squared Euclidean norm of the centred coefficients.
pub fn norm2(polys: &[Poly]) -> u64 {
    polys
        .iter()
        .flat_map(|p| {
            p.coefficients_i128()
                .unwrap()
                .iter()
                .map(|x| (x * x) as u64)
                .collect::<Vec<_>>()
        })
        .sum()
}

/// Split a ternary vector into binary vectors $`t=t^+-t^-`$.
pub fn split_ternary(t: &[i128]) -> (Vec<i128>, Vec<i128>) {
    (
        t.iter().map(|x| (*x).max(0)).collect(),
        t.iter().map(|x| (-*x).max(0)).collect(),
    )
}

/// Exact product in $`\mathbb Z[X]/(X^d+1)`$ (schoolbook, no reduction modulo q).
pub fn negacyclic(a: &[i128], b: &[i128]) -> Vec<i128> {
    let d = a.len();
    let mut out = vec![0i128; d];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            if i + j < d {
                out[i + j] += x * y;
            } else {
                out[i + j - d] -= x * y;
            }
        }
    }
    out
}

fn pow_mod(mut b: i64, mut e: i64, q: i64) -> i64 {
    let mut r = 1i64;
    b = b.rem_euclid(q);
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % q;
        }
        b = b * b % q;
        e >>= 1;
    }
    r
}
fn trim(a: &mut Vec<i64>) {
    while a.last() == Some(&0) {
        a.pop();
    }
}
/// Inverse of `a` in $`\mathbb Z_q[X]/(X^d+1)`$ for a prime $`q<2^{31}`$, by the extended
/// Euclidean algorithm over $`\mathbb F_q[X]`$; `None` if `a` is not invertible. Coefficients of
/// the result lie in $`[0,q)`$.
pub fn inverse(a: &[i128], q: i128) -> Option<Vec<i128>> {
    let q = q as i64;
    let d = a.len();
    let mut r0: Vec<i64> = vec![0; d + 1];
    r0[0] = 1;
    r0[d] = 1;
    let mut r1: Vec<i64> = a.iter().map(|x| (*x as i64).rem_euclid(q)).collect();
    trim(&mut r1);
    let (mut t0, mut t1): (Vec<i64>, Vec<i64>) = (vec![], vec![1]);
    while !r1.is_empty() {
        let mut rem = r0.clone();
        let lead_inverse = pow_mod(*r1.last().unwrap(), q - 2, q);
        let mut quotient = vec![0i64; rem.len().saturating_sub(r1.len()) + 1];
        while rem.len() >= r1.len() && !rem.is_empty() {
            let shift = rem.len() - r1.len();
            let c = rem.last().unwrap() * lead_inverse % q;
            quotient[shift] = c;
            for (i, x) in r1.iter().enumerate() {
                rem[i + shift] = (rem[i + shift] - c * x).rem_euclid(q);
            }
            trim(&mut rem);
        }
        let mut t2 = vec![0i64; (quotient.len() + t1.len()).max(t0.len())];
        t2[..t0.len()].copy_from_slice(&t0);
        for (i, a) in quotient.iter().enumerate() {
            for (j, b) in t1.iter().enumerate() {
                t2[i + j] = (t2[i + j] - a * b).rem_euclid(q);
            }
        }
        trim(&mut t2);
        r0 = std::mem::replace(&mut r1, rem);
        t0 = std::mem::replace(&mut t1, t2);
    }
    // r0 is a greatest common divisor: a is invertible iff it is a nonzero constant.
    if r0.len() != 1 {
        return None;
    }
    let c = pow_mod(r0[0], q - 2, q);
    let mut out = vec![0i128; d];
    for (i, x) in t0.iter().enumerate() {
        let (index, sign) = if i < d { (i, 1) } else { (i - d, -1) };
        out[index] = (out[index] + sign * (*x * c % q) as i128).rem_euclid(q as i128);
    }
    Some(out)
}

/// The integer value of a form at an integer witness, with an independent schoolbook product
/// on the centred coefficients of the form's own ring.
pub fn integer_value(form: &QuadEq, w: &[Vec<i128>]) -> Vec<i128> {
    let values = |p: &Poly| p.coefficients_i128().unwrap().to_vec();
    let mut out = values(&form.r0);
    let mut add = |v: Vec<i128>| out.iter_mut().zip(v).for_each(|(o, v)| *o += v);
    for (i, a) in form.r1.entries() {
        add(negacyclic(&values(a), &w[usize::from(i)]));
    }
    for ((i, j), a) in form.r2.entries() {
        let product = negacyclic(&w[usize::from(i)], &w[usize::from(j)]);
        add(negacyclic(&values(a), &product));
    }
    out
}

/// $`\lceil\sqrt x\rceil`$.
pub fn ceil_sqrt(x: u128) -> u128 {
    let r = x.isqrt();
    if r * r == x { r } else { r + 1 }
}

/// The compiler's bound $`F`$ on the integer value of `form`, recomputed from its formula
/// $`\|r_0\|_\infty+\sum_i\lceil\sqrt{\|r_{1,i}\|^2\beta_i}\rceil
/// +\sum_{i\le j}\|r_{2,ij}\|_1\lceil\sqrt{\beta_i\beta_j}\rceil`$, where $`\beta_i`$ is the
/// squared bound of the block of variable $`i`$ (for a binary block, its number of
/// coefficients) and the norms are of centred coefficients.
pub fn f_bound(form: &QuadEq, beta: &[u128]) -> u128 {
    let values = |p: &Poly| p.coefficients_i128().unwrap().to_vec();
    let mut f = values(&form.r0)
        .iter()
        .map(|x| x.unsigned_abs())
        .max()
        .unwrap();
    for (i, a) in form.r1.entries() {
        let norm: u128 = values(a).iter().map(|x| x.unsigned_abs().pow(2)).sum();
        f += ceil_sqrt(norm * beta[usize::from(i)]);
    }
    for ((i, j), a) in form.r2.entries() {
        let l1: u128 = values(a).iter().map(|x| x.unsigned_abs()).sum();
        f += l1 * ceil_sqrt(beta[usize::from(i)] * beta[usize::from(j)]);
    }
    f
}

/// A variable's bound for [`f_bound_linf`]: the squared bound of its block (for a binary block,
/// its number of coefficients), a bound on the absolute value of every coefficient, or such a
/// bound on a variable with at most `n` nonzero coefficients (a subring variable of degree `n`).
#[derive(Clone, Copy, Debug)]
pub enum VarBound {
    Squared(u128),
    Linf(u128),
    LinfIn(u128, u128),
}

/// The compiler's bound $`F`$ on the integer value of `form` when some variables are bounded in
/// $`\ell_\infty`$, recomputed from its formula: the terms of [`f_bound`] between exact-norm
/// and binary variables, and for $`y,y'`$ bounded by $`b,b'`$ in $`\ell_\infty`$ with $`n,n'`$
/// possibly nonzero coefficients ($`d`$, the degree of the form's ring, for `Linf`),
/// $`\|r\|_1b`$ for $`r\,y`$, $`\|r\|_1\lceil\sqrt{nB}\rceil b`$ for $`r\,y\,s`$ with $`s`$ in a
/// block of squared bound $`B`$, and $`\|r\|_1\lceil\sqrt{nn'}\rceil b\,b'`$ for $`r\,y\,y'`$
/// ($`\|r\|_1d\,b\,b'`$ at full degree). Panics from $`2^{128}`$ on.
pub fn f_bound_linf(form: &QuadEq, bound: &[VarBound]) -> u128 {
    let values = |p: &Poly| p.coefficients_i128().unwrap().to_vec();
    let l1 = |p: &Poly| values(p).iter().map(|x| x.unsigned_abs()).sum::<u128>();
    let d = form.r0.ring().degree() as u128;
    let mut f = values(&form.r0)
        .iter()
        .map(|x| x.unsigned_abs())
        .max()
        .unwrap();
    let mut add = |x: Option<u128>| f = f.checked_add(x.expect("below 2^128")).unwrap();
    // (bound, coefficients) of an l_inf variable.
    let linf = |b: VarBound| match b {
        VarBound::Linf(x) => Some((x, d)),
        VarBound::LinfIn(x, n) => Some((x, n)),
        VarBound::Squared(_) => None,
    };
    for (i, a) in form.r1.entries() {
        add(match (bound[usize::from(i)], linf(bound[usize::from(i)])) {
            (VarBound::Squared(b), _) => {
                let norm: u128 = values(a).iter().map(|x| x.unsigned_abs().pow(2)).sum();
                Some(ceil_sqrt(norm * b))
            }
            (_, Some((b, _))) => l1(a).checked_mul(b),
            _ => unreachable!(),
        });
    }
    for ((i, j), a) in form.r2.entries() {
        let (x, y) = (bound[usize::from(i)], bound[usize::from(j)]);
        let term = match (x, y, linf(x), linf(y)) {
            (VarBound::Squared(x), VarBound::Squared(y), _, _) => Some(ceil_sqrt(x * y)),
            (_, VarBound::Squared(s), Some((b, n)), _)
            | (VarBound::Squared(s), _, _, Some((b, n))) => ceil_sqrt(n * s).checked_mul(b),
            (_, _, Some((b, n)), Some((c, m))) => ceil_sqrt(n * m)
                .checked_mul(b)
                .and_then(|v| v.checked_mul(c)),
            _ => unreachable!(),
        };
        add(term.and_then(|t| t.checked_mul(l1(a))));
    }
    f
}
