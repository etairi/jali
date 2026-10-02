//! Degree lowering, including quadratic forms.
use super::{Poly, PolyMat, Ring, SparsePolyMat};
use crate::Error;
use std::{collections::BTreeMap, sync::Arc};

/// Split $`a(X)=\sum_{i<k}X^i a_i(X^k)`$ into $`k=d'/d`$ polynomials.
pub fn split(poly: &Poly, target: Arc<Ring>) -> Result<Vec<Poly>, Error> {
    if poly.ring().modulus() != target.modulus() {
        return Err(Error::RingMismatch);
    }
    if !poly.ring().degree().is_multiple_of(target.degree()) {
        return Err(Error::Dimension);
    }
    let k = poly.ring().degree() / target.degree();
    Ok((0..k)
        .map(|i| {
            Poly::from_canonical(
                target.clone(),
                (0..target.degree())
                    .map(|j| poly.coefficients()[k * j + i])
                    .collect(),
            )
        })
        .collect())
}

/// Invert [`split`] with coefficient interleaving.
pub fn join(parts: &[Poly], target: Arc<Ring>) -> Result<Poly, Error> {
    let first = parts.first().ok_or(Error::Dimension)?;
    if parts.iter().any(|p| p.ring() != first.ring()) || first.ring().modulus() != target.modulus()
    {
        return Err(Error::RingMismatch);
    }
    if parts.len().checked_mul(first.ring().degree()) != Some(target.degree()) {
        return Err(Error::Dimension);
    }
    Ok(Poly::from_canonical(
        target.clone(),
        (0..target.degree())
            .map(|i| parts[i % parts.len()].coefficients()[i / parts.len()])
            .collect(),
    ))
}

/// Embed $`v(Z)`$ as $`v(X^K)`$ in `target`, of the same modulus and degree $`K\deg v`$: a ring
/// homomorphism, since $`(X^K)^{\deg v}=X^{\deg\,\mathrm{target}}=-1`$. Builds the witness of a
/// subring block (`statement::Statement::var_subring`). Fails with [`Error::RingMismatch`] for
/// another modulus and [`Error::Dimension`] when the degree of `target` is not a multiple.
pub fn embed(poly: &Poly, target: Arc<Ring>) -> Result<Poly, Error> {
    if poly.ring().modulus() != target.modulus() {
        return Err(Error::RingMismatch);
    }
    if !target.degree().is_multiple_of(poly.ring().degree()) {
        return Err(Error::Dimension);
    }
    let k = target.degree() / poly.ring().degree();
    let mut coefficients = vec![crypto_bigint::U256::ZERO; target.degree()];
    for (i, x) in poly.coefficients().iter().enumerate() {
        coefficients[k * i] = *x;
    }
    Ok(Poly::from_canonical(target, coefficients))
}

/// Block matrix representing multiplication by a public polynomial over the lower ring.
pub fn multiplication_matrix(poly: &Poly, target: Arc<Ring>) -> Result<PolyMat, Error> {
    let parts = split(poly, target.clone())?;
    let k = parts.len();
    let mut entries = Vec::with_capacity(k * k);
    for t in 0..k {
        for l in 0..k {
            entries.push(if t >= l {
                parts[t - l].clone()
            } else {
                parts[t + k - l].rotate(1)
            });
        }
    }
    PolyMat::new(target, k, k, entries)
}

/// Lower an upper-triangular quadratic form into one form per output component.
///
/// Each $`(a,b,i,j,m)`$ contributes $`Y^{\lfloor(i+j+m)/k\rfloor}R_{ab,m}`$.
/// Contributions below the diagonal are added to their mirror, including square terms.
pub fn quadratic(form: &SparsePolyMat, target: Arc<Ring>) -> Result<Vec<SparsePolyMat>, Error> {
    if form.ring().modulus() != target.modulus() {
        return Err(Error::RingMismatch);
    }
    if !form.ring().degree().is_multiple_of(target.degree()) {
        return Err(Error::Dimension);
    }
    let k = form.ring().degree() / target.degree();
    let dim = form.dimension().checked_mul(k).ok_or(Error::Dimension)?;
    if dim > u16::MAX as usize + 1 {
        return Err(Error::Dimension);
    }
    let mut forms = vec![BTreeMap::<(u16, u16), Poly>::new(); k];
    for ((a, b), p) in form.entries() {
        let parts = split(p, target.clone())?;
        for i in 0..k {
            for j in 0..k {
                for (m, part) in parts.iter().enumerate() {
                    let degree = i + j + m;
                    let r = (usize::from(a) * k + i) as u16;
                    let c = (usize::from(b) * k + j) as u16;
                    let key = (r.min(c), r.max(c));
                    let value = part.rotate((degree / k) as i64);
                    let entry = forms[degree % k]
                        .entry(key)
                        .or_insert_with(|| Poly::zero(target.clone()));
                    *entry = entry.add(&value)?;
                }
            }
        }
    }
    forms
        .into_iter()
        .map(|entries| {
            SparsePolyMat::new(
                target.clone(),
                dim,
                entries.into_iter().map(|((a, b), p)| (a, b, p)).collect(),
            )
        })
        .collect()
}
