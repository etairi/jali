//! The principal relation of BS23 §5.1: witness vectors $`s_1,\dots,s_r\in\mathcal R_q^n`$
//! with $`\sum_i\|s_i\|^2\le\beta^2`$ that satisfy dot-product constraints
//! $`f(s)=\sum_{i,j}a_{ij}\langle s_i,s_j\rangle+\sum_i\langle\varphi_i,s_i\rangle-b`$: each
//! constraint of the family $`\mathcal F`$ vanishes in $`\mathcal R_q`$, each of the family
//! $`\mathcal F'`$ has a vanishing constant coefficient. Here
//! $`\langle a,b\rangle=\sum_ka_kb_k\in\mathcal R_q`$, and
//! $`\mathrm{ct}(\langle\sigma_{-1}(a),b\rangle)`$ is the integer dot product of the
//! coefficient vectors, which constant-term constraints can use to express integer relations.
use super::{
    params::{Shape, pair_index},
    ring::Rq,
};
use crate::{Error, transcript::Transcript};
use std::sync::Arc;
use zeroize::Zeroizing;

/// A dot-product function $`f(s)=\sum a_{ij}\langle s_i,s_j\rangle+\sum_i\langle\varphi_i,
/// s_i\rangle-b`$. Polynomials are given by their $`d`$ canonical coefficients in $`[0,q)`$.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Constraint {
    /// Terms $`a\langle s_i,s_j\rangle`$ as $`(i,j,a)`$.
    quadratic: Vec<(usize, usize, Vec<u64>)>,
    /// Terms $`\langle\varphi,s_i[o..o+\ell]\rangle`$ as $`(i,o,\varphi)`$, with $`\varphi`$ the
    /// $`\ell d`$ coefficients of $`\ell`$ polynomials.
    linear: Vec<(usize, usize, Vec<u64>)>,
    /// $`b`$. A constant-term constraint uses only its constant coefficient.
    constant: Vec<u64>,
}

impl Constraint {
    /// A constraint $`f(s)=-b`$ with no terms yet, for the $`d`$ coefficients of $`b`$.
    pub fn new(constant: Vec<u64>) -> Self {
        Self {
            constant,
            ..Self::default()
        }
    }

    /// Add $`a\langle s_i,s_j\rangle`$.
    pub fn quadratic(mut self, i: usize, j: usize, a: Vec<u64>) -> Self {
        self.quadratic.push((i, j, a));
        self
    }

    /// Add $`\langle\varphi,s_i[o..o+\ell]\rangle`$ for the $`\ell d`$ coefficients of
    /// $`\varphi`$, starting at polynomial $`o`$ of $`s_i`$.
    pub fn linear(mut self, i: usize, offset: usize, phi: Vec<u64>) -> Self {
        self.linear.push((i, offset, phi));
        self
    }

    fn check(&self, ring: &Rq, r: usize, n: usize) -> Result<(), Error> {
        let (q, d) = (ring.modulus(), ring.degree());
        let canonical = |p: &[u64]| p.iter().all(|x| *x < q);
        if self.constant.len() != d || !canonical(&self.constant) {
            return Err(Error::Dimension);
        }
        for (i, j, a) in &self.quadratic {
            if *i >= r || *j >= r {
                return Err(Error::Index);
            }
            if a.len() != d || !canonical(a) {
                return Err(Error::Dimension);
            }
        }
        for (i, offset, phi) in &self.linear {
            if *i >= r {
                return Err(Error::Index);
            }
            if phi.is_empty() || phi.len() % d != 0 || !canonical(phi) {
                return Err(Error::Dimension);
            }
            if offset.checked_add(phi.len() / d).is_none_or(|end| end > n) {
                return Err(Error::Index);
            }
        }
        Ok(())
    }

    /// The canonical encoding absorbed into the transcript: counts and indices as `LE32`,
    /// coefficients as `LE64`. A constant-term constraint has one constant coefficient.
    fn encode(&self, out: &mut Vec<u8>, full: bool) {
        let le32 = |out: &mut Vec<u8>, x: usize| out.extend_from_slice(&(x as u32).to_le_bytes());
        let coeffs = |out: &mut Vec<u8>, p: &[u64]| {
            for x in p {
                out.extend_from_slice(&x.to_le_bytes());
            }
        };
        le32(out, self.quadratic.len());
        for (i, j, a) in &self.quadratic {
            le32(out, *i);
            le32(out, *j);
            coeffs(out, a);
        }
        le32(out, self.linear.len());
        for (i, offset, phi) in &self.linear {
            le32(out, *i);
            le32(out, *offset);
            le32(out, phi.len());
            coeffs(out, phi);
        }
        if full {
            coeffs(out, &self.constant);
        } else {
            coeffs(out, &self.constant[..1]);
        }
    }
}

/// A statement of the principal relation: the ring, the multiplicity $`r`$, the rank $`n`$,
/// the bound $`\beta^2`$ on $`\sum_i\|s_i\|^2`$, and the two families of constraints.
#[derive(Clone, Debug)]
pub struct Statement {
    ring: Arc<Rq>,
    r: usize,
    n: usize,
    norm_sq: u128,
    full: Vec<Constraint>,
    constant_term: Vec<Constraint>,
}

/// The largest witness of any level, $`2^{24}`$ coefficients.
pub const MAX_WITNESS_COEFFICIENTS: usize = 1 << 24;
/// The largest multiplicity of any level.
pub const MAX_MULTIPLICITY: usize = 1024;

impl Statement {
    /// A statement over `ring` with `multiplicity` vectors of `rank` polynomials and the bound
    /// `norm_bound_squared` on $`\sum_i\|s_i\|^2`$ (centred coefficients), with no constraints.
    pub fn new(
        ring: Arc<Rq>,
        multiplicity: usize,
        rank: usize,
        norm_bound_squared: u128,
    ) -> Result<Self, Error> {
        if multiplicity == 0
            || multiplicity > MAX_MULTIPLICITY
            || rank == 0
            || multiplicity
                .checked_mul(rank)
                .and_then(|x| x.checked_mul(ring.degree()))
                .is_none_or(|x| x > MAX_WITNESS_COEFFICIENTS)
        {
            return Err(Error::Dimension);
        }
        Ok(Self {
            ring,
            r: multiplicity,
            n: rank,
            norm_sq: norm_bound_squared,
            full: Vec::new(),
            constant_term: Vec::new(),
        })
    }

    /// Add a constraint that must vanish in $`\mathcal R_q`$.
    pub fn push(&mut self, constraint: Constraint) -> Result<(), Error> {
        constraint.check(&self.ring, self.r, self.n)?;
        self.full.push(constraint);
        Ok(())
    }

    /// Add a constraint whose constant coefficient must vanish. Only the constant coefficient
    /// of its $`b`$ matters.
    pub fn push_constant_term(&mut self, constraint: Constraint) -> Result<(), Error> {
        constraint.check(&self.ring, self.r, self.n)?;
        self.constant_term.push(constraint);
        Ok(())
    }

    /// The ring.
    pub fn ring(&self) -> &Arc<Rq> {
        &self.ring
    }

    /// The number of witness vectors.
    pub fn multiplicity(&self) -> usize {
        self.r
    }

    /// The number of polynomials of each witness vector.
    pub fn rank(&self) -> usize {
        self.n
    }

    pub(crate) fn shape(&self) -> Shape {
        Shape {
            r: self.r,
            n: self.n,
            norm_sq: self.norm_sq,
        }
    }

    /// Absorb the statement: $`q,d,r,n,\beta^2`$, then the full constraints and the
    /// constant-term constraints, each family preceded by its count.
    pub(crate) fn absorb(&self, transcript: &mut Transcript) -> Result<(), Error> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.ring.modulus().to_le_bytes());
        for x in [self.ring.degree(), self.r, self.n] {
            out.extend_from_slice(&(x as u32).to_le_bytes());
        }
        out.extend_from_slice(&self.norm_sq.to_le_bytes());
        for (family, full) in [(&self.full, true), (&self.constant_term, false)] {
            out.extend_from_slice(&(family.len() as u32).to_le_bytes());
            for c in family {
                c.encode(&mut out, full);
                if out.len() >= 1 << 16 {
                    transcript.absorb(&out);
                    out.clear();
                }
            }
        }
        transcript.absorb(&out);
        Ok(())
    }

    /// Whether `witness` satisfies the statement: [`Error::Witness`] if not, with no
    /// indication of which condition fails.
    pub fn check(&self, witness: &Witness) -> Result<(), Error> {
        let (ring, d) = (&self.ring, self.ring.degree());
        if witness.vectors.len() != self.r || witness.vectors.iter().any(|v| v.len() != self.n * d)
        {
            return Err(Error::Dimension);
        }
        let norm: u128 = witness.vectors.iter().map(|v| ring.norm_sq(v)).sum();
        if norm > self.norm_sq {
            return Err(Error::Witness);
        }
        let hats: Zeroizing<Vec<Vec<u64>>> =
            Zeroizing::new(witness.vectors.iter().map(|v| ring.ntt_many(v)).collect());
        let pairs = Zeroizing::new(all_pairs(ring, &hats));
        for c in &self.full {
            let mut value = evaluate(ring, c, &hats, &pairs, self.r);
            ring.sub_assign(&mut value, &c.constant);
            if value.iter().any(|x| *x != 0) {
                return Err(Error::Witness);
            }
        }
        for c in &self.constant_term {
            let value = evaluate(ring, c, &hats, &pairs, self.r);
            if value[0] != c.constant[0] {
                return Err(Error::Witness);
            }
        }
        Ok(())
    }
}

/// A witness: $`r`$ vectors of $`nd`$ canonical coefficients, wiped on drop.
#[derive(Clone)]
pub struct Witness {
    pub(crate) vectors: Vec<Zeroizing<Vec<u64>>>,
}

impl core::fmt::Debug for Witness {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Witness").finish_non_exhaustive()
    }
}

impl Witness {
    /// A witness from centred (or any) integer coefficients, one vector of $`nd`$ of them for
    /// each $`s_i`$, reduced modulo $`q`$.
    pub fn new(ring: &Rq, vectors: Vec<Vec<i64>>) -> Self {
        Self {
            vectors: vectors
                .into_iter()
                .map(|v| {
                    let v = Zeroizing::new(v);
                    Zeroizing::new(v.iter().map(|x| ring.lift(*x)).collect())
                })
                .collect(),
        }
    }
}

/// $`\langle s_i,s_j\rangle`$ for every pair $`i\le j`$, from the transforms.
pub(crate) fn all_pairs(ring: &Rq, hats: &[Vec<u64>]) -> Vec<Vec<u64>> {
    let r = hats.len();
    let mut out = Vec::with_capacity(r * (r + 1) / 2);
    for i in 0..r {
        for j in i..r {
            out.push(ring.inner_hat(&hats[i], &hats[j]));
        }
    }
    out
}

/// $`f(s)+b`$: the quadratic and linear terms of `c` on the witness with transforms `hats`
/// and pair products `pairs`.
fn evaluate(
    ring: &Rq,
    c: &Constraint,
    hats: &[Vec<u64>],
    pairs: &[Vec<u64>],
    r: usize,
) -> Vec<u64> {
    let (d, k) = (ring.degree(), ring.hat_len());
    let mut value = vec![0u64; d];
    for (i, j, a) in &c.quadratic {
        let (lo, hi) = if i <= j { (*i, *j) } else { (*j, *i) };
        ring.add_assign(&mut value, &ring.poly_mul(a, &pairs[pair_index(lo, hi, r)]));
    }
    let mut acc = vec![0u64; k];
    for (i, offset, phi) in &c.linear {
        for (x, poly) in phi.chunks_exact(d).enumerate() {
            let at = (offset + x) * k;
            ring.mac(&mut acc, &ring.ntt(poly), &hats[*i][at..at + k]);
        }
    }
    ring.add_assign(&mut value, &ring.intt(acc));
    value
}

/// An aggregated function $`F(s)=\sum_{i\le j}a_{ij}\langle s_i,s_j\rangle+\sum_i\langle
/// \varphi_i,s_i\rangle-b`$: `quad` holds the nonzero $`a_{ij}`$ as $`(i,j,a_{ij})`$ with
/// $`i\le j`$, each pair once, `phi` the $`r`$ vectors $`\varphi_i`$, `constant` $`b`$.
pub(crate) struct Aggregate {
    pub(crate) quad: Vec<(usize, usize, Vec<u64>)>,
    pub(crate) phi: Vec<Vec<u64>>,
    pub(crate) constant: Vec<u64>,
}

/// What a level needs of the relation it proves.
pub(crate) trait Relation {
    fn shape(&self) -> Shape;
    /// $`|\mathcal F|`$ and $`|\mathcal F'|`$.
    fn full_count(&self) -> usize;
    fn constant_term_count(&self) -> usize;
    /// The constant coefficients $`b'_0`$ of $`\mathcal F'`$.
    fn constant_term_constants(&self) -> Vec<u64>;
    /// The prover's values $`f'(s)+b'`$ of $`\mathcal F'`$, as full polynomials.
    fn constant_term_values(&self, hats: &[Vec<u64>], pairs: &[Vec<u64>]) -> Vec<Vec<u64>>;
    /// $`\sum_k\alpha_kf^{(k)}+\sum_l\gamma_l(f'^{(l)}+b'^{(l)})`$: the constants of
    /// $`\mathcal F'`$ are left out, since the prover's lifted polynomials replace them.
    fn aggregate(&self, alpha: &[Vec<u64>], gamma: &[Vec<u64>]) -> Result<Aggregate, Error>;
}

impl Relation for Statement {
    fn shape(&self) -> Shape {
        Statement::shape(self)
    }

    fn full_count(&self) -> usize {
        self.full.len()
    }

    fn constant_term_count(&self) -> usize {
        self.constant_term.len()
    }

    fn constant_term_constants(&self) -> Vec<u64> {
        self.constant_term.iter().map(|c| c.constant[0]).collect()
    }

    fn constant_term_values(&self, hats: &[Vec<u64>], pairs: &[Vec<u64>]) -> Vec<Vec<u64>> {
        self.constant_term
            .iter()
            .map(|c| evaluate(&self.ring, c, hats, pairs, self.r))
            .collect()
    }

    fn aggregate(&self, alpha: &[Vec<u64>], gamma: &[Vec<u64>]) -> Result<Aggregate, Error> {
        if alpha.len() != self.full.len() || gamma.len() != self.constant_term.len() {
            return Err(Error::Dimension);
        }
        let ring = &self.ring;
        let (d, k, r, n) = (ring.degree(), ring.hat_len(), self.r, self.n);
        let mut quad: Vec<(usize, usize, Vec<u64>)> = Vec::new();
        let mut phi_hat = vec![vec![0u64; n * k]; r];
        let mut touched = vec![false; r];
        let mut constant = vec![0u64; d];
        let weighted = self
            .full
            .iter()
            .zip(alpha)
            .chain(self.constant_term.iter().zip(gamma));
        for (index, (c, weight)) in weighted.enumerate() {
            if weight.iter().all(|x| *x == 0) {
                continue;
            }
            for (i, j, a) in &c.quadratic {
                let (lo, hi) = if i <= j { (*i, *j) } else { (*j, *i) };
                let product = ring.poly_mul(weight, a);
                match quad.iter_mut().find(|(x, y, _)| (*x, *y) == (lo, hi)) {
                    Some((_, _, sum)) => ring.add_assign(sum, &product),
                    None => quad.push((lo, hi, product)),
                }
            }
            if !c.linear.is_empty() {
                let w = ring.ntt(weight);
                for (i, offset, phi) in &c.linear {
                    touched[*i] = true;
                    for (x, poly) in phi.chunks_exact(d).enumerate() {
                        let at = (offset + x) * k;
                        ring.mac(&mut phi_hat[*i][at..at + k], &w, &ring.ntt(poly));
                    }
                }
            }
            if index < self.full.len() {
                ring.add_assign(&mut constant, &ring.poly_mul(weight, &c.constant));
            }
        }
        let phi = phi_hat
            .into_iter()
            .zip(touched)
            .map(|(mut hat, touched)| {
                let mut out = vec![0u64; n * d];
                if touched {
                    for (h, o) in hat.chunks_exact_mut(k).zip(out.chunks_exact_mut(d)) {
                        ring.intt_into(h, o);
                    }
                }
                out
            })
            .collect();
        Ok(Aggregate {
            quad,
            phi,
            constant,
        })
    }
}

#[cfg(test)]
impl Constraint {
    pub(crate) fn constant_mut(&mut self) -> &mut Vec<u64> {
        &mut self.constant
    }
}

#[cfg(test)]
impl Statement {
    pub(crate) fn full_mut(&mut self) -> &mut Vec<Constraint> {
        &mut self.full
    }
    pub(crate) fn constant_term_mut(&mut self) -> &mut Vec<Constraint> {
        &mut self.constant_term
    }
}
