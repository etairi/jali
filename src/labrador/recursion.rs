//! The statement of the next level (BS23 §5.3): the verification equations of a level, which
//! its last message $`(z,\hat t,\hat g,\hat h)`$ satisfies, kept in structured form.
//!
//! The next witness has $`r'=2\nu+\mu`$ vectors of $`n'`$ polynomials: the digits
//! $`z=z_0+bz_1`$ split into $`\nu`$ parts each (vectors $`0..\nu`$ and $`\nu..2\nu`$), then
//! $`v=\hat t\,\|\,\hat g\,\|\,\hat h`$ split into $`\mu`$ parts, all zero-padded. In
//! $`v`$, digit $`m`$ of row $`k`$ of $`t_i`$ is at $`(it_1+m)\kappa+k`$, digit $`m`$ of
//! $`g_{ij}`$ ($`i\le j`$) at $`o_g+pt_2+m`$ and digit $`m`$ of $`h_{ij}`$ at $`o_h+pt_1+m`$,
//! for the pair index $`p`$. With $`g_{ij}=\langle s_i,s_j\rangle`$ and
//! $`h_{ii}=\langle\varphi_i,s_i\rangle`$, $`h_{ij}=\langle\varphi_i,s_j\rangle+\langle
//! \varphi_j,s_i\rangle`$ for $`i<j`$ (twice the $`h_{ij}`$ of BS23, which halves them),
//! the $`K'=\kappa+\kappa_1+\kappa_2+3`$ equations are, in order:
//! * $`Az-\sum_ic_it_i=0`$ ($`\kappa`$ rows),
//! * $`(B\mid C)(\hat t\|\hat g)-u_1=0`$ ($`\kappa_1`$ rows) and $`D\hat h-u_2=0`$
//!   ($`\kappa_2`$ rows),
//! * $`\langle z,z\rangle-\sum_{i\le j}w_{ij}c_ic_jg_{ij}=0`$ with $`w_{ii}=1`$, $`w_{ij}=2`$,
//! * $`\langle\varphi_z,z\rangle-\sum_{i\le j}c_ic_jh_{ij}=0`$ with
//!   $`\varphi_z=\sum_ic_i\varphi_i`$,
//! * $`\sum_{i\le j}a_{ij}g_{ij}+\sum_ih_{ii}-b=0`$,
//!
//! where $`t_i`$, $`g_{ij}`$, $`h_{ij}`$ are the recompositions of their digits and
//! $`(a_{ij},\varphi_i,b)`$ the level's aggregated function.
use super::{
    decompose::{decompose, powers},
    params::{Layout, LevelParams, Shape, pair_index},
    ring::{Rq, Sparse},
    sample::{Matrix, purpose},
    statement::{Aggregate, Relation},
};
use crate::Error;
use crate::wipe::Secret;
use std::sync::Arc;

/// The next statement, produced by the level of index `level`.
pub(crate) struct Next {
    pub(crate) ring: Arc<Rq>,
    pub(crate) pp_seed: [u8; 32],
    pub(crate) level: u32,
    pub(crate) params: LevelParams,
    pub(crate) layout: Layout,
    pub(crate) challenges: Vec<Sparse>,
    pub(crate) u1: Vec<u64>,
    pub(crate) u2: Vec<u64>,
    /// $`\varphi_z=\sum_ic_i\varphi_i`$, $`n`$ polynomials.
    pub(crate) phi_z: Vec<u64>,
    /// The level's aggregated quadratic coefficients $`(i,j,a_{ij})`$, $`i\le j`$, and its
    /// aggregated constant.
    pub(crate) quad: Vec<(usize, usize, Vec<u64>)>,
    pub(crate) constant: Vec<u64>,
}

/// The vector and polynomial position of polynomial `x` of a part split into pieces of
/// `n_next`, the first piece being vector `first`.
#[inline]
fn slot(first: usize, x: usize, n_next: usize) -> (usize, usize) {
    (first + x / n_next, x % n_next)
}

/// The next witness: $`z`$ ($`n`$ polynomials) decomposed in base $`b`$ and $`v`$
/// ($`m`$ polynomials), placed by the layout.
pub(crate) fn next_witness(
    ring: &Rq,
    params: &LevelParams,
    layout: &Layout,
    z: &[u64],
    v: &[u64],
) -> Vec<Secret<Vec<u64>>> {
    let d = ring.degree();
    let n_next = layout.n_next;
    let mut out: Vec<Secret<Vec<u64>>> = (0..layout.r_next)
        .map(|_| Secret::new(vec![0u64; n_next * d]))
        .collect();
    let digits = Secret::new(decompose(ring, z, params.b, 2));
    for (digit, part) in digits.iter().enumerate() {
        for (x, poly) in part.chunks_exact(d).enumerate() {
            let (vector, pos) = slot(digit * params.nu, x, n_next);
            out[vector][pos * d..(pos + 1) * d].copy_from_slice(poly);
        }
    }
    for (y, poly) in v.chunks_exact(d).enumerate() {
        let (vector, pos) = slot(2 * params.nu, y, n_next);
        out[vector][pos * d..(pos + 1) * d].copy_from_slice(poly);
    }
    out
}

impl Next {
    fn k_total(&self) -> usize {
        self.params.kappa + self.params.kappa1 + self.params.kappa2 + 3
    }

    /// `acc[x] += w ⊙ entry` over the transforms of a public matrix's row of `cols` entries.
    fn accumulate_row(
        &self,
        matrix: &mut Matrix,
        weight: &[u64],
        acc: &mut [u64],
        cols: usize,
    ) -> Result<(), Error> {
        let ring = &self.ring;
        let (d, k) = (ring.degree(), ring.hat_len());
        let w = ring.ntt(weight);
        let mut entry = vec![0u64; d];
        let mut entry_hat = vec![0u64; k];
        for x in 0..cols {
            matrix.next_poly(&mut entry)?;
            ring.ntt_into(&entry, &mut entry_hat);
            ring.mac(&mut acc[x * k..(x + 1) * k], &w, &entry_hat);
        }
        Ok(())
    }
}

impl Relation for Next {
    fn shape(&self) -> Shape {
        Shape {
            r: self.layout.r_next,
            n: self.layout.n_next,
            norm_sq: self.params.next_norm_sq,
        }
    }

    fn full_count(&self) -> usize {
        self.k_total()
    }

    fn constant_term_count(&self) -> usize {
        0
    }

    fn constant_term_constants(&self) -> Vec<u64> {
        Vec::new()
    }

    fn constant_term_values(&self, _: &[Vec<u64>], _: &[Vec<u64>]) -> Vec<Vec<u64>> {
        Vec::new()
    }

    fn aggregate(&self, alpha: &[Vec<u64>], gamma: &[Vec<u64>]) -> Result<Aggregate, Error> {
        let (ring, p, l) = (&self.ring, &self.params, &self.layout);
        if alpha.len() != self.k_total() || !gamma.is_empty() {
            return Err(Error::Dimension);
        }
        let (d, k, q) = (ring.degree(), ring.hat_len(), ring.modulus());
        let (kappa, k1, k2) = (p.kappa, p.kappa1, p.kappa2);
        let alpha_a = &alpha[..kappa];
        let alpha_b = &alpha[kappa..kappa + k1];
        let alpha_d = &alpha[kappa + k1..kappa + k1 + k2];
        let (alpha_g, alpha_h, alpha_f) = (
            &alpha[kappa + k1 + k2],
            &alpha[kappa + k1 + k2 + 1],
            &alpha[kappa + k1 + k2 + 2],
        );
        let (nu, r_next, n_next) = (p.nu, l.r_next, l.n_next);
        debug_assert!(nu <= r_next);
        let b = p.b % q;
        let scaled = |poly: &[u64], s: u64| {
            let mut out = poly.to_vec();
            ring.scale_assign(&mut out, s);
            out
        };

        // <z, z> = sum over parts of <z0, z0> + 2b <z0, z1> + b^2 <z1, z1>.
        let mut quad = Vec::with_capacity(3 * nu);
        for part in 0..nu {
            quad.push((part, part, alpha_g.clone()));
            quad.push((part, nu + part, scaled(alpha_g, ring.mul(2, b))));
            quad.push((nu + part, nu + part, scaled(alpha_g, ring.mul(b, b))));
        }

        let mut phi = vec![vec![0u64; n_next * d]; r_next];

        // The coefficient vector of z: sum_k alpha_k A_k + alpha_h phi_z.
        let mut z_hat = vec![0u64; l.n * k];
        let mut a = Matrix::new(ring, &self.pp_seed, purpose::MATRIX_A, self.level);
        for row in alpha_a {
            self.accumulate_row(&mut a, row, &mut z_hat, l.n)?;
        }
        let wh = ring.ntt(alpha_h);
        for (x, poly) in self.phi_z.chunks_exact(d).enumerate() {
            ring.mac(&mut z_hat[x * k..(x + 1) * k], &wh, &ring.ntt(poly));
        }
        let mut poly = vec![0u64; d];
        for x in 0..l.n {
            ring.intt_into(&mut z_hat[x * k..(x + 1) * k], &mut poly);
            let (vector, pos) = slot(0, x, n_next);
            phi[vector][pos * d..(pos + 1) * d].copy_from_slice(&poly);
            ring.scale_assign(&mut poly, b);
            phi[nu + vector][pos * d..(pos + 1) * d].copy_from_slice(&poly);
        }

        // The coefficient vector of v: the outer commitment rows, then the terms of the
        // amortization and garbage equations.
        let mut v_hat = vec![0u64; l.m * k];
        let mut bc = Matrix::new(ring, &self.pp_seed, purpose::MATRIX_B, self.level);
        for row in alpha_b {
            self.accumulate_row(&mut bc, row, &mut v_hat[..l.off_h * k], l.off_h)?;
        }
        let mut dm = Matrix::new(ring, &self.pp_seed, purpose::MATRIX_D, self.level);
        for row in alpha_d {
            self.accumulate_row(&mut dm, row, &mut v_hat[l.off_h * k..], l.m - l.off_h)?;
        }
        let mut v_phi = vec![0u64; l.m * d];
        for (h, o) in v_hat.chunks_exact_mut(k).zip(v_phi.chunks_exact_mut(d)) {
            ring.intt_into(h, o);
        }
        let p1 = powers(ring, p.b1, p.t1);
        let p2 = powers(ring, p.b2, p.t2);
        let mut term = vec![0u64; d];
        for (i, c) in self.challenges.iter().enumerate() {
            for (row, a_row) in alpha_a.iter().enumerate() {
                term.fill(0);
                ring.mul_sparse_add(c, a_row, &mut term);
                for (digit, scale) in p1.iter().enumerate() {
                    let y = (i * p.t1 + digit) * kappa + row;
                    ring.add_scaled_assign(&mut v_phi[y * d..(y + 1) * d], &term, ring.neg(*scale));
                }
            }
        }
        let r = l.r;
        let mut a_of: Vec<Option<&Vec<u64>>> = vec![None; l.pairs];
        for (i, j, a) in &self.quad {
            a_of[pair_index(*i, *j, r)] = Some(a);
        }
        let one = {
            let mut one = vec![0u64; d];
            one[0] = 1;
            one
        };
        for (i, c_i) in self.challenges.iter().enumerate() {
            let ci = c_i.dense(ring);
            for (j, c_j) in self.challenges.iter().enumerate().skip(i) {
                let pair = pair_index(i, j, r);
                let mut cc = vec![0u64; d];
                ring.mul_sparse_add(c_j, &ci, &mut cc);
                let w = if i == j { 1 } else { 2 };
                // g digits: alpha_f a_ij - w alpha_g c_i c_j.
                let mut g_term = match a_of[pair] {
                    Some(a) => ring.poly_mul(alpha_f, a),
                    None => vec![0u64; d],
                };
                ring.add_scaled_assign(&mut g_term, &ring.poly_mul(alpha_g, &cc), q - w);
                for (digit, scale) in p2.iter().enumerate() {
                    let y = l.off_g + pair * p.t2 + digit;
                    ring.add_scaled_assign(&mut v_phi[y * d..(y + 1) * d], &g_term, *scale);
                }
                // h digits: [i = j] alpha_f - alpha_h c_i c_j.
                let mut h_term = ring.poly_mul(alpha_h, &cc);
                ring.scale_assign(&mut h_term, q - 1);
                if i == j {
                    ring.add_assign(&mut h_term, &ring.poly_mul(alpha_f, &one));
                }
                for (digit, scale) in p1.iter().enumerate() {
                    let y = l.off_h + pair * p.t1 + digit;
                    ring.add_scaled_assign(&mut v_phi[y * d..(y + 1) * d], &h_term, *scale);
                }
            }
        }
        for (y, poly) in v_phi.chunks_exact(d).enumerate() {
            let (vector, pos) = slot(2 * nu, y, n_next);
            phi[vector][pos * d..(pos + 1) * d].copy_from_slice(poly);
        }

        let mut constant = ring.poly_mul(alpha_f, &self.constant);
        for (row, w) in alpha_b.iter().enumerate() {
            ring.add_assign(
                &mut constant,
                &ring.poly_mul(w, &self.u1[row * d..(row + 1) * d]),
            );
        }
        for (row, w) in alpha_d.iter().enumerate() {
            ring.add_assign(
                &mut constant,
                &ring.poly_mul(w, &self.u2[row * d..(row + 1) * d]),
            );
        }
        Ok(Aggregate {
            quad,
            phi,
            constant,
        })
    }
}

#[cfg(test)]
mod tests;
