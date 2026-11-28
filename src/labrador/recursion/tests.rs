//! The structured next statement against the $`K'`$ equations of BS23 Fig. 3 written out
//! directly: the honest next witness satisfies each of them, and on an arbitrary witness the
//! aggregation equals $`\sum_k\alpha_k e_k`$ of the explicit equations $`e_k`$.
use super::*;
use crate::wipe::Secret;
use crate::{
    labrador::{
        Policy,
        decompose::powers,
        level::{self, Ctx, evaluate_aggregate},
        params::lambda,
        sample::ChallengeSet,
        test_rng::TestRng,
        tests::instance,
    },
    math::Ring,
    transcript::{Protocol, Transcript},
};

fn ctx(ring: &Arc<Rq>) -> Ctx {
    Ctx {
        ring: ring.clone(),
        set: ChallengeSet::for_degree(64).unwrap(),
        eta_ring: Ring::new(i128::from(ring.modulus()), 64).unwrap(),
        policy: Policy::default(),
        pp_seed: [5u8; 32],
        lambda: lambda(ring.modulus()),
    }
}

/// The explicit equations $`e_k(w)`$ of the next statement, in its order.
fn explicit(next: &Next, w: &[Secret<Vec<u64>>]) -> Vec<Vec<u64>> {
    let ring = &next.ring;
    let (p, l) = (&next.params, &next.layout);
    let d = ring.degree();
    let (n, r, nn) = (l.n, l.r, l.n_next);
    // Unpack z0, z1 and v from the vectors.
    let get = |first: usize, x: usize| -> &[u64] {
        let (v, pos) = (first + x / nn, x % nn);
        &w[v][pos * d..(pos + 1) * d]
    };
    let mut z = vec![0u64; n * d];
    for x in 0..n {
        let (lo, hi) = (get(0, x), get(p.nu, x));
        let mut poly = lo.to_vec();
        ring.add_scaled_assign(&mut poly, hi, p.b % ring.modulus());
        z[x * d..(x + 1) * d].copy_from_slice(&poly);
    }
    let v: Vec<u64> = (0..l.m).flat_map(|y| get(2 * p.nu, y).to_vec()).collect();
    let poly = |y: usize| &v[y * d..(y + 1) * d];
    let p1 = powers(ring, p.b1, p.t1);
    let p2 = powers(ring, p.b2, p.t2);
    let t_of = |i: usize, row: usize| {
        let mut out = vec![0u64; d];
        for (digit, s) in p1.iter().enumerate() {
            ring.add_scaled_assign(&mut out, poly((i * p.t1 + digit) * p.kappa + row), *s);
        }
        out
    };
    let g_of = |pair: usize| {
        let mut out = vec![0u64; d];
        for (digit, s) in p2.iter().enumerate() {
            ring.add_scaled_assign(&mut out, poly(l.off_g + pair * p.t2 + digit), *s);
        }
        out
    };
    let h_of = |pair: usize| {
        let mut out = vec![0u64; d];
        for (digit, s) in p1.iter().enumerate() {
            ring.add_scaled_assign(&mut out, poly(l.off_h + pair * p.t1 + digit), *s);
        }
        out
    };
    let dense: Vec<Vec<u64>> = next.challenges.iter().map(|c| c.dense(ring)).collect();
    let mut eqs = Vec::new();
    // A z - sum c_i t_i.
    let mut a = Matrix::new(ring, &next.pp_seed, purpose::MATRIX_A, next.level);
    let mut entry = vec![0u64; d];
    for row in 0..p.kappa {
        let mut e = vec![0u64; d];
        for x in 0..n {
            a.next_poly(&mut entry).unwrap();
            ring.add_assign(&mut e, &ring.poly_mul(&entry, &z[x * d..(x + 1) * d]));
        }
        for (i, c) in dense.iter().enumerate() {
            ring.sub_assign(&mut e, &ring.poly_mul(c, &t_of(i, row)));
        }
        eqs.push(e);
    }
    // (B | C) v1 - u1 and D v2 - u2.
    for (purpose, rows, range, u) in [
        (purpose::MATRIX_B, p.kappa1, 0..l.off_h, &next.u1),
        (purpose::MATRIX_D, p.kappa2, l.off_h..l.m, &next.u2),
    ] {
        let mut m = Matrix::new(ring, &next.pp_seed, purpose, next.level);
        for row in 0..rows {
            let mut e = vec![0u64; d];
            for y in range.clone() {
                m.next_poly(&mut entry).unwrap();
                ring.add_assign(&mut e, &ring.poly_mul(&entry, poly(y)));
            }
            ring.sub_assign(&mut e, &u[row * d..(row + 1) * d]);
            eqs.push(e);
        }
    }
    // <z, z> - sum w c_i c_j g_ij, <phi_z, z> - sum c_i c_j h_ij, sum a_ij g_ij + sum h_ii - b.
    let zz = ring.inner_hat(&ring.ntt_many(&z), &ring.ntt_many(&z));
    let pz = ring.inner_hat(&ring.ntt_many(&next.phi_z), &ring.ntt_many(&z));
    let (mut eg, mut eh) = (zz, pz);
    let mut ef = vec![0u64; d];
    for i in 0..r {
        for j in i..r {
            let pair = pair_index(i, j, r);
            let cc = ring.poly_mul(&dense[i], &dense[j]);
            let w = if i == j { 1 } else { 2 };
            ring.add_scaled_assign(
                &mut eg,
                &ring.poly_mul(&cc, &g_of(pair)),
                ring.modulus() - w,
            );
            ring.sub_assign(&mut eh, &ring.poly_mul(&cc, &h_of(pair)));
            if i == j {
                ring.add_assign(&mut ef, &h_of(pair));
            }
        }
    }
    for (i, j, a) in &next.quad {
        ring.add_assign(&mut ef, &ring.poly_mul(a, &g_of(pair_index(*i, *j, r))));
    }
    ring.sub_assign(&mut ef, &next.constant);
    eqs.extend([eg, eh, ef]);
    eqs
}

#[test]
fn the_next_statement_is_the_verification_equations() {
    let mut rng = TestRng::new(20);
    let (statement, witness) = instance(&mut rng, 3, 8, 3, 2, 2);
    let ring = statement.ring().clone();
    let ctx = ctx(&ring);
    let mut t = Transcript::new(Protocol::Labrador, b"test").unwrap();
    let mut out = Vec::new();
    let (next, wit) = level::prove(
        &ctx,
        &statement,
        &witness.vectors,
        0,
        &mut t,
        &mut out,
        None,
    )
    .unwrap();
    // The honest next witness satisfies every equation.
    let eqs = explicit(&next, &wit);
    assert_eq!(eqs.len(), next.full_count());
    for (k, e) in eqs.iter().enumerate() {
        assert!(e.iter().all(|x| *x == 0), "equation {k}");
    }
    // On an arbitrary witness, the aggregation is the combination of the equations.
    let d = ring.degree();
    let q = ring.modulus();
    let shape = next.shape();
    let random: Vec<Secret<Vec<u64>>> = (0..shape.r)
        .map(|_| Secret::new((0..shape.n * d).map(|_| rng.below(q)).collect()))
        .collect();
    let alpha: Vec<Vec<u64>> = (0..next.full_count())
        .map(|_| (0..d).map(|_| rng.below(q)).collect())
        .collect();
    let agg = next.aggregate(&alpha, &[]).unwrap();
    let hats: Vec<Vec<u64>> = random.iter().map(|v| ring.ntt_many(v)).collect();
    let got = evaluate_aggregate(&ring, &agg, &hats);
    let mut want = vec![0u64; d];
    for (a, e) in alpha.iter().zip(explicit(&next, &random)) {
        ring.add_assign(&mut want, &ring.poly_mul(a, &e));
    }
    // The explicit equations read only the coefficients that carry values, so random values
    // in the padding of v must not change the aggregation either.
    assert_eq!(got, want);
}
