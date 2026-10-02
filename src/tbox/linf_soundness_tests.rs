//! What a verified proof says about a variable bounded in $`\ell_\infty`$ ([`Norm::Linf`]): its
//! coefficients are rows of the approximate range proof, whose verifier bounds the extracted
//! value by $`E=2\cdot`$`z4_bound` (LNP22 Lemma 2.7), not by the declared $`\beta`$ or by
//! `linf_bound`. With the witness check skipped, a coefficient of $`\beta+1`$ proves and
//! verifies; one of $`E+1`$ does not: the prover's eight range attempts fail, as a projection of
//! the long coefficient makes rejection sampling or the bound `z4_bound` refuse each response (a
//! proof, if one came out, would have to pass the unmodified verifier). For the second case the
//! parameters allow a long bounded witness, so that the commitment, which bounds
//! $`\|s_1\|^2`$ by `alpha_squared`, does not refuse it first.
use super::*;
use crate::{
    math::{Ring, iso},
    params::TboxParams,
    statement::{Extraction, Norm, Statement as Source},
};

/// Over $`\mathbb Z_{13}`$ at degree 128: `s` (eight polynomials, $`\|s\|^2\le128`$) and `u`
/// ($`\ell_\infty\le1`$), with no constraint: the approximate range proof holds only the two
/// proof-ring polynomials of `u`.
fn source() -> Source {
    let mut st = Source::new(Ring::new(13, 128).unwrap());
    st.var("s", 8, Norm::L2Squared(128)).unwrap();
    st.var("u", 1, Norm::Linf(1)).unwrap();
    st
}

/// The possession parameters of `lifting_soundness_tests` for `source`: one exact-norm block
/// of 16 rows, the two rows of `u` at bound 1 and no messages, with the MLWE rank that the
/// shorter message part leaves and $`\sigma_4`$ as the parameter tool sizes it.
fn params() -> TboxParams {
    let mut p = crate::params::toy_d64();
    p.id = "linf-soundness-test-only".into();
    p.m1 = 18;
    p.m2 = 55;
    p.n_msis = 15;
    p.l = 0;
    p.alpha_squared = 256;
    p.n_bin = 0;
    p.l2_rows = vec![16];
    p.l2_bounds_squared = vec![128];
    p.n_prime = 2;
    p.linf_bound = 1;
    p.log_sigma = [14, 12, 10, 9];
    p.d_bits = 6;
    p.mlwe_rank = 28;
    p
}

/// `params` with room for a coefficient of about $`E`$ in the bounded witness: a larger
/// `alpha_squared`, $`\sigma_1`$ and MSIS rank, with the same $`\sigma_4`$ and so the same
/// $`E`$.
fn long_params(e: u128) -> TboxParams {
    let mut p = params();
    p.id = "linf-soundness-long-test-only".into();
    p.alpha_squared = u64::try_from((e + 1) * (e + 1)).unwrap() + 256;
    p.log_sigma[0] = 22;
    p.n_msis = 23;
    p.m2 = 63;
    p
}

/// The bounded witness in the proof ring: `s` zero and $`u_0=c`$, each polynomial split into its
/// two degree-64 components.
fn witness(scheme: &Abdlop, c: i128) -> PolyVec {
    let ring = scheme.ring().clone();
    let lifted = Ring::new(ring.modulus_i128(), 128).unwrap();
    let mut bounded = vec![Poly::zero(lifted.clone()); 9];
    bounded[8] = Poly::constant(lifted, c);
    PolyVec::new(
        ring.clone(),
        bounded
            .iter()
            .flat_map(|p| iso::split(p, ring.clone()).unwrap())
            .collect(),
    )
    .unwrap()
}

#[test]
fn a_linf_coefficient_is_bounded_by_the_extraction_bound_only() {
    let compiled = source().compile(params()).unwrap();
    let scheme = Abdlop::new([61; 32], params()).unwrap();
    let e = scheme.checked.approx_extraction_bound;
    assert_eq!(e, 2 * scheme.checked.z4_bound);
    assert_eq!(compiled.extraction_bound("u"), Ok(Extraction::Linf(e)));
    let statement = compiled.statement();
    let messages = PolyVec::zero(scheme.ring().clone(), 0);
    // The range rows are u's two components; the honest prover refuses u_0 = 2 > beta.
    let two = witness(&scheme, 2);
    let rows = statement
        .arp
        .as_ref()
        .unwrap()
        .evaluate(&two, &messages)
        .unwrap();
    assert_eq!(rows.entries(), &two.entries()[16..]);
    assert_eq!(
        statement.check_witness(&scheme, &two, &messages).err(),
        Some(Error::Witness)
    );
    assert_eq!(
        prove_with_seed(&scheme, statement, &two, &messages, b"linf", [62; 32]).err(),
        Some(Error::Witness)
    );
    // Without the check it proves, and the unmodified verifier accepts: the proof does not
    // enforce beta.
    let seed = Zeroizing::new([63; 32]);
    let proof = prove_rounds(
        &scheme,
        statement,
        &two,
        &messages,
        b"linf",
        &seed,
        secret::MAX_ATTEMPTS,
    )
    .unwrap();
    verify(&scheme, statement, &proof, b"linf").unwrap();
    compiled.verify([61; 32], &proof, b"linf").unwrap();
    // u_0 = E + 1, with parameters whose commitment admits it and whose E is the same.
    let long = long_params(e);
    let compiled = source().compile(long.clone()).unwrap();
    let scheme = Abdlop::new([64; 32], long).unwrap();
    assert_eq!(scheme.checked.approx_extraction_bound, e);
    let statement = compiled.statement();
    let beyond = witness(&scheme, e as i128 + 1);
    assert_eq!(
        statement.check_witness(&scheme, &beyond, &messages).err(),
        Some(Error::Witness)
    );
    // Eight range attempts, each refused by rejection sampling or by z4_bound.
    let seed = Zeroizing::new([65; 32]);
    match prove_rounds(&scheme, statement, &beyond, &messages, b"linf", &seed, 8) {
        Err(e) => assert_eq!(e, Error::RestartLimit),
        Ok(proof) => assert!(verify(&scheme, statement, &proof, b"linf").is_err()),
    }
}
