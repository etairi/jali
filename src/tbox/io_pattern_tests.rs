//! The IO patterns of draft-irtf-cfrg-fiat-shamir-04 as the verifiers run them, recorded call by
//! call (`transcript::record`): one session per proof, the instance first, each prover message
//! absorbed as the bytes it has in the NARG string, each verifier message squeezed with the
//! length its type and the instance fix, and a NARG string that is exactly the concatenation of
//! the absorbed messages that are sent, then the compact final round (challenge and response),
//! with no other byte.
use super::stream_reuse_tests::toolbox_instance;
use super::*;
use crate::{
    fiat_shamir::ns,
    transcript::{
        CHALLENGE_BYTES,
        record::{Event, messages, recording},
    },
};

/// The lengths of the fixed-width fields of `scheme`'s proofs, in bytes.
struct Lengths {
    /// $`t_{A,1}`$ and $`t_B`$ of the toolbox's commitment.
    commitment: usize,
    /// $`z^{(e)}`$, $`z^{(d)}`$ and the garbage commitments.
    second: usize,
    /// $`h`$, or one polynomial modulo $`q`$ ($`t`$, $`v`$).
    h: usize,
    poly: usize,
    /// $`w_1`$ modulo $`m`$.
    w1: usize,
    /// The final round's challenge, responses and hints.
    response: usize,
}
fn lengths(scheme: &Abdlop) -> Lengths {
    let p = &scheme.parameters;
    let c = &scheme.checked;
    let d = p.degree;
    let k_q = scheme.ring().coefficient_bits() as usize;
    let k_m = (128
        - (crate::math::int::low_u128(&scheme.compression.hint_modulus()) - 1).leading_zeros())
        as usize;
    let k_c = (128 - ((2 * c.omega) as u128).leading_zeros()) as usize;
    let e = if c.n_ex > 0 { 256 / d } else { 0 };
    let a = if p.n_prime > 0 { 256 / d } else { 0 };
    let g = c.lambda / 2;
    let bits = |n: usize, k: usize| n * d * k / 8;
    let w = |t: u32| response_width(t) as usize;
    Lengths {
        commitment: bits(p.n_msis, scheme.high_bits() as usize) + bits(p.l + e + a + 1, k_q),
        second: bits(e, w(p.log_sigma[2])) + bits(a, w(p.log_sigma[3])) + bits(g, k_q),
        h: bits(g, k_q),
        poly: bits(1, k_q),
        w1: bits(p.n_msis, k_m),
        response: d / 2 * k_c / 8
            + bits(scheme.bounded_len(), w(p.log_sigma[0]))
            + bits(p.m2 - p.n_msis, w(p.log_sigma[1]))
            + bits(p.n_msis, c.hint_width as usize),
    }
}

/// The squeezes of the toolbox's verifier messages after the commitment: the projection
/// matrices (2 bits per entry), $`\Gamma`$ and $`\mu`$ (`DecodeUint`, $`N_s+16`$ bytes per
/// scalar modulo $`q`$), and the challenge.
fn squeezes(scheme: &Abdlop, statement: &Statement) -> [usize; 4] {
    let p = &scheme.parameters;
    let d = p.degree;
    let e = if scheme.checked.n_ex > 0 { 256 / d } else { 0 };
    let a = if p.n_prime > 0 { 256 / d } else { 0 };
    let extended = scheme.extend_messages(e + a + 1).unwrap();
    let forms = forms(scheme, &extended, statement).unwrap();
    let width = ns(&scheme.ring().modulus()).unwrap() + 16;
    let rows = |block: &[QuadEq]| if block.is_empty() { 0 } else { 256 };
    let evals = forms.evals.len() + rows(&forms.exact) + rows(&forms.approx);
    let finals = forms.eqs.len() + scheme.checked.lambda / 2;
    [
        256 * (forms.exact.len() + forms.approx.len()) * d / 4,
        scheme.checked.lambda * evals * width,
        finals * d * width,
        CHALLENGE_BYTES,
    ]
}

/// The absorbed bytes of the event, or a panic.
fn absorbed(event: &Event) -> &[u8] {
    match event {
        Event::Absorb(bytes) => bytes,
        other => panic!("an absorb, not {other:?}"),
    }
}
fn squeezed(event: &Event) -> usize {
    match event {
        Event::Squeeze(len) => *len,
        other => panic!("a squeeze, not {other:?}"),
    }
}

#[test]
fn the_toolbox_verifier_runs_the_documented_io_pattern_and_the_narg_is_its_messages() {
    let scheme = Abdlop::new([1; 32], crate::params::toy_d64()).unwrap();
    let (statement, s1, m) = toolbox_instance(&scheme);
    let context = b"io-pattern";
    let proof = prove_with_seed(&scheme, &statement, &s1, &m, context, [204; 32]).unwrap();
    let narg = crate::codec::proof::encode(&scheme, &proof).unwrap();
    let decoded = crate::codec::proof::decode(&scheme, &narg).unwrap();
    let (result, events) = recording(|| verify(&scheme, &statement, &decoded, context));
    result.unwrap();
    let events = messages(&events);
    assert_eq!(events[0], Event::Start(Protocol::Toolbox, context.to_vec()));
    assert_eq!(events.len(), 9);
    let l = lengths(&scheme);
    // The instance starts with SerializeVarLenString of the parameters, then the public seed.
    let params = scheme.parameter_bytes();
    let first = absorbed(&events[1]);
    assert_eq!(first[..4], (params.len() as u32).to_le_bytes());
    assert_eq!(first[4..4 + params.len()], params[..]);
    assert_eq!(first[4 + params.len()..36 + params.len()], scheme.seed);
    // Prover messages: the commitment closes the first absorb, then z_e, z_d and the garbage
    // commitments, h, and the final message (t, v, w1), whose t only is sent.
    let mut at = 0;
    let mut take = |n: usize| {
        at += n;
        &narg[at - n..at]
    };
    assert!(first.ends_with(take(l.commitment)));
    assert_eq!(absorbed(&events[3]), take(l.second));
    assert_eq!(absorbed(&events[5]), take(l.h));
    let last = absorbed(&events[7]);
    assert_eq!(last.len(), 2 * l.poly + l.w1);
    assert_eq!(&last[..l.poly], take(l.poly));
    // The verifier messages, each of the length its type fixes.
    let [projections, gamma, mu, challenge] = squeezes(&scheme, &statement);
    assert_eq!(
        [2, 4, 6, 8].map(|i| squeezed(&events[i])),
        [projections, gamma, mu, challenge]
    );
    // Then the compact final round: the challenge and the response, and nothing more.
    assert_eq!(narg.len(), at + l.response);
    assert_eq!(narg.len(), scheme.checked.estimated_proof_bytes);
}

#[test]
fn the_extractable_verifier_squeezes_the_linhc_key_and_absorbs_the_linhc_messages() {
    let scheme = Abdlop::new([1; 32], crate::params::toy_d64()).unwrap();
    let extraction = crate::params::toy_d64_extraction();
    let (statement, s1, m) = toolbox_instance(&scheme);
    let context = b"io-pattern-sl";
    let proof = prove_extractable_with_seed(
        &scheme,
        &extraction,
        &statement,
        &s1,
        &m,
        context,
        [205; 32],
    )
    .unwrap();
    let narg = crate::codec::proof::encode_extractable(&scheme, &extraction, &proof).unwrap();
    assert_eq!(
        narg.len(),
        extraction.proof_bytes(&scheme.parameters).unwrap()
    );
    let (result, events) =
        recording(|| verify_extractable(&scheme, &extraction, &statement, &proof, context));
    result.unwrap();
    let events = messages(&events);
    assert_eq!(
        events[0],
        Event::Start(Protocol::ToolboxExtractable, context.to_vec())
    );
    assert_eq!(events.len(), 11);
    let (linhc, _) = extraction
        .linhc(&scheme.parameters, &scheme.checked)
        .unwrap();
    let d = linhc.degree;
    let k = (128 - (linhc.modulus - 1).leading_zeros()) as usize;
    let pair = 2 * linhc.len * d * k / 8;
    let width = ns(&crate::math::U256::from_u128(linhc.modulus)).unwrap() + 16;
    let l = lengths(&scheme);
    // The instance ends with the extraction set, then the key: A and B, m' x n entries each.
    let set = extraction.transcript_bytes();
    let mut tail = (set.len() as u32).to_le_bytes().to_vec();
    tail.extend_from_slice(&set);
    assert!(absorbed(&events[1]).ends_with(&tail));
    assert_eq!(squeezed(&events[2]), 2 * linhc.len * linhc.rank * d * width);
    // The first message ends with the LinHC commitment, and the final one has the LinHC mask
    // commitment between (t, v) and w1.
    let mut at = 0;
    let mut take = |n: usize| {
        at += n;
        &narg[at - n..at]
    };
    assert_eq!(absorbed(&events[3]), take(l.commitment + pair));
    assert_eq!(absorbed(&events[5]), take(l.second));
    assert_eq!(absorbed(&events[7]), take(l.h));
    let last = absorbed(&events[9]);
    assert_eq!(last.len(), 2 * l.poly + pair + l.w1);
    assert_eq!(&last[..l.poly], take(l.poly));
    let [projections, gamma, mu, challenge] = squeezes(&scheme, &statement);
    assert_eq!(
        [4, 6, 8, 10].map(|i| squeezed(&events[i])),
        [projections, gamma, mu, challenge]
    );
    // The final round: challenge, response and hints, then the LinHC opening.
    let opening = (linhc.rank + 2 * linhc.len) * d * response_width(linhc.log_sigma) as usize / 8;
    assert_eq!(narg.len(), at + l.response + opening);
}

#[test]
fn the_opening_proof_on_its_own_is_one_compact_round() {
    let (scheme, commitment, opening) =
        crate::abdlop::verifier_check_tests::commitment_for(crate::params::toy_d64());
    let proof = scheme
        .prove_with_seed(&commitment, &opening, b"ctx", [7; 32])
        .unwrap();
    let narg = scheme.encode_proof(&proof).unwrap();
    let (result, events) = recording(|| scheme.verify(&commitment, &proof, b"ctx"));
    result.unwrap();
    let events = messages(&events);
    assert_eq!(events.len(), 3);
    assert_eq!(events[0], Event::Start(Protocol::Opening, b"ctx".to_vec()));
    // The instance (scheme and commitment) and w1 in one absorb, then the challenge.
    let l = lengths(&scheme);
    let instance = 4
        + scheme.parameter_bytes().len()
        + 32
        + scheme.commitment_bytes(&commitment).unwrap().len();
    assert_eq!(absorbed(&events[1]).len(), instance + l.w1);
    assert_eq!(squeezed(&events[2]), CHALLENGE_BYTES);
    assert_eq!(narg.len(), l.response);
}
