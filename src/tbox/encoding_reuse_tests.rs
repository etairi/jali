//! Encodings computed once per proof: the toolbox key and the prefix of a range attempt from
//! `EncodedForms`, reused across attempts, against a reference that absorbs the forms as it
//! encodes them, and
//! `absorb_encoded` against `absorb_equations`.
use super::projection_tests::{Rng, statement};
use super::*;

/// Reference `toolbox_key`, encoding the forms as it absorbs them.
fn toolbox_key_by_forms(
    scheme: &Abdlop,
    extended: &Abdlop,
    forms: &Forms,
    context: &[u8],
    s1: &PolyVec,
    m: &PolyVec,
    seed: &[u8; 32],
) -> Result<Zeroizing<[u8; 32]>, Error> {
    let mut statement = Transcript::new(
        b"LNP22-toolbox-statement",
        &extended.parameter_bytes(),
        &extended.seed,
        b"",
    );
    absorb_statement(&mut statement, extended, forms)?;
    Ok(derive_key(
        b"tbox/proof",
        seed,
        &[
            &scheme.fingerprint(),
            &statement.digest(),
            context,
            &abdlop::secret_bytes(s1),
            &abdlop::secret_bytes(m),
        ],
    ))
}

#[test]
fn encoded_forms_give_the_keys_and_prefixes_of_the_forms() {
    let mut rng = Rng(31);
    let scheme = Abdlop::new([9; 32], crate::params::toy_d64()).unwrap();
    let ring = scheme.ring().clone();
    let extended = scheme.extend_messages(9).unwrap();
    let vector = |rng: &mut Rng, n: usize| {
        PolyVec::new(ring.clone(), (0..n).map(|_| rng.poly(&ring)).collect()).unwrap()
    };
    for case in 0u8..6 {
        let mut forms = forms(&scheme, &extended, &statement(&mut rng, &scheme)).unwrap();
        if case % 3 == 1 {
            forms.approx.clear();
        } else if case % 3 == 2 {
            forms.exact.clear();
            forms.eqs.clear();
        }
        let encoded = EncodedForms::new(&forms).unwrap();
        let (s1, m) = (
            vector(&mut rng, scheme.bounded_len()),
            vector(&mut rng, scheme.message_len()),
        );
        for context in [b"".as_slice(), b"application"] {
            assert_eq!(
                toolbox_key(&scheme, &extended, &encoded, context, &s1, &m, &[case; 32]).unwrap(),
                toolbox_key_by_forms(&scheme, &extended, &forms, context, &s1, &m, &[case; 32])
                    .unwrap()
            );
            // One encoding serves every attempt's commitment.
            for _ in 0..3 {
                let commitment = Commitment {
                    t_a: PolyVec::zero(ring.clone(), extended.parameters.n_msis),
                    t_b: vector(&mut rng, extended.message_len()),
                };
                assert_eq!(
                    round_prefix_encoded(&extended, &commitment, &encoded, context)
                        .unwrap()
                        .digest(),
                    round_prefix(&extended, &commitment, &forms, context)
                        .unwrap()
                        .digest()
                );
            }
        }
    }
}

#[test]
fn absorbing_encodings_equals_absorbing_the_equations() {
    let mut rng = Rng(32);
    let scheme = Abdlop::new([9; 32], crate::params::toy_d64()).unwrap();
    let extended = scheme.extend_messages(9).unwrap();
    let forms = forms(&scheme, &extended, &statement(&mut rng, &scheme)).unwrap();
    for family in forms.families() {
        for count in [0, 1, family.len()] {
            let equations = &family[..count.min(family.len())];
            let mut streamed = Transcript::new(b"t", b"", &[0; 32], b"");
            let mut encoded = streamed.clone();
            quad_many::absorb_equations(&mut streamed, b"label", equations).unwrap();
            quad_many::absorb_encoded(
                &mut encoded,
                b"label",
                &quad_many::encode_equations(equations).unwrap(),
            );
            assert_eq!(streamed, encoded);
        }
    }
}
