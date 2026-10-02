//! The encoding of polynomials for statement hashing: a zero polynomial is one run of zero
//! bits, which must equal its $`d`$ uniform codes, at every
//! alignment of the writer and for narrow and wide moduli.
use super::*;

/// Reference `encode_poly`: one uniform code per coefficient.
fn reference_encode_poly(w: &mut BitWriter, p: &Poly) -> Result<(), Error> {
    let q = p.ring().modulus();
    match int::to_u128(&q) {
        Some(m) => {
            for x in p.coefficients() {
                w.uniform(int::low_u128(x), m)?;
            }
        }
        None => {
            for x in p.coefficients() {
                w.uniform_u256(x, &q)?;
            }
        }
    }
    Ok(())
}

#[test]
fn zero_polynomials_encode_as_their_uniform_codes() {
    let two = |k: u32| U256::ONE.shl_vartime(k);
    for q in [
        U256::from_u8(2),
        U256::from_u8(3),
        U256::from_u8(13),
        U256::from_u64(1099511627917),
        two(64),
        two(100).wrapping_sub(&U256::from_u8(15)),
        two(128).wrapping_add(&U256::from_u8(165)),
        two(255).wrapping_sub(&U256::from_u8(19)),
    ] {
        for d in [64, 128] {
            let ring = Ring::with_modulus(q, d).unwrap();
            let zero = Poly::zero(ring.clone());
            let other = Poly::constant(ring.clone(), -1).rotate(5);
            for offset in 0..70u32 {
                let (mut new, mut expected) = (BitWriter::new(), BitWriter::new());
                for w in [&mut new, &mut expected] {
                    w.unsigned((1 << offset) - 1, offset).unwrap();
                }
                for p in [&zero, &other, &zero, &zero] {
                    encode_poly(&mut new, p).unwrap();
                    reference_encode_poly(&mut expected, p).unwrap();
                }
                assert_eq!(
                    new.finish(),
                    expected.finish(),
                    "q {q}, d {d}, offset {offset}"
                );
            }
        }
    }
}
