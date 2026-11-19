//! ABDLOP commitment and compressed opening proof (LNP22 Fig. 4 and Fig. 18).
//!
//! The verifier checks the joint norm of both randomness-response components. The opening
//! proof on its own runs the transcript `JALI-V02-ABDLOP-OPENING-C1-SHAKE128`: its instance is
//! the parameters, the public seed and the commitment, its one prover message $`w_1`$ and its
//! verifier message the challenge (`docs/encodings.md`).
use crate::{
    Error,
    codec::{BitReader, BitWriter, fields},
    dcompress::Compression,
    linhc,
    math::{
        I256, Poly, PolyMat, PolyVec, Ring, U256, int,
        terms::{Shape, Term},
    },
    params::{CheckedParams, TboxParams, response_width},
    rand::{
        AesPrg, ByteStream, bounded, derive_key, domain, gaussian,
        reject::{self, Policy, Variance},
        secret, take_seed, uniform_poly,
    },
    transcript::{Protocol, Transcript},
};
use std::sync::Arc;
use zeroize::Zeroizing;

/// Coin bytes of one attempt of the opening proof: the coins of Rej_1 and of Rej_2, in that
/// order, of [`reject::COIN_BYTES`] each.
pub(crate) const OPENING_COIN_BYTES: usize = 2 * reject::COIN_BYTES;

/// Public commitment with compressed Ajtai part and full BDLOP messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Commitment {
    /// High bits $`t_{A,1}`$, of length n.
    pub t_a: PolyVec,
    /// $`t_B`$, of length l.
    pub t_b: PolyVec,
}

#[cfg(test)]
mod coin_tests;
#[cfg(test)]
mod restart_tests;
#[cfg(test)]
mod speculation_tests;
#[cfg(test)]
pub(crate) mod verifier_check_tests;

#[cfg(test)]
mod extraction_tests {
    use super::*;

    #[test]
    fn large_z21_satisfying_the_reconstruction_equation_is_rejected_by_the_joint_bound() {
        let scheme = Abdlop::new([91; 32], crate::params::toy_d64()).unwrap();
        let ring = scheme.ring().clone();
        let c = Poly::zero(ring.clone());
        let commitment = Commitment {
            t_a: PolyVec::zero(ring.clone(), scheme.parameters.n_msis),
            t_b: PolyVec::zero(ring.clone(), scheme.message_len()),
        };
        let mut values = vec![Poly::zero(ring.clone()); scheme.a2.cols()];
        values[0] = Poly::constant(ring.clone(), 1 << 30);
        let proof = OpeningProof {
            challenge: c.clone(),
            z1: PolyVec::zero(ring.clone(), scheme.bounded_len()),
            z21: PolyVec::new(ring.clone(), values).unwrap(),
            hint: PolyVec::zero(ring, scheme.parameters.n_msis),
        };
        let r = scheme.a2.mul(&proof.z21).unwrap();
        let w1 = map(&r, |x| scheme.compression.decompose(x).0).unwrap();
        let z22 = sub(&r, &scale_scalar(&w1, &scheme.gamma()).unwrap()).unwrap();
        // This passes the incomplete z22-only norm check. Fixing the interactive
        // challenge isolates the inequality from Fiat–Shamir rejection.
        assert!(z22.norm_squared().unwrap() < U256::from(scheme.checked.b_squared));
        assert!(proof.z21.norm_squared().unwrap() > U256::from(scheme.checked.b_squared));
        assert_eq!(
            scheme.verify_core(&commitment, &proof, |_| Ok(c)),
            Err(Error::InvalidProof)
        );
    }

    #[test]
    fn two_challenges_recover_an_opening_of_the_compressed_commitment() {
        let scheme = Abdlop::new([91; 32], crate::params::toy_d64()).unwrap();
        let ring = scheme.ring().clone();
        let s1 = PolyVec::new(
            ring.clone(),
            vec![Poly::constant(ring.clone(), 1); scheme.bounded_len()],
        )
        .unwrap();
        let m = PolyVec::new(
            ring.clone(),
            vec![Poly::constant(ring.clone(), 3); scheme.message_len()],
        )
        .unwrap();
        let (commitment, opening) = scheme
            .commit_with_seed(s1.clone(), m.clone(), [92; 32])
            .unwrap();
        let c1 = Poly::constant(ring.clone(), 1);
        let c2 = Poly::constant(ring.clone(), 2);
        // Rewinding: one transcript prefix, and so one key, answered with two challenges.
        let binding = scheme
            .opening_transcript(&commitment, b"")
            .unwrap()
            .binding();
        let mut forks = None;
        for seed in 0..32 {
            let seed = Zeroizing::new([seed; 32]);
            let fork = |c: &Poly| {
                scheme
                    .prove_core(
                        &commitment,
                        &opening,
                        Caller::Opening,
                        &binding,
                        &seed,
                        |_, _, _| Ok((c.clone(), ())),
                    )
                    .unwrap()
                    .0
            };
            let (p1, p2) = (fork(&c1), fork(&c2));
            if sub(&p2.z1, &p1.z1).unwrap() == s1
                && sub(&p2.z21, &p1.z21).unwrap() == part(&opening.s2, 0, scheme.a2.cols()).unwrap()
            {
                forks = Some((p1, p2));
                break;
            }
        }
        let (p1, p2) = forks.expect("same accepted mask prefix for two challenges");
        scheme.verify_core(&commitment, &p1, |_| Ok(c1)).unwrap();
        scheme.verify_core(&commitment, &p2, |_| Ok(c2)).unwrap();
        let reconstruct = |proof: &OpeningProof| {
            let r = sub(
                &add(
                    &scheme.a1.mul(&proof.z1).unwrap(),
                    &scheme.a2.mul(&proof.z21).unwrap(),
                )
                .unwrap(),
                &scale(
                    &commitment.t_a,
                    &proof
                        .challenge
                        .scale_u256(&scheme.rounding_power())
                        .unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
            let w1 = scheme.use_hints(&proof.hint, &r).unwrap();
            let z22 = sub(&scale_scalar(&w1, &scheme.gamma()).unwrap(), &r).unwrap();
            (w1, z22)
        };
        let (w1, z221) = reconstruct(&p1);
        let (w2, z222) = reconstruct(&p2);
        assert_eq!(w1, w2);
        let recovered_s1 = sub(&p2.z1, &p1.z1).unwrap();
        let recovered_s21 = sub(&p2.z21, &p1.z21).unwrap();
        let recovered_s22 = sub(&z222, &z221).unwrap();
        let ta = add(
            &add(
                &scheme.a1.mul(&recovered_s1).unwrap(),
                &scheme.a2.mul(&recovered_s21).unwrap(),
            )
            .unwrap(),
            &recovered_s22,
        )
        .unwrap();
        assert_eq!(
            ta,
            scale_scalar(&commitment.t_a, &scheme.rounding_power()).unwrap()
        );
        assert_eq!(
            sub(&commitment.t_b, &scheme.b.mul(&recovered_s21).unwrap()).unwrap(),
            m
        );
    }
}

/// Secret opening. Polynomial buffers zeroize on drop; debug output omits coefficients.
#[derive(Clone)]
pub struct Opening {
    pub(crate) s1: PolyVec,
    pub(crate) s2: PolyVec,
    pub(crate) m: PolyVec,
    pub(crate) low: PolyVec,
}
impl core::fmt::Debug for Opening {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Opening").finish_non_exhaustive()
    }
}

/// The fixed-shape compact proof `(c, z1, z21, hint)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningProof {
    /// Sigma-stable challenge.
    pub challenge: Poly,
    /// Ajtai response.
    pub z1: PolyVec,
    /// First randomness-response component.
    pub z21: PolyVec,
    /// Hints reconstructing compressed mask commitments.
    pub hint: PolyVec,
}

/// Expanded public matrices and validated parameters for an ABDLOP instance.
#[derive(Clone, Debug)]
pub struct Abdlop {
    pub(crate) parameters: TboxParams,
    pub(crate) checked: CheckedParams,
    pub(crate) ring: Arc<Ring>,
    pub(crate) seed: [u8; 32],
    pub(crate) a1: PolyMat,
    pub(crate) a2: PolyMat,
    pub(crate) b: PolyMat,
    pub(crate) compression: Compression,
}

pub(crate) fn matrix(
    ring: Arc<Ring>,
    rows: usize,
    cols: usize,
    seed: &[u8; 32],
    dom: u32,
) -> Result<PolyMat, Error> {
    let entries = matrix_entries(&ring, 0..matrix_size(rows, cols)?, seed, dom)?;
    PolyMat::new(ring, rows, cols, entries)
}

/// The number of entries of a public matrix, at most $`2^{20}`$.
pub(crate) fn matrix_size(rows: usize, cols: usize) -> Result<usize, Error> {
    rows.checked_mul(cols)
        .filter(|n| *n <= 1 << 20)
        .ok_or(Error::Dimension)
}

/// The entries with row-major indices in `range` of the matrices that [`matrix`] expands from
/// `seed` and `dom`. Entry $`i`$ is read from its own stream $`(dom,i)`$, whatever the shape, so
/// rows can be expanded on their own.
pub(crate) fn matrix_entries(
    ring: &Arc<Ring>,
    range: std::ops::Range<usize>,
    seed: &[u8; 32],
    dom: u32,
) -> Result<Vec<Poly>, Error> {
    range
        .map(|i| uniform_poly(&mut AesPrg::new(seed, domain(dom, i as u32)), ring))
        .collect()
}

/// Gaussian mask polynomials together with the exact variance they were sampled at. Rejection
/// tests take the variance from here, so the width is chosen once, where the mask is sampled.
pub(crate) struct Mask {
    pub(crate) values: PolyVec,
    pub(crate) variance: Variance,
}

pub(crate) fn sample_gaussian(
    ring: Arc<Ring>,
    len: usize,
    t: u32,
    seed: &[u8; 32],
    dom: u32,
) -> Result<Mask, Error> {
    let mut entries = Vec::with_capacity(len);
    for i in 0..len {
        let stream = AesPrg::new(seed, domain(dom, (i + 1) as u32));
        // On x86 the sampler's short reads come from 1 KiB batches of the keystream, the batch
        // of VAES-512, and the stream is dropped after the mask: the same bytes in the same
        // order as from the stream itself.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let mut stream = crate::rand::ReadAhead::<_, 1024>::new(stream);
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let mut stream = stream;
        let mut values = Zeroizing::new(gaussian(&mut stream, t, ring.degree())?);
        // A width at which a sample can exceed q/2 is a parameter error; another attempt
        // would not fix it.
        if values
            .iter()
            .any(|x| U256::from_u128(x.unsigned_abs()) > ring.half)
        {
            return Err(Error::Parameter("Gaussian mask above q/2"));
        }
        entries.push(Poly::new(ring.clone(), std::mem::take(&mut *values))?);
    }
    Ok(Mask {
        values: PolyVec::new(ring, entries)?,
        variance: Variance::gaussian(t)?,
    })
}
pub(crate) fn add(a: &PolyVec, b: &PolyVec) -> Result<PolyVec, Error> {
    if a.len() != b.len() {
        return Err(Error::Dimension);
    }
    PolyVec::new(
        a.ring().clone(),
        a.entries()
            .iter()
            .zip(b.entries())
            .map(|(a, b)| a.add(b))
            .collect::<Result<_, _>>()?,
    )
}
pub(crate) fn sub(a: &PolyVec, b: &PolyVec) -> Result<PolyVec, Error> {
    if a.len() != b.len() {
        return Err(Error::Dimension);
    }
    PolyVec::new(
        a.ring().clone(),
        a.entries()
            .iter()
            .zip(b.entries())
            .map(|(a, b)| a.sub(b))
            .collect::<Result<_, _>>()?,
    )
}
/// $`(a_ib)_i`$, the values of `a_i.mul(b)`: `b` classified once and, if general, transformed
/// once, and each general $`a_i`$ transformed once.
pub(crate) fn scale(a: &PolyVec, b: &Poly) -> Result<PolyVec, Error> {
    if b.ring() != a.ring() {
        return Err(Error::RingMismatch);
    }
    let term = Term::new(b);
    let product = |x: &Poly| -> Result<Poly, Error> {
        let mut out = Poly::zero(a.ring().clone());
        match &term {
            Term::Cheap(shape) => {
                shape.add_product_to(&mut out, x)?;
            }
            Term::General(b_ntt) => match Shape::of(x) {
                Shape::General => return x.to_ntt().product(b_ntt)?.reduce(),
                cheap => {
                    cheap.add_product_to(&mut out, b)?;
                }
            },
        }
        Ok(out)
    };
    PolyVec::new(
        a.ring().clone(),
        a.entries().iter().map(product).collect::<Result<_, _>>()?,
    )
}
pub(crate) fn part(a: &PolyVec, start: usize, end: usize) -> Result<PolyVec, Error> {
    PolyVec::new(
        a.ring().clone(),
        a.entries()
            .get(start..end)
            .ok_or(Error::Dimension)?
            .to_vec(),
    )
}
/// Apply `f` to every canonical coefficient; `f` must return canonical values.
pub(crate) fn map(a: &PolyVec, f: impl Fn(&U256) -> U256) -> Result<PolyVec, Error> {
    PolyVec::new(
        a.ring().clone(),
        a.entries()
            .iter()
            .map(|a| Poly::from_canonical(a.ring().clone(), a.values().map(|x| f(&x)).collect()))
            .collect(),
    )
}
/// Multiply every entry by a scalar.
pub(crate) fn scale_scalar(a: &PolyVec, scalar: &U256) -> Result<PolyVec, Error> {
    PolyVec::new(
        a.ring().clone(),
        a.entries()
            .iter()
            .map(|a| a.scale_u256(scalar))
            .collect::<Result<_, _>>()?,
    )
}
/// Centred coefficients of short vectors as `i128`. Fails with [`Error::Overflow`] if one does
/// not fit.
pub(crate) fn flatten(a: &PolyVec) -> Result<Zeroizing<Vec<i128>>, Error> {
    let mut out = Zeroizing::new(Vec::with_capacity(a.len() * a.ring().degree()));
    for p in a.entries() {
        out.extend_from_slice(&p.coefficients_i128()?);
    }
    Ok(out)
}
/// The proof that runs the opening-proof core. Its tag is a part of the core's key, so the two
/// callers never share a key, even if their transcript digests were ever equal.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Caller {
    /// `Abdlop::prove_with_seed`.
    Opening,
    /// `quad::prove_with_seed`, also inside `quad_many` and `quad_eval`.
    Quadratic,
}
impl Caller {
    fn tag(self) -> &'static [u8] {
        match self {
            Self::Opening => b"opening",
            Self::Quadratic => b"quadratic",
        }
    }
}
/// Secret polynomials as key-derivation input: each centred coefficient as 16 LE bytes
/// (two's complement) when $`q<2^{128}`$, and as 32 LE bytes otherwise. The width depends on
/// the ring only, which the key derivation binds as well.
pub(crate) fn secret_bytes(a: &PolyVec) -> Zeroizing<Vec<u8>> {
    let ring = a.ring();
    let wide = int::to_u128(&ring.q).is_none();
    let width = if wide { 32 } else { 16 };
    let mut out = Zeroizing::new(Vec::with_capacity(width * a.len() * ring.degree()));
    for p in a.entries() {
        for i in 0..ring.degree() {
            if wide {
                // The two's complement bits, wiped like the narrow value.
                let x = Zeroizing::new(p.coefficient(i));
                let value = Zeroizing::new(*ring.centred_i256(&x).as_uint());
                for word in value.as_words() {
                    out.extend_from_slice(&word.to_le_bytes());
                }
            } else {
                let value = Zeroizing::new(p.centred(i).expect("q below 2^128"));
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    out
}

impl Abdlop {
    /// Expand A1, A2' and B' from domains 0, 1 and 2 of the public seed.
    pub fn new(seed: [u8; 32], parameters: TboxParams) -> Result<Self, Error> {
        let checked = parameters.check()?;
        Self::with_checked(seed, parameters, checked)
    }
    /// Test only: construct over parameters whose range blocks use a two-prime modulus.
    #[cfg(test)]
    pub(crate) fn new_allowing_composite_range_blocks(
        seed: [u8; 32],
        parameters: TboxParams,
    ) -> Result<Self, Error> {
        let checked = parameters.check_allowing_composite_range_blocks()?;
        Self::with_checked(seed, parameters, checked)
    }
    pub(crate) fn with_checked(
        seed: [u8; 32],
        parameters: TboxParams,
        checked: CheckedParams,
    ) -> Result<Self, Error> {
        let ring = Ring::with_modulus(checked.q, parameters.degree)?;
        let m1 = parameters.m1 + parameters.l2_rows.len();
        let cols = parameters.m2 - parameters.n_msis;
        let a1 = matrix(ring.clone(), parameters.n_msis, m1, &seed, 0)?;
        let a2 = matrix(ring.clone(), parameters.n_msis, cols, &seed, 1)?;
        let b = matrix(ring.clone(), parameters.l, cols, &seed, 2)?;
        let compression = Compression::with_modulus(
            checked.q,
            U256::from_u64(parameters.gamma),
            parameters.d_bits,
        )?;
        Ok(Self {
            parameters,
            checked,
            ring,
            seed,
            a1,
            a2,
            b,
            compression,
        })
    }
    /// Proof ring shared by all inputs.
    pub fn ring(&self) -> &Arc<Ring> {
        &self.ring
    }
    /// Expected length of the bounded witness, including slack coordinates.
    pub fn bounded_len(&self) -> usize {
        self.a1.cols()
    }
    /// Expected BDLOP message length.
    pub fn message_len(&self) -> usize {
        self.parameters.l
    }
    pub(crate) fn extend_messages(&self, count: usize) -> Result<Self, Error> {
        if count >= self.checked.l_ext {
            return Err(Error::Dimension);
        }
        let mut out = self.clone();
        out.parameters.l += count;
        out.checked.l_ext -= count;
        // B is `matrix(.., l, cols, seed, 2)` and its entries are independent streams, so the
        // first rows of the larger matrix are those of B: only the new rows are expanded.
        let cols = out.a2.cols();
        let size = matrix_size(out.parameters.l, cols)?;
        debug_assert_eq!(self.b.entries().len(), self.parameters.l * cols);
        let mut entries = Vec::with_capacity(size);
        entries.extend_from_slice(self.b.entries());
        entries.extend(matrix_entries(
            &out.ring,
            entries.len()..size,
            &out.seed,
            2,
        )?);
        out.b = PolyMat::new(out.ring.clone(), out.parameters.l, cols, entries)?;
        Ok(out)
    }
    fn shape(&self, a: &PolyVec, len: usize) -> Result<(), Error> {
        if a.len() != len {
            return Err(Error::Dimension);
        }
        if a.ring() != &self.ring {
            return Err(Error::RingMismatch);
        }
        Ok(())
    }
    /// Commit with explicitly supplied, bounded randomness, for composition and testing.
    pub fn commit_with_randomness(
        &self,
        s1: PolyVec,
        m: PolyVec,
        s2: PolyVec,
    ) -> Result<(Commitment, Opening), Error> {
        self.shape(&s1, self.bounded_len())?;
        self.shape(&m, self.message_len())?;
        self.shape(&s2, self.parameters.m2)?;
        let bound = self.parameters.alpha_squared as u128
            + (self.parameters.l2_rows.len() * self.parameters.degree) as u128;
        if s1.norm_squared()? > U256::from(bound)
            || s2.entries().iter().any(|p| p.norm_infinity() > U256::ONE)
        {
            return Err(Error::Witness);
        }
        let s21 = part(&s2, 0, self.a2.cols())?;
        let s22 = part(&s2, self.a2.cols(), s2.len())?;
        let ta = add(&add(&self.a1.mul(&s1)?, &self.a2.mul(&s21)?)?, &s22)?;
        let commitment = Commitment {
            t_a: map(&ta, |x| self.compression.power2round(x).0)?,
            t_b: add(&self.b.mul(&s21)?, &m)?,
        };
        let low = map(&ta, |x| {
            self.ring.reduce_i256(&self.compression.power2round(x).1)
        })?;
        Ok((commitment, Opening { s1, s2, m, low }))
    }
    /// Commit with randomness sampled from $`[-1,1]^{m_2d}`$ under a key derived from a seed.
    ///
    /// The seed must be secret and uniformly random, and it may be reused. The key depends on
    /// the seed, the parameters, the public seed and the committed values: committing to other
    /// values, or under other parameters, reads independent randomness (assuming SHAKE128 and
    /// AES-256 behave as pseudorandom functions), while committing to the same values again
    /// returns the same commitment, so the two are linkable.
    ///
    /// With a reused seed, hiding holds only relative to the equality pattern of the committed
    /// values: whether two commitments from one seed are equal reveals whether their values
    /// are. Where that matters, use `commit` with a `rand_core::CryptoRng`, which draws a fresh
    /// seed per call.
    pub fn commit_with_seed(
        &self,
        s1: PolyVec,
        m: PolyVec,
        mut seed: [u8; 32],
    ) -> Result<(Commitment, Opening), Error> {
        self.commit_seeded(s1, m, &take_seed(&mut seed))
    }
    fn commit_seeded(
        &self,
        s1: PolyVec,
        m: PolyVec,
        seed: &Zeroizing<[u8; 32]>,
    ) -> Result<(Commitment, Opening), Error> {
        let key = self.commitment_key(&s1, &m, seed);
        self.commit_in_domain(s1, m, &key, secret::COMMITMENT)
    }
    /// The key of `commit_with_seed`.
    pub(crate) fn commitment_key(
        &self,
        s1: &PolyVec,
        m: &PolyVec,
        seed: &[u8; 32],
    ) -> Zeroizing<[u8; 32]> {
        derive_key(
            b"abdlop/commitment",
            seed,
            &[&self.fingerprint(), &secret_bytes(s1), &secret_bytes(m)],
        )
    }
    /// Commit with randomness read from the streams $`(word, i+1)`$ of a secret key.
    pub(crate) fn commit_in_domain(
        &self,
        s1: PolyVec,
        m: PolyVec,
        key: &[u8; 32],
        word: u32,
    ) -> Result<(Commitment, Opening), Error> {
        let entries = (0..self.parameters.m2)
            .map(|i| {
                Poly::new(
                    self.ring.clone(),
                    bounded(
                        &mut AesPrg::new(key, domain(word, (i + 1) as u32)),
                        -1,
                        1,
                        self.ring.degree(),
                    )?,
                )
            })
            .collect::<Result<_, Error>>()?;
        self.commit_with_randomness(s1, m, PolyVec::new(self.ring.clone(), entries)?)
    }
    /// Commit with an explicit cryptographic RNG (rand_core 0.10).
    pub fn commit(
        &self,
        s1: PolyVec,
        m: PolyVec,
        rng: &mut impl rand_core::CryptoRng,
    ) -> Result<(Commitment, Opening), Error> {
        let mut seed = Zeroizing::new([0; 32]);
        rand_core::Rng::fill_bytes(rng, &mut *seed);
        self.commit_seeded(s1, m, &seed)
    }
    /// Every parameter field and derived bound that transcripts bind.
    pub(crate) fn parameter_bytes(&self) -> Vec<u8> {
        let mut params = Vec::new();
        params.extend_from_slice(&self.parameters.transcript_bytes());
        // q as 16 LE bytes below 2^128 and as 32 otherwise. q is a function of the prime
        // factors, which the parameter bytes already encode injectively.
        let q = self.ring.modulus().to_le_bytes();
        params.extend_from_slice(if int::to_u128(&self.ring.q).is_some() {
            &q[..16]
        } else {
            &q[..]
        });
        for x in [
            self.ring.degree() as u128,
            self.bounded_len() as u128,
            self.parameters.m2 as u128,
            self.parameters.n_msis as u128,
            self.parameters.l as u128,
            self.parameters.gamma as u128,
            self.parameters.d_bits as u128,
            self.checked.b_squared,
            self.checked.z1_bound_squared,
            self.checked.omega as u128,
            self.checked.eta as u128,
            self.parameters.alpha_squared as u128,
            self.parameters.log_sigma[0] as u128,
            self.parameters.log_sigma[1] as u128,
        ] {
            params.extend_from_slice(&x.to_le_bytes());
        }
        params
    }
    /// Digest of the parameters and the public seed, which the keys of `commit_with_seed` and of
    /// the toolbox bind. Unchanged since version 0.1: the 32-byte state of that version's
    /// transcript chain after the protocol `abdlop-scheme`, the parameter bytes, the seed and an
    /// empty statement, each length-framed (`legacy_chain`). It is a key-derivation input, not
    /// part of any Fiat–Shamir transcript.
    pub(crate) fn fingerprint(&self) -> [u8; 32] {
        legacy_chain(&[
            (b"protocol", b"abdlop-scheme"),
            (b"parameters", &self.parameter_bytes()),
            (b"pp-seed", &self.seed),
            (b"statement", b""),
        ])
    }
    /// Absorb the scheme, the first part of every instance: `SerializeVarLenString` of the
    /// parameter bytes, then the 32-byte public seed, which determines the public matrices.
    pub(crate) fn absorb_scheme(&self, transcript: &mut Transcript) -> Result<(), Error> {
        transcript.absorb_varlen(&self.parameter_bytes())?;
        transcript.absorb(&self.seed);
        Ok(())
    }
    /// The transcript of the opening proof on its own, after its instance: the scheme, then the
    /// commitment's serialization ([`Self::commitment_bytes`]).
    pub(crate) fn opening_transcript(
        &self,
        commitment: &Commitment,
        context: &[u8],
    ) -> Result<Transcript, Error> {
        let mut transcript = Transcript::new(Protocol::Opening, context)?;
        self.absorb_scheme(&mut transcript)?;
        transcript.absorb(&self.commitment_bytes(commitment)?);
        Ok(transcript)
    }
    /// The commitment's shape, and its compressed part within the high-part range.
    fn check_commitment(&self, commitment: &Commitment) -> Result<(), Error> {
        self.shape(&commitment.t_a, self.parameters.n_msis)?;
        self.shape(&commitment.t_b, self.parameters.l)?;
        let max_high = self.max_high();
        if commitment
            .t_a
            .entries()
            .iter()
            .any(|p| p.values().any(|x| x > max_high))
        {
            return Err(Error::InvalidProof);
        }
        Ok(())
    }
    /// The serialization of a commitment, after the checks of [`Self::check_commitment`]: the
    /// high parts $`t_{A,1}`$ in $`\mathrm{bitlen}(q-1)-D`$ bits each, then $`t_B`$ modulo
    /// $`q`$.
    pub(crate) fn commitment_bytes(&self, commitment: &Commitment) -> Result<Vec<u8>, Error> {
        self.check_commitment(commitment)?;
        let mut w = BitWriter::new();
        self.write_commitment(&mut w, commitment)?;
        w.into_bytes()
    }
    /// The width of a compressed high part: $`\mathrm{bitlen}(q-1)-D`$.
    pub(crate) fn high_bits(&self) -> u32 {
        self.ring.coefficient_bits() - self.parameters.d_bits
    }
    /// Write the fields of [`Self::commitment_bytes`], refusing a high part above
    /// [`Self::max_high`].
    pub(crate) fn write_commitment(
        &self,
        w: &mut BitWriter,
        commitment: &Commitment,
    ) -> Result<(), Error> {
        let (bits, max_high) = (self.high_bits(), self.max_high());
        for p in commitment.t_a.entries() {
            for x in p.values() {
                if x > max_high {
                    return Err(Error::Encoding);
                }
                w.unsigned_u256(&x, bits)?;
            }
        }
        fields::write_below(w, &commitment.t_b, &self.ring.q)
    }
    /// Read a commitment written by [`Self::write_commitment`] with `messages` rows in $`t_B`$,
    /// refusing high parts above [`Self::max_high`] and values not below $`q`$.
    pub(crate) fn read_commitment(
        &self,
        r: &mut BitReader<'_>,
        messages: usize,
    ) -> Result<Commitment, Error> {
        let (bits, max_high) = (self.high_bits(), self.max_high());
        let mut t_a = Vec::with_capacity(self.parameters.n_msis);
        for _ in 0..self.parameters.n_msis {
            let coefficients = (0..self.ring.degree())
                .map(|_| {
                    let x = r.unsigned_u256(bits)?;
                    if x > max_high {
                        return Err(Error::Encoding);
                    }
                    Ok(x)
                })
                .collect::<Result<Vec<_>, Error>>()?;
            t_a.push(Poly::from_canonical(self.ring.clone(), coefficients));
        }
        Ok(Commitment {
            t_a: PolyVec::new(self.ring.clone(), t_a)?,
            t_b: fields::read_below(r, &self.ring, messages, &self.ring.q)?,
        })
    }
    /// The serialization of high parts $`w_1`$, each in $`[0,m)`$ in $`\mathrm{bitlen}(m-1)`$
    /// bits for $`m=(q-1)/\gamma`$: the opening proof's first message, absorbed and never sent.
    pub(crate) fn high_bytes(&self, w1: &PolyVec) -> Result<Vec<u8>, Error> {
        let mut w = BitWriter::new();
        fields::write_below(&mut w, w1, &self.compression.hint_modulus())?;
        w.into_bytes()
    }
    /// The challenge after the first message `w1` of the opening proof on its own: on a copy of
    /// `transcript`, absorb [`Self::high_bytes`] of `w1` and squeeze a challenge
    /// ([`Self::squeeze_challenge`]). Prover and verifier call this same function.
    pub(crate) fn challenge(&self, transcript: &Transcript, w1: &PolyVec) -> Result<Poly, Error> {
        let mut transcript = transcript.clone();
        transcript.absorb(&self.high_bytes(w1)?);
        self.squeeze_challenge(&mut transcript)
    }
    /// Test only: the bytes that [`Self::challenge`] decodes its challenge from.
    #[cfg(test)]
    pub(crate) fn challenge_bytes(
        &self,
        transcript: &Transcript,
        w1: &PolyVec,
    ) -> Result<Vec<u8>, Error> {
        let mut transcript = transcript.clone();
        transcript.absorb(&self.high_bytes(w1)?);
        let mut bytes = vec![0u8; crate::transcript::CHALLENGE_BYTES];
        transcript.squeeze(&mut bytes);
        Ok(bytes)
    }
    /// The challenge of the set (LNP22 §2.7, with this set's $`\omega`$ and $`\eta`$) decoded
    /// from the transcript's next `CHALLENGE_BYTES` bytes (`rand::challenge_from_bytes`): a
    /// challenge within $`\eta`$ in every case.
    pub(crate) fn squeeze_challenge(&self, transcript: &mut Transcript) -> Result<Poly, Error> {
        transcript.challenge(self.ring.clone(), self.checked.omega, self.checked.eta)
    }
    fn rejection_constants(&self) -> Result<[U256; 2], Error> {
        let p = &self.parameters;
        let alpha = p.alpha_squared as f64 + (p.l2_rows.len() * p.degree) as f64;
        let s1 = 1.55 * (2.0f64).powi(p.log_sigma[0] as i32);
        let s2 = 1.55 * (2.0f64).powi(p.log_sigma[1] as i32);
        let g1 = s1 / (self.checked.eta as f64 * alpha.sqrt());
        let g2 = s2 / (self.checked.eta as f64 * ((p.m2 * p.degree) as f64).sqrt());
        // Integral ceilings are conservative M values: lower acceptance, same rejection law.
        let m1 = (((258.0 * std::f64::consts::LN_2).sqrt() / g1 + 0.5 / g1.powi(2)).exp() + 1e-10)
            .ceil();
        let m2 = ((0.5 / g2.powi(2)).exp() + 1e-10).ceil();
        if m1 >= 65536.0 || m2 >= 65536.0 {
            return Err(Error::Parameter("rejection constant capacity"));
        }
        Ok([
            U256::from(m1 as u64).shl_vartime(128),
            U256::from(m2 as u64).shl_vartime(128),
        ])
    }
    /// Produce a Fig. 18 proof, using Rej_1 for z1 and paper Rej_2 for z2.
    /// The per-round rejection loop is bounded at 4096 attempts.
    ///
    /// The seed must be secret and uniformly random, and it may be reused, also for the
    /// commitment being proven. The masks are read under a key derived from the seed, the
    /// transcript before the first message (parameters, public seed, context and commitment)
    /// and the opening: identical calls return identical proofs, and calls that differ in any
    /// input read independent masks (assuming SHAKE128 and AES-256 behave as pseudorandom
    /// functions).
    ///
    /// With a reused seed, zero-knowledge holds only relative to the equality pattern of the
    /// inputs: identical inputs give identical proofs. Uses that need multi-theorem
    /// zero-knowledge, and settings where faults can be injected (a fault that changes a
    /// challenge but not the key can reveal the witness), should use `prove`, which draws a
    /// fresh seed per call.
    pub fn prove_with_seed(
        &self,
        commitment: &Commitment,
        opening: &Opening,
        context: &[u8],
        mut seed: [u8; 32],
    ) -> Result<OpeningProof, Error> {
        self.prove_seeded(commitment, opening, context, &take_seed(&mut seed))
    }
    fn prove_seeded(
        &self,
        commitment: &Commitment,
        opening: &Opening,
        context: &[u8],
        seed: &Zeroizing<[u8; 32]>,
    ) -> Result<OpeningProof, Error> {
        let transcript = self.opening_transcript(commitment, context)?;
        let (proof, ()) = self.prove_core(
            commitment,
            opening,
            Caller::Opening,
            &transcript.binding(),
            seed,
            |_, _, w1| Ok((self.challenge(&transcript, w1)?, ())),
        )?;
        Ok(proof)
    }
    /// The key of the masks and rejection coins, bound to the proof that runs the core and to
    /// the transcript digest `binding` that precedes the first prover message.
    pub(crate) fn opening_key(
        opening: &Opening,
        caller: Caller,
        binding: &[u8; 32],
        seed: &[u8; 32],
    ) -> Zeroizing<[u8; 32]> {
        // "/v3" since the duplex-sponge transcript and the fixed-width codes of version 0.2,
        // "/v2" since the 256-bit sampler, the 256-bit rejection coins and the challenges
        // within eta. Each version reads these streams under its own key: masks that two
        // samplers read from one stream are correlated, and two proofs with correlated masks can
        // reveal the witness.
        Self::opening_key_labeled(b"abdlop/opening-proof/v3", opening, caller, binding, seed)
    }
    /// The key of [`Self::opening_key`] under another label. The extractable mode reads more
    /// streams per attempt (the LinHC masks and coin), so it takes its own label,
    /// `abdlop/opening-proof-sl/v2`.
    pub(crate) fn opening_key_labeled(
        label: &[u8],
        opening: &Opening,
        caller: Caller,
        binding: &[u8; 32],
        seed: &[u8; 32],
    ) -> Zeroizing<[u8; 32]> {
        derive_key(
            label,
            seed,
            &[
                caller.tag(),
                binding,
                &secret_bytes(&opening.s1),
                &secret_bytes(&opening.m),
                &secret_bytes(&opening.s2),
            ],
        )
    }
    /// The opening proof with a caller-defined challenge. `binding` must be the
    /// `Transcript::binding` of the transcript that the challenge extends, which binds
    /// everything it depends on besides the prover's messages, so that one key never meets two
    /// challenge functions.
    ///
    /// The challenge function returns the challenge with side data of its own, which is
    /// returned with the proof: that of the accepted attempt. With the `parallel` feature
    /// attempts run concurrently and some after the accepted one are computed, so the function
    /// is called for those too, in no particular order.
    pub(crate) fn prove_core<T: Send>(
        &self,
        commitment: &Commitment,
        opening: &Opening,
        caller: Caller,
        binding: &[u8; 32],
        seed: &Zeroizing<[u8; 32]>,
        challenge: impl Fn(&PolyVec, &PolyVec, &PolyVec) -> Result<(Poly, T), Error> + Sync,
    ) -> Result<(OpeningProof, T), Error> {
        self.prove_core_with_attempts(
            commitment,
            opening,
            caller,
            binding,
            seed,
            secret::MAX_ATTEMPTS,
            challenge,
        )
    }
    /// `prove_core` with at most `max_attempts` rejection-sampling attempts (and never more
    /// than `MAX_ATTEMPTS`, which bounds the mask domains). Tests lower the limit.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prove_core_with_attempts<T: Send>(
        &self,
        commitment: &Commitment,
        opening: &Opening,
        caller: Caller,
        binding: &[u8; 32],
        seed: &Zeroizing<[u8; 32]>,
        max_attempts: u32,
        challenge: impl Fn(&PolyVec, &PolyVec, &PolyVec) -> Result<(Poly, T), Error> + Sync,
    ) -> Result<(OpeningProof, T), Error> {
        let (_, proof, side) = self.prove_core_speculative(
            commitment,
            opening,
            caller,
            binding,
            seed,
            max_attempts,
            crate::par::speculation_width(),
            challenge,
        )?;
        Ok((proof, side))
    }
    /// `prove_core_with_attempts` with `width` attempts computed at a time, which also returns
    /// the index of the accepted attempt. The result does not depend on `width`.
    ///
    /// Attempt $`a`$ reads its masks from the streams of the words
    /// `OPENING_MASKS` $`+2a`$ and $`+2a+1`$ and its rejection coins, two 32-byte values
    /// ([`reject::coin`]: that of Rej_1 for $`z_1`$, then that of Rej_2 for $`z_2`$), from bytes
    /// $`64a`$ to $`64a+63`$ of the coin stream: an attempt that does not fail runs both
    /// rejection tests, so it reads exactly 64 bytes, and one that fails ends the loop. Its
    /// outcome is therefore a function of the key and $`a`$ alone, and the first attempt in
    /// order whose outcome is not a rejection (an accepted proof, or an error) gives the result
    /// of the sequential loop. The challenge function must not read the coin stream either.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prove_core_speculative<T: Send>(
        &self,
        commitment: &Commitment,
        opening: &Opening,
        caller: Caller,
        binding: &[u8; 32],
        seed: &Zeroizing<[u8; 32]>,
        max_attempts: u32,
        width: u32,
        challenge: impl Fn(&PolyVec, &PolyVec, &PolyVec) -> Result<(Poly, T), Error> + Sync,
    ) -> Result<(u32, OpeningProof, T), Error> {
        let (attempt, proof, _, side) = self.prove_core_ext(
            commitment,
            opening,
            caller,
            binding,
            seed,
            max_attempts,
            width,
            None,
            |y1, y21, w1, _| challenge(y1, y21, w1),
        )?;
        Ok((attempt, proof, side))
    }
    /// [`Self::prove_core_speculative`] with Katsumata's extractable commitment beside the
    /// opening when `ext` is given.
    ///
    /// The challenge function then also receives each attempt's LinHC mask commitment
    /// $`(w'_1,w'_2)`$ to $`(y_1\|y_{21})`$, which it must absorb before deriving the challenge,
    /// and an attempt is kept only if the LinHC opening passes its own rejection test too. The
    /// attempt reads its LinHC masks from the words `LINHC_MASKS` $`+3a`$ to $`+3a+2`$ and its
    /// LinHC coin from the stream `(LINHC_COINS, a)`, under a key with the label
    /// `abdlop/opening-proof-sl/v2`, so its outcome is still a function of the key and $`a`$.
    /// Without `ext` this is the plain opening proof, with its own key, streams and challenges.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(crate) fn prove_core_ext<T: Send>(
        &self,
        commitment: &Commitment,
        opening: &Opening,
        caller: Caller,
        binding: &[u8; 32],
        seed: &Zeroizing<[u8; 32]>,
        max_attempts: u32,
        width: u32,
        ext: Option<&linhc::ProverExt<'_>>,
        challenge: impl Fn(
            &PolyVec,
            &PolyVec,
            &PolyVec,
            Option<linhc::MaskCommitment<'_>>,
        ) -> Result<(Poly, T), Error>
        + Sync,
    ) -> Result<(u32, OpeningProof, Option<linhc::LinhcOpening>, T), Error> {
        let (expected, _) =
            self.commit_with_randomness(opening.s1.clone(), opening.m.clone(), opening.s2.clone())?;
        if expected != *commitment {
            return Err(Error::Witness);
        }
        let constants = self.rejection_constants()?;
        let key = match ext {
            None => Self::opening_key(opening, caller, binding, seed),
            Some(_) => Self::opening_key_labeled(
                b"abdlop/opening-proof-sl/v2",
                opening,
                caller,
                binding,
                seed,
            ),
        };
        let mut random = AesPrg::new(&key, domain(secret::OPENING_COINS, 0));
        let limit = max_attempts.min(secret::MAX_ATTEMPTS);
        let mut attempts =
            crate::par::Attempts::new(width, limit, &mut random, |attempt, coins, abandon| {
                self.attempt(
                    opening, &key, &constants, attempt, coins, ext, &challenge, abandon,
                )
            });
        // The outcomes in order: the first that is not a rejection decides.
        for attempt in 0..max_attempts.min(secret::MAX_ATTEMPTS) {
            if let Some((proof, linhc, side)) = attempts.outcome(attempt)? {
                return Ok((attempt, proof, linhc, side));
            }
        }
        Err(Error::RestartLimit)
    }
    /// One attempt of the opening proof, with its coin bytes, the coins of Rej_1 and Rej_2 in
    /// that order: `Ok(None)` if it is rejected, or abandoned because an earlier attempt of its
    /// batch has decided (`par::Attempts`).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn attempt<T>(
        &self,
        opening: &Opening,
        key: &[u8; 32],
        constants: &[U256; 2],
        attempt: u32,
        coins: &[u8; OPENING_COIN_BYTES],
        ext: Option<&linhc::ProverExt<'_>>,
        challenge: &impl Fn(
            &PolyVec,
            &PolyVec,
            &PolyVec,
            Option<linhc::MaskCommitment<'_>>,
        ) -> Result<(Poly, T), Error>,
        abandon: &crate::par::Abandon<'_>,
    ) -> Result<Option<(OpeningProof, Option<linhc::LinhcOpening>, T)>, Error> {
        if abandon.requested() {
            return Ok(None);
        }
        let Mask {
            values: y1,
            variance: variance1,
        } = sample_gaussian(
            self.ring.clone(),
            self.bounded_len(),
            self.parameters.log_sigma[0],
            key,
            secret::OPENING_MASKS + attempt * 2,
        )?;
        let Mask {
            values: y2,
            variance: variance2,
        } = sample_gaussian(
            self.ring.clone(),
            self.parameters.m2,
            self.parameters.log_sigma[1],
            key,
            secret::OPENING_MASKS + attempt * 2 + 1,
        )?;
        if abandon.requested() {
            return Ok(None);
        }
        let y21 = part(&y2, 0, self.a2.cols())?;
        let y22 = part(&y2, self.a2.cols(), y2.len())?;
        let w = add(&add(&self.a1.mul(&y1)?, &self.a2.mul(&y21)?)?, &y22)?;
        let w1 = map(&w, |x| self.compression.decompose(x).0)?;
        let w0 = map(&w, |x| {
            self.ring.reduce_i256(&self.compression.decompose(x).1)
        })?;
        if abandon.requested() {
            return Ok(None);
        }
        // The LinHC mask commitment to (y1 || y21), part of this attempt's first message.
        let linhc_mask = match ext {
            Some(ext) => {
                let r = linhc::message(ext.key.a.ring(), &y1, &y21)?;
                let dom = attempt
                    .checked_mul(3)
                    .and_then(|x| x.checked_add(secret::LINHC_MASKS))
                    .ok_or(Error::Overflow)?;
                Some(linhc::mask(ext.key, ext.params, &r, key, dom)?)
            }
            None => None,
        };
        let (c, side) = challenge(&y1, &y21, &w1, linhc_mask.as_ref().map(|m| (&m.w1, &m.w2)))?;
        if abandon.requested() {
            return Ok(None);
        }
        let v1 = scale(&opening.s1, &c)?;
        let v2 = scale(&opening.s2, &c)?;
        let z1 = add(&y1, &v1)?;
        let z2 = add(&y2, &v2)?;
        let mut keep = true;
        for (((z, v), (variance, m), policy), u) in [(&z1, &v1), (&z2, &v2)]
            .into_iter()
            .zip([variance1, variance2].into_iter().zip(*constants))
            .zip([Policy::Standard, Policy::Rej2])
            .map(|((zv, vm), policy)| (zv, vm, policy))
            .zip(coins.as_chunks::<{ reject::COIN_BYTES }>().0)
        {
            let (dot, norm) = reject::moments(&flatten(z)?, &flatten(v)?)?;
            keep &= reject::accept(policy, dot, norm, variance, m, reject::coin(u))?;
        }
        if !keep || !self.z1_within_bound(&z1)? {
            return Ok(None);
        }
        let z21 = part(&z2, 0, self.a2.cols())?;
        let z22 = sub(
            &sub(
                &part(&z2, self.a2.cols(), z2.len())?,
                &scale(&opening.low, &c)?,
            )?,
            &w0,
        )?;
        if !self.joint_within_bound(&z21, &z22)? {
            return Ok(None);
        }
        // The fixed-width codes: a response coefficient beyond its width restarts the attempt,
        // a test of the accepted response alone.
        if !fields::fits_signed(&z1, response_width(self.parameters.log_sigma[0]))
            || !fields::fits_signed(&z21, response_width(self.parameters.log_sigma[1]))
        {
            return Ok(None);
        }
        // The LinHC opening of c (s1 || s21) + (y1 || y21), with its own rejection coin.
        let linhc_opening = match (ext, &linhc_mask) {
            (Some(ext), Some(mask)) => {
                let c_lifted = linhc::lift_poly(ext.key.a.ring(), &c)?;
                let mut coin = [0u8; reject::COIN_BYTES];
                AesPrg::new(key, domain(secret::LINHC_COINS, attempt)).fill(&mut coin)?;
                match linhc::open(ext.checked, &c_lifted, ext.secret, mask, &coin)? {
                    Some(opening)
                        if opening.z.iter().all(|z| {
                            fields::fits_signed(z, response_width(ext.params.log_sigma))
                        }) =>
                    {
                        Some(opening)
                    }
                    _ => return Ok(None),
                }
            }
            _ => None,
        };
        let r = sub(&scale_scalar(&w1, &self.gamma())?, &z22)?;
        let hints = r
            .entries()
            .iter()
            .zip(z22.entries())
            .map(|(r, z)| {
                Poly::from_canonical(
                    self.ring.clone(),
                    r.values()
                        .zip(z.values())
                        .map(|(r, z)| self.ring.reduce_i256(&self.compression.make_hint(&z, &r)))
                        .collect(),
                )
            })
            .collect();
        let hint = PolyVec::new(self.ring.clone(), hints)?;
        // Hints beyond the hint code's width restart the attempt too: they are a function of
        // the accepted response and the public data (`docs/security.md`).
        if !fields::fits_signed(&hint, self.checked.hint_width) {
            return Ok(None);
        }
        Ok(Some((
            OpeningProof {
                challenge: c,
                z1,
                z21,
                hint,
            },
            linhc_opening,
            side,
        )))
    }
    /// Randomized proof generation with explicit caller RNG, under an application `context`
    /// that the verifier must supply too. Each call draws a fresh seed.
    pub fn prove(
        &self,
        commitment: &Commitment,
        opening: &Opening,
        context: &[u8],
        rng: &mut impl rand_core::CryptoRng,
    ) -> Result<OpeningProof, Error> {
        let mut seed = Zeroizing::new([0; 32]);
        rand_core::Rng::fill_bytes(rng, &mut *seed);
        self.prove_seeded(commitment, opening, context, &seed)
    }
    /// Verify a proof, including the complete `(z21,z22)` norm from Fig. 18.
    pub fn verify(
        &self,
        commitment: &Commitment,
        proof: &OpeningProof,
        context: &[u8],
    ) -> Result<(), Error> {
        let transcript = self.opening_transcript(commitment, context)?;
        self.verify_core(commitment, proof, |w1| self.challenge(&transcript, w1))
    }
    /// Whether `c` has the form of a challenge: in this ring, sigma-stable, with coefficients of
    /// absolute value at most omega and a zero coefficient at d/2. The last condition of the
    /// challenge set, the operator-norm bound eta, is not tested here: `verify_core` accepts
    /// only the challenge it derives, and the derivation (`Abdlop::squeeze_challenge`, through
    /// `rand::challenge_from_bytes`) never returns one above eta. A challenge above eta is therefore
    /// refused there, and the exact test, the costly part, runs once per derived draw rather
    /// than once more here.
    pub(crate) fn challenge_well_formed(&self, c: &Poly) -> bool {
        c.ring() == &self.ring
            && c.auto() == *c
            && c.norm_infinity() <= U256::from_u128(self.checked.omega as u128)
            && c.coefficient(self.ring.degree() / 2) == U256::ZERO
    }
    /// Whether $`\|z_1\|^2`$ is within `z1_bound_squared`. Prover and verifier share this test.
    pub(crate) fn z1_within_bound(&self, z1: &PolyVec) -> Result<bool, Error> {
        Ok(z1.norm_squared()? <= U256::from(self.checked.z1_bound_squared))
    }
    /// Whether the joint norm $`\|z_{21}\|^2+\|z_{22}\|^2`$ is within $`B^2`$ (Fig. 18). Prover
    /// and verifier share this test.
    pub(crate) fn joint_within_bound(&self, z21: &PolyVec, z22: &PolyVec) -> Result<bool, Error> {
        let norm = z21.norm_squared()?.saturating_add(&z22.norm_squared()?);
        Ok(norm <= U256::from(self.checked.b_squared))
    }
    /// The opening-proof verifier, with `challenge` deriving the challenge from the
    /// reconstructed `w1`. Before that, only the form of the proof's challenge is checked
    /// (`challenge_well_formed`); its bound eta holds because it must equal the derived one.
    /// `challenge` must therefore return challenges of the set, as `Abdlop::challenge` does;
    /// the tests that fix a challenge instead fix one within eta.
    pub(crate) fn verify_core(
        &self,
        commitment: &Commitment,
        proof: &OpeningProof,
        challenge: impl FnOnce(&PolyVec) -> Result<Poly, Error>,
    ) -> Result<(), Error> {
        self.verify_core_ext(commitment, proof, None, |w1, _| challenge(w1))
    }
    /// [`Self::verify_core`] with Katsumata's extractable commitment when `ext` is given: after
    /// the joint bound, the verifier recomputes the LinHC mask commitment $`(w'_1,w'_2)`$ from
    /// the proof's challenge, the response $`(z_1\|z_{21})`$ and the LinHC opening (checking
    /// the opening's block bounds), and the challenge function must absorb it before deriving
    /// the challenge it compares with the proof's.
    pub(crate) fn verify_core_ext(
        &self,
        commitment: &Commitment,
        proof: &OpeningProof,
        ext: Option<&linhc::VerifierExt<'_>>,
        challenge: impl FnOnce(&PolyVec, Option<linhc::MaskCommitment<'_>>) -> Result<Poly, Error>,
    ) -> Result<(), Error> {
        self.shape(&proof.z1, self.bounded_len())?;
        self.shape(&proof.z21, self.a2.cols())?;
        self.shape(&proof.hint, self.parameters.n_msis)?;
        let c = &proof.challenge;
        if !self.challenge_well_formed(c) {
            return Err(Error::InvalidProof);
        }
        if !self.z1_within_bound(&proof.z1)? {
            return Err(Error::InvalidProof);
        }
        let r = sub(
            &add(&self.a1.mul(&proof.z1)?, &self.a2.mul(&proof.z21)?)?,
            &scale(&commitment.t_a, &c.scale_u256(&self.rounding_power())?)?,
        )?;
        let w1 = self.use_hints(&proof.hint, &r)?;
        let z22 = sub(&r, &scale_scalar(&w1, &self.gamma())?)?;
        if !self.joint_within_bound(&proof.z21, &z22)? {
            return Err(Error::InvalidProof);
        }
        let mask = match ext {
            Some(ext) => {
                let ring = ext.key.a.ring();
                Some(linhc::recompute(
                    ext.key,
                    ext.params,
                    ext.checked,
                    &linhc::lift_poly(ring, c)?,
                    ext.commitment,
                    &linhc::message(ring, &proof.z1, &proof.z21)?,
                    ext.opening,
                )?)
            }
            None => None,
        };
        if challenge(&w1, mask.as_ref().map(|(u, v)| (u, v)))? != *c {
            return Err(Error::InvalidProof);
        }
        Ok(())
    }
    /// The compression divisor as a ring scalar.
    pub(crate) fn gamma(&self) -> U256 {
        U256::from_u64(self.parameters.gamma)
    }
    /// $`2^D`$, which the compression check keeps below q.
    pub(crate) fn rounding_power(&self) -> U256 {
        U256::ONE.shl_vartime(self.parameters.d_bits)
    }
    /// The largest high part of a canonical value, that of $`q-1`$.
    pub(crate) fn max_high(&self) -> U256 {
        self.compression
            .power2round(&self.ring.q.wrapping_sub(&U256::ONE))
            .0
    }
    /// UseHint on every coefficient: the high parts, as ring elements.
    pub(crate) fn use_hints(&self, hints: &PolyVec, r: &PolyVec) -> Result<PolyVec, Error> {
        let entries = r
            .entries()
            .iter()
            .zip(hints.entries())
            .map(|(r, h)| {
                Ok(Poly::from_canonical(
                    self.ring.clone(),
                    r.values()
                        .zip(h.values())
                        .map(|(r, h)| self.compression.use_hint(&self.ring.centred_i256(&h), &r))
                        .collect::<Result<_, _>>()?,
                ))
            })
            .collect::<Result<_, Error>>()?;
        PolyVec::new(self.ring.clone(), entries)
    }
    /// Write the hints in the hint code's width ([`CheckedParams::hint_width`]), each checked
    /// against the canonical hint interval $`(-m/2,m/2]`$.
    pub(crate) fn write_hints(&self, w: &mut BitWriter, hints: &PolyVec) -> Result<(), Error> {
        for p in hints.entries() {
            for x in p.values() {
                self.compression
                    .use_hint(&self.ring.centred_i256(&x), &U256::ZERO)?;
            }
        }
        fields::write_signed(w, hints, self.checked.hint_width)
    }
    /// Read `n` polynomials of hints written by [`Self::write_hints`], refusing one outside the
    /// canonical interval.
    pub(crate) fn read_hints(&self, r: &mut BitReader<'_>) -> Result<PolyVec, Error> {
        let hints = fields::read_signed(
            r,
            &self.ring,
            self.parameters.n_msis,
            self.checked.hint_width,
        )?;
        for p in hints.entries() {
            for i in 0..self.ring.degree() {
                let x = p.centred(i).ok_or(Error::Encoding)?;
                self.compression
                    .use_hint(&I256::from_i128(x), &U256::ZERO)?;
            }
        }
        Ok(hints)
    }
    /// Write the opening proof's last fields, as the compact flavor sends them: the challenge,
    /// then $`z_1`$ and $`z_{21}`$ in their response widths, then the hints.
    pub(crate) fn write_opening(
        &self,
        w: &mut BitWriter,
        proof: &OpeningProof,
    ) -> Result<(), Error> {
        self.shape(&proof.z1, self.bounded_len())?;
        self.shape(&proof.z21, self.a2.cols())?;
        self.shape(&proof.hint, self.parameters.n_msis)?;
        if proof.challenge.ring() != &self.ring {
            return Err(Error::RingMismatch);
        }
        fields::write_challenge(w, &proof.challenge, self.checked.omega)?;
        fields::write_signed(w, &proof.z1, response_width(self.parameters.log_sigma[0]))?;
        fields::write_signed(w, &proof.z21, response_width(self.parameters.log_sigma[1]))?;
        self.write_hints(w, &proof.hint)
    }
    /// Read the fields of [`Self::write_opening`].
    pub(crate) fn read_opening(&self, r: &mut BitReader<'_>) -> Result<OpeningProof, Error> {
        Ok(OpeningProof {
            challenge: fields::read_challenge(r, &self.ring, self.checked.omega)?,
            z1: fields::read_signed(
                r,
                &self.ring,
                self.bounded_len(),
                response_width(self.parameters.log_sigma[0]),
            )?,
            z21: fields::read_signed(
                r,
                &self.ring,
                self.a2.cols(),
                response_width(self.parameters.log_sigma[1]),
            )?,
            hint: self.read_hints(r)?,
        })
    }
    /// Encode the proof as its compact NARG string: the challenge, $`z_1`$, $`z_{21}`$ and the
    /// hints in their fixed widths, every length from the parameters.
    pub fn encode_proof(&self, proof: &OpeningProof) -> Result<Vec<u8>, Error> {
        let mut w = BitWriter::new();
        self.write_opening(&mut w, proof)?;
        w.into_bytes()
    }
    /// Decode the encoding of [`Self::encode_proof`] strictly: exact lengths, every field in its
    /// range, and no trailing bytes.
    pub fn decode_proof(&self, bytes: &[u8]) -> Result<OpeningProof, Error> {
        let mut reader = crate::codec::proof_reader(bytes)?;
        let proof = self.read_opening(&mut reader)?;
        reader.finish_bytes()?;
        Ok(proof)
    }
}

/// The 32-byte chain of the transcript of version 0.1 over the framed `(label, payload)` items:
/// from $`s=\mathrm{SHAKE128}(\texttt{LNP22-RUST-v1})`$, each item gives
/// $`s\leftarrow\mathrm{SHAKE128}(s\,\|\,\mathrm{LE64}(|label|)\,\|\,label\,\|\,
/// \mathrm{LE64}(|payload|)\,\|\,payload)`$, each hash taken to 32 bytes. Only
/// [`Abdlop::fingerprint`] uses it, so that the keys of `commit_with_seed` stay those of 0.1.
fn legacy_chain(items: &[(&[u8], &[u8])]) -> [u8; 32] {
    use shake::{ExtendableOutput, Shake128, Update, XofReader};
    let hash = |parts: &[&[u8]]| {
        let mut h = Shake128::default();
        for part in parts {
            h.update(part);
        }
        let mut out = [0u8; 32];
        h.finalize_xof().read(&mut out);
        out
    };
    let mut state = hash(&[b"LNP22-RUST-v1"]);
    for (label, payload) in items {
        state = hash(&[
            &state,
            &(label.len() as u64).to_le_bytes(),
            label,
            &(payload.len() as u64).to_le_bytes(),
            payload,
        ]);
    }
    state
}

#[cfg(test)]
mod seed_reuse_tests {
    use super::*;

    #[test]
    fn masks_never_read_the_commitment_streams_of_a_shared_seed() {
        // A caller may commit and prove with one seed. The masks must then come from streams
        // the commitment never reads: were y1 read from the stream of s2, the published
        // z1 = y1 + c s1 would be a function of the randomness that hides the message.
        let scheme = Abdlop::new([91; 32], crate::params::toy_d64()).unwrap();
        let ring = scheme.ring().clone();
        let s1 = PolyVec::new(
            ring.clone(),
            vec![Poly::constant(ring.clone(), 1); scheme.bounded_len()],
        )
        .unwrap();
        let m = PolyVec::new(
            ring.clone(),
            vec![Poly::constant(ring.clone(), 3); scheme.message_len()],
        )
        .unwrap();
        let seed = [93; 32];
        let key = scheme.commitment_key(&s1, &m, &seed);
        let (commitment, opening) = scheme.commit_with_seed(s1, m, seed).unwrap();
        let transcript = scheme.opening_transcript(&commitment, b"").unwrap();
        // The masks of every attempt computed, also speculative ones after the accepted one.
        let masks = std::sync::Mutex::new(Vec::new());
        let (proof, ()) = scheme
            .prove_core(
                &commitment,
                &opening,
                Caller::Opening,
                &transcript.binding(),
                &Zeroizing::new(seed),
                |y1, y21, w1| {
                    masks.lock().unwrap().push((y1.clone(), y21.clone()));
                    Ok((scheme.challenge(&transcript, w1)?, ()))
                },
            )
            .unwrap();
        scheme.verify(&commitment, &proof, b"").unwrap();
        let masks = masks.into_inner().unwrap();
        assert!(!masks.is_empty());
        // The mask polynomial that commitment stream i would give at width t.
        let from_commitment_stream = |i: usize, t: u32| {
            let stream = &mut AesPrg::new(&key, domain(secret::COMMITMENT, (i + 1) as u32));
            Poly::new(ring.clone(), gaussian(stream, t, ring.degree()).unwrap()).unwrap()
        };
        let t = scheme.parameters.log_sigma;
        for (y1, y21) in &masks {
            for (i, y) in y1.entries().iter().enumerate().take(scheme.parameters.m2) {
                assert_ne!(*y, from_commitment_stream(i, t[0]), "y1[{i}]");
            }
            for (i, y) in y21.entries().iter().enumerate() {
                assert_ne!(*y, from_commitment_stream(i, t[1]), "y2[{i}]");
            }
        }
    }
}
