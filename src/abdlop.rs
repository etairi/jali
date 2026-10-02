//! ABDLOP commitment and compressed opening proof (LNP22 Fig. 4 and Fig. 18).
//!
//! The verifier checks the joint norm of both randomness-response components. Transcript
//! prefixes bind all public parameters, the matrix seed, caller context and commitment.
use crate::{
    Error,
    codec::{BitReader, BitWriter},
    dcompress::Compression,
    math::{I256, Poly, PolyMat, PolyVec, Ring, U256, int},
    params::{CheckedParams, TboxParams},
    rand::{
        AesPrg, autostable, bounded, derive_key, domain, gaussian,
        reject::{self, Policy, Variance},
        secret, take_seed, uniform_ring,
    },
    transcript::Transcript,
};
use std::sync::Arc;
use zeroize::Zeroizing;

/// Public commitment with compressed Ajtai part and full BDLOP messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Commitment {
    /// High bits $`t_{A,1}`$, of length n.
    pub t_a: PolyVec,
    /// $`t_B`$, of length l.
    pub t_b: PolyVec,
}

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
        let binding = scheme.prefix(&commitment, b"").unwrap().digest();
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
        .map(|i| {
            let values = uniform_ring(
                &mut AesPrg::new(seed, domain(dom, i as u32)),
                ring,
                ring.degree(),
            )?;
            Ok(Poly::from_canonical(ring.clone(), values))
        })
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
        let mut values = Zeroizing::new(gaussian(
            &mut AesPrg::new(seed, domain(dom, (i + 1) as u32)),
            t,
            ring.degree(),
        )?);
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
pub(crate) fn scale(a: &PolyVec, b: &Poly) -> Result<PolyVec, Error> {
    PolyVec::new(
        a.ring().clone(),
        a.entries()
            .iter()
            .map(|a| a.mul(b))
            .collect::<Result<_, _>>()?,
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
            .map(|a| {
                Poly::from_canonical(a.ring().clone(), a.coefficients().iter().map(&f).collect())
            })
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
/// Encode each coefficient's centred representative modulo `modulus` with the uniform code.
/// With `modulus` equal to q that is the canonical representative.
pub(crate) fn encode_vec(w: &mut BitWriter, a: &PolyVec, modulus: &U256) -> Result<(), Error> {
    let ring = a.ring();
    let narrow = int::to_u128(modulus);
    let nonzero = crypto_bigint::NonZero::new(*modulus)
        .into_option()
        .ok_or(Error::Encoding)?;
    for p in a.entries() {
        for x in p.coefficients() {
            let value = if *modulus == ring.q {
                *x
            } else {
                // The only such caller encodes high parts in [0, m), m < q/2: never negative.
                let r = ring.magnitude(x).rem_vartime(&nonzero);
                if ring.is_negative(x) && r != U256::ZERO {
                    modulus.wrapping_sub(&r)
                } else {
                    r
                }
            };
            match narrow {
                Some(m) => w.uniform(int::low_u128(&value), m)?,
                None => w.uniform_u256(&value, modulus)?,
            }
        }
    }
    Ok(())
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
/// Gaussian codes of every centred coefficient; a coefficient outside `i128` cannot be one.
pub(crate) fn write_gaussians(w: &mut BitWriter, v: &PolyVec, t: u32) -> Result<(), Error> {
    for p in v.entries() {
        for x in p.coefficients() {
            let x = p.ring().centred_i128(x).ok_or(Error::Encoding)?;
            w.gaussian(x, t, 1 << 20)?;
        }
    }
    Ok(())
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
        for x in p.coefficients() {
            if wide {
                // The two's complement bits, wiped like the narrow value.
                let value = Zeroizing::new(*ring.centred_i256(x).as_uint());
                for word in value.as_words() {
                    out.extend_from_slice(&word.to_le_bytes());
                }
            } else {
                let value = Zeroizing::new(ring.centred_i128(x).expect("q below 2^128"));
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
    /// Digest of the parameters and the public seed.
    pub(crate) fn fingerprint(&self) -> [u8; 32] {
        Transcript::new(b"abdlop-scheme", &self.parameter_bytes(), &self.seed, b"").digest()
    }
    pub(crate) fn prefix(
        &self,
        commitment: &Commitment,
        context: &[u8],
    ) -> Result<Transcript, Error> {
        self.shape(&commitment.t_a, self.parameters.n_msis)?;
        self.shape(&commitment.t_b, self.parameters.l)?;
        let max_high = self.max_high();
        if commitment
            .t_a
            .entries()
            .iter()
            .any(|p| p.coefficients().iter().any(|x| *x > max_high))
        {
            return Err(Error::InvalidProof);
        }
        let mut t = Transcript::new(
            b"abdlop-opening",
            &self.parameter_bytes(),
            &self.seed,
            context,
        );
        let mut w = BitWriter::new();
        encode_vec(&mut w, &commitment.t_a, &self.ring.q)?;
        encode_vec(&mut w, &commitment.t_b, &self.ring.q)?;
        t.absorb(b"commitment", &w.finish());
        Ok(t)
    }
    pub(crate) fn challenge(&self, prefix: &Transcript, w1: &PolyVec) -> Result<Poly, Error> {
        let mut writer = BitWriter::new();
        encode_vec(&mut writer, w1, &self.compression.hint_modulus())?;
        let mut t = prefix.clone();
        t.absorb(b"w1", &writer.finish());
        autostable(
            &mut AesPrg::new(&t.challenge_seed(b"c"), 0),
            self.ring.clone(),
            self.checked.omega,
        )
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
        let prefix = self.prefix(commitment, context)?;
        let (proof, ()) = self.prove_core(
            commitment,
            opening,
            Caller::Opening,
            &prefix.digest(),
            seed,
            |_, _, w1| Ok((self.challenge(&prefix, w1)?, ())),
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
        derive_key(
            b"abdlop/opening-proof",
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
    /// The opening proof with a caller-defined challenge. `binding` must be the digest of a
    /// transcript that the challenge extends and that binds everything it depends on besides
    /// the prover's messages, so that one key never meets two challenge functions.
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
    /// `OPENING_MASKS` $`+2a`$ and $`+2a+1`$ and its rejection coins, two 16-byte values, from
    /// bytes $`32a`$ to $`32a+31`$ of the coin stream: an attempt that does not fail runs both
    /// rejection tests, so it reads exactly 32 bytes, and one that fails ends the loop. Its
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
        let (expected, _) =
            self.commit_with_randomness(opening.s1.clone(), opening.m.clone(), opening.s2.clone())?;
        if expected != *commitment {
            return Err(Error::Witness);
        }
        let constants = self.rejection_constants()?;
        let key = Self::opening_key(opening, caller, binding, seed);
        let mut random = AesPrg::new(&key, domain(secret::OPENING_COINS, 0));
        let limit = max_attempts.min(secret::MAX_ATTEMPTS);
        let mut attempts =
            crate::par::Attempts::new(width, limit, &mut random, |attempt, coins, abandon| {
                self.attempt(
                    opening, &key, &constants, attempt, coins, &challenge, abandon,
                )
            });
        // The outcomes in order: the first that is not a rejection decides.
        for attempt in 0..max_attempts.min(secret::MAX_ATTEMPTS) {
            if let Some((proof, side)) = attempts.outcome(attempt)? {
                return Ok((attempt, proof, side));
            }
        }
        Err(Error::RestartLimit)
    }
    /// One attempt of the opening proof, with its 32 coin bytes: `Ok(None)` if it is rejected,
    /// or abandoned because an earlier attempt of its batch has decided (`par::Attempts`).
    #[allow(clippy::too_many_arguments)]
    fn attempt<T>(
        &self,
        opening: &Opening,
        key: &[u8; 32],
        constants: &[U256; 2],
        attempt: u32,
        coins: &[u8; 32],
        challenge: &impl Fn(&PolyVec, &PolyVec, &PolyVec) -> Result<(Poly, T), Error>,
        abandon: &crate::par::Abandon<'_>,
    ) -> Result<Option<(OpeningProof, T)>, Error> {
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
        let (c, side) = challenge(&y1, &y21, &w1)?;
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
            .zip(coins.as_chunks::<16>().0)
        {
            let (dot, norm) = reject::moments(&flatten(z)?, &flatten(v)?)?;
            keep &= reject::accept(policy, dot, norm, variance, m, u128::from_le_bytes(*u))?;
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
        let r = sub(&scale_scalar(&w1, &self.gamma())?, &z22)?;
        let hints = r
            .entries()
            .iter()
            .zip(z22.entries())
            .map(|(r, z)| {
                Poly::from_canonical(
                    self.ring.clone(),
                    r.coefficients()
                        .iter()
                        .zip(z.coefficients())
                        .map(|(r, z)| self.ring.reduce_i256(&self.compression.make_hint(z, r)))
                        .collect(),
                )
            })
            .collect();
        Ok(Some((
            OpeningProof {
                challenge: c,
                z1,
                z21,
                hint: PolyVec::new(self.ring.clone(), hints)?,
            },
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
        let prefix = self.prefix(commitment, context)?;
        self.verify_core(commitment, proof, |w1| self.challenge(&prefix, w1))
    }
    /// Whether `c` lies in the challenge set: in this ring, sigma-stable, with coefficients of
    /// absolute value at most omega and a zero coefficient at d/2.
    pub(crate) fn challenge_in_set(&self, c: &Poly) -> bool {
        c.ring() == &self.ring
            && c.auto() == *c
            && c.norm_infinity() <= U256::from_u128(self.checked.omega as u128)
            && c.coefficients()[self.ring.degree() / 2] == U256::ZERO
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
    pub(crate) fn verify_core(
        &self,
        commitment: &Commitment,
        proof: &OpeningProof,
        challenge: impl FnOnce(&PolyVec) -> Result<Poly, Error>,
    ) -> Result<(), Error> {
        self.shape(&proof.z1, self.bounded_len())?;
        self.shape(&proof.z21, self.a2.cols())?;
        self.shape(&proof.hint, self.parameters.n_msis)?;
        let c = &proof.challenge;
        if !self.challenge_in_set(c) {
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
        if !self.joint_within_bound(&proof.z21, &z22)? || challenge(&w1)? != *c {
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
                    r.coefficients()
                        .iter()
                        .zip(h.coefficients())
                        .map(|(r, h)| self.compression.use_hint(&self.ring.centred_i256(h), r))
                        .collect::<Result<_, _>>()?,
                ))
            })
            .collect::<Result<_, Error>>()?;
        PolyVec::new(self.ring.clone(), entries)
    }
    /// The challenge code: each coefficient plus omega, in $`[0,2\omega]`$.
    pub(crate) fn write_challenge(&self, w: &mut BitWriter, c: &Poly) -> Result<(), Error> {
        let omega = self.checked.omega;
        for x in c.coefficients() {
            let x = self.ring.centred_i128(x).ok_or(Error::Encoding)?;
            if x.unsigned_abs() > omega as u128 {
                return Err(Error::Encoding);
            }
            w.uniform((x + omega) as u128, (2 * omega + 1) as u128)?;
        }
        Ok(())
    }
    /// The hint code, after checking the canonical hint interval.
    pub(crate) fn write_hints(&self, w: &mut BitWriter, hints: &PolyVec) -> Result<(), Error> {
        for p in hints.entries() {
            for x in p.coefficients() {
                self.compression
                    .use_hint(&self.ring.centred_i256(x), &U256::ZERO)?;
                w.hint(self.ring.centred_i128(x).ok_or(Error::Encoding)?, 1 << 20)?;
            }
        }
        Ok(())
    }
    /// Read one hint, bounded by $`m/2`$ and checked against the canonical interval.
    pub(crate) fn read_hint(&self, reader: &mut BitReader<'_>) -> Result<i128, Error> {
        let max_abs =
            int::to_u128(&self.compression.hint_modulus().shr_vartime(1)).unwrap_or(u128::MAX);
        let hint = reader.hint(max_abs, 1 << 20)?;
        self.compression
            .use_hint(&I256::from_i128(hint), &U256::ZERO)?;
        Ok(hint)
    }
    /// Encode the proof in a strict, parameter-sized bit stream.
    pub fn encode_proof(&self, proof: &OpeningProof) -> Result<Vec<u8>, Error> {
        self.shape(&proof.z1, self.bounded_len())?;
        self.shape(&proof.z21, self.a2.cols())?;
        self.shape(&proof.hint, self.parameters.n_msis)?;
        if proof.challenge.ring() != &self.ring {
            return Err(Error::RingMismatch);
        }
        let mut w = BitWriter::new();
        self.write_challenge(&mut w, &proof.challenge)?;
        self.write_hints(&mut w, &proof.hint)?;
        for (v, t) in [
            (&proof.z1, self.parameters.log_sigma[0]),
            (&proof.z21, self.parameters.log_sigma[1]),
        ] {
            write_gaussians(&mut w, v, t)?;
        }
        Ok(w.finish())
    }
    /// Decode with exact dimensions, canonical padding and no trailing bytes.
    pub fn decode_proof(&self, bytes: &[u8]) -> Result<OpeningProof, Error> {
        let mut reader = crate::codec::proof_reader(bytes)?;
        let challenge = Poly::new(
            self.ring.clone(),
            (0..self.ring.degree())
                .map(|_| {
                    Ok(
                        reader.uniform((2 * self.checked.omega + 1) as u128)? as i128
                            - self.checked.omega,
                    )
                })
                .collect::<Result<_, Error>>()?,
        )?;
        let hint = self.read_vector(&mut reader, self.parameters.n_msis, None)?;
        let z1 = self.read_vector(
            &mut reader,
            self.bounded_len(),
            Some(self.parameters.log_sigma[0]),
        )?;
        let z21 = self.read_vector(
            &mut reader,
            self.a2.cols(),
            Some(self.parameters.log_sigma[1]),
        )?;
        reader.finish()?;
        Ok(OpeningProof {
            challenge,
            z1,
            z21,
            hint,
        })
    }
    fn read_vector(
        &self,
        reader: &mut BitReader<'_>,
        len: usize,
        t: Option<u32>,
    ) -> Result<PolyVec, Error> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            let coeffs = (0..self.ring.degree())
                .map(|_| {
                    let x = if let Some(t) = t {
                        reader.gaussian(t, 1 << 20)?
                    } else {
                        self.read_hint(reader)?
                    };
                    if U256::from_u128(x.unsigned_abs()) > self.ring.half {
                        return Err(Error::Encoding);
                    }
                    Ok(x)
                })
                .collect::<Result<_, _>>()?;
            out.push(Poly::new(self.ring.clone(), coeffs)?);
        }
        PolyVec::new(self.ring.clone(), out)
    }
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
        let prefix = scheme.prefix(&commitment, b"").unwrap();
        // The masks of every attempt computed, also speculative ones after the accepted one.
        let masks = std::sync::Mutex::new(Vec::new());
        let (proof, ()) = scheme
            .prove_core(
                &commitment,
                &opening,
                Caller::Opening,
                &prefix.digest(),
                &Zeroizing::new(seed),
                |y1, y21, w1| {
                    masks.lock().unwrap().push((y1.clone(), y21.clone()));
                    Ok((scheme.challenge(&prefix, w1)?, ()))
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
