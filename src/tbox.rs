//! LNP22 toolbox reduction (Fig. 10): binary, exact norm and approximate range blocks.
//! The default transcript absorbs the application context and the statement before the first
//! challenge, range responses before Gamma, then h before mu.
use crate::{
    Error,
    abdlop::{self, Abdlop, Commitment},
    lnp::Statement,
    math::{
        Poly, PolyNtt, PolyVec, Ring, SparsePolyVec, U256,
        terms::{Accumulator, Term},
    },
    quad::{self, QuadEq},
    quad_eval::{self, EvalProof},
    quad_many,
    rand::{
        AesPrg, ByteStream, binomial, derive_key, domain,
        reject::{self, Policy},
        secret, take_seed,
    },
    transcript::Transcript,
};
use std::{collections::BTreeMap, sync::Arc};
use zeroize::Zeroizing;

#[cfg(test)]
mod encoding_reuse_tests;
#[cfg(test)]
mod lifting_soundness_tests;
#[cfg(test)]
mod linf_exact_soundness_tests;
#[cfg(test)]
mod linf_soundness_tests;
#[cfg(test)]
mod projection_tests;
#[cfg(test)]
mod range_tests;
#[cfg(test)]
mod restart_tests;
#[cfg(test)]
mod soundness_tests;
#[cfg(test)]
mod stream_reuse_tests;
#[cfg(test)]
mod two_prime_forgery_tests;
#[cfg(test)]
mod verifier_check_tests;

/// Toolbox proof, including the public commitment and range responses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    /// Commitment to the original and range-mask messages (before garbage extension).
    pub commitment: Commitment,
    /// Exact-range response, empty when no binary or exact block is present.
    pub z_exact: PolyVec,
    /// Approximate-range response, empty when that block is absent.
    pub z_approx: PolyVec,
    /// Evaluation proof binding every equation family.
    pub evaluation: EvalProof,
}
struct Forms {
    eqs: Vec<QuadEq>,
    evals: Vec<QuadEq>,
    exact: Vec<QuadEq>,
    approx: Vec<QuadEq>,
    sign_e: QuadEq,
    sign_d: QuadEq,
    mask_e: usize,
    mask_d: usize,
}
impl Forms {
    /// The four equation families, in the order the transcript absorbs them.
    fn families(&self) -> [&[QuadEq]; 4] {
        [&self.eqs, &self.evals, &self.exact, &self.approx]
    }
}
fn variable(scheme: &Abdlop, dimension: usize, index: usize) -> Result<QuadEq, Error> {
    let mut eq = QuadEq::zero(scheme.ring().clone(), dimension)?;
    eq.r1 = SparsePolyVec::new(
        scheme.ring().clone(),
        dimension,
        vec![(
            u16::try_from(2 * index).map_err(|_| Error::Dimension)?,
            Poly::constant(scheme.ring().clone(), 1),
        )],
    )?;
    Ok(eq)
}
fn constant(scheme: &Abdlop, dim: usize, p: Poly) -> Result<QuadEq, Error> {
    let mut eq = QuadEq::zero(scheme.ring().clone(), dim)?;
    eq.r0 = p;
    Ok(eq)
}
fn forms(scheme: &Abdlop, extended: &Abdlop, statement: &Statement) -> Result<Forms, Error> {
    statement.check(scheme)?;
    let p = &scheme.parameters;
    let ring = scheme.ring();
    let dim = 2 * (extended.bounded_len() + extended.message_len());
    if dim > u16::MAX as usize + 1 {
        return Err(Error::Dimension);
    }
    let mask_e = scheme.bounded_len() + p.l;
    let mask_d = mask_e
        + if scheme.checked.n_ex > 0 {
            256 / p.degree
        } else {
            0
        };
    let sign = mask_d + if p.n_prime > 0 { 256 / p.degree } else { 0 };
    let packed = variable(scheme, dim, sign)?;
    let half = Poly::constant(ring.clone(), 1).rotate((p.degree / 2) as i64);
    let sign_e = packed.trace()?;
    let sign_d = packed.scale(&half)?.trace()?;
    let indices: Vec<usize> = (0..2 * (p.m1 + p.l))
        .map(|i| {
            if i < 2 * p.m1 {
                i
            } else {
                i + 2 * p.l2_rows.len()
            }
        })
        .collect();
    let mut eqs = statement
        .quadratic
        .iter()
        .map(|eq| eq.remap(&indices, dim))
        .collect::<Result<Vec<_>, _>>()?;
    let mut evals = statement
        .evaluation
        .iter()
        .map(|eq| eq.remap(&indices, dim))
        .collect::<Result<Vec<_>, _>>()?;
    for sign in [&sign_e, &sign_d] {
        let mut equation = sign.product_affine(sign)?;
        equation.r0 = equation.r0.sub(&Poly::constant(ring.clone(), 1))?;
        eqs.push(equation);
        for j in 1..p.degree {
            evals.push(sign.scale(&Poly::constant(ring.clone(), 1).rotate(-(j as i64)))?);
        }
    }
    let binary = statement
        .binary
        .as_ref()
        .map(|block| block.forms(scheme, dim))
        .transpose()?
        .unwrap_or_default();
    let slack = (0..p.l2_rows.len())
        .map(|i| variable(scheme, dim, p.m1 + i))
        .collect::<Result<Vec<_>, _>>()?;
    let mut binary_eq = QuadEq::zero(ring.clone(), dim)?;
    let ones = Poly::new(ring.clone(), vec![-1; p.degree])?;
    for affine in &binary {
        let shifted = affine.add(&constant(scheme, dim, ones.clone())?)?;
        binary_eq.add_product_affine(&affine.conjugate()?, &shifted)?;
    }
    if !binary.is_empty() {
        evals.push(binary_eq);
    }
    // Keep each slack polynomial's binary equation separate: the parameter checker uses
    // sqrt(d), not sqrt(Z*d), in its slack-lifting inequality.
    for affine in &slack {
        let shifted = affine.add(&constant(scheme, dim, ones.clone())?)?;
        evals.push(affine.conjugate()?.product_affine(&shifted)?);
    }
    let mut exact = binary;
    for (i, block) in statement.l2.iter().enumerate() {
        let block_forms = block.map.forms(scheme, dim)?;
        let mut norm = QuadEq::zero(ring.clone(), dim)?;
        for affine in &block_forms {
            norm.add_product_affine(&affine.conjugate()?, affine)?;
        }
        let mut powers = vec![0i128; p.degree];
        for (j, value) in powers.iter_mut().enumerate().take(64) {
            if (1u64 << j) <= block.bound_squared {
                *value = 1i128 << j;
            }
        }
        let powers = Poly::new(ring.clone(), powers)?.auto();
        norm.add_scaled_assign(&slack[i], &powers)?;
        norm.r0 = norm
            .r0
            .sub(&Poly::constant(ring.clone(), block.bound_squared.into()))?;
        evals.push(norm);
        exact.extend(block_forms);
    }
    exact.extend(slack);
    let approx = statement
        .arp
        .as_ref()
        .map(|block| block.forms(scheme, dim))
        .transpose()?
        .unwrap_or_default();
    Ok(Forms {
        eqs,
        evals,
        exact,
        approx,
        sign_e,
        sign_d,
        mask_e,
        mask_d,
    })
}
/// The toolbox's version tag, which its transcripts take as their statement field. Version 3
/// draws its masks with the 256-bit Gaussian sampler and its challenges within eta.
const TOOLBOX_TAG: &[u8] = b"LNP22-toolbox-v3";
fn round_prefix(
    scheme: &Abdlop,
    commitment: &Commitment,
    forms: &Forms,
    context: &[u8],
) -> Result<Transcript, Error> {
    // The toolbox's version tag, then the application context right after the commitment.
    let mut prefix = scheme.prefix(commitment, TOOLBOX_TAG)?;
    prefix.absorb(b"application-context", context);
    absorb_statement(&mut prefix, scheme, forms)?;
    Ok(prefix)
}
/// [`round_prefix`] from the forms' encodings: the same transcript, without encoding again.
fn round_prefix_encoded(
    scheme: &Abdlop,
    commitment: &Commitment,
    encoded: &EncodedForms,
    context: &[u8],
) -> Result<Transcript, Error> {
    let mut prefix = scheme.prefix(commitment, TOOLBOX_TAG)?;
    prefix.absorb(b"application-context", context);
    absorb_encoded_statement(&mut prefix, scheme, encoded);
    Ok(prefix)
}
const FAMILIES: [&[u8]; 4] = [b"quadratic", b"evaluation", b"exact-map", b"approx-map"];
/// The equation families and range parameters, which the verifier derives from the statement.
fn absorb_statement(prefix: &mut Transcript, scheme: &Abdlop, forms: &Forms) -> Result<(), Error> {
    for (label, eqs) in FAMILIES.into_iter().zip(forms.families()) {
        quad_many::absorb_equations(prefix, label, eqs)?;
    }
    absorb_range_parameters(prefix, scheme);
    Ok(())
}
/// [`absorb_statement`] from the forms' encodings.
fn absorb_encoded_statement(prefix: &mut Transcript, scheme: &Abdlop, encoded: &EncodedForms) {
    for (label, eqs) in FAMILIES.into_iter().zip(&encoded.0) {
        quad_many::absorb_encoded(prefix, label, eqs);
    }
    absorb_range_parameters(prefix, scheme);
}
fn absorb_range_parameters(prefix: &mut Transcript, scheme: &Abdlop) {
    let mut params = Vec::new();
    for x in [
        scheme.checked.z3_bound_squared,
        scheme.checked.z4_bound,
        scheme.parameters.linf_bound as u128,
        scheme.parameters.log_sigma[2] as u128,
        scheme.parameters.log_sigma[3] as u128,
        scheme.checked.lambda as u128,
    ] {
        params.extend_from_slice(&x.to_le_bytes());
    }
    prefix.absorb(b"range-parameters", &params);
}
/// The encodings of the four equation families, computed once per proof: the toolbox key and
/// the transcript of every range attempt absorb them.
struct EncodedForms([Vec<Vec<u8>>; 4]);
impl EncodedForms {
    fn new(forms: &Forms) -> Result<Self, Error> {
        let [eqs, evals, exact, approx] = forms.families();
        Ok(Self([
            quad_many::encode_equations(eqs)?,
            quad_many::encode_equations(evals)?,
            quad_many::encode_equations(exact)?,
            quad_many::encode_equations(approx)?,
        ]))
    }
}
/// The toolbox key: the seed, the scheme, the statement, the context and the witness. Calls
/// that agree on all of these commit identically and so read identical transcripts.
fn toolbox_key(
    scheme: &Abdlop,
    extended: &Abdlop,
    encoded: &EncodedForms,
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
    absorb_encoded_statement(&mut statement, extended, encoded);
    // "/v2" since the 256-bit sampler, the 256-bit rejection coins and the challenges
    // within eta. The earlier version read these streams differently: masks that two
    // samplers read from one stream are correlated, and two proofs with correlated masks can
    // reveal the witness.
    Ok(derive_key(
        b"tbox/proof/v2",
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
fn row(seed: &[u8; 32], exact: bool, index: usize, count: usize) -> Result<Vec<i128>, Error> {
    binomial(
        &mut AesPrg::new(
            seed,
            domain(if exact { 0x52584500 } else { 0x52584400 }, index as u32),
        ),
        1,
        count,
    )
}
fn projections(seed: &[u8; 32], exact: bool, values: &[i128]) -> Result<Vec<i128>, Error> {
    let compute_row = |i| {
        let row = row(seed, exact, i, values.len())?;
        row.iter().zip(values).try_fold(0i128, |sum, (r, v)| {
            sum.checked_add(r.checked_mul(*v).ok_or(Error::Overflow)?)
                .ok_or(Error::Overflow)
        })
    };
    // In parallel with the `parallel` feature, in order.
    crate::par::try_map(256, compute_row)
}
/// The response of one range block, or `None` to restart: the bimodal test reads one 256-bit
/// coin ([`reject::coin`]) from `random`, after the check against q/2 and before the bound.
fn range_response(
    scheme: &Abdlop,
    seed: &[u8; 32],
    exact: bool,
    values: &[i128],
    mask: &abdlop::Mask,
    sign: i128,
    random: &mut impl ByteStream,
) -> Result<Option<PolyVec>, Error> {
    if values.is_empty() {
        return Ok(Some(PolyVec::zero(scheme.ring().clone(), 0)));
    }
    let mut v = Zeroizing::new(projections(seed, exact, values)?);
    for x in v.iter_mut() {
        *x *= sign;
    }
    let y = abdlop::flatten(&mask.values)?;
    let z = Zeroizing::new(
        y.iter()
            .zip(v.iter())
            .map(|(a, b)| a.checked_add(*b).ok_or(Error::Overflow))
            .collect::<Result<Vec<_>, _>>()?,
    );
    if z.iter()
        .any(|x| U256::from_u128(x.unsigned_abs()) > scheme.ring().half)
    {
        return Ok(None);
    }
    let m = range_rejection_m(scheme, exact)?;
    let (dot, norm) = reject::moments(&z, &v)?;
    let mut u = Zeroizing::new([0u8; reject::COIN_BYTES]);
    random.fill(&mut *u)?;
    if !reject::accept(
        Policy::Bimodal,
        dot,
        norm,
        mask.variance,
        U256::from(m).shl_vartime(128),
        reject::coin(&u),
    )? {
        return Ok(None);
    }
    if exact && crate::math::int::squared_norm(&z)? > U256::from(scheme.checked.z3_bound_squared) {
        return Ok(None);
    }
    if !exact && z.iter().any(|x| x.unsigned_abs() > scheme.checked.z4_bound) {
        return Ok(None);
    }
    Ok(Some(PolyVec::new(
        scheme.ring().clone(),
        z.chunks_exact(scheme.ring().degree())
            .map(|p| Poly::new(scheme.ring().clone(), p.to_vec()))
            .collect::<Result<_, _>>()?,
    )?))
}
/// The bimodal constant for one range block, from the Euclidean bound on its projected vector.
/// For the approximate block that bound is $`\sqrt{n'd}`$ times `linf_bound`, not `linf_bound`.
fn range_rejection_m(scheme: &Abdlop, exact: bool) -> Result<u64, Error> {
    let (t, alpha_squared) = if exact {
        (
            scheme.parameters.log_sigma[2],
            scheme.checked.exact_alpha_squared,
        )
    } else {
        (
            scheme.parameters.log_sigma[3],
            scheme.checked.approx_alpha_squared,
        )
    };
    crate::params::range_rejection_constant(t, alpha_squared)
}
fn projection_equations(
    scheme: &Abdlop,
    forms: &Forms,
    seed: &[u8; 32],
    ze: &PolyVec,
    zd: &PolyVec,
) -> Result<Vec<QuadEq>, Error> {
    let dim = 2 * (scheme.bounded_len() + scheme.message_len());
    let ring = scheme.ring();
    let mut equations = forms.evals.clone();
    for (exact, affines, sign, offset, z) in [
        (true, &forms.exact, &forms.sign_e, forms.mask_e, ze),
        (false, &forms.approx, &forms.sign_d, forms.mask_d, zd),
    ] {
        if affines.is_empty() {
            if !z.is_empty() {
                return Err(Error::Dimension);
            }
            continue;
        }
        if z.len() != 256 / ring.degree() || z.ring() != ring {
            return Err(Error::Dimension);
        }
        let response = abdlop::flatten(z)?;
        let prepared = Prepared::new(ring, dim, affines)?;
        // Each row reads its own stream: the rows are computed in parallel (`parallel`) and
        // kept in order.
        let rows = crate::par::try_map(256, |i| {
            let r = row(seed, exact, i, affines.len() * ring.degree())?;
            let rows = r
                .chunks_exact(ring.degree())
                .map(|coeffs| Ok(Poly::new(ring.clone(), coeffs.to_vec())?.auto()))
                .collect::<Result<Vec<_>, Error>>()?;
            let projection = match &prepared {
                Some(prepared) => prepared.projection(ring, dim, &rows)?,
                None => projection(ring, dim, affines, &rows)?,
            };
            let mask = variable(scheme, dim, offset + i / ring.degree())?
                .scale(&Poly::constant(ring.clone(), 1).rotate(-((i % ring.degree()) as i64)))?;
            // In place: the keys and values of `sign.product_affine(&projection)?.add(&mask)?`.
            let mut equation = QuadEq::zero(ring.clone(), dim)?;
            equation.add_product_affine(sign, &projection)?;
            equation.add_assign(&mask)?;
            equation
                .r0
                .sub_assign(&Poly::constant(ring.clone(), response[i]))?;
            Ok(equation)
        })?;
        equations.extend(rows);
    }
    Ok(equations)
}
/// A projection row $`\sum_j\sigma(r_j)f_j`$ of the forms $`f_j`$, with the rows
/// $`\sigma(r_j)`$ given: the keys of every form and the exact sums, as the sum of
/// `f_j.scale(sigma(r_j))` over $`j`$ returns them.
fn projection(
    ring: &Arc<Ring>,
    dim: usize,
    affines: &[QuadEq],
    rows: &[Poly],
) -> Result<QuadEq, Error> {
    let mut projection = QuadEq::zero(ring.clone(), dim)?;
    for (affine, row) in affines.iter().zip(rows) {
        projection.add_scaled_assign(affine, row)?;
    }
    Ok(projection)
}
/// The forms of a range block prepared for its 256 projection rows: each coefficient is
/// classified once and, if general, transformed once, and the coefficients are grouped by key.
/// A row then transforms each $`\sigma(r_j)`$ at most once and reduces each key's sum of
/// general products once per [`NADDS`](crate::params::moduli::NADDS) products; the scalar and
/// monomial ones are taken in the coefficient domain. The keys and values are those of
/// [`projection`].
struct Prepared {
    /// Every key of any form, with each form's coefficient there.
    columns: Vec<(u16, Vec<(usize, Term)>)>,
    /// Each form's constant.
    constants: Vec<(usize, Term)>,
}
impl Prepared {
    /// `None` when a form has quadratic entries, which [`projection`] then handles.
    fn new(ring: &Arc<Ring>, dim: usize, affines: &[QuadEq]) -> Result<Option<Self>, Error> {
        for affine in affines {
            affine.check(ring, dim)?;
        }
        if affines.iter().any(|a| a.r2.entries().next().is_some()) {
            return Ok(None);
        }
        let mut columns: BTreeMap<u16, Vec<(usize, Term)>> = BTreeMap::new();
        let mut constants = Vec::with_capacity(affines.len());
        for (j, affine) in affines.iter().enumerate() {
            for (key, p) in affine.r1.entries() {
                columns.entry(key).or_default().push((j, Term::new(p)));
            }
            constants.push((j, Term::new(&affine.r0)));
        }
        Ok(Some(Self {
            columns: columns.into_iter().collect(),
            constants,
        }))
    }
    /// $`\sum_jt_j\,\mathrm{rows}_j`$ over the terms $`(j,t_j)`$, exactly.
    fn combine(
        ring: &Arc<Ring>,
        terms: &[(usize, Term)],
        rows: &[Poly],
        rows_ntt: &mut [Option<PolyNtt>],
    ) -> Result<Poly, Error> {
        let mut out = Poly::zero(ring.clone());
        let mut sum = Accumulator::new(ring);
        for (j, term) in terms {
            match term {
                Term::Cheap(shape) => {
                    shape.add_product_to(&mut out, &rows[*j])?;
                }
                Term::General(a) => {
                    let b = rows_ntt[*j].get_or_insert_with(|| rows[*j].to_ntt());
                    sum.mac(a, b, &mut out)?;
                }
            }
        }
        sum.finish_into(&mut out)?;
        Ok(out)
    }
    fn projection(&self, ring: &Arc<Ring>, dim: usize, rows: &[Poly]) -> Result<QuadEq, Error> {
        let mut rows_ntt = vec![None; rows.len()];
        let entries = self
            .columns
            .iter()
            .map(|(key, terms)| Ok((*key, Self::combine(ring, terms, rows, &mut rows_ntt)?)))
            .collect::<Result<Vec<_>, Error>>()?;
        let mut out = QuadEq::zero(ring.clone(), dim)?;
        out.r1 = SparsePolyVec::new(ring.clone(), dim, entries)?;
        out.r0 = Self::combine(ring, &self.constants, rows, &mut rows_ntt)?;
        Ok(out)
    }
}
fn bind_responses(prefix: &mut Transcript, ze: &PolyVec, zd: &PolyVec) -> Result<(), Error> {
    let mut writer = crate::codec::BitWriter::new();
    abdlop::encode_vec(&mut writer, ze, &ze.ring().modulus())?;
    abdlop::encode_vec(&mut writer, zd, &zd.ring().modulus())?;
    prefix.absorb(b"range-responses-before-Gamma", &writer.finish());
    Ok(())
}

/// Prove a full statement, checking its original witness before inserting slack and masks.
/// The proof verifies only under the same application `context`.
///
/// The seed must be secret and uniformly random, and it may be reused. The commitment inside
/// the proof and every mask are read under keys derived from the seed, the parameters, the
/// public seed, the statement, the context and the witness: identical calls return identical
/// proofs, and calls that differ in any input read independent randomness (assuming SHAKE128
/// and AES-256 behave as pseudorandom functions).
///
/// With a reused seed, zero-knowledge holds only relative to the equality pattern of the
/// inputs: identical inputs give identical proofs. Uses that need multi-theorem
/// zero-knowledge, and settings where faults can be injected (a fault that changes a challenge
/// but not the key can reveal the witness), need a fresh seed per call, as `lnp::Prover::prove`
/// and `statement::Compiled::prove` draw from a `rand_core::CryptoRng`.
pub fn prove_with_seed(
    scheme: &Abdlop,
    statement: &Statement,
    s1: &PolyVec,
    m: &PolyVec,
    context: &[u8],
    mut seed: [u8; 32],
) -> Result<Proof, Error> {
    prove_seeded(scheme, statement, s1, m, context, &take_seed(&mut seed))
}
pub(crate) fn prove_seeded(
    scheme: &Abdlop,
    statement: &Statement,
    s1: &PolyVec,
    m: &PolyVec,
    context: &[u8],
    seed: &Zeroizing<[u8; 32]>,
) -> Result<Proof, Error> {
    prove_with_attempts(
        scheme,
        statement,
        s1,
        m,
        context,
        seed,
        secret::MAX_ATTEMPTS,
    )
}
/// `prove_seeded` with at most `max_attempts` range-round attempts (and never more than
/// `MAX_ATTEMPTS`, which bounds the mask domains). Tests lower the limit.
pub(crate) fn prove_with_attempts(
    scheme: &Abdlop,
    statement: &Statement,
    s1: &PolyVec,
    m: &PolyVec,
    context: &[u8],
    seed: &Zeroizing<[u8; 32]>,
    max_attempts: u32,
) -> Result<Proof, Error> {
    statement.check_witness(scheme, s1, m)?;
    prove_rounds(scheme, statement, s1, m, context, seed, max_attempts)
}
/// The prover after the witness check, which it does not repeat: a witness outside the
/// relation must not reach it, except in tests of what the verifier then does.
fn prove_rounds(
    scheme: &Abdlop,
    statement: &Statement,
    s1: &PolyVec,
    m: &PolyVec,
    context: &[u8],
    seed: &Zeroizing<[u8; 32]>,
    max_attempts: u32,
) -> Result<Proof, Error> {
    let p = &scheme.parameters;
    let ring = scheme.ring();
    let mut bounded = s1.entries().to_vec();
    for block in &statement.l2 {
        let norm = block.map.evaluate(s1, m)?.norm_squared()?;
        let bytes = norm.to_le_bytes();
        let norm = u64::from_le_bytes(bytes[..8].try_into().expect("fixed length"));
        let slack = block.bound_squared - norm;
        let mut coeffs = vec![0; p.degree];
        for (i, x) in coeffs.iter_mut().enumerate().take(64) {
            *x = i128::from((slack >> i) & 1);
        }
        bounded.push(Poly::new(ring.clone(), coeffs)?);
    }
    let bounded = PolyVec::new(ring.clone(), bounded)?;
    let e_slots = if scheme.checked.n_ex > 0 {
        256 / p.degree
    } else {
        0
    };
    let d_slots = if p.n_prime > 0 { 256 / p.degree } else { 0 };
    let extended = scheme.extend_messages(e_slots + d_slots + 1)?;
    let forms = forms(scheme, &extended, statement)?;
    let encoded = EncodedForms::new(&forms)?;
    let key = toolbox_key(scheme, &extended, &encoded, context, s1, m, seed)?;
    let (_, opening) =
        scheme.commit_in_domain(bounded.clone(), m.clone(), &key, secret::TOOLBOX_COMMITMENT)?;
    let mut random = AesPrg::new(&key, domain(secret::TOOLBOX, 0));
    for attempt in 0..max_attempts.min(secret::MAX_ATTEMPTS) {
        let ye = abdlop::sample_gaussian(
            ring.clone(),
            e_slots,
            p.log_sigma[2],
            &key,
            secret::RANGE_MASKS + 2 * attempt,
        )?;
        let yd = abdlop::sample_gaussian(
            ring.clone(),
            d_slots,
            p.log_sigma[3],
            &key,
            secret::RANGE_MASKS + 2 * attempt + 1,
        )?;
        let mut signs = [0u8; 1];
        random.fill(&mut signs)?;
        let be = if signs[0] & 1 == 0 { 1 } else { -1 };
        let bd = if signs[0] & 2 == 0 { 1 } else { -1 };
        let packed = Poly::constant(ring.clone(), be)
            .sub(&Poly::constant(ring.clone(), bd).rotate((p.degree / 2) as i64))?;
        let mut messages = m.entries().to_vec();
        messages.extend(ye.values.entries().iter().cloned());
        messages.extend(yd.values.entries().iter().cloned());
        messages.push(packed);
        let messages = PolyVec::new(ring.clone(), messages)?;
        let witness = quad::interleave(&bounded, &messages)?;
        let (commitment, full_opening) =
            extended.commit_with_randomness(bounded.clone(), messages, opening.s2.clone())?;
        let mut prefix = round_prefix_encoded(&extended, &commitment, &encoded, context)?;
        let projection_seed = prefix.challenge_seed(b"range-matrices");
        // Evaluated in parallel (`parallel`) and in order.
        let exact = PolyVec::new(
            ring.clone(),
            crate::par::try_map(forms.exact.len(), |i| forms.exact[i].evaluate(&witness))?,
        )?;
        let approx = PolyVec::new(
            ring.clone(),
            crate::par::try_map(forms.approx.len(), |i| forms.approx[i].evaluate(&witness))?,
        )?;
        let Some(ze) = range_response(
            scheme,
            &projection_seed,
            true,
            &abdlop::flatten(&exact)?,
            &ye,
            be,
            &mut random,
        )?
        else {
            continue;
        };
        let Some(zd) = range_response(
            scheme,
            &projection_seed,
            false,
            &abdlop::flatten(&approx)?,
            &yd,
            bd,
            &mut random,
        )?
        else {
            continue;
        };
        let evals = projection_equations(&extended, &forms, &projection_seed, &ze, &zd)?;
        bind_responses(&mut prefix, &ze, &zd)?;
        let mut proof_seed = Zeroizing::new([0; 32]);
        random.fill(&mut *proof_seed)?;
        #[cfg(test)]
        crate::rand::stream_log::child(&key, &proof_seed);
        // The encodings are not needed past the last range attempt.
        drop(encoded);
        let evaluation = quad_eval::prove_seeded(
            &extended,
            &commitment,
            &full_opening,
            &forms.eqs,
            &evals,
            &prefix.digest(),
            &proof_seed,
        )?;
        return Ok(Proof {
            commitment,
            z_exact: ze,
            z_approx: zd,
            evaluation,
        });
    }
    Err(Error::RestartLimit)
}

/// The verifier's checks of the range responses: one exact and one approximate slot per 256
/// projections when the block is present, the proof ring, $`\|z^{(e)}\|^2\le`$`z3_bound_squared`
/// and $`\|z^{(d)}\|_\infty\le`$`z4_bound`. Returns the two slot counts.
fn response_shape_and_bounds(scheme: &Abdlop, proof: &Proof) -> Result<(usize, usize), Error> {
    let p = &scheme.parameters;
    let e_slots = if scheme.checked.n_ex > 0 {
        256 / p.degree
    } else {
        0
    };
    let d_slots = if p.n_prime > 0 { 256 / p.degree } else { 0 };
    if proof.z_exact.len() != e_slots || proof.z_approx.len() != d_slots {
        return Err(Error::Dimension);
    }
    if proof.z_exact.ring() != scheme.ring() || proof.z_approx.ring() != scheme.ring() {
        return Err(Error::RingMismatch);
    }
    if proof.z_exact.norm_squared()? > U256::from(scheme.checked.z3_bound_squared)
        || proof
            .z_approx
            .entries()
            .iter()
            .any(|p| p.norm_infinity() > U256::from_u128(scheme.checked.z4_bound))
    {
        return Err(Error::InvalidProof);
    }
    Ok((e_slots, d_slots))
}

/// Verify bounds, reconstruct every equation, and verify the complete evaluation proof under
/// the application `context`.
pub fn verify(
    scheme: &Abdlop,
    statement: &Statement,
    proof: &Proof,
    context: &[u8],
) -> Result<(), Error> {
    statement.check(scheme)?;
    let (e_slots, d_slots) = response_shape_and_bounds(scheme, proof)?;
    let extended = scheme.extend_messages(e_slots + d_slots + 1)?;
    let forms = forms(scheme, &extended, statement)?;
    let mut prefix = round_prefix(&extended, &proof.commitment, &forms, context)?;
    let evals = projection_equations(
        &extended,
        &forms,
        &prefix.challenge_seed(b"range-matrices"),
        &proof.z_exact,
        &proof.z_approx,
    )?;
    bind_responses(&mut prefix, &proof.z_exact, &proof.z_approx)?;
    quad_eval::verify(
        &extended,
        &proof.commitment,
        &forms.eqs,
        &evals,
        &proof.evaluation,
        &prefix.digest(),
    )
}
