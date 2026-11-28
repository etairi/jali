//! LNP22 toolbox reduction (Fig. 10): binary, exact norm and approximate range blocks.
//!
//! One transcript runs through the whole proof, `JALI-V02-LNP22-TBOX-C1-SHAKE128`, or
//! `JALI-V02-LNP22-TBOX-SL-C1-SHAKE128` in the straight-line extractable mode, with the caller's
//! context in the tag. Its instance is the parameters, the public seed and the statement's four
//! equation families, then the extraction set in the extractable mode. The extractable mode
//! squeezes the LinHC key first. The prover messages and the verifier messages then follow in
//! protocol order: the commitment (with the LinHC commitment), the projection matrices, the
//! range responses and the garbage commitments, $`\Gamma`$, $`h`$, $`\mu`$, the final message
//! $`(t,v,w_1)`$ (with the LinHC mask commitment before $`w_1`$), and the challenge
//! (`docs/encodings.md`).
use crate::wipe::Secret;
use crate::{
    Error,
    abdlop::{self, Abdlop, Commitment},
    codec::{BitWriter, fields},
    linhc::{self, LinhcChecked, LinhcCommitment, LinhcKey, LinhcParams},
    lnp::Statement,
    math::{
        Poly, PolyNtt, PolyVec, Ring, SparsePolyVec, U256, int,
        terms::{Accumulator, Term},
        ternary::{PATTERN, pattern_signs},
    },
    params::{ExtractionParams, response_width},
    quad::{self, QuadEq},
    quad_eval::{self, Combination, EvalProof, EvalRows},
    quad_many,
    rand::{
        AesPrg, ByteStream, derive_key, domain,
        reject::{self, Policy},
        secret, take_seed,
    },
    transcript::{Protocol, Transcript},
};
use std::{borrow::Cow, collections::BTreeMap, sync::Arc};

#[cfg(test)]
mod extractable_tests;
#[cfg(test)]
mod io_pattern_tests;
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

/// A proof of the straight-line extractable mode: the toolbox proof with Katsumata's
/// extractable linear homomorphic commitment (LinHC, ePrint 2021/927, Sec. 3.4) beside its final
/// opening proof.
///
/// $`(t_1,t_2)`$ commits to the final opening's witness $`(s_1\|s_{21})`$ and
/// $`(z'_1,z'_2,z'_3)`$ opens the commitment of $`c(s_1\|s_{21})+(y_1\|y_{21})`$ to the
/// response $`(z_1\|z_{21})`$. They are over the LinHC ring
/// $`\mathbb Z_{q'}[X]/(X^d+1)`$ of the [`ExtractionParams`]. The toolbox proof's transcript
/// binds them, so it does not verify without them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtractableProof {
    /// The toolbox proof, under the extractable mode's transcript.
    pub proof: Proof,
    /// LinHC commitment $`t_1=p(As_1'+s_2')`$, of length $`m'`$.
    pub t1: PolyVec,
    /// LinHC commitment $`t_2=p(Bs_1'+s_3')+(s_1\|s_{21})`$, of length $`m'`$.
    pub t2: PolyVec,
    /// LinHC opening $`z'_1`$, of length the LinHC rank $`n`$.
    pub z1: PolyVec,
    /// LinHC opening $`z'_2`$, of length $`m'`$.
    pub z2: PolyVec,
    /// LinHC opening $`z'_3`$, of length $`m'`$.
    pub z3: PolyVec,
}

/// The prover's two modes: plain Fiat–Shamir, or straight-line extractable with Katsumata's
/// transform, with the LinHC key squeezed from the transcript unless one is given.
#[derive(Clone, Copy)]
enum Mode<'a> {
    Plain,
    Extractable {
        extraction: &'a ExtractionParams,
        key: Option<&'a LinhcKey>,
    },
}

/// What prover and verifier of the extractable mode share: the LinHC parameters and its key.
struct Extractable {
    params: LinhcParams,
    checked: LinhcChecked,
    key: LinhcKey,
}

/// The toolbox's transcript after its instance: the tag of the mode with `context`, the
/// parameters and the public seed of `scheme` (`Abdlop::absorb_scheme`), the four equation
/// families that `families` absorbs, in the order of [`Forms::families`], and in the
/// extractable mode the extraction set as `SerializeVarLenString`.
fn instance(
    scheme: &Abdlop,
    extraction: Option<&ExtractionParams>,
    context: &[u8],
    families: impl FnOnce(&mut Transcript) -> Result<(), Error>,
) -> Result<Transcript, Error> {
    let protocol = match extraction {
        None => Protocol::Toolbox,
        Some(_) => Protocol::ToolboxExtractable,
    };
    let mut transcript = Transcript::new(protocol, context)?;
    scheme.absorb_scheme(&mut transcript)?;
    families(&mut transcript)?;
    if let Some(extraction) = extraction {
        transcript.absorb_varlen(&extraction.transcript_bytes())?;
    }
    Ok(transcript)
}

/// The extractable mode's first verifier message: the LinHC key, squeezed from the transcript
/// after the instance ([`LinhcKey::squeeze`]), so that it is a random-oracle output on the
/// statement and the context. A given `key` (tests use a trapdoored one, as an extractor
/// programs the oracle) replaces the squeezed key, which is squeezed all the same.
fn extractable_setup(
    extended: &Abdlop,
    extraction: &ExtractionParams,
    transcript: &mut Transcript,
    key: Option<&LinhcKey>,
) -> Result<Extractable, Error> {
    let (params, checked) = extraction.linhc(&extended.parameters, &extended.checked)?;
    let squeezed = LinhcKey::squeeze(&params.ring()?, &params, transcript)?;
    Ok(Extractable {
        params,
        checked,
        key: key.cloned().unwrap_or(squeezed),
    })
}

/// The toolbox's first prover message: the commitment's serialization, then in the extractable
/// mode the LinHC commitment's (`linhc::write_pair`).
fn commitment_message(
    extended: &Abdlop,
    commitment: &Commitment,
    linhc_commitment: Option<&LinhcCommitment>,
) -> Result<Vec<u8>, Error> {
    let mut bytes = extended.commitment_bytes(commitment)?;
    if let Some(linhc_commitment) = linhc_commitment {
        let mut w = BitWriter::new();
        linhc::write_pair(&mut w, &linhc_commitment.t1, &linhc_commitment.t2)?;
        bytes.extend_from_slice(&w.into_bytes()?);
    }
    Ok(bytes)
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
/// Absorb the four equation families of the statement into the instance, in the order of
/// [`Forms::families`], each as `quad_many::absorb_equations`.
fn absorb_families(transcript: &mut Transcript, forms: &Forms) -> Result<(), Error> {
    for eqs in forms.families() {
        quad_many::absorb_equations(transcript, eqs)?;
    }
    Ok(())
}
/// The toolbox key under `label`: the seed, the transcript after the instance (the mode, the
/// context, the parameters, the public seed, the statement and in the extractable mode the
/// extraction set) through its `Transcript::binding`, and the witness. Calls that agree on all
/// of these commit identically and so read identical transcripts.
fn toolbox_key(
    label: &[u8],
    binding: &[u8; 32],
    s1: &PolyVec,
    m: &PolyVec,
    seed: &[u8; 32],
) -> Secret<[u8; 32]> {
    derive_key(
        label,
        seed,
        &[binding, &abdlop::secret_bytes(s1), &abdlop::secret_bytes(m)],
    )
}
/// The projection matrices of the range proofs, the toolbox's first verifier message: 256 rows
/// per block, one entry per coefficient of the block's vector, row-major, the exact block's
/// matrix first, each entry from 2 squeezed bits (`transcript::ternary_entry`). The squeezed
/// bytes are kept and rows decoded when read, so a matrix takes 64 bytes per column. An absent
/// block's matrix is empty, and a block of more than $`2^{24}`$ columns is refused.
struct Projections {
    exact: Vec<u8>,
    approx: Vec<u8>,
    exact_cols: usize,
    approx_cols: usize,
}
impl Projections {
    fn squeeze(transcript: &mut Transcript, forms: &Forms, degree: usize) -> Result<Self, Error> {
        let exact_cols = forms.exact.len() * degree;
        let approx_cols = forms.approx.len() * degree;
        if exact_cols.max(approx_cols) > 1 << 24 {
            return Err(Error::Dimension);
        }
        Ok(Self {
            exact: transcript.ternary_packed(256 * exact_cols)?,
            approx: transcript.ternary_packed(256 * approx_cols)?,
            exact_cols,
            approx_cols,
        })
    }
    /// Test only: matrices of `exact_cols` and `approx_cols` columns squeezed from a
    /// transcript whose context is `seed`.
    #[cfg(test)]
    fn for_tests(seed: &[u8; 32], exact_cols: usize, approx_cols: usize) -> Self {
        let mut transcript = Transcript::new(Protocol::Toolbox, seed).unwrap();
        Self {
            exact: transcript.ternary_packed(256 * exact_cols).unwrap(),
            approx: transcript.ternary_packed(256 * approx_cols).unwrap(),
            exact_cols,
            approx_cols,
        }
    }
    /// The packed matrix of the exact block, or of the approximate block, and its width.
    fn block(&self, exact: bool) -> (&[u8], usize) {
        if exact {
            (&self.exact, self.exact_cols)
        } else {
            (&self.approx, self.approx_cols)
        }
    }
    /// Row `index` of the exact block's matrix, or of the approximate block's.
    fn row(&self, exact: bool, index: usize) -> Vec<i8> {
        let (matrix, cols) = self.block(exact);
        (index * cols..(index + 1) * cols)
            .map(|i| crate::transcript::ternary_entry(matrix, i))
            .collect()
    }
}
/// The 256 projections $`\sum_jr_{ij}v_j`$ of `values` by the rows $`r_i`$ of a block's
/// matrix, exact (an overflow of `i128` is an error). Byte-aligned rows of values at most
/// $`(2^{63}-1)/n`$ in magnitude, for $`n`$ columns, are summed in `i64` straight from the
/// packed bytes, where no partial sum can overflow, the others by [`projections_checked`].
fn projections(matrices: &Projections, exact: bool, values: &[i128]) -> Result<Vec<i128>, Error> {
    let (matrix, cols) = matrices.block(exact);
    if values.len() != cols {
        return Err(Error::Dimension);
    }
    let max = values.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
    if !cols.is_multiple_of(4) || max > i64::MAX as u128 / cols.max(1) as u128 {
        return projections_checked(matrices, exact, values);
    }
    let words = Secret::new(values.iter().map(|v| *v as i64).collect::<Vec<_>>());
    let row_bytes = cols / 4;
    // In parallel with the `parallel` feature, in order.
    crate::par::try_map(256, |i| {
        Ok(i128::from(signed_sum(
            &matrix[i * row_bytes..(i + 1) * row_bytes],
            &words,
        )))
    })
}
/// The sum of `values` with the entries of one packed row (four per byte, as
/// [`crate::transcript::ternary_entry`] reads them), each entry's term selected by masks: the
/// value where its first bit is set, minus the value where its second is.
fn signed_sum(row: &[u8], values: &[i64]) -> i64 {
    let mut sums = [0i64; 4];
    for (byte, v) in row.iter().zip(values.as_chunks::<4>().0) {
        for (k, (sum, x)) in sums.iter_mut().zip(v).enumerate() {
            let plus = -i64::from((byte >> (2 * k)) & 1);
            let minus = -i64::from((byte >> (2 * k + 1)) & 1);
            *sum += (x & plus) - (x & minus);
        }
    }
    sums.iter().sum()
}
/// [`projections`] row by row in checked `i128` arithmetic, for every width and magnitude.
fn projections_checked(
    matrices: &Projections,
    exact: bool,
    values: &[i128],
) -> Result<Vec<i128>, Error> {
    let compute_row = |i| {
        let row = matrices.row(exact, i);
        if row.len() != values.len() {
            return Err(Error::Dimension);
        }
        row.iter().zip(values).try_fold(0i128, |sum, (r, v)| {
            sum.checked_add(i128::from(*r).checked_mul(*v).ok_or(Error::Overflow)?)
                .ok_or(Error::Overflow)
        })
    };
    // In parallel with the `parallel` feature, in order.
    crate::par::try_map(256, compute_row)
}
/// The response of one range block, or `None` to restart: the bimodal test reads one 256-bit
/// coin ([`reject::coin`]) from `random`, after the check against q/2 and before the bound and
/// the width of the response's fixed-width code.
fn range_response(
    scheme: &Abdlop,
    matrices: &Projections,
    exact: bool,
    values: &[i128],
    mask: &abdlop::Mask,
    sign: i128,
    random: &mut impl ByteStream,
) -> Result<Option<PolyVec>, Error> {
    if values.is_empty() {
        return Ok(Some(PolyVec::zero(scheme.ring().clone(), 0)));
    }
    let mut v = Secret::new(projections(matrices, exact, values)?);
    for x in v.iter_mut() {
        *x *= sign;
    }
    let y = abdlop::flatten(&mask.values)?;
    let z = Secret::new(
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
    let mut u = Secret::new([0u8; reject::COIN_BYTES]);
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
    // The fixed-width code: a coefficient beyond it restarts, a test of the response alone.
    let t = scheme.parameters.log_sigma[if exact { 2 } else { 3 }];
    let max = (1u128 << (response_width(t) - 1)) - 1;
    if z.iter().any(|x| x.unsigned_abs() > max) {
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
/// The evaluation equations of a toolbox proof, in order: the statement's own, then one per
/// row of each nonempty projection family, exact first.
///
/// The evaluation proof only needs, for each of its λ challenge rows, the combination of the
/// equations with that row's weights. A family's 256 equations $`e_j`$ are linear in their
/// matrix rows $`r_j`$, masks and responses, so their combination $`\sum_jw_je_j`$ is formed
/// directly ([`ProjectionRows::combined`]): one projection of the combined matrix row
/// $`\sum_jw_jr_j`$ per challenge row, instead of 256 projections that the combination then
/// adds up, with the same keys and values. The rows themselves are never formed.
struct ProjectionRows<'a> {
    scheme: &'a Abdlop,
    dim: usize,
    evals: &'a [QuadEq],
    matrices: &'a Projections,
    families: Vec<Family<'a>>,
}
/// A nonempty projection family of [`ProjectionRows`].
struct Family<'a> {
    exact: bool,
    affines: &'a [QuadEq],
    sign: &'a QuadEq,
    offset: usize,
    response: Secret<Vec<i128>>,
    prepared: Option<Prepared>,
}
impl<'a> ProjectionRows<'a> {
    /// The equations for the statement's evaluation equations `evals` and the families
    /// `(exact, affines, sign, mask offset, response)`, checked before any row is produced: a
    /// nonempty family needs $`256/d`$ responses over the ring, an empty one none.
    fn new(
        scheme: &'a Abdlop,
        evals: &'a [QuadEq],
        families: [(bool, &'a [QuadEq], &'a QuadEq, usize, &PolyVec); 2],
        matrices: &'a Projections,
    ) -> Result<Self, Error> {
        let dim = 2 * (scheme.bounded_len() + scheme.message_len());
        let ring = scheme.ring();
        let mut nonempty = Vec::new();
        for (exact, affines, sign, offset, z) in families {
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
            nonempty.push(Family {
                exact,
                affines,
                sign,
                offset,
                response,
                prepared: Prepared::new(ring, dim, affines)?,
            });
        }
        Ok(Self {
            scheme,
            dim,
            evals,
            matrices,
            families: nonempty,
        })
    }
    /// The equation of row `i` of `family`: the reference that [`ProjectionRows::combined`]
    /// is tested against.
    #[cfg(test)]
    fn row(&self, family: &Family<'_>, i: usize) -> Result<QuadEq, Error> {
        let (scheme, dim) = (self.scheme, self.dim);
        let ring = scheme.ring();
        let r = self.matrices.row(family.exact, i);
        if r.len() != family.affines.len() * ring.degree() {
            return Err(Error::Dimension);
        }
        let rows = r
            .chunks_exact(ring.degree())
            .map(|coeffs| {
                Ok(Poly::new(
                    ring.clone(),
                    coeffs.iter().map(|x| i128::from(*x)).collect(),
                )?
                .auto())
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let projection = match &family.prepared {
            Some(prepared) => prepared.projection(ring, dim, &rows)?,
            None => projection(ring, dim, family.affines, &rows)?,
        };
        let mask = variable(scheme, dim, family.offset + i / ring.degree())?
            .scale(&Poly::constant(ring.clone(), 1).rotate(-((i % ring.degree()) as i64)))?;
        // In place: the keys and values of `sign.product_affine(&projection)?.add(&mask)?`.
        let mut equation = QuadEq::zero(ring.clone(), dim)?;
        equation.add_product_affine(family.sign, &projection)?;
        equation.add_assign(&mask)?;
        equation
            .r0
            .sub_assign(&Poly::constant(ring.clone(), family.response[i]))?;
        Ok(equation)
    }
}
impl ProjectionRows<'_> {
    /// For each weight vector `weights[k]` (one weight per row), the combination
    /// $`\sum_jw_{kj}e_j`$ of the 256 equations $`e_j`$ of `family`. Row $`j`$ is
    /// $`\mathrm{sign}\cdot\sum_a\sigma(r_{j,a})f_a`$ plus its mask
    /// $`y_{\lfloor j/d\rfloor}X^{-(j\bmod d)}`$ minus its response $`z_j`$, linear in each, so
    /// the combination takes the combined matrix row $`\sum_jw_{kj}r_j`$ through one
    /// projection, each mask polynomial with the factor $`\sum_ew_{k,vd+e}X^{-e}`$, and the
    /// response $`\sum_jw_{kj}z_j`$. Each part adds the keys that the rows' parts add, so the
    /// keys and values are those of the combination of the rows.
    fn combined(&self, family: &Family<'_>, weights: &[&[U256]]) -> Result<Vec<QuadEq>, Error> {
        let (scheme, dim) = (self.scheme, self.dim);
        let ring = scheme.ring();
        let d = ring.degree();
        let cols = family.affines.len() * d;
        if weights.iter().any(|w| w.len() != 256) {
            return Err(Error::Dimension);
        }
        let sums = weighted_rows(ring, self.matrices, family.exact, cols, weights)?;
        // In parallel (`parallel`) over the challenge rows, in order.
        crate::par::try_map(weights.len(), |k| {
            let rows: Vec<Poly> = sums[k]
                .chunks_exact(d)
                .map(|c| Poly::from_canonical(ring.clone(), c.to_vec()).auto())
                .collect();
            let projection = match &family.prepared {
                Some(prepared) => prepared.projection(ring, dim, &rows)?,
                None => projection(ring, dim, family.affines, &rows)?,
            };
            let mut equation = QuadEq::zero(ring.clone(), dim)?;
            equation.add_product_affine(family.sign, &projection)?;
            for v in 0..256 / d {
                let mut factor = Poly::zero(ring.clone());
                for e in 0..d {
                    let w = Poly::constant_u256(ring.clone(), &weights[k][v * d + e]);
                    factor.add_assign(&w.rotate(-(e as i64)))?;
                }
                equation.add_assign(&variable(scheme, dim, family.offset + v)?.scale(&factor)?)?;
            }
            let response = weights[k]
                .iter()
                .zip(family.response.iter())
                .fold(U256::ZERO, |sum, (w, z)| {
                    ring.add(&sum, &ring.mul(w, &ring.reduce_i128(*z)))
                });
            equation
                .r0
                .sub_assign(&Poly::constant_u256(ring.clone(), &response))?;
            Ok(equation)
        })
    }
}
/// For each weight vector `weights[k]` (256 canonical weights), the canonical
/// $`\sum_jw_{kj}r_{j,c}`$ for every column $`c`$ of the 256 matrix rows $`r_j`$ of a family,
/// whose entries are in $`\{-1,0,1\}`$. From [`TABLE_MIN_COLS`] columns on, by the table of
/// sign patterns of [`weighted_rows_by_patterns`] below $`2^{55}`$ and by the byte table of
/// [`weighted_rows_by_table`] below $`2^{119}`$. Otherwise below $`2^{120}`$ each row is
/// decoded once and a column's positive and negative weights are summed in `u128`, where 256 of
/// them stay below $`2^{128}`$, and reduced once. Above, each entry is a modular addition.
fn weighted_rows(
    ring: &Ring,
    matrices: &Projections,
    exact: bool,
    cols: usize,
    weights: &[&[U256]],
) -> Result<Vec<Vec<U256>>, Error> {
    let lambda = weights.len();
    let table = cols >= TABLE_MIN_COLS && cols.is_multiple_of(4);
    match int::to_u128(&ring.modulus()).filter(|q| *q < 1 << 120) {
        Some(q) if q < 1 << 55 && table => {
            weighted_rows_by_patterns(matrices, exact, cols, weights, q as u64)
        }
        Some(q) if q < 1 << 119 && table => {
            weighted_rows_by_table(matrices, exact, cols, weights, q)
        }
        Some(q) => {
            let w: Vec<Vec<u128>> = weights
                .iter()
                .map(|w| w.iter().map(int::low_u128).collect())
                .collect();
            let (mut pos, mut neg) = (vec![0u128; lambda * cols], vec![0u128; lambda * cols]);
            for j in 0..256 {
                let row = matrices.row(exact, j);
                if row.len() != cols {
                    return Err(Error::Dimension);
                }
                for (c, e) in row.iter().enumerate() {
                    let sums = match e {
                        1 => &mut pos,
                        -1 => &mut neg,
                        _ => continue,
                    };
                    for (k, w) in w.iter().enumerate() {
                        sums[k * cols + c] += w[j];
                    }
                }
            }
            Ok((0..lambda)
                .map(|k| {
                    (0..cols)
                        .map(|c| {
                            let (p, n) = (pos[k * cols + c] % q, neg[k * cols + c] % q);
                            U256::from_u128(if p >= n { p - n } else { p + q - n })
                        })
                        .collect()
                })
                .collect())
        }
        None => weighted_rows_general(ring, matrices, exact, cols, weights),
    }
}
/// The narrowest matrix that [`weighted_rows`] sums by table. The table's set-up writes
/// $`2^{14}`$ sums per challenge row whatever the width, about the work of summing 64 columns
/// directly: on an Apple M4, with 2 to 8 challenge rows, the direct sums were 1.1 to 1.6 times
/// faster at 64 columns, and the table 1.2 to 1.7 times faster at 128.
const TABLE_MIN_COLS: usize = 128;
/// [`weighted_rows`] below $`2^{119}`$, by the method of the LaBRADOR projection: the matrix
/// transposed so that byte $`g`$ of a column holds its entries of rows $`4g`$ to $`4g+3`$ (2
/// bits each, as [`crate::transcript::ternary_entry`] packs them), a table of the weighted sum
/// of the 4 entries of every byte value for each group $`g`$ and each challenge row $`k`$, and
/// each column's sum as 64 lookups. The table's set-up does not depend on the width, hence
/// [`TABLE_MIN_COLS`]. The sums are exact in `i128`, as 256 weights below $`q`$
/// stay below $`256q<2^{127}`$ in magnitude, and reduced once.
fn weighted_rows_by_table(
    matrices: &Projections,
    exact: bool,
    cols: usize,
    weights: &[&[U256]],
    q: u128,
) -> Result<Vec<Vec<U256>>, Error> {
    let lambda = weights.len();
    let columns = matrix_columns(matrices, exact, cols)?;
    let w: Vec<Vec<i128>> = weights
        .iter()
        .map(|w| w.iter().map(|x| int::low_u128(x) as i128).collect())
        .collect();
    // table[(256g + byte) lambda + k] = sum over e of entry e of `byte` times w_k[4g + e], where
    // the field 01 is 1, 10 is -1, and 00 and 11 are 0: the sum of entries 0 and 1 (the low 4
    // bits) from a table of 16, plus that of entries 2 and 3 (the high 4 bits) from another.
    let entry = |field: usize, w: i128| match field {
        1 => w,
        2 => -w,
        _ => 0,
    };
    let mut halves = vec![([0i128; 16], [0i128; 16]); lambda];
    let mut table = vec![0i128; 64 * 256 * lambda];
    for (g, block) in table.chunks_exact_mut(256 * lambda).enumerate() {
        for ((low, high), w) in halves.iter_mut().zip(&w) {
            for (n, (l, h)) in low.iter_mut().zip(high.iter_mut()).enumerate() {
                *l = entry(n & 3, w[4 * g]) + entry(n >> 2, w[4 * g + 1]);
                *h = entry(n & 3, w[4 * g + 2]) + entry(n >> 2, w[4 * g + 3]);
            }
        }
        for (byte, sums) in block.chunks_exact_mut(lambda).enumerate() {
            for (sum, (low, high)) in sums.iter_mut().zip(&halves) {
                *sum = low[byte & 15] + high[byte >> 4];
            }
        }
    }
    let mut out = vec![vec![U256::ZERO; cols]; lambda];
    let modulus = q as i128;
    let mut write = |c: usize, sums: &[i128]| {
        for (o, sum) in out.iter_mut().zip(sums) {
            o[c] = U256::from_u128(sum.rem_euclid(modulus) as u128);
        }
    };
    fn fixed<const L: usize>(table: &[i128], column: &[u8]) -> [i128; L] {
        let table = table.as_chunks::<L>().0;
        let mut acc = [0i128; L];
        for (g, b) in column.iter().enumerate() {
            for (a, t) in acc.iter_mut().zip(&table[g * 256 + usize::from(*b)]) {
                *a += *t;
            }
        }
        acc
    }
    for (c, column) in columns.as_chunks::<64>().0.iter().enumerate() {
        match lambda {
            1 => write(c, &fixed::<1>(&table, column)),
            2 => write(c, &fixed::<2>(&table, column)),
            3 => write(c, &fixed::<3>(&table, column)),
            4 => write(c, &fixed::<4>(&table, column)),
            5 => write(c, &fixed::<5>(&table, column)),
            6 => write(c, &fixed::<6>(&table, column)),
            7 => write(c, &fixed::<7>(&table, column)),
            8 => write(c, &fixed::<8>(&table, column)),
            _ => {
                let sums: Vec<i128> = (0..lambda)
                    .map(|k| {
                        column
                            .iter()
                            .enumerate()
                            .map(|(g, b)| table[(g * 256 + usize::from(*b)) * lambda + k])
                            .sum()
                    })
                    .collect();
                write(c, &sums);
            }
        }
    }
    Ok(out)
}
/// A block's matrix with each column as 64 bytes: byte $`g`$ holds the column's entries of rows
/// $`4g`$ to $`4g+3`$, 2 bits each as [`crate::transcript::ternary_entry`] packs them. Rows are
/// byte aligned (`cols` is a multiple of 4): the 4 rows of group $`g`$ give, byte by byte, the
/// entries of 4 columns, which become byte $`g`$ of each of those columns.
fn matrix_columns(matrices: &Projections, exact: bool, cols: usize) -> Result<Vec<u8>, Error> {
    let (matrix, matrix_cols) = matrices.block(exact);
    if matrix_cols != cols || matrix.len() != 64 * cols || !cols.is_multiple_of(4) {
        return Err(Error::Dimension);
    }
    let row_bytes = cols / 4;
    let mut columns = vec![0u8; 64 * cols];
    // The bytes b of rows 4g to 4g + 3 as the word x = r0 | r1 << 8 | r2 << 16 | r3 << 24 hold
    // field e of row i at bit 8i + 2e; byte e of the column bytes holds it at 8e + 2i. Swapping
    // the 2 x 2 blocks of fields (distance 12) and then the fields within them (distance 6)
    // transposes the 4 x 4 fields. Sixteen row bytes at a time (64 columns), so that the
    // columns written and the rows read stay in the first-level cache.
    let transpose = |x: u32| {
        let t = ((x >> 12) ^ x) & 0x0000_f0f0;
        let x = x ^ t ^ (t << 12);
        let t = ((x >> 6) ^ x) & 0x00cc_00cc;
        x ^ t ^ (t << 6)
    };
    for first in (0..row_bytes).step_by(16) {
        let block = first..(first + 16).min(row_bytes);
        for g in 0..64 {
            let row = |e: usize| &matrix[(4 * g + e) * row_bytes..(4 * g + e + 1) * row_bytes];
            let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
            for b in block.clone() {
                let y = transpose(u32::from_le_bytes([r0[b], r1[b], r2[b], r3[b]]));
                for (e, byte) in y.to_le_bytes().into_iter().enumerate() {
                    columns[(4 * b + e) * 64 + g] = byte;
                }
            }
        }
    }
    Ok(columns)
}
/// [`weighted_rows`] below $`2^{55}`$ by a table of sign patterns, as LaBRADOR sums its
/// projection: for each group $`g`$ of 4 rows, each of the 81 patterns of 4 entries
/// ([`PATTERN`]) and each challenge row $`k`$, the weighted sum of the entries in `i64`, and
/// each column's sums as 64 lookups through the pattern of each of its bytes. 256 weights below
/// $`q<2^{55}`$ stay below $`256q<2^{63}`$ in magnitude, so the sums are exact, and reduced once.
/// The table holds 5184 sums per challenge row, against the byte table's 16384.
fn weighted_rows_by_patterns(
    matrices: &Projections,
    exact: bool,
    cols: usize,
    weights: &[&[U256]],
    q: u64,
) -> Result<Vec<Vec<U256>>, Error> {
    let lambda = weights.len();
    let columns = matrix_columns(matrices, exact, cols)?;
    let w: Vec<Vec<i64>> = weights
        .iter()
        .map(|w| w.iter().map(|x| int::low_u128(x) as i64).collect())
        .collect();
    let mut table = vec![0i64; 64 * 81 * lambda];
    for (g, block) in table.chunks_exact_mut(81 * lambda).enumerate() {
        for (p, sums) in block.chunks_exact_mut(lambda).enumerate() {
            let signs = pattern_signs(p);
            for (sum, w) in sums.iter_mut().zip(&w) {
                *sum = (0..4).map(|e| i64::from(signs[e]) * w[4 * g + e]).sum();
            }
        }
    }
    let mut out = vec![vec![U256::ZERO; cols]; lambda];
    let modulus = q as i64;
    let mut write = |c: usize, sums: &[i64]| {
        for (o, sum) in out.iter_mut().zip(sums) {
            o[c] = U256::from_u64(sum.rem_euclid(modulus) as u64);
        }
    };
    fn fixed<const L: usize>(table: &[i64], column: &[u8]) -> [i64; L] {
        let table = table.as_chunks::<L>().0;
        let mut acc = [0i64; L];
        for (g, b) in column.iter().enumerate() {
            let entry = &table[g * 81 + usize::from(PATTERN[usize::from(*b)])];
            for (a, t) in acc.iter_mut().zip(entry) {
                *a += *t;
            }
        }
        acc
    }
    for (c, column) in columns.as_chunks::<64>().0.iter().enumerate() {
        match lambda {
            1 => write(c, &fixed::<1>(&table, column)),
            2 => write(c, &fixed::<2>(&table, column)),
            3 => write(c, &fixed::<3>(&table, column)),
            4 => write(c, &fixed::<4>(&table, column)),
            5 => write(c, &fixed::<5>(&table, column)),
            6 => write(c, &fixed::<6>(&table, column)),
            7 => write(c, &fixed::<7>(&table, column)),
            8 => write(c, &fixed::<8>(&table, column)),
            _ => {
                let sums: Vec<i64> = (0..lambda)
                    .map(|k| {
                        column
                            .iter()
                            .enumerate()
                            .map(|(g, b)| {
                                table[(g * 81 + usize::from(PATTERN[usize::from(*b)])) * lambda + k]
                            })
                            .sum()
                    })
                    .collect();
                write(c, &sums);
            }
        }
    }
    Ok(out)
}
/// [`weighted_rows`] by modular additions, for every modulus.
fn weighted_rows_general(
    ring: &Ring,
    matrices: &Projections,
    exact: bool,
    cols: usize,
    weights: &[&[U256]],
) -> Result<Vec<Vec<U256>>, Error> {
    let rows = (0..256).map(|j| matrices.row(exact, j)).collect::<Vec<_>>();
    if rows.iter().any(|row| row.len() != cols) {
        return Err(Error::Dimension);
    }
    Ok(weights
        .iter()
        .map(|w| {
            let mut sums = vec![U256::ZERO; cols];
            for (row, w) in rows.iter().zip(w.iter()) {
                for (sum, e) in sums.iter_mut().zip(row) {
                    match e {
                        1 => *sum = ring.add(sum, w),
                        -1 => *sum = ring.sub(sum, w),
                        _ => {}
                    }
                }
            }
            sums
        })
        .collect())
}
impl EvalRows for ProjectionRows<'_> {
    fn count(&self) -> usize {
        self.evals.len() + 256 * self.families.len()
    }
    fn add_to(
        &self,
        combination: &mut Combination<'_>,
        witness: Option<&PolyVec>,
    ) -> Result<(), Error> {
        self.evals.add_to(combination, witness)?;
        for family in &self.families {
            let weights = combination.next_weights(256)?;
            let rows = self.combined(family, &weights)?;
            // The prover checks the combined rows, which vanish at the constant coefficient
            // when the 256 rows do (and miss a row that does not with probability 1/q each).
            if let Some(witness) = witness {
                quad_eval::check_constants(&rows, witness, &mut quad::PairProducts::empty())?;
            }
            combination.add_combined(rows, 256)?;
        }
        Ok(())
    }
}
/// Every equation of [`ProjectionRows`] in one list.
#[cfg(test)]
fn projection_equations(
    scheme: &Abdlop,
    forms: &Forms,
    matrices: &Projections,
    ze: &PolyVec,
    zd: &PolyVec,
) -> Result<Vec<QuadEq>, Error> {
    let rows = ProjectionRows::new(
        scheme,
        &forms.evals,
        [
            (true, &forms.exact[..], &forms.sign_e, forms.mask_e, ze),
            (false, &forms.approx[..], &forms.sign_d, forms.mask_d, zd),
        ],
        matrices,
    )?;
    let mut out = rows.evals.to_vec();
    for family in &rows.families {
        for i in 0..256 {
            out.push(rows.row(family, i)?);
        }
    }
    Ok(out)
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
    QuadEq::combination(ring, dim, affines, rows)
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
/// The serialization of the range responses, the toolbox's second prover message (with the
/// garbage commitments after it): $`z^{(e)}`$ then $`z^{(d)}`$, each in its response width.
fn range_message(scheme: &Abdlop, ze: &PolyVec, zd: &PolyVec) -> Result<Vec<u8>, Error> {
    let mut w = BitWriter::new();
    fields::write_signed(&mut w, ze, response_width(scheme.parameters.log_sigma[2]))?;
    fields::write_signed(&mut w, zd, response_width(scheme.parameters.log_sigma[3]))?;
    w.into_bytes()
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
    seed: &Secret<[u8; 32]>,
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
    seed: &Secret<[u8; 32]>,
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
    seed: &Secret<[u8; 32]>,
    max_attempts: u32,
) -> Result<Proof, Error> {
    Ok(prove_rounds_mode(
        scheme,
        statement,
        s1,
        m,
        context,
        seed,
        max_attempts,
        Mode::Plain,
    )?
    .0)
}
/// [`prove_rounds`] in either mode: in the extractable mode it also returns the LinHC
/// commitment and the accepted attempt's LinHC opening.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn prove_rounds_mode(
    scheme: &Abdlop,
    statement: &Statement,
    s1: &PolyVec,
    m: &PolyVec,
    context: &[u8],
    seed: &Secret<[u8; 32]>,
    max_attempts: u32,
    mode: Mode<'_>,
) -> Result<(Proof, Option<(LinhcCommitment, linhc::LinhcOpening)>), Error> {
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
    let extraction = match mode {
        Mode::Plain => None,
        Mode::Extractable { extraction, .. } => Some(extraction),
    };
    let mut base = instance(scheme, extraction, context, |t| absorb_families(t, &forms))?;
    // "/v3" and "/v2" since the duplex-sponge transcript of version 0.2, which the binding
    // comes from. The earlier versions read these streams under other keys: masks that two
    // samplers read from one stream are correlated, and two proofs with correlated masks can
    // reveal the witness. The extractable mode reads more streams (its LinHC commitment's
    // randomness), so it has a label of its own.
    let label: &[u8] = match mode {
        Mode::Plain => b"tbox/proof/v3",
        Mode::Extractable { .. } => b"tbox/proof-sl/v2",
    };
    let key = toolbox_key(label, &base.binding(), s1, m, seed);
    let (_, opening) =
        scheme.commit_in_domain(bounded.clone(), m.clone(), &key, secret::TOOLBOX_COMMITMENT)?;
    // The extractable mode squeezes the LinHC key after the instance and commits, once, to the
    // final opening's witness (s1 || s21) with it.
    let extractable = match mode {
        Mode::Plain => None,
        Mode::Extractable {
            extraction,
            key: given,
        } => {
            let setup = extractable_setup(&extended, extraction, &mut base, given)?;
            let witness = linhc::message(
                setup.key.a.ring(),
                &bounded,
                &abdlop::part(&opening.s2, 0, scheme.a2.cols())?,
            )?;
            let (commitment, secret) = linhc::commit(
                &setup.key,
                &setup.params,
                &witness,
                &key,
                secret::LINHC_COMMITMENT,
            )?;
            Some((setup, commitment, secret))
        }
    };
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
        // Each attempt forks the transcript after the instance (and the LinHC key).
        let mut transcript = base.clone();
        transcript.absorb(&commitment_message(
            &extended,
            &commitment,
            extractable.as_ref().map(|(_, c, _)| c),
        )?);
        let matrices = Projections::squeeze(&mut transcript, &forms, ring.degree())?;
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
            &matrices,
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
            &matrices,
            false,
            &abdlop::flatten(&approx)?,
            &yd,
            bd,
            &mut random,
        )?
        else {
            continue;
        };
        let evals = ProjectionRows::new(
            &extended,
            &forms.evals,
            [
                (true, &forms.exact[..], &forms.sign_e, forms.mask_e, &ze),
                (false, &forms.approx[..], &forms.sign_d, forms.mask_d, &zd),
            ],
            &matrices,
        )?;
        transcript.absorb(&range_message(scheme, &ze, &zd)?);
        let mut proof_seed = Secret::new([0; 32]);
        random.fill(&mut *proof_seed)?;
        #[cfg(test)]
        crate::rand::stream_log::child(&key, &proof_seed);
        let ext = extractable
            .as_ref()
            .map(|(setup, _, secret)| linhc::ProverExt {
                key: &setup.key,
                params: &setup.params,
                checked: &setup.checked,
                secret,
            });
        // The equations move into the evaluation proof: every path below returns.
        let (evaluation, linhc_opening) = quad_eval::prove_in(
            &extended,
            &commitment,
            &full_opening,
            Cow::Owned(forms.eqs),
            &evals,
            transcript,
            &proof_seed,
            ext.as_ref(),
        )?;
        let proof = Proof {
            commitment,
            z_exact: ze,
            z_approx: zd,
            evaluation,
        };
        return Ok(match (extractable, linhc_opening) {
            (None, _) => (proof, None),
            (Some((_, linhc_commitment, _)), Some(linhc_opening)) => {
                (proof, Some((linhc_commitment, linhc_opening)))
            }
            (Some(_), None) => return Err(Error::Parameter("extractable mode")),
        });
    }
    Err(Error::RestartLimit)
}

/// Prove a full statement in the straight-line extractable mode (Katsumata's transform), under
/// the application `context`, with the extraction set `extraction` for the scheme's parameters.
///
/// The proof is a toolbox proof under the extractable mode's own transcript, with the LinHC
/// commitment to the final opening's witness and the LinHC opening beside it
/// ([`ExtractableProof`]). An extractor that holds the LinHC key's trapdoor recovers the
/// committed witness from one proof, without rewinding: it decrypts the responses for other
/// challenges. In the random-oracle model the key, squeezed from the transcript after the
/// instance, is the oracle's output on the statement and the context, so the extractor programs
/// it. `docs/security.md` gives the argument and its
/// status.
///
/// Seeds work as for [`prove_with_seed`], with keys under labels of this mode
/// (`tbox/proof-sl/v2`, `abdlop/opening-proof-sl/v2`).
pub fn prove_extractable_with_seed(
    scheme: &Abdlop,
    extraction: &ExtractionParams,
    statement: &Statement,
    s1: &PolyVec,
    m: &PolyVec,
    context: &[u8],
    mut seed: [u8; 32],
) -> Result<ExtractableProof, Error> {
    prove_extractable_seeded(
        scheme,
        extraction,
        statement,
        s1,
        m,
        context,
        &take_seed(&mut seed),
        None,
    )
}
/// [`prove_extractable_with_seed`] from a seed in a wiping buffer, with a given LinHC key in
/// place of the derived one when `key` is given (tests use a trapdoored key).
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_extractable_seeded(
    scheme: &Abdlop,
    extraction: &ExtractionParams,
    statement: &Statement,
    s1: &PolyVec,
    m: &PolyVec,
    context: &[u8],
    seed: &Secret<[u8; 32]>,
    key: Option<&LinhcKey>,
) -> Result<ExtractableProof, Error> {
    statement.check_witness(scheme, s1, m)?;
    let (proof, linhc) = prove_rounds_mode(
        scheme,
        statement,
        s1,
        m,
        context,
        seed,
        secret::MAX_ATTEMPTS,
        Mode::Extractable { extraction, key },
    )?;
    let (commitment, opening) = linhc.ok_or(Error::Parameter("extractable mode"))?;
    let [z1, z2, z3] = opening.z;
    Ok(ExtractableProof {
        proof,
        t1: commitment.t1,
        t2: commitment.t2,
        z1,
        z2,
        z3,
    })
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
    let mut transcript = instance(scheme, None, context, |t| absorb_families(t, &forms))?;
    transcript.absorb(&commitment_message(&extended, &proof.commitment, None)?);
    let matrices = Projections::squeeze(&mut transcript, &forms, scheme.ring().degree())?;
    let evals = ProjectionRows::new(
        &extended,
        &forms.evals,
        [
            (
                true,
                &forms.exact[..],
                &forms.sign_e,
                forms.mask_e,
                &proof.z_exact,
            ),
            (
                false,
                &forms.approx[..],
                &forms.sign_d,
                forms.mask_d,
                &proof.z_approx,
            ),
        ],
        &matrices,
    )?;
    transcript.absorb(&range_message(scheme, &proof.z_exact, &proof.z_approx)?);
    quad_eval::verify_in(
        &extended,
        &proof.commitment,
        Cow::Owned(forms.eqs),
        &evals,
        &proof.evaluation,
        transcript,
        None,
    )
}

/// Verify a proof of the straight-line extractable mode under the application `context` and
/// the extraction set `extraction`: the toolbox checks under the extractable mode's transcript,
/// the LinHC opening's block bounds, and the challenge derived after the LinHC mask commitment
/// that the verifier recomputes from the opening.
pub fn verify_extractable(
    scheme: &Abdlop,
    extraction: &ExtractionParams,
    statement: &Statement,
    proof: &ExtractableProof,
    context: &[u8],
) -> Result<(), Error> {
    verify_extractable_with_key(scheme, extraction, statement, proof, context, None)
}
/// [`verify_extractable`] with a given LinHC key in place of the derived one when `key` is
/// given.
pub(crate) fn verify_extractable_with_key(
    scheme: &Abdlop,
    extraction: &ExtractionParams,
    statement: &Statement,
    proof: &ExtractableProof,
    context: &[u8],
    key: Option<&LinhcKey>,
) -> Result<(), Error> {
    statement.check(scheme)?;
    let (e_slots, d_slots) = response_shape_and_bounds(scheme, &proof.proof)?;
    let extended = scheme.extend_messages(e_slots + d_slots + 1)?;
    let forms = forms(scheme, &extended, statement)?;
    let mut transcript = instance(scheme, Some(extraction), context, |t| {
        absorb_families(t, &forms)
    })?;
    let setup = extractable_setup(&extended, extraction, &mut transcript, key)?;
    let ring = setup.key.a.ring();
    let p = &setup.params;
    for (v, len) in [
        (&proof.t1, p.len),
        (&proof.t2, p.len),
        (&proof.z1, p.rank),
        (&proof.z2, p.len),
        (&proof.z3, p.len),
    ] {
        if v.len() != len {
            return Err(Error::Dimension);
        }
        if v.ring() != ring {
            return Err(Error::RingMismatch);
        }
    }
    let linhc_commitment = LinhcCommitment {
        t1: proof.t1.clone(),
        t2: proof.t2.clone(),
    };
    let linhc_opening = linhc::LinhcOpening {
        z: [proof.z1.clone(), proof.z2.clone(), proof.z3.clone()],
    };
    transcript.absorb(&commitment_message(
        &extended,
        &proof.proof.commitment,
        Some(&linhc_commitment),
    )?);
    let matrices = Projections::squeeze(&mut transcript, &forms, scheme.ring().degree())?;
    let evals = ProjectionRows::new(
        &extended,
        &forms.evals,
        [
            (
                true,
                &forms.exact[..],
                &forms.sign_e,
                forms.mask_e,
                &proof.proof.z_exact,
            ),
            (
                false,
                &forms.approx[..],
                &forms.sign_d,
                forms.mask_d,
                &proof.proof.z_approx,
            ),
        ],
        &matrices,
    )?;
    transcript.absorb(&range_message(
        scheme,
        &proof.proof.z_exact,
        &proof.proof.z_approx,
    )?);
    let ext = linhc::VerifierExt {
        key: &setup.key,
        params: &setup.params,
        checked: &setup.checked,
        commitment: &linhc_commitment,
        opening: &linhc_opening,
    };
    quad_eval::verify_in(
        &extended,
        &proof.proof.commitment,
        Cow::Owned(forms.eqs),
        &evals,
        &proof.proof.evaluation,
        transcript,
        Some(&ext),
    )
}
