//! Parameters of the straight-line extractable mode: Katsumata's transform (CRYPTO 2021,
//! ePrint 2021/927) over the toolbox, with the extractable linear homomorphic commitment
//! (LinHC) of its Sec. 3.4 beside the final opening proof.
use super::{CheckedParams, TboxParams};
use crate::{
    Error,
    linhc::{self, LinhcChecked, LinhcParams},
};

/// The oracle model that an extraction parameter set targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
#[non_exhaustive]
pub enum OracleModel {
    /// The classical random-oracle model.
    Rom,
}

/// The mask width of the LinHC in units of its shift bound: $`\sigma'\ge53T`$, which keeps
/// the rejection constant of the LinHC opening near 1.2 to 1.3.
pub(crate) const PHI: u32 = 53;

/// Parameters of the straight-line extractable mode for one [`TboxParams`].
///
/// The LinHC commits to the final opening's witness $`(s_1\|s_{21})`$, of
/// $`m'=m_1+|\text{l2\_rows}|+m_2-n`$ polynomials, over a ring of the proof degree and of a
/// prime modulus $`q'`$. Its other parameters are derived, not supplied: the plaintext modulus
/// $`p=4B+1`$ for the bound $`B`$ that the toolbox verifier puts on every coefficient of
/// $`(z_1\|z_{21})`$, the mask width $`\sigma'\ge53T`$, and the least prime $`q'`$ above
/// $`2(pN+B)`$, the decryption bound. The MLWE rank of the commitment randomness, its
/// estimate and the modulus that estimate is for are supplied evidence, as for the toolbox's own
/// MLWE: [`ExtractionParams::check`] requires the modulus to equal the derived one and the root
/// Hermite factor to be at most 1.0044, not that the estimate is true.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct ExtractionParams {
    /// The oracle model the set is derived for.
    pub model: OracleModel,
    /// MLWE rank $`n`$ of the LinHC commitment randomness, supplied by the parameter
    /// estimator. It must not exceed $`m'`$.
    pub rank: usize,
    /// The LinHC modulus $`q'`$ that the estimate is for. In JSON, a number below $`2^{64}`$
    /// and a decimal string otherwise.
    #[cfg_attr(feature = "serde", serde(with = "crate::json::u128_field"))]
    pub modulus: u128,
    /// Root Hermite factor of the estimate behind `rank`, at most 1.0044.
    pub mlwe_delta: f64,
    /// Estimator version and model provenance. Bound into the extractable mode's transcripts.
    pub estimator: String,
}

impl ExtractionParams {
    /// Check the set against the toolbox parameters `params`: the parameters themselves, a
    /// nonzero rank of at most $`m'`$, a root Hermite factor in $`[1,1.0044]`$, a nonempty
    /// estimator string, and a modulus equal to the derived $`q'`$ (with every LinHC
    /// inequality, see the module documentation of the LinHC).
    pub fn check(&self, params: &TboxParams) -> Result<(), Error> {
        self.linhc(params, &params.check()?).map(|_| ())
    }

    /// The LinHC modulus $`q'`$ that the rank `rank` gives for the toolbox parameters `params`:
    /// the modulus an estimate for that rank must be made for, and the value of `modulus` that
    /// [`ExtractionParams::check`] accepts.
    pub fn modulus_for(params: &TboxParams, rank: usize) -> Result<u128, Error> {
        Ok(derive(params, &params.check()?, rank)?.modulus)
    }

    /// The LinHC parameters this set gives for `params`, whose checked values are `checked`.
    pub(crate) fn linhc(
        &self,
        params: &TboxParams,
        checked: &CheckedParams,
    ) -> Result<(LinhcParams, LinhcChecked), Error> {
        if !self.mlwe_delta.is_finite() || !(1.0..=1.0044).contains(&self.mlwe_delta) {
            return Err(Error::Parameter("extraction MLWE estimate"));
        }
        if self.estimator.is_empty() {
            return Err(Error::Parameter("extraction estimator"));
        }
        let derived = derive(params, checked, self.rank)?;
        if derived.modulus != self.modulus {
            return Err(Error::Parameter("extraction modulus"));
        }
        let linhc_checked = derived.check()?;
        Ok((derived, linhc_checked))
    }

    /// The set's encoding in the extractable mode's transcripts.
    pub(crate) fn transcript_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(match self.model {
            OracleModel::Rom => 1u8,
        });
        out.extend_from_slice(&(self.rank as u64).to_le_bytes());
        out.extend_from_slice(&self.modulus.to_le_bytes());
        out.extend_from_slice(&self.mlwe_delta.to_bits().to_le_bytes());
        out.extend_from_slice(&(self.estimator.len() as u64).to_le_bytes());
        out.extend_from_slice(self.estimator.as_bytes());
        out
    }

    /// Strict JSON decoding followed by [`ExtractionParams::check`] against `params`.
    ///
    /// The fields are `model` (`"rom"`), `rank`, `modulus`, `mlwe_delta` and `estimator`.
    #[cfg(feature = "serde")]
    pub fn from_json(json: &str, params: &TboxParams) -> Result<Self, Error> {
        let extraction: Self = serde_json::from_str(json).map_err(|_| Error::Encoding)?;
        extraction.check(params)?;
        Ok(extraction)
    }
}

/// The derived LinHC parameters for `params` and `rank`: the message length
/// $`m'=m_1+|\text{l2\_rows}|+m_2-n`$, the message bound $`B`$ and the least valid moduli.
fn derive(params: &TboxParams, checked: &CheckedParams, rank: usize) -> Result<LinhcParams, Error> {
    let len = (params.m1 + params.l2_rows.len())
        .checked_add(params.m2 - params.n_msis)
        .ok_or(Error::Overflow)?;
    if rank == 0 || rank > len {
        return Err(Error::Parameter("extraction rank"));
    }
    // Every coefficient of a response the verifier accepts is at most its Euclidean norm: z1
    // within z1_bound_squared, and z21 within the joint bound of (z21, z22).
    let bound = linhc::ceil_sqrt(checked.z1_bound_squared).max(linhc::ceil_sqrt(checked.b_squared));
    linhc::derive(params.degree, rank, len, bound, checked.eta, PHI)
}
