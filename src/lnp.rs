//! Generic statement and witness objects for the LNP22 toolbox (LNP22 §5).
use crate::{
    Error,
    abdlop::Abdlop,
    math::{PolyMat, PolyVec, Ring, SparsePolyVec, U256},
    params::TboxParams,
    quad::QuadEq,
    rand::take_seed,
    tbox,
};
use std::sync::Arc;
use zeroize::Zeroizing;

/// Affine block $`E_s s_1+E_m m+v`$ with explicit dimensions, including absent matrices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffineBlock {
    /// Number of output polynomials.
    pub rows: usize,
    /// Coefficients on the original bounded witness (before slack insertion).
    pub s: Option<PolyMat>,
    /// Coefficients on the BDLOP witness.
    pub m: Option<PolyMat>,
    /// Public affine offset.
    pub offset: Option<PolyVec>,
}
impl AffineBlock {
    pub(crate) fn check(
        &self,
        ring: &Arc<Ring>,
        m1: usize,
        l: usize,
        rows: usize,
    ) -> Result<(), Error> {
        if self.rows != rows {
            return Err(Error::Dimension);
        }
        for (matrix, cols) in [(&self.s, m1), (&self.m, l)] {
            if let Some(matrix) = matrix {
                if matrix.rows() != rows || matrix.cols() != cols {
                    return Err(Error::Dimension);
                }
                if matrix.ring() != ring {
                    return Err(Error::RingMismatch);
                }
            }
        }
        if let Some(offset) = &self.offset {
            if offset.len() != rows {
                return Err(Error::Dimension);
            }
            if offset.ring() != ring {
                return Err(Error::RingMismatch);
            }
        }
        Ok(())
    }
    /// Evaluate on ring-checked witness vectors.
    pub fn evaluate(&self, s1: &PolyVec, m: &PolyVec) -> Result<PolyVec, Error> {
        self.check(s1.ring(), s1.len(), m.len(), self.rows)?;
        if m.ring() != s1.ring() {
            return Err(Error::RingMismatch);
        }
        let mut out = self
            .offset
            .clone()
            .unwrap_or_else(|| PolyVec::zero(s1.ring().clone(), self.rows));
        if let Some(matrix) = &self.s {
            out = crate::abdlop::add(&out, &matrix.mul(s1)?)?;
        }
        if let Some(matrix) = &self.m {
            out = crate::abdlop::add(&out, &matrix.mul(m)?)?;
        }
        Ok(out)
    }
    pub(crate) fn forms(&self, scheme: &Abdlop, dimension: usize) -> Result<Vec<QuadEq>, Error> {
        let p = &scheme.parameters;
        let ring = scheme.ring();
        self.check(ring, p.m1, p.l, self.rows)?;
        (0..self.rows)
            .map(|row| {
                let mut entries = Vec::new();
                if let Some(matrix) = &self.s {
                    for j in 0..p.m1 {
                        entries.push(((2 * j) as u16, matrix.get(row, j)?.clone()));
                    }
                }
                if let Some(matrix) = &self.m {
                    for j in 0..p.l {
                        entries.push((
                            (2 * (scheme.bounded_len() + j)) as u16,
                            matrix.get(row, j)?.clone(),
                        ));
                    }
                }
                let mut equation = QuadEq::zero(ring.clone(), dimension)?;
                equation.r1 = SparsePolyVec::new(ring.clone(), dimension, entries)?;
                if let Some(offset) = &self.offset {
                    equation.r0 = offset.entries()[row].clone();
                }
                Ok(equation)
            })
            .collect()
    }
}

/// One exact Euclidean-norm block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct L2Block {
    /// Public affine map.
    pub map: AffineBlock,
    /// Exact integer squared bound, required to match the parameter set.
    pub bound_squared: u64,
}
/// Binary constraint on every coefficient of an affine block.
pub type BinBlock = AffineBlock;
/// Approximate infinity-norm constraint on an affine block.
pub type ArpBlock = AffineBlock;

/// The full public toolbox statement, indexed from zero in the original witness space.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Statement {
    /// Whole-polynomial equations.
    pub quadratic: Vec<QuadEq>,
    /// Equations whose constant coefficient vanishes.
    pub evaluation: Vec<QuadEq>,
    /// Exact norm blocks, in parameter-set order.
    pub l2: Vec<L2Block>,
    /// Optional binary block.
    pub binary: Option<BinBlock>,
    /// Optional approximate block.
    pub arp: Option<ArpBlock>,
}
impl Statement {
    /// Check all rings, dimensions and block bounds before any transcript is generated.
    pub fn check(&self, scheme: &Abdlop) -> Result<(), Error> {
        let p = &scheme.parameters;
        let ring = scheme.ring();
        if self.quadratic.len() > 65535
            || self.evaluation.len() > 65535
            || self.l2.len() != p.l2_rows.len()
            || self.binary.is_some() != (p.n_bin > 0)
            || self.arp.is_some() != (p.n_prime > 0)
        {
            return Err(Error::Dimension);
        }
        for eq in self.quadratic.iter().chain(&self.evaluation) {
            eq.check(ring, 2 * (p.m1 + p.l))?;
        }
        for (i, block) in self.l2.iter().enumerate() {
            if block.bound_squared != p.l2_bounds_squared[i] {
                return Err(Error::Parameter("exact block bound"));
            }
            block.map.check(ring, p.m1, p.l, p.l2_rows[i])?;
        }
        if let Some(block) = &self.binary {
            block.check(ring, p.m1, p.l, p.n_bin)?;
        }
        if let Some(block) = &self.arp {
            block.check(ring, p.m1, p.l, p.n_prime)?;
        }
        Ok(())
    }
    pub(crate) fn check_witness(
        &self,
        scheme: &Abdlop,
        s1: &PolyVec,
        m: &PolyVec,
    ) -> Result<(), Error> {
        self.check(scheme)?;
        let p = &scheme.parameters;
        if s1.len() != p.m1 || m.len() != p.l {
            return Err(Error::Dimension);
        }
        if s1.ring() != scheme.ring() || m.ring() != scheme.ring() {
            return Err(Error::RingMismatch);
        }
        if s1.norm_squared()? > U256::from(p.alpha_squared) {
            return Err(Error::Witness);
        }
        let witness = crate::quad::interleave(s1, m)?;
        // Evaluated in parallel (`parallel`); the first failure in order is returned.
        let (quadratic, evaluation) = (&self.quadratic, &self.evaluation);
        crate::par::try_map(quadratic.len() + evaluation.len(), |i| {
            let holds = match i.checked_sub(quadratic.len()) {
                None => quadratic[i].evaluate(&witness)?.is_zero(),
                Some(j) => evaluation[j].evaluate(&witness)?.coefficients()[0].is_zero_vartime(),
            };
            match holds {
                true => Ok(()),
                false => Err(Error::Witness),
            }
        })?;
        for block in &self.l2 {
            if block.map.evaluate(s1, m)?.norm_squared()? > U256::from(block.bound_squared) {
                return Err(Error::Witness);
            }
        }
        if let Some(block) = &self.binary
            && block
                .evaluate(s1, m)?
                .entries()
                .iter()
                .any(|p| p.coefficients().iter().any(|x| *x > U256::ONE))
        {
            return Err(Error::Witness);
        }
        if let Some(block) = &self.arp
            && block
                .evaluate(s1, m)?
                .entries()
                .iter()
                .any(|poly| poly.norm_infinity() > U256::from_u64(p.linf_bound))
        {
            return Err(Error::Witness);
        }
        Ok(())
    }
}

/// Stateful prover owning its witness. All polynomial buffers zeroize on drop.
#[derive(Debug)]
pub struct Prover {
    scheme: Abdlop,
    statement: Statement,
    witness: Option<(PolyVec, PolyVec)>,
}
/// Public verifier with the same statement setters as the prover.
#[derive(Clone, Debug)]
pub struct Verifier {
    scheme: Abdlop,
    statement: Statement,
}

macro_rules! setters {
    () => {
        /// Replace the entire public statement, validating all block dimensions.
        pub fn set_statement(&mut self, statement: Statement) -> Result<(), Error> {
            statement.check(&self.scheme)?;
            self.statement = statement;
            Ok(())
        }
        /// Replace the whole-polynomial equations; full validation also runs at prove and
        /// verify time.
        pub fn set_quadratic_eqs(&mut self, eqs: &[QuadEq]) -> Result<(), Error> {
            for e in eqs {
                e.check(
                    self.scheme.ring(),
                    2 * (self.scheme.parameters.m1 + self.scheme.parameters.l),
                )?;
            }
            self.statement.quadratic = eqs.to_vec();
            Ok(())
        }
        /// Replace constant-coefficient equations.
        pub fn set_eval_eqs(&mut self, eqs: &[QuadEq]) -> Result<(), Error> {
            for e in eqs {
                e.check(
                    self.scheme.ring(),
                    2 * (self.scheme.parameters.m1 + self.scheme.parameters.l),
                )?;
            }
            self.statement.evaluation = eqs.to_vec();
            Ok(())
        }
        /// Set the exact norm blocks after checking their dimensions and bounds.
        pub fn set_l2_blocks(&mut self, blocks: &[L2Block]) -> Result<(), Error> {
            let p = &self.scheme.parameters;
            if blocks.len() != p.l2_rows.len() {
                return Err(Error::Dimension);
            }
            for (i, block) in blocks.iter().enumerate() {
                block.map.check(self.ring(), p.m1, p.l, p.l2_rows[i])?;
                if block.bound_squared != p.l2_bounds_squared[i] {
                    return Err(Error::Parameter("exact block bound"));
                }
            }
            self.statement.l2 = blocks.to_vec();
            Ok(())
        }
        /// Set the optional binary block.
        pub fn set_binary_block(&mut self, block: Option<BinBlock>) -> Result<(), Error> {
            let p = &self.scheme.parameters;
            if block.is_some() != (p.n_bin > 0) {
                return Err(Error::Dimension);
            }
            if let Some(block) = &block {
                block.check(self.ring(), p.m1, p.l, p.n_bin)?;
            }
            self.statement.binary = block;
            Ok(())
        }
        /// Set the optional approximate range block.
        pub fn set_arp_block(&mut self, block: Option<ArpBlock>) -> Result<(), Error> {
            let p = &self.scheme.parameters;
            if block.is_some() != (p.n_prime > 0) {
                return Err(Error::Dimension);
            }
            if let Some(block) = &block {
                block.check(self.ring(), p.m1, p.l, p.n_prime)?;
            }
            self.statement.arp = block;
            Ok(())
        }
        /// Ring used by the public statement.
        pub fn ring(&self) -> &Arc<Ring> {
            self.scheme.ring()
        }
    };
}
impl Prover {
    /// Prove and encode with caller-supplied cryptographic randomness, under an application
    /// `context` that the verifier must supply too. Each call draws a fresh seed.
    pub fn prove_bytes(
        &self,
        context: &[u8],
        rng: &mut impl rand_core::CryptoRng,
    ) -> Result<Vec<u8>, Error> {
        crate::codec::proof::encode(&self.scheme, &self.prove(context, rng)?)
    }
    /// Prove and encode using the fixed-format toolbox codec.
    ///
    /// The seed must be secret and uniformly random, and it may be reused. The commitment
    /// inside the proof and every mask are read under keys derived from the seed, the
    /// parameters, the public seed, the statement, the context and the witness: identical calls
    /// return identical proofs, and calls that differ in any input read independent randomness
    /// (assuming SHAKE128 and AES-256 behave as pseudorandom functions).
    ///
    /// With a reused seed, zero-knowledge holds only relative to the equality pattern of the
    /// inputs: identical inputs give identical proofs. Uses that need multi-theorem
    /// zero-knowledge, and settings where faults can be injected (a fault that changes a
    /// challenge but not the key can reveal the witness), should use `prove_bytes`, which draws
    /// a fresh seed per call.
    pub fn prove_bytes_with_seed(
        &self,
        context: &[u8],
        mut seed: [u8; 32],
    ) -> Result<Vec<u8>, Error> {
        crate::codec::proof::encode(
            &self.scheme,
            &self.prove_seeded(context, &take_seed(&mut seed))?,
        )
    }
    /// Construct with validated parameters and an empty statement.
    pub fn new(pp_seed: [u8; 32], params: Arc<TboxParams>) -> Result<Self, Error> {
        Ok(Self {
            scheme: Abdlop::new(pp_seed, (*params).clone())?,
            statement: Statement::default(),
            witness: None,
        })
    }
    setters!();
    /// Replace the secret witness, checking rings and original dimensions.
    pub fn set_witness(&mut self, s1: &PolyVec, m: &PolyVec) -> Result<(), Error> {
        if s1.len() != self.scheme.parameters.m1 || m.len() != self.scheme.parameters.l {
            return Err(Error::Dimension);
        }
        if s1.ring() != self.ring() || m.ring() != self.ring() {
            return Err(Error::RingMismatch);
        }
        self.witness = Some((s1.clone(), m.clone()));
        Ok(())
    }
    /// Prove using explicit deterministic randomness, under an application `context`.
    ///
    /// The seed must be secret and uniformly random, and it may be reused. The commitment
    /// inside the proof and every mask are read under keys derived from the seed, the
    /// parameters, the public seed, the statement, the context and the witness: identical calls
    /// return identical proofs, and calls that differ in any input read independent randomness
    /// (assuming SHAKE128 and AES-256 behave as pseudorandom functions).
    ///
    /// With a reused seed, zero-knowledge holds only relative to the equality pattern of the
    /// inputs: identical inputs give identical proofs. Uses that need multi-theorem
    /// zero-knowledge, and settings where faults can be injected (a fault that changes a
    /// challenge but not the key can reveal the witness), should use `prove`, which draws a
    /// fresh seed per call.
    pub fn prove_with_seed(
        &self,
        context: &[u8],
        mut seed: [u8; 32],
    ) -> Result<tbox::Proof, Error> {
        self.prove_seeded(context, &take_seed(&mut seed))
    }
    /// Prove with caller-supplied cryptographic randomness, under an application `context`.
    /// Each call draws a fresh seed.
    pub fn prove(
        &self,
        context: &[u8],
        rng: &mut impl rand_core::CryptoRng,
    ) -> Result<tbox::Proof, Error> {
        let mut seed = Zeroizing::new([0; 32]);
        rand_core::Rng::fill_bytes(rng, &mut *seed);
        self.prove_seeded(context, &seed)
    }
    fn prove_seeded(
        &self,
        context: &[u8],
        seed: &Zeroizing<[u8; 32]>,
    ) -> Result<tbox::Proof, Error> {
        let (s1, m) = self.witness.as_ref().ok_or(Error::Witness)?;
        tbox::prove_seeded(&self.scheme, &self.statement, s1, m, context, seed)
    }
}
impl Verifier {
    /// Strictly decode and verify a wire-format proof made under `context`.
    pub fn verify_bytes(&self, bytes: &[u8], context: &[u8]) -> Result<(), Error> {
        self.verify(&crate::codec::proof::decode(&self.scheme, bytes)?, context)
    }
    /// Construct with validated parameters and an empty statement.
    pub fn new(pp_seed: [u8; 32], params: Arc<TboxParams>) -> Result<Self, Error> {
        Ok(Self {
            scheme: Abdlop::new(pp_seed, (*params).clone())?,
            statement: Statement::default(),
        })
    }
    setters!();
    /// Verify against this verifier's own public statement and the application `context`.
    pub fn verify(&self, proof: &tbox::Proof, context: &[u8]) -> Result<(), Error> {
        tbox::verify(&self.scheme, &self.statement, proof, context)
    }
}
