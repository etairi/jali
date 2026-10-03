//! Named witness blocks and quadratic statements over other rings.
//! The compiler lowers degrees, lifts linear relations implicitly, and commits to the carries of
//! quadratic relations and of constant-coefficient clauses, packing the latter $`d`$ to a
//! proof-ring polynomial. Each constraint has its own source modulus, that of the ring its form
//! is built over. Blocks bounded in $`\ell_\infty`$ share the one approximate range proof with
//! the carries (LNP22 §5.1), or are bounded exactly through bits ([`Norm::LinfExact`]). A block
//! may hold elements of a subring of smaller degree ([`Statement::var_subring`]), of which only
//! the nonzero components are committed. Each block sits in the part of the commitment its
//! declaration names ([`Placement`]). Parameters must be generated for the resulting shape and
//! independently estimated.
use crate::{
    Error,
    abdlop::Abdlop,
    lnp::{self, AffineBlock, L2Block},
    math::{I256, Poly, PolyMat, PolyVec, Ring, SparsePolyMat, SparsePolyVec, U256, int, iso},
    params::TboxParams,
    quad::{Affine, QuadEq},
    rand::take_seed,
    tbox,
};
use crypto_bigint::{CheckedAdd, I512, NonZero, U1024};
use std::{
    collections::{BTreeMap, btree_map::Entry},
    sync::Arc,
};

#[cfg(test)]
mod digits_tests;
#[cfg(test)]
mod selector_tests;

/// Integer witness bounds, applied to the whole named block.
///
/// In JSON: `{"l2_squared": B}`, `"binary"`, `{"linf": β}`, `"unbounded"` or
/// `{"linf_exact": β}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Norm {
    /// Exact squared Euclidean bound.
    L2Squared(u64),
    /// Every coefficient is either zero or one.
    Binary,
    /// Every coefficient has absolute value at most $`\beta\ge1`$, proven approximately: the
    /// coefficients join the one approximate range proof (LNP22 §5.1), with the carries and
    /// quotients, under its common bound `linf_bound`. The honest prover needs $`\beta`$, but a
    /// proof guarantees only the extraction bound $`E=2\cdot`$`z4_bound` of that proof
    /// ([`Compiled::extraction_bound`]), which is far larger. Every soundness check of the
    /// compiler uses $`E`$ ([witness relation](Statement)).
    Linf(u64),
    /// No norm claim. Only constraints at the proof modulus $`q`$, which are not lifted, can use
    /// this block, and only if $`q`$ divides the statement modulus $`p`$ or $`p\ge q`$ (the
    /// range condition of the [witness relation](Statement)).
    Unbounded,
    /// Every coefficient has absolute value at most $`\beta`$, $`1\le\beta<2^{63}`$, proven
    /// exactly. The compiler writes each coefficient as $`v=\sum_{b<n}c_b\,x_b-\beta`$ with
    /// $`n=\mathrm{bitlen}(2\beta)`$ bits $`x_b`$ in the binary block and the weights
    /// $`c_b=\lfloor(2\beta+2^b)/2^{b+1}\rfloor`$, whose subset sums are exactly
    /// $`[0,2\beta]`$, and substitutes that form into every constraint; $`v`$ itself is not
    /// committed. A proof bounds an extracted coefficient by $`\beta`$ exactly
    /// ([`Extraction::LinfExact`]), and every soundness check uses $`\beta`$. Each committed
    /// proof-ring polynomial of the block becomes $`n`$ binary ones: 2 for $`\beta=1`$, 11 for
    /// $`\beta=512`$. (Last, so that the serde variant indices of the others stay.)
    LinfExact(u64),
}

/// The part of the ABDLOP commitment that holds a block (LNP22 §3.1, Fig. 4).
///
/// The Ajtai part $`s_1`$ holds short vectors: `alpha_squared` bounds its Euclidean norm, and
/// the proof sends its masked opening with a Gaussian whose width $`\sigma_1`$ grows with that
/// bound. The BDLOP part $`m`$ holds messages, which the commitment carries at full size modulo
/// $`q`$. Every norm proof is an affine map of both parts (LNP22 §5.2), so a bounded block in
/// the BDLOP part keeps its exact, binary or approximate proof: it leaves `m1` and
/// `alpha_squared` and joins `l`. That shrinks $`\sigma_1`$, and can shrink the proof, when its
/// bound is large next to the rest of the Ajtai part. In JSON, `"ajtai"` or `"bdlop"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Placement {
    /// The Ajtai part, where [`Statement::var`] puts every bounded block.
    Ajtai,
    /// The BDLOP part, the only place for an unbounded block.
    Bdlop,
}
impl Placement {
    /// The placement that [`Statement::var`] gives: the Ajtai part for a bounded block and the
    /// BDLOP part for an unbounded one.
    pub fn default_for(norm: Norm) -> Self {
        if norm == Norm::Unbounded {
            Placement::Bdlop
        } else {
            Placement::Ajtai
        }
    }
}

/// One named block as a parameter search needs it ([`Statement::blocks`]).
///
/// With the requirements of the same statement, this lets a tool move a bounded block to the
/// other part without the statement: its `rows` move between `m1` and `l`, and its share of
/// `alpha_squared` counts only in the Ajtai part. That share is the bound $`B`$ of an exact
/// block, $`\mathrm{rows}\cdot d`$ for a binary block and for the bits of a
/// [`Norm::LinfExact`] block, and $`\mathrm{rows}\cdot d\beta^2`$ for an $`\ell_\infty`$ block,
/// $`d`$ the proof degree. No other field of the requirements depends on the placement.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct BlockRequirement {
    /// The name given to [`Statement::var`].
    pub name: String,
    /// The proof-ring polynomials the block commits: its count times $`d_v/d`$ for a block of
    /// degree $`d_v`$ ([`Statement::var_subring`]; $`d_v=d'`$ for [`Statement::var`]), times
    /// the number of bits for a [`Norm::LinfExact`] block.
    pub rows: usize,
    /// The declared norm.
    pub norm: Norm,
    /// The declared part.
    pub placement: Placement,
}

/// Dimensions and coefficient bounds exported before offline parameter selection.
///
/// In JSON, the integer bounds are numbers below $`2^{64}`$ and decimal strings otherwise, and
/// an absent `approx_alpha_squared` and an empty `lifted_moduli` are left out, so that a
/// statement whose constraints all use the statement modulus, and whose range rows all hold up
/// to `linf_bound` in every coefficient, exports only the fields that every statement has. Serde
/// formats that are not human-readable, such as postcard, take the format's own integers, every
/// 256-bit value as 32 little-endian bytes, and always write every field, since they read
/// fields by position.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct Requirements {
    /// Witness polynomials of the Ajtai part in the proof ring, before slack insertion: the
    /// rows of the bounded blocks placed there ([`Placement`], [`BlockRequirement::rows`]).
    pub m1: usize,
    /// BDLOP message polynomials: bounded blocks placed there, unbounded blocks and committed
    /// carries: $`d'/d`$ for each lifted quadratic `eq_mod_p`, and $`\lceil m/d\rceil`$ for the
    /// $`m`$ lifted constant-coefficient clauses together, which share packed carry
    /// polynomials ([`Statement::const_coeff_zero`]).
    pub l: usize,
    /// Sufficient squared bound on the Ajtai part.
    #[cfg_attr(feature = "serde", serde(with = "crate::json::u128_field"))]
    pub alpha_squared: u128,
    /// Binary selector rows: those of the binary blocks and the bits of the
    /// [`Norm::LinfExact`] blocks.
    pub n_bin: usize,
    /// Exact-norm selector rows per named block.
    pub l2_rows: Vec<usize>,
    /// Exact squared bounds.
    #[cfg_attr(feature = "serde", serde(with = "crate::json::u64_list"))]
    pub l2_bounds_squared: Vec<u64>,
    /// Approximate-range rows: $`d'/d`$ for the implicit quotients or committed carries of each
    /// lifted `eq_mod_p`, $`\lceil m/d\rceil`$ for the packed carries of the $`m`$ lifted
    /// constant-coefficient clauses, and one per committed proof-ring polynomial of each
    /// [`Norm::Linf`] variable.
    pub n_prime: usize,
    /// Sufficient integer bound on quotients and carries, and the $`\beta`$ of every
    /// [`Norm::Linf`] variable.
    #[cfg_attr(feature = "serde", serde(with = "crate::json::u128_field"))]
    pub linf_bound: u128,
    /// Conservative bound on the integer evaluation of any lifted expression.
    #[cfg_attr(feature = "serde", serde(with = "crate::json::u256"))]
    pub max_integer_coefficient: U256,
    /// A per-slot bound on the squared Euclidean norm of the approximate-range vector of every
    /// honest witness, present exactly when it is below $`n'd\beta_\infty^2`$
    /// ($`\beta_\infty`$ = `linf_bound`) and $`2^{128}`$:
    /// $`\sum_js_j\lceil F_j/p_j\rceil^2+\sum_v\mathrm{count}_v\,d_v\beta_v^2`$, over the lifted
    /// constraints $`j`$, with $`s_j=d'`$ slots for an `eq_mod_p` and $`s_j=1`$ for a
    /// constant-coefficient clause, and over the [`Norm::Linf`] variables $`v`$ of degree
    /// $`d_v`$ ($`d'`$ unless declared with [`Statement::var_subring`]); unused slots of packed
    /// carry polynomials hold 0. A parameter tool may size $`\sigma_4`$ from it.
    /// Nothing else reads it: the prover's rejection constant still uses
    /// $`n'd\beta_\infty^2`$, which the witness check enforces, so rejection sampling stays
    /// exact for every $`\sigma_4`$ that `TboxParams::check` accepts, and soundness depends on
    /// $`\sigma_4`$ only through the extraction bound $`E`$. In JSON, left out when absent, and
    /// otherwise a number below $`2^{64}`$ or a decimal string.
    #[cfg_attr(feature = "serde", serde(with = "crate::json::u128_option", default))]
    pub approx_alpha_squared: Option<u128>,
    /// The moduli of the lifted constraints in ascending order, each with the largest bound
    /// $`F_j`$ among the constraints at it. `compile` requires
    /// $`q>2(F_j+p_j\beta_\infty\psi)`$ for each, with $`\psi`$ as in [`Statement::compile`].
    /// Empty when every lifted constraint uses the statement modulus: the lifting is then
    /// described by that modulus and `max_integer_coefficient`.
    /// These bounds are honest ones: for a constraint with an $`\ell_\infty`$ variable, `compile`
    /// uses the bound of `linf` instead.
    #[cfg_attr(feature = "serde", serde(default))]
    pub lifted_moduli: Vec<LiftedModulus>,
    /// The conditions on the extraction bound of the approximate range proof, present exactly
    /// when the statement declares a [`Norm::Linf`] variable. JSON leaves it out when absent.
    #[cfg_attr(feature = "serde", serde(default))]
    pub linf: Option<LinfRequirements>,
}

/// What the soundness of [`Norm::Linf`] variables requires of the extraction bound
/// $`E=`$`approx_extraction_bound` $`=2\cdot`$`z4_bound`, which the parameters fix through
/// `log_sigma[3]`. `compile` checks both conditions with the actual $`E`$. Variables bounded
/// exactly ([`Norm::LinfExact`]) need neither: they keep their $`\beta`$.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinfRequirements {
    /// Every lifted constraint that uses a [`Norm::Linf`] variable, in declaration order.
    /// `compile` requires $`q>2(F_j(E)+p_j\beta_\infty\psi)`$ for each, with the bound
    /// $`F_j(E)`$ of [`LinfLifting`] in place of the honest one.
    pub lifting: Vec<LinfLifting>,
    /// $`\lfloor(p-1)/2\rfloor`$ for the statement modulus $`p`$, when a [`Norm::Linf`]
    /// variable has a nonzero coefficient in a constraint whose modulus does not divide $`p`$:
    /// the range condition of the [witness relation](Statement) then needs
    /// $`E\le\lfloor(p-1)/2\rfloor`$. Absent otherwise. In JSON, `null` or the number rule.
    #[cfg_attr(feature = "serde", serde(with = "crate::json::u256_option", default))]
    pub extraction_limit: Option<U256>,
}

/// A lifted constraint with [`Norm::Linf`] variables. On every witness whose [`Norm::Linf`]
/// variables are bounded by $`E`$ and whose other variables meet their declared norms, its
/// integer value is at most
/// $`F_j(E)=`$`exact`$`+`$`linear`$`\cdot E+`$`quadratic`$`\cdot E^2`$ in absolute value.
///
/// In JSON, every field is a number below $`2^{64}`$ and a decimal string otherwise.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinfLifting {
    /// The constraint modulus $`p_j`$.
    #[cfg_attr(feature = "serde", serde(with = "crate::json::u256"))]
    pub modulus: U256,
    /// The terms without [`Norm::Linf`] variables, as in `max_integer_coefficient`.
    #[cfg_attr(feature = "serde", serde(with = "crate::json::u256"))]
    pub exact: U256,
    /// The weight of $`E`$: $`\|r\|_1`$ for a linear term $`r\,y`$,
    /// $`\|r\|_1\lceil\sqrt{n_yB}\rceil`$ for a quadratic term $`r\,y\,s`$ with $`s`$ in a
    /// block of squared bound $`B`$ (for a binary block, its number of coefficients), and
    /// $`\|r\|_1\lceil\sqrt{n_yn_{y'}}\rceil\beta'`$ for $`r\,y\,y'`$ with $`y'`$ bounded by
    /// $`\beta'`$ exactly ([`Norm::LinfExact`]), where $`n_y`$ is the degree of $`y`$: the
    /// statement degree, or the subring degree of [`Statement::var_subring`].
    #[cfg_attr(feature = "serde", serde(with = "crate::json::u256"))]
    pub linear: U256,
    /// The weight of $`E^2`$: $`\|r\|_1\lceil\sqrt{n_yn_{y'}}\rceil`$ for a quadratic term
    /// $`r\,y\,y'`$ of two [`Norm::Linf`] variables ($`d'\|r\|_1`$ when both have the statement
    /// degree $`d'`$).
    #[cfg_attr(feature = "serde", serde(with = "crate::json::u256"))]
    pub quadratic: U256,
}
impl LinfLifting {
    /// $`F_j(E)`$ in 1024 bits.
    fn at(&self, e: u128) -> Result<Bound, Error> {
        let e = Bound::from_u128(e);
        let wide = |x: &U256| x.resize::<{ Bound::LIMBS }>();
        let linear = mul_bound(wide(&self.linear), e)?;
        let quadratic = mul_bound(mul_bound(wide(&self.quadratic), e)?, e)?;
        add_bound(add_bound(wide(&self.exact), linear)?, quadratic)
    }
}

/// A modulus of lifted constraints, with the largest bound on their integer evaluation.
///
/// In JSON, both are numbers below $`2^{64}`$ and decimal strings otherwise.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LiftedModulus {
    /// The constraint modulus $`p_j`$.
    #[cfg_attr(feature = "serde", serde(with = "crate::json::u256"))]
    pub modulus: U256,
    /// The largest $`F_j`$ among the lifted constraints at this modulus.
    #[cfg_attr(feature = "serde", serde(with = "crate::json::u256"))]
    pub max_integer_coefficient: U256,
}

/// Written by hand so that JSON can leave an absent `approx_alpha_squared`, an empty
/// `lifted_moduli` and an absent `linf` out while formats that read fields by position, such
/// as postcard, always get them.
#[cfg(feature = "serde")]
impl serde::Serialize for Requirements {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use crate::json::{U64List, U128Field, U128Option, U256Field};
        use serde::ser::SerializeStruct;
        let approx = !(s.is_human_readable() && self.approx_alpha_squared.is_none());
        let lifted = !(s.is_human_readable() && self.lifted_moduli.is_empty());
        let linf = !(s.is_human_readable() && self.linf.is_none());
        let fields = 9 + usize::from(approx) + usize::from(lifted) + usize::from(linf);
        let mut out = s.serialize_struct("Requirements", fields)?;
        out.serialize_field("m1", &self.m1)?;
        out.serialize_field("l", &self.l)?;
        out.serialize_field("alpha_squared", &U128Field(&self.alpha_squared))?;
        out.serialize_field("n_bin", &self.n_bin)?;
        out.serialize_field("l2_rows", &self.l2_rows)?;
        out.serialize_field("l2_bounds_squared", &U64List(&self.l2_bounds_squared))?;
        out.serialize_field("n_prime", &self.n_prime)?;
        out.serialize_field("linf_bound", &U128Field(&self.linf_bound))?;
        out.serialize_field(
            "max_integer_coefficient",
            &U256Field(&self.max_integer_coefficient),
        )?;
        if approx {
            out.serialize_field(
                "approx_alpha_squared",
                &U128Option(&self.approx_alpha_squared),
            )?;
        } else {
            out.skip_field("approx_alpha_squared")?;
        }
        if lifted {
            out.serialize_field("lifted_moduli", &self.lifted_moduli)?;
        } else {
            out.skip_field("lifted_moduli")?;
        }
        if linf {
            out.serialize_field("linf", &self.linf)?;
        } else {
            out.skip_field("linf")?;
        }
        out.end()
    }
}
#[derive(Clone, Debug)]
struct Variable {
    name: String,
    start: usize,
    count: usize,
    /// The degree $`d_v`$ of the subring the elements lie in, $`d'`$ for a full block.
    degree: usize,
    norm: Norm,
    placement: Placement,
}
impl Variable {
    /// The proof-ring polynomials the block commits at proof degree `d`, which divides
    /// $`d_v`$: $`d_v/d`$ components per element, each as $`n`$ bits for
    /// [`Norm::LinfExact`].
    fn rows(&self, d: usize) -> usize {
        let bits = match self.norm {
            Norm::LinfExact(beta) => bit_count(beta),
            _ => 1,
        };
        self.count * (self.degree / d) * bits
    }
}

/// The number of bits of a [`Norm::LinfExact`] coefficient: $`\mathrm{bitlen}(2\beta)`$.
fn bit_count(beta: u64) -> usize {
    (u64::BITS - (2 * beta).leading_zeros()) as usize
}
/// The weights $`c_b=\lfloor(B+2^b)/2^{b+1}\rfloor`$, $`b<\mathrm{bitlen}(B)`$, of the bits of a
/// [`Norm::LinfExact`] coefficient, $`B=2\beta<2^{64}`$, computed as
/// $`\lfloor B/2^b\rfloor-\lfloor B/2^{b+1}\rfloor`$.
///
/// Their subset sums are exactly $`[0,B]`$. By Hermite's identity
/// $`\lfloor y+\tfrac12\rfloor=\lfloor2y\rfloor-\lfloor y\rfloor`$ at $`y=B/2^{b+1}`$ the two
/// forms agree, so the weights sum to $`B`$ (the differences telescope) and those after $`b`$ to
/// $`a=\lfloor B/2^{b+1}\rfloor`$. As $`\lfloor B/2^b\rfloor\in\{2a,2a+1\}`$ is at least 1 for
/// $`b<\mathrm{bitlen}(B)`$, $`1\le c_b\le a+1`$. Every subset sum is therefore in $`[0,B]`$,
/// and `digits` reaches every $`x\in[0,B]`$: in index order it keeps the remainder at most the
/// sum of the weights not yet visited, since taking $`c_b`$ from at most $`c_b+a`$ leaves at
/// most $`a`$, and a remainder below $`c_b\le a+1`$ is at most $`a`$; after the last weight
/// the remainder is 0. Plain powers of two would be wrong: for $`B=10`$ they reach 15.
fn weights(beta: u64) -> Vec<u64> {
    let b = 2 * beta;
    (0..bit_count(beta) as u32)
        .map(|i| (b >> i) - b.checked_shr(i + 1).unwrap_or(0))
        .collect()
}
/// The greedy bits of $`x\in[0,B]`$ for `weights`, from the first weight on, into `bits`: bit
/// $`b`$ is 1 when the remainder is at least $`c_b`$, taken from the borrow of the subtraction
/// rather than from a branch in the source (the crate makes no constant-time claim).
fn digits(mut x: u64, weights: &[u64], bits: &mut [u64]) {
    for (w, bit) in weights.iter().zip(bits.iter_mut()) {
        let (_, below) = x.overflowing_sub(*w);
        *bit = u64::from(!below);
        x -= *bit * w;
    }
    debug_assert_eq!(x, 0);
}

/// Where a lowered variable, a component of a statement polynomial, lives among the proof-ring
/// witness polynomials.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lowered {
    /// Committed as this proof-ring polynomial.
    Committed(usize),
    /// Identically zero: a component outside the subring of its variable.
    Zero,
    /// A [`Norm::LinfExact`] component, $`\sum_bc_bx_b-\beta J`$ for the bits $`x_b`$ from
    /// `first` on and the all-ones polynomial $`J`$.
    Bits { first: usize, beta: u64 },
}
impl Lowered {
    /// The affine form that replaces the lowered variable, in interleaved indices.
    fn affine(&self, ring: &Arc<Ring>) -> Result<Affine, Error> {
        let index = |i: usize| u16::try_from(2 * i).map_err(|_| Error::Dimension);
        Ok(match *self {
            Lowered::Committed(i) => Affine {
                terms: vec![(index(i)?, Poly::constant(ring.clone(), 1))],
                constant: Poly::zero(ring.clone()),
            },
            Lowered::Zero => Affine {
                terms: Vec::new(),
                constant: Poly::zero(ring.clone()),
            },
            Lowered::Bits { first, beta } => Affine {
                terms: weights(beta)
                    .into_iter()
                    .enumerate()
                    .map(|(b, w)| Ok((index(first + b)?, Poly::constant(ring.clone(), w.into()))))
                    .collect::<Result<_, Error>>()?,
                constant: Poly::new(ring.clone(), vec![-i128::from(beta); ring.degree()])?,
            },
        })
    }
}
/// The committed proof-ring polynomials of a variable in layout order: its components', or
/// their bits.
fn committed(layout: &[Lowered], var: &Variable, k: usize) -> Vec<usize> {
    layout[var.start * k..(var.start + var.count) * k]
        .iter()
        .flat_map(|x| match *x {
            Lowered::Committed(i) => i..i + 1,
            Lowered::Bits { first, beta } => first..first + bit_count(beta),
            Lowered::Zero => 0..0,
        })
        .collect()
}
#[derive(Clone, Debug)]
struct Constraint {
    form: QuadEq,
    constant_only: bool,
}
impl Constraint {
    /// The ring of the form: the constraint holds modulo its modulus.
    fn ring(&self) -> &Arc<Ring> {
        self.form.r0.ring()
    }
    /// Whether lifting commits carries, as for constant-coefficient and quadratic constraints,
    /// rather than leaving the quotients implicit.
    fn commits_carries(&self) -> bool {
        self.constant_only || self.form.r2.entries().any(|(_, p)| !p.is_zero())
    }
}

/// Public statement builder. Declare all variables before constructing expressions.
///
/// Each constraint holds modulo the modulus of the ring its form is built over: the statement
/// ring for `variable` and `constant`, and any ring of the statement's degree for
/// `variable_in` and `constant_in`. `compile` compiles constraints at the proof modulus
/// natively and lifts every other one, each with its own modulus. `var` puts a bounded block in
/// the Ajtai part of the commitment; `var_placed` can put it in the BDLOP part ([`Placement`]),
/// which changes the parameters but not the relation. `var_subring` declares a block of
/// subring elements, of which only the nonzero components are committed.
///
/// # Witness relation
///
/// Witness blocks are polynomials over the statement ring, and every coefficient stands for
/// the integer of its centred representative modulo the statement modulus $`p`$, in
/// $`(-p/2,p/2]`$. A witness satisfies the statement if these integers meet every declared
/// norm, every coefficient of a block declared with `var_subring` outside its subring is 0, and
/// every constraint $`f_j`$ vanishes on them modulo its modulus $`p_j`$.
///
/// A proof establishes an integer relation: knowledge of integers $`\bar s`$ within the declared
/// norms, zero outside the subring of each subring block, with values modulo the proof modulus
/// $`q`$ for unbounded blocks and, for an $`\ell_\infty`$ block ([`Norm::Linf`]), every value
/// bounded by the extraction bound $`E`$ of the approximate range proof instead of its
/// $`\beta`$, such that $`f_j(\bar s)\equiv0\pmod{p_j}`$ for every constraint ($`p_j=q`$ for a
/// native one). An $`\ell_\infty`$ block is thus a relaxed claim: the prover needs $`\beta`$,
/// the proof shows $`E`$ ([`Compiled::extraction_bound`]). A block bounded exactly
/// ([`Norm::LinfExact`]) keeps its $`\beta`$. Reducing $`\bar s`$ modulo $`p`$ (an unbounded
/// value through its centred representative modulo $`q`$) never increases an absolute value, so
/// it keeps these bounds, and it keeps every constraint whose modulus divides $`p`$, among them
/// those over the statement ring. It gives a witness of the relaxed relation under a range
/// condition, a soundness condition that `requirements` and `compile` enforce: for every
/// constraint whose modulus does not divide $`p`$, every variable it uses with a nonzero
/// coefficient is binary, has a bound $`\|s\|^2\le B`$ with $`\lfloor\sqrt B\rfloor\le(p-1)/2`$,
/// is bounded exactly by $`\beta\le(p-1)/2`$, is an $`\ell_\infty`$ block with
/// $`E\le(p-1)/2`$, or is unbounded with $`p\ge q`$, so that the reduction changes no value the
/// constraint reads. Both refuse a statement that violates the condition with "variable range
/// above statement modulus"; for an $`\ell_\infty`$ block, whose $`E`$ depends on
/// `log_sigma[3]`, `requirements` exports the limit ([`LinfRequirements::extraction_limit`])
/// and `compile` checks it. Without the check, a proof could verify for a statement with no
/// witness: over $`\mathbb Z_{13}`$ with $`\|x\|^2\le64`$, no value in $`[-6,6]`$ satisfies
/// $`\mathrm{ct}(x_0)=7`$ modulo 12 and modulo 13, but the integer 7 does.
#[derive(Clone, Debug)]
pub struct Statement {
    ring: Arc<Ring>,
    variables: Vec<Variable>,
    count: usize,
    constraints: Vec<Constraint>,
}
impl Statement {
    /// Export the shape needed by the parameter tool for a chosen degree and modulus.
    /// Constraints whose modulus is `proof_modulus` are compiled natively, the others lifted;
    /// set `proof_modulus` to the statement modulus for native-ring compilation. Refuses a
    /// statement that violates the range condition of the [witness relation](Statement).
    ///
    /// A bounded block counts its rows ([`BlockRequirement::rows`]) in `m1` and its share in
    /// `alpha_squared` in the Ajtai part, and its rows in `l` in the BDLOP part
    /// ([`Placement`]). An $`\ell_\infty`$ variable ([`Norm::Linf`]) adds
    /// $`\mathrm{count}\cdot d_v\beta^2`$ to `alpha_squared` for its degree $`d_v`$ (in the
    /// Ajtai part), one approximate-range row per committed proof-ring polynomial to `n_prime`,
    /// and $`\beta`$ to the maximum `linf_bound`. Its soundness conditions, which depend on
    /// `log_sigma[3]`, are in `linf`. A [`Norm::LinfExact`] variable adds its bits to `n_bin`
    /// and $`\mathrm{count}\cdot d_v`$ per bit to `alpha_squared` (in the Ajtai part), and
    /// nothing to `n_prime` or `linf_bound`. The lifted constant-coefficient clauses add
    /// $`\lceil m/d\rceil`$ to `l` and `n_prime` together ([`Statement::const_coeff_zero`]).
    /// Refuses, as "target ring", a proof degree that does not divide the degree of every
    /// block.
    pub fn requirements(
        &self,
        proof_degree: usize,
        proof_modulus: U256,
    ) -> Result<Requirements, Error> {
        if proof_modulus < U256::from_u8(2) {
            return Err(Error::Parameter("target ring"));
        }
        self.check_proof_degree(proof_degree)?;
        let k = self.ring.degree() / proof_degree;
        let mut out = Requirements {
            m1: 0,
            l: 0,
            alpha_squared: 0,
            n_bin: 0,
            l2_rows: Vec::new(),
            l2_bounds_squared: Vec::new(),
            n_prime: 0,
            linf_bound: 0,
            max_integer_coefficient: U256::ZERO,
            approx_alpha_squared: None,
            lifted_moduli: Vec::new(),
            linf: None,
        };
        let mut linf_rows = 0;
        // The per-slot squared bound of the range vector: slots times squared bound, per row
        // kind (Requirements::approx_alpha_squared).
        let mut slots = Bound::ZERO;
        let statement_degree = Bound::from_u64(self.ring.degree() as u64);
        let squared = |beta: u128| {
            let beta = Bound::from_u128(beta);
            mul_bound(beta, beta)
        };
        for var in &self.variables {
            let rows = var.rows(proof_degree);
            match var.norm {
                Norm::L2Squared(bound) => {
                    out.l2_rows.push(rows);
                    out.l2_bounds_squared.push(bound);
                }
                // The bits of a LinfExact block are binary rows.
                Norm::Binary | Norm::LinfExact(_) => out.n_bin += rows,
                Norm::Linf(beta) => {
                    linf_rows += rows;
                    out.linf_bound = out.linf_bound.max(u128::from(beta));
                    // count * d_v coefficients of absolute value at most beta.
                    let coefficients = Bound::from_u64((var.count * var.degree) as u64);
                    slots = add_bound(slots, mul_bound(coefficients, squared(u128::from(beta))?)?)?;
                }
                Norm::Unbounded => {}
            }
            if var.placement == Placement::Ajtai {
                out.m1 += rows;
                out.alpha_squared = self
                    .alpha_share(var)?
                    .checked_add(out.alpha_squared)
                    .ok_or(Error::Overflow)?;
            } else {
                out.l += rows;
            }
        }
        // Every constraint not at the proof modulus is lifted; the largest F per modulus.
        let mut lifted = BTreeMap::<U256, U256>::new();
        let mut linf_lifting = Vec::new();
        // Lifted constant-coefficient clauses, whose carries share packed polynomials.
        let mut clauses = 0usize;
        for constraint in &self.constraints {
            let ring = constraint.ring();
            if ring.modulus() == proof_modulus {
                continue;
            }
            let bound = form_bound(self, &constraint.form)?;
            let f = bound.honest;
            let quotient = quotient_bound(&f, ring)?;
            // One slot for a clause, d' for an eq_mod_p, each at most ceil(F_j / p_j).
            let row_slots = if constraint.constant_only {
                clauses += 1;
                Bound::ONE
            } else {
                out.n_prime += k;
                if constraint.commits_carries() {
                    out.l += k;
                }
                statement_degree
            };
            slots = add_bound(slots, mul_bound(row_slots, squared(quotient)?)?)?;
            out.max_integer_coefficient = out.max_integer_coefficient.max(f);
            out.linf_bound = out.linf_bound.max(quotient);
            let largest = lifted.entry(ring.modulus()).or_insert(U256::ZERO);
            *largest = (*largest).max(f);
            if let Some(entry) = bound.linf {
                linf_lifting.push(entry);
            }
        }
        // d carries per packed polynomial, each a committed message with one range row.
        let packed = clauses.div_ceil(proof_degree);
        out.n_prime += packed;
        out.l += packed;
        // The rows of the ℓ∞ variables follow those of the constraints.
        out.n_prime += linf_rows;
        let extraction_limit = self.check_ranges(&proof_modulus)?;
        // A present ARP block must have a positive bound, even for an identically zero map.
        if out.n_prime > 0 {
            out.linf_bound = out.linf_bound.max(1);
        }
        // n'd coefficients of absolute value at most linf_bound, the bound the prover uses.
        let rows = Bound::from_u64((out.n_prime * proof_degree) as u64);
        if slots < mul_bound(rows, squared(out.linf_bound)?)? {
            out.approx_alpha_squared = narrow_bound(slots).ok().and_then(|x| int::to_u128(&x));
        }
        if lifted.keys().any(|p| *p != self.ring.modulus()) {
            out.lifted_moduli = lifted
                .into_iter()
                .map(|(modulus, max_integer_coefficient)| LiftedModulus {
                    modulus,
                    max_integer_coefficient,
                })
                .collect();
        }
        if self
            .variables
            .iter()
            .any(|v| matches!(v.norm, Norm::Linf(_)))
        {
            out.linf = Some(LinfRequirements {
                lifting: linf_lifting,
                extraction_limit,
            });
        }
        Ok(out)
    }
    /// The range condition of the witness relation: every variable of a constraint whose
    /// modulus does not divide the statement modulus $`p`$ takes only values in
    /// $`(-p/2,p/2]`$ in every integer witness, so that reducing modulo $`p`$ keeps them. An
    /// $`\ell_\infty`$ variable ([`Norm::Linf`]) is bounded by the extraction bound $`E`$,
    /// which `compile` knows: when one is involved, returns the limit $`\lfloor(p-1)/2\rfloor`$
    /// on $`E`$. One bounded exactly keeps its $`\beta`$.
    fn check_ranges(&self, proof_modulus: &U256) -> Result<Option<U256>, Error> {
        let p = self.ring.modulus();
        let mut norms = vec![Norm::Unbounded; self.count];
        for var in &self.variables {
            norms[var.start..var.start + var.count].fill(var.norm);
        }
        let mut limit = None;
        for constraint in &self.constraints {
            let ring = constraint.ring();
            if p.div_rem_vartime(&ring.nonzero).1 == U256::ZERO {
                continue;
            }
            let form = &constraint.form;
            let linear = form
                .r1
                .entries()
                .filter(|(_, c)| !c.is_zero())
                .map(|(i, _)| i);
            let quadratic = form.r2.entries().filter(|(_, c)| !c.is_zero());
            for i in linear.chain(quadratic.flat_map(|((i, j), _)| [i, j])) {
                let fits = match norms[usize::from(i)] {
                    Norm::Binary => true,
                    // Every value is at most isqrt(B) in absolute value: 2 isqrt(B) < p.
                    Norm::L2Squared(b) => U256::from_u64(2 * b.isqrt()) < p,
                    // Every extracted value is at most beta < 2^63: 2 beta < p, strictly, as for
                    // an even p the bits also encode -p/2, which a witness reads as p/2.
                    Norm::LinfExact(beta) => U256::from_u64(2 * beta) < p,
                    // Every extracted value is at most E: `compile` checks 2E < p.
                    Norm::Linf(_) => {
                        limit = Some(p.wrapping_sub(&U256::ONE).shr_vartime(1));
                        true
                    }
                    // Only in constraints at q (lifting refuses it), with every value
                    // modulo q represented in (-p/2, p/2].
                    Norm::Unbounded => ring.modulus() == *proof_modulus && p >= *proof_modulus,
                };
                if !fits {
                    return Err(Error::Parameter("variable range above statement modulus"));
                }
            }
        }
        Ok(limit)
    }
    /// A block's share of `alpha_squared` in the Ajtai part: the squared Euclidean bound of the
    /// $`\mathrm{count}\cdot d_v`$ possibly nonzero coefficients of its elements of degree
    /// $`d_v`$, or of their bits.
    fn alpha_share(&self, var: &Variable) -> Result<u128, Error> {
        let coefficients = (var.count * var.degree) as u128;
        match var.norm {
            Norm::L2Squared(bound) => Ok(u128::from(bound)),
            Norm::Binary => Ok(coefficients),
            // count * d_v coefficients of absolute value at most beta.
            Norm::Linf(beta) => u128::from(beta)
                .checked_mul(u128::from(beta))
                .and_then(|x| x.checked_mul(coefficients))
                .ok_or(Error::Overflow),
            // Bits: count * d_v binary coefficients per bit.
            Norm::LinfExact(beta) => Ok(coefficients * bit_count(beta) as u128),
            Norm::Unbounded => Ok(0),
        }
    }
    /// A proof degree of 64 or 128 that divides the degree of every block, and so the
    /// statement degree; "target ring" otherwise.
    fn check_proof_degree(&self, proof_degree: usize) -> Result<(), Error> {
        if !matches!(proof_degree, 64 | 128)
            || !self.ring.degree().is_multiple_of(proof_degree)
            || self
                .variables
                .iter()
                .any(|v| !v.degree.is_multiple_of(proof_degree))
        {
            return Err(Error::Parameter("target ring"));
        }
        Ok(())
    }
    /// The declared blocks in declaration order, with their committed proof-ring rows for
    /// `proof_degree`, for a parameter search that moves blocks between the parts
    /// ([`BlockRequirement`]). Refuses a degree that `requirements` refuses.
    pub fn blocks(&self, proof_degree: usize) -> Result<Vec<BlockRequirement>, Error> {
        self.check_proof_degree(proof_degree)?;
        Ok(self
            .variables
            .iter()
            .map(|var| BlockRequirement {
                name: var.name.clone(),
                rows: var.rows(proof_degree),
                norm: var.norm,
                placement: var.placement,
            })
            .collect())
    }
    /// Start a statement over a validated statement ring.
    pub fn new(ring: Arc<Ring>) -> Self {
        Self {
            ring,
            variables: Vec::new(),
            count: 0,
            constraints: Vec::new(),
        }
    }
    /// Declare a uniquely named witness block. Returns its first scalar polynomial index.
    /// Refuses [`Norm::Linf`]`(0)`, [`Norm::LinfExact`]`(0)` and [`Norm::LinfExact`] from
    /// $`2^{63}`$ on. A bounded block goes to the Ajtai part of the commitment, an unbounded one
    /// to the BDLOP part ([`Placement::default_for`]).
    pub fn var(
        &mut self,
        name: impl Into<String>,
        count: usize,
        norm: Norm,
    ) -> Result<usize, Error> {
        self.var_placed(name, count, norm, Placement::default_for(norm))
    }
    /// [`Statement::var`] in the given part of the commitment ([`Placement`]). Refuses an
    /// unbounded block in the Ajtai part, which bounds everything it holds.
    pub fn var_placed(
        &mut self,
        name: impl Into<String>,
        count: usize,
        norm: Norm,
        placement: Placement,
    ) -> Result<usize, Error> {
        let degree = self.ring.degree();
        self.var_subring(name, count, degree, norm, placement)
    }
    /// [`Statement::var_placed`] for a block of `count` elements of the subring
    /// $`S=\{v(X^K)\}`$ of degree $`d_v`$ = `degree`, $`K=d'/d_v`$, a power of two with
    /// $`64\le d_v\le d'`$ ("variable declaration" otherwise; $`d_v=d'`$ is `var_placed`).
    ///
    /// $`Z\mapsto X^K`$ is a ring homomorphism, so every form may use the block, and a witness
    /// element is a statement-ring polynomial whose coefficients at indices not divisible by
    /// $`K`$ are 0 (build it with [`iso::embed`]; the witness map refuses any other).
    /// [`Statement::coefficient_in`] reads its coefficients. Lowered to proof degree $`d`$, the
    /// components $`c\equiv0\pmod K`$ of each element are those of $`v`$ as a polynomial of
    /// degree $`d_v`$, and every other one is 0: the compiler commits only these $`d_v/d`$
    /// components per element (for [`Norm::LinfExact`], their bits) and drops the terms of the
    /// others, so the block costs $`d_v/d`$ rows per element instead of $`d'/d`$. A proof degree
    /// that does not divide $`d_v`$ is refused ("target ring"). The norm applies to the whole
    /// block as for `var`: a binary block's count of coefficients, and an $`\ell_\infty`$
    /// block's Euclidean norm, use $`\mathrm{count}\cdot d_v`$ coefficients.
    pub fn var_subring(
        &mut self,
        name: impl Into<String>,
        count: usize,
        degree: usize,
        norm: Norm,
        placement: Placement,
    ) -> Result<usize, Error> {
        let name = name.into();
        if name.is_empty()
            || name.len() > 256
            || count == 0
            || matches!(norm, Norm::Linf(0) | Norm::LinfExact(0))
            || matches!(norm, Norm::LinfExact(beta) if beta >= 1 << 63)
            || !degree.is_power_of_two()
            || !(64..=self.ring.degree()).contains(&degree)
            || (norm == Norm::Unbounded && placement == Placement::Ajtai)
            || self.variables.iter().any(|v| v.name == name)
            || !self.constraints.is_empty()
        {
            return Err(Error::Parameter("variable declaration"));
        }
        let next = self
            .count
            .checked_add(count)
            .filter(|n| *n <= 32768)
            .ok_or(Error::Dimension)?;
        let start = self.count;
        self.variables.push(Variable {
            name,
            start,
            count,
            degree,
            norm,
            placement,
        });
        self.count = next;
        Ok(start)
    }
    /// Expression for one component of a named variable, using the current variable space.
    /// Compose expressions with `QuadEq::{add,scale,product_affine}`.
    pub fn variable(&self, name: &str, component: usize) -> Result<QuadEq, Error> {
        self.variable_in(&self.ring, name, component)
    }
    /// [`Statement::variable`] over `ring`, which must have the statement's degree and may have
    /// another modulus: a constraint built over it holds modulo that modulus. Fails with
    /// [`Error::RingMismatch`] for another degree.
    pub fn variable_in(
        &self,
        ring: &Arc<Ring>,
        name: &str,
        component: usize,
    ) -> Result<QuadEq, Error> {
        if ring.degree() != self.ring.degree() {
            return Err(Error::RingMismatch);
        }
        let var = self
            .variables
            .iter()
            .find(|v| v.name == name)
            .ok_or(Error::Index)?;
        if component >= var.count {
            return Err(Error::Index);
        }
        let mut eq = QuadEq::zero(ring.clone(), self.count)?;
        eq.r1 = SparsePolyVec::new(
            ring.clone(),
            self.count,
            vec![(
                (var.start + component) as u16,
                Poly::constant(ring.clone(), 1),
            )],
        )?;
        Ok(eq)
    }
    /// [`Statement::variable_in`] times $`X^{-K\cdot\mathrm{index}}`$, $`K=d'/d_v`$ for the
    /// variable's degree $`d_v`$ ([`Statement::var_subring`]; $`K=1`$ for a full block), whose
    /// constant coefficient is coefficient `index` of the element $`v`$, for
    /// [`Statement::const_coeff_zero`] clauses: in $`X^{-Ki}\sum_jv_jX^{Kj}`$ only $`j=i`$ gives
    /// exponent 0, as $`X^{K(j-i)}=-X^{d'+K(j-i)}`$ for $`j<i`$. Fails with [`Error::Index`] for
    /// an `index` of at least $`d_v`$, and as `variable_in` otherwise.
    pub fn coefficient_in(
        &self,
        ring: &Arc<Ring>,
        name: &str,
        component: usize,
        index: usize,
    ) -> Result<QuadEq, Error> {
        let form = self.variable_in(ring, name, component)?;
        let var = self
            .variables
            .iter()
            .find(|v| v.name == name)
            .ok_or(Error::Index)?;
        if index >= var.degree {
            return Err(Error::Index);
        }
        let stride = self.ring.degree() / var.degree;
        form.scale(&Poly::constant(ring.clone(), 1).rotate(-((stride * index) as i64)))
    }
    /// Public constant expression over the statement ring.
    pub fn constant(&self, value: Poly) -> Result<QuadEq, Error> {
        if value.ring() != &self.ring {
            return Err(Error::RingMismatch);
        }
        self.constant_in(value)
    }
    /// Public constant expression over the ring of `value`, which must have the statement's
    /// degree and may have another modulus, as in [`Statement::variable_in`].
    pub fn constant_in(&self, value: Poly) -> Result<QuadEq, Error> {
        if value.ring().degree() != self.ring.degree() {
            return Err(Error::RingMismatch);
        }
        let mut eq = QuadEq::zero(value.ring().clone(), self.count)?;
        eq.r0 = value;
        Ok(eq)
    }
    /// Require an expression to vanish in the ring it is built over, that is, modulo that
    /// ring's modulus.
    pub fn eq_mod_p(&mut self, form: QuadEq) -> Result<&mut Self, Error> {
        self.push(form, false)
    }
    /// Require only its constant coefficient to vanish modulo the modulus of its ring.
    ///
    /// When that modulus $`p_j`$ is not the proof modulus $`q`$, `compile` lifts the clause with
    /// a committed carry, the integer quotient of the constant coefficient by $`p_j`$, which the
    /// approximate range proof bounds. The carries of all such clauses are packed: with proof
    /// degree $`d`$, clause $`i`$ of the $`m`$ lifted ones (in declaration order) holds
    /// coefficient $`t=i\bmod d`$ of the packed carry polynomial $`C_r`$,
    /// $`r=\lfloor i/d\rfloor`$, and its compiled evaluation equation adds $`-p_jX^{-t}C_r`$,
    /// whose constant coefficient is $`-p_jC_{r,t}`$ in $`\mathbb Z_q[X]/(X^d+1)`$. The
    /// $`\lceil m/d\rceil`$ packed polynomials are BDLOP messages with one range row each,
    /// placed where the first clause's carry is, and unused coefficients hold 0; one clause
    /// compiles as before packing. Every clause keeps its own lifting condition
    /// ([`Statement::compile`]): the range proof bounds each coefficient of $`C_r`$, and the
    /// clause reads only its own.
    pub fn const_coeff_zero(&mut self, form: QuadEq) -> Result<&mut Self, Error> {
        self.push(form, true)
    }
    /// A form over one ring of the statement's degree, in the statement's variable space.
    fn push(&mut self, form: QuadEq, constant_only: bool) -> Result<&mut Self, Error> {
        form.check(form.r0.ring(), self.count)?;
        if form.r0.ring().degree() != self.ring.degree() {
            return Err(Error::RingMismatch);
        }
        self.constraints.push(Constraint {
            form,
            constant_only,
        });
        Ok(self)
    }
    /// Compile against an offline parameter set, checking dimensions and lifting inequalities.
    ///
    /// A constraint of modulus $`p_j\ne q`$ is lifted: the proof shows
    /// $`f_j(s)-p_jc_j\equiv0\pmod q`$ with each quotient or carry $`c_j`$ in the approximate
    /// range proof (for a constant-coefficient clause, one coefficient of a packed carry
    /// polynomial, [`Statement::const_coeff_zero`]). An extracted witness satisfies the declared
    /// exact norms, and its carries and
    /// $`\ell_\infty`$ variables are bounded by $`E=`$ `approx_extraction_bound` (LNP22 Lemma
    /// 2.7), so $`|f_j(s)|\le F_j`$, the bound computed with $`E`$ in place of the $`\beta`$ of
    /// every $`\ell_\infty`$ variable ([`LinfLifting`]), and $`|c_j|\le E`$. The compiler
    /// requires, for every lifted constraint, $`q>2(F_j+p_j\beta_\infty\psi)`$ with
    /// $`\beta_\infty`$ = `linf_bound`, $`t_4`$ = `log_sigma[3]` and
    /// $`\psi=\lceil28\cdot1.55\cdot2^{t_4}/\beta_\infty\rceil`$, which exceeds
    /// $`F_j+p_jE`$. Then
    /// $`f_j(s)-p_jc_j`$ is a multiple of $`q`$ smaller than $`q`$, hence zero, and
    /// $`f_j(s)\equiv0\pmod{p_j}`$ (LNP22 §1.3, Eq. (12)). Each constraint uses only its own
    /// modulus, which may be composite; $`q`$ must be prime for the range proof. A modulus above
    /// $`q`$ fails this check. Through `requirements`, the compiler also refuses a statement that
    /// violates the range condition of the [witness relation](Statement), under which these
    /// integer relations give a witness over the statement ring. With $`\ell_\infty`$ variables
    /// it also requires `linf_bound` $`\ge\beta`$ for each, $`2E+1<q`$, so that $`E`$ bounds
    /// anything, and the range condition's limit on $`E`$.
    pub fn compile(&self, params: TboxParams) -> Result<Compiled, Error> {
        compile(self.clone(), params)
    }
}

/// Compiled toolbox statement and the deterministic witness map used by the prover.
#[derive(Clone, Debug)]
pub struct Compiled {
    source: Statement,
    params: TboxParams,
    statement: lnp::Statement,
    ring: Arc<Ring>,
    k: usize,
    /// Where each lowered variable lives, indexed by statement polynomial times $`k`$ plus
    /// component.
    layout: Vec<Lowered>,
    bounded: usize,
    carries: Vec<Carry>,
    /// `approx_extraction_bound` of the parameters.
    extraction: u128,
}

/// Where the committed carries of a lifted constraint sit among the proof-ring messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Carry {
    /// A quadratic `eq_mod_p`: its carry polynomial's `count` components from `start`.
    Components {
        constraint: usize,
        start: usize,
        count: usize,
    },
    /// A constant-coefficient clause: coefficient `coefficient` of the packed polynomial `poly`.
    Slot {
        constraint: usize,
        poly: usize,
        coefficient: usize,
    },
}
impl Carry {
    fn constraint(&self) -> usize {
        match self {
            Carry::Components { constraint, .. } | Carry::Slot { constraint, .. } => *constraint,
        }
    }
}

/// What a verified proof guarantees about one named variable of the witness that extraction
/// obtains ([`Compiled::extraction_bound`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extraction {
    /// $`\|s\|^2\le B`$, the declared bound, which the exact-norm proof extracts exactly.
    L2Squared(u64),
    /// Every coefficient zero or one, as declared.
    Binary,
    /// Every coefficient of absolute value at most this bound: the approximate range proof's
    /// `approx_extraction_bound` $`=2\cdot`$`z4_bound` (LNP22 Lemma 2.7), whatever the declared
    /// $`\beta`$ of the [`Norm::Linf`] variable.
    Linf(u128),
    /// Every coefficient of absolute value at most the declared $`\beta`$ of the
    /// [`Norm::LinfExact`] variable, exactly: its bits are binary, and their weights reach
    /// exactly $`[0,2\beta]`$.
    LinfExact(u64),
    /// No bound: values modulo the proof modulus.
    Unbounded,
}

/// Bounds are computed exactly in 1024 bits: squared norms of 256-bit coefficients over 1024
/// coefficients need up to 522 bits, times a 64-bit bound.
type Bound = U1024;

fn add_bound(a: Bound, b: Bound) -> Result<Bound, Error> {
    Option::from(a.checked_add(&b)).ok_or(Error::Overflow)
}
fn mul_bound(a: Bound, b: Bound) -> Result<Bound, Error> {
    Option::from(a.checked_mul(&b)).ok_or(Error::Overflow)
}
fn ceil_sqrt(x: Bound) -> Bound {
    let r = x.floor_sqrt_vartime();
    if r.wrapping_mul(&r) == x {
        r
    } else {
        r.wrapping_add(&Bound::ONE)
    }
}
/// Exact squared Euclidean norm of the centred representatives.
fn exact_norm_squared(p: &Poly) -> Result<Bound, Error> {
    p.coefficients().iter().try_fold(Bound::ZERO, |sum, x| {
        let m = p.ring().magnitude(x).resize::<{ Bound::LIMBS }>();
        add_bound(sum, m.wrapping_mul(&m))
    })
}
/// $`\|p\|_1`$ of the centred representatives.
fn l1_norm(p: &Poly) -> Result<Bound, Error> {
    p.coefficients().iter().try_fold(Bound::ZERO, |sum, x| {
        add_bound(sum, p.ring().magnitude(x).resize::<{ Bound::LIMBS }>())
    })
}
/// A value below $`2^{256}`$, and [`Error::Overflow`] otherwise.
fn narrow_bound(x: Bound) -> Result<U256, Error> {
    if x.bits_vartime() > 256 {
        return Err(Error::Overflow);
    }
    Ok(x.resize::<{ U256::LIMBS }>())
}
/// The bound on a form's integer value: `honest` on every witness within the declared norms,
/// and, for a form with $`\ell_\infty`$ variables, as a function of a common bound $`E`$ on them.
struct FormBound {
    honest: U256,
    linf: Option<LinfLifting>,
}
impl FormBound {
    /// The bound on an extracted witness, whose $`\ell_\infty`$ variables are at most $`e`$.
    fn extracted(&self, e: u128) -> Result<Bound, Error> {
        match &self.linf {
            Some(linf) => linf.at(e),
            None => Ok(self.honest.resize::<{ Bound::LIMBS }>()),
        }
    }
}
/// The bound $`F`$ on the integer evaluation, exactly, and [`Error::Overflow`] from $`2^{256}`$
/// on, where no proof modulus can exceed $`2F`$. A term with an $`\ell_\infty`$ variable $`y`$
/// of bound $`b`$ and degree $`n_y`$ (the statement degree $`d'`$, or the subring degree of
/// [`Statement::var_subring`], whose other coefficients are 0) uses $`\|y\|_\infty\le b`$ and
/// $`\|y\|_2\le\sqrt{n_y}\,b`$: $`\|r\|_1b`$ for $`r\,y`$, $`\|r\|_1\lceil\sqrt{n_yB}\rceil b`$
/// for $`r\,y\,s`$ with $`s`$ in a block of squared bound $`B`$ (for a binary block, its
/// $`\mathrm{count}\cdot d_v`$ coefficients), and $`\|r\|_1\lceil\sqrt{n_yn_{y'}}\rceil bb'`$ for
/// $`r\,y\,y'`$ ($`\|r\|_1d'bb'`$ at full degree). A [`Norm::LinfExact`] variable's $`\beta`$ is
/// its bound both honestly and at extraction, so its terms count in `exact`; a
/// [`Norm::Linf`] one's is $`E`$ at extraction.
fn form_bound(statement: &Statement, form: &QuadEq) -> Result<FormBound, Error> {
    /// A variable's bound: squared Euclidean, or on every coefficient with its degree, taken
    /// at extraction as $`E`$ (`Linf`) or as declared (`Exact`).
    #[derive(Clone, Copy)]
    enum Kind {
        Squared(Bound),
        Linf(Bound, Bound),
        Exact(Bound, Bound),
        Unbounded,
    }
    let mut kind = vec![Kind::Unbounded; statement.count];
    for var in &statement.variables {
        let degree = Bound::from_u64(var.degree as u64);
        let b = match var.norm {
            Norm::L2Squared(b) => Kind::Squared(Bound::from_u64(b)),
            Norm::Binary => Kind::Squared(Bound::from_u64((var.count * var.degree) as u64)),
            Norm::Linf(beta) => Kind::Linf(Bound::from_u64(beta), degree),
            Norm::LinfExact(beta) => Kind::Exact(Bound::from_u64(beta), degree),
            Norm::Unbounded => Kind::Unbounded,
        };
        kind[var.start..var.start + var.count].fill(b);
    }
    let unbounded = || Error::Parameter("unbounded variable in lifted relation");
    let mut exact = form.r0.norm_infinity().resize::<{ Bound::LIMBS }>();
    // The weights of E and E^2, and the value of those terms at the declared bounds.
    let (mut linear, mut quadratic, mut declared) = (Bound::ZERO, Bound::ZERO, Bound::ZERO);
    let mut uses_linf = false;
    for (i, p) in form.r1.entries() {
        if p.is_zero() {
            continue;
        }
        match kind[usize::from(i)] {
            Kind::Squared(b) => {
                exact = add_bound(exact, ceil_sqrt(mul_bound(exact_norm_squared(p)?, b)?))?;
            }
            Kind::Linf(beta, _) => {
                let w = l1_norm(p)?;
                linear = add_bound(linear, w)?;
                declared = add_bound(declared, mul_bound(w, beta)?)?;
                uses_linf = true;
            }
            Kind::Exact(beta, _) => exact = add_bound(exact, mul_bound(l1_norm(p)?, beta)?)?,
            Kind::Unbounded => return Err(unbounded()),
        }
    }
    for ((i, j), p) in form.r2.entries() {
        if p.is_zero() {
            continue;
        }
        // ||r||_1 ceil(sqrt(x y)).
        let weight = |x: Bound, y: Bound| -> Result<Bound, Error> {
            mul_bound(l1_norm(p)?, ceil_sqrt(mul_bound(x, y)?))
        };
        match (kind[usize::from(i)], kind[usize::from(j)]) {
            (Kind::Squared(a), Kind::Squared(b)) => exact = add_bound(exact, weight(a, b)?)?,
            (Kind::Linf(beta, n), Kind::Squared(b)) | (Kind::Squared(b), Kind::Linf(beta, n)) => {
                let w = weight(n, b)?;
                linear = add_bound(linear, w)?;
                declared = add_bound(declared, mul_bound(w, beta)?)?;
                uses_linf = true;
            }
            (Kind::Linf(a, n), Kind::Linf(b, m)) => {
                let w = weight(n, m)?;
                quadratic = add_bound(quadratic, w)?;
                declared = add_bound(declared, mul_bound(mul_bound(w, a)?, b)?)?;
                uses_linf = true;
            }
            (Kind::Exact(beta, n), Kind::Squared(b)) | (Kind::Squared(b), Kind::Exact(beta, n)) => {
                exact = add_bound(exact, mul_bound(weight(n, b)?, beta)?)?;
            }
            (Kind::Exact(a, n), Kind::Exact(b, m)) => {
                exact = add_bound(exact, mul_bound(mul_bound(weight(n, m)?, a)?, b)?)?;
            }
            (Kind::Exact(a, n), Kind::Linf(b, m)) | (Kind::Linf(b, m), Kind::Exact(a, n)) => {
                let w = mul_bound(weight(n, m)?, a)?;
                linear = add_bound(linear, w)?;
                declared = add_bound(declared, mul_bound(w, b)?)?;
                uses_linf = true;
            }
            _ => return Err(unbounded()),
        }
    }
    let honest = narrow_bound(add_bound(exact, declared)?)?;
    // Every declared bound is at least 1, so each part is at most `honest`.
    let linf = if uses_linf {
        Some(LinfLifting {
            modulus: form.r0.ring().modulus(),
            exact: narrow_bound(exact)?,
            linear: narrow_bound(linear)?,
            quadratic: narrow_bound(quadratic)?,
        })
    } else {
        None
    };
    Ok(FormBound { honest, linf })
}
/// $`\lceil F/p\rceil`$ for the modulus $`p`$ of a constraint's ring, as `u128`.
fn quotient_bound(f: &U256, ring: &Ring) -> Result<u128, Error> {
    let (quotient, rest) = f.div_rem_vartime(&ring.nonzero);
    let quotient = if rest == U256::ZERO {
        quotient
    } else {
        quotient.wrapping_add(&U256::ONE)
    };
    int::to_u128(&quotient).ok_or(Error::Overflow)
}
/// The centred representative of each coefficient, reduced into the target ring.
fn lift(p: &Poly, target: Arc<Ring>) -> Result<Poly, Error> {
    let source = p.ring();
    if source.q == target.q {
        return Ok(Poly::from_canonical(target, p.coefficients().to_vec()));
    }
    let values = p
        .coefficients()
        .iter()
        .map(|x| target.reduce_i256(&source.centred_i256(x)))
        .collect();
    Ok(Poly::from_canonical(target, values))
}

/// Lower a quadratic form into one output equation per lower-ring component.
pub fn lower(form: &QuadEq, target: Arc<Ring>) -> Result<Vec<QuadEq>, Error> {
    form.check(form.r0.ring(), form.r2.dimension())?;
    let source_q = Ring::with_modulus(target.modulus(), form.r0.ring().degree())?;
    if !source_q.degree().is_multiple_of(target.degree()) {
        return Err(Error::Dimension);
    }
    let k = source_q.degree() / target.degree();
    let dim = form.r2.dimension() * k;
    let lifted = SparsePolyMat::new(
        source_q.clone(),
        form.r2.dimension(),
        form.r2
            .entries()
            .map(|((i, j), p)| Ok((i, j, lift(p, source_q.clone())?)))
            .collect::<Result<_, Error>>()?,
    )?;
    let matrices = iso::quadratic(&lifted, target.clone())?;
    let constants = iso::split(&lift(&form.r0, source_q.clone())?, target.clone())?;
    let mut linear = vec![BTreeMap::<u16, Poly>::new(); k];
    for (i, p) in form.r1.entries() {
        let matrix = iso::multiplication_matrix(&lift(p, source_q.clone())?, target.clone())?;
        for (row, entries) in linear.iter_mut().enumerate() {
            for col in 0..k {
                entries.insert(
                    u16::try_from(usize::from(i) * k + col).map_err(|_| Error::Dimension)?,
                    matrix.get(row, col)?.clone(),
                );
            }
        }
    }
    matrices
        .into_iter()
        .zip(linear)
        .zip(constants)
        .map(|((r2, linear), r0)| {
            Ok(QuadEq {
                r2,
                r1: SparsePolyVec::new(target.clone(), dim, linear.into_iter().collect())?,
                r0,
            })
        })
        .collect()
}
fn selector(
    ring: &Arc<Ring>,
    indices: &[usize],
    m1: usize,
    l: usize,
) -> Result<AffineBlock, Error> {
    // The rows' entries in one buffer of exactly their number: collected from the flattened
    // rows, whose number the iterator does not report, the buffer grew by doubling, to up to
    // twice the size of the entries.
    let matrix = |start: usize, cols: usize| {
        let mut entries =
            Vec::with_capacity(indices.len().checked_mul(cols).ok_or(Error::Dimension)?);
        for idx in indices {
            entries.extend(
                (0..cols).map(|j| Poly::constant(ring.clone(), i128::from(*idx == start + j))),
            );
        }
        PolyMat::new(ring.clone(), indices.len(), cols, entries)
    };
    Ok(AffineBlock {
        rows: indices.len(),
        s: Some(matrix(0, m1)?),
        m: Some(matrix(m1, l)?),
        offset: None,
    })
}
/// The linear form `coefficient` times proof-ring witness polynomial `index` (interleaved index
/// $`2\cdot`$`index`) in a space of dimension `dim`.
fn term(ring: &Arc<Ring>, dim: usize, index: usize, coefficient: Poly) -> Result<QuadEq, Error> {
    let mut form = QuadEq::zero(ring.clone(), dim)?;
    form.r1 = SparsePolyVec::new(
        ring.clone(),
        dim,
        vec![(
            u16::try_from(2 * index).map_err(|_| Error::Dimension)?,
            coefficient,
        )],
    )?;
    Ok(form)
}
fn affine_rows(
    ring: &Arc<Ring>,
    rows: &[QuadEq],
    m1: usize,
    l: usize,
) -> Result<AffineBlock, Error> {
    let mut a = vec![Poly::zero(ring.clone()); rows.len() * m1];
    let mut b = vec![Poly::zero(ring.clone()); rows.len() * l];
    let mut offsets = Vec::new();
    for (row, form) in rows.iter().enumerate() {
        if form.r2.entries().any(|(_, p)| !p.is_zero()) {
            return Err(Error::Parameter("nonlinear ARP map"));
        }
        for (i, p) in form.r1.entries() {
            let i = usize::from(i);
            if i % 2 != 0 {
                return Err(Error::Index);
            }
            let i = i / 2;
            if i < m1 {
                a[row * m1 + i] = p.clone();
            } else {
                b[row * l + i - m1] = p.clone();
            }
        }
        offsets.push(form.r0.clone());
    }
    Ok(AffineBlock {
        rows: rows.len(),
        s: Some(PolyMat::new(ring.clone(), rows.len(), m1, a)?),
        m: Some(PolyMat::new(ring.clone(), rows.len(), l, b)?),
        offset: Some(PolyVec::new(ring.clone(), offsets)?),
    })
}
fn compile(source: Statement, params: TboxParams) -> Result<Compiled, Error> {
    let checked = params.check()?;
    let requirements = source.requirements(params.degree, checked.q)?;
    if requirements.alpha_squared > u128::from(params.alpha_squared) {
        return Err(Error::Parameter("bounded witness norm budget"));
    }
    let ring = Ring::with_modulus(checked.q, params.degree)?;
    if !source.ring.degree().is_multiple_of(params.degree) {
        return Err(Error::Dimension);
    }
    let k = source.ring.degree() / params.degree;
    // The Ajtai part, then the BDLOP part: bounded blocks placed there, unbounded blocks and,
    // below, the carries. With `var`'s placements this is the order before `var_placed`.
    let parts: [fn(&Variable) -> bool; 3] = [
        |v| v.placement == Placement::Ajtai,
        |v| v.placement == Placement::Bdlop && v.norm != Norm::Unbounded,
        |v| v.norm == Norm::Unbounded,
    ];
    // Each variable's committed components in the order (polynomial, component): all k, or for
    // a subring of degree d_v the d_v/d components c = 0 mod d'/d_v, the others being 0; for
    // LinfExact, each as its bits. `requirements` checked that d divides d_v.
    let mut layout = vec![Lowered::Zero; source.count * k];
    let (mut bounded, mut total) = (0, 0);
    for (part, belongs) in parts.into_iter().enumerate() {
        for var in source.variables.iter().filter(|v| belongs(v)) {
            let stride = source.ring.degree() / var.degree;
            let own = &mut layout[var.start * k..(var.start + var.count) * k];
            // Entry i is component i mod k of its polynomial.
            for (i, entry) in own.iter_mut().enumerate() {
                if !(i % k).is_multiple_of(stride) {
                    continue;
                }
                *entry = match var.norm {
                    Norm::LinfExact(beta) => {
                        total += bit_count(beta);
                        Lowered::Bits {
                            first: total - bit_count(beta),
                            beta,
                        }
                    }
                    _ => {
                        total += 1;
                        Lowered::Committed(total - 1)
                    }
                };
            }
        }
        if part == 0 {
            bounded = total;
        }
    }
    // Every constraint not at the proof modulus is lifted: (F_j, p_j) for the check below.
    // The carries of lifted constant-coefficient clauses share packed polynomials, d slots
    // each, placed where the first clause's carry is: (first polynomial, count).
    let clauses = source
        .constraints
        .iter()
        .filter(|c| c.constant_only && c.ring().modulus() != ring.modulus())
        .count();
    let mut packed = None;
    let mut slot = 0;
    let mut carries = Vec::new();
    let mut lifted = Vec::new();
    let mut max_v = 0u128;
    for (i, constraint) in source.constraints.iter().enumerate() {
        let modulus = constraint.ring().modulus();
        if modulus == ring.modulus() {
            continue;
        }
        let f = form_bound(&source, &constraint.form)?;
        max_v = max_v.max(quotient_bound(&f.honest, constraint.ring())?);
        lifted.push((f, modulus));
        if constraint.constant_only {
            let (first, _) = *packed.get_or_insert_with(|| {
                let count = clauses.div_ceil(params.degree);
                total += count;
                (total - count, count)
            });
            carries.push(Carry::Slot {
                constraint: i,
                poly: first + slot / params.degree,
                coefficient: slot % params.degree,
            });
            slot += 1;
        } else if constraint.commits_carries() {
            carries.push(Carry::Components {
                constraint: i,
                start: total,
                count: k,
            });
            total += k;
        }
    }
    if params.m1 != bounded || params.l != total - bounded {
        return Err(Error::Parameter("compiled witness dimensions"));
    }
    let dim = 2 * total;
    // Without subring or LinfExact variables every lowered variable is committed, and the
    // equations are remapped as before; otherwise the committed ones, the zeros and the bit
    // forms are substituted.
    let mapped: Option<Vec<usize>> = layout
        .iter()
        .map(|x| match x {
            Lowered::Committed(i) => Some(2 * i),
            _ => None,
        })
        .collect();
    let affine = match mapped {
        Some(_) => Vec::new(),
        None => layout
            .iter()
            .map(|x| x.affine(&ring))
            .collect::<Result<_, _>>()?,
    };
    let mut statement = lnp::Statement::default();
    let mut binary = Vec::new();
    for var in &source.variables {
        let indices = committed(&layout, var, k);
        match var.norm {
            Norm::L2Squared(bound_squared) => statement.l2.push(L2Block {
                map: selector(&ring, &indices, bounded, params.l)?,
                bound_squared,
            }),
            // A LinfExact variable's bits, in its place among the binary blocks.
            Norm::Binary | Norm::LinfExact(_) => binary.extend(indices),
            Norm::Linf(_) | Norm::Unbounded => {}
        }
    }
    if !binary.is_empty() {
        statement.binary = Some(selector(&ring, &binary, bounded, params.l)?);
    }
    let mut arp = Vec::new();
    // p^-1 modulo q, the factor of implicit quotients, for the statement modulus and every
    // constraint modulus other than q. The statement modulus comes first, as before
    // per-constraint moduli, when it was the only one.
    let mut inverses = BTreeMap::new();
    let moduli = std::iter::once((source.ring.modulus(), "noninvertible statement modulus")).chain(
        source
            .constraints
            .iter()
            .map(|c| (c.ring().modulus(), "noninvertible constraint modulus")),
    );
    for (modulus, refusal) in moduli {
        if modulus != ring.modulus() && !inverses.contains_key(&modulus) {
            let inverse = Option::<U256>::from(ring.reduce(&modulus).invert_mod(&ring.nonzero))
                .ok_or(Error::Parameter(refusal))?;
            inverses.insert(modulus, inverse);
        }
    }
    let one = Poly::constant(ring.clone(), 1);
    for (i, constraint) in source.constraints.iter().enumerate() {
        let modulus = constraint.ring().modulus();
        // -p_j in the proof ring, the coefficient of each committed carry.
        let minus_p = Poly::constant_u256(ring.clone(), &ring.neg(&ring.reduce(&modulus)));
        let lowered = lower(&constraint.form, ring.clone())?;
        let carry = carries.iter().find(|c| c.constraint() == i);
        for (component, equation) in lowered
            .into_iter()
            .enumerate()
            .take(if constraint.constant_only { 1 } else { k })
        {
            let mut equation = match &mapped {
                Some(mapped) => equation.remap(mapped, dim)?,
                None => equation.substitute(&affine, dim)?,
            };
            match carry {
                Some(Carry::Components { start, .. }) => {
                    let index = start + component;
                    equation = equation.add(&term(&ring, dim, index, minus_p.clone())?)?;
                    arp.push(term(&ring, dim, index, one.clone())?);
                }
                // ct(X^-t C) = C_t in Z_q[X]/(X^d + 1): for u in [0, d), X^(u - t) has exponent
                // 0 only at u = t, and X^(u - t) = -X^(d + u - t) otherwise has one in (0, d).
                Some(Carry::Slot {
                    poly, coefficient, ..
                }) => {
                    let read = minus_p.rotate(-(*coefficient as i64));
                    equation = equation.add(&term(&ring, dim, *poly, read)?)?;
                    // Every packed polynomial's range row, at the first clause.
                    if let Some((first, count)) = packed
                        && (*poly, *coefficient) == (first, 0)
                    {
                        for index in first..first + count {
                            arp.push(term(&ring, dim, index, one.clone())?);
                        }
                    }
                }
                None => {
                    if let Some(inverse) = inverses.get(&modulus) {
                        arp.push(equation.scale(&Poly::constant_u256(ring.clone(), inverse))?);
                        continue;
                    }
                }
            }
            if constraint.constant_only {
                statement.evaluation.push(equation);
            } else {
                statement.quadratic.push(equation);
            }
        }
    }
    // Each coefficient of an ℓ∞ variable is a row of the approximate range proof, after the
    // carries and quotients: a selector, as any affine image of the witness can be.
    let extraction = checked.approx_extraction_bound;
    for var in &source.variables {
        if let Norm::Linf(beta) = var.norm {
            if u128::from(beta) > u128::from(params.linf_bound) {
                return Err(Error::Parameter("approximate range bound"));
            }
            for index in committed(&layout, var, k) {
                arp.push(term(&ring, dim, index, one.clone())?);
            }
        }
    }
    if let Some(linf) = &requirements.linf {
        // The range condition, with the extraction bound that bounds an extracted ℓ∞ variable.
        if linf
            .extraction_limit
            .is_some_and(|limit| U256::from_u128(extraction) > limit)
        {
            return Err(Error::Parameter("variable range above statement modulus"));
        }
        // Otherwise the extracted bound would say nothing modulo q.
        if U256::from_u128(extraction)
            .shl_vartime(1)
            .wrapping_add(&U256::ONE)
            >= ring.modulus()
        {
            return Err(Error::Parameter("approximate extraction bound"));
        }
    }
    if !lifted.is_empty() {
        if max_v > params.linf_bound as u128 {
            return Err(Error::Parameter("carry or quotient bound"));
        }
        // Extraction bounds every quotient, carry and ℓ∞ coefficient by E =
        // approx_extraction_bound = 2*z4_bound <= 32*sigma4 (LNP22 Lemma 2.7), so
        // |f_j| <= F_j(E), and f_j - p_j*carry is a multiple of q, so constraint j holds over the
        // integers once q > F_j(E) + p_j*E, whatever the other constraints' moduli. The check
        // below uses the paper's constant 28*sigma4, which belongs to its 14*sigma4 verifier
        // bound, but its factor 2 makes it stricter than that exact condition:
        // 2*(F_j + 28*sigma4*p_j) >= F_j + 56*sigma4*p_j. Round psi up before the exact check.
        let psi = (28.0 * 1.55 * (2.0f64).powi(params.log_sigma[3] as i32)
            / params.linf_bound as f64)
            .ceil();
        if !psi.is_finite() || psi > u64::MAX as f64 {
            return Err(Error::Parameter("lifting slack"));
        }
        // Exact in 1024 bits: p_j < 2^256, linf_bound and psi < 2^64.
        let slack = mul_bound(
            Bound::from_u64(params.linf_bound),
            Bound::from_u64(psi as u64),
        )?;
        for (f, modulus) in &lifted {
            let rhs = mul_bound(modulus.resize::<{ Bound::LIMBS }>(), slack)?;
            let rhs = mul_bound(add_bound(f.extracted(extraction)?, rhs)?, Bound::from_u8(2))?;
            if ring.modulus().resize::<{ Bound::LIMBS }>() <= rhs {
                return Err(Error::Parameter("modulus lifting bound"));
            }
        }
    }
    if !arp.is_empty() {
        statement.arp = Some(affine_rows(&ring, &arp, bounded, params.l)?);
    }
    let scheme = Abdlop::new([0; 32], params.clone())?;
    statement.check(&scheme)?;
    Ok(Compiled {
        source,
        params,
        statement,
        ring,
        k,
        layout,
        bounded,
        carries,
        extraction,
    })
}

/// Checked integer arithmetic for the lifted evaluation: `i128`, or 512 bits when a value does
/// not fit.
trait Integer: Copy + Default + Zeroize {
    fn centred(p: &Poly) -> Option<Zeroizing<Vec<Self>>>;
    fn add(self, rhs: Self) -> Option<Self>;
    fn mul(self, rhs: Self) -> Option<Self>;
    fn neg(self) -> Option<Self>;
}
impl Integer for i128 {
    fn centred(p: &Poly) -> Option<Zeroizing<Vec<Self>>> {
        p.coefficients_i128().ok()
    }
    fn add(self, rhs: Self) -> Option<Self> {
        self.checked_add(rhs)
    }
    fn mul(self, rhs: Self) -> Option<Self> {
        self.checked_mul(rhs)
    }
    fn neg(self) -> Option<Self> {
        self.checked_neg()
    }
}
/// A 512-bit integer; every centred 256-bit coefficient and every product of two fits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Wide(I512);
impl Zeroize for Wide {
    fn zeroize(&mut self) {
        self.0 = I512::ZERO;
    }
}
impl Integer for Wide {
    fn centred(p: &Poly) -> Option<Zeroizing<Vec<Self>>> {
        let ring = p.ring();
        Some(Zeroizing::new(
            p.coefficients()
                .iter()
                .map(|x| Wide(ring.centred_i256(x).resize::<{ I512::LIMBS }>()))
                .collect(),
        ))
    }
    fn add(self, rhs: Self) -> Option<Self> {
        Option::from(self.0.checked_add(&rhs.0)).map(Wide)
    }
    fn mul(self, rhs: Self) -> Option<Self> {
        Option::from(self.0.checked_mul(&rhs.0)).map(Wide)
    }
    fn neg(self) -> Option<Self> {
        Option::from(self.0.checked_neg()).map(Wide)
    }
}
fn integer_product<T: Integer>(a: &[T], b: &[T]) -> Result<Zeroizing<Vec<T>>, Error> {
    if a.len() != b.len() {
        return Err(Error::Dimension);
    }
    let d = a.len();
    let mut out = Zeroizing::new(vec![T::default(); d]);
    for (i, a) in a.iter().enumerate() {
        for (j, b) in b.iter().enumerate() {
            let mut value = a.mul(*b).ok_or(Error::Overflow)?;
            if i + j >= d {
                value = value.neg().ok_or(Error::Overflow)?;
            }
            out[(i + j) % d] = out[(i + j) % d].add(value).ok_or(Error::Overflow)?;
        }
    }
    Ok(out)
}
fn integer_evaluate_as<T: Integer>(
    form: &QuadEq,
    witness: &[Poly],
) -> Result<Zeroizing<Vec<T>>, Error> {
    let centred = |p: &Poly| T::centred(p).ok_or(Error::Overflow);
    let mut out = centred(&form.r0)?;
    for (i, a) in form.r1.entries() {
        let v = integer_product(&centred(a)?, &centred(&witness[usize::from(i)])?)?;
        for (o, v) in out.iter_mut().zip(v.iter()) {
            *o = o.add(*v).ok_or(Error::Overflow)?;
        }
    }
    for ((i, j), a) in form.r2.entries() {
        let v = integer_product(
            &centred(&witness[usize::from(i)])?,
            &centred(&witness[usize::from(j)])?,
        )?;
        let v = integer_product(&centred(a)?, &v)?;
        for (o, v) in out.iter_mut().zip(v.iter()) {
            *o = o.add(*v).ok_or(Error::Overflow)?;
        }
    }
    Ok(out)
}
/// The integer evaluation of a form on centred representatives: in `i128` when every step fits,
/// and otherwise again in 512 bits. Both give the same integers when both succeed.
fn integer_evaluate(form: &QuadEq, witness: &[Poly]) -> Result<Zeroizing<Vec<Wide>>, Error> {
    match integer_evaluate_as::<i128>(form, witness) {
        Ok(values) => Ok(Zeroizing::new(
            values.iter().map(|x| Wide(I512::from_i128(*x))).collect(),
        )),
        Err(_) => integer_evaluate_as::<Wide>(form, witness),
    }
}
/// Whether $`p`$ divides the value.
fn divides(p: &NonZero<U256>, value: &Wide) -> bool {
    value.0.rem_unsigned(p) == I256::ZERO
}
/// The exact quotient by $`p`$, which must fit `i128` (carries are bounded by linf_bound).
fn carry(p: &NonZero<U256>, value: &Wide) -> Result<i128, Error> {
    let (magnitude, negative) = value.0.div_unsigned(p).abs_sign();
    if magnitude.bits_vartime() > 127 {
        return Err(Error::Overflow);
    }
    let magnitude = int::low_u128(&magnitude) as i128;
    Ok(if bool::from(negative) {
        -magnitude
    } else {
        magnitude
    })
}
use zeroize::{Zeroize, Zeroizing};
impl Compiled {
    /// The checked parameters used by this compilation.
    pub fn parameters(&self) -> &TboxParams {
        &self.params
    }
    /// Prove and encode the full toolbox proof under an application `context`.
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
        pp_seed: [u8; 32],
        witness: &BTreeMap<String, Vec<Poly>>,
        context: &[u8],
        mut seed: [u8; 32],
    ) -> Result<Vec<u8>, Error> {
        let (scheme, proof) =
            self.prove_seeded(pp_seed, witness, context, &take_seed(&mut seed))?;
        crate::codec::proof::encode(&scheme, &proof)
    }
    /// The scheme of `pp_seed` and a proof under it, from a seed held in a wiping buffer.
    fn prove_seeded(
        &self,
        pp_seed: [u8; 32],
        witness: &BTreeMap<String, Vec<Poly>>,
        context: &[u8],
        seed: &Zeroizing<[u8; 32]>,
    ) -> Result<(Abdlop, tbox::Proof), Error> {
        let (s1, m) = self.map_witness(witness)?;
        let scheme = Abdlop::new(pp_seed, self.params.clone())?;
        let proof = tbox::prove_seeded(&scheme, &self.statement, &s1, &m, context, seed)?;
        Ok((scheme, proof))
    }
    /// Strictly decode and verify a proof against the compiled statement and `context`.
    pub fn verify_bytes(
        &self,
        pp_seed: [u8; 32],
        bytes: &[u8],
        context: &[u8],
    ) -> Result<(), Error> {
        let scheme = Abdlop::new(pp_seed, self.params.clone())?;
        tbox::verify(
            &scheme,
            &self.statement,
            &crate::codec::proof::decode(&scheme, bytes)?,
            context,
        )
    }
    /// Compiled public toolbox relation, useful for inspecting dimensions and selectors.
    pub fn statement(&self) -> &lnp::Statement {
        &self.statement
    }
    /// The bound that a verified proof guarantees for the named variable of an extracted
    /// witness ([witness relation](Statement)): the declared bound for exact-norm, binary and
    /// [`Norm::LinfExact`] variables, and for a [`Norm::Linf`] variable the approximate range
    /// proof's `approx_extraction_bound` $`=2\cdot`$`z4_bound`, not its $`\beta`$. Fails with
    /// [`Error::Index`] for an unknown name.
    pub fn extraction_bound(&self, name: &str) -> Result<Extraction, Error> {
        let var = self
            .source
            .variables
            .iter()
            .find(|v| v.name == name)
            .ok_or(Error::Index)?;
        Ok(match var.norm {
            Norm::L2Squared(b) => Extraction::L2Squared(b),
            Norm::Binary => Extraction::Binary,
            Norm::Linf(_) => Extraction::Linf(self.extraction),
            Norm::LinfExact(beta) => Extraction::LinfExact(beta),
            Norm::Unbounded => Extraction::Unbounded,
        })
    }
    /// The bit polynomials of component `component` of a [`Norm::LinfExact`] polynomial `p`,
    /// whose coefficients the witness map checked to be at most $`\beta`$: bit $`b`$ of the
    /// greedy digits of $`v+\beta`$ for each coefficient $`v`$ of the component, from the exact
    /// integer (the proof modulus may be below $`2\beta`$).
    fn bits(&self, p: &Poly, component: usize, beta: u64) -> Result<Vec<Poly>, Error> {
        let values = p.coefficients_i128()?;
        let weights = weights(beta);
        let d = self.ring.degree();
        let mut bits = Zeroizing::new(vec![vec![0i128; d]; weights.len()]);
        let mut digit = Zeroizing::new(vec![0u64; weights.len()]);
        for u in 0..d {
            // Coefficient k u + c of the statement polynomial is coefficient u of component c.
            let x = (values[self.k * u + component] + i128::from(beta)) as u64;
            digits(x, &weights, &mut digit);
            for (bit, value) in bits.iter_mut().zip(digit.iter()) {
                bit[u] = i128::from(*value);
            }
        }
        bits.iter_mut()
            .map(|bit| Poly::new(self.ring.clone(), std::mem::take(bit)))
            .collect()
    }
    /// Map named statement-ring witness blocks into the proof-ring partition and carries.
    ///
    /// Each block is checked against its norm and, for a subring block, for zeros outside the
    /// subring. Each constraint is checked modulo its own modulus on the integers that the
    /// centred coefficients stand for (see [`Statement`]), and each carry is the exact quotient
    /// of the integer value by that modulus. A [`Norm::LinfExact`] coefficient $`v`$ is written
    /// as the greedy bits of $`v+\beta\in[0,2\beta]`$.
    pub fn map_witness(
        &self,
        witness: &BTreeMap<String, Vec<Poly>>,
    ) -> Result<(PolyVec, PolyVec), Error> {
        if witness.len() != self.source.variables.len() {
            return Err(Error::Dimension);
        }
        let mut original = Vec::new();
        for var in &self.source.variables {
            let block = witness.get(&var.name).ok_or(Error::Witness)?;
            if block.len() != var.count || block.iter().any(|p| p.ring() != &self.source.ring) {
                return Err(Error::Dimension);
            }
            let block_vec = PolyVec::new(self.source.ring.clone(), block.clone())?;
            match var.norm {
                Norm::L2Squared(b) if block_vec.norm_squared()? > U256::from(b) => {
                    return Err(Error::Witness);
                }
                Norm::Binary
                    if block
                        .iter()
                        .any(|p| p.coefficients().iter().any(|x| *x > U256::ONE)) =>
                {
                    return Err(Error::Witness);
                }
                // The range proof checks only the common linf_bound: beta is checked here. For
                // LinfExact, the bits could not encode more.
                Norm::Linf(beta) | Norm::LinfExact(beta)
                    if block.iter().any(|p| {
                        p.coefficients()
                            .iter()
                            .any(|x| self.source.ring.magnitude(x) > U256::from_u64(beta))
                    }) =>
                {
                    return Err(Error::Witness);
                }
                _ => {}
            }
            // A subring element v(X^K) has zeros at every index not divisible by K.
            let stride = self.source.ring.degree() / var.degree;
            if block.iter().any(|p| {
                p.coefficients()
                    .iter()
                    .enumerate()
                    .any(|(i, x)| i % stride != 0 && !x.is_zero_vartime())
            }) {
                return Err(Error::Witness);
            }
            original.extend(block.iter().cloned());
        }
        let mut mapped = vec![Poly::zero(self.ring.clone()); self.params.m1 + self.params.l];
        let lifted_ring = Ring::with_modulus(self.ring.modulus(), self.source.ring.degree())?;
        for (i, p) in original.iter().enumerate() {
            for (component, value) in iso::split(&lift(p, lifted_ring.clone())?, self.ring.clone())?
                .into_iter()
                .enumerate()
            {
                match self.layout[i * self.k + component] {
                    Lowered::Committed(index) => mapped[index] = value,
                    Lowered::Zero => {}
                    Lowered::Bits { first, beta } => {
                        let bits = self.bits(p, component, beta)?;
                        mapped[first..first + bits.len()].clone_from_slice(&bits);
                    }
                }
            }
        }
        let original_vec = PolyVec::new(self.source.ring.clone(), original.clone())?;
        // The witness over each other constraint ring: its integers reduced modulo p_j.
        let mut reduced = BTreeMap::<U256, PolyVec>::new();
        for constraint in &self.source.constraints {
            let ring = constraint.ring();
            let witness = if ring == &self.source.ring {
                &original_vec
            } else {
                match reduced.entry(ring.modulus()) {
                    Entry::Occupied(entry) => &*entry.into_mut(),
                    Entry::Vacant(entry) => &*entry.insert(PolyVec::new(
                        ring.clone(),
                        original
                            .iter()
                            .map(|p| lift(p, ring.clone()))
                            .collect::<Result<_, _>>()?,
                    )?),
                }
            };
            let value = constraint.form.evaluate(witness)?;
            if if constraint.constant_only {
                !value.coefficients()[0].is_zero_vartime()
            } else {
                !value.is_zero()
            } {
                return Err(Error::Witness);
            }
        }
        for committed in &self.carries {
            let constraint = &self.source.constraints[committed.constraint()];
            let p = &constraint.ring().nonzero;
            let value = integer_evaluate(&constraint.form, &original)?;
            match *committed {
                // The constant coefficient's quotient, in its slot; unused slots stay 0.
                Carry::Slot {
                    poly, coefficient, ..
                } => {
                    if !divides(p, &value[0]) {
                        return Err(Error::Witness);
                    }
                    mapped[poly].set_coefficient(coefficient, carry(p, &value[0])?)?;
                }
                Carry::Components { start, count, .. } => {
                    if value.iter().any(|x| !divides(p, x)) {
                        return Err(Error::Witness);
                    }
                    let carry = Poly::new(
                        lifted_ring.clone(),
                        value
                            .iter()
                            .map(|x| carry(p, x))
                            .collect::<Result<_, _>>()?,
                    )?;
                    let parts = iso::split(&carry, self.ring.clone())?;
                    if parts.len() != count {
                        return Err(Error::Dimension);
                    }
                    mapped[start..start + count].clone_from_slice(&parts);
                }
            }
        }
        Ok((
            PolyVec::new(self.ring.clone(), mapped[..self.bounded].to_vec())?,
            PolyVec::new(self.ring.clone(), mapped[self.bounded..].to_vec())?,
        ))
    }
    /// Prove the compiled statement using named witness blocks, under an application `context`.
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
        pp_seed: [u8; 32],
        witness: &BTreeMap<String, Vec<Poly>>,
        context: &[u8],
        mut seed: [u8; 32],
    ) -> Result<tbox::Proof, Error> {
        Ok(self
            .prove_seeded(pp_seed, witness, context, &take_seed(&mut seed))?
            .1)
    }
    /// Prove with fresh randomness from a caller-supplied cryptographic RNG, under `context`.
    /// Each call draws a fresh seed.
    pub fn prove(
        &self,
        pp_seed: [u8; 32],
        witness: &BTreeMap<String, Vec<Poly>>,
        context: &[u8],
        rng: &mut impl rand_core::CryptoRng,
    ) -> Result<tbox::Proof, Error> {
        let mut seed = Zeroizing::new([0; 32]);
        rand_core::Rng::fill_bytes(rng, &mut *seed);
        Ok(self.prove_seeded(pp_seed, witness, context, &seed)?.1)
    }
    /// Prove and encode with fresh randomness from a cryptographic RNG, under `context`. Each
    /// call draws a fresh seed.
    pub fn prove_bytes(
        &self,
        pp_seed: [u8; 32],
        witness: &BTreeMap<String, Vec<Poly>>,
        context: &[u8],
        rng: &mut impl rand_core::CryptoRng,
    ) -> Result<Vec<u8>, Error> {
        let mut seed = Zeroizing::new([0; 32]);
        rand_core::Rng::fill_bytes(rng, &mut *seed);
        let (scheme, proof) = self.prove_seeded(pp_seed, witness, context, &seed)?;
        crate::codec::proof::encode(&scheme, &proof)
    }
    /// Verify against the compiled public statement and the application `context`.
    pub fn verify(
        &self,
        pp_seed: [u8; 32],
        proof: &tbox::Proof,
        context: &[u8],
    ) -> Result<(), Error> {
        tbox::verify(
            &Abdlop::new(pp_seed, self.params.clone())?,
            &self.statement,
            proof,
            context,
        )
    }
}
