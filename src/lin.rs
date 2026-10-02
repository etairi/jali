//! Linear front end with degree lowering and implicit-carry modulus lifting.
use crate::{
    Error,
    math::{PolyMat, PolyVec},
    params::TboxParams,
    statement::{Compiled, Norm, Placement, Statement},
};

/// Named partition of consecutive witness polynomials.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    /// Unique witness-map name.
    pub name: String,
    /// Number of statement-ring polynomials in this block.
    pub length: usize,
    /// The block's norm: binary, exact Euclidean, approximate $`\ell_\infty`$ or exact
    /// $`\ell_\infty`$ through bits ([`Norm`]).
    pub norm: Norm,
}
/// Compile $`Aw+t=0`$ over the matrix's statement ring into the generic toolbox.
/// Polynomials store canonical coefficients; the compiler bounds and lifts their centred
/// representatives. Every bounded block goes to the Ajtai part of the commitment.
pub fn compile(
    a: &PolyMat,
    t: &PolyVec,
    blocks: &[Block],
    params: TboxParams,
) -> Result<Compiled, Error> {
    compile_placed(a, t, blocks, &[], params)
}
/// [`compile`] with the blocks named in `bdlop` in the BDLOP part of the commitment
/// ([`Placement::Bdlop`]), as a parameter search may choose; the other blocks keep the
/// placement of [`Statement::var`]. Fails with [`Error::Index`] for a name that is no block.
pub fn compile_placed(
    a: &PolyMat,
    t: &PolyVec,
    blocks: &[Block],
    bdlop: &[&str],
    params: TboxParams,
) -> Result<Compiled, Error> {
    if a.rows() != t.len()
        || a.ring() != t.ring()
        || blocks
            .iter()
            .try_fold(0usize, |n, b| n.checked_add(b.length))
            .ok_or(Error::Dimension)?
            != a.cols()
    {
        return Err(Error::Dimension);
    }
    if bdlop
        .iter()
        .any(|name| blocks.iter().all(|b| b.name != *name))
    {
        return Err(Error::Index);
    }
    let mut statement = Statement::new(a.ring().clone());
    for block in blocks {
        let placement = if bdlop.contains(&block.name.as_str()) {
            Placement::Bdlop
        } else {
            Placement::default_for(block.norm)
        };
        statement.var_placed(&block.name, block.length, block.norm, placement)?;
    }
    for row in 0..a.rows() {
        let mut equation = statement.constant(t.entries()[row].clone())?;
        let mut col = 0;
        for block in blocks {
            for j in 0..block.length {
                equation = equation.add(
                    &statement
                        .variable(&block.name, j)?
                        .scale(a.get(row, col)?)?,
                )?;
                col += 1;
            }
        }
        statement.eq_mod_p(equation)?;
    }
    statement.compile(params)
}
