//! Packed ternary entries, four to a byte: entry $`e`$ of a byte is bit $`2e`$ less bit
//! $`2e+1`$, the centred binomial decoding with $`\eta=1`$ (`SamplePolyCBD` of FIPS 203). The
//! toolbox's projection matrices ([`crate::transcript::ternary_entry`]) and LaBRADOR's
//! projection columns are packed this way.

/// The four entries in $`\{-1,0,1\}`$ of each byte.
pub(crate) const DECODE: [[i8; 4]; 256] = {
    let mut out = [[0i8; 4]; 256];
    let mut b = 0;
    while b < 256 {
        let mut e = 0;
        while e < 4 {
            out[b][e] = ((b >> (2 * e)) & 1) as i8 - ((b >> (2 * e + 1)) & 1) as i8;
            e += 1;
        }
        b += 1;
    }
    out
};

/// The index of each byte among the 81 sign patterns of four entries,
/// $`\sum_e3^e(x_e+1)`$ for its entries $`x_e`$ ([`DECODE`]): bytes with the same four entries
/// share their index.
pub(crate) const PATTERN: [u8; 256] = {
    let mut out = [0u8; 256];
    let mut b = 0;
    while b < 256 {
        let mut index = 0;
        let mut e = 4;
        while e > 0 {
            e -= 1;
            index = 3 * index + (DECODE[b][e] + 1) as u8;
        }
        out[b] = index;
        b += 1;
    }
    out
};

/// The four entries of sign pattern `index` ([`PATTERN`]).
pub(crate) const fn pattern_signs(index: usize) -> [i8; 4] {
    let mut out = [0i8; 4];
    let (mut rest, mut e) = (index, 0);
    while e < 4 {
        out[e] = (rest % 3) as i8 - 1;
        rest /= 3;
        e += 1;
    }
    out
}
