//! The first nine primes of the search in `tools/params/moduli.py`; roots have order 2048.

/// Descending primes below $`2^{62}`$ and primitive 2048-th roots.
///
/// Nine primes cover every ring: their product exceeds $`2^{557}`$, and a ring needs
/// $`P>(q-1)^2\cdot d\cdot128`$, which is below $`2^{529}`$ for $`q<2^{256}`$ and
/// $`d\le1024`$. Rings take the primes in this order and stop once the bound holds, so a ring
/// that fitted the first four primes selects the same primes as before.
pub const NTT_PRIMES: [(u64, u64); 9] = [
    (4611686018427365377, 1482597879546526807),
    (4611686018427322369, 2953159431647451165),
    (4611686018427289601, 4233275892050583047),
    (4611686018427277313, 2797043744752489505),
    (4611686018427246593, 3639146724924666250),
    (4611686018427228161, 3997053597298999433),
    (4611686018427215873, 397491170059803391),
    (4611686018427199489, 1728412936777906485),
    (4611686018427185153, 3383440969586232429),
];

/// Maximum number of products before exact CRT reconstruction and reduction.
pub const NADDS: usize = 128;
