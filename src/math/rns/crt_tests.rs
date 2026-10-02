//! The CRT: `SmallCrt` (moduli below $`2^{63}`$) and `WideCrt` (from
//! $`2^{63}`$ on) against a num-bigint CRT at $`0`$, $`1`$, $`\lfloor P/2\rfloor`$ and its
//! neighbours, $`P-1`$ and random values, for one to nine primes and moduli from 2 to
//! $`2^{256}-1`$; against the multi-limb reference path on rings with every number of
//! primes; and the native residues against a reference formula.
use super::*;
use crate::{
    math::{
        int::Divisor64,
        ntt::NttPlan,
        ring::{SmallCrt, WideCrt},
    },
    params::moduli::NTT_PRIMES,
};
use crypto_bigint::NonZero;
use num_bigint::BigUint;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: &BigUint) -> BigUint {
        let words: Vec<u64> = (0..10).map(|_| self.next()).collect();
        BigUint::from_slice(
            &words
                .iter()
                .flat_map(|w| [*w as u32, (*w >> 32) as u32])
                .collect::<Vec<_>>(),
        ) % n
    }
}

const SMALL_MODULI: [u64; 6] = [2, 3, 12, 13, (1 << 62) - 1, (1 << 63) - 1];

/// The integers of the test plan below $`P`$: the ends, $`\lfloor P/2\rfloor`$ with its
/// neighbours, and random ones.
fn integers(product: &BigUint, rng: &mut Rng) -> Vec<BigUint> {
    let half = product >> 1u32;
    let one = BigUint::from(1u32);
    let mut out = vec![
        BigUint::from(0u32),
        one.clone(),
        &half - &one,
        half.clone(),
        &half + &one,
        product - 2u32,
        product - &one,
    ];
    out.extend((0..40).map(|_| rng.below(product)));
    out
}

/// The canonical value modulo `q` of the centred integer: $`X`$ up to $`\lfloor P/2\rfloor`$,
/// $`X-P`$ above.
fn centred_mod_big(x: &BigUint, product: &BigUint, q: &BigUint) -> BigUint {
    if x > &(product >> 1u32) {
        (q - ((product - x) % q)) % q
    } else {
        x % q
    }
}
fn centred_mod(x: &BigUint, product: &BigUint, q: u64) -> u64 {
    let r = centred_mod_big(x, product, &BigUint::from(q));
    r.to_u64_digits().first().copied().unwrap_or(0)
}
fn big(x: &U256) -> BigUint {
    BigUint::from_bytes_le(&x.to_le_bytes())
}
fn residues_of(xs: &[BigUint], primes: &[u64]) -> Vec<Vec<u64>> {
    primes
        .iter()
        .map(|p| {
            xs.iter()
                .map(|x| (x % p).to_u64_digits().first().copied().unwrap_or(0))
                .collect()
        })
        .collect()
}

#[test]
fn native_crt_equals_a_big_integer_crt_for_one_to_nine_primes() {
    let mut rng = Rng(8);
    for k in 1..=9 {
        let primes: Vec<u64> = NTT_PRIMES[..k].iter().map(|(p, _)| *p).collect();
        let product = primes
            .iter()
            .map(|p| BigUint::from(*p))
            .product::<BigUint>();
        for q in SMALL_MODULI {
            let crt = SmallCrt::new(q, &primes);
            let xs = integers(&product, &mut rng);
            let residues = residues_of(&xs, &primes);
            for (j, x) in xs.iter().enumerate() {
                assert_eq!(
                    crt.reconstruct(&residues, j),
                    centred_mod(x, &product, q),
                    "{k} primes, q {q}, X {x}"
                );
            }
        }
    }
}

/// Moduli from $`2^{63}`$ on: both sides of $`2^{64}`$ and $`2^{128}`$, primes of the wide
/// tests, rings of every number of primes from three to nine, and the largest values.
fn wide_moduli() -> Vec<U256> {
    let two = |k: u32| U256::ONE.shl_vartime(k);
    vec![
        two(63),
        two(63).wrapping_add(&U256::ONE),
        two(64).wrapping_sub(&U256::from_u8(59)),
        two(64),
        two(100).wrapping_sub(&U256::from_u8(15)),
        two(128).wrapping_sub(&U256::ONE),
        two(128).wrapping_add(&U256::from_u8(165)),
        two(160).wrapping_add(&U256::from_u8(7)),
        two(200).wrapping_sub(&U256::from_u8(75)),
        two(240).wrapping_add(&U256::from_u16(325)),
        U256::MAX.wrapping_sub(&U256::from_u16(434)),
        U256::MAX,
    ]
}

#[test]
fn wide_crt_equals_a_big_integer_crt_for_one_to_nine_primes() {
    let mut rng = Rng(11);
    for k in 1..=9 {
        let primes: Vec<u64> = NTT_PRIMES[..k].iter().map(|(p, _)| *p).collect();
        let product = primes
            .iter()
            .map(|p| BigUint::from(*p))
            .product::<BigUint>();
        for q in wide_moduli() {
            let nonzero = NonZero::new(q).unwrap();
            let small = int::to_u64(&q).map(Divisor64::new);
            let crt = WideCrt::new(&nonzero, &primes);
            let xs = integers(&product, &mut rng);
            let residues = residues_of(&xs, &primes);
            for (j, x) in xs.iter().enumerate() {
                assert_eq!(
                    big(&crt.reconstruct(&residues, j, &nonzero, small.as_ref())),
                    centred_mod_big(x, &product, &big(&q)),
                    "{k} primes, q {q}, X {x}"
                );
            }
        }
    }
}

/// Moduli below $`2^{63}`$ whose rings select one, two and three primes at degree 64 and 1024.
fn small_rings() -> Vec<Arc<Ring>> {
    let mut out = Vec::new();
    for q in SMALL_MODULI
        .into_iter()
        .chain([1099511627917, 1 << 40, (1 << 45) + 7])
    {
        for d in [64, 1024] {
            out.push(Ring::with_modulus(U256::from_u64(q), d).unwrap());
        }
    }
    let primes: Vec<usize> = out.iter().map(|r| r.plans.len()).collect();
    for k in 1..=3 {
        assert!(primes.contains(&k), "no ring with {k} primes");
    }
    out
}

#[test]
fn every_crt_equals_the_multi_limb_path_on_rings_with_one_to_nine_primes() {
    let mut rng = Rng(9);
    let mut rings = small_rings();
    for q in wide_moduli() {
        for d in [64, 1024] {
            rings.push(Ring::with_modulus(q, d).unwrap());
        }
    }
    let primes: Vec<usize> = rings.iter().map(|r| r.plans.len()).collect();
    for k in 1..=9 {
        assert!(primes.contains(&k), "no ring with {k} primes");
    }
    for ring in rings {
        let primes: Vec<u64> = ring.plans.iter().map(NttPlan::modulus).collect();
        let product = primes
            .iter()
            .map(|p| BigUint::from(*p))
            .product::<BigUint>();
        // d values per prime: the boundary integers first, then random residues.
        let mut residues: Vec<Vec<u64>> = primes
            .iter()
            .map(|p| (0..ring.d).map(|_| rng.next() % p).collect())
            .collect();
        for (j, x) in integers(&product, &mut rng).iter().take(7).enumerate() {
            for (i, p) in primes.iter().enumerate() {
                residues[i][j] = (x % p).to_u64_digits().first().copied().unwrap_or(0);
            }
        }
        let want = match &ring.multi_limb {
            Crt::Four(basis) => reconstruct_with(&ring, basis, &residues),
            Crt::Six(basis) => reconstruct_with(&ring, basis, &residues),
            Crt::Nine(basis) => reconstruct_with(&ring, basis, &residues),
        };
        for (j, want) in want.iter().enumerate() {
            let got = match &ring.crt {
                RingCrt::Small(small) => U256::from_u64(small.reconstruct(&residues, j)),
                RingCrt::Wide(wide) => {
                    wide.reconstruct(&residues, j, &ring.nonzero, ring.small.as_ref())
                }
            };
            assert_eq!(got, *want, "q {}, d {}, coefficient {j}", ring.q, ring.d);
        }
    }
}

#[test]
fn native_residues_equal_the_reference_formula() {
    let mut rng = Rng(10);
    for ring in small_rings() {
        let q = int::low_u64(&ring.q);
        let mut values: Vec<U256> = [0, 1, q / 2, q / 2 + 1, q - 1]
            .into_iter()
            .filter(|x| *x < q)
            .map(U256::from_u64)
            .collect();
        values.extend((0..64).map(|_| U256::from_u64(rng.next() % q)));
        for (i, plan) in ring.plans.iter().enumerate() {
            let p = plan.modulus();
            let q_mod_p = ring.q_mod_p[i];
            // The residue of the centred value by the reference formula.
            let expected: Vec<u64> = values
                .iter()
                .map(|x| {
                    let r = ring.residue(x, i);
                    if !ring.is_negative(x) {
                        r
                    } else if r >= q_mod_p {
                        r - q_mod_p
                    } else {
                        r + (p - q_mod_p)
                    }
                })
                .collect();
            assert_eq!(
                centred_residues(&ring, &values, i),
                expected,
                "q {q}, prime {i}"
            );
        }
    }
}
