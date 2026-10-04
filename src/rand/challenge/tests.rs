//! The exact test against a direct computation with num-bigint (all $`d`$ coefficients of
//! every product, no use of $`\sigma_{-1}`$-stability), its boundary and capacities, and the
//! resampling of [`challenge`].
use super::*;
use crate::rand::{AesPrg, domain};
use num_bigint::BigInt;
use num_traits::{Signed, Zero};

/// $`\|(\sigma_{-1}(c)c)^{32}\|_1`$ with full negacyclic products of big integers.
fn direct(c: &[i128]) -> BigInt {
    direct_stages(c).pop().unwrap()
}

/// The norms $`\|u^{2^{j-1}}\|_1`$, $`u=\sigma_{-1}(c)c`$, for $`j=1`$ to 6, the same way.
fn direct_stages(c: &[i128]) -> Vec<BigInt> {
    let d = c.len();
    let product = |a: &[BigInt], b: &[BigInt]| {
        let mut out = vec![BigInt::zero(); d];
        for (i, x) in a.iter().enumerate() {
            for (j, y) in b.iter().enumerate() {
                if i + j < d {
                    out[i + j] += x * y;
                } else {
                    out[i + j - d] -= x * y;
                }
            }
        }
        out
    };
    let c: Vec<BigInt> = c.iter().map(|x| BigInt::from(*x)).collect();
    let s: Vec<BigInt> = (0..d)
        .map(|i| if i == 0 { c[0].clone() } else { -&c[d - i] })
        .collect();
    let norm = |u: &[BigInt]| u.iter().map(|x| x.abs()).sum::<BigInt>();
    let mut u = product(&s, &c);
    let mut norms = vec![norm(&u)];
    for _ in 0..5 {
        u = product(&u, &u);
        norms.push(norm(&u));
    }
    norms
}

fn big(x: &U1024) -> BigInt {
    BigInt::from_bytes_le(num_bigint::Sign::Plus, &x.to_le_bytes())
}

fn poly(ring: &Arc<Ring>, c: &[i128]) -> Poly {
    Poly::new(ring.clone(), c.to_vec()).unwrap()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> i128 {
        i128::from(self.next() % n)
    }
}

#[test]
fn the_exact_norm_equals_the_direct_power_on_challenges_and_other_polynomials() {
    let mut rng = Rng(1);
    for (d, omega, count) in [(64, 8, 200), (128, 2, 200), (128, 8, 20)] {
        let ring = Ring::new(1099511627917, d).unwrap();
        // Challenges as the protocol draws them.
        for i in 0..count {
            let c = autostable(
                &mut AesPrg::new(&[5; 32], domain(d as u32, i)),
                ring.clone(),
                omega,
            )
            .unwrap();
            let coefficients = c.coefficients_i128().unwrap();
            assert_eq!(
                big(&eta_norm_power(&c).unwrap()),
                direct(&coefficients),
                "{d} {i}"
            );
        }
        // Polynomials that are not sigma-stable, with every coefficient in [-omega, omega]:
        // sigma_{-1}(c) c is stable all the same.
        for i in 0..count / 4 {
            let c: Vec<i128> = (0..d)
                .map(|_| rng.below(2 * omega as u64 + 1) - omega)
                .collect();
            if c.iter().map(|x| x.unsigned_abs()).sum::<u128>() > MAX_L1 {
                continue;
            }
            assert_eq!(
                big(&eta_norm_power(&poly(&ring, &c)).unwrap()),
                direct(&c),
                "{d} {i}"
            );
        }
    }
}

#[test]
fn extreme_challenges_and_the_capacity() {
    for d in [64usize, 128] {
        let ring = Ring::new(1099511627917, d).unwrap();
        let check = |c: &[i128]| {
            let norm = eta_norm_power(&poly(&ring, c)).unwrap();
            assert_eq!(big(&norm), direct(c));
            norm
        };
        // Zero, and X - X^(d-1) = X + X^(-1).
        assert_eq!(check(&vec![0; d]), U1024::ZERO);
        let mut c = vec![0; d];
        c[1] = 1;
        c[d - 1] = -1;
        check(&c);
        // Every coefficient at omega (and minus omega above d/2): far above eta.
        for (omega, eta) in [(8i128, 140u64), (2, 59)] {
            let mut c = vec![0; d];
            for i in 0..d / 2 {
                c[i] = omega;
                if i > 0 {
                    c[d - i] = -omega;
                }
            }
            if c.iter().map(|x| x.unsigned_abs()).sum::<u128>() <= MAX_L1 {
                check(&c);
                assert!(!within_eta(&poly(&ring, &c), eta).unwrap());
            }
        }
        // The l1 capacity: a constant 2^15 gives exactly 2^960; 2^15 + 1 is refused.
        let mut c = vec![0; d];
        c[0] = 1 << 15;
        assert_eq!(check(&c), U1024::ONE.shl_vartime(960));
        c[0] = -(1 << 15);
        assert_eq!(check(&c), U1024::ONE.shl_vartime(960));
        c[0] = (1 << 15) + 1;
        assert_eq!(
            eta_norm_power(&poly(&ring, &c)),
            Err(Error::Parameter("challenge norm capacity"))
        );
        c[0] = (1 << 15) - 1;
        c[d - 1] = 2;
        assert_eq!(
            within_eta(&poly(&ring, &c), 140),
            Err(Error::Parameter("challenge norm capacity"))
        );
        // Dense polynomials at the capacity, with every coefficient of one magnitude: all
        // positive, alternating, and sigma-stable with alternating signs.
        let a = (1i128 << 15) / d as i128;
        let dense: [Vec<i128>; 3] = [
            vec![a; d],
            (0..d).map(|i| if i % 2 == 0 { a } else { -a }).collect(),
            (0..d)
                .map(|i| match i {
                    0 => a,
                    i if i < d / 2 => a * if i % 2 == 0 { 1 } else { -1 },
                    i if i == d / 2 => 0,
                    i => -(a * if (d - i) % 2 == 0 { 1 } else { -1 }),
                })
                .collect(),
        ];
        for c in &dense {
            assert!(c.iter().map(|x| x.unsigned_abs()).sum::<u128>() <= MAX_L1);
            check(c);
        }
    }
}

#[test]
fn a_set_beyond_the_capacity_is_refused_before_any_read() {
    // ||c||_1 <= (d - 1) omega: (64, 520) and (128, 258) are within 2^15, (64, 521) and
    // (128, 259) are not, and are refused whatever the stream.
    for (d, omega, refused) in [
        (64, 520, false),
        (64, 521, true),
        (128, 258, false),
        (128, 259, true),
    ] {
        let ring = Ring::new(1099511627917, d).unwrap();
        let mut stream = AesPrg::new(&[4; 32], 0);
        let out = challenge(&mut stream, ring.clone(), omega, (1 << 16) - 1);
        let mut fresh = AesPrg::new(&[4; 32], 0);
        let (mut x, mut y) = ([0u8; 16], [0u8; 16]);
        stream.fill(&mut x).unwrap();
        fresh.fill(&mut y).unwrap();
        if refused {
            assert_eq!(
                out,
                Err(Error::Parameter("challenge norm capacity")),
                "{d} {omega}"
            );
            assert_eq!(x, y, "a refused set reads nothing");
        } else {
            let c = out.unwrap();
            assert!(within_eta(&c, (1 << 16) - 1).unwrap());
            assert_ne!(x, y);
        }
    }
}

#[test]
fn the_comparison_includes_its_boundary() {
    // A constant k: sigma_{-1}(k) k = k^2 and the norm is k^64, which eta = k meets and
    // eta = k - 1 does not.
    let ring = Ring::new(1099511627917, 64).unwrap();
    for k in [1i128, 2, 7, 8, 140, 1024] {
        let c = Poly::constant(ring.clone(), k);
        let eta = k as u64;
        assert_eq!(eta_norm_power(&c).unwrap(), eta_powers(eta).unwrap()[6]);
        assert!(within_eta(&c, eta).unwrap(), "{k}");
        assert!(!within_eta(&c, eta - 1).unwrap(), "{k}");
        let c = Poly::constant(ring.clone(), -k);
        assert!(within_eta(&c, eta).unwrap() && !within_eta(&c, eta - 1).unwrap());
    }
    // The zero polynomial is within every bound, eta = 0 included.
    assert!(within_eta(&Poly::zero(ring.clone()), 0).unwrap());
    // eta^64 for the largest eta, 2^16 - 1, still fits; 2^16 is refused.
    let largest = eta_powers((1 << 16) - 1).unwrap()[6];
    assert_eq!(
        big(&largest),
        BigInt::from((1u64 << 16) - 1).pow(64),
        "the largest power"
    );
    assert_eq!(
        within_eta(&Poly::zero(ring), 1 << 16),
        Err(Error::Parameter("challenge eta capacity"))
    );
}

/// The smallest eta that `c` meets: the ceiling of the 64th root of its norm power.
fn smallest_eta(c: &Poly) -> u64 {
    let norm = big(&eta_norm_power(c).unwrap());
    let root = norm.nth_root(64);
    let eta = if root.pow(64) == norm { root } else { root + 1 };
    u64::try_from(eta).unwrap()
}

#[test]
fn the_early_acceptance_decides_as_the_exact_test_at_every_boundary() {
    // For each polynomial, within_eta must accept at its smallest eta and refuse just below:
    // an early acceptance at eta - 1 or a missed one at eta fails here. Below its smallest
    // eta a polynomial runs all six products; at it, the stage that decides varies.
    let mut rng = Rng(7);
    let mut stops = [0u32; 7];
    for (d, omega, count) in [(64, 8, 300), (128, 2, 300), (128, 8, 40)] {
        let ring = Ring::new(1099511627917, d).unwrap();
        let mut polys: Vec<Poly> = (0..count)
            .map(|i| {
                autostable(
                    &mut AesPrg::new(&[6; 32], domain(d as u32, i)),
                    ring.clone(),
                    omega,
                )
                .unwrap()
            })
            .collect();
        // Not sigma-stable, and sparse ones with few large coefficients.
        for _ in 0..count / 4 {
            let c: Vec<i128> = (0..d)
                .map(|_| rng.below(2 * omega as u64 + 1) - omega)
                .collect();
            polys.push(poly(&ring, &c));
            let mut sparse = vec![0; d];
            for _ in 0..3 {
                sparse[rng.below(d as u64) as usize] = rng.below(201) - 100;
            }
            polys.push(poly(&ring, &sparse));
        }
        for c in &polys {
            let eta = smallest_eta(c);
            assert!(within_eta(c, eta).unwrap(), "{d} {eta}");
            if eta > 0 {
                assert!(!within_eta(c, eta - 1).unwrap(), "{d} {eta}");
            }
            // Where the decision at eta was made.
            let powers = eta_powers(eta).unwrap();
            let mut stage = 0;
            stage_norms(c, |j, norm| {
                stage = j;
                *norm <= powers[j]
            })
            .unwrap();
            stops[stage] += 1;
        }
    }
    // Each of the six stages decides some of these boundary cases ([0, 74, 1, 2, 153, 436,
    // 294] for j = 0 to 6 when written).
    assert!(stops[1..].iter().all(|n| *n > 0), "{stops:?}");
}

#[test]
fn each_stage_norm_is_exact_and_compared_inclusively() {
    for (d, omega) in [(64usize, 8i128), (128, 2)] {
        let ring = Ring::new(1099511627917, d).unwrap();
        for i in 0..4 {
            let c = autostable(&mut AesPrg::new(&[7; 32], i), ring.clone(), omega).unwrap();
            let mut norms = [U1024::ZERO; 7];
            stage_norms(&c, |j, n| {
                norms[j] = *n;
                false
            })
            .unwrap();
            let expected = direct_stages(&c.coefficients_i128().unwrap());
            for j in 1..=6 {
                assert_eq!(big(&norms[j]), expected[j - 1], "{d} {i} {j}");
            }
            // With every other threshold just below its norm, stage j alone decides: it
            // accepts at its norm and refuses one below.
            for j in 1..=6 {
                let mut powers = norms.map(|n| n.wrapping_sub(&U1024::ONE));
                powers[j] = norms[j];
                assert!(decide(&c, &powers).unwrap(), "{d} {i} {j}");
                powers[j] = norms[j].wrapping_sub(&U1024::ONE);
                assert!(!decide(&c, &powers).unwrap(), "{d} {i} {j}");
            }
        }
    }
}

/// The domains of 0 to 1999 whose first draw is rejected, for the seed `[9; 32]`, as the
/// Python oracle of `tools/kat/generate.py` finds them (`challenge`, one rejected draw each).
const REJECTED_FIRST_DRAWS: [(usize, i128, u64, &[u32]); 2] = [
    (
        64,
        8,
        140,
        &[
            165, 240, 312, 338, 415, 605, 614, 709, 960, 1073, 1295, 1447, 1501, 1610, 1799, 1827,
            1845, 1895, 1926, 1938,
        ],
    ),
    (
        128,
        2,
        59,
        &[
            48, 181, 492, 621, 706, 833, 853, 860, 861, 948, 1282, 1421, 1449, 1542, 1655, 1690,
            1710, 1729, 1766, 1796, 1802, 1824, 1923, 1965, 1973, 1978, 1999,
        ],
    ),
];

#[test]
fn a_challenge_is_the_first_draw_within_eta_from_one_stream() {
    for (d, omega, eta, expected) in REJECTED_FIRST_DRAWS {
        let ring = Ring::new(1099511627917, d).unwrap();
        let mut rejected = Vec::new();
        for i in 0..2000u32 {
            let seed = [9; 32];
            let mut stream = AesPrg::new(&seed, domain(0, i));
            let c = challenge(&mut stream, ring.clone(), omega, eta).unwrap();
            // The same draws, one at a time, on another copy of the stream.
            let mut copy = AesPrg::new(&seed, domain(0, i));
            let mut draws = 0;
            let first = loop {
                draws += 1;
                let x = autostable(&mut copy, ring.clone(), omega).unwrap();
                if within_eta(&x, eta).unwrap() {
                    break x;
                }
            };
            assert_eq!(c, first);
            if draws > 1 {
                rejected.push(i);
                assert_eq!(draws, 2, "{d} {i}");
            }
            // Both streams continue at the same byte.
            let (mut x, mut y) = ([0u8; 16], [0u8; 16]);
            stream.fill(&mut x).unwrap();
            copy.fill(&mut y).unwrap();
            assert_eq!(x, y);
        }
        // 1.0% and 1.35% of first draws, against about 0.76% and 1.13% expected.
        assert_eq!(rejected, expected, "{d}");
    }
}

#[test]
fn a_challenge_fails_after_the_last_draw() {
    // With eta = 0 only the zero polynomial passes, so every draw is rejected: the error
    // comes after exactly MAX_CHALLENGE_DRAWS draws.
    let ring = Ring::new(1099511627917, 64).unwrap();
    let mut stream = AesPrg::new(&[3; 32], 0);
    assert_eq!(
        challenge(&mut stream, ring.clone(), 8, 0).err(),
        Some(Error::Randomness)
    );
    let mut copy = AesPrg::new(&[3; 32], 0);
    for _ in 0..MAX_CHALLENGE_DRAWS {
        autostable(&mut copy, ring.clone(), 8).unwrap();
    }
    let (mut x, mut y) = ([0u8; 16], [0u8; 16]);
    stream.fill(&mut x).unwrap();
    copy.fill(&mut y).unwrap();
    assert_eq!(x, y);
    // A refused eta reads nothing.
    let mut stream = AesPrg::new(&[3; 32], 0);
    assert_eq!(
        challenge(&mut stream, ring, 8, 1 << 16).err(),
        Some(Error::Parameter("challenge eta capacity"))
    );
    let mut fresh = AesPrg::new(&[3; 32], 0);
    stream.fill(&mut x).unwrap();
    fresh.fill(&mut y).unwrap();
    assert_eq!(x, y);
}
