//! The sampler against a reference sampler, kept below as the oracle: equal
//! outputs and stream consumption on shared streams for $`t\in\{0,\dots,16,24,33,41,68,100\}`$;
//! uniform values forced at and around the exact cutoff, decided as the exact test decides
//! them and, at the cutoff's own top word, by the exact test itself; the width 1.55 ($`t=0`$)
//! on hand-made tapes; the integer enclosure around the exact cutoff on sampler inputs; and the
//! variance it is built for.
use super::*;
use crate::rand::{AesPrg, domain, reject::exp_negative};

/// The reference sampler: the exact cutoff for every candidate, every table entry compared
/// with the table value (not only the run of entries above it), and the offset bits read one
/// at a time.
struct Reference<'a, S> {
    stream: &'a mut S,
    signs: u64,
    remaining: u32,
}
impl<S: ByteStream> Reference<'_, S> {
    fn word(&mut self) -> Result<u64, Error> {
        let mut b = [0; 8];
        self.stream.fill(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn bit(&mut self) -> Result<i128, Error> {
        if self.remaining == 0 {
            self.signs = self.word()?;
            self.remaining = 64;
        }
        let bit = self.signs & 1;
        self.signs >>= 1;
        self.remaining -= 1;
        Ok(i128::from(bit))
    }
    /// $`\#\{j:V<T_j\}`$ over every entry, for $`V`$ of 32 bytes, little-endian.
    fn draw(&mut self, table: &[U256]) -> Result<i128, Error> {
        let mut bytes = [0u8; 32];
        self.stream.fill(&mut bytes)?;
        let value = U256::from_le_slice(&bytes);
        Ok(table.iter().filter(|entry| value < **entry).count() as i128)
    }
    fn narrow(&mut self) -> Result<i128, Error> {
        for _ in 0..4096 {
            let b = self.bit()?;
            match (b, self.draw(&HALF_1_55)?) {
                (1, 0) => continue,
                (1, k) => return Ok(k),
                (_, k) => return Ok(-k),
            }
        }
        Err(Error::Randomness)
    }
    fn base(&mut self, u: u128, t: u32) -> Result<i128, Error> {
        let variance = Variance::gaussian(t)?;
        for _ in 0..4096 {
            let b = self.bit()?;
            let k = self.draw(&HALF_3_1)?;
            let k = if b == 1 { k + 1 } else { -k };
            let scale = 1i128 << (t - 1);
            let a = U256::from((k * scale - u as i128).unsigned_abs());
            let v = U256::from(((k - b) * scale).unsigned_abs());
            let numerator = a
                .wrapping_mul(&a)
                .wrapping_sub(&v.wrapping_mul(&v))
                .wrapping_mul(&U256::from(variance.denominator));
            let denominator = variance.numerator.shl_vartime(1);
            let cutoff = exp_negative(numerator, denominator)?;
            let mut random = [0u8; 32];
            self.stream.fill(&mut random[..24])?;
            if U256::from_le_slice(&random) < cutoff {
                return Ok(k);
            }
        }
        Err(Error::Randomness)
    }
}
fn reference_gaussian(
    stream: &mut impl ByteStream,
    log_sigma: u32,
    count: usize,
) -> Result<Vec<i128>, Error> {
    // t - 1 offset bits per coefficient, none at t = 0.
    let offset = log_sigma.saturating_sub(1) as usize;
    let mut bytes = vec![0u8; (count * offset).div_ceil(8)];
    stream.fill(&mut bytes)?;
    let mut sampler = Reference {
        stream,
        signs: 0,
        remaining: 0,
    };
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        if log_sigma == 0 {
            out.push(sampler.narrow()?);
            continue;
        }
        let mut u = 0u128;
        for j in 0..offset {
            let index = i * offset + j;
            u |= u128::from((bytes[index / 8] >> (index % 8)) & 1) << j;
        }
        let k = sampler.base(u, log_sigma)?;
        out.push(k * (1i128 << offset) - u as i128);
    }
    Ok(out)
}

/// The widths of the test plan.
const WIDTHS: [u32; 22] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 33, 41, 68, 100,
];
/// The widths with a Bernoulli test: every one but $`t=0`$.
fn bernoulli_widths() -> impl Iterator<Item = u32> {
    WIDTHS.into_iter().filter(|t| *t > 0)
}
/// The largest base sample of width 3.1.
const LARGEST: i128 = HALF_3_1.len() as i128 - 1;

fn exact_tests() -> u64 {
    EXACT_TESTS.with(|n| n.get())
}

/// Both samplers on two copies of one stream: equal outputs, then equal next bytes. Returns
/// the number of exact tests the sampler ran.
fn compare(t: u32, count: usize, seed: u8) -> u64 {
    let mut new_stream = AesPrg::new(&[seed; 32], domain(t, 7));
    let mut reference_stream = AesPrg::new(&[seed; 32], domain(t, 7));
    let before = exact_tests();
    let new = gaussian(&mut new_stream, t, count).unwrap();
    let exact = exact_tests() - before;
    assert_eq!(
        new,
        reference_gaussian(&mut reference_stream, t, count).unwrap(),
        "t {t}"
    );
    let (mut x, mut y) = ([0u8; 48], [0u8; 48]);
    new_stream.fill(&mut x).unwrap();
    reference_stream.fill(&mut y).unwrap();
    assert_eq!(x, y, "consumption, t {t}");
    exact
}

#[test]
fn the_sampler_equals_the_reference_on_shared_streams() {
    // An exact test runs with probability about 2^-55 per candidate (the enclosure's width),
    // so none is expected among these candidates.
    let exact: u64 = WIDTHS.iter().map(|t| compare(*t, 2_000, 1)).sum();
    assert_eq!(exact, 0);
}

#[test]
#[ignore = "2.2 million coefficients against the exact sampler, one thread per width"]
fn the_sampler_equals_the_reference_on_100000_coefficients_per_width() {
    let exact: u64 = std::thread::scope(|scope| {
        let handles: Vec<_> = WIDTHS
            .iter()
            .map(|t| scope.spawn(move || compare(*t, 100_000, 2)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    });
    assert_eq!(exact, 0);
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
}

/// The operands $`(a,v)`$ of the trial for offset `u`, base sample `k` and sign `b` at width
/// exponent $`t\ge1`$, as the sampler forms them: at scale $`2^{t-1}`$.
fn operands(u: u128, k: i128, b: i128, t: u32) -> (u128, u128) {
    let k = if b == 1 { k + 1 } else { -k };
    let scale = 1i128 << (t - 1);
    (
        (k * scale - u as i128).unsigned_abs(),
        ((k - b) * scale).unsigned_abs(),
    )
}

/// Sampler inputs: every base sample and sign with offsets at the ends and random ones.
fn inputs(t: u32, rng: &mut Rng, random: usize) -> Vec<(u128, u128)> {
    let top = (1u128 << (t - 1)) - 1;
    let mut offsets = vec![0, top, top / 2];
    offsets
        .extend((0..random).map(|_| (u128::from(rng.next()) << 64 | u128::from(rng.next())) & top));
    let mut out = Vec::new();
    for k in 0..=LARGEST {
        for b in 0..=1 {
            for u in &offsets {
                out.push(operands(*u, k, b, t));
            }
        }
    }
    out
}

fn cutoff(a: u128, v: u128, t: u32) -> U256 {
    exact_cutoff(a, v, Variance::gaussian(t).unwrap()).unwrap()
}

#[test]
fn the_enclosure_contains_the_exact_cutoff_on_sampler_inputs() {
    let mut rng = Rng(3);
    let mut widest = 0;
    for t in bernoulli_widths() {
        for (a, v) in inputs(t, &mut rng, 8) {
            let [lower, upper] = bernoulli::exp_bounds(a, v, t).expect("sampler inputs");
            let c = cutoff(a, v, t);
            let (l, u) = (
                U256::from(lower).shl_vartime(129),
                U256::from(upper).shl_vartime(129),
            );
            // What the decisions need, (A) of `bernoulli`: within 2^128 of the enclosure.
            let slack = U256::ONE.shl_vartime(128);
            assert!(
                l < c.wrapping_add(&slack) && c < u.wrapping_add(&slack),
                "t {t}, a {a}"
            );
            // What `exp_negative` documents, about 2^22 units, holds here too.
            let documented = U256::ONE.shl_vartime(23);
            assert!(l <= c.wrapping_add(&documented) && c <= u.wrapping_add(&documented));
            widest = widest.max(upper - lower);
        }
    }
    // The width sets how often the exact test runs: at most about 2^13 of 2^64 top words.
    assert!(widest < 1 << 12, "width {widest}");
}

#[test]
fn decisions_at_and_near_the_cutoff_equal_the_exact_test() {
    let mut rng = Rng(4);
    let mut deltas: Vec<i128> = vec![0, 1, -1];
    for j in 1..127 {
        deltas.extend([
            1i128 << j,
            -(1i128 << j),
            (1i128 << j) + 1,
            -(1i128 << j) - 1,
        ]);
    }
    let top = U256::ONE.shl_vartime(192);
    for t in bernoulli_widths() {
        for (a, v) in inputs(t, &mut rng, 2) {
            let c = cutoff(a, v, t);
            if c < top {
                // The cutoff's own top word is never decided early.
                let h0 = crate::math::int::low_u64(&c.shr_vartime(128));
                assert_eq!(bernoulli::decide(a, v, t, h0), None, "t {t}, a {a}, v {v}");
            }
            for delta in &deltas {
                let magnitude = U256::from(delta.unsigned_abs());
                let value = if *delta < 0 {
                    c.wrapping_sub(&magnitude)
                } else {
                    c.wrapping_add(&magnitude)
                };
                if value >= top || (*delta < 0 && magnitude > c) {
                    continue;
                }
                let high = crate::math::int::low_u64(&value.shr_vartime(128));
                if let Some(accept) = bernoulli::decide(a, v, t, high) {
                    assert_eq!(accept, value < c, "t {t}, a {a}, v {v}, delta {delta}");
                }
            }
            // And across the whole range of top words.
            for _ in 0..8 {
                let high = rng.next();
                if let Some(accept) = bernoulli::decide(a, v, t, high) {
                    let low = U256::from(high).shl_vartime(128);
                    let high_end = U256::from(u128::from(high) + 1).shl_vartime(128);
                    assert!(if accept { high_end <= c } else { low >= c }, "t {t}");
                }
            }
        }
    }
}

/// A stream over fixed bytes.
struct Tape {
    bytes: Vec<u8>,
    position: usize,
}
impl ByteStream for Tape {
    fn fill(&mut self, output: &mut [u8]) -> Result<(), Error> {
        let end = self.position + output.len();
        output.copy_from_slice(
            self.bytes
                .get(self.position..end)
                .ok_or(Error::Randomness)?,
        );
        self.position = end;
        Ok(())
    }
}

#[test]
fn uniform_values_at_the_cutoff_take_the_exact_test_in_the_sampler() {
    // One coefficient: its offset u, then a candidate with sign b, base sample 0 (the table
    // value 2^256 - 1 is above every entry) and the uniform value C + delta; if that
    // candidate is rejected, a second one (sign 0, base sample 0, uniform value 0) is accepted.
    let mut rng = Rng(5);
    let mut exact_runs = 0;
    for t in bernoulli_widths() {
        let top = (1u128 << (t - 1)) - 1;
        for u in [
            0,
            top,
            top / 3,
            (u128::from(rng.next()) << 64 | u128::from(rng.next())) & top,
        ] {
            for b in 0..=1 {
                let (a, v) = operands(u, 0, b, t);
                let c = cutoff(a, v, t);
                for delta in [-2i128, -1, 0, 1, 2, 1 << 64, -(1 << 64)] {
                    let magnitude = U256::from(delta.unsigned_abs());
                    let value = if delta < 0 {
                        c.wrapping_sub(&magnitude)
                    } else {
                        c.wrapping_add(&magnitude)
                    };
                    if value >= U256::ONE.shl_vartime(192) {
                        continue;
                    }
                    // t - 1 offset bits, then the sign word and the two candidates.
                    let mut bytes = u.to_le_bytes()[..(t as usize - 1).div_ceil(8)].to_vec();
                    bytes.extend((b as u64).to_le_bytes());
                    bytes.extend([0xff; 32]);
                    bytes.extend(&value.to_le_bytes()[..24]);
                    bytes.extend([0xff; 32]);
                    bytes.extend([0; 24]);
                    let mut new_tape = Tape {
                        bytes: bytes.clone(),
                        position: 0,
                    };
                    let mut reference_tape = Tape { bytes, position: 0 };
                    let before = exact_tests();
                    let new = gaussian(&mut new_tape, t, 1).unwrap();
                    exact_runs += exact_tests() - before;
                    let expected = reference_gaussian(&mut reference_tape, t, 1).unwrap();
                    assert_eq!(new, expected, "t {t}, u {u}, b {b}, delta {delta}");
                    assert_eq!(new_tape.position, reference_tape.position);
                    // Accepted at once (value below C), or by the second candidate.
                    let first = new_tape.position == new_tape.bytes.len() - 56;
                    assert_eq!(first, value < c, "t {t}, u {u}, b {b}, delta {delta}");
                }
            }
        }
    }
    // Deltas -2..=2 put the uniform value's top word at the cutoff's (unless that word
    // changes), and those candidates take the exact test.
    let widths = bernoulli_widths().count() as u64;
    assert!(exact_runs >= 4 * 2 * widths, "{exact_runs} exact tests");
}

#[test]
fn width_1_55_draws_a_signed_base_sample_and_draws_minus_zero_again() {
    // t = 0: no offset, then per attempt a sign bit of the cached word and a 32-byte table
    // value; the table value T_{k-1} - 1 gives K = k, T_{k-1} itself K = k - 1, 2^256 - 1
    // gives K = 0.
    let le = |x: U256| {
        let mut out = [0u8; 32];
        out.copy_from_slice(&x.to_le_bytes());
        out
    };
    let below = |k: usize| le(HALF_1_55[k - 1].wrapping_sub(&U256::ONE));
    let at = |k: usize| le(HALF_1_55[k - 1]);
    let largest = HALF_1_55.len() - 1;
    // Signs, least significant bit first: 1 (with K = 0: drawn again), 1, 0, 0, 1, 1, 0.
    let signs: u64 = 0b011_0011;
    let mut bytes = signs.to_le_bytes().to_vec();
    for value in [
        [0xff; 32],
        below(3),
        [0xff; 32],
        below(5),
        below(largest),
        at(2),
        below(1),
    ] {
        bytes.extend(value);
    }
    let expected = [3, 0, -5, largest as i128, 1, -1];
    let mut tape = Tape {
        bytes: bytes.clone(),
        position: 0,
    };
    assert_eq!(gaussian(&mut tape, 0, 6).unwrap(), expected);
    assert_eq!(tape.position, 8 + 7 * 32);
    let mut tape = Tape { bytes, position: 0 };
    assert_eq!(reference_gaussian(&mut tape, 0, 6).unwrap(), expected);
    assert_eq!(tape.position, 8 + 7 * 32);
    // No Bernoulli test runs at t = 0: a sample is exactly a draw of the table.
    assert_eq!(largest, 29);
}

#[test]
fn the_enclosure_is_built_for_the_sampler_variance() {
    for t in 0..=crate::rand::MAX_LOG_SIGMA {
        assert_eq!(
            Variance::gaussian(t).unwrap(),
            Variance {
                numerator: U256::from(961u16).shl_vartime(2 * t),
                denominator: 400,
            }
        );
    }
    assert_ne!((200u128 << 64) % 961, 0);
}
