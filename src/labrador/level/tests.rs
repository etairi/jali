//! The projection's fast paths against the code they replaced: `project` (i32 lanes) against
//! `project_wide` (i128 throughout) for small, flush-boundary and large coefficients, and the
//! interleaved tables of `projection_blocks` against one table per k, each from one reader of
//! the whole stream, also over several blocks and waves of blocks.
use super::*;
use crate::labrador::{
    params::{Layout, Shape},
    test_rng::TestRng,
};

fn witness(
    rng: &mut TestRng,
    q: u64,
    vectors: usize,
    len: usize,
    bound: u64,
) -> Vec<Zeroizing<Vec<u64>>> {
    (0..vectors)
        .map(|_| {
            Zeroizing::new(
                (0..len)
                    .map(|i| {
                        // a zero in every fifth position, the bound itself in every seventh
                        let m = if i % 7 == 0 {
                            bound
                        } else {
                            rng.below(bound + 1)
                        };
                        let v = if i % 5 == 0 { 0 } else { m };
                        if rng.below(2) == 0 { v } else { (q - v) % q }
                    })
                    .collect(),
            )
        })
        .collect()
}

#[test]
fn lane_projections_equal_wide_ones() {
    let mut rng = TestRng::new(81);
    // 2^32 - 99: centred values reach 2^31 - 50, so a lane flushes after every step there.
    for q in [65557u64, 4294967197, 281474976710597] {
        let ring = Rq::new(q, 64).unwrap();
        // 8-bit lanes up to 127 (flushed every step there), 32-bit lanes from 128 on.
        for bound in [1u64, 2, 126, 127, 128, 1 << 15, (q - 1) / 2] {
            // Within one block, then over several blocks, a partial last one and two waves.
            for len in [64 * 5, 64 * 1400 + 3] {
                let w = witness(&mut rng, q, 3, len, bound);
                let seed = [rng.below(256) as u8; 32];
                assert_eq!(
                    project(&ring, &seed, 2, &w).unwrap(),
                    project_wide(&ring, &seed, 2, &w).unwrap(),
                    "q {q} bound {bound} len {len}"
                );
            }
        }
    }
}

/// `projection_blocks` with one table per k and one reader of the whole stream, as before the
/// tables were interleaved and the stream read in blocks.
fn projection_polys_reference(
    ring: &Rq,
    seed: &[u8; 32],
    level: u32,
    omega: &[Vec<u64>],
    shape: &Shape,
) -> Vec<Vec<Vec<u64>>> {
    let d = ring.degree();
    // Exact integer sums of canonical entries, reduced once at the end, for every modulus.
    let q = i128::from(ring.modulus());
    let tables: Vec<Vec<i128>> = omega
        .iter()
        .map(|w| {
            let mut table = vec![0i128; 64 * 256];
            for group in 0..64 {
                for byte in 0..256usize {
                    let mut s = 0i128;
                    for e in 0..4 {
                        let v =
                            ((byte >> (2 * e)) & 1) as i128 - ((byte >> (2 * e + 1)) & 1) as i128;
                        s += v * i128::from(w[4 * group + e]);
                    }
                    table[group * 256 + byte] = s;
                }
            }
            table
        })
        .collect();
    let mut projection = Projection::new(seed, level);
    let mut out = Vec::new();
    let mut tmp = vec![0u64; d];
    let mut column = [0u8; 64];
    for _ in 0..shape.r {
        for _ in 0..shape.n {
            let mut polys = vec![vec![0u64; d]; omega.len()];
            for c in 0..d {
                projection.column_bytes(&mut column).unwrap();
                for (poly, table) in polys.iter_mut().zip(&tables) {
                    let s: i128 = column
                        .iter()
                        .enumerate()
                        .map(|(g, b)| table[g * 256 + usize::from(*b)])
                        .sum();
                    poly[c] = s.rem_euclid(q) as u64;
                }
            }
            for poly in &mut polys {
                ring.sigma_minus_one(poly, &mut tmp);
                poly.copy_from_slice(&tmp);
            }
            out.push(polys);
        }
    }
    out
}

/// Moduli on both sides of the switch of the column sums to `i128` at $`2^{58}`$, and the
/// largest below $`2^{62}`$ (prime, 5 mod 8).
const LARGE: [u64; 4] = [
    93266870910952069,
    288230376151711717,
    288230376151711813,
    4611686018427387733,
];

#[test]
fn interleaved_tables_give_the_same_polynomials() {
    let mut rng = TestRng::new(82);
    for q in [65557u64, 4294967197, 281474976710597]
        .into_iter()
        .chain(LARGE)
    {
        let ring = Rq::new(q, 64).unwrap();
        // the general loop (1, 2) and every specialized width (3 to 8), the largest λ of the
        // moduli above 2^16
        for lambda in 1usize..=MAX_LAMBDA {
            let omega: Vec<Vec<u64>> = (0..lambda)
                .map(|_| (0..256).map(|_| rng.below(q)).collect())
                .collect();
            // One block, a partial last block, and more blocks than a wave.
            let shapes: &[(usize, usize)] = if lambda == 5 {
                &[(2, 3), (2, 97), (3, 1400)]
            } else {
                &[(2, 3)]
            };
            for &(r, n) in shapes {
                let shape = Shape { r, n, norm_sq: 0 };
                let seed = [rng.below(256) as u8; 32];
                let mut got: Vec<Vec<Vec<u64>>> = Vec::new();
                projection_blocks(&ring, &seed, 1, &omega, &shape, Ok, |block| {
                    for (index, polys) in block.iter() {
                        assert_eq!(index, got.len());
                        got.push(
                            polys
                                .chunks_exact(ring.degree())
                                .map(|p| p.to_vec())
                                .collect(),
                        );
                    }
                    Ok(())
                })
                .unwrap();
                assert_eq!(
                    got,
                    projection_polys_reference(&ring, &seed, 1, &omega, &shape),
                    "q {q} lambda {lambda} r {r} n {n}"
                );
            }
        }
    }
}

/// A layout with `nu` parts of `z0` and `z1` of `n_next` polynomials and `m` polynomials of `v`.
fn final_layout(nu: usize, n_next: usize, m: usize) -> Layout {
    Layout {
        r: 1,
        n: nu * n_next,
        pairs: 1,
        m,
        off_g: 0,
        off_h: 0,
        r_next: 2 * nu + m.div_ceil(n_next),
        n_next,
    }
}

/// A final witness: uniform digits in `[-h, h]` for `z0` and `v`, values of magnitude about
/// `spread` for `z1` (none past the length that `final_length` gives).
fn final_witness(
    ring: &Rq,
    rng: &mut TestRng,
    layout: &Layout,
    nu: usize,
    h: u64,
    spread: u64,
) -> Vec<Zeroizing<Vec<u64>>> {
    let d = ring.degree();
    (0..layout.r_next)
        .map(|i| {
            let len = final_length(layout, nu, i) * d;
            let mut v: Vec<u64> = (0..len)
                .map(|_| {
                    let x = if gaussian_part(nu, i) {
                        // a sum of uniforms, roughly Gaussian
                        (0..4).map(|_| rng.below(spread + 1) as i64).sum::<i64>()
                            - 2 * spread as i64
                    } else {
                        rng.below(2 * h + 1) as i64 - h as i64
                    };
                    ring.lift(x)
                })
                .collect();
            v.resize(layout.n_next * d, 0);
            Zeroizing::new(v)
        })
        .collect()
}

fn round_trip(ring: &Rq, layout: &Layout, nu: usize, witness: &[Zeroizing<Vec<u64>>]) -> usize {
    let bytes = encode_final(ring, layout, nu, witness).unwrap();
    let mut input = &bytes[..];
    let back = decode_final(ring, layout, nu, &mut input).unwrap();
    assert!(input.is_empty());
    assert_eq!(
        back.iter().map(|v| v.to_vec()).collect::<Vec<_>>(),
        witness.iter().map(|v| v.to_vec()).collect::<Vec<_>>()
    );
    bytes.len()
}

#[test]
fn final_witnesses_round_trip() {
    let mut rng = TestRng::new(83);
    for q in [65557u64, 4294967197, 281474976710597] {
        let ring = Rq::new(q, 64).unwrap();
        let half = q / 2;
        for (nu, n_next, m) in [(1, 4, 6), (2, 3, 7), (1, 2, 0), (3, 1, 1)] {
            let layout = final_layout(nu, n_next, m);
            for (h, spread) in [(1, 1), (8, 5), (6, 100), (half, half / 4)] {
                let witness = final_witness(&ring, &mut rng, &layout, nu, h, spread);
                round_trip(&ring, &layout, nu, &witness);
            }
            // Zero vectors, and every vector at the extremes +-(q-1)/2.
            let zero = vec![Zeroizing::new(vec![0u64; n_next * 64]); layout.r_next];
            round_trip(&ring, &layout, nu, &zero);
            let mut extreme = final_witness(&ring, &mut rng, &layout, nu, 1, 1);
            for (i, v) in extreme.iter_mut().enumerate() {
                let len = final_length(&layout, nu, i) * 64;
                for (k, x) in v[..len].iter_mut().enumerate() {
                    *x = if k % 2 == 0 { half } else { q - half };
                }
            }
            round_trip(&ring, &layout, nu, &extreme);
        }
    }
}

#[test]
fn final_witnesses_take_about_their_entropy() {
    // Uniform digits in [-8, 8] and z1 of magnitude about 4: the fixed-width code took 5 and at
    // least 5 bits per coefficient, the radix code takes log2(17) = 4.09 bits.
    let mut rng = TestRng::new(84);
    let ring = Rq::new(4294967197, 64).unwrap();
    let layout = final_layout(1, 64, 64);
    let witness = final_witness(&ring, &mut rng, &layout, 1, 8, 4);
    let bytes = round_trip(&ring, &layout, 1, &witness) as f64;
    let coefficients = (3 * 64 * 64) as f64;
    assert!(
        8.0 * bytes / coefficients < 4.3,
        "{} bits",
        8.0 * bytes / coefficients
    );
}

/// [`encode_final`] with one field of vector `at` changed (none for `None`): `Width` writes the
/// radix header one bit wider, `Max` declares a largest magnitude one above the true one,
/// `Parameter` codes the Gaussian part with the next parameter, `Chunk` writes the first radix
/// chunk as R^k. Returns the bytes and the number of bits before the padding.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Tweak {
    Width,
    Max,
    Parameter,
    Chunk,
}

fn encode_tweaked(
    ring: &Rq,
    layout: &Layout,
    nu: usize,
    witness: &[Zeroizing<Vec<u64>>],
    at: usize,
    tweak: Option<Tweak>,
) -> (Vec<u8>, usize) {
    let d = ring.degree();
    let half = u128::from(ring.modulus() / 2);
    let mut writer = BitWriter::new();
    for (i, v) in witness.iter().enumerate() {
        let tweak = if i == at { tweak } else { None };
        let len = final_length(layout, nu, i) * d;
        let values: Vec<i128> = v[..len]
            .iter()
            .map(|x| i128::from(ring.center(*x)))
            .collect();
        if gaussian_part(nu, i) {
            let mut p = gaussian_parameter(&values);
            if tweak == Some(Tweak::Parameter) {
                p += 1;
            }
            writer.unsigned(u128::from(p), 8).unwrap();
            for x in &values {
                writer.gaussian(*x, p, unary_limit(half, p)).unwrap();
            }
        } else {
            let mut h = values.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0);
            if tweak == Some(Tweak::Max) {
                h += 1;
            }
            let mut w = signed_width(h);
            if tweak == Some(Tweak::Width) {
                w += 1;
            }
            writer.unsigned(u128::from(w), 8).unwrap();
            writer.unsigned(h, w - 1).unwrap();
            if tweak == Some(Tweak::Chunk) {
                // R^k itself, in the bits of the code of [0, R^k), then the rest as usual
                let radix = 2 * h + 1;
                let k = radix_chunk(radix);
                let power = (0..k).fold(1u128, |p, _| p * radix);
                writer
                    .unsigned(power, 128 - (power - 1).leading_zeros())
                    .unwrap();
                write_radix(&mut writer, &values[k..], h).unwrap();
            } else if h > 0 {
                write_radix(&mut writer, &values, h).unwrap();
            }
        }
    }
    let payload = writer.bit_len();
    writer.unsigned(0, ((8 - payload % 8) % 8) as u32).unwrap();
    (writer.into_bytes().unwrap(), payload)
}

#[test]
fn final_witnesses_have_one_encoding() {
    let mut rng = TestRng::new(85);
    let ring = Rq::new(4294967197, 64).unwrap();
    let (nu, layout) = (1, final_layout(1, 4, 6));
    let decode = |bytes: &[u8]| {
        let mut input = bytes;
        decode_final(&ring, &layout, nu, &mut input)
    };
    // A witness whose code does not end on a byte, so that the last byte holds padding.
    let (witness, bytes, payload) = loop {
        let witness = final_witness(&ring, &mut rng, &layout, nu, 8, 5);
        let (bytes, payload) = encode_tweaked(&ring, &layout, nu, &witness, 0, None);
        if payload % 8 != 0 {
            break (witness, bytes, payload);
        }
    };
    assert_eq!(bytes, encode_final(&ring, &layout, nu, &witness).unwrap());
    assert!(decode(&bytes).is_ok());
    for (at, tweak) in [
        (0, Tweak::Width),
        (2, Tweak::Width),
        (0, Tweak::Max),
        (3, Tweak::Max),
        (1, Tweak::Parameter),
        (0, Tweak::Chunk),
        (2, Tweak::Chunk),
    ] {
        let (changed, _) = encode_tweaked(&ring, &layout, nu, &witness, at, Some(tweak));
        assert_eq!(
            decode(&changed).err(),
            Some(Error::Encoding),
            "{at} {tweak:?}"
        );
    }
    // The first padding bit set, and a missing byte.
    let mut changed = bytes.clone();
    *changed.last_mut().unwrap() |= 1 << (payload % 8);
    assert_eq!(decode(&changed).err(), Some(Error::Encoding));
    assert_eq!(
        decode(&bytes[..bytes.len() - 1]).err(),
        Some(Error::Encoding)
    );
}

/// The column sums at the largest magnitudes their bounds allow, every lane at
/// $`\pm32(q-1)`$, and on random tables, against exact integer sums, on both sides of
/// $`2^{32}`$ and $`2^{58}`$, for every width (the specialized 3 to 8 and the general loop for
/// 1 and 2) and for one, two and four columns at once.
#[test]
fn column_sums_are_exact_at_their_bounds() {
    let mut rng = TestRng::new(83);
    for q in [65557u64, 4294967197].into_iter().chain(LARGE) {
        let ring = Rq::new(q, 64).unwrap();
        let half = ((q - 1) / 2) as i64;
        for lambda in 1usize..=MAX_LAMBDA {
            let n = 64 * 81 * lambda;
            let all: Vec<Vec<i64>> = vec![
                vec![half; n],
                vec![-half; n],
                (0..n)
                    .map(|i| if i % 2 == 0 { half } else { -half })
                    .collect(),
                (0..n).map(|_| rng.below(q) as i64 - half).collect(),
            ];
            for values in all {
                let mut tables = vec![ProjectionTable {
                    lambda,
                    entries: TableEntries::Wide(values.clone()),
                }];
                if q < 1 << 32 {
                    tables.push(ProjectionTable {
                        lambda,
                        entries: TableEntries::Narrow(values.iter().map(|x| *x as i32).collect()),
                    });
                }
                // The same sums padded from an unaligned start as the AVX-512 layouts store
                // them (eight per pattern below 2^32, four from 2^58 for λ at most 4).
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    let padded = |stride: usize, start: usize| -> Vec<i64> {
                        let mut out = vec![0i64; start + 64 * 81 * stride];
                        for (i, v) in values.iter().enumerate() {
                            out[start + (i / lambda) * stride + i % lambda] = *v;
                        }
                        out
                    };
                    if lambda <= 4 {
                        tables.push(ProjectionTable {
                            lambda,
                            entries: TableEntries::WidePadded {
                                table: padded(4, 3),
                                start: 3,
                            },
                        });
                    }
                    if q < 1 << 32 {
                        tables.push(ProjectionTable {
                            lambda,
                            entries: TableEntries::NarrowPadded {
                                table: padded(8, 5).iter().map(|x| *x as i32).collect(),
                                start: 5,
                            },
                        });
                    }
                }
                for table in &tables {
                    let columns: [[u8; 64]; 4] = [
                        [0u8; 64],
                        [255u8; 64],
                        core::array::from_fn(|_| rng.below(256) as u8),
                        core::array::from_fn(|_| rng.below(256) as u8),
                    ];
                    let exact = |column: &[u8; 64], k: usize| -> u64 {
                        let e: i128 = column
                            .iter()
                            .enumerate()
                            .map(|(g, b)| i128::from(table.entry(g, *b, k)))
                            .sum();
                        e.rem_euclid(i128::from(q)) as u64
                    };
                    let mut four = [[0u64; 8]; 4];
                    table.sums(&ring, &columns, &mut four);
                    let mut two = [[0u64; 8]; 2];
                    table.sums(&ring, &[columns[2], columns[3]], &mut two);
                    for (c, column) in columns.iter().enumerate() {
                        let mut one = [[0u64; 8]; 1];
                        table.sums(&ring, &[*column], &mut one);
                        for k in 0..lambda {
                            let e = exact(column, k);
                            assert_eq!(one[0][k], e, "q {q} lambda {lambda} one column");
                            assert_eq!(four[c][k], e, "q {q} lambda {lambda} four columns");
                            if c >= 2 {
                                assert_eq!(two[c - 2][k], e, "q {q} lambda {lambda} two columns");
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The table of `projection_table`, for every group, byte and $`k`$, holds the sum of the
/// table indexed by the byte itself (the one of 0.3.3, built here as it was), so that the
/// polynomials of `block_polys` are those of a byte table: below $`2^{32}`$, between $`2^{32}`$
/// and $`2^{58}`$ and above, for every λ of the supported moduli.
#[test]
fn the_pattern_table_holds_the_byte_table_sums() {
    let mut rng = TestRng::new(84);
    for q in [
        65557u64,
        4294967197,
        (1 << 40) + 21,
        (1 << 61) - 1,
        LARGE[0],
    ] {
        let Ok(ring) = Rq::new(q, 64) else { continue };
        for lambda in 3..=MAX_LAMBDA {
            let omega: Vec<Vec<u64>> = (0..lambda)
                .map(|_| (0..256).map(|_| rng.below(q)).collect())
                .collect();
            let mut reference = vec![0i64; 64 * 256 * lambda];
            for (k, w) in omega.iter().enumerate() {
                for group in 0..64 {
                    for byte in 0..256usize {
                        let mut s = 0u64;
                        for (e, v) in DECODE[byte].iter().enumerate() {
                            match v {
                                1 => s = ring.add(s, w[4 * group + e]),
                                -1 => s = ring.sub(s, w[4 * group + e]),
                                _ => {}
                            }
                        }
                        reference[(group * 256 + byte) * lambda + k] = ring.center(s);
                    }
                }
            }
            let table = projection_table(&ring, &omega);
            for group in 0..64 {
                for byte in 0..=255u8 {
                    for k in 0..lambda {
                        assert_eq!(
                            table.entry(group, byte, k),
                            reference[(group * 256 + usize::from(byte)) * lambda + k],
                            "q {q} lambda {lambda} group {group} byte {byte} k {k}"
                        );
                    }
                }
            }
        }
    }
}

/// Every pattern index is below 81, and two bytes share one exactly when their four entries
/// agree.
#[test]
fn patterns_index_the_sign_patterns() {
    for a in 0..256usize {
        assert!(PATTERN[a] < 81);
        for b in 0..256usize {
            assert_eq!(
                PATTERN[a] == PATTERN[b],
                DECODE[a] == DECODE[b],
                "bytes {a} {b}"
            );
        }
    }
}

/// The masks and shifts of `gaussian_bits` against the divisions they replaced.
#[test]
fn gaussian_bits_match_the_division_formula() {
    fn reference(x: i128, p: u32) -> u64 {
        let scale = 1i128 << (p + 1);
        let r = x.rem_euclid(scale);
        let low = if r >= scale / 2 { r - scale } else { r };
        let quotient = (x - low) / scale;
        let n = if quotient > 0 {
            2 * quotient - 1
        } else {
            -2 * quotient
        };
        n as u64 + 1 + u64::from(p + 1)
    }
    let mut rng = TestRng::new(84);
    let mut values: Vec<i128> = vec![0, 1, -1, 2, -2, 3, -3];
    for k in 1..100 {
        let v = 1i128 << k;
        values.extend([v, -v, v - 1, 1 - v, v + 1, -v - 1]);
    }
    values.extend((0..200).map(|_| rng.small(1 << 40) as i128));
    for p in 0..=100 {
        for x in &values {
            // Quotients of more than 2^62 would overflow the bit count of both versions.
            if (x.unsigned_abs() >> (p + 1)) < 1 << 60 {
                assert_eq!(gaussian_bits(*x, p), reference(*x, p), "x {x} p {p}");
            }
        }
    }
}

/// The digits of the radix code from 64-bit pieces against one division per digit, for small
/// and large radices, the general path above 2^64 included, and every chunk length.
#[test]
fn radix_digits_match_digit_by_digit_division() {
    let mut rng = TestRng::new(85);
    for h in [
        1u128,
        2,
        3,
        7,
        127,
        1 << 20,
        (1 << 31) + 5,
        (1 << 62) - 1,
        1 << 63,
        1 << 70,
    ] {
        let radix = 2 * h + 1;
        let k = radix_chunk(radix);
        for len in 1..=k {
            let power = (0..len).fold(1u128, |p, _| p * radix);
            for value in [
                0,
                power - 1,
                (u128::from(rng.next_u64()) << 64 | u128::from(rng.next_u64())) % power,
            ] {
                let mut want = Vec::new();
                let mut v = value;
                for _ in 0..len {
                    want.push((v % radix) as i128 - h as i128);
                    v /= radix;
                }
                let mut got = Vec::new();
                push_radix_digits(&mut got, value, len, radix, h);
                assert_eq!(got, want, "h {h} len {len}");
            }
        }
    }
}

/// `combine_hats` and `combine` against the sparse products they replaced
/// (`Rq::mul_sparse_add`), for sparse polynomials with coefficients ±1 and ±2 at distinct
/// positions (the shape of the challenges) and random vectors, at moduli on both sides of
/// $`2^{40}`$.
#[test]
fn transform_combinations_equal_the_sparse_products() {
    let mut rng = TestRng::new(85);
    for q in [4294967197u64, LARGE[0], LARGE[3]] {
        let ring = Rq::new(q, 64).unwrap();
        let d = ring.degree();
        let (r, n) = (3usize, 5usize);
        let challenges: Vec<Sparse> = (0..r)
            .map(|_| {
                let mut positions: Vec<usize> = (0..d).collect();
                for i in 0..d {
                    positions.swap(i, i + rng.below((d - i) as u64) as usize);
                }
                let values = [1i64, -1, 2, -2];
                Sparse {
                    terms: positions[..54]
                        .iter()
                        .map(|p| (*p, values[rng.below(4) as usize]))
                        .collect(),
                }
            })
            .collect();
        let vectors: Vec<Vec<u64>> = (0..r)
            .map(|_| (0..n * d).map(|_| rng.below(q)).collect())
            .collect();
        let mut reference = vec![0u64; n * d];
        for (c, v) in challenges.iter().zip(&vectors) {
            for x in 0..n {
                ring.mul_sparse_add(
                    c,
                    &v[x * d..(x + 1) * d],
                    &mut reference[x * d..(x + 1) * d],
                );
            }
        }
        let c_hats = challenge_hats(&ring, &challenges);
        let v_hats: Vec<Vec<u64>> = vectors.iter().map(|v| ring.ntt_many(v)).collect();
        assert_eq!(
            *combine_hats(&ring, &c_hats, &v_hats, n),
            reference,
            "q {q} stored"
        );
        assert_eq!(
            combine(&ring, &c_hats, &vectors, n),
            reference,
            "q {q} computed"
        );
    }
}
