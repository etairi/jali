//! The AVX-512 transforms against the scalar ones: every NTT prime, every degree from 16 to
//! 1024, random residues and the edges 0 and p - 1, forward, inverse and one after the other.
//! On a processor without AVX-512 the test says so and passes, the scalar code being the only
//! code that runs there.
use crate::math::ntt::{NttPlan, pow_mod};
use crate::params::moduli::NTT_PRIMES;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

#[test]
fn avx512_transforms_equal_the_scalar_ones() {
    if !super::avx512() {
        eprintln!("no AVX-512 here (or JALI_SCALAR set): only the scalar transforms run");
        return;
    }
    let mut rng = Rng(41);
    for (p, root) in NTT_PRIMES {
        for d in [16usize, 32, 64, 128, 256, 512, 1024] {
            let plan = NttPlan::new(p, d, pow_mod(root, (1024 / d) as u64, p)).unwrap();
            for case in 0..5 {
                let a: Vec<u64> = (0..d)
                    .map(|i| match (case, i % 7) {
                        (0, _) => 0,
                        (1, _) => p - 1,
                        (2, 0) => p - 1,
                        (2, 1) => 0,
                        _ => rng.next() % p,
                    })
                    .collect();
                let (mut scalar, mut vector) = (a.clone(), a.clone());
                plan.forward_mont_scalar(&mut scalar);
                assert!(super::forward_mont(&plan.simd_tables(), &mut vector));
                assert_eq!(vector, scalar, "forward p {p} d {d} case {case}");
                let (mut scalar, mut vector) = (a.clone(), a.clone());
                plan.inverse_mont_scalar(&mut scalar);
                assert!(super::inverse_mont(&plan.simd_tables(), &mut vector));
                assert_eq!(vector, scalar, "inverse p {p} d {d} case {case}");
                let mut x = a.clone();
                plan.forward_mont(&mut x);
                plan.inverse_mont(&mut x);
                assert_eq!(x, a, "round trip p {p} d {d} case {case}");
            }
        }
    }
}

#[test]
fn the_bmi_permutation_equals_the_keccak_crate() {
    if !super::bmi() {
        eprintln!("no BMI1 and BMI2 here (or JALI_SCALAR set): the keccak crate permutes");
        return;
    }
    let mut rng = Rng(43);
    let keccak = keccak::Keccak::new();
    for case in 0..2000 {
        let mut a: [u64; 25] = core::array::from_fn(|_| match case {
            0 => 0,
            1 => u64::MAX,
            _ => rng.next(),
        });
        let mut b = a;
        keccak.with_f1600(|f| f(&mut a));
        assert!(super::keccak_f1600(&mut b));
        assert_eq!(a, b, "case {case}");
    }
}

#[test]
fn elementwise_kernels_equal_the_scalar_formulas() {
    use crate::math::ntt::{add_mod, shoup, shoup_mul, sub_mod};
    if !super::avx512() {
        eprintln!("no AVX-512 here (or JALI_SCALAR set): only the scalar loops run");
        return;
    }
    let mut rng = Rng(44);
    // The NTT primes, LaBRADOR moduli (32, 61.1 bits) and a small one, every length up to 40.
    let mut moduli: Vec<u64> = NTT_PRIMES.iter().map(|(p, _)| *p).collect();
    moduli.extend([4294967197, 2402958527344632421, 65557]);
    for &p in &moduli {
        for len in 0..=40usize {
            let mut value = |i: usize| match i % 9 {
                0 => 0,
                1 => p - 1,
                _ => rng.next() % p,
            };
            let a: Vec<u64> = (0..len).map(&mut value).collect();
            let b: Vec<u64> = (0..len).map(|i| value(i + 4)).collect();
            let w = rng.next() % p;
            let ws = shoup(w, p);
            let check = |kernel: &dyn Fn(&mut [u64]) -> bool,
                         scalar: &dyn Fn(u64, u64) -> u64,
                         what: &str| {
                let mut x = a.clone();
                assert!(kernel(&mut x), "{what} ran");
                let expect: Vec<u64> = a.iter().zip(&b).map(|(x, y)| scalar(*x, *y)).collect();
                assert_eq!(x, expect, "{what} p {p} len {len}");
            };
            check(
                &|x| super::add_mod(x, &b, p),
                &|x, y| add_mod(x, y, p),
                "add_mod",
            );
            check(
                &|x| super::sub_mod(x, &b, p),
                &|x, y| sub_mod(x, y, p),
                "sub_mod",
            );
            check(
                &|x| super::add_scaled(x, &b, w, ws, p),
                &|x, y| add_mod(x, shoup_mul(y, w, ws, p), p),
                "add_scaled",
            );
            check(
                &|x| super::sub_scaled(x, &b, w, ws, p),
                &|x, y| sub_mod(x, shoup_mul(y, w, ws, p), p),
                "sub_scaled",
            );
            check(
                &|x| super::scale(x, w, ws, p),
                &|x, _| shoup_mul(x, w, ws, p),
                "scale",
            );
        }
    }
    // The Montgomery accumulation for each NTT prime, and the centred residues of each
    // LaBRADOR modulus below it.
    for (p, root) in NTT_PRIMES {
        let plan = NttPlan::new(p, 64, pow_mod(root, 16, p)).unwrap();
        for len in 0..=40usize {
            let a: Vec<u64> = (0..len)
                .map(|i| if i == 0 { p - 1 } else { rng.next() % p })
                .collect();
            let b: Vec<u64> = (0..len)
                .map(|i| if i == 1 { p - 1 } else { rng.next() % p })
                .collect();
            let acc: Vec<u64> = (0..len).map(|_| rng.next() % p).collect();
            let mut x = acc.clone();
            assert!(super::mont_mac(&mut x, &a, &b, p, plan.neg_inv()));
            let expect: Vec<u64> = (0..len)
                .map(|i| add_mod(acc[i], plan.mont_mul(a[i], b[i]), p))
                .collect();
            assert_eq!(x, expect, "mont_mac p {p} len {len}");
            for q in [4294967197u64, 2402958527344632421, 65557] {
                let v: Vec<u64> = (0..len)
                    .map(|i| match i % 5 {
                        0 => 0,
                        1 => q - 1,
                        2 => q / 2,
                        3 => q / 2 + 1,
                        _ => rng.next() % q,
                    })
                    .collect();
                let mut out = vec![0; len];
                assert!(super::centred(&v, q, p, &mut out));
                let expect: Vec<u64> = v
                    .iter()
                    .map(|&x| if x > q / 2 { p - (q - x) } else { x })
                    .collect();
                assert_eq!(out, expect, "centred q {q} p {p} len {len}");
            }
        }
    }
}

#[test]
fn rns_kernels_equal_the_scalar_formulas() {
    use crate::math::ntt::{add_mod, shoup, shoup_mul, sub_mod};
    if !super::avx512() {
        eprintln!("no AVX-512 here (or JALI_SCALAR set): only the scalar loops run");
        return;
    }
    let mut rng = Rng(45);
    let wide = |rng: &mut Rng| (u128::from(rng.next()) << 64) | u128::from(rng.next());
    for (p, _) in NTT_PRIMES {
        let c = ((1u128 << 64) % u128::from(p)) as u64;
        // Moduli on both sides of 2^64 and up to 2^126, and one-word ones below 2^63.
        for q in [
            (1u128 << 63) + 25,
            (1u128 << 64) + 13,
            800_602_353_498_606_932_557,
            (1u128 << 126) - 137,
        ] {
            let k = super::ResidueConstants {
                p,
                c,
                c_shoup: shoup(c, p),
                one_shoup: shoup(1, p),
                q_mod_p: (q % u128::from(p)) as u64,
            };
            let half = q >> 1;
            for len in [0usize, 1, 7, 8, 9, 16, 64, 67] {
                let values: Vec<u128> = (0..len)
                    .map(|i| match i % 7 {
                        0 => 0,
                        1 => q - 1,
                        2 => half,
                        3 => half + 1,
                        // the high word of half, the low word around it
                        4 => (half & !u128::from(u64::MAX)) | u128::from(rng.next()),
                        _ => wide(&mut rng) % q,
                    })
                    .collect();
                let out = super::centred_wide(&values, half, &k).expect("AVX-512");
                let expect: Vec<u64> = values
                    .iter()
                    .map(|&v| {
                        let r = add_mod(
                            shoup_mul((v >> 64) as u64, c, k.c_shoup, p),
                            shoup_mul(v as u64, 1, k.one_shoup, p),
                            p,
                        );
                        if v > half {
                            sub_mod(r, k.q_mod_p, p)
                        } else {
                            r
                        }
                    })
                    .collect();
                assert_eq!(out, expect, "centred_wide p {p} q {q} len {len}");
            }
        }
        for q in [3u64, 65557, 1099511627917, (1 << 62) + 135, (1 << 63) - 25] {
            let (half, qp) = (q / 2, q % p);
            for len in [0usize, 5, 8, 13, 64] {
                let values: Vec<u64> = (0..len)
                    .map(|i| match i % 5 {
                        0 => 0,
                        1 => q - 1,
                        2 => half,
                        3 => half + 1,
                        _ => rng.next() % q,
                    })
                    .collect();
                let out = super::centred_word(&values, half, p, qp).expect("AVX-512");
                let expect: Vec<u64> = values
                    .iter()
                    .map(|&v| {
                        let r = v % p;
                        if v > half { sub_mod(r, qp, p) } else { r }
                    })
                    .collect();
                assert_eq!(out, expect, "centred_word p {p} q {q} len {len}");
            }
        }
        // Sums of up to 15 products of residues on sums below p, as the lazy sums bound them.
        for len in [0usize, 3, 8, 11, 64] {
            let mut sums: Vec<u128> = (0..len).map(|_| u128::from(rng.next() % p)).collect();
            let mut expect = sums.clone();
            for _ in 0..15 {
                let x: Vec<u64> = (0..len)
                    .map(|i| if i == 0 { p - 1 } else { rng.next() % p })
                    .collect();
                let y: Vec<u64> = (0..len)
                    .map(|i| if i == 0 { p - 1 } else { rng.next() % p })
                    .collect();
                assert!(super::lazy_mac(&mut sums, &x, &y));
                for ((e, a), b) in expect.iter_mut().zip(&x).zip(&y) {
                    *e += u128::from(*a) * u128::from(*b);
                }
            }
            assert_eq!(sums, expect, "lazy_mac p {p} len {len}");
        }
    }
}

#[test]
fn table_sums_equal_the_scalar_sums() {
    if !super::avx512() {
        eprintln!("no AVX-512 here (or JALI_SCALAR set): only the scalar loops run");
        return;
    }
    let mut rng = Rng(46);
    for lambda in 1..=8usize {
        // Centred entries of at most 2^57 in magnitude, the bound below q = 2^58.
        let table: Vec<i64> = (0..64 * 256 * lambda)
            .map(|i| match i % 11 {
                0 => (1 << 57) - 1,
                1 => -(1 << 57) + 1,
                _ => (rng.next() >> 7) as i64 - (1 << 56),
            })
            .collect();
        for _ in 0..20 {
            let columns: [[u8; 64]; 8] =
                core::array::from_fn(|_| core::array::from_fn(|_| rng.next() as u8));
            let mut out = [[0i64; 8]; 8];
            assert!(super::table_sums8(&table, lambda, &columns, &mut out));
            for (c, column) in columns.iter().enumerate() {
                for k in 0..lambda {
                    let expect: i64 = (0..64)
                        .map(|g| table[(256 * g + usize::from(column[g])) * lambda + k])
                        .sum();
                    assert_eq!(out[c][k], expect, "lambda {lambda} column {c} k {k}");
                }
            }
        }
    }
}

/// The AVX-512 byte kernel of LaBRADOR's projection against the portable update, for
/// coefficients up to ±127 in chunks of the flush length, with columns of every entry (all +1,
/// all −1, cancelling bits) and random ones.
#[test]
fn byte_projection_equals_the_portable_update() {
    if !super::avx512bw() {
        eprintln!("no AVX-512BW here (or JALI_SCALAR set): only the portable loops run");
        return;
    }
    let mut rng = Rng(46);
    for max in [1i8, 2, 63, 127] {
        let flush = 127 / max as usize;
        for len in [0usize, 1, 3, flush] {
            let mut columns: Vec<[u8; 64]> = vec![[0x55; 64], [0xaa; 64], [0xff; 64]];
            columns.extend((3..len.max(3)).map(|_| core::array::from_fn(|_| rng.next() as u8)));
            columns.truncate(len);
            let values: Vec<i8> = (0..len)
                .map(|i| {
                    let m = if i % 3 == 0 {
                        max
                    } else {
                        1 + (rng.next() % max as u64) as i8
                    };
                    if rng.next().is_multiple_of(2) { m } else { -m }
                })
                .collect();
            let mut got = [[0i8; 64]; 4];
            assert!(super::project_bytes(&mut got, &columns, &values));
            let mut expect = [[0i8; 64]; 4];
            for (column, s) in columns.iter().zip(&values) {
                for (e, lane) in expect.iter_mut().enumerate() {
                    for (a, b) in lane.iter_mut().zip(column) {
                        let entry = i8::from((b >> (2 * e)) & 1 == 1)
                            - i8::from((b >> (2 * e + 1)) & 1 == 1);
                        *a += entry * s;
                    }
                }
            }
            assert_eq!(got, expect, "max {max} len {len}");
        }
    }
}

/// The kernels of the table of sign patterns, with AVX-512 VBMI where it runs and without it,
/// against the sums of the table's entries, for λ from 1 to 8, at the largest 32-bit entries
/// and on random ones, with a map onto every pattern.
#[test]
fn pattern_sums_equal_the_scalar_sums() {
    if !super::avx512vl() {
        eprintln!("no AVX-512VL here (or JALI_SCALAR set): only the portable sums run");
        return;
    }
    let mut rng = Rng(81);
    let map: [u8; 256] = core::array::from_fn(|b| ((b * 7) % 81) as u8);
    for lambda in 1..=8usize {
        let n = 64 * 81 * lambda;
        let tables: [Vec<i32>; 3] = [
            vec![i32::MAX; n],
            vec![-i32::MAX; n],
            (0..n).map(|_| rng.next() as i32).collect(),
        ];
        for table in &tables {
            let mut columns = [[0u8; 64]; 8];
            columns[1] = [255; 64];
            for c in columns.iter_mut().skip(2) {
                *c = core::array::from_fn(|_| rng.next() as u8);
            }
            let mut expect = [[0i64; 8]; 8];
            for (e, column) in expect.iter_mut().zip(&columns) {
                for (g, b) in column.iter().enumerate() {
                    let at = (81 * g + usize::from(map[usize::from(*b)])) * lambda;
                    for (k, x) in e[..lambda].iter_mut().enumerate() {
                        *x += i64::from(table[at + k]);
                    }
                }
            }
            let mut got = [[0i64; 8]; 8];
            assert!(super::pattern_sums8(
                table, lambda, &map, &columns, &mut got
            ));
            let mut plain = [[0i64; 8]; 8];
            // SAFETY: the processor has AVX-512F and VL (checked above).
            unsafe {
                super::modular_avx512::pattern_sums8(table, lambda, &map, &columns, &mut plain)
            };
            for c in 0..8 {
                assert_eq!(
                    got[c][..lambda],
                    expect[c][..lambda],
                    "lambda {lambda} column {c}"
                );
                assert_eq!(
                    plain[c][..lambda],
                    expect[c][..lambda],
                    "plain lambda {lambda} column {c}"
                );
            }
        }
    }
}
