//! The projection's fast paths against the code they replaced: `project` (i32 lanes) against
//! `project_wide` (i128 throughout) for small, flush-boundary and large coefficients, and the
//! interleaved tables of `for_each_projection_poly` against one table per k.
use super::*;
use crate::labrador::{params::Shape, test_rng::TestRng};

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
        for bound in [1u64, 1 << 15, (q - 1) / 2] {
            let w = witness(&mut rng, q, 3, 64 * 5, bound);
            let seed = [rng.below(256) as u8; 32];
            assert_eq!(
                project(&ring, &seed, 2, &w).unwrap(),
                project_wide(&ring, &seed, 2, &w).unwrap(),
                "q {q} bound {bound}"
            );
        }
    }
}

/// `for_each_projection_poly` with one table per k, as before the tables were interleaved.
fn projection_polys_reference(
    ring: &Rq,
    seed: &[u8; 32],
    level: u32,
    omega: &[Vec<u64>],
    shape: &Shape,
) -> Vec<Vec<Vec<u64>>> {
    let d = ring.degree();
    let q = ring.modulus() as i64;
    let tables: Vec<Vec<i64>> = omega
        .iter()
        .map(|w| {
            let mut table = vec![0i64; 64 * 256];
            for group in 0..64 {
                for byte in 0..256usize {
                    let mut s = 0i64;
                    for e in 0..4 {
                        let v = ((byte >> (2 * e)) & 1) as i64 - ((byte >> (2 * e + 1)) & 1) as i64;
                        s += v * w[4 * group + e] as i64;
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
                    let s: i64 = column
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

#[test]
fn interleaved_tables_give_the_same_polynomials() {
    let mut rng = TestRng::new(82);
    for q in [65557u64, 4294967197, 281474976710597] {
        let ring = Rq::new(q, 64).unwrap();
        // the general loop (1, 2, 9) and every specialized width (3 to 8)
        for lambda in 1usize..=9 {
            let omega: Vec<Vec<u64>> = (0..lambda)
                .map(|_| (0..256).map(|_| rng.below(q)).collect())
                .collect();
            let shape = Shape {
                r: 2,
                n: 3,
                norm_sq: 0,
            };
            let seed = [rng.below(256) as u8; 32];
            let mut got = Vec::new();
            for_each_projection_poly(&ring, &seed, 1, &omega, &shape, |_, _, polys| {
                got.push(polys.to_vec());
                Ok(())
            })
            .unwrap();
            assert_eq!(
                got,
                projection_polys_reference(&ring, &seed, 1, &omega, &shape),
                "q {q} lambda {lambda}"
            );
        }
    }
}
