//! The fast integer evaluation against the checked one: equal integers wherever its bound
//! holds, all coefficients and the constant one alone, for forms with zero, scalar, monomial
//! and general entries, and the fallback where coefficients or sums are too large.
use super::*;
use crate::math::SparsePolyMat;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform in [-b, b].
    fn small(&mut self, b: u64) -> i128 {
        (self.next() % (2 * b + 1)) as i128 - b as i128
    }
}

/// A polynomial of each shape in turn, coefficients bounded by `b` in magnitude.
fn shaped(ring: &Arc<Ring>, rng: &mut Rng, kind: u64, b: u64) -> Poly {
    let d = ring.degree();
    let mut values = vec![0i128; d];
    match kind % 4 {
        0 => {}
        1 => values[0] = rng.small(b),
        2 => {
            let k = 1 + (rng.next() as usize) % (d - 1);
            values[k] = rng.small(b);
        }
        _ => {
            for v in values.iter_mut() {
                *v = rng.small(b);
            }
        }
    }
    Poly::new(ring.clone(), values).unwrap()
}

fn form(ring: &Arc<Ring>, rng: &mut Rng, dim: usize, b: u64, dense: bool) -> QuadEq {
    let mut eq = QuadEq::zero(ring.clone(), dim).unwrap();
    let mut r2 = Vec::new();
    for r in 0..dim as u16 {
        for c in r..dim as u16 {
            if !rng.next().is_multiple_of(3) {
                let kind = if dense { 3 } else { rng.next() };
                r2.push((r, c, shaped(ring, rng, kind, b)));
            }
        }
    }
    eq.r2 = SparsePolyMat::new(ring.clone(), dim, r2).unwrap();
    let mut r1 = Vec::new();
    for i in 0..dim as u16 {
        let kind = rng.next();
        r1.push((i, shaped(ring, rng, kind, b)));
    }
    eq.r1 = SparsePolyVec::new(ring.clone(), dim, r1).unwrap();
    eq.r0 = shaped(ring, rng, 3, b);
    eq
}

/// The checked evaluation of the code before the fast path.
fn checked(form: &QuadEq, witness: &[Poly]) -> Vec<Wide> {
    match integer_evaluate_as::<i128>(form, witness) {
        Ok(values) => values.iter().map(|x| Wide(I512::from_i128(*x))).collect(),
        Err(_) => integer_evaluate_as::<Wide>(form, witness).unwrap().to_vec(),
    }
}

#[test]
fn fast_integer_evaluations_equal_checked_ones() {
    let mut rng = Rng(51);
    let mut fast_runs = 0;
    for q in [13i128, 1099511627917, 3606003552620118107] {
        for d in [64, 128] {
            let ring = Ring::new(q, d).unwrap();
            for round in 0..10u64 {
                // Witness bound 1, 2^20 or q/2: the last ones leave the fast path's bound.
                let wb = [1u64, 1 << 20, (q / 2) as u64][(round % 3) as usize];
                let witness: Vec<Poly> = (0..5)
                    .map(|i| shaped(&ring, &mut rng, i + round, wb))
                    .collect();
                let eq = form(&ring, &mut rng, 5, (q / 2) as u64, round % 2 == 0);
                let want = checked(&eq, &witness);
                let full = integer_evaluate(&eq, &witness).unwrap();
                assert_eq!(full.to_vec(), want, "q {q} d {d} round {round}");
                let constant = integer_evaluate_constant(&eq, &witness).unwrap();
                assert_eq!(constant, want[0], "q {q} d {d} round {round}");
                if integer_evaluate_fast(&eq, &witness, false).is_some() {
                    fast_runs += 1;
                }
            }
        }
    }
    // The fast path ran for the small witnesses and declined for some of the large ones.
    assert!(
        fast_runs > 10 && fast_runs < 60,
        "fast path taken {fast_runs} times"
    );
}

#[test]
fn the_fast_path_declines_coefficients_beyond_i64() {
    // q > 2^64: centred coefficients up to q/2 do not fit i64.
    let q = (1i128 << 100) - 15;
    let ring = Ring::new(q, 64).unwrap();
    let mut rng = Rng(52);
    let witness: Vec<Poly> = (0..3).map(|i| shaped(&ring, &mut rng, i, 1)).collect();
    let mut eq = form(&ring, &mut rng, 3, 1, true);
    eq.r0 = Poly::constant(ring.clone(), q / 2);
    assert!(integer_evaluate_fast(&eq, &witness, false).is_none());
    assert_eq!(
        integer_evaluate(&eq, &witness).unwrap().to_vec(),
        checked(&eq, &witness)
    );
}
