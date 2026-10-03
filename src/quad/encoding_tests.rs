//! The encoding of polynomials for statement hashing: a zero polynomial is one run of zero
//! bits, which must equal its $`d`$ uniform codes, at every
//! alignment of the writer and for narrow and wide moduli. The segments of an equation hold the
//! bytes that `to_bytes` returned before them, here a reference with one code per coefficient:
//! for random equations, zero polynomials first and throughout, encodings that end just short
//! of a chunk or just past it, and zero runs that start around the end of a chunk; and an
//! equation that cannot be encoded fails with the same error.
use super::*;
use crate::codec::{CHUNK, MIN_ZERO_RUN};

/// The header of a record of segments: a literal and a zero byte count.
const HEADER: usize = 8;

/// Reference `encode_poly`: one uniform code per coefficient.
fn reference_encode_poly(w: &mut BitWriter, p: &Poly) -> Result<(), Error> {
    let q = p.ring().modulus();
    match int::to_u128(&q) {
        Some(m) => {
            for x in p.coefficients() {
                w.uniform(int::low_u128(x), m)?;
            }
        }
        None => {
            for x in p.coefficients() {
                w.uniform_u256(x, &q)?;
            }
        }
    }
    Ok(())
}

#[test]
fn zero_polynomials_encode_as_their_uniform_codes() {
    let two = |k: u32| U256::ONE.shl_vartime(k);
    for q in [
        U256::from_u8(2),
        U256::from_u8(3),
        U256::from_u8(13),
        U256::from_u64(1099511627917),
        two(64),
        two(100).wrapping_sub(&U256::from_u8(15)),
        two(128).wrapping_add(&U256::from_u8(165)),
        two(255).wrapping_sub(&U256::from_u8(19)),
    ] {
        for d in [64, 128] {
            let ring = Ring::with_modulus(q, d).unwrap();
            let zero = Poly::zero(ring.clone());
            let other = Poly::constant(ring.clone(), -1).rotate(5);
            for offset in 0..70u32 {
                let (mut new, mut expected) = (BitWriter::new(), BitWriter::new());
                for w in [&mut new, &mut expected] {
                    w.unsigned((1 << offset) - 1, offset).unwrap();
                }
                for p in [&zero, &other, &zero, &zero] {
                    encode_poly(&mut new, p).unwrap();
                    reference_encode_poly(&mut expected, p).unwrap();
                }
                assert_eq!(
                    new.finish(),
                    expected.finish(),
                    "q {q}, d {d}, offset {offset}"
                );
            }
        }
    }
}

/// [`QuadEq::to_bytes`] as it was before segments: the entries collected first, every
/// coefficient its own uniform code ([`reference_encode_poly`]), into one growing buffer.
pub(crate) fn reference_to_bytes(eq: &QuadEq) -> Result<Vec<u8>, Error> {
    eq.check(eq.r0.ring(), eq.r2.dimension())?;
    let mut w = BitWriter::new();
    w.unsigned(eq.r2.dimension() as u128, 32)?;
    let matrix: Vec<_> = eq.r2.entries().collect();
    let vector: Vec<_> = eq.r1.entries().collect();
    w.unsigned(matrix.len() as u128, 32)?;
    for ((r, c), p) in matrix {
        w.unsigned(r.into(), 16)?;
        w.unsigned(c.into(), 16)?;
        reference_encode_poly(&mut w, p)?;
    }
    w.unsigned(vector.len() as u128, 32)?;
    for (i, p) in vector {
        w.unsigned(i.into(), 16)?;
        reference_encode_poly(&mut w, p)?;
    }
    reference_encode_poly(&mut w, &eq.r0)?;
    Ok(w.finish())
}

pub(crate) struct Rng(pub(crate) u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub(crate) fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// A uniform 256-bit value, from bytes, so that it does not depend on the limb width.
    fn uniform(&mut self) -> U256 {
        let words = [self.next(), self.next(), self.next(), self.next()];
        U256::from_le_slice(&words.map(u64::to_le_bytes).concat())
    }
    /// A polynomial with coefficient 0 equal to one and the others uniform.
    fn nonzero_poly(&mut self, ring: &Arc<Ring>) -> Poly {
        let mut coeffs: Vec<_> = (0..ring.degree())
            .map(|_| ring.reduce(&self.uniform()))
            .collect();
        coeffs[0] = U256::ONE;
        Poly::from_u256(ring.clone(), coeffs).unwrap()
    }
    /// Zero `zero_percent` percent of the time; otherwise a scalar, a monomial or dense.
    fn poly(&mut self, ring: &Arc<Ring>, zero_percent: u64) -> Poly {
        if self.below(100) < zero_percent {
            return Poly::zero(ring.clone());
        }
        let d = ring.degree();
        let mut coeffs = vec![U256::ZERO; d];
        match self.below(3) {
            0 => coeffs[0] = ring.reduce(&self.uniform()),
            1 => coeffs[self.below(d as u64) as usize] = ring.reduce(&self.uniform()),
            _ => coeffs
                .iter_mut()
                .for_each(|x| *x = ring.reduce(&self.uniform())),
        }
        Poly::from_u256(ring.clone(), coeffs).unwrap()
    }
}

/// An equation in `dim` coordinates with up to `keys` matrix and `keys` vector entries, each
/// polynomial zero `zero_percent` percent of the time.
pub(crate) fn random_equation(
    rng: &mut Rng,
    ring: &Arc<Ring>,
    dim: usize,
    keys: u64,
    zero_percent: u64,
) -> QuadEq {
    let mut matrix = BTreeMap::new();
    let mut vector = BTreeMap::new();
    for _ in 0..keys {
        let (i, j) = (rng.below(dim as u64) as u16, rng.below(dim as u64) as u16);
        let p = rng.poly(ring, zero_percent);
        matrix.insert((i.min(j), i.max(j)), p);
        let p = rng.poly(ring, zero_percent);
        vector.insert(rng.below(dim as u64) as u16, p);
    }
    QuadEq {
        r2: SparsePolyMat::new(
            ring.clone(),
            dim,
            matrix.into_iter().map(|((i, j), p)| (i, j, p)).collect(),
        )
        .unwrap(),
        r1: SparsePolyVec::new(ring.clone(), dim, vector.into_iter().collect()).unwrap(),
        r0: rng.poly(ring, zero_percent),
    }
}

/// The equation's segments hold the bytes of the reference and of `to_bytes`, in chunks of
/// at most `CHUNK` bytes: allocated with that capacity, the last one shrunk to its length.
/// Returns the chunks' lengths.
pub(crate) fn assert_same_encoding(eq: &QuadEq) -> Vec<usize> {
    let reference = reference_to_bytes(eq).unwrap();
    assert_eq!(eq.to_bytes().unwrap(), reference);
    let segments = eq.to_segments().unwrap();
    assert_eq!(segments.len(), reference.len() as u64);
    assert_eq!(segments.to_vec(), reference);
    let shapes = segments.chunk_shapes();
    let (last, full) = shapes.split_last().expect("an encoding is never empty");
    assert!(full.iter().all(|(len, cap)| len <= cap && *cap == CHUNK));
    assert!(last.0 == last.1 && last.1 <= CHUNK);
    shapes.into_iter().map(|(len, _)| len).collect()
}

/// Narrow and wide moduli, at degrees 64 and 128: zero polynomials of 256 bits and below,
/// which stay bytes, and of 2624 bits and above, which become counts.
fn rings() -> Vec<Arc<Ring>> {
    let two = |k: u32| U256::ONE.shl_vartime(k);
    let mut rings = Vec::new();
    for q in [
        U256::from_u8(3),
        U256::from_u8(13),
        U256::from_u64(1099511627917),
        two(64),
        two(100).wrapping_sub(&U256::from_u8(15)),
        two(128).wrapping_add(&U256::from_u8(165)),
        two(255).wrapping_sub(&U256::from_u8(19)),
    ] {
        for d in [64, 128] {
            rings.push(Ring::with_modulus(q, d).unwrap());
        }
    }
    rings
}

#[test]
fn segments_hold_the_encoding_of_random_equations() {
    let mut rng = Rng(77);
    let mut chunks = 0;
    for ring in rings() {
        for zero_percent in [0, 50, 90, 100] {
            for keys in [0, 1, 6, 60, 400] {
                let dim = 2 + rng.below(400) as usize;
                let eq = random_equation(&mut rng, &ring, dim, keys, zero_percent);
                chunks += assert_same_encoding(&eq).len();
            }
        }
    }
    assert!(chunks > 300, "{chunks}");
}

#[test]
fn zero_polynomials_first_and_throughout() {
    for ring in rings() {
        let zero = || Poly::zero(ring.clone());
        let mut rng = Rng(ring.coefficient_bits().into());
        // The first polynomial zero (the case that took the doubling buffer from 576 bytes),
        // then nonzero and zero entries in turn.
        let eq = QuadEq {
            r2: SparsePolyMat::new(
                ring.clone(),
                8,
                vec![
                    (0, 0, zero()),
                    (0, 1, rng.nonzero_poly(&ring)),
                    (1, 1, zero()),
                    (2, 7, zero()),
                ],
            )
            .unwrap(),
            r1: SparsePolyVec::new(
                ring.clone(),
                8,
                vec![(0, zero()), (3, rng.nonzero_poly(&ring)), (7, zero())],
            )
            .unwrap(),
            r0: zero(),
        };
        assert_same_encoding(&eq);
        // Every polynomial zero, over many keys: held in a few hundred bytes when the zero
        // codes of a polynomial span at least MIN_ZERO_RUN bytes after its first word.
        let keys: Vec<_> = (0..60u16)
            .flat_map(|i| (i..60).map(move |j| (i, j)))
            .take(1500)
            .collect();
        let eq = QuadEq {
            r2: SparsePolyMat::new(
                ring.clone(),
                60,
                keys.iter().map(|(i, j)| (*i, *j, zero())).collect(),
            )
            .unwrap(),
            r1: SparsePolyVec::new(ring.clone(), 60, (0..60).map(|i| (i, zero())).collect())
                .unwrap(),
            r0: zero(),
        };
        let held: usize = assert_same_encoding(&eq).iter().sum();
        let run_bits = (ring.degree() as u64) * u64::from(ring.coefficient_bits());
        if run_bits >= 8 * (MIN_ZERO_RUN + 16) {
            let joined = eq.to_bytes().unwrap().len();
            assert!(held * 10 < joined, "held {held}, joined {joined}");
        }
        // No entries at all.
        assert_same_encoding(&QuadEq::zero(ring.clone(), 5).unwrap());
    }
}

#[test]
fn encodings_at_and_around_the_end_of_a_chunk() {
    // Equations of `matrix` and `vector` nonzero entries, one zero vector entry if `gap`, three
    // more nonzero vector entries and a nonzero constant, over q = 2^bits at degree 64: every
    // code is a multiple of 16 bits, so an encoding has an odd number of bytes.
    let shape = |bits: u32, matrix: u16, vector: u16, gap: bool| {
        let ring = Ring::with_modulus(U256::ONE.shl_vartime(bits), 64).unwrap();
        let mut rng = Rng(u64::from(bits) << 32 | u64::from(matrix) << 16 | u64::from(vector));
        let r2 = (0..matrix)
            .map(|i| (i, i, rng.nonzero_poly(&ring)))
            .collect();
        let mut r1: Vec<_> = (0..vector + 4)
            .map(|i| (i, rng.nonzero_poly(&ring)))
            .collect();
        if gap {
            r1[usize::from(vector)].1 = Poly::zero(ring.clone());
        }
        QuadEq {
            r2: SparsePolyMat::new(ring.clone(), 2048, r2).unwrap(),
            r1: SparsePolyVec::new(ring.clone(), 2048, r1).unwrap(),
            r0: rng.nonzero_poly(&ring),
        }
    };
    let mut found = [0; 5];
    for bits in 9..=40u32 {
        let code = 64 * bits as usize;
        for matrix in 0..8 {
            let before = 64 + matrix * (32 + code) + 32;
            let vector = (CHUNK * 8 - before) / (16 + code);
            for vector in vector.saturating_sub(7)..=vector + 1 {
                // Without a zero entry: the bytes are the literal bytes of one record per chunk.
                let len = (before + (vector + 4) * (16 + code) + code + 1).div_ceil(8);
                let case = match len {
                    // One byte short of a full chunk, or one byte past it.
                    _ if len == CHUNK - HEADER - 1 => Some((0, vec![CHUNK - 1])),
                    _ if len == CHUNK - HEADER + 1 => Some((1, vec![CHUNK, HEADER + 1])),
                    _ => None,
                };
                if let Some((i, shapes)) = case {
                    let eq = shape(bits, matrix as u16, vector as u16, false);
                    assert_eq!(assert_same_encoding(&eq), shapes, "bits {bits}");
                    found[i] += 1;
                }
                // With the zero entry: its run of whole zero bytes starts after the word that
                // holds its first bit, here 8 bytes before the end of the first chunk, at its
                // end, or 8 bytes past it.
                let start = before + vector * (16 + code) + 16;
                let run_start = 8 * (start / 64 + 1);
                let run = 8 * ((code - (64 - start % 64)) / 64) as u64;
                let first = match CHUNK as isize - (HEADER + run_start) as isize {
                    // The next literal byte needs a new chunk, since a header does not fit.
                    8 => Some((2, CHUNK - 8)),
                    // The first chunk is full when the run starts.
                    0 => Some((3, CHUNK)),
                    // The first literal bytes spill over: the run follows them in chunk 2.
                    -8 => Some((4, CHUNK)),
                    _ => None,
                };
                if let (Some((i, first)), true) = (first, run >= MIN_ZERO_RUN) {
                    let eq = shape(bits, matrix as u16, vector as u16, true);
                    let lengths = assert_same_encoding(&eq);
                    assert_eq!(lengths[0], first, "bits {bits}");
                    assert!(lengths.len() >= 2);
                    found[i] += 1;
                }
            }
        }
    }
    assert!(found.iter().all(|n| *n > 0), "{found:?}");
}

#[test]
fn equations_that_fail_to_encode_fail_alike() {
    let ring = Ring::new(1099511627917, 64).unwrap();
    let other = Ring::new(13, 64).unwrap();
    let mut rng = Rng(5);
    let mut wrong_dimension = random_equation(&mut rng, &ring, 40, 5, 50);
    wrong_dimension.r1 = SparsePolyVec::new(ring.clone(), 42, vec![]).unwrap();
    let mut wrong_ring = random_equation(&mut rng, &ring, 40, 5, 50);
    wrong_ring.r0 = Poly::zero(other);
    for (eq, error) in [
        (wrong_dimension, Error::Dimension),
        (wrong_ring, Error::RingMismatch),
    ] {
        assert_eq!(reference_to_bytes(&eq), Err(error.clone()));
        assert_eq!(eq.to_bytes(), Err(error.clone()));
        assert_eq!(eq.to_segments().err(), Some(error));
    }
}
