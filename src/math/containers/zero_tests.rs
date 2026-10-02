//! Explicit zero entries share one zero polynomial: the keys, values,
//! equality and `Debug` output are those of a map that holds a zero polynomial per entry, and
//! the in-place algebra keeps explicit zeros without giving them polynomials of their own.
use super::*;
use crate::quad::QuadEq;

/// The derived `Debug` of maps that hold a polynomial per entry.
mod dense {
    use crate::math::{Poly, Ring};
    use std::{collections::BTreeMap, sync::Arc};

    #[derive(Debug)]
    #[allow(dead_code)]
    pub(super) struct SparsePolyMat {
        pub(super) ring: Arc<Ring>,
        pub(super) dim: usize,
        pub(super) entries: BTreeMap<(u16, u16), Poly>,
    }
    #[derive(Debug)]
    #[allow(dead_code)]
    pub(super) struct SparsePolyVec {
        pub(super) ring: Arc<Ring>,
        pub(super) dim: usize,
        pub(super) entries: BTreeMap<u16, Poly>,
    }
}

impl SparsePolyMat {
    /// Test only: how many entries hold a polynomial of their own.
    pub(crate) fn held(&self) -> usize {
        self.entries
            .values()
            .filter(|s| matches!(s, Slot::Value(_)))
            .count()
    }
}
impl SparsePolyVec {
    /// Test only: how many entries hold a polynomial of their own.
    pub(crate) fn held(&self) -> usize {
        self.entries
            .values()
            .filter(|s| matches!(s, Slot::Value(_)))
            .count()
    }
}

fn ring() -> Arc<Ring> {
    Ring::with_modulus(U256::from_u64((1 << 40) - 87), 64).unwrap()
}

/// A form in 12 variables with explicit zeros, scalars and dense entries. The kind of an r2
/// entry depends on (c + seed) mod 3 and its presence on (r + c + seed) mod 3, so that both
/// parts hold explicit zeros.
fn form(ring: &Arc<Ring>, seed: u64) -> QuadEq {
    let mut s = seed;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut poly = |kind: u64| match kind % 3 {
        0 => Poly::zero(ring.clone()),
        1 => Poly::constant(ring.clone(), (next() % 1000) as i128 - 500),
        _ => Poly::new(
            ring.clone(),
            (0..64).map(|_| (next() % 2001) as i128 - 1000).collect(),
        )
        .unwrap(),
    };
    let r2 = (0..12u16)
        .flat_map(|r| (r..12).map(move |c| (r, c)))
        .filter(|(r, c)| !(r * 7 + c + seed as u16).is_multiple_of(3))
        .map(|(r, c)| (r, c, poly(u64::from(3 * r + c) + seed)))
        .collect();
    let r1 = (0..12u16)
        .filter(|i| (i + seed as u16).is_multiple_of(2))
        .map(|i| (i, poly(u64::from(i) + seed)))
        .collect();
    QuadEq {
        r2: SparsePolyMat::new(ring.clone(), 12, r2).unwrap(),
        r1: SparsePolyVec::new(ring.clone(), 12, r1).unwrap(),
        r0: poly(seed),
    }
}

#[test]
fn explicit_zeros_read_as_zero_polynomials_with_dense_equality_and_debug_output() {
    let ring = ring();
    let one = Poly::constant(ring.clone(), 1);
    let entries = vec![
        (0u16, 1u16, Poly::zero(ring.clone())),
        (1, 3, one.clone()),
        (2, 2, Poly::zero(ring.clone())),
    ];
    let m = SparsePolyMat::new(ring.clone(), 4, entries.clone()).unwrap();
    assert_eq!(m.held(), 1);
    let read: Vec<_> = m.entries().map(|(k, p)| (k, p.clone())).collect();
    assert_eq!(
        read,
        entries
            .iter()
            .map(|(r, c, p)| ((*r, *c), p.clone()))
            .collect::<Vec<_>>()
    );
    let dense = dense::SparsePolyMat {
        ring: ring.clone(),
        dim: 4,
        entries: entries
            .iter()
            .map(|(r, c, p)| ((*r, *c), p.clone()))
            .collect(),
    };
    assert_eq!(format!("{m:?}"), format!("{dense:?}"));
    assert_eq!(format!("{m:#?}"), format!("{dense:#?}"));
    // A value that is zero after a cancellation equals an explicit zero.
    let mut cancelled = SparsePolyMat::new(ring.clone(), 4, vec![(0, 1, one.clone())]).unwrap();
    cancelled.add_at(0, 1, &one.neg()).unwrap();
    cancelled.add_at(1, 3, &one).unwrap();
    cancelled.add_at(2, 2, &Poly::zero(ring.clone())).unwrap();
    assert_eq!(cancelled, m);
    assert_ne!(
        SparsePolyMat::new(ring.clone(), 4, vec![(0, 1, Poly::zero(ring.clone()))]).unwrap(),
        SparsePolyMat::new(ring.clone(), 4, vec![]).unwrap(),
        "a zero key is part of the value"
    );
    let v = SparsePolyVec::new(
        ring.clone(),
        4,
        vec![(0, Poly::zero(ring.clone())), (3, one.clone())],
    )
    .unwrap();
    assert_eq!(v.held(), 1);
    let dense = dense::SparsePolyVec {
        ring: ring.clone(),
        dim: 4,
        entries: [(0, Poly::zero(ring.clone())), (3, one)].into(),
    };
    assert_eq!(format!("{v:?}"), format!("{dense:?}"));
}

#[test]
fn the_in_place_algebra_keeps_explicit_zeros_without_polynomials() {
    let ring = ring();
    let zero_form = |seed| {
        let f = form(&ring, seed);
        // Only explicit zeros: every key of `f` with a zero value.
        QuadEq {
            r2: SparsePolyMat::new(
                ring.clone(),
                12,
                f.r2.entries()
                    .map(|((r, c), _)| (r, c, Poly::zero(ring.clone())))
                    .collect(),
            )
            .unwrap(),
            r1: SparsePolyVec::new(
                ring.clone(),
                12,
                f.r1.entries()
                    .map(|(i, _)| (i, Poly::zero(ring.clone())))
                    .collect(),
            )
            .unwrap(),
            r0: Poly::zero(ring.clone()),
        }
    };
    let (z1, z2) = (zero_form(1), zero_form(2));
    let weight = Poly::new(ring.clone(), (1..=64).collect()).unwrap();
    let mut acc = QuadEq::zero(ring.clone(), 12).unwrap();
    acc.add_assign(&z1).unwrap();
    acc.add_scaled_scalar_assign(&z2, &U256::from_u64(7))
        .unwrap();
    acc.add_scaled_assign(&z1, &weight).unwrap();
    let held = |f: &QuadEq| f.r2.held() + f.r1.held();
    assert_eq!(held(&acc), 0);
    assert_eq!(acc, z1.add(&z2).unwrap());
    let traced = acc.clone().into_trace().unwrap();
    assert_eq!(held(&traced), 0);
    assert_eq!(traced, acc.trace().unwrap());
    // Products of linear forms with zero coefficients keep every pair key, without values.
    let linear = |seed| {
        let mut f = zero_form(seed);
        f.r2 = SparsePolyMat::new(ring.clone(), 12, vec![]).unwrap();
        f
    };
    let mut product = QuadEq::zero(ring.clone(), 12).unwrap();
    product.add_product_affine(&linear(3), &linear(4)).unwrap();
    assert_eq!(held(&product), 0);
    assert_eq!(product, linear(3).product_affine(&linear(4)).unwrap());
    // Values that are not zero are held; the keys and values equal the copying operations'.
    let (f1, f2) = (form(&ring, 5), form(&ring, 6));
    let mut acc = f1.clone();
    acc.add_scaled_assign(&f2, &weight).unwrap();
    assert_eq!(acc, f1.add(&f2.scale(&weight).unwrap()).unwrap());
    assert!(held(&acc) <= acc.r2.entries().count() + acc.r1.entries().count());
    let nonzero = |f: &QuadEq| {
        f.r2.entries().filter(|(_, p)| !p.is_zero()).count()
            + f.r1.entries().filter(|(_, p)| !p.is_zero()).count()
    };
    assert!(held(&acc) >= nonzero(&acc));
}

#[test]
fn conjugation_equals_the_entrywise_definition_with_explicit_zeros() {
    let ring = ring();
    for seed in 1..8 {
        let f = form(&ring, seed);
        // The definition: entry (r, c) at the key of
        // (r ^ 1, c ^ 1) with its automorphism image, zero entries included.
        let expected = QuadEq {
            r2: SparsePolyMat::new(
                ring.clone(),
                12,
                f.r2.entries()
                    .map(|((r, c), p)| ((r ^ 1).min(c ^ 1), (r ^ 1).max(c ^ 1), p.auto()))
                    .collect(),
            )
            .unwrap(),
            r1: SparsePolyVec::new(
                ring.clone(),
                12,
                f.r1.entries().map(|(i, p)| (i ^ 1, p.auto())).collect(),
            )
            .unwrap(),
            r0: f.r0.auto(),
        };
        let zeros = |q: &QuadEq| {
            q.r2.entries().filter(|(_, p)| p.is_zero()).count()
                + q.r1.entries().filter(|(_, p)| p.is_zero()).count()
        };
        assert!(
            f.r2.entries()
                .any(|((r, c), p)| p.is_zero() && (r ^ 1, c ^ 1) != (c, r)),
            "seed {seed}: an explicit zero that conjugation moves"
        );
        assert!(zeros(&f) > 0);
        let conjugate = f.conjugate().unwrap();
        assert_eq!(conjugate, expected, "seed {seed}");
        let keys = |q: &QuadEq| q.r2.entries().map(|(k, _)| k).collect::<Vec<_>>();
        assert_eq!(keys(&conjugate), keys(&expected), "seed {seed}");
        // Explicit zeros stay without polynomials of their own.
        assert_eq!(conjugate.r2.held(), f.r2.held(), "seed {seed}");
        assert_eq!(conjugate.r1.held(), f.r1.held(), "seed {seed}");
    }
}
