//! The forms of an affine block against the column-by-column construction that they replaced:
//! the same keys, explicit zeros included, and values, for dense and sparse matrices.
use super::*;
use crate::math::{Poly, PolyMat, PolyVec};

/// Every column through `get`, as before.
fn column_by_column(block: &AffineBlock, scheme: &Abdlop, dimension: usize) -> Vec<QuadEq> {
    let p = &scheme.parameters;
    let ring = scheme.ring();
    (0..block.rows)
        .map(|row| {
            let mut entries = Vec::new();
            if let Some(matrix) = &block.s {
                for j in 0..p.m1 {
                    entries.push(((2 * j) as u16, matrix.get(row, j).unwrap().clone()));
                }
            }
            if let Some(matrix) = &block.m {
                for j in 0..p.l {
                    let key = (2 * (scheme.bounded_len() + j)) as u16;
                    entries.push((key, matrix.get(row, j).unwrap().clone()));
                }
            }
            let mut equation = QuadEq::zero(ring.clone(), dimension).unwrap();
            equation.r1 = SparsePolyVec::new(ring.clone(), dimension, entries).unwrap();
            if let Some(offset) = &block.offset {
                equation.r0 = offset.entries()[row].clone();
            }
            equation
        })
        .collect()
}

#[test]
fn forms_equal_the_column_by_column_ones() {
    let scheme = Abdlop::new([5; 32], crate::params::toy_d64()).unwrap();
    let (m1, l) = (scheme.parameters.m1, scheme.parameters.l);
    let ring = scheme.ring().clone();
    let dimension = 2 * (scheme.bounded_len() + scheme.message_len());
    for rows in 1..5 {
        let value = |i: usize| match i % 4 {
            0 => Poly::zero(ring.clone()),
            1 => Poly::constant(ring.clone(), 1),
            2 => Poly::constant(ring.clone(), (i % 7) as i128 - 3).rotate(i as i64),
            _ => Poly::new(
                ring.clone(),
                (0..ring.degree())
                    .map(|c| ((c * i) % 11) as i128 - 5)
                    .collect(),
            )
            .unwrap(),
        };
        let dense =
            PolyMat::new(ring.clone(), rows, m1, (0..rows * m1).map(value).collect()).unwrap();
        let sparse = |cols: usize| {
            let listed = (0..rows)
                .filter(|_| cols > 0)
                .map(|r| (r, (3 * r) % cols, value(r + 1)))
                .collect();
            PolyMat::sparse(ring.clone(), rows, cols, listed).unwrap()
        };
        let offset = PolyVec::new(
            ring.clone(),
            (0..rows)
                .map(|i| Poly::constant(ring.clone(), i as i128 + 2))
                .collect(),
        )
        .unwrap();
        for (s, m, with_offset) in [
            (Some(dense.clone()), Some(sparse(l)), true),
            (Some(sparse(m1)), None, false),
            (None, Some(sparse(l)), true),
            (
                Some(dense.clone()),
                Some(
                    PolyMat::new(ring.clone(), rows, l, (0..rows * l).map(value).collect())
                        .unwrap(),
                ),
                false,
            ),
        ] {
            let block = AffineBlock {
                rows,
                s,
                m,
                offset: with_offset.then(|| offset.clone()),
            };
            assert_eq!(
                block.forms(&scheme, dimension).unwrap(),
                column_by_column(&block, &scheme, dimension),
                "rows {rows}"
            );
        }
    }
}
