//! The matrices of `selector`: the entries, in order, that collecting their flattened rows
//! gives, held in a buffer of exactly their number.
use super::*;

/// The matrix of `selector` from `start` on, `cols` wide, collected from its flattened rows.
fn collected(ring: &Arc<Ring>, indices: &[usize], start: usize, cols: usize) -> PolyMat {
    PolyMat::new(
        ring.clone(),
        indices.len(),
        cols,
        indices
            .iter()
            .flat_map(|idx| {
                (0..cols).map(move |j| Poly::constant(ring.clone(), i128::from(*idx == start + j)))
            })
            .collect(),
    )
    .unwrap()
}

#[test]
fn selector_matrices_are_the_collected_ones_in_buffers_of_their_exact_size() {
    let ring = Ring::new(1099511627917, 64).unwrap();
    // Empty parts, indices in the bounded part, in the message part (from m1 on) and past both,
    // unsorted and repeated indices, and a larger set.
    let spread: Vec<usize> = (0..70).map(|i| (i * 37) % 95).collect();
    let cases: [(&[usize], usize, usize); 8] = [
        (&[], 3, 2),
        (&[0, 1], 2, 0),
        (&[0, 1], 0, 2),
        (&[2, 0, 5, 9], 4, 3),
        (&[3, 3, 1], 4, 1),
        (&[6, 7], 5, 3),
        (&[0], 1, 1),
        (&spread, 80, 12),
    ];
    for (indices, m1, l) in cases {
        let block = selector(&ring, indices, m1, l).unwrap();
        assert_eq!(block.rows, indices.len());
        assert!(block.offset.is_none());
        let parts = [(block.s.as_ref(), 0, m1), (block.m.as_ref(), m1, l)];
        for (matrix, start, cols) in parts {
            let matrix = matrix.unwrap();
            assert_eq!(matrix, &collected(&ring, indices, start, cols));
            assert_eq!(
                matrix.capacity(),
                indices.len() * cols,
                "{indices:?} {m1} {l}"
            );
        }
    }
}
