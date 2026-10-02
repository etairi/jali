//! Row-only expansion of the public matrix $`B`$: `Abdlop::extend_messages`
//! and the extra row of the quadratic proof equal the rows of the fully expanded matrices, and
//! both keep the size limit of `abdlop::matrix`.
use super::*;
use crate::params::toy_d64;

#[test]
fn extended_and_extra_rows_equal_the_full_expansion() {
    for seed in [[1u8; 32], [7; 32]] {
        let scheme = Abdlop::new(seed, toy_d64()).unwrap();
        let (l, cols) = (scheme.message_len(), scheme.a2.cols());
        let full = |rows| abdlop::matrix(scheme.ring.clone(), rows, cols, &seed, 2).unwrap();
        assert_eq!(scheme.b, full(l));
        for count in [1, 2, scheme.checked.l_ext - 1] {
            let extended = scheme.extend_messages(count).unwrap();
            assert_eq!(extended.b, full(l + count), "count {count}");
            assert_eq!(extended.message_len(), l + count);
            // The extra row of the extended scheme, as the quadratic proofs take it.
            let row = extra_row(&extended).unwrap();
            let next = full(l + count + 1);
            assert_eq!(
                row.entries(),
                &next.entries()[(l + count) * cols..],
                "count {count}"
            );
        }
        let row = extra_row(&scheme).unwrap();
        assert_eq!(row.entries(), &full(l + 1).entries()[l * cols..]);
        assert_eq!(
            scheme.extend_messages(scheme.checked.l_ext).err(),
            Some(Error::Dimension)
        );
    }
}

#[test]
fn row_expansion_keeps_the_size_limit() {
    assert_eq!(abdlop::matrix_size(1 << 10, 1 << 10), Ok(1 << 20));
    assert_eq!(
        abdlop::matrix_size((1 << 10) + 1, 1 << 10),
        Err(Error::Dimension)
    );
    assert_eq!(abdlop::matrix_size(usize::MAX, 2), Err(Error::Dimension));
}
