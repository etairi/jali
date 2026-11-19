//! The byte kernel of LaBRADOR's projection on AVX-512 registers of 64 bytes, one register for
//! each entry position of the 256 rows: lane $`g`$ of register $`e`$ is row $`4g+e`$, whose
//! entry in a column is bit $`2e`$ of byte $`g`$ less bit $`2e+1`$ (the decoding of
//! `labrador::sample::Projection`). Masks from bit tests select the lanes that add or subtract
//! the column's coefficient, so that each column takes eight masked operations.
use core::arch::x86_64::*;

/// `lanes[e][g] += s` where bit $`2e`$ of `column[g]` is set and `lanes[e][g] -= s` where bit
/// $`2e+1`$ is, for each column of `columns` and its coefficient $`s`$ in `values`, in wrapping
/// 8-bit arithmetic: the update of `level::project_bytes`, exact while every partial sum stays
/// within $`\pm127`$ (the caller adds the lanes into wider sums often enough).
#[target_feature(enable = "avx512f,avx512bw")]
pub(super) fn project_bytes(lanes: &mut [[i8; 64]; 4], columns: &[[u8; 64]], values: &[i8]) {
    // SAFETY: each load and store reads or writes the 64 bytes of one array, unaligned.
    let mut acc = unsafe {
        [
            _mm512_loadu_si512(lanes[0].as_ptr().cast()),
            _mm512_loadu_si512(lanes[1].as_ptr().cast()),
            _mm512_loadu_si512(lanes[2].as_ptr().cast()),
            _mm512_loadu_si512(lanes[3].as_ptr().cast()),
        ]
    };
    let plus = [
        _mm512_set1_epi8(1),
        _mm512_set1_epi8(4),
        _mm512_set1_epi8(16),
        _mm512_set1_epi8(64),
    ];
    let minus = [
        _mm512_set1_epi8(2),
        _mm512_set1_epi8(8),
        _mm512_set1_epi8(32),
        _mm512_set1_epi8(-128),
    ];
    for (column, s) in columns.iter().zip(values) {
        // SAFETY: as above, the 64 bytes of the column.
        let c = unsafe { _mm512_loadu_si512(column.as_ptr().cast()) };
        let s = _mm512_set1_epi8(*s);
        for e in 0..4 {
            let p = _mm512_test_epi8_mask(c, plus[e]);
            let m = _mm512_test_epi8_mask(c, minus[e]);
            acc[e] = _mm512_mask_add_epi8(acc[e], p, acc[e], s);
            acc[e] = _mm512_mask_sub_epi8(acc[e], m, acc[e], s);
        }
    }
    for (lane, a) in lanes.iter_mut().zip(acc) {
        // SAFETY: as above.
        unsafe { _mm512_storeu_si512(lane.as_mut_ptr().cast(), a) };
    }
}
