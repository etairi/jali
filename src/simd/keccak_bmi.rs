//! Keccak-f[1600] (FIPS 202) compiled for BMI1 and BMI2, whose `andn` and `rorx` take the
//! complements and rotations of the scalar code without extra moves: every step of a round
//! unrolled, the 25 lanes in locals, lane $`i`$ the word $`x+5y=i`$. The `keccak` crate's
//! permutation is the reference (`super::tests`).

/// The round constants of Keccak-f[1600], round 0 first.
const RC: [u64; 24] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808a,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808b,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008a,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000a,
    0x0000_0000_8000_808b,
    0x8000_0000_0000_008b,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800a,
    0x8000_0000_8000_000a,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];

/// The permutation of `state`: theta's column parities `c` and their combinations `d`, rho
/// and pi into `b` (lane $`(x,y)`$ to $`(y,2x+3y)`$, rotated by its offset), chi and iota.
#[target_feature(enable = "bmi1,bmi2")]
pub(super) fn f1600(state: &mut [u64; 25]) {
    let [
        mut a0,
        mut a1,
        mut a2,
        mut a3,
        mut a4,
        mut a5,
        mut a6,
        mut a7,
        mut a8,
        mut a9,
        mut a10,
        mut a11,
        mut a12,
        mut a13,
        mut a14,
        mut a15,
        mut a16,
        mut a17,
        mut a18,
        mut a19,
        mut a20,
        mut a21,
        mut a22,
        mut a23,
        mut a24,
    ] = *state;
    for rc in RC {
        let c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20;
        let c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21;
        let c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22;
        let c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23;
        let c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24;
        let d0 = c4 ^ c1.rotate_left(1);
        let d1 = c0 ^ c2.rotate_left(1);
        let d2 = c1 ^ c3.rotate_left(1);
        let d3 = c2 ^ c4.rotate_left(1);
        let d4 = c3 ^ c0.rotate_left(1);
        let b0 = a0 ^ d0;
        let b1 = (a6 ^ d1).rotate_left(44);
        let b2 = (a12 ^ d2).rotate_left(43);
        let b3 = (a18 ^ d3).rotate_left(21);
        let b4 = (a24 ^ d4).rotate_left(14);
        let b5 = (a3 ^ d3).rotate_left(28);
        let b6 = (a9 ^ d4).rotate_left(20);
        let b7 = (a10 ^ d0).rotate_left(3);
        let b8 = (a16 ^ d1).rotate_left(45);
        let b9 = (a22 ^ d2).rotate_left(61);
        let b10 = (a1 ^ d1).rotate_left(1);
        let b11 = (a7 ^ d2).rotate_left(6);
        let b12 = (a13 ^ d3).rotate_left(25);
        let b13 = (a19 ^ d4).rotate_left(8);
        let b14 = (a20 ^ d0).rotate_left(18);
        let b15 = (a4 ^ d4).rotate_left(27);
        let b16 = (a5 ^ d0).rotate_left(36);
        let b17 = (a11 ^ d1).rotate_left(10);
        let b18 = (a17 ^ d2).rotate_left(15);
        let b19 = (a23 ^ d3).rotate_left(56);
        let b20 = (a2 ^ d2).rotate_left(62);
        let b21 = (a8 ^ d3).rotate_left(55);
        let b22 = (a14 ^ d4).rotate_left(39);
        let b23 = (a15 ^ d0).rotate_left(41);
        let b24 = (a21 ^ d1).rotate_left(2);
        a0 = b0 ^ (!b1 & b2);
        a1 = b1 ^ (!b2 & b3);
        a2 = b2 ^ (!b3 & b4);
        a3 = b3 ^ (!b4 & b0);
        a4 = b4 ^ (!b0 & b1);
        a5 = b5 ^ (!b6 & b7);
        a6 = b6 ^ (!b7 & b8);
        a7 = b7 ^ (!b8 & b9);
        a8 = b8 ^ (!b9 & b5);
        a9 = b9 ^ (!b5 & b6);
        a10 = b10 ^ (!b11 & b12);
        a11 = b11 ^ (!b12 & b13);
        a12 = b12 ^ (!b13 & b14);
        a13 = b13 ^ (!b14 & b10);
        a14 = b14 ^ (!b10 & b11);
        a15 = b15 ^ (!b16 & b17);
        a16 = b16 ^ (!b17 & b18);
        a17 = b17 ^ (!b18 & b19);
        a18 = b18 ^ (!b19 & b15);
        a19 = b19 ^ (!b15 & b16);
        a20 = b20 ^ (!b21 & b22);
        a21 = b21 ^ (!b22 & b23);
        a22 = b22 ^ (!b23 & b24);
        a23 = b23 ^ (!b24 & b20);
        a24 = b24 ^ (!b20 & b21);
        a0 ^= rc;
    }
    *state = [
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
        a20, a21, a22, a23, a24,
    ];
}
