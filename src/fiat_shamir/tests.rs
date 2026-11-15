//! The vectors of draft-irtf-cfrg-fiat-shamir-04, Appendix C (`kat/ietf/fiat-shamir-04.json`,
//! extracted by `tools/kat/ietf_vectors.py`), for the suite SHAKE128 and the suite-independent
//! ones, and the sumcheck example of its Appendix B against its NARG vectors. The TurboSHAKE128
//! vectors and the big-endian field serialization are outside this crate.
use super::*;
use serde_json::Value;

const VECTORS: &str = include_str!("../../kat/ietf/fiat-shamir-04.json");

fn vectors() -> Vec<Value> {
    let doc: Value = serde_json::from_str(VECTORS).unwrap();
    doc["vectors"].as_array().unwrap().clone()
}

fn field<'a>(v: &'a Value, key: &str) -> &'a str {
    v[key]
        .as_str()
        .unwrap_or_else(|| panic!("{}: no {key}", v["Id"]))
}

fn bytes(s: &str) -> Vec<u8> {
    if s == "\"\"" {
        Vec::new()
    } else {
        hex::decode(s).unwrap()
    }
}

/// An integer written in decimal or with the prefix `0x`.
fn int(s: &str) -> U256 {
    match s.strip_prefix("0x") {
        Some(h) => U256::from_be_hex(&format!("{h:0>64}")),
        None => U256::from_u128(s.parse().unwrap()),
    }
}

/// A value below 2^64 as `u64`, whatever the limb width.
fn small(x: &U256) -> u64 {
    let le = x.to_le_bytes();
    assert!(le[8..].iter().all(|&b| b == 0), "not below 2^64");
    u64::from_le_bytes(le[..8].try_into().unwrap())
}

fn session_id(v: &Value) -> [u8; 32] {
    bytes(field(v, "SessionId")).try_into().unwrap()
}

/// Runs the `Operations` of a vector on the sponge, returning every squeezed byte.
fn run(sponge: &mut DuplexSponge, v: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    for op in v["Operations"].as_array().unwrap() {
        let op = op.as_str().unwrap();
        if let Some(x) = op.strip_prefix("absorb ") {
            sponge.absorb(&bytes(x));
        } else if let Some(n) = op.strip_prefix("squeeze ") {
            let mut buf = vec![0u8; n.parse().unwrap()];
            sponge.squeeze(&mut buf);
            out.extend_from_slice(&buf);
        } else {
            panic!("unknown operation {op}");
        }
    }
    out
}

/// Whether a vector is in scope: suite SHAKE128 or no suite, and the default byte order.
fn in_scope(v: &Value) -> bool {
    v.get("Suite").is_none_or(|s| s == "SHAKE128") && v.get("ByteOrder").is_none()
}

/// The sumcheck example of Appendix B over the prime `p < 2^32`.
mod sumcheck {
    use super::*;

    fn le4(x: u64) -> [u8; 4] {
        u32::try_from(x).unwrap().to_le_bytes()
    }

    /// `r := LE2IP(state.Squeeze(Ns)) mod p` with `Ns = 4`.
    fn challenge(state: &mut DuplexSponge, p: u64) -> u64 {
        u64::from(u32::from_le_bytes(state.squeeze_array::<4>())) % p
    }

    fn start(session_id: &[u8; 32], v: u32, s: u64) -> DuplexSponge {
        let mut state = DuplexSponge::new(session_id);
        let mut instance = v.to_le_bytes().to_vec();
        instance.extend_from_slice(&le4(s));
        state.absorb(&instance);
        state
    }

    /// `SumcheckProve`: the NARG string and the final evaluation.
    pub(super) fn prove(session_id: &[u8; 32], p: u64, v: u32, w: &[u64]) -> (Vec<u8>, u64) {
        let mut w: Vec<u64> = w.iter().map(|x| x % p).collect();
        let s = w.iter().fold(0, |a, x| (a + x) % p);
        let mut state = start(session_id, v, s);
        let mut narg = Vec::new();
        for _ in 0..v {
            let a0 = w.iter().step_by(2).fold(0, |a, x| (a + x) % p);
            let odd = w.iter().skip(1).step_by(2).fold(0, |a, x| (a + x) % p);
            let a1 = (odd + p - a0) % p;
            let mut msg = le4(a0).to_vec();
            msg.extend_from_slice(&le4(a1));
            state.absorb(&msg);
            narg.extend_from_slice(&msg);
            let r = challenge(&mut state, p);
            w = w
                .chunks(2)
                .map(|c| (c[0] + r * ((c[1] + p - c[0]) % p)) % p)
                .collect();
        }
        (narg, w[0])
    }

    /// `SumcheckVerify`, with the reason of a rejection.
    pub(super) fn verify(
        session_id: &[u8; 32],
        p: u64,
        v: u32,
        claimed: u64,
        narg: &[u8],
        y: u64,
    ) -> Result<(), &'static str> {
        let modulus = U256::from_u64(p);
        let mut state = start(session_id, v, claimed);
        let (mut s, mut rest) = (claimed, narg);
        for _ in 0..v {
            let (a0, r1) = deserialize_uint(rest, &modulus).ok_or("deserialization")?;
            let (a1, r2) = deserialize_uint(r1, &modulus).ok_or("deserialization")?;
            rest = r2;
            let (a0, a1) = (small(&a0), small(&a1));
            if (2 * a0 + a1) % p != s {
                return Err("round sum");
            }
            let mut msg = le4(a0).to_vec();
            msg.extend_from_slice(&le4(a1));
            state.absorb(&msg);
            let r = challenge(&mut state, p);
            s = (a0 + a1 * r) % p;
        }
        if !rest.is_empty() {
            return Err("trailing bytes");
        }
        if s != y {
            return Err("final evaluation");
        }
        Ok(())
    }
}

#[test]
fn the_vectors_of_the_draft() {
    let all = vectors();
    let by_id = |id: &str| all.iter().find(|v| v["Id"] == id).unwrap().clone();
    let mut ran = 0;
    for v in all.iter().filter(|v| in_scope(v)) {
        let id = field(v, "Id");
        match field(v, "Function") {
            "DuplexSponge" => {
                let out = run(&mut DuplexSponge::new(&session_id(v)), v);
                assert_eq!(hex::encode(out), field(v, "Output"), "{id}");
            }
            "DeriveSessionID" => {
                let sid = derive_session_id(&bytes(field(v, "Tag")));
                assert_eq!(hex::encode(sid), field(v, "SessionId"), "{id}");
            }
            "DecodeUint" => {
                let m = int(field(v, "Modulus"));
                let buf = if v.get("Operations").is_some() {
                    let out = run(&mut DuplexSponge::new(&session_id(v)), v);
                    assert_eq!(hex::encode(&out), field(v, "Output"), "{id}");
                    out
                } else {
                    bytes(field(v, "Input"))
                };
                assert_eq!(ns(&m).unwrap() + 16, buf.len(), "{id}");
                let got = decode_uint(&buf, &m).unwrap();
                assert_eq!(got, int(field(v, "VerifierMessage")), "{id}");
            }
            "SerializeVarLenString" => {
                let out = serialize_varlen(&bytes(field(v, "Input"))).unwrap();
                assert_eq!(hex::encode(out), field(v, "Output"), "{id}");
            }
            "SerializeUint" => {
                let out = serialize_uint(&int(field(v, "Value")), &int(field(v, "Modulus")));
                assert_eq!(hex::encode(out.unwrap()), field(v, "Output"), "{id}");
            }
            "DeserializeVarLenString" => {
                assert_eq!(field(v, "Expected"), "reject", "{id}");
                assert!(
                    deserialize_varlen(&bytes(field(v, "Input"))).is_none(),
                    "{id}"
                );
            }
            "DeserializeUint" => {
                assert_eq!(field(v, "Expected"), "reject", "{id}");
                let m = int(field(v, "Modulus"));
                assert!(
                    deserialize_uint(&bytes(field(v, "Input")), &m).is_none(),
                    "{id}"
                );
            }
            "DeserializeField" => {
                // coordinate by coordinate, every one canonical
                assert_eq!(field(v, "Expected"), "reject", "{id}");
                let m = int(field(v, "Modulus"));
                let degree: usize = field(v, "ExtensionDegree").parse().unwrap();
                let input = bytes(field(v, "Input"));
                let mut rest = input.as_slice();
                let mut ok = true;
                for _ in 0..degree {
                    match deserialize_uint(rest, &m) {
                        Some((_, r)) => rest = r,
                        None => ok = false,
                    }
                }
                assert!(!ok, "{id}");
            }
            "Sumcheck" => {
                let p = small(&int(field(v, "Modulus")));
                let nv: u32 = field(v, "NumVariables").parse().unwrap();
                let claimed = small(&int(field(v, "ClaimedSum")));
                let sid = session_id(v);
                if let Some(tag) = v.get("Tag") {
                    let tag = bytes(tag.as_str().unwrap());
                    assert_eq!(derive_session_id(&tag), sid, "{id}");
                }
                if field(v, "Expected") == "accept" {
                    let w: Vec<u64> = v["Witness"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_str().unwrap().parse().unwrap())
                        .collect();
                    let (narg, y) = sumcheck::prove(&sid, p, nv, &w);
                    assert_eq!(hex::encode(&narg), field(v, "NargString"), "{id}");
                    assert_eq!(U256::from_u64(y), int(field(v, "FinalEvaluation")), "{id}");
                    assert_eq!(sumcheck::verify(&sid, p, nv, claimed, &narg, y), Ok(()));
                } else {
                    let narg = bytes(field(v, "NargString"));
                    // the baseline's final evaluation, so that only the vector's own defect fails
                    let y = v
                        .get("BaseId")
                        .map(|b| int(field(&by_id(b.as_str().unwrap()), "FinalEvaluation")))
                        .map_or(0, |y| small(&y));
                    let reason = if id.ends_with("trailing_bytes") {
                        "trailing bytes"
                    } else {
                        "deserialization"
                    };
                    assert_eq!(
                        sumcheck::verify(&sid, p, nv, claimed, &narg, y),
                        Err(reason),
                        "{id}"
                    );
                }
            }
            other => panic!("{id}: no runner for {other}"),
        }
        ran += 1;
    }
    // every vector ran but the 12 of TurboSHAKE128 and the big-endian serialization
    let turbo = all
        .iter()
        .filter(|v| v.get("Suite").is_some_and(|s| s == "TurboSHAKE128"));
    assert_eq!((all.len(), turbo.count()), (35, 12));
    assert_eq!(ran, 35 - 12 - 1, "vectors run");
}

#[test]
fn absorbing_in_pieces_and_squeezing_in_pieces_are_one_stream() {
    let sid = [7u8; 32];
    let mut a = DuplexSponge::new(&sid);
    a.absorb(b"abc");
    a.absorb(b"");
    a.absorb(b"def");
    let mut b = DuplexSponge::new(&sid);
    b.absorb(b"abcdef");
    let whole: [u8; 300] = b.squeeze_array();
    let mut pieces = Vec::new();
    for n in [1usize, 167, 0, 2, 130] {
        let mut buf = vec![0u8; n];
        a.squeeze(&mut buf);
        pieces.extend_from_slice(&buf);
    }
    assert_eq!(pieces, whole);
    // a clone continues the stream where its original stood
    let mut c = a.clone();
    assert_eq!(c.squeeze_array::<16>(), a.squeeze_array::<16>());
}

#[test]
fn the_codecs_refuse_what_they_cannot_represent() {
    let m = U256::from_u64(1_000_003);
    assert_eq!(ns(&m), Some(3));
    assert_eq!(ns(&U256::from_u64(1 << 24)), Some(3));
    assert_eq!(ns(&U256::from_u64((1 << 24) + 1)), Some(4));
    assert_eq!(ns(&U256::ONE), None);
    assert_eq!(serialize_uint(&m, &m), None);
    let x = U256::from_u64(999_999);
    let s = serialize_uint(&x, &m).unwrap();
    assert_eq!(deserialize_uint(&s, &m), Some((x, &[][..])));
    assert_eq!(decode_uint(&[0u8; 65], &m), None);
    assert_eq!(decode_uint(&[0xff; 19], &m), {
        // (2^152 - 1) mod 1,000,003, as an integer computation
        let big = num_bigint::BigUint::from_bytes_le(&[0xff; 19]) % 1_000_003u32;
        Some(U256::from_u64(big.try_into().unwrap()))
    });
    assert_eq!(deserialize_varlen(&[1, 0, 0]), None);
    assert_eq!(
        deserialize_varlen(&[1, 0, 0, 0, 9, 8]),
        Some((&[9u8][..], &[8u8][..]))
    );
}
