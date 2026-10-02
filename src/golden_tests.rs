//! Pinned values: parameter transcripts, scheme fingerprints, opening-proof bytes and ring
//! arithmetic. A change to any of them changes what verifiers accept, so each must stay byte for
//! byte unless the parameters themselves change.
use crate::{
    abdlop::Abdlop,
    math::{Poly, PolyVec, Ring, int},
    params::{TboxParams, toy_d64},
};
use shake::{ExtendableOutput, Shake128, Update, XofReader};

fn digest(bytes: &[u8]) -> String {
    let mut h = Shake128::default();
    h.update(bytes);
    let mut out = [0u8; 32];
    h.finalize_xof().read(&mut out);
    hex::encode(out)
}
/// The possession set of `tests/statement.rs`.
fn possession() -> TboxParams {
    let mut p = toy_d64();
    p.id = "quadratic-possession-test-only".into();
    p.m1 = 20;
    p.m2 = 55;
    p.n_msis = 15;
    p.l = 2;
    p.alpha_squared = 192;
    p.n_bin = 0;
    p.l2_rows = vec![16, 4];
    p.n_prime = 2;
    p.linf_bound = 13;
    p.log_sigma = [14, 12, 10, 13];
    p.d_bits = 6;
    p
}
fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}
/// Canonical coefficients as 16 LE bytes each, as the pinned digests were taken.
fn canonical_bytes(p: &Poly) -> Vec<u8> {
    p.coefficients()
        .iter()
        .flat_map(|x| int::low_u128(x).to_le_bytes())
        .collect()
}

#[test]
fn parameter_transcripts_fingerprints_and_opening_proofs_are_unchanged() {
    for (p, transcript, fingerprint, prefix, length, proof_digest) in [
        (
            toy_d64(),
            "b62dab81063396d79799ad6df6fe49f97cc65b008a74945592dbffc457c55ad0",
            "5ade81f1ea9aae9d97121f8ecc4148eaeef0e9982f502d88cb2aee62bcd816f3",
            "8cdf0f0af4ad3e5dced199c44d288761f28784c005af309e8412aa490433d213",
            6870,
            "dbff8fc3d5d388003838302bd77b24b5e6aef4cf131a3c68f033694dfaa3548a",
        ),
        (
            possession(),
            "7f3cb670ea5a236803f5d7ecc2b99bbde64bf3943372479d25f3e00be1abe768",
            "485842745d59ecd6766db654dddd8923ff752970efa6dc9957269f444c7ed4d5",
            "c04b68734ec1e840dc329c5c6a532212690e16a8b798af3e12d7a69f2cec716f",
            8026,
            "edee2bbfbb48ce46595d695982a964482e306fbc9749479de6b9dd3a5f1d1646",
        ),
    ] {
        assert_eq!(digest(&p.transcript_bytes()), transcript);
        let scheme = Abdlop::new([91; 32], p).unwrap();
        assert_eq!(hex::encode(scheme.fingerprint()), fingerprint);
        let ring = scheme.ring().clone();
        let s1 = PolyVec::new(
            ring.clone(),
            vec![Poly::constant(ring.clone(), 1); scheme.bounded_len()],
        )
        .unwrap();
        let m = PolyVec::new(
            ring.clone(),
            vec![Poly::constant(ring.clone(), -3); scheme.message_len()],
        )
        .unwrap();
        let (commitment, opening) = scheme.commit_with_seed(s1, m, [92; 32]).unwrap();
        assert_eq!(
            hex::encode(scheme.prefix(&commitment, b"ctx").unwrap().digest()),
            prefix
        );
        let proof = scheme
            .prove_with_seed(&commitment, &opening, b"ctx", [93; 32])
            .unwrap();
        let bytes = scheme.encode_proof(&proof).unwrap();
        assert_eq!(
            (bytes.len(), digest(&bytes).as_str()),
            (length, proof_digest)
        );
    }
}

#[cfg(feature = "serde")]
#[test]
fn fixture_transcripts_and_fingerprints_are_unchanged() {
    for (text, transcript, fingerprint) in [
        (
            include_str!("../tests/fixtures/params/tag-preimage-crt-d64.json"),
            "98e07c478ccb67988a05c250a998e16036340a8a82afd5176d6409bed81463bd",
            "8da8ec28bc62817c166abfd71989215d2575ac13459704c7fc33e064ce4371d4",
        ),
        (
            include_str!("../tests/fixtures/params/tag-preimage-crt-d128.json"),
            "a2292d6837f647a60461a3a0b35e9e51b7cd67c4e4e5a63985a1f87c2fb77841",
            "b40027d4cd3c8e2985b3266dd73446f474cddba7f964cba3fd9b492188e8dbcf",
        ),
        (
            include_str!("../tests/fixtures/params/dense-blocks-7.json"),
            "08499e38815141151a9ecfea8c98a6e3e76bbe7f4f560b16fab77b1b37fb3375",
            "361f6e18ed65503a0753d2d3aa83ade9306e3dd3a92152a1625d9dc1bf317c7f",
        ),
        (
            include_str!("../tests/fixtures/params/decoder-ranges.json"),
            "63d830a4babab264d0a224a051682a057a671ce80038cde06771d76f4096186e",
            "aae151e9b62e9c81d98fa5248661f9e3e060dad4aebbfb04296c278f2dd815ce",
        ),
        (
            include_str!("../examples/fixtures/possession.json"),
            "7f3cb670ea5a236803f5d7ecc2b99bbde64bf3943372479d25f3e00be1abe768",
            "485842745d59ecd6766db654dddd8923ff752970efa6dc9957269f444c7ed4d5",
        ),
        (
            include_str!("params/sets/toy-d64.json"),
            "b62dab81063396d79799ad6df6fe49f97cc65b008a74945592dbffc457c55ad0",
            "5ade81f1ea9aae9d97121f8ecc4148eaeef0e9982f502d88cb2aee62bcd816f3",
        ),
    ] {
        let p = TboxParams::from_json(text).unwrap();
        assert_eq!(digest(&p.transcript_bytes()), transcript);
        let scheme = Abdlop::new([91; 32], p).unwrap();
        assert_eq!(hex::encode(scheme.fingerprint()), fingerprint);
    }
}

/// The sets of the parameter tool that the crate ships; `tests/param_sets.rs` checks that
/// their JSON files load as these.
#[test]
fn shipped_set_transcripts_and_fingerprints_are_unchanged() {
    for (p, transcript, fingerprint) in [
        (
            crate::params::kyber1024_d64(),
            "a12d1c9518970b00dab69579a75da67f3283f701c57d210caf9ce04361a622cc",
            "045fb5afcd1ab2f970be492a14d57b21e6a5dceb6309106021f4c5456f9ece78",
        ),
        (
            crate::params::kyber1024_d128(),
            "dbf89a00b6c603bc2a78c45a8218a23fe2ca2fd7eba95148a1b315bce91ed4e4",
            "0001f87db56064c48c4e3a5f7956faa670eea84a7b08ae1e465f2d3e7be375f2",
        ),
        (
            crate::params::demo_d64(),
            "56a56aa24e888c8d7f635d50205de76bd1568f401017b23c92577e90b89aa548",
            "78ff684f6f7283cbe047cdb2b1af215858b79830290fda3e8d83cb9254b5cb7e",
        ),
    ] {
        let id = p.id.clone();
        let t = digest(&p.transcript_bytes());
        let f = hex::encode(Abdlop::new([91; 32], p).unwrap().fingerprint());
        assert_eq!((t.as_str(), f.as_str()), (transcript, fingerprint), "{id}");
    }
}

/// Products, sums, differences, scaling, the automorphism and rotation on pseudorandom inputs,
/// and the number of RNS primes, for moduli from 2 to $`2^{100}-1`$ at every degree.
#[test]
fn ring_arithmetic_and_prime_selection_are_unchanged() {
    const RINGS: [(i128, usize, usize, &str); 40] = [
        (
            13,
            64,
            1,
            "728dc649178133b878ed6caf608f9f35e9d1ba495e261a979e13ec022cbcb3c0",
        ),
        (
            13,
            128,
            1,
            "fdae8d1c349093e49fe87aba82ede5018e45afb552bd6426f2884e89f3df66f6",
        ),
        (
            13,
            256,
            1,
            "0e81a4c5f016b7748b297b1fbca75a82932cd6396dec36dd20f65b8f1db40690",
        ),
        (
            13,
            512,
            1,
            "7c0594fdbde5200ba440dc7b67a130e3cb92317c6ed1df9df567409f9308f40a",
        ),
        (
            13,
            1024,
            1,
            "5779f122455cd8a13145a9625b31b37ddffe7b6e630c0a85a0eb8c44298e9d44",
        ),
        (
            1099511627917,
            64,
            2,
            "54b875fd45a9fe1b9174dc155474ada120314a118438b2f2af0ca87cc026f0a6",
        ),
        (
            1099511627917,
            128,
            2,
            "b1faa547ddcd3b06be3be3d02387d1decda603ccfa6bd71052343a2ea9b2c0d5",
        ),
        (
            1099511627917,
            256,
            2,
            "fb32bbbe53181b8654a47de62b2e3a70024d5b1b47cce1b02456de94b96b4919",
        ),
        (
            1099511627917,
            512,
            2,
            "dbf03de108b9cb13979dadfa9f9b705188d59feb2847f9db02b997e5ae4bc3e5",
        ),
        (
            1099511627917,
            1024,
            2,
            "8c4e808278ad4bc04e7726942ffff0261970b234916a29bc5a62c48141a8adc2",
        ),
        (
            302231454903657293676757,
            64,
            3,
            "82068e55ce3477ab4534ea49d30c83b57abe9d56191b91def581c59c6559fa5d",
        ),
        (
            302231454903657293676757,
            128,
            3,
            "a206aeefa249ce31c02e3e00af99ef5fe29916682050e6b02e730b38445ccff9",
        ),
        (
            302231454903657293676757,
            256,
            3,
            "8dc2302c97519149b9d5ea50b41d7442fb0315a299abeabbad1016268ef249f3",
        ),
        (
            302231454903657293676757,
            512,
            3,
            "c7dcacd2a04432a3b1351bcbe883cb8fa7bcfdd39eff201097b5e409e7ba3b25",
        ),
        (
            302231454903657293676757,
            1024,
            3,
            "92654809a536bb1397ce37f06cd20065aab3e625e5cfe514e7d22d8f04735d95",
        ),
        (
            633825300114114700748351602671,
            64,
            4,
            "654295c42429b9692d05e315bdcb6a7357112f3065469d46fe1eea0f15332cc4",
        ),
        (
            633825300114114700748351602671,
            128,
            4,
            "55d959bfc45bc64f48abb3f4ba39f5d76ebde03ece0171559c2da332e484d145",
        ),
        (
            633825300114114700748351602671,
            256,
            4,
            "e99704d15b466478111fc70846ad867a1e7e67bf9243dc0a41e0f92f60c24c9f",
        ),
        (
            633825300114114700748351602671,
            512,
            4,
            "c2b8e12d57abc8ecca2f6128b546941f6c079e66f3c84d428eac700ec3ef4082",
        ),
        (
            633825300114114700748351602671,
            1024,
            4,
            "969a7890c358624cefb3de1a484706d0e5d10797ae0d127904b6e471ac8b6db6",
        ),
        (
            1267650600228229401496703205375,
            64,
            4,
            "5f5efa2bfd16222d8e44d77a2785bd6369a7383b6e1cd948c9e3945ea7b54e2b",
        ),
        (
            1267650600228229401496703205375,
            128,
            4,
            "33b52f9457df0f38cb6bf77993578d2e1d467d358558960d9f12d96c2fde066c",
        ),
        (
            1267650600228229401496703205375,
            256,
            4,
            "d745873f7019e2c0f3b3552b4d76836f8ba4684af0b16fe2ffa2d8e01249f98e",
        ),
        (
            1267650600228229401496703205375,
            512,
            4,
            "4f8b8822666506c53c056eaaaeae1636116b04ecb3385a74ae9b9f912cab2f1e",
        ),
        (
            1267650600228229401496703205375,
            1024,
            4,
            "22b31efa2a18d8ea64f1dfaccc8553478e97b9064969044772e8aff4ab15344e",
        ),
        (
            2,
            64,
            1,
            "ebf94ab28ed09402ab5c580ae1b81f6934ea0c1a0d9ac3a7a2478044c50a0510",
        ),
        (
            2,
            128,
            1,
            "8d06ebbc3612d4b5945017bcd0ab60bd7c1915f3d7398dca298ac0c036c7c489",
        ),
        (
            2,
            256,
            1,
            "bc8ad42660e5a6bc79567046aa10c4f102c2c01ca740649323d0d3048e9555c9",
        ),
        (
            2,
            512,
            1,
            "631fa87e9851dfb23f2b8d20c07bbd38ba398e05b85077c665317599866c8b44",
        ),
        (
            2,
            1024,
            1,
            "bcc014a07f567de81070c652ca068dccb51f553c7829b2390b8bf33ccc9f092d",
        ),
        (
            3,
            64,
            1,
            "25858ac4dfe16bdd60c5e55d972b268f7b95ab24482a4034f4b5886a41165f7a",
        ),
        (
            3,
            128,
            1,
            "6b661bc0ff69be77a0780349f2da73d993fcb3e711ead4b7d738636d57fdfed0",
        ),
        (
            3,
            256,
            1,
            "efd9dbeabcea85fa60e70e4f4c8201aeaf58f7e7025aac4ad7291a7efea6f118",
        ),
        (
            3,
            512,
            1,
            "53022bc87485d21af4d5d91d81a59bf4b85a4fbb241c2e284ccc7b6091f9b2aa",
        ),
        (
            3,
            1024,
            1,
            "f6a87aaae018566c54dead3b615952e82c315f4d48489840a45bc7c64cb93633",
        ),
        (
            18446744073709551557,
            64,
            3,
            "0063881bde32bcadde8d2af68ff4d4971c6878271c3ff2ceeab208c71cee9b79",
        ),
        (
            18446744073709551557,
            128,
            3,
            "5b545e1f6e50434bd4a460ee1e5d28e5283752f57e54f1835c6bcfca7e43dc18",
        ),
        (
            18446744073709551557,
            256,
            3,
            "7bc5096a9013c952a70220a5dcee653659433fc09f8de03339a5654c02649ed6",
        ),
        (
            18446744073709551557,
            512,
            3,
            "a5194470fdc090103bb8bb81811b42d95392ce9c39ea599523821a9bc091729d",
        ),
        (
            18446744073709551557,
            1024,
            3,
            "697cb7b94f51a41e0cbf987b9f6906ecb83d6dd7e52e025b91318c82dbb331b0",
        ),
    ];
    for (q, d, primes, expected) in RINGS {
        let ring = Ring::new(q, d).unwrap();
        assert_eq!(ring.rns_primes(), primes, "q={q} d={d}");
        let mut s = 0x9e37_79b9_7f4a_7c15u64 ^ (q as u64) ^ d as u64;
        let mut rnd = || {
            (((u128::from(xorshift(&mut s)) << 64) | u128::from(xorshift(&mut s))) % q as u128)
                as i128
        };
        let a = Poly::new(ring.clone(), (0..d).map(|_| rnd()).collect()).unwrap();
        let b = Poly::new(ring.clone(), (0..d).map(|_| rnd()).collect()).unwrap();
        let mut all = canonical_bytes(&a.mul(&b).unwrap());
        all.extend(canonical_bytes(&a.add(&b).unwrap()));
        all.extend(canonical_bytes(&a.sub(&b).unwrap()));
        all.extend(canonical_bytes(&a.scale(-7).unwrap()));
        all.extend(canonical_bytes(&a.auto()));
        all.extend(canonical_bytes(&a.rotate(5)));
        assert_eq!(digest(&all), expected, "q={q} d={d}");
    }
}
