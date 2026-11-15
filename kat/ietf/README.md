# Vectors of the CFRG drafts

`fiat-shamir-04.json` holds the test vectors of draft-irtf-cfrg-fiat-shamir-04 (Orrù and Yun,
Crypto Forum Research Group, 2 November 2026), extracted from the text form of the draft by
`tools/kat/ietf_vectors.py`. The JSON records the SHA-256 of the text it was extracted from.
`src/fiat_shamir/tests.rs` runs the SHAKE128 vectors and the draft's sumcheck example against
the crate's duplex sponge and asserts how many vectors it runs.

```sh
curl -O https://www.ietf.org/archive/id/draft-irtf-cfrg-fiat-shamir-04.txt
python3 tools/kat/ietf_vectors.py draft-irtf-cfrg-fiat-shamir-04.txt \
    --output kat/ietf/fiat-shamir-04.json
```

The vectors come from an IETF document: copyright (c) 2026 IETF Trust and the persons
identified as the document authors, subject to BCP 78 and the IETF Trust's Legal Provisions
Relating to IETF Documents (https://trustee.ietf.org/license-info). They are reproduced here
for testing an implementation of the draft and are not covered by this crate's MIT license.
