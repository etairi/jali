//! Serde forms of integer fields that can exceed $`2^{64}`$ (prime factors and bounds).
//!
//! In human-readable formats such as JSON, writers emit a number below $`2^{64}`$ and a decimal
//! string otherwise. Readers accept both: a nonnegative integer below $`2^{64}`$, or a string of
//! ASCII digits without sign or leading zeros. A number from $`2^{64}`$ on is refused, since a
//! JSON parser may round it to a float. Out-of-range values are refused.
//!
//! Reading a number or a string needs `deserialize_any`, which formats that do not describe
//! themselves, such as postcard or bincode, do not support. Formats that are not human-readable
//! therefore use fixed forms: `u64` and `u128` fields are the format's own integers, and a
//! `U256` is 32 little-endian bytes.
use crate::math::U256;
use crypto_bigint::CheckedAdd;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

/// Decimal digits of a 256-bit value.
pub(crate) fn to_decimal(x: &U256) -> String {
    if *x == U256::ZERO {
        return "0".into();
    }
    let mut digits = Vec::new();
    let mut rest = *x;
    let ten = crypto_bigint::NonZero::new(crypto_bigint::Limb(10)).expect("ten");
    while rest != U256::ZERO {
        let (quotient, digit) = rest.div_rem_limb(ten);
        digits.push(b'0' + digit.0 as u8);
        rest = quotient;
    }
    digits.reverse();
    String::from_utf8(digits).expect("ASCII digits")
}

/// Parse the canonical decimal form: digits only, no sign, no leading zeros.
pub(crate) fn from_decimal(text: &str) -> Option<U256> {
    let bytes = text.as_bytes();
    if bytes.is_empty() || (bytes.len() > 1 && bytes[0] == b'0') {
        return None;
    }
    let ten = U256::from_u8(10);
    let mut x = U256::ZERO;
    for b in bytes {
        if !b.is_ascii_digit() {
            return None;
        }
        x = Option::<U256>::from(x.checked_mul(&ten))?;
        x = Option::<U256>::from(x.checked_add(&U256::from_u8(b - b'0')))?;
    }
    Some(x)
}

fn write<S: Serializer>(x: &U256, s: S) -> Result<S::Ok, S::Error> {
    if !s.is_human_readable() {
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&x.to_le_bytes());
        return bytes.serialize(s);
    }
    match crate::math::int::to_u64(x) {
        Some(small) => s.serialize_u64(small),
        None => s.serialize_str(&to_decimal(x)),
    }
}

struct Visitor;
impl de::Visitor<'_> for Visitor {
    type Value = U256;
    fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("a nonnegative integer below 2^64 or a decimal string")
    }
    fn visit_u64<E: de::Error>(self, v: u64) -> Result<U256, E> {
        Ok(U256::from_u64(v))
    }
    fn visit_str<E: de::Error>(self, v: &str) -> Result<U256, E> {
        from_decimal(v).ok_or_else(|| E::custom("not a canonical decimal integer below 2^256"))
    }
}
fn read<'de, D: Deserializer<'de>>(d: D) -> Result<U256, D::Error> {
    if !d.is_human_readable() {
        return <[u8; 32]>::deserialize(d).map(|bytes| U256::from_le_slice(&bytes));
    }
    d.deserialize_any(Visitor)
}

/// A `U256` field.
pub(crate) mod u256 {
    use super::*;
    pub(crate) fn serialize<S: Serializer>(x: &U256, s: S) -> Result<S::Ok, S::Error> {
        write(x, s)
    }
    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<U256, D::Error> {
        read(d)
    }
}

/// A `Vec<U256>` field.
pub(crate) mod u256_list {
    use super::*;
    use serde::ser::SerializeSeq;
    pub(crate) fn serialize<S: Serializer>(x: &[U256], s: S) -> Result<S::Ok, S::Error> {
        struct Item<'a>(&'a U256);
        impl serde::Serialize for Item<'_> {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                write(self.0, s)
            }
        }
        let mut seq = s.serialize_seq(Some(x.len()))?;
        for value in x {
            seq.serialize_element(&Item(value))?;
        }
        seq.end()
    }
    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<U256>, D::Error> {
        struct Item(U256);
        impl<'de> serde::Deserialize<'de> for Item {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                read(d).map(Item)
            }
        }
        let items: Vec<Item> = serde::Deserialize::deserialize(d)?;
        Ok(items.into_iter().map(|x| x.0).collect())
    }
}

/// An `Option<U256>` field: `null` or the form of [`u256`] in JSON, and the format's option of
/// 32 bytes otherwise.
pub(crate) mod u256_option {
    use super::*;
    pub(crate) fn serialize<S: Serializer>(x: &Option<U256>, s: S) -> Result<S::Ok, S::Error> {
        match x {
            None => s.serialize_none(),
            Some(x) => s.serialize_some(&U256Field(x)),
        }
    }
    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<U256>, D::Error> {
        struct Item(U256);
        impl<'de> serde::Deserialize<'de> for Item {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                read(d).map(Item)
            }
        }
        Ok(Option::<Item>::deserialize(d)?.map(|x| x.0))
    }
}

fn narrow<T: TryFrom<u128>, E: de::Error>(x: U256) -> Result<T, E> {
    crate::math::int::to_u128(&x)
        .and_then(|x| T::try_from(x).ok())
        .ok_or_else(|| E::custom("integer out of range"))
}

/// A `u64` field: written as a number, read from a number or a decimal string.
pub(crate) mod u64_field {
    use super::*;
    pub(crate) fn serialize<S: Serializer>(x: &u64, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(*x)
    }
    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<u64, D::Error> {
        if !d.is_human_readable() {
            return u64::deserialize(d);
        }
        narrow(read(d)?)
    }
}

/// A `Vec<u64>` field, as [`u64_field`].
pub(crate) mod u64_list {
    use super::*;
    pub(crate) fn serialize<S: Serializer>(x: &[u64], s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(x)
    }
    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u64>, D::Error> {
        if !d.is_human_readable() {
            return Vec::deserialize(d);
        }
        let values = super::u256_list::deserialize(d)?;
        values.into_iter().map(narrow).collect()
    }
}

/// The forms above as values, for a hand-written `Serialize`: a `u128`, an `Option<u128>`, a
/// list of `u64` and a `U256`.
pub(crate) struct U128Field<'a>(pub(crate) &'a u128);
pub(crate) struct U128Option<'a>(pub(crate) &'a Option<u128>);
pub(crate) struct U64List<'a>(pub(crate) &'a [u64]);
pub(crate) struct U256Field<'a>(pub(crate) &'a U256);
impl Serialize for U128Field<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        u128_field::serialize(self.0, s)
    }
}
impl Serialize for U128Option<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        u128_option::serialize(self.0, s)
    }
}
impl Serialize for U64List<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        u64_list::serialize(self.0, s)
    }
}
impl Serialize for U256Field<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        write(self.0, s)
    }
}

/// A `u128` field: a number below $`2^{64}`$, a decimal string otherwise.
pub(crate) mod u128_field {
    use super::*;
    pub(crate) fn serialize<S: Serializer>(x: &u128, s: S) -> Result<S::Ok, S::Error> {
        if !s.is_human_readable() {
            return s.serialize_u128(*x);
        }
        write(&U256::from_u128(*x), s)
    }
    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<u128, D::Error> {
        if !d.is_human_readable() {
            return u128::deserialize(d);
        }
        narrow(read(d)?)
    }
}

/// An `Option<u128>` field: `null` or the form of [`u128_field`] in JSON, and the format's option
/// of a `u128` otherwise. A struct that leaves `None` out of JSON writes it by hand.
pub(crate) mod u128_option {
    use super::*;
    pub(crate) fn serialize<S: Serializer>(x: &Option<u128>, s: S) -> Result<S::Ok, S::Error> {
        match x {
            None => s.serialize_none(),
            Some(x) => s.serialize_some(&U128Field(x)),
        }
    }
    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<u128>, D::Error> {
        struct Item(u128);
        impl<'de> Deserialize<'de> for Item {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                u128_field::deserialize(d).map(Item)
            }
        }
        Ok(Option::<Item>::deserialize(d)?.map(|x| x.0))
    }
}
