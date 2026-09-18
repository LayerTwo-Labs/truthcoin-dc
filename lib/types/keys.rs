use borsh::BorshSerialize;
use libes::{auth::HmacSha256, enc::Aes256Gcm, key::X25519};
use serde::{Deserialize, Serialize};
use serde_with::{DeserializeAs, DisplayFromStr, FromInto, SerializeAs};
use thiserror::Error;
use utoipa::ToSchema;

#[derive(Debug, Error)]
#[error("Wrong Bech32 HRP. Expected {expected} but decoded {decoded}")]
pub struct WrongHrpError {
    expected: bech32::Hrp,
    decoded: bech32::Hrp,
}

#[derive(Debug, Error)]
pub enum Bech32mDecodeError {
    #[error(transparent)]
    Bech32m(#[from] bech32::DecodeError),
    #[error("Invalid bytes: {}", const_hex::encode(.bytes))]
    InvalidBytes {
        bytes: [u8; 32],
        source: Box<ed25519_dalek::SignatureError>,
    },
    #[error(transparent)]
    WrongHrp(#[from] Box<WrongHrpError>),
    #[error("Wrong decoded byte length. Must decode to 32 bytes of data.")]
    WrongSize,
    #[error("Wrong Bech32 variant. Only Bech32m is accepted.")]
    WrongVariant,
}

fn borsh_serialize_x25519_pubkey<W>(
    pk: &x25519_dalek::PublicKey,
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    borsh::BorshSerialize::serialize(pk.as_bytes(), writer)
}

/// Wrapper around x25519 pubkeys
#[derive(BorshSerialize, Clone, Copy, Debug, Eq, Hash, PartialEq, ToSchema)]
#[repr(transparent)]
#[schema(value_type = String)]
pub struct EncryptionPubKey(
    #[borsh(serialize_with = "borsh_serialize_x25519_pubkey")]
    pub  x25519_dalek::PublicKey,
);

impl EncryptionPubKey {
    /// HRP for Bech32m encoding
    const BECH32M_HRP: bech32::Hrp = bech32::Hrp::parse_unchecked("ba-enc");

    /// Encode to Bech32m format
    pub fn bech32m_encode(&self) -> String {
        bech32::encode::<bech32::Bech32m>(Self::BECH32M_HRP, self.0.as_bytes())
            .expect("Bech32m Encoding should not fail")
    }

    /// Decode from Bech32m format
    pub fn bech32m_decode(s: &str) -> Result<Self, Bech32mDecodeError> {
        let (hrp, data) = bech32::decode(s)?;
        if hrp != Self::BECH32M_HRP {
            let err = WrongHrpError {
                expected: Self::BECH32M_HRP,
                decoded: hrp,
            };
            return Err(Box::new(err).into());
        }
        let Ok(bytes) = <[u8; 32]>::try_from(data) else {
            return Err(Bech32mDecodeError::WrongSize);
        };
        let res = Self::from(bytes);
        if s != res.bech32m_encode() {
            return Err(Bech32mDecodeError::WrongVariant);
        }
        Ok(res)
    }
}

impl std::fmt::Display for EncryptionPubKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.bech32m_encode().fmt(f)
    }
}

impl<T> From<T> for EncryptionPubKey
where
    x25519_dalek::PublicKey: From<T>,
{
    fn from(value: T) -> Self {
        Self(value.into())
    }
}

impl std::str::FromStr for EncryptionPubKey {
    type Err = Bech32mDecodeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::bech32m_decode(s)
    }
}

impl<'de> Deserialize<'de> for EncryptionPubKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        serde_with::IfIsHumanReadable::<
            DisplayFromStr,
            FromInto<x25519_dalek::PublicKey>,
        >::deserialize_as(deserializer)
    }
}

impl Serialize for EncryptionPubKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if serializer.is_human_readable() {
            DisplayFromStr::serialize_as(self, serializer)
        } else {
            self.0.serialize(serializer)
        }
    }
}

fn borsh_serialize_ed25519_vk<W>(
    vk: &ed25519_dalek::VerifyingKey,
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    borsh::BorshSerialize::serialize(vk.as_bytes(), writer)
}

/// Wrapper around x25519 pubkeys
#[derive(BorshSerialize, Clone, Copy, Debug, Eq, Hash, PartialEq, ToSchema)]
#[repr(transparent)]
#[schema(value_type = String)]
pub struct VerifyingKey(
    #[borsh(serialize_with = "borsh_serialize_ed25519_vk")]
    pub  ed25519_dalek::VerifyingKey,
);

impl VerifyingKey {
    /// HRP for Bech32m encoding
    const BECH32M_HRP: bech32::Hrp = bech32::Hrp::parse_unchecked("ba-svk");

    const BYTE_LEN: usize = ed25519_dalek::PUBLIC_KEY_LENGTH;

    /// Encode to Bech32m format
    pub fn bech32m_encode(&self) -> String {
        bech32::encode::<bech32::Bech32m>(Self::BECH32M_HRP, self.0.as_bytes())
            .expect("Bech32m Encoding should not fail")
    }

    /// Decode from Bech32m format
    pub fn bech32m_decode(s: &str) -> Result<Self, Bech32mDecodeError> {
        let (hrp, data) = bech32::decode(s)?;
        if hrp != Self::BECH32M_HRP {
            let err = WrongHrpError {
                expected: Self::BECH32M_HRP,
                decoded: hrp,
            };
            return Err(Box::new(err).into());
        }
        let Ok(bytes) = <[u8; Self::BYTE_LEN]>::try_from(data) else {
            return Err(Bech32mDecodeError::WrongSize);
        };
        let res = match ed25519_dalek::VerifyingKey::from_bytes(&bytes) {
            Ok(vk) => Self(vk),
            Err(err) => {
                let err = Bech32mDecodeError::InvalidBytes {
                    bytes,
                    source: Box::new(err),
                };
                return Err(err);
            }
        };
        if s != res.bech32m_encode() {
            return Err(Bech32mDecodeError::WrongVariant);
        }
        Ok(res)
    }

    pub fn to_bytes(&self) -> [u8; Self::BYTE_LEN] {
        self.0.to_bytes()
    }
}

impl std::fmt::Display for VerifyingKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.bech32m_encode().fmt(f)
    }
}

impl From<ed25519_dalek::VerifyingKey> for VerifyingKey {
    fn from(vk: ed25519_dalek::VerifyingKey) -> Self {
        Self(vk)
    }
}

impl From<VerifyingKey> for ed25519_dalek::VerifyingKey {
    fn from(vk: VerifyingKey) -> Self {
        vk.0
    }
}

impl TryFrom<&[u8; VerifyingKey::BYTE_LEN]> for VerifyingKey {
    type Error = ed25519_dalek::SignatureError;

    fn try_from(
        bytes: &[u8; VerifyingKey::BYTE_LEN],
    ) -> Result<Self, Self::Error> {
        ed25519_dalek::VerifyingKey::from_bytes(bytes).map(Self)
    }
}

impl std::str::FromStr for VerifyingKey {
    type Err = Bech32mDecodeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::bech32m_decode(s)
    }
}

impl<'de> Deserialize<'de> for VerifyingKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        serde_with::IfIsHumanReadable::<
            DisplayFromStr,
            FromInto<ed25519_dalek::VerifyingKey>,
        >::deserialize_as(deserializer)
    }
}

impl Serialize for VerifyingKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serde_with::IfIsHumanReadable::<
            DisplayFromStr,
            FromInto<ed25519_dalek::VerifyingKey>,
        >::serialize_as(self, serializer)
    }
}

/// ECIES crypto scheme over x25519
pub type Ecies = libes::Ecies<X25519, Aes256Gcm, HmacSha256>;
