use std::net::{SocketAddrV4, SocketAddrV6};

use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};
use utoipa::{
    PartialSchema, ToSchema,
    openapi::{RefOr, Schema},
};

use crate::types::{EncryptionPubKey, Hash, VerifyingKey};

#[derive(
    BorshSerialize,
    Clone,
    Debug,
    Default,
    Deserialize,
    Eq,
    Hash,
    PartialEq,
    Serialize,
    ToSchema,
)]
#[cfg_attr(feature = "clap", derive(clap::Args))]
#[cfg_attr(feature = "clap", group(required = false, multiple = true))]
pub struct TruthcoinData {
    /// Commitment to arbitrary data
    #[cfg_attr(feature = "clap", arg(
        long,
        value_parser = |s: &str| <Hash as hex::FromHex>::from_hex(s)
    ))]
    #[schema(value_type = Option<String>)]
    pub commitment: Option<Hash>,
    /// Optional ipv4 addr
    #[cfg_attr(feature = "clap", arg(long))]
    #[schema(value_type = Option<String>)]
    pub socket_addr_v4: Option<SocketAddrV4>,
    /// Optional ipv6 addr
    #[cfg_attr(feature = "clap", arg(long))]
    #[schema(value_type = Option<String>)]
    pub socket_addr_v6: Option<SocketAddrV6>,
    /// Optional pubkey used for encryption
    #[cfg_attr(feature = "clap", arg(long))]
    pub encryption_pubkey: Option<EncryptionPubKey>,
    /// Optional pubkey used for signing messages
    #[cfg_attr(feature = "clap", arg(long))]
    pub signing_pubkey: Option<VerifyingKey>,
}

/// Delete, retain, or set a value
#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize)]
pub enum Update<T> {
    Delete,
    Retain,
    Set(T),
}

impl<T> Update<T> {
    /// Create a schema from a schema for `T`.
    fn schema(schema_t: RefOr<Schema>) -> RefOr<Schema> {
        let schema_delete = utoipa::openapi::ObjectBuilder::new()
            .schema_type(utoipa::openapi::Type::String)
            .enum_values(Some(["Delete"]));
        let schema_retain = utoipa::openapi::ObjectBuilder::new()
            .schema_type(utoipa::openapi::Type::String)
            .enum_values(Some(["Retain"]));
        let schema_set = utoipa::openapi::ObjectBuilder::new()
            .property("Set", schema_t)
            .required("Set");
        let schema = utoipa::openapi::OneOfBuilder::new()
            .item(schema_delete)
            .item(schema_retain)
            .item(schema_set)
            .build()
            .into();
        RefOr::T(schema)
    }
}

impl PartialSchema for Update<Hash> {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        Self::schema(<String as PartialSchema>::schema())
    }
}

impl ToSchema for Update<Hash> {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("UpdateHash")
    }
}

impl PartialSchema for Update<SocketAddrV4> {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        Self::schema(<String as PartialSchema>::schema())
    }
}

impl ToSchema for Update<SocketAddrV4> {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("UpdateSocketAddrV4")
    }
}

impl PartialSchema for Update<SocketAddrV6> {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        Self::schema(<String as PartialSchema>::schema())
    }
}

impl ToSchema for Update<SocketAddrV6> {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("UpdateSocketAddrV6")
    }
}

impl PartialSchema for Update<EncryptionPubKey> {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        Self::schema(<String as PartialSchema>::schema())
    }
}

impl ToSchema for Update<EncryptionPubKey> {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("UpdateEncryptionPubKey")
    }
}

impl PartialSchema for Update<VerifyingKey> {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        Self::schema(<String as PartialSchema>::schema())
    }
}

impl ToSchema for Update<VerifyingKey> {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("UpdateVerifyingKey")
    }
}

impl PartialSchema for Update<u64> {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        Self::schema(<u64 as PartialSchema>::schema())
    }
}

impl ToSchema for Update<u64> {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("UpdateU64")
    }
}

/// Updates to the data associated with a Truthcoin
#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct TruthcoinDataUpdates {
    /// Commitment to arbitrary data
    #[schema(schema_with = <Update<Hash> as PartialSchema>::schema)]
    pub commitment: Update<Hash>,
    /// Optional ipv4 addr
    #[schema(schema_with = <Update<SocketAddrV4> as PartialSchema>::schema)]
    pub socket_addr_v4: Update<SocketAddrV4>,
    /// Optional ipv6 addr
    #[schema(schema_with = <Update<SocketAddrV6> as PartialSchema>::schema)]
    pub socket_addr_v6: Update<SocketAddrV6>,
    /// Optional pubkey used for encryption
    #[schema(schema_with = <Update<EncryptionPubKey> as PartialSchema>::schema)]
    pub encryption_pubkey: Update<EncryptionPubKey>,
    /// Optional pubkey used for signing messages
    #[schema(schema_with = <Update<VerifyingKey> as PartialSchema>::schema)]
    pub signing_pubkey: Update<VerifyingKey>,
}
