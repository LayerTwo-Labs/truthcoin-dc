use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};

use crate::decision::DecisionId;

#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, BorshSerialize,
)]
pub enum DimensionSpec {
    Single(DecisionId),
    Categorical(DecisionId),
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Ord,
    PartialOrd,
    Serialize,
    Deserialize,
    borsh::BorshSerialize,
    borsh::BorshDeserialize,
)]
pub struct MarketId(pub [u8; 6]);

impl MarketId {
    pub fn new(data: [u8; 6]) -> Self {
        Self(data)
    }

    pub fn as_bytes(&self) -> &[u8; 6] {
        &self.0
    }
}

impl std::fmt::Display for MarketId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", const_hex::encode(self.0))
    }
}

impl AsRef<[u8]> for MarketId {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl utoipa::PartialSchema for MarketId {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::Schema> {
        let schema = utoipa::openapi::ObjectBuilder::new()
            .description(Some("6-byte market identifier"))
            .examples([serde_json::json!("0x0123456789ab")])
            .build();
        utoipa::openapi::RefOr::T(utoipa::openapi::Schema::Object(schema))
    }
}

impl utoipa::ToSchema for MarketId {
    fn name() -> std::borrow::Cow<'static, str> {
        "MarketId".into()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketState {
    Trading = 1,
    Cancelled = 3,
    Invalid = 4,
    Settled = 5,
}

impl MarketState {
    pub fn can_transition_to(&self, new_state: MarketState) -> bool {
        use MarketState::*;
        match (self, new_state) {
            (Trading, Trading)
            | (Cancelled, Cancelled)
            | (Invalid, Invalid)
            | (Settled, Settled) => true,
            (Trading, Cancelled | Invalid | Settled) => true,
            (Invalid, Settled) => true,
            (Cancelled, Trading | Invalid | Settled) => false,
            (Invalid, Trading | Cancelled) => false,
            (Settled, Trading | Cancelled | Invalid) => false,
        }
    }

    pub fn allows_trading(&self) -> bool {
        matches!(self, MarketState::Trading)
    }
}
