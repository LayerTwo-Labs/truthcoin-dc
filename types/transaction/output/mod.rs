use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    Address, AssetId, GetBitcoinValue, InPoint, OutPoint,
    serde_display_fromstr_human_readable, serde_hexstr_human_readable,
};

mod content;
pub use content::{
    AssetContent, BitcoinContent, Content, FilledContent, WithdrawalContent,
};

#[derive(
    BorshSerialize,
    Clone,
    Debug,
    Deserialize,
    Eq,
    PartialEq,
    Serialize,
    ToSchema,
)]
pub struct Output<OutputContent = Content> {
    #[serde(with = "serde_display_fromstr_human_readable")]
    pub address: Address,
    pub content: OutputContent,
    #[serde(with = "serde_hexstr_human_readable")]
    pub memo: Vec<u8>,
}

impl<Content> Output<Content> {
    pub fn new(address: Address, content: Content) -> Self {
        Self {
            address,
            content,
            memo: Vec::new(),
        }
    }

    pub fn map_content<C, F>(self, f: F) -> Output<C>
    where
        F: FnOnce(Content) -> C,
    {
        Output {
            address: self.address,
            content: f(self.content),
            memo: self.memo,
        }
    }

    pub fn map_content_opt<C, F>(self, f: F) -> Option<Output<C>>
    where
        F: FnOnce(Content) -> Option<C>,
    {
        Some(Output {
            address: self.address,
            content: f(self.content)?,
            memo: self.memo,
        })
    }
}

pub type TxOutput = Output;

impl TxOutput {
    /// `true` if the output content corresponds to a Bitcoin Value
    pub fn is_bitcoin(&self) -> bool {
        self.content.is_bitcoin()
    }

    /// `true` if the output content corresponds to a Bitcoin Withdrawal
    pub fn is_withdrawal(&self) -> bool {
        self.content.is_withdrawal()
    }

    /// `true` if the output corresponds to an asset output
    pub fn is_asset(&self) -> bool {
        self.content.is_asset()
    }
}

impl GetBitcoinValue for TxOutput {
    #[inline(always)]
    fn get_bitcoin_value(&self) -> bitcoin::Amount {
        self.content.get_bitcoin_value()
    }
}

pub type BitcoinOutput = Output<BitcoinContent>;

impl From<TxOutput> for Option<BitcoinOutput> {
    fn from(output: Output) -> Option<BitcoinOutput> {
        output.map_content_opt(Content::as_bitcoin)
    }
}

pub type AssetOutput = Output<AssetContent>;

impl From<TxOutput> for Option<AssetOutput> {
    fn from(output: Output) -> Option<AssetOutput> {
        output.map_content_opt(Content::as_asset)
    }
}

pub type FilledOutput = Output<FilledContent>;

impl FilledOutput {
    pub fn asset_value(&self) -> Option<(AssetId, u64)> {
        self.content.asset_value()
    }

    pub fn content(&self) -> &FilledContent {
        &self.content
    }

    pub fn is_bitcoin(&self) -> bool {
        self.content.is_bitcoin()
    }

    /// True if the output content corresponds to a withdrawal
    pub fn is_withdrawal(&self) -> bool {
        self.content.is_withdrawal()
    }
}

impl From<FilledOutput> for Output {
    fn from(filled: FilledOutput) -> Self {
        Self {
            address: filled.address,
            content: filled.content.into(),
            memo: filled.memo,
        }
    }
}

impl GetBitcoinValue for FilledOutput {
    fn get_bitcoin_value(&self) -> bitcoin::Amount {
        self.content.get_bitcoin_value()
    }
}

/// Representation of a spent output
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize, ToSchema)]
pub struct SpentOutput<OutputContent = FilledContent> {
    #[schema(inline)]
    pub output: Output<OutputContent>,
    pub inpoint: InPoint,
}

#[derive(
    BorshSerialize,
    Clone,
    Debug,
    Deserialize,
    Eq,
    PartialEq,
    Serialize,
    ToSchema,
)]
pub struct Pointed<OutputContent = Content> {
    pub outpoint: OutPoint,
    #[schema(inline)]
    pub output: Output<OutputContent>,
}
