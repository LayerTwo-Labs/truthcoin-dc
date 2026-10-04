use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    address::Address,
    transaction::{GetValue, outpoint::OutPoint},
};

mod content;
pub use content::Content;

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
pub struct Output {
    pub address: Address,
    pub content: Content,
}

impl Output {
    /// Canonical size in bytes. The canonical encoding is used for hashing,
    /// but other encodings may be used at eg. networking, rpc levels.
    #[inline(always)]
    pub(crate) fn canonical_size(&self) -> borsh::io::Result<u64> {
        borsh::object_length(self).map(|size| size as u64)
    }
}

impl GetValue for Output {
    #[inline(always)]
    fn get_value(&self) -> bitcoin::Amount {
        self.content.get_value()
    }
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
pub struct Pointed<Output = crate::transaction::output::Output> {
    pub outpoint: OutPoint,
    pub output: Output,
}

/// Useful when computing hashes for Utreexo,
/// without needing to clone an output
#[derive(BorshSerialize, Clone, Copy, Debug)]
pub struct PointedOutputRef<'a> {
    pub outpoint: OutPoint,
    pub output: &'a Output,
}
