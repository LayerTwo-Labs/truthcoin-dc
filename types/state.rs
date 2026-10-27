use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{M6id, OutPoint, Output, WithdrawalBundle};

/// Information we have regarding a withdrawal bundle
#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub enum WithdrawalBundleInfo {
    /// Withdrawal bundle is known
    Known(WithdrawalBundle),
    /// Withdrawal bundle is unknown but unconfirmed / failed
    Unknown,
    /// If an unknown withdrawal bundle is confirmed, ALL UTXOs are
    /// considered spent.
    UnknownConfirmed {
        spend_utxos: BTreeMap<OutPoint, Output>,
    },
}

/// A coin movement that a block applied outside its body
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize, ToSchema)]
pub enum TwoWayPegEvent {
    /// A mainchain deposit created this output
    Deposit { outpoint: OutPoint, output: Output },
    /// A withdrawal bundle spent this output
    BundleSpend { outpoint: OutPoint, m6id: M6id },
    /// A failed withdrawal bundle returned this output to the UTXO set
    BundleReturn {
        outpoint: OutPoint,
        output: Output,
        m6id: M6id,
    },
}

/// What a market output that no transaction created or spent holds
#[derive(
    Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ToSchema,
)]
pub enum MarketUtxoReason {
    /// The market treasury
    Treasury,
    /// The fees of the market author
    AuthorFee,
    /// The proceeds of a sell trade, to the seller
    SellPayout,
    /// The change of a buy trade, to the buyer
    BuyChange,
    /// The change of the inputs of a sell trade, to the seller
    SellInputChange,
    /// The payout of settled shares, to the shareholder
    SharePayout,
    /// A share of the author fees at settlement
    FeePayout,
    /// The refund to the market creator at settlement
    CreatorRefund,
}

/// An output that market code created or removed outside any transaction
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize, ToSchema)]
pub struct MarketUtxo {
    pub outpoint: OutPoint,
    pub output: Output,
    pub reason: MarketUtxoReason,
}

/// The market outputs that one block created and removed outside its body,
/// each list in the order the node applied it
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct MarketUtxoChanges {
    pub creates: Vec<MarketUtxo>,
    pub deletes: Vec<MarketUtxo>,
}

impl MarketUtxoChanges {
    pub fn is_empty(&self) -> bool {
        self.creates.is_empty() && self.deletes.is_empty()
    }
}
