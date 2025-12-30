use sneed::{db::error as db, env::error as env, rwtxn::error as rwtxn};
use thiserror::Error;
use transitive::Transitive;

use crate::types::{
    AmountOverflowError, AmountUnderflowError, BlockHash, M6id, MerkleRoot,
    OutPoint, WithdrawalBundleError,
};

#[derive(Debug, Error)]
pub enum InvalidHeader {
    #[error("expected block hash {expected}, but computed {computed}")]
    BlockHash {
        expected: BlockHash,
        computed: BlockHash,
    },
    #[error(
        "expected previous sidechain block hash {expected:?}, but received {received:?}"
    )]
    PrevSideHash {
        expected: Option<BlockHash>,
        received: Option<BlockHash>,
    },
}

#[derive(Debug)]
pub struct FillTxOutputContents(pub Box<crate::types::FilledTransaction>);

impl std::fmt::Display for FillTxOutputContents {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let txid = self.0.txid();
        write!(
            f,
            "failed to fill tx output contents ({txid}): invalid transaction"
        )?;
        if f.alternate() {
            let tx_json = serde_json::to_string(&self.0).unwrap();
            write!(f, " ({tx_json})")?;
        }
        Ok(())
    }
}

impl std::error::Error for FillTxOutputContents {}

#[allow(clippy::duplicated_attributes)]
#[derive(Debug, Error, Transitive)]
#[transitive(from(db::Clear, db::Error))]
#[transitive(from(db::Delete, db::Error))]
#[transitive(from(db::Error, sneed::Error))]
#[transitive(from(db::IterInit, db::Error))]
#[transitive(from(db::IterItem, db::Error))]
#[transitive(from(db::Last, db::Error))]
#[transitive(from(db::Put, db::Error))]
#[transitive(from(db::TryGet, db::Error))]
#[transitive(from(env::CreateDb, env::Error))]
#[transitive(from(env::Error, sneed::Error))]
#[transitive(from(env::WriteTxn, env::Error))]
#[transitive(from(rwtxn::Commit, rwtxn::Error))]
#[transitive(from(rwtxn::Error, sneed::Error))]
pub enum Error {
    #[error(transparent)]
    Market(#[from] crate::state::markets::MarketError),
    #[error(transparent)]
    AmountOverflow(#[from] AmountOverflowError),
    #[error(transparent)]
    AmountUnderflow(#[from] AmountUnderflowError),
    #[error("failed to verify authorization")]
    AuthorizationError,
    #[error("bad coinbase output content")]
    BadCoinbaseOutputContent,
    #[error("genesis slots have already been initialized")]
    GenesisAlreadyInitialized,
    #[error("inconsistent decision type")]
    InconsistentDecisionType,
    #[error("invalid range: min must be less than max")]
    InvalidRange,
    #[error("invalid slot ID: {reason}")]
    InvalidSlotId { reason: String },
    #[error("invalid slot state: {reason}")]
    InvalidSlotState { reason: String },
    #[error("invalid timestamp")]
    InvalidTimestamp,
    #[error("invalid transaction: {reason}")]
    InvalidTransaction { reason: String },
    #[error("slot {slot_id:?} is already claimed")]
    SlotAlreadyClaimed {
        slot_id: crate::state::slots::SlotId,
    },
    #[error("slot {slot_id:?} is not available: {reason}")]
    SlotNotAvailable {
        slot_id: crate::state::slots::SlotId,
        reason: String,
    },
    #[error("timestamp out of range")]
    TimestampOutOfRange,

    #[error("bundle too heavy {weight} > {max_weight}")]
    BundleTooHeavy { weight: u64, max_weight: u64 },
    #[error(transparent)]
    BorshSerialize(borsh::io::Error),
    #[error("Database consistency error: {0}")]
    DatabaseError(String),
    #[error(transparent)]
    Db(#[from] sneed::Error),
    #[error(transparent)]
    FillTxOutputContents(#[from] FillTxOutputContents),
    #[error(
        "invalid body: expected merkle root {expected}, but computed {computed}"
    )]
    InvalidBody {
        expected: MerkleRoot,
        computed: MerkleRoot,
    },
    #[error("invalid header: {0}")]
    InvalidHeader(InvalidHeader),

    #[error("deposit block doesn't exist")]
    NoDepositBlock,
    #[error("total fees less than coinbase value")]
    NotEnoughFees,
    #[error("value in is less than value out")]
    NotEnoughValueIn,
    #[error("no tip")]
    NoTip,
    #[error("stxo {outpoint} doesn't exist")]
    NoStxo { outpoint: OutPoint },
    #[error("utxo {outpoint} doesn't exist")]
    NoUtxo { outpoint: OutPoint },
    #[error("Withdrawal bundle event block doesn't exist")]
    NoWithdrawalBundleEventBlock,

    #[error(transparent)]
    SignatureError(#[from] ed25519_dalek::SignatureError),

    #[error("unbalanced Votecoin: {inputs} inputs, {outputs} outputs")]
    UnbalancedVotecoin { inputs: u32, outputs: u32 },

    #[error("Unknown withdrawal bundle: {m6id}")]
    UnknownWithdrawalBundle { m6id: M6id },
    #[error(
        "Unknown withdrawal bundle confirmed in {event_block_hash}: {m6id}"
    )]
    UnknownWithdrawalBundleConfirmed {
        event_block_hash: bitcoin::BlockHash,
        m6id: M6id,
    },
    #[error("utxo double spent")]
    UtxoDoubleSpent,
    #[error(transparent)]
    WithdrawalBundle(#[from] WithdrawalBundleError),
    #[error("wrong public key for address")]
    WrongPubKeyForAddress,
    #[error(
        "consensus not yet calculated for period {0:?} - must be calculated by protocol during block connection"
    )]
    ConsensusNotYetCalculated(crate::state::voting::types::VotingPeriodId),
}
