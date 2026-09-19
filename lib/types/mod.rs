use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap},
    sync::LazyLock,
};

use bitcoin::amount::CheckedSum as _;
use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use thiserror::Error;
use utoipa::ToSchema;

pub use crate::authorization::Authorization;

mod address;
pub mod hashes;
pub mod keys;
pub mod net;
pub mod proto;
pub mod schema;
mod transaction;
pub mod tx_pow;

pub use address::Address;
pub use hashes::{AssetId, BlockHash, Hash, M6id, MerkleRoot, Txid};
pub use keys::{EncryptionPubKey, VerifyingKey};
pub use transaction::{
    AssetOutput, AssetOutputContent, Authorized, AuthorizedTransaction,
    BallotItem, BitcoinOutput, BitcoinOutputContent, ClaimDecisionPayload,
    DecisionClaimEntry, FilledOutput, FilledOutputContent, FilledTransaction,
    InPoint, MarketCreationView, OutPoint, OutPointKey, Output, OutputContent,
    PointedOutput, SpentOutput, Transaction, TransactionData, TxData, TxInputs,
    WithdrawalOutputContent,
};

pub const THIS_SIDECHAIN: u8 = 13;

#[derive(Debug, Error)]
#[error("Bitcoin amount overflow")]
pub struct AmountOverflowError;

#[derive(Debug, Error)]
#[error("Bitcoin amount underflow")]
pub struct AmountUnderflowError;

#[derive(Debug, Error)]
#[error("body has fewer authorizations than transaction inputs")]
pub struct MalformedBodyError;

/// (de)serialize as Display/FromStr for human-readable forms like json,
/// and default serialization for non human-readable forms like bincode
mod serde_display_fromstr_human_readable {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use serde_with::{DeserializeAs, DisplayFromStr, SerializeAs};
    use std::{fmt::Display, str::FromStr};

    pub fn serialize<S, T>(data: T, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize + Display,
    {
        if serializer.is_human_readable() {
            DisplayFromStr::serialize_as(&data, serializer)
        } else {
            data.serialize(serializer)
        }
    }

    pub fn deserialize<'de, D, T>(deserializer: D) -> Result<T, D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de> + FromStr,
        <T as FromStr>::Err: Display,
    {
        if deserializer.is_human_readable() {
            DisplayFromStr::deserialize_as(deserializer)
        } else {
            T::deserialize(deserializer)
        }
    }
}

/// Optimized (de)serialize as hex strings for human-readable forms like json,
/// and default serialization for non human-readable formats like bincode
mod serde_hexstr_human_readable {
    use const_hex::{FromHex, ToHexExt};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[inline]
    pub fn serialize<S, T>(data: T, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize + ToHexExt,
    {
        if serializer.is_human_readable() {
            data.encode_hex().serialize(serializer)
        } else {
            data.serialize(serializer)
        }
    }

    #[inline]
    pub fn deserialize<'de, D, T>(deserializer: D) -> Result<T, D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de> + FromHex,
        <T as FromHex>::Error: std::fmt::Display,
    {
        if deserializer.is_human_readable() {
            const_hex::serde::deserialize(deserializer)
        } else {
            T::deserialize(deserializer)
        }
    }
}

pub trait GetAddress {
    fn get_address(&self) -> Address;
}

pub trait GetBitcoinValue {
    /// Bitcoin value in sats
    fn get_bitcoin_value(&self) -> bitcoin::Amount;
}

#[derive(Debug, Error)]
pub enum Bech32mDecodeError {
    #[error(transparent)]
    Bech32m(#[from] bech32::DecodeError),
    #[error(
        "Wrong Bech32 HRP. Perhaps this key is being used somewhere it shouldn't be."
    )]
    WrongHrp,
    #[error("Wrong decoded byte length. Must decode to 32 bytes of data.")]
    WrongSize,
    #[error("Wrong Bech32 variant. Only Bech32m is accepted.")]
    WrongVariant,
}

fn borsh_serialize_bitcoin_block_hash<W>(
    block_hash: &bitcoin::BlockHash,
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    let bytes: &[u8; 32] = block_hash.as_ref();
    borsh::BorshSerialize::serialize(bytes, writer)
}

#[derive(
    BorshSerialize,
    Clone,
    Debug,
    Deserialize,
    Eq,
    Hash,
    PartialEq,
    Serialize,
    ToSchema,
)]
pub struct Header {
    pub merkle_root: MerkleRoot,
    pub prev_side_hash: Option<BlockHash>,
    #[borsh(serialize_with = "borsh_serialize_bitcoin_block_hash")]
    #[schema(value_type = crate::types::schema::BitcoinBlockHash)]
    pub prev_main_hash: bitcoin::BlockHash,
}

impl Header {
    pub fn hash(&self) -> BlockHash {
        hashes::hash_with_scratch_buffer(self).into()
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum WithdrawalBundleEventStatus {
    Confirmed,
    Failed,
    Submitted,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum WithdrawalBundleStatus {
    Confirmed,
    Dropped,
    Failed,
    Pending,
    Submitted,
    SubmittedUnexpected,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WithdrawalBundleEvent {
    pub m6id: M6id,
    pub status: WithdrawalBundleEventStatus,
}

/// Coin movements that a block body does not carry: a mainchain deposit, and
/// the outputs a withdrawal bundle removed
#[derive(
    Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize, ToSchema,
)]
pub struct BlockIndexEvents {
    /// Outputs that mainchain deposits created
    pub deposits: Vec<(OutPoint, FilledOutput)>,
    /// Outputs that a withdrawal bundle removed, with the bundle that took them
    pub bundle_spends: Vec<(OutPoint, M6id)>,
}

impl BlockIndexEvents {
    /// True when the block moved no coins outside its body
    pub fn is_empty(&self) -> bool {
        self.deposits.is_empty() && self.bundle_spends.is_empty()
    }
}

/// One transaction of a block, with the fields its body omits
#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct BlockIndexTx {
    pub txid: Txid,
    /// Borsh size in bytes
    pub size: u64,
    /// Borsh encoding, as hex
    pub raw: String,
}

/// One output a mainchain deposit created
#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct BlockIndexDeposit {
    pub outpoint: OutPoint,
    pub output: FilledOutput,
}

/// One output a withdrawal bundle removed, with the bundle that took it
#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct BlockIndexSpend {
    pub outpoint: OutPoint,
    pub m6id: M6id,
}

/// Everything about a block that its body does not carry
//  Each pair is a named struct: a tuple of ref schemas does not compose.
#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct BlockIndex {
    /// Transactions in body order
    pub txs: Vec<BlockIndexTx>,
    /// Outputs that mainchain deposits created
    pub deposits: Vec<BlockIndexDeposit>,
    /// Outputs that a withdrawal bundle removed
    pub bundle_spends: Vec<BlockIndexSpend>,
}

pub static OP_DRIVECHAIN_SCRIPT: LazyLock<bitcoin::ScriptBuf> =
    LazyLock::new(|| {
        let mut script = bitcoin::ScriptBuf::new();
        script.push_opcode(bitcoin::opcodes::all::OP_RETURN);
        script.push_instruction(bitcoin::script::Instruction::PushBytes(
            &bitcoin::script::PushBytesBuf::from([THIS_SIDECHAIN]),
        ));
        script.push_opcode(bitcoin::opcodes::OP_TRUE);
        script
    });

#[derive(Debug, Error)]
enum WithdrawalBundleErrorInner {
    #[error("bundle too heavy: weight `{weight}` > max weight `{max_weight}`")]
    BundleTooHeavy { weight: u64, max_weight: u64 },
}

#[derive(Debug, Error)]
#[error("Withdrawal bundle error")]
pub struct WithdrawalBundleError(#[from] WithdrawalBundleErrorInner);

#[serde_as]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize, ToSchema)]
pub struct WithdrawalBundle {
    #[schema(value_type = Vec<(
        transaction::OutPoint,
        transaction::FilledOutput)>
    )]
    #[serde_as(as = "serde_with::IfIsHumanReadable<serde_with::Seq<(_, _)>>")]
    spend_utxos: BTreeMap<OutPoint, FilledOutput>,
    #[schema(value_type = schema::BitcoinTransaction)]
    tx: bitcoin::Transaction,
}

impl WithdrawalBundle {
    /// Compute the size of a single txout
    pub const fn txout_size(spk_size: u32) -> Option<u32> {
        let Some(size) = (bitcoin::Amount::SIZE as u32)
            .checked_add(bitcoin::VarInt(spk_size as u64).size() as u32)
        else {
            return None;
        };
        size.checked_add(spk_size)
    }

    /// Predict the weight of a withdrawal bundle, based on the number of
    /// outputs (not including the commitment/treasury outputs) and the
    /// sum of sizes of txouts (not including the commitment/treasury outputs).
    /// Returns None if the predicted weight exceeds the maximum tx weight.
    pub const fn predict_weight(
        n_outputs: u32,
        sum_txout_sizes: u32,
    ) -> Option<bitcoin::Weight> {
        use bitcoin::{VarInt, Weight};
        const fn txin_base_size(script_sig_size: u32) -> Option<u32> {
            const OUTPOINT_SIZE: u8 = 36;
            const SEQUENCE_SIZE: u8 = 4;
            let script_sig_len_size: u8 =
                VarInt(script_sig_size as u64).size() as u8;
            let Some(res) = ((OUTPOINT_SIZE + script_sig_len_size) as u32)
                .checked_add(script_sig_size)
            else {
                return None;
            };
            res.checked_add(SEQUENCE_SIZE as u32)
        }
        const fn tx_base_size(
            n_inputs: u32,
            sum_txin_base_sizes: u32,
            n_outputs: u32,
            sum_txout_sizes: u32,
        ) -> Option<u32> {
            const VERSION_SIZE: u8 = 4;
            const fn vin_base_size(
                n_inputs: u32,
                sum_txin_base_sizes: u32,
            ) -> Option<u32> {
                let len_size = VarInt(n_inputs as u64).size() as u8;
                (len_size as u32).checked_add(sum_txin_base_sizes)
            }
            const fn vout_size(
                n_outputs: u32,
                sum_txout_sizes: u32,
            ) -> Option<u32> {
                let len_size = VarInt(n_outputs as u64).size() as u8;
                (len_size as u32).checked_add(sum_txout_sizes)
            }
            const LOCKTIME_SIZE: u8 = bitcoin::absolute::LockTime::SIZE as u8;
            let res = VERSION_SIZE as u32;
            let Some(vin_base_size) =
                vin_base_size(n_inputs, sum_txin_base_sizes)
            else {
                return None;
            };
            let Some(res) = res.checked_add(vin_base_size) else {
                return None;
            };
            let Some(vout_size) = vout_size(n_outputs, sum_txout_sizes) else {
                return None;
            };
            let Some(res) = res.checked_add(vout_size) else {
                return None;
            };
            res.checked_add(LOCKTIME_SIZE as u32)
        }
        const N_INPUTS: u32 = 1;
        const SUM_TXIN_BASE_SIZES: u32 = {
            const TREASURY_TXIN_BASE_SIZE: u32 = {
                const TREASURY_SCRIPT_SIG_SIZE: u32 = 0;
                txin_base_size(TREASURY_SCRIPT_SIG_SIZE).unwrap()
            };
            TREASURY_TXIN_BASE_SIZE
        };
        let Some(n_outputs) = n_outputs.checked_add(2) else {
            return None;
        };
        let Some(sum_txout_sizes) = ({
            const INPUTS_COMMITMENT_TXOUT_SIZE: u32 = {
                const INPUTS_COMMITMENT_OUTPUT_SPK_SIZE: u8 = 34;
                WithdrawalBundle::txout_size(
                    INPUTS_COMMITMENT_OUTPUT_SPK_SIZE as u32,
                )
                .unwrap()
            };
            const MAINCHAIN_FEE_COMMITMENT_TXOUT_SIZE: u32 = {
                const MAINCHAIN_FEE_COMMITMENT_OUTPUT_SPK_SIZE: u8 = 10;
                WithdrawalBundle::txout_size(
                    MAINCHAIN_FEE_COMMITMENT_OUTPUT_SPK_SIZE as u32,
                )
                .unwrap()
            };
            (INPUTS_COMMITMENT_TXOUT_SIZE + MAINCHAIN_FEE_COMMITMENT_TXOUT_SIZE)
                .checked_add(sum_txout_sizes)
        }) else {
            return None;
        };
        let Some(tx_base_size) = tx_base_size(
            N_INPUTS,
            SUM_TXIN_BASE_SIZES,
            n_outputs,
            sum_txout_sizes,
        ) else {
            return None;
        };
        let Some(tx_weight_wu) =
            (tx_base_size as u64).checked_mul(Weight::WITNESS_SCALE_FACTOR)
        else {
            return None;
        };
        if tx_weight_wu <= bitcoin::Transaction::MAX_STANDARD_WEIGHT.to_wu() {
            Some(Weight::from_wu(tx_weight_wu))
        } else {
            None
        }
    }

    pub fn new(
        block_height: u32,
        fee: bitcoin::Amount,
        spend_utxos: BTreeMap<OutPoint, FilledOutput>,
        bundle_outputs: Vec<bitcoin::TxOut>,
    ) -> Result<Self, WithdrawalBundleError> {
        let inputs_commitment_txout = {
            // Create inputs commitment.
            let inputs: Vec<OutPoint> = [
                // Commit to inputs.
                spend_utxos.keys().copied().collect(),
                // Commit to block height.
                vec![OutPoint::Regular {
                    txid: [0; 32].into(),
                    vout: block_height,
                }],
            ]
            .concat();
            let commitment = hashes::hash_with_scratch_buffer(&inputs);
            let script_pubkey = bitcoin::script::Builder::new()
                .push_opcode(bitcoin::opcodes::all::OP_RETURN)
                .push_slice(commitment)
                .into_script();
            bitcoin::TxOut {
                value: bitcoin::Amount::ZERO,
                script_pubkey,
            }
        };
        let mainchain_fee_txout = {
            let script_pubkey = bitcoin::script::Builder::new()
                .push_opcode(bitcoin::opcodes::all::OP_RETURN)
                .push_slice(fee.to_sat().to_be_bytes())
                .into_script();
            bitcoin::TxOut {
                value: bitcoin::Amount::ZERO,
                script_pubkey,
            }
        };
        let outputs = Vec::from_iter(
            [mainchain_fee_txout, inputs_commitment_txout]
                .into_iter()
                .chain(bundle_outputs),
        );
        let tx = bitcoin::Transaction {
            version: bitcoin::transaction::Version::TWO,
            lock_time: bitcoin::blockdata::locktime::absolute::LockTime::ZERO,
            input: Vec::new(),
            output: outputs,
        };
        if tx.weight().to_wu() > bitcoin::policy::MAX_STANDARD_TX_WEIGHT as u64
        {
            Err(WithdrawalBundleErrorInner::BundleTooHeavy {
                weight: tx.weight().to_wu(),
                max_weight: bitcoin::policy::MAX_STANDARD_TX_WEIGHT as u64,
            })?;
        }
        Ok(Self { spend_utxos, tx })
    }

    pub fn compute_m6id(&self) -> M6id {
        M6id(self.tx.compute_txid())
    }

    pub fn spend_utxos(&self) -> &BTreeMap<OutPoint, FilledOutput> {
        &self.spend_utxos
    }

    pub fn tx(&self) -> &bitcoin::Transaction {
        &self.tx
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct TwoWayPegData {
    pub deposits: HashMap<OutPoint, Output>,
    pub deposit_block_hash: Option<bitcoin::BlockHash>,
    pub bundle_statuses: HashMap<M6id, WithdrawalBundleEvent>,
}

#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct Body {
    pub coinbase: Vec<Output>,
    pub transactions: Vec<Transaction>,
    pub authorizations: Vec<Authorization>,
    #[serde(default)]
    pub actor_proofs: Vec<Option<Authorization>>,
}

impl Body {
    /// Size limit in bytes
    pub const MAX_SIZE: usize = 8 * 1024 * 1024;

    pub fn new(
        authorized_transactions: Vec<AuthorizedTransaction>,
        coinbase: Vec<Output>,
    ) -> Self {
        let mut authorizations = Vec::with_capacity(
            authorized_transactions
                .iter()
                .map(|t| t.transaction.inputs.len())
                .sum(),
        );
        let mut transactions =
            Vec::with_capacity(authorized_transactions.len());
        let mut actor_proofs =
            Vec::with_capacity(authorized_transactions.len());
        for at in authorized_transactions.into_iter() {
            authorizations.extend(at.authorizations);
            actor_proofs.push(at.actor_proof);
            transactions.push(at.transaction);
        }
        Self {
            coinbase,
            transactions,
            authorizations,
            actor_proofs,
        }
    }

    pub fn authorized_transactions(
        &self,
    ) -> Result<Vec<AuthorizedTransaction>, MalformedBodyError> {
        let mut authorizations_iter = self.authorizations.iter();
        let mut actor_proofs_iter = self.actor_proofs.iter();
        self.transactions
            .iter()
            .map(|tx| {
                let mut authorizations = Vec::with_capacity(tx.inputs.len());
                for _ in 0..tx.inputs.len() {
                    let auth =
                        authorizations_iter.next().ok_or(MalformedBodyError)?;
                    authorizations.push(auth.clone());
                }
                let actor_proof = actor_proofs_iter.next().cloned().flatten();
                Ok(AuthorizedTransaction {
                    transaction: tx.clone(),
                    authorizations,
                    actor_proof,
                })
            })
            .collect()
    }

    pub fn compute_merkle_root(
        coinbase: &[Output],
        txs: &[Transaction],
    ) -> MerkleRoot {
        let coinbase_hash: Hash = hashes::hash_with_scratch_buffer(coinbase);
        let mut leaves: Vec<Hash> = std::iter::once(coinbase_hash)
            .chain(txs.iter().map(|tx| tx.txid().into()))
            .collect();
        while leaves.len() > 1 {
            let mut next_level = Vec::with_capacity(leaves.len().div_ceil(2));
            for pair in leaves.chunks(2) {
                let left = pair[0].as_ref();
                let right = if pair.len() == 2 {
                    pair[1].as_ref()
                } else {
                    pair[0].as_ref()
                };
                let mut combined = [0u8; 64];
                combined[..32].copy_from_slice(left);
                combined[32..].copy_from_slice(right);
                next_level.push(*blake3::hash(&combined).as_bytes());
            }
            leaves = next_level;
        }
        leaves[0].into()
    }

    pub fn get_inputs(&self) -> Vec<OutPoint> {
        self.transactions
            .iter()
            .flat_map(|tx| tx.inputs.iter())
            .copied()
            .collect()
    }

    pub fn get_outputs(&self) -> HashMap<OutPoint, Output> {
        let mut outputs = HashMap::new();
        let merkle_root =
            Body::compute_merkle_root(&self.coinbase, &self.transactions);
        for (vout, output) in self.coinbase.iter().enumerate() {
            let vout = vout as u32;
            let outpoint = OutPoint::Coinbase { merkle_root, vout };
            outputs.insert(outpoint, output.clone());
        }
        for transaction in &self.transactions {
            let txid = transaction.txid();
            for (vout, output) in transaction.outputs.iter().enumerate() {
                let vout = vout as u32;
                let outpoint = OutPoint::Regular { txid, vout };
                outputs.insert(outpoint, output.clone());
            }
        }
        outputs
    }

    pub fn get_coinbase_value(
        &self,
    ) -> Result<bitcoin::Amount, AmountOverflowError> {
        self.coinbase
            .iter()
            .map(|output| output.get_bitcoin_value())
            .checked_sum()
            .ok_or(AmountOverflowError)
    }
}

pub trait Verify {
    type Error;
    fn verify_transaction(
        transaction: &AuthorizedTransaction,
    ) -> Result<(), Self::Error>;
    fn verify_body(body: &Body) -> Result<(), Self::Error>;
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct Block {
    #[serde(flatten)]
    pub header: Header,
    #[serde(flatten)]
    pub body: Body,
    pub height: u32,
}

#[derive(Eq, PartialEq, Clone, Debug)]
pub struct AggregatedWithdrawal {
    pub spend_utxos: HashMap<OutPoint, FilledOutput>,
    pub main_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
    pub value: bitcoin::Amount,
    pub main_fee: bitcoin::Amount,
}

impl Ord for AggregatedWithdrawal {
    fn cmp(&self, other: &Self) -> Ordering {
        // A *total* order (lexicographic by main_fee, value, main_address). The
        // previous `OR of >` was not antisymmetric/transitive, so the
        // withdrawal-bundle output order (and hence compute_m6id) depended on
        // HashMap iteration order and could differ across nodes. A real total order makes
        // the sorted bundle canonical regardless of aggregation order.
        (self.main_fee, self.value, &self.main_address).cmp(&(
            other.main_fee,
            other.value,
            &other.main_address,
        ))
    }
}

impl PartialOrd for AggregatedWithdrawal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Transaction index
#[derive(Clone, Copy, Debug, Deserialize, Serialize, ToSchema)]
pub struct TxIn {
    pub block_hash: BlockHash,
    pub idx: u32,
}

/// Step of the sync with the mainchain
#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize, ToSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum MainchainSyncPhase {
    #[default]
    Idle,
    /// Fetch mainchain headers from the enforcer
    Headers,
    /// Write the fetched mainchain headers to the archive
    Writing,
}

/// Progress of the sync with the mainchain
#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize, ToSchema,
)]
pub struct MainchainSyncProgress {
    pub phase: MainchainSyncPhase,
    pub done: u32,
    pub total: u32,
    /// Height of the mainchain block that the sync moves to
    pub tip_height: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum BmmResult {
    Verified,
    Failed,
}

/// A tip refers to both a sidechain block AND the mainchain block that commits
/// to it.
#[derive(
    BorshSerialize,
    Clone,
    Copy,
    Debug,
    Deserialize,
    Eq,
    Hash,
    PartialEq,
    Serialize,
)]
pub struct Tip {
    pub block_hash: BlockHash,
    #[borsh(serialize_with = "borsh_serialize_bitcoin_block_hash")]
    pub main_block_hash: bitcoin::BlockHash,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum, strum::Display))]
pub enum Network {
    #[default]
    Betanet,
    Forknet,
    Regtest,
    Signet,
}

/// Semver-compatible version
#[derive(
    BorshSerialize,
    Clone,
    Copy,
    Debug,
    Deserialize,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
)]
pub struct Version {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl From<semver::Version> for Version {
    fn from(version: semver::Version) -> Self {
        let semver::Version {
            major,
            minor,
            patch,
            pre: _,
            build: _,
        } = version;
        Self {
            major,
            minor,
            patch,
        }
    }
}
// Do not make this public outside of this crate, as it could break semver
pub(crate) static VERSION: LazyLock<Version> = LazyLock::new(|| {
    const VERSION_STR: &str = env!("CARGO_PKG_VERSION");
    semver::Version::parse(VERSION_STR).unwrap().into()
});

pub const SECONDS_PER_QUARTER: u64 = 3600 * 24 * 91;

#[cfg(test)]
mod withdrawal_bundle_order_regression {
    use super::{AggregatedWithdrawal, M6id, WithdrawalBundle};
    use bitcoin::{Address, Amount, address::NetworkUnchecked};
    use std::collections::{BTreeMap, HashMap};

    fn aw(value: u64, main_fee: u64) -> AggregatedWithdrawal {
        // value/main_fee drive the comparison; one address is enough to expose it.
        let addr: Address<NetworkUnchecked> =
            "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"
                .parse()
                .unwrap();
        AggregatedWithdrawal {
            spend_utxos: HashMap::new(),
            main_address: addr,
            value: Amount::from_sat(value),
            main_fee: Amount::from_sat(main_fee),
        }
    }

    // Build the bundle m6id exactly as `collect_withdrawal_bundle` does, for a given
    // (HashMap-determined) input order.
    fn bundle_m6id(mut aggregated: Vec<AggregatedWithdrawal>) -> M6id {
        aggregated.sort_by_key(|a| std::cmp::Reverse(a.clone()));
        let outputs: Vec<bitcoin::TxOut> = aggregated
            .iter()
            .map(|a| bitcoin::TxOut {
                value: a.value,
                script_pubkey: a
                    .main_address
                    .assume_checked_ref()
                    .script_pubkey(),
            })
            .collect();
        WithdrawalBundle::new(0, Amount::ZERO, BTreeMap::new(), outputs)
            .unwrap()
            .compute_m6id()
    }

    // The withdrawal bundle's m6id must not depend on the order in which withdrawals
    // were aggregated (HashMap iteration order is randomized per process). Before the
    // total-order fix, the comparator was non-transitive and this failed.
    #[test]
    fn m6id_is_independent_of_aggregation_order() {
        let a = aw(1, 3);
        let b = aw(3, 2);
        let c = aw(2, 1);
        let m = bundle_m6id(vec![a.clone(), b.clone(), c.clone()]);
        for perm in [
            vec![c.clone(), b.clone(), a.clone()],
            vec![b.clone(), a.clone(), c.clone()],
            vec![a.clone(), c.clone(), b.clone()],
        ] {
            assert_eq!(
                m,
                bundle_m6id(perm),
                "m6id must not depend on aggregation order"
            );
        }
    }
}
