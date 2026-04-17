use std::collections::{BTreeMap, HashMap, HashSet};
use std::num::NonZeroU32;

use fallible_iterator::FallibleIterator;
use futures::Stream;
use heed::types::SerdeBincode;
use serde::{Deserialize, Serialize};
use sneed::{DatabaseUnique, RoDatabaseUnique, RoTxn, RwTxn, UnitKey};

use crate::{
    types::{
        Address, AmountOverflowError, Authorized, AuthorizedTransaction,
        BlockHash, Body, FilledOutput, FilledTransaction, GetBitcoinValue as _,
        Header, InPoint, M6id, MerkleRoot, OutPoint, OutPointKey, SpentOutput,
        Transaction, VERSION, Version, WithdrawalBundle,
        WithdrawalBundleStatus, proto::mainchain::TwoWayPegData,
    },
    util::Watchable,
    validation::DecisionValidationInterface,
};

pub trait UtxoManager {
    fn insert_utxo(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
        filled_output: &FilledOutput,
    ) -> Result<(), Error>;
    fn delete_utxo(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
    ) -> Result<bool, Error>;
    fn clear_utxos(&self, rwtxn: &mut RwTxn) -> Result<(), Error>;
}

pub mod block;
pub mod decisions;
pub mod error;
pub mod markets;
mod rollback;
pub mod type_aliases;
pub mod undo;
use decisions::{Decision, DecisionId};
pub mod reputation;
mod two_way_peg_data;
pub mod voting;

pub use decisions::period_to_name;
pub use error::Error;
pub use markets::{
    Market, MarketBuilder, MarketId, MarketState, MarketsDatabase, ShareAccount,
};
use rollback::{HeightStamped, RollBack};
pub use voting::VotingSystem;

pub const WITHDRAWAL_BUNDLE_FAILURE_GAP: u32 = 4;

/// Prevalidated block data containing computed values from validation
/// to avoid redundant computation during connection
pub struct PrevalidatedBlock {
    pub filled_transactions: Vec<FilledTransaction>,
    pub computed_merkle_root: MerkleRoot,
    pub coinbase_value: bitcoin::Amount,
    pub next_height: u32,
}

#[derive(Debug, Deserialize, Serialize)]
enum WithdrawalBundleInfo {
    Known(WithdrawalBundle),
    Unknown,
    UnknownConfirmed {
        spend_utxos: BTreeMap<OutPoint, FilledOutput>,
    },
}

impl WithdrawalBundleInfo {
    fn is_known(&self) -> bool {
        match self {
            Self::Known(_) => true,
            Self::Unknown | Self::UnknownConfirmed { .. } => false,
        }
    }
}

type WithdrawalBundlesDb = DatabaseUnique<
    SerdeBincode<M6id>,
    SerdeBincode<(
        WithdrawalBundleInfo,
        RollBack<HeightStamped<WithdrawalBundleStatus>>,
    )>,
>;

#[derive(Clone)]
pub struct State {
    tip: DatabaseUnique<UnitKey, SerdeBincode<BlockHash>>,
    height: DatabaseUnique<UnitKey, SerdeBincode<u32>>,
    mainchain_timestamp: DatabaseUnique<UnitKey, SerdeBincode<u64>>,
    genesis_timestamp: DatabaseUnique<UnitKey, SerdeBincode<u64>>,
    reputation: reputation::ReputationDbs,
    decisions: decisions::Dbs,
    markets: MarketsDatabase,
    voting: VotingSystem,
    utxos: DatabaseUnique<OutPointKey, SerdeBincode<FilledOutput>>,
    utxos_by_address:
        DatabaseUnique<SerdeBincode<(Address, OutPoint)>, SerdeBincode<()>>,
    stxos: DatabaseUnique<OutPointKey, SerdeBincode<SpentOutput>>,
    pending_withdrawal_bundle:
        DatabaseUnique<UnitKey, SerdeBincode<(WithdrawalBundle, u32)>>,
    latest_failed_withdrawal_bundle:
        DatabaseUnique<UnitKey, SerdeBincode<RollBack<HeightStamped<M6id>>>>,
    withdrawal_bundles: WithdrawalBundlesDb,
    deposit_blocks: DatabaseUnique<
        SerdeBincode<u32>,
        SerdeBincode<(bitcoin::BlockHash, u32)>,
    >,
    withdrawal_bundle_event_blocks: DatabaseUnique<
        SerdeBincode<u32>,
        SerdeBincode<(bitcoin::BlockHash, u32)>,
    >,
    _version: DatabaseUnique<UnitKey, SerdeBincode<Version>>,
    // Undo databases for disconnect_tip chain reorganization support
    ossification_undo: DatabaseUnique<
        SerdeBincode<u32>,
        SerdeBincode<undo::OssificationUndoData>,
    >,
    consensus_undo: DatabaseUnique<
        SerdeBincode<u32>,
        SerdeBincode<undo::ConsensusUndoData>,
    >,
    consolidation_undo: DatabaseUnique<
        SerdeBincode<u32>,
        SerdeBincode<undo::ConsolidationUndoData>,
    >,
    pub(crate) minting_undo:
        DatabaseUnique<SerdeBincode<u32>, SerdeBincode<u32>>,
    reputation_transfer_undo: DatabaseUnique<
        SerdeBincode<u32>,
        SerdeBincode<undo::ReputationTransferUndoData>,
    >,
}

impl DecisionValidationInterface for State {
    fn validate_decision_claim(
        &self,
        rotxn: &RoTxn,
        decision_id: DecisionId,
        decision: &Decision,
        current_ts: u64,
        current_height: Option<u32>,
        genesis_ts: u64,
    ) -> Result<(), Error> {
        self.decisions().validate_decision_claim(
            rotxn,
            decision_id,
            decision,
            current_ts,
            current_height,
            genesis_ts,
        )
    }

    fn try_get_height(&self, rotxn: &RoTxn) -> Result<Option<u32>, Error> {
        self.try_get_height(rotxn)
    }

    fn try_get_genesis_timestamp(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<u64>, Error> {
        self.try_get_genesis_timestamp(rotxn)
    }

    fn try_get_mainchain_timestamp(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<u64>, Error> {
        self.try_get_mainchain_timestamp(rotxn)
    }

    fn get_standard_claimed_count_in_period(
        &self,
        rotxn: &RoTxn,
        period_index: u32,
    ) -> Result<u64, Error> {
        self.decisions()
            .get_standard_claimed_count_in_period(rotxn, period_index)
    }

    fn get_available_decisions(
        &self,
        rotxn: &RoTxn,
        period: u32,
        current_ts: u64,
        current_height: Option<u32>,
        genesis_ts: u64,
    ) -> Result<u64, Error> {
        self.decisions().get_available_decisions(
            rotxn,
            period,
            current_ts,
            current_height,
            genesis_ts,
        )
    }
}

impl State {
    const BASE_DBS: u32 = 13;
    const UNDO_DBS: u32 = 5;

    pub const NUM_DBS: u32 = reputation::ReputationDbs::NUM_DBS
        + decisions::Dbs::NUM_DBS
        + MarketsDatabase::NUM_DBS
        + VotingSystem::NUM_DBS
        + Self::BASE_DBS
        + Self::UNDO_DBS;

    pub fn new(
        env: &sneed::Env,
        decision_config_testing: Option<u32>,
    ) -> Result<Self, Error> {
        let mut rwtxn = env.write_txn()?;
        let tip = DatabaseUnique::create(env, &mut rwtxn, "tip")?;
        let height = DatabaseUnique::create(env, &mut rwtxn, "height")?;
        let mainchain_timestamp =
            DatabaseUnique::create(env, &mut rwtxn, "mainchain_timestamp")?;
        let genesis_timestamp =
            DatabaseUnique::create(env, &mut rwtxn, "genesis_timestamp")?;
        let reputation = reputation::ReputationDbs::new(env, &mut rwtxn)?;
        let decisions = if let Some(blocks_per_period) = decision_config_testing
        {
            let nz = NonZeroU32::new(blocks_per_period).ok_or_else(|| {
                Error::InvalidTransaction {
                    reason: "decision_config_testing blocks_per_period \
                             must be > 0"
                        .into(),
                }
            })?;
            decisions::Dbs::new_with_config(
                env,
                &mut rwtxn,
                decisions::DecisionConfig::testing(nz),
            )?
        } else {
            decisions::Dbs::new(env, &mut rwtxn)?
        };
        let markets = MarketsDatabase::new(env, &mut rwtxn)?;
        let voting = VotingSystem::new(env, &mut rwtxn)?;
        let utxos = DatabaseUnique::create(env, &mut rwtxn, "utxos")?;
        let utxos_by_address =
            DatabaseUnique::create(env, &mut rwtxn, "utxos_by_address")?;
        let stxos = DatabaseUnique::create(env, &mut rwtxn, "stxos")?;
        let pending_withdrawal_bundle = DatabaseUnique::create(
            env,
            &mut rwtxn,
            "pending_withdrawal_bundle",
        )?;
        let latest_failed_withdrawal_bundle = DatabaseUnique::create(
            env,
            &mut rwtxn,
            "latest_failed_withdrawal_bundle",
        )?;
        let withdrawal_bundles =
            DatabaseUnique::create(env, &mut rwtxn, "withdrawal_bundles")?;
        let deposit_blocks =
            DatabaseUnique::create(env, &mut rwtxn, "deposit_blocks")?;
        let withdrawal_bundle_event_blocks = DatabaseUnique::create(
            env,
            &mut rwtxn,
            "withdrawal_bundle_event_blocks",
        )?;
        let version = DatabaseUnique::create(env, &mut rwtxn, "state_version")?;
        if version.try_get(&rwtxn, &())?.is_none() {
            version.put(&mut rwtxn, &(), &*VERSION)?;
        }
        let ossification_undo =
            DatabaseUnique::create(env, &mut rwtxn, "ossification_undo")?;
        let consensus_undo =
            DatabaseUnique::create(env, &mut rwtxn, "consensus_undo")?;
        let consolidation_undo =
            DatabaseUnique::create(env, &mut rwtxn, "consolidation_undo")?;
        let minting_undo =
            DatabaseUnique::create(env, &mut rwtxn, "minting_undo")?;
        let reputation_transfer_undo = DatabaseUnique::create(
            env,
            &mut rwtxn,
            "reputation_transfer_undo",
        )?;
        rwtxn.commit()?;
        Ok(Self {
            tip,
            height,
            mainchain_timestamp,
            genesis_timestamp,
            reputation,
            decisions,
            markets,
            voting,
            utxos,
            utxos_by_address,
            stxos,
            pending_withdrawal_bundle,
            latest_failed_withdrawal_bundle,
            withdrawal_bundles,
            withdrawal_bundle_event_blocks,
            deposit_blocks,
            _version: version,
            ossification_undo,
            consensus_undo,
            consolidation_undo,
            minting_undo,
            reputation_transfer_undo,
        })
    }

    pub fn reputation(&self) -> &reputation::ReputationDbs {
        &self.reputation
    }

    pub fn decisions(&self) -> &decisions::Dbs {
        &self.decisions
    }

    pub fn markets(&self) -> &MarketsDatabase {
        &self.markets
    }

    pub fn voting(&self) -> &VotingSystem {
        &self.voting
    }

    pub fn deposit_blocks(
        &self,
    ) -> &RoDatabaseUnique<
        SerdeBincode<u32>,
        SerdeBincode<(bitcoin::BlockHash, u32)>,
    > {
        &self.deposit_blocks
    }

    pub fn stxos(
        &self,
    ) -> &RoDatabaseUnique<OutPointKey, SerdeBincode<SpentOutput>> {
        &self.stxos
    }

    pub fn withdrawal_bundle_event_blocks(
        &self,
    ) -> &RoDatabaseUnique<
        SerdeBincode<u32>,
        SerdeBincode<(bitcoin::BlockHash, u32)>,
    > {
        &self.withdrawal_bundle_event_blocks
    }

    pub fn try_get_tip(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<BlockHash>, Error> {
        let tip = self.tip.try_get(rotxn, &())?;
        Ok(tip)
    }

    pub fn try_get_height(&self, rotxn: &RoTxn) -> Result<Option<u32>, Error> {
        let height = self.height.try_get(rotxn, &())?;
        Ok(height)
    }

    pub fn try_get_mainchain_timestamp(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<u64>, Error> {
        let timestamp = self.mainchain_timestamp.try_get(rotxn, &())?;
        Ok(timestamp)
    }

    pub fn try_get_genesis_timestamp(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<u64>, Error> {
        let timestamp = self.genesis_timestamp.try_get(rotxn, &())?;
        Ok(timestamp)
    }

    pub fn get_utxos(
        &self,
        rotxn: &RoTxn,
    ) -> Result<HashMap<OutPoint, FilledOutput>, Error> {
        let utxos: HashMap<OutPoint, FilledOutput> = self
            .utxos
            .iter(rotxn)?
            .map(|(key, output)| Ok((key.to_outpoint(), output)))
            .collect()?;
        Ok(utxos)
    }

    pub fn get_utxos_by_addresses(
        &self,
        rotxn: &RoTxn,
        addresses: &HashSet<Address>,
    ) -> Result<HashMap<OutPoint, FilledOutput>, Error> {
        let mut result = HashMap::with_capacity(addresses.len() * 4);

        let mut iter = self.utxos_by_address.iter(rotxn)?;
        while let Some(((addr, outpoint), _)) = iter.next()? {
            if addresses.contains(&addr)
                && let Some(filled_output) = self
                    .utxos
                    .try_get(rotxn, &OutPointKey::from_outpoint(&outpoint))?
            {
                result.insert(outpoint, filled_output);
            }
        }

        Ok(result)
    }
}

impl UtxoManager for State {
    fn insert_utxo(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
        filled_output: &FilledOutput,
    ) -> Result<(), Error> {
        let key = OutPointKey::from_outpoint(outpoint);
        self.utxos.put(rwtxn, &key, filled_output)?;
        self.utxos_by_address.put(
            rwtxn,
            &(filled_output.address, *outpoint),
            &(),
        )?;
        Ok(())
    }

    fn delete_utxo(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
    ) -> Result<bool, Error> {
        let key = OutPointKey::from_outpoint(outpoint);
        let filled_output =
            if let Some(output) = self.utxos.try_get(rwtxn, &key)? {
                output
            } else {
                return Ok(false);
            };

        self.utxos_by_address
            .delete(rwtxn, &(filled_output.address, *outpoint))?;

        let deleted = self.utxos.delete(rwtxn, &key)?;

        if !deleted {
            // Restore address index
            self.utxos_by_address.put(
                rwtxn,
                &(filled_output.address, *outpoint),
                &(),
            )?;
            return Ok(false);
        }

        Ok(true)
    }

    fn clear_utxos(&self, rwtxn: &mut RwTxn) -> Result<(), Error> {
        self.utxos.clear(rwtxn)?;
        self.utxos_by_address.clear(rwtxn)?;
        Ok(())
    }
}

impl State {
    pub fn get_mempool_shares(
        &self,
        rotxn: &RoTxn,
        market_id: &crate::state::markets::MarketId,
    ) -> Result<Option<ndarray::Array1<i64>>, Error> {
        self.markets.get_mempool_shares(rotxn, market_id)
    }

    pub fn put_mempool_shares(
        &self,
        rwtxn: &mut RwTxn,
        market_id: &crate::state::markets::MarketId,
        shares: &ndarray::Array1<i64>,
    ) -> Result<(), Error> {
        self.markets.put_mempool_shares(rwtxn, market_id, shares)
    }

    pub fn clear_mempool_shares(
        &self,
        rwtxn: &mut RwTxn,
        market_id: &crate::state::markets::MarketId,
    ) -> Result<(), Error> {
        self.markets.clear_mempool_shares(rwtxn, market_id)
    }

    pub fn get_mempool_treasury_delta(
        &self,
        rotxn: &RoTxn,
        market_id: &crate::state::markets::MarketId,
    ) -> Result<u64, Error> {
        self.markets.get_mempool_treasury_delta(rotxn, market_id)
    }

    pub fn put_mempool_treasury_delta(
        &self,
        rwtxn: &mut RwTxn,
        market_id: &crate::state::markets::MarketId,
        delta: u64,
    ) -> Result<(), Error> {
        self.markets
            .put_mempool_treasury_delta(rwtxn, market_id, delta)
    }

    pub fn clear_mempool_treasury_delta(
        &self,
        rwtxn: &mut RwTxn,
        market_id: &crate::state::markets::MarketId,
    ) -> Result<(), Error> {
        self.markets.clear_mempool_treasury_delta(rwtxn, market_id)
    }

    pub fn get_latest_failed_withdrawal_bundle(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<(u32, M6id)>, Error> {
        let Some(latest_failed_m6id) =
            self.latest_failed_withdrawal_bundle.try_get(rotxn, &())?
        else {
            return Ok(None);
        };
        let latest_failed_m6id = latest_failed_m6id.latest().value;
        let (_bundle, bundle_status) = self
            .withdrawal_bundles
            .try_get(rotxn, &latest_failed_m6id)?
            .ok_or_else(|| {
                Error::DatabaseError(format!(
                    "latest failed m6id {latest_failed_m6id} \
                     not found in withdrawal_bundles"
                ))
            })?;
        let bundle_status = bundle_status.latest();
        if bundle_status.value != WithdrawalBundleStatus::Failed {
            return Err(Error::DatabaseError(format!(
                "latest failed bundle has status {:?}, expected Failed",
                bundle_status.value,
            )));
        }
        Ok(Some((bundle_status.height, latest_failed_m6id)))
    }

    pub fn fill_transaction(
        &self,
        rotxn: &RoTxn,
        transaction: &Transaction,
    ) -> Result<FilledTransaction, Error> {
        let mut spent_utxos = Vec::with_capacity(transaction.inputs.len());
        for input in &transaction.inputs {
            let key = OutPointKey::from_outpoint(input);
            let utxo = self
                .utxos
                .try_get(rotxn, &key)?
                .ok_or(Error::NoUtxo { outpoint: *input })?;
            spent_utxos.push(utxo);
        }
        Ok(FilledTransaction {
            spent_utxos,
            transaction: transaction.clone(),
            actor_address: None,
        })
    }

    pub fn fill_transaction_from_stxos(
        &self,
        rotxn: &RoTxn,
        tx: Transaction,
    ) -> Result<FilledTransaction, Error> {
        let txid = tx.txid();
        let mut spent_utxos = Vec::with_capacity(tx.inputs.len());
        for (vin, input) in tx.inputs.iter().enumerate().rev() {
            let key = OutPointKey::from_outpoint(input);
            let stxo = self
                .stxos
                .try_get(rotxn, &key)?
                .ok_or(Error::NoStxo { outpoint: *input })?;
            assert_eq!(
                stxo.inpoint,
                InPoint::Regular {
                    txid,
                    vin: vin as u32
                }
            );
            spent_utxos.push(stxo.output);
        }
        spent_utxos.reverse();
        Ok(FilledTransaction {
            spent_utxos,
            transaction: tx,
            actor_address: None,
        })
    }

    pub fn fill_authorized_transaction(
        &self,
        rotxn: &RoTxn,
        transaction: AuthorizedTransaction,
    ) -> Result<Authorized<FilledTransaction>, Error> {
        let filled_tx =
            self.fill_transaction(rotxn, &transaction.transaction)?;
        let authorizations = transaction.authorizations;
        Ok(Authorized {
            transaction: filled_tx,
            authorizations,
            actor_proof: transaction.actor_proof,
        })
    }

    pub fn get_pending_withdrawal_bundle(
        &self,
        txn: &RoTxn,
    ) -> Result<Option<(WithdrawalBundle, u32)>, Error> {
        Ok(self.pending_withdrawal_bundle.try_get(txn, &())?)
    }

    pub fn validate_filled_transaction(
        &self,
        archive: &crate::archive::Archive,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        override_height: Option<u32>,
    ) -> Result<bitcoin::Amount, Error> {
        crate::validation::BlockValidator::validate_filled_transaction(
            self,
            archive,
            rotxn,
            tx,
            override_height,
        )
    }

    pub fn validate_transaction(
        &self,
        archive: &crate::archive::Archive,
        rotxn: &RoTxn,
        transaction: &AuthorizedTransaction,
    ) -> Result<bitcoin::Amount, Error> {
        crate::validation::BlockValidator::validate_transaction(
            self,
            archive,
            rotxn,
            transaction,
        )
    }

    pub fn get_last_deposit_block_hash(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<bitcoin::BlockHash>, Error> {
        let block_hash = self
            .deposit_blocks
            .last(rotxn)?
            .map(|(_, (block_hash, _))| block_hash);
        Ok(block_hash)
    }

    pub fn get_last_withdrawal_bundle_event_block_hash(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<bitcoin::BlockHash>, Error> {
        let block_hash = self
            .withdrawal_bundle_event_blocks
            .last(rotxn)?
            .map(|(_, (block_hash, _))| block_hash);
        Ok(block_hash)
    }

    /// Compute the total value of all deposit UTXOs (on-demand, no cache).
    pub fn compute_deposit_utxo_value(
        &self,
        rotxn: &RoTxn,
    ) -> Result<u64, Error> {
        let mut total = 0u64;
        let mut iter = self.utxos.iter(rotxn)?;
        while let Some((outpoint_key, filled_output)) = iter.next()? {
            let outpoint = outpoint_key.to_outpoint();
            if matches!(outpoint, OutPoint::Deposit(_)) {
                total = total
                    .saturating_add(filled_output.get_bitcoin_value().to_sat());
            }
        }
        Ok(total)
    }

    /// Compute the total value of all spent deposit outputs (on-demand, no cache).
    pub fn compute_deposit_stxo_value(
        &self,
        rotxn: &RoTxn,
    ) -> Result<u64, Error> {
        let mut total = 0u64;
        let mut iter = self.stxos.iter(rotxn)?;
        while let Some((outpoint_key, spent_output)) = iter.next()? {
            let outpoint = outpoint_key.to_outpoint();
            if matches!(outpoint, OutPoint::Deposit(_)) {
                total = total.saturating_add(
                    spent_output.output.get_bitcoin_value().to_sat(),
                );
            }
        }
        Ok(total)
    }

    /// Compute the total value of all withdrawal spent outputs (on-demand, no cache).
    pub fn compute_withdrawal_stxo_value(
        &self,
        rotxn: &RoTxn,
    ) -> Result<u64, Error> {
        let mut total = 0u64;
        let mut iter = self.stxos.iter(rotxn)?;
        while let Some((_outpoint_key, spent_output)) = iter.next()? {
            if matches!(spent_output.inpoint, InPoint::Withdrawal { .. }) {
                total = total.saturating_add(
                    spent_output.output.get_bitcoin_value().to_sat(),
                );
            }
        }
        Ok(total)
    }

    pub fn sidechain_wealth(
        &self,
        rotxn: &RoTxn,
    ) -> Result<bitcoin::Amount, Error> {
        let total_deposit_utxo_value =
            bitcoin::Amount::from_sat(self.compute_deposit_utxo_value(rotxn)?);
        let total_deposit_stxo_value =
            bitcoin::Amount::from_sat(self.compute_deposit_stxo_value(rotxn)?);
        let total_withdrawal_stxo_value = bitcoin::Amount::from_sat(
            self.compute_withdrawal_stxo_value(rotxn)?,
        );

        let total_wealth = total_deposit_utxo_value
            .checked_add(total_deposit_stxo_value)
            .and_then(|sum| sum.checked_sub(total_withdrawal_stxo_value))
            .ok_or(AmountOverflowError)?;

        Ok(total_wealth)
    }

    pub fn prevalidate_block(
        &self,
        archive: &crate::archive::Archive,
        rotxn: &RoTxn,
        header: &Header,
        body: &Body,
    ) -> Result<PrevalidatedBlock, Error> {
        crate::validation::BlockValidator::prevalidate(
            self, archive, rotxn, header, body,
        )
    }

    pub fn connect_prevalidated_block(
        &self,
        rwtxn: &mut RwTxn,
        header: &Header,
        body: &Body,
        mainchain_timestamp: u64,
        prevalidated: PrevalidatedBlock,
    ) -> Result<(), Error> {
        block::connect_prevalidated(
            self,
            rwtxn,
            header,
            body,
            mainchain_timestamp,
            prevalidated,
        )
    }

    pub fn apply_block(
        &self,
        archive: &crate::archive::Archive,
        rwtxn: &mut RwTxn,
        header: &Header,
        body: &Body,
        mainchain_timestamp: u64,
    ) -> Result<(), Error> {
        let prevalidated =
            self.prevalidate_block(archive, rwtxn, header, body)?;
        self.connect_prevalidated_block(
            rwtxn,
            header,
            body,
            mainchain_timestamp,
            prevalidated,
        )
    }

    pub fn disconnect_tip(
        &self,
        rwtxn: &mut RwTxn,
        header: &Header,
        body: &Body,
    ) -> Result<(), Error> {
        block::disconnect_tip(self, rwtxn, header, body)
    }

    pub fn connect_two_way_peg_data(
        &self,
        rwtxn: &mut RwTxn,
        two_way_peg_data: &TwoWayPegData,
    ) -> Result<(), Error> {
        two_way_peg_data::connect(self, rwtxn, two_way_peg_data)
    }

    pub fn disconnect_two_way_peg_data(
        &self,
        rwtxn: &mut RwTxn,
        two_way_peg_data: &TwoWayPegData,
    ) -> Result<(), Error> {
        two_way_peg_data::disconnect(self, rwtxn, two_way_peg_data)
    }

    fn period_context(
        &self,
        rotxn: &RoTxn,
    ) -> Result<(u64, Option<u32>, u64), Error> {
        let current_ts = self.try_get_mainchain_timestamp(rotxn)?.unwrap_or(0);
        let current_height = self.try_get_height(rotxn)?;
        let genesis_ts = self.try_get_genesis_timestamp(rotxn)?.unwrap_or(0);
        Ok((current_ts, current_height, genesis_ts))
    }

    pub fn get_all_decision_periods(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Vec<(u32, u64)>, Error> {
        let (current_ts, current_height, genesis_ts) =
            self.period_context(rotxn)?;
        self.decisions.get_active_periods(
            rotxn,
            current_ts,
            current_height,
            genesis_ts,
        )
    }

    pub fn get_decisions_for_period(
        &self,
        rotxn: &RoTxn,
        period: u32,
    ) -> Result<u64, Error> {
        let (current_ts, current_height, genesis_ts) =
            self.period_context(rotxn)?;
        self.decisions.total_for(
            rotxn,
            period,
            current_ts,
            current_height,
            genesis_ts,
        )
    }

    pub fn get_available_decisions_in_period(
        &self,
        rotxn: &RoTxn,
        period_index: u32,
    ) -> Result<Vec<crate::state::decisions::DecisionId>, Error> {
        let (current_ts, current_height, genesis_ts) =
            self.period_context(rotxn)?;
        self.decisions.get_available_decisions_in_period(
            rotxn,
            period_index,
            current_ts,
            current_height,
            genesis_ts,
        )
    }

    pub fn get_ossified_decisions(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Vec<crate::state::decisions::DecisionEntry>, Error> {
        let (current_ts, current_height, genesis_ts) =
            self.period_context(rotxn)?;
        self.decisions.get_ossified_decisions(
            rotxn,
            current_ts,
            current_height,
            genesis_ts,
        )
    }

    pub fn is_decision_in_voting(
        &self,
        rotxn: &RoTxn,
        decision_id: crate::state::decisions::DecisionId,
    ) -> Result<bool, Error> {
        self.decisions.is_decision_in_voting(rotxn, decision_id)
    }

    pub fn get_voting_periods(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Vec<(u32, u64, u64)>, Error> {
        let (current_ts, current_height, genesis_ts) =
            self.period_context(rotxn)?;
        self.decisions.get_voting_periods(
            rotxn,
            current_ts,
            current_height,
            genesis_ts,
        )
    }

    pub fn get_period_summary(
        &self,
        rotxn: &RoTxn,
    ) -> Result<type_aliases::PeriodSummary, Error> {
        let (current_ts, current_height, genesis_ts) =
            self.period_context(rotxn)?;
        self.decisions.get_period_summary(
            rotxn,
            current_ts,
            current_height,
            genesis_ts,
        )
    }

    pub fn claimed_count_in_period(
        &self,
        rotxn: &RoTxn,
        period_index: u32,
    ) -> Result<u64, Error> {
        self.decisions.claimed_count_in_period(rotxn, period_index)
    }
}

impl Watchable<()> for State {
    type WatchStream = impl Stream<Item = ()>;
    fn watch(&self) -> Self::WatchStream {
        tokio_stream::wrappers::WatchStream::new(self.tip.watch().clone())
    }
}
