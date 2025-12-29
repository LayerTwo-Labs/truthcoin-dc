use std::collections::{BTreeMap, HashMap, HashSet};

use fallible_iterator::FallibleIterator;
use futures::Stream;
use heed::types::SerdeBincode;
use serde::{Deserialize, Serialize};
use sneed::{DatabaseUnique, RoDatabaseUnique, RoTxn, RwTxn, UnitKey};

use crate::{
    authorization::Authorization,
    types::{
        Address, AmountOverflowError, Authorized, AuthorizedTransaction,
        BlockHash, Body, FilledOutput, FilledTransaction, GetAddress as _,
        GetBitcoinValue as _, Header, InPoint, M6id, MerkleRoot, OutPoint,
        SpentOutput, Transaction, VERSION, Verify as _, Version,
        WithdrawalBundle, WithdrawalBundleStatus,
        proto::mainchain::TwoWayPegData,
    },
    util::Watchable,
    validation::{MarketValidator, SlotValidationInterface, SlotValidator},
};

pub trait UtxoManager {
    fn insert_utxo_with_address_index(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
        filled_output: &FilledOutput,
    ) -> Result<(), Error>;
    fn delete_utxo_with_address_index(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
    ) -> Result<bool, Error>;
    fn clear_utxos_and_address_index(
        &self,
        rwtxn: &mut RwTxn,
    ) -> Result<(), Error>;

    fn insert_utxo_supply_neutral(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
        filled_output: &FilledOutput,
    ) -> Result<(), Error>;

    fn delete_utxo_supply_neutral(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
    ) -> Result<bool, Error>;
}

pub mod block;
pub mod error;
pub mod markets;
mod rollback;
pub mod slots;
pub mod type_aliases;
use slots::{Decision, SlotId};
mod two_way_peg_data;
pub mod votecoin;
pub mod voting;

pub use error::Error;
pub use markets::{
    BatchedMarketTrade, Market, MarketBuilder, MarketId, MarketState,
    MarketsDatabase, ShareAccount,
};
use rollback::{HeightStamped, RollBack};
pub use slots::{period_to_name, timestamp_to_period};
pub use voting::VotingSystem;

pub const WITHDRAWAL_BUNDLE_FAILURE_GAP: u32 = 4;

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
    votecoin: votecoin::Dbs,
    slots: slots::Dbs,
    markets: MarketsDatabase,
    voting: VotingSystem,
    utxos: DatabaseUnique<SerdeBincode<OutPoint>, SerdeBincode<FilledOutput>>,
    utxos_by_address:
        DatabaseUnique<SerdeBincode<(Address, OutPoint)>, SerdeBincode<()>>,
    stxos: DatabaseUnique<SerdeBincode<OutPoint>, SerdeBincode<SpentOutput>>,
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
    votecoin_balances: DatabaseUnique<SerdeBincode<Address>, SerdeBincode<u32>>,
    _version: DatabaseUnique<UnitKey, SerdeBincode<Version>>,
}

impl SlotValidationInterface for State {
    fn validate_slot_claim(
        &self,
        rotxn: &RoTxn,
        slot_id: SlotId,
        decision: &Decision,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> Result<(), Error> {
        self.slots().validate_slot_claim(
            rotxn,
            slot_id,
            decision,
            current_ts,
            current_height,
        )
    }

    fn try_get_height(&self, rotxn: &RoTxn) -> Result<Option<u32>, Error> {
        self.try_get_height(rotxn)
    }
}

impl State {
    pub const NUM_DBS: u32 = votecoin::Dbs::NUM_DBS
        + slots::Dbs::NUM_DBS
        + MarketsDatabase::NUM_DBS
        + VotingSystem::NUM_DBS
        + 15;

    pub fn new(env: &sneed::Env) -> Result<Self, Error> {
        let mut rwtxn = env.write_txn()?;
        let tip = DatabaseUnique::create(env, &mut rwtxn, "tip")?;
        let height = DatabaseUnique::create(env, &mut rwtxn, "height")?;
        let mainchain_timestamp =
            DatabaseUnique::create(env, &mut rwtxn, "mainchain_timestamp")?;
        let votecoin = votecoin::Dbs::new(env, &mut rwtxn)?;
        let slots = slots::Dbs::new(env, &mut rwtxn)?;
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
        let votecoin_balances =
            DatabaseUnique::create(env, &mut rwtxn, "votecoin_balances")?;
        let version = DatabaseUnique::create(env, &mut rwtxn, "state_version")?;
        if version.try_get(&rwtxn, &())?.is_none() {
            version.put(&mut rwtxn, &(), &*VERSION)?;
        }
        rwtxn.commit()?;
        Ok(Self {
            tip,
            height,
            mainchain_timestamp,
            votecoin,
            slots,
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
            votecoin_balances,
            _version: version,
        })
    }

    pub fn votecoin(&self) -> &votecoin::Dbs {
        &self.votecoin
    }

    pub fn slots(&self) -> &slots::Dbs {
        &self.slots
    }

    pub fn markets(&self) -> &MarketsDatabase {
        &self.markets
    }

    pub fn voting(&self) -> &VotingSystem {
        &self.voting
    }

    pub fn voting_mut(&mut self) -> &mut VotingSystem {
        &mut self.voting
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
    ) -> &RoDatabaseUnique<SerdeBincode<OutPoint>, SerdeBincode<SpentOutput>>
    {
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

    pub fn get_utxos(
        &self,
        rotxn: &RoTxn,
    ) -> Result<HashMap<OutPoint, FilledOutput>, Error> {
        let utxos = self.utxos.iter(rotxn)?.collect()?;
        Ok(utxos)
    }

    pub fn get_utxos_by_addresses(
        &self,
        rotxn: &RoTxn,
        addresses: &HashSet<Address>,
    ) -> Result<HashMap<OutPoint, FilledOutput>, Error> {
        // NOTE: This implementation scans all UTXOs and filters by address.
        // For optimal performance, sneed would need to expose heed's range() method
        // to allow per-address range queries on the (Address, OutPoint) composite key.
        // This is O(total_utxos) instead of O(matching_utxos).
        let mut result = HashMap::with_capacity(addresses.len() * 4);

        let mut iter = self.utxos_by_address.iter(rotxn)?;
        while let Some(((addr, outpoint), _)) = iter.next()? {
            if addresses.contains(&addr)
                && let Some(filled_output) =
                    self.utxos.try_get(rotxn, &outpoint)?
            {
                result.insert(outpoint, filled_output);
            }
        }

        Ok(result)
    }
}

impl UtxoManager for State {
    fn insert_utxo_with_address_index(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
        filled_output: &FilledOutput,
    ) -> Result<(), Error> {
        self.utxos.put(rwtxn, outpoint, filled_output)?;

        self.utxos_by_address.put(
            rwtxn,
            &(filled_output.address, *outpoint),
            &(),
        )?;

        // Update votecoin balances index (not a cache - this is a denormalized index for O(1) lookups)
        if let crate::types::FilledOutputContent::Votecoin(amount) =
            &filled_output.content
        {
            let current_balance = self
                .votecoin_balances
                .try_get(rwtxn, &filled_output.address)?
                .unwrap_or(0);
            let new_balance = current_balance.saturating_add(*amount);
            self.votecoin_balances.put(
                rwtxn,
                &filled_output.address,
                &new_balance,
            )?;
        }

        Ok(())
    }

    fn delete_utxo_with_address_index(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
    ) -> Result<bool, Error> {
        let filled_output =
            if let Some(output) = self.utxos.try_get(rwtxn, outpoint)? {
                output
            } else {
                return Ok(false);
            };

        // Update votecoin balances index (not a cache - this is a denormalized index for O(1) lookups)
        if let crate::types::FilledOutputContent::Votecoin(amount) =
            &filled_output.content
        {
            let current_balance = self
                .votecoin_balances
                .try_get(rwtxn, &filled_output.address)?
                .unwrap_or(0);
            let new_balance = current_balance.saturating_sub(*amount);
            self.votecoin_balances.put(
                rwtxn,
                &filled_output.address,
                &new_balance,
            )?;
        }

        self.utxos_by_address
            .delete(rwtxn, &(filled_output.address, *outpoint))?;

        let deleted = self.utxos.delete(rwtxn, outpoint)?;

        if !deleted {
            // Restore address index
            self.utxos_by_address.put(
                rwtxn,
                &(filled_output.address, *outpoint),
                &(),
            )?;
            // Restore votecoin balance
            if let crate::types::FilledOutputContent::Votecoin(amount) =
                &filled_output.content
            {
                let current_balance = self
                    .votecoin_balances
                    .try_get(rwtxn, &filled_output.address)?
                    .unwrap_or(0);
                let restored_balance = current_balance.saturating_add(*amount);
                self.votecoin_balances.put(
                    rwtxn,
                    &filled_output.address,
                    &restored_balance,
                )?;
            }
            return Ok(false);
        }

        Ok(true)
    }

    fn clear_utxos_and_address_index(
        &self,
        rwtxn: &mut RwTxn,
    ) -> Result<(), Error> {
        self.utxos.clear(rwtxn)?;
        self.utxos_by_address.clear(rwtxn)?;
        self.votecoin_balances.clear(rwtxn)?;
        Ok(())
    }

    fn insert_utxo_supply_neutral(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
        filled_output: &FilledOutput,
    ) -> Result<(), Error> {
        self.utxos.put(rwtxn, outpoint, filled_output)?;

        self.utxos_by_address.put(
            rwtxn,
            &(filled_output.address, *outpoint),
            &(),
        )?;

        if let crate::types::FilledOutputContent::Votecoin(amount) =
            &filled_output.content
        {
            let current_balance = self
                .votecoin_balances
                .try_get(rwtxn, &filled_output.address)?
                .unwrap_or(0);
            let new_balance = current_balance.saturating_add(*amount);
            self.votecoin_balances.put(
                rwtxn,
                &filled_output.address,
                &new_balance,
            )?;
        }

        Ok(())
    }

    fn delete_utxo_supply_neutral(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
    ) -> Result<bool, Error> {
        let filled_output =
            if let Some(output) = self.utxos.try_get(rwtxn, outpoint)? {
                output
            } else {
                return Ok(false);
            };

        if let crate::types::FilledOutputContent::Votecoin(amount) =
            &filled_output.content
        {
            let current_balance = self
                .votecoin_balances
                .try_get(rwtxn, &filled_output.address)?
                .unwrap_or(0);
            let new_balance = current_balance.saturating_sub(*amount);
            self.votecoin_balances.put(
                rwtxn,
                &filled_output.address,
                &new_balance,
            )?;
        }

        self.utxos_by_address
            .delete(rwtxn, &(filled_output.address, *outpoint))?;

        let deleted = self.utxos.delete(rwtxn, outpoint)?;

        if !deleted {
            self.utxos_by_address.put(
                rwtxn,
                &(filled_output.address, *outpoint),
                &(),
            )?;
            if let crate::types::FilledOutputContent::Votecoin(amount) =
                &filled_output.content
            {
                let current_balance = self
                    .votecoin_balances
                    .try_get(rwtxn, &filled_output.address)?
                    .unwrap_or(0);
                let restored_balance = current_balance.saturating_add(*amount);
                self.votecoin_balances.put(
                    rwtxn,
                    &filled_output.address,
                    &restored_balance,
                )?;
            }
        }

        Ok(deleted)
    }
}

impl State {
    pub fn get_mempool_shares(
        &self,
        rotxn: &RoTxn,
        market_id: &crate::state::markets::MarketId,
    ) -> Result<Option<ndarray::Array1<f64>>, Error> {
        self.markets.get_mempool_shares(rotxn, market_id)
    }

    pub fn put_mempool_shares(
        &self,
        rwtxn: &mut RwTxn,
        market_id: &crate::state::markets::MarketId,
        shares: &ndarray::Array1<f64>,
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
        let (_bundle, bundle_status) = self.withdrawal_bundles.try_get(rotxn, &latest_failed_m6id)?
            .expect("Inconsistent DBs: latest failed m6id should exist in withdrawal_bundles");
        let bundle_status = bundle_status.latest();
        assert_eq!(bundle_status.value, WithdrawalBundleStatus::Failed);
        Ok(Some((bundle_status.height, latest_failed_m6id)))
    }

    pub fn fill_transaction(
        &self,
        rotxn: &RoTxn,
        transaction: &Transaction,
    ) -> Result<FilledTransaction, Error> {
        let mut spent_utxos = vec![];
        for input in &transaction.inputs {
            let utxo = self
                .utxos
                .try_get(rotxn, input)?
                .ok_or(Error::NoUtxo { outpoint: *input })?;
            spent_utxos.push(utxo);
        }
        Ok(FilledTransaction {
            spent_utxos,
            transaction: transaction.clone(),
        })
    }

    pub fn fill_transaction_from_stxos(
        &self,
        rotxn: &RoTxn,
        tx: Transaction,
    ) -> Result<FilledTransaction, Error> {
        let txid = tx.txid();
        let mut spent_utxos = vec![];
        for (vin, input) in tx.inputs.iter().enumerate().rev() {
            let stxo = self
                .stxos
                .try_get(rotxn, input)?
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
        })
    }

    pub fn get_pending_withdrawal_bundle(
        &self,
        txn: &RoTxn,
    ) -> Result<Option<(WithdrawalBundle, u32)>, Error> {
        Ok(self.pending_withdrawal_bundle.try_get(txn, &())?)
    }

    pub fn validate_votecoin(
        &self,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        override_height: Option<u32>,
    ) -> Result<(), Error> {
        let votecoin_inputs: u32 = tx
            .spent_votecoin()
            .filter_map(|(_, output)| output.votecoin())
            .sum();
        let votecoin_outputs: u32 = tx
            .votecoin_outputs()
            .filter_map(|output| {
                if let crate::types::OutputContent::Votecoin(amount) =
                    &output.content
                {
                    Some(*amount)
                } else {
                    None
                }
            })
            .sum();

        let block_height = match override_height {
            Some(height) => height,
            None => self.try_get_height(rotxn)?.unwrap_or(0),
        };
        let is_genesis = block_height == 0;

        if is_genesis {
            Ok(())
        } else {
            if votecoin_inputs != votecoin_outputs {
                return Err(Error::UnbalancedVotecoin {
                    inputs: votecoin_inputs,
                    outputs: votecoin_outputs,
                });
            }
            Ok(())
        }
    }

    pub fn validate_decision_slot_claim(
        &self,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        override_height: Option<u32>,
    ) -> Result<(), Error> {
        SlotValidator::validate_complete_decision_slot_claim(
            self,
            rotxn,
            tx,
            override_height,
        )
    }

    pub fn validate_market_creation(
        &self,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        override_height: Option<u32>,
    ) -> Result<(), Error> {
        MarketValidator::validate_market_creation(
            self,
            rotxn,
            tx,
            override_height,
        )
    }

    pub fn validate_buy_shares(
        &self,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        override_height: Option<u32>,
    ) -> Result<(), Error> {
        MarketValidator::validate_buy_shares(self, rotxn, tx, override_height)
    }

    pub fn validate_filled_transaction(
        &self,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        override_height: Option<u32>,
    ) -> Result<bitcoin::Amount, Error> {
        use crate::validation::VoteValidator;

        let () = self.validate_votecoin(rotxn, tx, override_height)?;

        if tx.is_claim_decision_slot() {
            self.validate_decision_slot_claim(rotxn, tx, override_height)?;
        }

        if tx.is_create_market() {
            self.validate_market_creation(rotxn, tx, override_height)?;
        }

        if tx
            .transaction
            .data
            .as_ref()
            .is_some_and(|data| data.is_buy_shares())
        {
            self.validate_buy_shares(rotxn, tx, override_height)?;
        }

        if tx.is_submit_vote() {
            VoteValidator::validate_vote_submission(
                self,
                rotxn,
                tx,
                override_height,
            )?;
        }

        if tx.is_submit_vote_batch() {
            VoteValidator::validate_vote_batch(
                self,
                rotxn,
                tx,
                override_height,
            )?;
        }

        if tx.is_register_voter() {
            let voter_address = tx
                .spent_utxos
                .first()
                .ok_or_else(|| Error::InvalidTransaction {
                    reason: "Voter registration transaction must have inputs"
                        .to_string(),
                })?
                .address;

            let votecoin_balance =
                self.get_votecoin_balance(rotxn, &voter_address)?;
            if votecoin_balance == 0 {
                return Err(Error::InvalidTransaction {
                    reason: "Voter registration requires Votecoin balance"
                        .to_string(),
                });
            }
        }

        tx.bitcoin_fee()?.ok_or(Error::NotEnoughValueIn)
    }

    pub fn validate_transaction(
        &self,
        rotxn: &RoTxn,
        transaction: &AuthorizedTransaction,
    ) -> Result<bitcoin::Amount, Error> {
        let filled_transaction =
            self.fill_transaction(rotxn, &transaction.transaction)?;
        for (authorization, spent_utxo) in transaction
            .authorizations
            .iter()
            .zip(filled_transaction.spent_utxos.iter())
        {
            if authorization.get_address() != spent_utxo.address {
                return Err(Error::WrongPubKeyForAddress);
            }
        }
        if Authorization::verify_transaction(transaction).is_err() {
            return Err(Error::AuthorizationError);
        }
        let fee =
            self.validate_filled_transaction(rotxn, &filled_transaction, None)?;
        Ok(fee)
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
        while let Some((outpoint, filled_output)) = iter.next()? {
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
        while let Some((outpoint, spent_output)) = iter.next()? {
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
        while let Some((_outpoint, spent_output)) = iter.next()? {
            if matches!(spent_output.inpoint, InPoint::Withdrawal { .. }) {
                total = total.saturating_add(
                    spent_output.output.get_bitcoin_value().to_sat(),
                );
            }
        }
        Ok(total)
    }

    /// Compute the total votecoin supply (on-demand, no cache).
    pub fn compute_votecoin_supply(&self, rotxn: &RoTxn) -> Result<u32, Error> {
        let mut total = 0u32;
        let mut iter = self.votecoin_balances.iter(rotxn)?;
        while let Some((_address, balance)) = iter.next()? {
            total = total.saturating_add(balance);
        }
        Ok(total)
    }

    pub fn sidechain_wealth(
        &self,
        rotxn: &RoTxn,
    ) -> Result<bitcoin::Amount, Error> {
        // Compute values on-demand from source data (no caches)
        let deposit_utxo_value = self.compute_deposit_utxo_value(rotxn)?;
        let deposit_stxo_value = self.compute_deposit_stxo_value(rotxn)?;
        let withdrawal_stxo_value =
            self.compute_withdrawal_stxo_value(rotxn)?;

        let total_deposit_utxo_value =
            bitcoin::Amount::from_sat(deposit_utxo_value);
        let total_deposit_stxo_value =
            bitcoin::Amount::from_sat(deposit_stxo_value);
        let total_withdrawal_stxo_value =
            bitcoin::Amount::from_sat(withdrawal_stxo_value);

        let total_wealth = total_deposit_utxo_value
            .checked_add(total_deposit_stxo_value)
            .and_then(|sum| sum.checked_sub(total_withdrawal_stxo_value))
            .ok_or(AmountOverflowError)?;

        Ok(total_wealth)
    }

    pub fn validate_block(
        &self,
        rotxn: &RoTxn,
        header: &Header,
        body: &Body,
    ) -> Result<(bitcoin::Amount, Vec<FilledTransaction>, MerkleRoot), Error>
    {
        block::validate(self, rotxn, header, body)
    }

    pub fn connect_block(
        &self,
        rwtxn: &mut RwTxn,
        header: &Header,
        body: &Body,
        mainchain_timestamp: u64,
        filled_txs: Vec<FilledTransaction>,
        merkle_root: MerkleRoot,
    ) -> Result<(), Error> {
        block::connect(
            self,
            rwtxn,
            header,
            body,
            mainchain_timestamp,
            filled_txs,
            merkle_root,
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

    pub fn get_all_slot_quarters(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Vec<(u32, u64)>, Error> {
        let current_ts = self.try_get_mainchain_timestamp(rotxn)?.unwrap_or(0);
        let current_height = self.try_get_height(rotxn)?;
        self.slots
            .get_active_periods(rotxn, current_ts, current_height)
    }

    pub fn get_slots_for_quarter(
        &self,
        rotxn: &RoTxn,
        quarter: u32,
    ) -> Result<u64, Error> {
        let current_ts = self.try_get_mainchain_timestamp(rotxn)?.unwrap_or(0);
        let current_height = self.try_get_height(rotxn)?;
        self.slots
            .total_for(rotxn, quarter, current_ts, current_height)
    }

    pub fn get_available_slots_in_period(
        &self,
        rotxn: &RoTxn,
        period_index: u32,
    ) -> Result<Vec<crate::state::slots::SlotId>, Error> {
        let current_ts = self.try_get_mainchain_timestamp(rotxn)?.unwrap_or(0);
        let current_height = self.try_get_height(rotxn)?;
        self.slots.get_available_slots_in_period(
            rotxn,
            period_index,
            current_ts,
            current_height,
        )
    }

    pub fn get_ossified_slots(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Vec<crate::state::slots::Slot>, Error> {
        let current_ts = self.try_get_mainchain_timestamp(rotxn)?.unwrap_or(0);
        let current_height = self.try_get_height(rotxn)?;
        self.slots
            .get_ossified_slots(rotxn, current_ts, current_height)
    }

    pub fn is_slot_in_voting(
        &self,
        rotxn: &RoTxn,
        slot_id: crate::state::slots::SlotId,
    ) -> Result<bool, Error> {
        self.slots.is_slot_in_voting(rotxn, slot_id)
    }

    pub fn get_voting_periods(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Vec<(u32, u64, u64)>, Error> {
        let current_ts = self.try_get_mainchain_timestamp(rotxn)?.unwrap_or(0);
        let current_height = self.try_get_height(rotxn)?;
        self.slots
            .get_voting_periods(rotxn, current_ts, current_height)
    }

    pub fn get_period_summary(
        &self,
        rotxn: &RoTxn,
    ) -> Result<type_aliases::PeriodSummary, Error> {
        let current_ts = self.try_get_mainchain_timestamp(rotxn)?.unwrap_or(0);
        let current_height = self.try_get_height(rotxn)?;
        self.slots
            .get_period_summary(rotxn, current_ts, current_height)
    }

    pub fn get_claimed_slot_count_in_period(
        &self,
        rotxn: &RoTxn,
        period_index: u32,
    ) -> Result<u64, Error> {
        self.slots
            .get_claimed_slot_count_in_period(rotxn, period_index)
    }

    pub fn get_votecoin_balance(
        &self,
        rotxn: &RoTxn,
        address: &Address,
    ) -> Result<u32, Error> {
        Ok(self.votecoin_balances.try_get(rotxn, address)?.unwrap_or(0))
    }

    pub fn get_votecoin_balances_batch(
        &self,
        rotxn: &RoTxn,
        addresses: &HashSet<Address>,
    ) -> Result<HashMap<Address, u32>, Error> {
        let mut balances = HashMap::with_capacity(addresses.len());
        for &address in addresses {
            let balance = self
                .votecoin_balances
                .try_get(rotxn, &address)?
                .unwrap_or(0);
            balances.insert(address, balance);
        }
        Ok(balances)
    }

    pub fn get_total_votecoin_supply(
        &self,
        rotxn: &RoTxn,
    ) -> Result<u32, Error> {
        self.compute_votecoin_supply(rotxn)
    }

    pub fn get_votecoin_proportion(
        &self,
        rotxn: &RoTxn,
        address: &Address,
    ) -> Result<f64, Error> {
        let balance = self.get_votecoin_balance(rotxn, address)?;
        let total_supply = self.get_total_votecoin_supply(rotxn)?;
        if total_supply == 0 {
            return Ok(0.0);
        }
        Ok(balance as f64 / total_supply as f64)
    }

    pub fn get_votecoin_proportions_batch(
        &self,
        rotxn: &RoTxn,
        addresses: &HashSet<Address>,
    ) -> Result<HashMap<Address, f64>, Error> {
        let balances = self.get_votecoin_balances_batch(rotxn, addresses)?;
        let total_supply = self.get_total_votecoin_supply(rotxn)?;
        let mut proportions = HashMap::new();
        if total_supply == 0 {
            for &address in addresses {
                proportions.insert(address, 0.0);
            }
            return Ok(proportions);
        }
        for (&address, &balance) in &balances {
            let proportion = balance as f64 / total_supply as f64;
            proportions.insert(address, proportion);
        }
        Ok(proportions)
    }
}

impl Watchable<()> for State {
    type WatchStream = impl Stream<Item = ()>;
    fn watch(&self) -> Self::WatchStream {
        tokio_stream::wrappers::WatchStream::new(self.tip.watch().clone())
    }
}
