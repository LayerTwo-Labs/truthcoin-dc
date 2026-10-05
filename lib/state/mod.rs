//! Sidechain state as of the current sidechain tip

use std::{
    collections::{HashMap, HashSet},
    num::NonZeroU32,
};

use fallible_iterator::FallibleIterator as _;
use heed::types::SerdeBincode;
use sneed::{
    DatabaseUnique, RoTxn, RwTxn, UnitKey,
    db::error::{self as db_error, Error as DbError},
    env::Error as EnvError,
    rwtxn::Error as RwTxnError,
};

use crate::{
    types::{
        Accumulator, Address, AmountOverflowError, AmountUnderflowError,
        Authorized, AuthorizedTransaction, BlockHash, Body, FilledTransaction,
        GetAddress, GetValue, Header, InPoint, M6id, MerkleRoot, OutPoint,
        OutPointKey, Output, PointedOutput, PointedOutputRef, SpentOutput,
        Transaction, UtreexoNodeHash, UtreexoProof, VERSION, Version,
        WithdrawalBundle, WithdrawalBundleStatus,
        authorization::{self, BatchVerificationContext},
        proto::mainchain::TwoWayPegData,
        state::{TwoWayPegEvent, WithdrawalBundleInfo},
    },
    util::Watchable,
    validation::DecisionValidationInterface,
};

mod block;
pub mod decisions;
mod error;
pub mod markets;
pub mod reputation;
mod rollback;
mod two_way_peg_data;
pub mod type_aliases;
pub mod undo;
pub mod voting;

pub use decisions::period_to_name;
use decisions::{Decision, DecisionId};
pub use error::Error;
pub use markets::{
    Market, MarketBuilder, MarketId, MarketState, MarketsDatabase, ShareAccount,
};
use rollback::RollBack;
pub use voting::VotingSystem;

pub const WITHDRAWAL_BUNDLE_FAILURE_GAP: u32 = 4;

/// Prevalidated block data containing computed values from validation
/// to avoid redundant computation during connection
#[derive(Clone, Debug)]
pub struct PrevalidatedBlock {
    pub filled_transactions: Vec<FilledTransaction>,
    pub computed_merkle_root: MerkleRoot,
    pub total_fees: bitcoin::Amount,
    pub coinbase_value: bitcoin::Amount,
    /// Precomputed next height to avoid DB read in write txn
    pub next_height: u32,
    pub accumulator_diff: crate::types::AccumulatorDiff,
}

#[derive(Clone)]
pub struct State {
    /// Current tip
    tip: DatabaseUnique<UnitKey, SerdeBincode<BlockHash>>,
    /// Current height
    height: DatabaseUnique<UnitKey, SerdeBincode<u32>>,
    pub utxos: DatabaseUnique<OutPointKey, SerdeBincode<Output>>,
    pub stxos: DatabaseUnique<OutPointKey, SerdeBincode<SpentOutput>>,
    /// Pending withdrawal bundle. MUST exist in withdrawal_bundles
    pub pending_withdrawal_bundle: DatabaseUnique<UnitKey, SerdeBincode<M6id>>,
    /// Latest failed (known) withdrawal bundle
    latest_failed_withdrawal_bundle:
        DatabaseUnique<UnitKey, SerdeBincode<RollBack<M6id>>>,
    /// Withdrawal bundles and their status.
    /// Some withdrawal bundles may be unknown.
    /// in which case they are `None`.
    withdrawal_bundles: DatabaseUnique<
        SerdeBincode<M6id>,
        SerdeBincode<(WithdrawalBundleInfo, RollBack<WithdrawalBundleStatus>)>,
    >,
    /// deposit blocks and the height at which they were applied, keyed sequentially
    pub deposit_blocks: DatabaseUnique<
        SerdeBincode<u32>,
        SerdeBincode<(bitcoin::BlockHash, u32)>,
    >,
    /// withdrawal bundle event blocks and the height at which they were applied, keyed sequentially
    pub withdrawal_bundle_event_blocks: DatabaseUnique<
        SerdeBincode<u32>,
        SerdeBincode<(bitcoin::BlockHash, u32)>,
    >,
    /// Coin movements that no block body carries, keyed by the height that
    /// applied them, in the order the node applied them
    two_way_peg_events:
        DatabaseUnique<SerdeBincode<u32>, SerdeBincode<Vec<TwoWayPegEvent>>>,
    pub utreexo_accumulator: DatabaseUnique<UnitKey, SerdeBincode<Accumulator>>,
    _version: DatabaseUnique<UnitKey, SerdeBincode<Version>>,
    /// Timestamp of the mainchain block that the tip names
    mainchain_timestamp: DatabaseUnique<UnitKey, SerdeBincode<u64>>,
    /// Timestamp of the mainchain block that the genesis block names
    genesis_timestamp: DatabaseUnique<UnitKey, SerdeBincode<u64>>,
    reputation: reputation::ReputationDbs,
    decisions: decisions::Dbs,
    markets: MarketsDatabase,
    voting: VotingSystem,
    settlement_undo: DatabaseUnique<
        SerdeBincode<u32>,
        SerdeBincode<undo::SettlementUndoData>,
    >,
    consensus_undo: DatabaseUnique<
        SerdeBincode<u32>,
        SerdeBincode<undo::ConsensusUndoData>,
    >,
    consolidation_undo: DatabaseUnique<
        SerdeBincode<u32>,
        SerdeBincode<undo::ConsolidationUndoData>,
    >,
    minting_undo: DatabaseUnique<SerdeBincode<u32>, SerdeBincode<u32>>,
    reputation_transfer_undo: DatabaseUnique<
        SerdeBincode<u32>,
        SerdeBincode<undo::ReputationTransferUndoData>,
    >,
    skipped_tx_indices_undo:
        DatabaseUnique<SerdeBincode<u32>, SerdeBincode<Vec<u32>>>,
}

impl State {
    pub const NUM_DBS: u32 = 20
        + reputation::ReputationDbs::NUM_DBS
        + decisions::Dbs::NUM_DBS
        + MarketsDatabase::NUM_DBS
        + VotingSystem::NUM_DBS;

    pub fn new<Tls>(
        env: &sneed::Env<Tls>,
        decision_config_testing: Option<u32>,
    ) -> Result<Self, Error> {
        let mut rwtxn = env.write_txn().map_err(EnvError::from)?;
        let tip = DatabaseUnique::create(env, &mut rwtxn, "tip")
            .map_err(EnvError::from)?;
        let height = DatabaseUnique::create(env, &mut rwtxn, "height")
            .map_err(EnvError::from)?;
        let utxos = DatabaseUnique::create(env, &mut rwtxn, "utxos")
            .map_err(EnvError::from)?;
        let stxos = DatabaseUnique::create(env, &mut rwtxn, "stxos")
            .map_err(EnvError::from)?;
        let pending_withdrawal_bundle = DatabaseUnique::create(
            env,
            &mut rwtxn,
            "pending_withdrawal_bundle",
        )
        .map_err(EnvError::from)?;
        let latest_failed_withdrawal_bundle = DatabaseUnique::create(
            env,
            &mut rwtxn,
            "latest_failed_withdrawal_bundle",
        )
        .map_err(EnvError::from)?;
        let withdrawal_bundles =
            DatabaseUnique::create(env, &mut rwtxn, "withdrawal_bundles")
                .map_err(EnvError::from)?;
        let deposit_blocks =
            DatabaseUnique::create(env, &mut rwtxn, "deposit_blocks")
                .map_err(EnvError::from)?;
        let withdrawal_bundle_event_blocks = DatabaseUnique::create(
            env,
            &mut rwtxn,
            "withdrawal_bundle_event_blocks",
        )
        .map_err(EnvError::from)?;
        let two_way_peg_events =
            DatabaseUnique::create(env, &mut rwtxn, "two_way_peg_events")
                .map_err(EnvError::from)?;
        let utreexo_accumulator =
            DatabaseUnique::create(env, &mut rwtxn, "utreexo_accumulator")
                .map_err(EnvError::from)?;
        let version = DatabaseUnique::create(env, &mut rwtxn, "state_version")
            .map_err(EnvError::from)?;
        match version.try_get(&rwtxn, &())? {
            Some(db_version)
                if db_version
                    < Version {
                        major: 0,
                        minor: 20,
                        patch: 0,
                    } =>
            {
                return Err(Error::IncompatibleVersion {
                    version: db_version,
                    db_path: env.path().to_path_buf(),
                });
            }
            Some(_) => (),
            None => version.put(&mut rwtxn, &(), &*VERSION)?,
        };
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
        let settlement_undo =
            DatabaseUnique::create(env, &mut rwtxn, "settlement_undo")?;
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
        let skipped_tx_indices_undo =
            DatabaseUnique::create(env, &mut rwtxn, "skipped_tx_indices_undo")?;
        rwtxn.commit().map_err(RwTxnError::from)?;
        Ok(Self {
            tip,
            height,
            utxos,
            stxos,
            pending_withdrawal_bundle,
            latest_failed_withdrawal_bundle,
            withdrawal_bundles,
            deposit_blocks,
            withdrawal_bundle_event_blocks,
            two_way_peg_events,
            utreexo_accumulator,
            _version: version,
            mainchain_timestamp,
            genesis_timestamp,
            reputation,
            decisions,
            markets,
            voting,
            settlement_undo,
            consensus_undo,
            consolidation_undo,
            minting_undo,
            reputation_transfer_undo,
            skipped_tx_indices_undo,
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

    pub fn get_mempool_shares(
        &self,
        rotxn: &RoTxn,
        market_id: &MarketId,
    ) -> Result<Option<ndarray::Array1<i64>>, Error> {
        self.markets.get_mempool_shares(rotxn, market_id)
    }

    pub fn put_mempool_shares(
        &self,
        rwtxn: &mut RwTxn,
        market_id: &MarketId,
        shares: &ndarray::Array1<i64>,
    ) -> Result<(), Error> {
        self.markets.put_mempool_shares(rwtxn, market_id, shares)
    }

    pub fn clear_mempool_shares(
        &self,
        rwtxn: &mut RwTxn,
        market_id: &MarketId,
    ) -> Result<(), Error> {
        self.markets.clear_mempool_shares(rwtxn, market_id)
    }

    /// Coin movements that the block at this height applied outside its body,
    /// in the order the node applied them
    pub fn get_two_way_peg_events(
        &self,
        rotxn: &RoTxn,
        height: u32,
    ) -> Result<Vec<TwoWayPegEvent>, Error> {
        let events = self
            .two_way_peg_events
            .try_get(rotxn, &height)?
            .unwrap_or_default();
        Ok(events)
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

    pub fn get_stxos_by_addresses(
        &self,
        rotxn: &RoTxn,
        addresses: &HashSet<Address>,
    ) -> Result<HashMap<OutPoint, SpentOutput>, db_error::Iter> {
        let stxos: HashMap<OutPoint, _> = self
            .stxos
            .iter(rotxn)?
            .filter_map(|(key, output)| {
                if addresses.contains(&output.output.address) {
                    Ok(Some((key.into(), output)))
                } else {
                    Ok(None)
                }
            })
            .collect()?;
        Ok(stxos)
    }

    pub fn get_utxos(
        &self,
        rotxn: &RoTxn,
    ) -> Result<HashMap<OutPoint, Output>, db_error::Iter> {
        let utxos: HashMap<OutPoint, Output> = self
            .utxos
            .iter(rotxn)?
            .map(|(key, output)| Ok((key.into(), output)))
            .collect()?;
        Ok(utxos)
    }

    pub fn get_utxos_by_addresses(
        &self,
        rotxn: &RoTxn,
        addresses: &HashSet<Address>,
    ) -> Result<HashMap<OutPoint, Output>, db_error::Iter> {
        let utxos: HashMap<OutPoint, Output> = self
            .utxos
            .iter(rotxn)?
            .filter(|(_, output)| Ok(addresses.contains(&output.address)))
            .map(|(key, output)| Ok((key.into(), output)))
            .collect()?;
        Ok(utxos)
    }

    /// Get the latest failed withdrawal bundle, and the height at which it failed
    pub fn get_latest_failed_withdrawal_bundle(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<(u32, M6id)>, db_error::TryGet> {
        let Some(latest_failed_m6id) =
            self.latest_failed_withdrawal_bundle.try_get(rotxn, &())?
        else {
            return Ok(None);
        };
        let latest_failed_m6id = latest_failed_m6id.latest().value;
        let (_bundle, bundle_status) = self.withdrawal_bundles.try_get(rotxn, &latest_failed_m6id)?
            .unwrap_or_else(||
                panic!("Inconsistent DBs: latest failed m6id {latest_failed_m6id} should exist in withdrawal_bundles")
            );
        let failed_height = bundle_status
            .iter()
            .rev()
            .find_map(|status| match status.value {
                WithdrawalBundleStatus::Failed => Some(status.height),
                WithdrawalBundleStatus::Confirmed
                | WithdrawalBundleStatus::Dropped
                | WithdrawalBundleStatus::Pending
                | WithdrawalBundleStatus::Submitted
                | WithdrawalBundleStatus::SubmittedUnexpected => None,
            })
            .unwrap_or_else(|| {
                panic!("missing failure status for {latest_failed_m6id}")
            });
        Ok(Some((failed_height, latest_failed_m6id)))
    }

    pub fn try_get_withdrawal_bundle(
        &self,
        rotxn: &RoTxn,
        m6id: &M6id,
    ) -> Result<
        Option<(WithdrawalBundleInfo, WithdrawalBundleStatus)>,
        db_error::TryGet,
    > {
        let Some((bundle_info, bundle_status)) =
            self.withdrawal_bundles.try_get(rotxn, m6id)?
        else {
            return Ok(None);
        };
        Ok(Some((bundle_info, bundle_status.latest().value)))
    }

    /// Get the current Utreexo accumulator
    pub fn get_accumulator(&self, rotxn: &RoTxn) -> Result<Accumulator, Error> {
        let accumulator = self
            .utreexo_accumulator
            .try_get(rotxn, &())
            .map_err(DbError::from)?
            .unwrap_or_default();
        Ok(accumulator)
    }

    /// Regenerate utreexo proof for a tx.
    ///
    /// An input that `unconfirmed` answers has no leaf in the accumulator, so
    /// it is not a proof target. The transaction that made it proves it.
    pub fn regenerate_proof(
        &self,
        rotxn: &RoTxn,
        unconfirmed: &HashMap<OutPoint, Output>,
        tx: &mut Transaction,
    ) -> Result<(), Error> {
        let accumulator = self.get_accumulator(rotxn)?;
        let targets: Vec<_> = tx
            .inputs
            .iter()
            .filter(|(outpoint, _)| !unconfirmed.contains_key(outpoint))
            .map(|(_, utxo_hash)| utxo_hash.into())
            .collect();
        tx.proof = accumulator.prove(&targets)?;
        Ok(())
    }

    /// Get a Utreexo proof for the provided utxos
    pub fn get_utreexo_proof<'a, Utxos>(
        &self,
        rotxn: &RoTxn,
        utxos: Utxos,
    ) -> Result<UtreexoProof, Error>
    where
        Utxos: IntoIterator<Item = &'a PointedOutput>,
    {
        let accumulator = self.get_accumulator(rotxn)?;
        let targets: Vec<UtreexoNodeHash> =
            utxos.into_iter().map(UtreexoNodeHash::from).collect();
        let proof = accumulator.prove(&targets)?;
        Ok(proof)
    }

    /// Fill a transaction with the outputs it spends.
    ///
    /// `unconfirmed` holds the outputs of transactions that the chain does not
    /// carry yet: the earlier transactions of a block body, or the ancestors a
    /// mempool holds. Pass an empty map to read the confirmed set alone.
    pub fn fill_transaction(
        &self,
        rotxn: &RoTxn,
        unconfirmed: &HashMap<OutPoint, Output>,
        transaction: &Transaction,
    ) -> Result<FilledTransaction, Error> {
        let mut spent_utxos = Vec::with_capacity(transaction.inputs.len());
        for (outpoint, _) in &transaction.inputs {
            let key = OutPointKey::from(outpoint);
            let utxo = match self.utxos.try_get(rotxn, &key)? {
                Some(utxo) => utxo,
                None => {
                    unconfirmed.get(outpoint).cloned().ok_or(error::NoUtxo {
                        outpoint: *outpoint,
                    })?
                }
            };
            spent_utxos.push(utxo);
        }
        Ok(FilledTransaction {
            spent_utxos,
            transaction: transaction.clone(),
            actor_address: None,
        })
    }

    /// Fill a transaction of the active chain from the STXOs it spent
    pub fn fill_transaction_from_stxos(
        &self,
        rotxn: &RoTxn,
        tx: Transaction,
    ) -> Result<FilledTransaction, Error> {
        let txid = tx.txid();
        let mut spent_utxos = Vec::with_capacity(tx.inputs.len());
        for (vin, (outpoint, _)) in tx.inputs.iter().enumerate().rev() {
            let key = OutPointKey::from(outpoint);
            let stxo =
                self.stxos.try_get(rotxn, &key)?.ok_or(Error::NoStxo {
                    outpoint: *outpoint,
                })?;
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
        unconfirmed: &HashMap<OutPoint, Output>,
        transaction: AuthorizedTransaction,
    ) -> Result<Authorized<FilledTransaction>, Error> {
        let mut filled_tx = self.fill_transaction(
            rotxn,
            unconfirmed,
            &transaction.transaction,
        )?;
        filled_tx.actor_address = transaction
            .actor_proof
            .as_ref()
            .map(|auth| auth.get_address());
        let authorizations = transaction.authorizations;
        Ok(Authorized {
            transaction: filled_tx,
            authorizations,
            actor_proof: transaction.actor_proof,
        })
    }

    /// Get pending withdrawal bundle and block height
    pub fn try_get_pending_withdrawal_bundle(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<(WithdrawalBundle, u32)>, Error> {
        let Some(m6id) = self.pending_withdrawal_bundle.try_get(rotxn, &())?
        else {
            return Ok(None);
        };
        let (bundle_info, bundle_status) =
            self.withdrawal_bundles.get(rotxn, &m6id)?;
        let bundle = match bundle_info {
            WithdrawalBundleInfo::Known(bundle) => bundle,
            WithdrawalBundleInfo::Unknown
            | WithdrawalBundleInfo::UnknownConfirmed { spend_utxos: _ } => {
                return Err(error::PendingWithdrawalBundleUnknown(m6id).into());
            }
        };
        let height = bundle_status.latest().height;
        Ok(Some((bundle, height)))
    }

    fn validate_utxo_hashes(
        transaction: &FilledTransaction,
    ) -> Result<(), Error> {
        for (outpoint, utxo_hash, output) in transaction.inputs() {
            let outpoint = *outpoint;
            let utxo_hash = *utxo_hash;
            let computed_utxo_hash =
                crate::types::hash(&PointedOutputRef { outpoint, output });
            if utxo_hash != computed_utxo_hash {
                return Err(Error::UtxoHashMismatch {
                    computed: computed_utxo_hash,
                    outpoint,
                    input_hash: utxo_hash,
                });
            }
        }
        Ok(())
    }

    pub fn validate_filled_transaction(
        &self,
        rotxn: &RoTxn,
        transaction: &FilledTransaction,
        archive: &crate::archive::Archive,
        override_height: Option<u32>,
    ) -> Result<bitcoin::Amount, Error> {
        let () = Self::validate_utxo_hashes(transaction)?;
        let mut value_in = bitcoin::Amount::ZERO;
        let mut value_out = bitcoin::Amount::ZERO;
        for (outpoint, _, utxo) in transaction.inputs() {
            // a withdrawal output is committed to a bundle and can only be
            // spent by the bundle, never by a transaction
            if utxo.content.is_withdrawal() {
                return Err(Error::SpendWithdrawalOutput {
                    outpoint: *outpoint,
                });
            }
            value_in = value_in
                .checked_add(utxo.get_value())
                .ok_or(AmountOverflowError)?;
        }
        if let Some(fee) = crate::validation::validate_market_transaction(
            self,
            rotxn,
            transaction,
            archive,
            override_height,
        )? {
            return Ok(fee);
        }
        for output in &transaction.transaction.outputs {
            value_out = value_out
                .checked_add(output.get_value())
                .ok_or(AmountOverflowError)?;
        }
        if value_out > value_in {
            return Err(Error::NotEnoughValueIn);
        }
        value_in
            .checked_sub(value_out)
            .ok_or_else(|| AmountUnderflowError.into())
    }

    pub fn validate_transaction(
        &self,
        rotxn: &RoTxn,
        batch_verification_ctxt: &BatchVerificationContext,
        unconfirmed: &HashMap<OutPoint, Output>,
        transaction: &AuthorizedTransaction,
        archive: &crate::archive::Archive,
    ) -> Result<bitcoin::Amount, Error> {
        let mut filled_transaction = self.fill_transaction(
            rotxn,
            unconfirmed,
            &transaction.transaction,
        )?;
        filled_transaction.actor_address = transaction
            .actor_proof
            .as_ref()
            .map(|auth| auth.get_address());
        for (authorization, spent_utxo) in transaction
            .authorizations
            .iter()
            .zip(filled_transaction.spent_utxos.iter())
        {
            if authorization.get_address() != spent_utxo.address {
                return Err(Error::WrongPubKeyForAddress);
            }
        }
        if authorization::verify_transaction(
            batch_verification_ctxt,
            transaction,
        )
        .is_err()
        {
            return Err(Error::Authorization);
        }
        let fee = self.validate_filled_transaction(
            rotxn,
            &filled_transaction,
            archive,
            None,
        )?;
        Ok(fee)
    }

    const LIMIT_GROWTH_EXPONENT: f64 = 1.04;

    pub fn body_sigops_limit(height: u32) -> usize {
        // Starting body size limit is 8MB = 8 * 1024 * 1024 B
        // 2 input 2 output transaction is 392 B
        // 2 * ceil(8 * 1024 * 1024 B / 392 B) = 42800
        const START: usize = 42800;
        let month = height / (6 * 24 * 30);
        if month < 120 {
            (START as f64 * Self::LIMIT_GROWTH_EXPONENT.powi(month as i32))
                .floor() as usize
        } else {
            // 1.04 ** 120 = 110.6625
            // So we are rounding up.
            START * 111
        }
    }

    // in bytes
    pub fn body_size_limit(height: u32) -> usize {
        // 8MB starting body size limit.
        const START: usize = 8 * 1024 * 1024;
        let month = height / (6 * 24 * 30);
        if month < 120 {
            (START as f64 * Self::LIMIT_GROWTH_EXPONENT.powi(month as i32))
                .floor() as usize
        } else {
            // 1.04 ** 120 = 110.6625
            // So we are rounding up.
            START * 111
        }
    }

    pub fn get_last_deposit_block_hash(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<bitcoin::BlockHash>, Error> {
        let block_hash = self
            .deposit_blocks
            .last(rotxn)
            .map_err(DbError::from)?
            .map(|(_, (block_hash, _))| block_hash);
        Ok(block_hash)
    }

    pub fn get_last_withdrawal_bundle_event_block_hash(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<bitcoin::BlockHash>, Error> {
        let block_hash = self
            .withdrawal_bundle_event_blocks
            .last(rotxn)
            .map_err(DbError::from)?
            .map(|(_, (block_hash, _))| block_hash);
        Ok(block_hash)
    }

    /// Get total sidechain wealth in Bitcoin
    pub fn sidechain_wealth(
        &self,
        rotxn: &RoTxn,
    ) -> Result<bitcoin::Amount, Error> {
        let mut total_deposit_utxo_value = bitcoin::Amount::ZERO;
        self.utxos
            .iter(rotxn)
            .map_err(DbError::from)?
            .map_err(|err| DbError::from(err).into())
            .for_each(|(outpoint_key, output)| {
                let outpoint: OutPoint = outpoint_key.into();
                if let OutPoint::Deposit(_) = outpoint {
                    total_deposit_utxo_value = total_deposit_utxo_value
                        .checked_add(output.get_value())
                        .ok_or(AmountOverflowError)?;
                }
                Ok::<_, Error>(())
            })?;
        let mut total_deposit_stxo_value = bitcoin::Amount::ZERO;
        let mut total_withdrawal_stxo_value = bitcoin::Amount::ZERO;
        self.stxos
            .iter(rotxn)
            .map_err(DbError::from)?
            .map_err(|err| DbError::from(err).into())
            .for_each(|(outpoint_key, spent_output)| {
                let outpoint: OutPoint = outpoint_key.into();
                if let OutPoint::Deposit(_) = outpoint {
                    total_deposit_stxo_value = total_deposit_stxo_value
                        .checked_add(spent_output.output.get_value())
                        .ok_or(AmountOverflowError)?;
                }
                if let InPoint::Withdrawal { .. } = spent_output.inpoint {
                    total_withdrawal_stxo_value = total_withdrawal_stxo_value
                        .checked_add(spent_output.output.get_value())
                        .ok_or(AmountOverflowError)?;
                }
                Ok::<_, Error>(())
            })?;

        let total_wealth: bitcoin::Amount = total_deposit_utxo_value
            .checked_add(total_deposit_stxo_value)
            .ok_or(AmountOverflowError)?
            .checked_sub(total_withdrawal_stxo_value)
            .ok_or(AmountOverflowError)?;
        Ok(total_wealth)
    }

    pub fn validate_block(
        &self,
        rotxn: &RoTxn,
        batch_verification_ctxt: &BatchVerificationContext,
        header: &Header,
        body: &Body,
        archive: &crate::archive::Archive,
    ) -> Result<(bitcoin::Amount, MerkleRoot), Error> {
        block::validate(
            batch_verification_ctxt,
            self,
            rotxn,
            header,
            body,
            archive,
        )
    }

    pub fn connect_block(
        &self,
        rwtxn: &mut RwTxn,
        header: &Header,
        body: &Body,
        mainchain_timestamp: u64,
    ) -> Result<MerkleRoot, Error> {
        block::connect(self, rwtxn, header, body, mainchain_timestamp)
    }

    /// Prevalidate a block under a read transaction, computing values reused on connect.
    pub fn prevalidate_block(
        &self,
        rotxn: &RoTxn,
        batch_verification_ctxt: &BatchVerificationContext,
        header: &Header,
        body: &Body,
        archive: &crate::archive::Archive,
    ) -> Result<PrevalidatedBlock, Error> {
        block::prevalidate(
            batch_verification_ctxt,
            self,
            rotxn,
            header,
            body,
            archive,
        )
    }

    /// Connect a block using prevalidated data to avoid recomputation.
    pub fn connect_prevalidated_block(
        &self,
        rwtxn: &mut RwTxn,
        header: &Header,
        body: &Body,
        prevalidated: PrevalidatedBlock,
        mainchain_timestamp: u64,
    ) -> Result<MerkleRoot, Error> {
        block::connect_prevalidated(
            self,
            rwtxn,
            header,
            body,
            prevalidated,
            mainchain_timestamp,
        )
    }

    /// Convenience: prevalidate then connect using the same write transaction.
    pub fn apply_block(
        &self,
        rwtxn: &mut RwTxn,
        batch_verification_ctxt: &BatchVerificationContext,
        header: &Header,
        body: &Body,
        archive: &crate::archive::Archive,
        mainchain_timestamp: u64,
    ) -> Result<(), Error> {
        let pre = self.prevalidate_block(
            rwtxn,
            batch_verification_ctxt,
            header,
            body,
            archive,
        )?;
        let _: MerkleRoot = self.connect_prevalidated_block(
            rwtxn,
            header,
            body,
            pre,
            mainchain_timestamp,
        )?;
        Ok(())
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

    pub fn get_settled_decisions(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Vec<crate::state::decisions::DecisionEntry>, Error> {
        let (current_ts, current_height, genesis_ts) =
            self.period_context(rotxn)?;
        self.decisions.get_settled_decisions(
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

    fn fee_for_decision_id(
        &self,
        rotxn: &RoTxn,
        decision_id: DecisionId,
    ) -> Result<u64, Error> {
        DecisionValidationInterface::fee_for_decision_id(
            self.decisions(),
            rotxn,
            decision_id,
        )
    }
}

impl Watchable<()> for State {
    type WatchStream = tokio_stream::wrappers::WatchStream<()>;

    /// Get a signal that notifies whenever the tip changes
    fn watch(&self) -> Self::WatchStream {
        tokio_stream::wrappers::WatchStream::new(self.tip.watch().clone())
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use bitcoin::hashes::Hash as _;

    use crate::{
        state::State,
        types::{
            Address, FilledTransaction, InPoint, M6id, OutPoint, OutPointKey,
            Output, OutputContent, PointedOutputRef, SpentOutput, Transaction,
            hash, state::TwoWayPegEvent,
        },
    };

    fn temp_dir(test_name: &str) -> anyhow::Result<temp_dir::TempDir> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let res = temp_dir::TempDir::with_prefix(format!(
            "truthcoin-{test_name}-{}-{nanos}",
            std::process::id()
        ))?;
        Ok(res)
    }

    // open a fresh state-backed env in a unique temp dir
    pub fn temp_env(
        test_name: &str,
    ) -> anyhow::Result<(temp_dir::TempDir, sneed::Env)> {
        let temp_dir = temp_dir(test_name)?;
        let mut opts = heed::EnvOpenOptions::new();
        opts.map_size(64 * 1024 * 1024)
            .max_dbs(State::NUM_DBS + crate::archive::Archive::NUM_DBS);
        let env = unsafe { sneed::Env::open(&opts, temp_dir.path()) }?;
        Ok((temp_dir, env))
    }

    pub fn fresh_state(
        test_name: &str,
    ) -> anyhow::Result<(temp_dir::TempDir, sneed::Env, State)> {
        let (temp_dir, env) = temp_env(test_name)?;
        let state = State::new(&env, None)?;
        Ok((temp_dir, env, state))
    }

    /// Create a value output
    pub fn value_output(addr: Address, sats: u64) -> Output {
        Output {
            address: addr,
            content: OutputContent::Value(bitcoin::Amount::from_sat(sats)),
        }
    }

    #[test]
    fn cannot_spend_withdrawal_output() -> anyhow::Result<()> {
        let (_temp_dir, env, state) =
            fresh_state("cannot-spend-withdrawal-output")?;
        let archive = crate::archive::Archive::new(&env)?;
        let main_address = {
            let pkh = bitcoin::PubkeyHash::hash(b"test pubkey");
            bitcoin::Address::p2pkh(pkh, bitcoin::NetworkKind::Test)
                .into_unchecked()
        };
        let withdrawal = Output {
            address: Address::ALL_ZEROS,
            content: OutputContent::Withdrawal {
                value: bitcoin::Amount::from_sat(1000),
                main_fee: bitcoin::Amount::from_sat(300),
                main_address,
            },
        };
        let outpoint = OutPoint::Regular {
            txid: [1; 32].into(),
            vout: 0,
        };
        let utxo_hash = hash(&PointedOutputRef {
            outpoint,
            output: &withdrawal,
        });
        let tx = FilledTransaction {
            transaction: Transaction {
                inputs: vec![(outpoint, utxo_hash)].into(),
                outputs: vec![value_output(Address::ALL_ZEROS, 1300)].into(),
                ..Default::default()
            },
            spent_utxos: vec![withdrawal],
            actor_address: None,
        };
        let rotxn = env.read_txn()?;
        assert!(matches!(
            state.validate_filled_transaction(&rotxn, &tx, &archive, None),
            Err(crate::state::Error::SpendWithdrawalOutput { .. })
        ));
        Ok(())
    }

    #[test]
    fn fill_authorized_transaction_sets_actor_address() -> anyhow::Result<()> {
        use crate::types::authorization::{SigningKey, authorize, get_address};

        let (_temp_dir, env, state) =
            fresh_state("fill-authorized-transaction-sets-actor-address")?;
        let key = SigningKey::from_scalar(
            curve25519_dalek::Scalar::from_bytes_mod_order([7; 32]),
        )?;
        let actor = get_address(&(&key).into());
        let mut tx =
            authorize(rand::rng(), &[(actor, &key)], Transaction::default())?;
        tx.actor_proof = tx.authorizations.pop().map(Box::new);
        let rotxn = env.read_txn()?;
        let filled =
            state.fill_authorized_transaction(&rotxn, &HashMap::new(), tx)?;
        anyhow::ensure!(filled.transaction.actor_address == Some(actor));
        Ok(())
    }

    #[test]
    fn sidechain_wealth() -> anyhow::Result<()> {
        use std::str::FromStr;

        use bitcoin::hashes::Hash as _;

        let (_temp_dir, env, state) = fresh_state("sidechain-wealth")?;
        {
            let mut rwtxn = env.write_txn()?;

            // One unspent DEPOSIT UTXO: 50 sats.
            let deposit_utxo_op = OutPoint::Deposit(bitcoin::OutPoint {
                txid: bitcoin::Txid::from_str(
                    "0000000000000000000000000000000000000000000000000000000000000001",
                )?,
                vout: 0,
            });
            state.utxos.put(
                &mut rwtxn,
                &OutPointKey::from(&deposit_utxo_op),
                &value_output(Address::ALL_ZEROS, 50),
            )?;

            // Two spent DEPOSIT STXOs: 100 + 100 sats.
            for (i, sats) in [(2u8, 100u64), (3u8, 100u64)] {
                let op = OutPoint::Deposit(bitcoin::OutPoint {
                    txid: bitcoin::Txid::from_byte_array([i; 32]),
                    vout: 0,
                });
                let stxo = SpentOutput {
                    output: value_output(Address::ALL_ZEROS, sats),
                    inpoint: InPoint::Regular {
                        txid: [i; 32].into(),
                        vin: 0,
                    },
                };
                state
                    .stxos
                    .put(&mut rwtxn, &OutPointKey::from(&op), &stxo)?;
            }

            // Two WITHDRAWAL STXOs: 10 + 10 sats
            for (i, sats) in [(4u8, 10u64), (5u8, 10u64)] {
                let op = OutPoint::Regular {
                    txid: [i; 32].into(),
                    vout: 0,
                };
                let stxo = SpentOutput {
                    output: value_output(Address::ALL_ZEROS, sats),
                    inpoint: InPoint::Withdrawal {
                        m6id: crate::types::M6id(
                            bitcoin::Txid::from_byte_array([i; 32]),
                        ),
                    },
                };
                state
                    .stxos
                    .put(&mut rwtxn, &OutPointKey::from(&op), &stxo)?;
            }

            rwtxn.commit()?;
        }

        let rotxn = env.read_txn()?;
        let sidechain_wealth = state.sidechain_wealth(&rotxn)?;

        // Correct value: deposit UTXO 50 + deposit STXOs 200 - withdrawal
        // STXOs 20 = 230 sats.
        let expected_sidechain_wealth = bitcoin::Amount::from_sat(230);
        anyhow::ensure!(
            sidechain_wealth == expected_sidechain_wealth,
            "Expected sidechain wealth ({}), but computed ({})",
            expected_sidechain_wealth,
            sidechain_wealth,
        );
        Ok(())
    }

    #[test]
    fn state_opens_a_database_it_created() -> anyhow::Result<()> {
        let (_temp_dir, env, state) = fresh_state("state-reopen")?;
        drop(state);
        State::new(&env, None)?;
        Ok(())
    }

    #[test]
    fn two_way_peg_events_round_trip() -> anyhow::Result<()> {
        let (_temp_dir, env, state) = fresh_state("two-way-peg-events")?;
        let deposit_outpoint = |byte: u8| {
            OutPoint::Deposit(bitcoin::OutPoint {
                txid: bitcoin::Txid::from_byte_array([byte; 32]),
                vout: 0,
            })
        };
        let m6id = M6id(bitcoin::Txid::from_byte_array([3; 32]));
        let events = vec![
            TwoWayPegEvent::Deposit {
                outpoint: deposit_outpoint(1),
                output: value_output(Address::ALL_ZEROS, 5000),
            },
            TwoWayPegEvent::BundleSpend {
                outpoint: deposit_outpoint(2),
                m6id,
            },
            TwoWayPegEvent::BundleReturn {
                outpoint: deposit_outpoint(2),
                output: value_output(Address::ALL_ZEROS, 7000),
                m6id,
            },
        ];
        {
            let mut rwtxn = env.write_txn()?;
            state.two_way_peg_events.put(&mut rwtxn, &7, &events)?;
            rwtxn.commit()?;
        }
        {
            let rotxn = env.read_txn()?;
            anyhow::ensure!(state.get_two_way_peg_events(&rotxn, 7)? == events);
            // A height that moved nothing outside its body reads as empty.
            anyhow::ensure!(
                state.get_two_way_peg_events(&rotxn, 8)?.is_empty()
            );
        }

        // A disconnect drops the events, so a reorg leaves nothing behind for
        // the block that takes the height.
        {
            let mut rwtxn = env.write_txn()?;
            state.two_way_peg_events.delete(&mut rwtxn, &7)?;
            rwtxn.commit()?;
        }
        let rotxn = env.read_txn()?;
        anyhow::ensure!(state.get_two_way_peg_events(&rotxn, 7)?.is_empty());

        // A height that moved nothing writes no row, so deleting it again is
        // still safe.
        {
            let mut rwtxn = env.write_txn()?;
            state.two_way_peg_events.delete(&mut rwtxn, &8)?;
            rwtxn.commit()?;
        }
        Ok(())
    }
}
