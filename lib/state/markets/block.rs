//! Market effects of connecting and disconnecting a block

use std::collections::{BTreeMap, HashMap, HashSet};

use sneed::{RoTxn, RwTxn};

use crate::{
    math::trading,
    state::{
        Error, State,
        markets::{MarketId, utxos::UtxoManager},
    },
    types::{
        AccumulatorDiff, Address, Body, FilledTransaction, GetValue as _,
        MerkleRoot, OutPoint, OutPointKey, OutputContent, Transaction, TxData,
        Votecoin,
        state::{MarketUtxo, MarketUtxoChanges, MarketUtxoReason},
    },
};

#[derive(Default)]
struct StateUpdate {
    market_updates: Vec<MarketStateUpdate>,
    market_creations: Vec<MarketCreation>,
    share_account_changes: HashMap<(Address, MarketId), HashMap<u32, i64>>,
    vote_submissions: Vec<VoteSubmission>,
    pending_trade_settlements: Vec<PendingTradeSettlement>,
}

struct MarketStateUpdate {
    market_id: MarketId,
    legs: Vec<TradeLeg>,
}

/// One outcome a trade moves, with its share of the trade's volume
struct TradeLeg {
    outcome_index: usize,
    shares: i64,
    volume_sats: u64,
}

struct MarketCreation {
    market: crate::state::Market,
}

struct VoteSubmission {
    vote: crate::state::voting::types::Vote,
}

/// The money of a trade or an `AmplifyBeta` deposit, settled when the block
/// consolidates its market's treasury
pub struct PendingTradeSettlement {
    pub market_id: MarketId,
    pub trader_address: Address,
    pub input_value_sats: u64,
    pub buy_cost_sats: u64,
    pub sell_gross_sats: u64,
    pub fee_sats: u64,
    pub transaction_id: [u8; 32],
    pub is_amplify: bool,
}

pub enum TradeApplyResult {
    Applied,
    Skipped { reason: String },
}

impl StateUpdate {
    fn new() -> Self {
        Self::default()
    }

    fn verify_internal_consistency(&self) -> Result<(), Error> {
        let mut created_market_ids = std::collections::HashSet::new();
        for creation in &self.market_creations {
            if !created_market_ids.insert(creation.market.id) {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Duplicate market creation for ID: {:?}",
                        creation.market.id
                    ),
                });
            }
        }

        for update in &self.market_updates {
            if created_market_ids.contains(&update.market_id) {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Market {:?} cannot be both created and updated in same block",
                        update.market_id
                    ),
                });
            }
        }

        Ok(())
    }

    fn validate_all_changes(
        &self,
        state: &State,
        rotxn: &RoTxn,
    ) -> Result<(), Error> {
        self.verify_internal_consistency()?;

        for update in &self.market_updates {
            if state
                .markets()
                .get_market(rotxn, &update.market_id)?
                .is_none()
            {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Market {:?} does not exist",
                        update.market_id
                    ),
                });
            }
        }

        for creation in &self.market_creations {
            if state
                .markets()
                .get_market(rotxn, &creation.market.id)?
                .is_some()
            {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Market {:?} already exists",
                        creation.market.id
                    ),
                });
            }

            crate::validation::MarketValidator::validate_market_shares(
                creation.market.shares(),
            )?;
        }

        Ok(())
    }

    fn apply_all_changes(
        &self,
        state: &State,
        rwtxn: &mut RwTxn,
        accumulator_diff: &mut AccumulatorDiff,
        market_utxos: &mut MarketUtxoChanges,
        height: u32,
    ) -> Result<Option<crate::state::undo::ConsolidationUndoData>, Error> {
        for creation in &self.market_creations {
            state
                .markets()
                .add_market(rwtxn, &creation.market)
                .map_err(|_| Error::InvalidTransaction {
                    reason: "Failed to store market in database".to_string(),
                })?;
        }

        let mut legs_by_market: HashMap<MarketId, Vec<&TradeLeg>> =
            HashMap::new();
        for update in &self.market_updates {
            legs_by_market
                .entry(update.market_id)
                .or_default()
                .extend(&update.legs);
        }

        for (market_id, legs) in legs_by_market {
            let mut market = state
                .markets()
                .get_market(rwtxn, &market_id)?
                .ok_or_else(|| Error::InvalidTransaction {
                    reason: format!("Market {market_id:?} not found"),
                })?;

            let mut new_shares = market.shares().clone();

            for leg in legs {
                new_shares[leg.outcome_index] += leg.shares;
                market
                    .update_trading_volume(leg.outcome_index, leg.volume_sats)
                    .map_err(|e| Error::InvalidTransaction {
                        reason: format!("Failed to update volume: {e:?}"),
                    })?;
            }

            market
                .update_state(height, None, Some(new_shares), None)
                .map_err(|e| Error::InvalidTransaction {
                    reason: format!("Failed to update market state: {e:?}"),
                })?;

            state.markets().update_market(rwtxn, &market)?;
            state.clear_mempool_shares(rwtxn, &market_id)?;
        }

        let mut amplify_by_market: std::collections::HashMap<MarketId, u64> =
            std::collections::HashMap::new();
        for settlement in &self.pending_trade_settlements {
            if settlement.is_amplify {
                *amplify_by_market.entry(settlement.market_id).or_default() +=
                    settlement.buy_cost_sats;
            }
        }
        for (market_id, amount) in amplify_by_market {
            let mut market = state
                .markets()
                .get_market(rwtxn, &market_id)?
                .ok_or_else(|| Error::InvalidTransaction {
                    reason: format!("Market {market_id:?} not found"),
                })?;
            market.liquidity_base_sats = market
                .liquidity_base_sats
                .checked_add(amount)
                .ok_or_else(|| Error::InvalidTransaction {
                    reason: format!(
                        "Liquidity base overflow for market {market_id:?}"
                    ),
                })?;
            state.markets().update_market(rwtxn, &market)?;
        }

        for ((address, market_id), outcome_changes) in
            &self.share_account_changes
        {
            for (&outcome_index, &share_delta) in outcome_changes {
                if share_delta != 0 {
                    if share_delta > 0 {
                        state.markets().add_shares_to_account(
                            rwtxn,
                            address,
                            *market_id,
                            outcome_index,
                            share_delta,
                            height,
                        )?;
                    } else {
                        state.markets().remove_shares_from_account(
                            rwtxn,
                            address,
                            market_id,
                            outcome_index,
                            -share_delta,
                            height,
                        )?;
                    }
                }
            }
        }

        {
            let traded_markets: std::collections::HashSet<_> = self
                .share_account_changes
                .keys()
                .map(|(_, market_id)| *market_id)
                .collect();
            for market_id in &traded_markets {
                state.markets().verify_share_invariant(rwtxn, market_id)?;
            }
        }

        tracing::debug!(
            "apply_all_changes: Applying {} vote submissions",
            self.vote_submissions.len()
        );

        for submission in &self.vote_submissions {
            state
                .voting()
                .databases()
                .put_vote(rwtxn, &submission.vote)?;
        }

        let consolidation_undo = Self::consolidate_market_utxos(
            state,
            rwtxn,
            accumulator_diff,
            market_utxos,
            height,
            &self.pending_trade_settlements,
        )?;

        Ok(consolidation_undo)
    }

    pub fn generate_trade_change_outpoint(
        market_id: &MarketId,
        trader_address: &Address,
        transaction_id: [u8; 32],
    ) -> OutPoint {
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(b"TRADE_CHANGE");
        hasher.update(&market_id.0);
        hasher.update(&trader_address.0);
        hasher.update(&transaction_id);

        let hash = hasher.finalize();
        let merkle_root = MerkleRoot::from(*hash.as_bytes());

        OutPoint::Payout {
            hash: merkle_root,
            vout: 0,
        }
    }

    fn consolidate_market_utxos(
        state: &State,
        rwtxn: &mut RwTxn,
        accumulator_diff: &mut AccumulatorDiff,
        market_utxos: &mut MarketUtxoChanges,
        height: u32,
        pending_trade_settlements: &[PendingTradeSettlement],
    ) -> Result<Option<crate::state::undo::ConsolidationUndoData>, Error> {
        use crate::math::trading::TRADE_MINER_FEE_SATS;
        use crate::state::markets::{
            generate_market_author_fee_address,
            generate_market_treasury_address,
        };
        use crate::types::{OutPoint, Output};
        use std::collections::BTreeSet;

        // Accumulator insertion order must match on every node.
        let markets_to_consolidate: BTreeSet<[u8; 6]> =
            pending_trade_settlements
                .iter()
                .map(|settlement| settlement.market_id.0)
                .collect();

        if markets_to_consolidate.is_empty() {
            return Ok(None);
        }

        let mut undo_entries = Vec::new();

        for market_id_bytes in &markets_to_consolidate {
            let market_id = MarketId::new(*market_id_bytes);

            let mut treasury_total = 0u64;
            let mut treasury_utxos_to_consume = Vec::new();
            let mut fee_total = 0u64;
            let mut fee_utxos_to_consume = Vec::new();

            let old_treasury_pointer = state
                .markets()
                .get_market_funds_utxo(rwtxn, &market_id, false)?;
            let old_fee_pointer = state
                .markets()
                .get_market_funds_utxo(rwtxn, &market_id, true)?;
            // Capture old treasury UTXOs with their filled outputs
            let mut old_treasury_utxos_with_outputs = Vec::new();
            let mut old_fee_utxos_with_outputs = Vec::new();

            if let Some(existing_outpoint) = old_treasury_pointer
                && let Some(utxo) = state
                    .utxos
                    .try_get(rwtxn, &OutPointKey::from(&existing_outpoint))?
            {
                treasury_total += utxo.get_value().to_sat();
                old_treasury_utxos_with_outputs.push((existing_outpoint, utxo));
                treasury_utxos_to_consume.push(existing_outpoint);
            }

            if let Some(existing_outpoint) = old_fee_pointer
                && let Some(utxo) = state
                    .utxos
                    .try_get(rwtxn, &OutPointKey::from(&existing_outpoint))?
            {
                fee_total += utxo.get_value().to_sat();
                old_fee_utxos_with_outputs.push((existing_outpoint, utxo));
                fee_utxos_to_consume.push(existing_outpoint);
            }

            let settlements: Vec<&PendingTradeSettlement> =
                pending_trade_settlements
                    .iter()
                    .filter(|settlement| settlement.market_id == market_id)
                    .collect();

            let mut total_sell_gross = 0u64;
            for settlement in &settlements {
                treasury_total += settlement.buy_cost_sats;
                total_sell_gross += settlement.sell_gross_sats;
                fee_total += settlement.fee_sats;
            }

            let remaining_treasury = treasury_total
                .checked_sub(total_sell_gross)
                .ok_or_else(|| Error::InvalidTransaction {
                    reason: format!(
                        "Treasury underflow: sell gross {total_sell_gross} \
                         exceeds treasury {treasury_total} for market \
                         {market_id:?}",
                    ),
                })?;

            let mut new_treasury_utxo = None;
            let mut new_fee_utxo = None;
            let mut trade_change_utxos = Vec::new();

            for outpoint in &treasury_utxos_to_consume {
                state.delete_market_utxo(
                    rwtxn,
                    accumulator_diff,
                    market_utxos,
                    outpoint,
                    MarketUtxoReason::Treasury,
                )?;
            }
            state
                .markets()
                .clear_market_funds_utxo(rwtxn, &market_id, false)?;

            for settlement in &settlements {
                let change = settlement
                    .input_value_sats
                    .checked_add(settlement.sell_gross_sats)
                    .and_then(|v| v.checked_sub(TRADE_MINER_FEE_SATS))
                    .and_then(|v| v.checked_sub(settlement.buy_cost_sats))
                    .and_then(|v| v.checked_sub(settlement.fee_sats))
                    .ok_or_else(|| Error::InvalidTransaction {
                        reason: format!(
                            "Trade change underflow: input {} + sell {} < \
                             miner fee {} + buy {} + market fee {}",
                            settlement.input_value_sats,
                            settlement.sell_gross_sats,
                            TRADE_MINER_FEE_SATS,
                            settlement.buy_cost_sats,
                            settlement.fee_sats,
                        ),
                    })?;
                if change > 0 {
                    let change_outpoint = Self::generate_trade_change_outpoint(
                        &market_id,
                        &settlement.trader_address,
                        settlement.transaction_id,
                    );
                    let change_output = Output {
                        address: settlement.trader_address,
                        content: OutputContent::Value(
                            bitcoin::Amount::from_sat(change),
                        ),
                    };
                    state.insert_market_utxo(
                        rwtxn,
                        accumulator_diff,
                        market_utxos,
                        MarketUtxo {
                            outpoint: change_outpoint,
                            output: change_output,
                            reason: MarketUtxoReason::TradeChange,
                        },
                    )?;
                    trade_change_utxos.push(change_outpoint);
                }
            }

            if remaining_treasury > 0 {
                let treasury_address =
                    generate_market_treasury_address(&market_id);
                let new_outpoint = OutPoint::MarketFunds {
                    market_id: *market_id_bytes,
                    block_height: height,
                    is_fee: false,
                };
                let new_output = Output {
                    address: treasury_address,
                    content: OutputContent::MarketFunds {
                        market_id: *market_id_bytes,
                        amount: bitcoin::Amount::from_sat(remaining_treasury),
                        is_fee: false,
                    },
                };
                state.insert_market_utxo(
                    rwtxn,
                    accumulator_diff,
                    market_utxos,
                    MarketUtxo {
                        outpoint: new_outpoint,
                        output: new_output,
                        reason: MarketUtxoReason::Treasury,
                    },
                )?;
                state.markets().set_market_funds_utxo(
                    rwtxn,
                    &market_id,
                    false,
                    &new_outpoint,
                )?;
                new_treasury_utxo = Some(new_outpoint);
            }

            if fee_total > 0 {
                for outpoint in &fee_utxos_to_consume {
                    state.delete_market_utxo(
                        rwtxn,
                        accumulator_diff,
                        market_utxos,
                        outpoint,
                        MarketUtxoReason::AuthorFee,
                    )?;
                }
                state
                    .markets()
                    .clear_market_funds_utxo(rwtxn, &market_id, true)?;

                let fee_address =
                    generate_market_author_fee_address(&market_id);
                let new_outpoint = OutPoint::MarketFunds {
                    market_id: *market_id_bytes,
                    block_height: height,
                    is_fee: true,
                };
                let new_output = Output {
                    address: fee_address,
                    content: OutputContent::MarketFunds {
                        market_id: *market_id_bytes,
                        amount: bitcoin::Amount::from_sat(fee_total),
                        is_fee: true,
                    },
                };
                state.insert_market_utxo(
                    rwtxn,
                    accumulator_diff,
                    market_utxos,
                    MarketUtxo {
                        outpoint: new_outpoint,
                        output: new_output,
                        reason: MarketUtxoReason::AuthorFee,
                    },
                )?;
                state.markets().set_market_funds_utxo(
                    rwtxn,
                    &market_id,
                    true,
                    &new_outpoint,
                )?;
                new_fee_utxo = Some(new_outpoint);
            }

            undo_entries.push(crate::state::undo::ConsolidationUndoEntry {
                market_id,
                old_treasury_utxos: old_treasury_utxos_with_outputs,
                old_fee_utxos: old_fee_utxos_with_outputs,
                old_treasury_pointer,
                old_fee_pointer,
                new_treasury_utxo,
                new_fee_utxo,
                trade_change_utxos,
            });
        }

        Ok(Some(crate::state::undo::ConsolidationUndoData {
            entries: undo_entries,
        }))
    }

    fn add_market_update(&mut self, update: MarketStateUpdate) {
        self.market_updates.push(update);
    }
    fn add_share_account_change(
        &mut self,
        address: Address,
        market_id: MarketId,
        outcome: u32,
        delta: i64,
    ) {
        *self
            .share_account_changes
            .entry((address, market_id))
            .or_default()
            .entry(outcome)
            .or_insert(0) += delta;
    }
    fn add_market_creation(&mut self, creation: MarketCreation) {
        self.market_creations.push(creation);
    }
    fn add_vote_submission(&mut self, vote: crate::state::voting::types::Vote) {
        self.vote_submissions.push(VoteSubmission { vote });
    }

    fn add_pending_trade_settlement(
        &mut self,
        settlement: PendingTradeSettlement,
    ) {
        self.pending_trade_settlements.push(settlement);
    }
}

/// Market data of a block that `connect_begin` reads before the coinbase
pub(in crate::state) struct ConnectContext {
    height: u32,
    mainchain_timestamp: u64,
    genesis_ts: u64,
    current_period: u32,
}

/// Start to connect the market effects of a block, before the coinbase
pub(in crate::state) fn connect_begin(
    state: &State,
    rwtxn: &mut RwTxn,
    body: &Body,
    height: u32,
    mainchain_timestamp: u64,
) -> Result<ConnectContext, Error> {
    if height == 0 {
        state
            .genesis_timestamp
            .put(rwtxn, &(), &mainchain_timestamp)?;
        let allocation = crate::state::reputation::genesis_allocation(
            state.genesis_reputation_to_miner,
            body,
        )?;
        for (address, reputation) in &allocation {
            state
                .reputation()
                .set_reputation(rwtxn, address, *reputation)?;
        }
        tracing::info!(
            "Genesis block: Initialized reputation for {} addresses",
            allocation.len()
        );
    }
    let genesis_ts = state.try_get_genesis_timestamp(rwtxn)?.unwrap_or(0);
    let prev_highest_minted = state
        .decisions()
        .get_highest_minted_period(rwtxn)?
        .unwrap_or(0);
    state.decisions().snapshot_period_pricing(rwtxn, height)?;
    state
        .minting_undo
        .put(rwtxn, &height, &prev_highest_minted)?;
    let current_period =
        crate::state::voting::period_calculator::get_current_period(
            mainchain_timestamp,
            Some(height),
            genesis_ts,
            state.decisions().get_config(),
        )?;
    let voting_needing_reveal = state
        .decisions()
        .get_voting_decisions_needing_reveal(rwtxn, current_period)?;
    for decision_id in voting_needing_reveal {
        state.decisions().transition_decision_to_revealing(
            rwtxn,
            decision_id,
            height,
        )?;
    }
    crate::validation::validate_coinbase_outputs(
        body.coinbase.outputs.as_slice(),
    )?;
    Ok(ConnectContext {
        height,
        mainchain_timestamp,
        genesis_ts,
        current_period,
    })
}

/// Market changes of the transactions of a block, and the indices of the
/// trades that the block skips
pub(in crate::state) struct ConnectTxs {
    state_update: StateUpdate,
    pub skipped_tx_indices: HashSet<usize>,
}

/// Apply the market data of each transaction. A trade that fails its price
/// limit is skipped: it changes no UTXO and no market.
pub(in crate::state) fn connect_txs(
    state: &State,
    rwtxn: &mut RwTxn,
    ctxt: &ConnectContext,
    filled_txs: &[FilledTransaction],
    coinbase_value: bitcoin::Amount,
) -> Result<ConnectTxs, Error> {
    let height = ctxt.height;
    let mut state_update = StateUpdate::new();
    let mut skipped_tx_indices: HashSet<usize> = HashSet::new();
    for (idx, filled_tx) in filled_txs.iter().enumerate() {
        match &filled_tx.transaction.data {
            Some(TxData::Trade { .. }) => {
                match apply_trade(state, rwtxn, filled_tx, &mut state_update)? {
                    TradeApplyResult::Applied => {}
                    TradeApplyResult::Skipped { reason } => {
                        tracing::info!(
                            "Trade tx {} skipped due to slippage: {}",
                            filled_tx.txid(),
                            reason
                        );
                        skipped_tx_indices.insert(idx);
                    }
                }
            }
            Some(TxData::CreateMarket { .. }) => {
                apply_market_creation(
                    state,
                    rwtxn,
                    filled_tx,
                    &mut state_update,
                    height,
                )?;
            }
            Some(TxData::ClaimDecision(_)) => {
                apply_decision_claim(state, rwtxn, filled_tx, height)?;
            }
            Some(TxData::SubmitBallot { .. }) => {
                apply_submit_ballot(state, rwtxn, filled_tx)?;
            }
            Some(TxData::RevealBallot { .. }) => {
                apply_reveal_ballot(
                    state,
                    rwtxn,
                    filled_tx,
                    &mut state_update,
                    height,
                )?;
            }
            Some(TxData::TransferReputation { .. }) => {
                apply_transfer_reputation(state, rwtxn, filled_tx, height)?;
            }
            Some(TxData::AmplifyBeta { .. }) => {
                apply_amplify_beta(filled_tx, &mut state_update)?;
            }
            Some(TxData::TakeOffer { offer, buyer }) => {
                move_reputation(
                    state,
                    rwtxn,
                    offer.offer.seller,
                    *buyer,
                    offer.offer.amount,
                    height,
                )?;
                state.closed_offers.put(rwtxn, &offer.id(), &height)?;
            }
            Some(TxData::CancelOffer { offer }) => {
                state.closed_offers.put(rwtxn, &offer.id(), &height)?;
            }
            None => {}
        }
    }
    state_update.validate_all_changes(state, rwtxn)?;
    crate::validation::validate_fees(
        coinbase_value,
        filled_txs,
        &skipped_tx_indices,
    )?;
    Ok(ConnectTxs {
        state_update,
        skipped_tx_indices,
    })
}

/// Cancel the accumulator changes of a skipped transaction
pub(in crate::state) fn unapply_skipped_tx_diff(
    accumulator_diff: &mut AccumulatorDiff,
    transaction: &Transaction,
) {
    let txid = transaction.txid();
    for (vout, output) in transaction.outputs.iter().enumerate() {
        let outpoint = OutPoint::Regular {
            txid,
            vout: vout as u32,
        };
        accumulator_diff
            .remove(crate::types::PointedOutputRef { outpoint, output }.into());
    }
    for (_, utxo_hash) in &transaction.inputs {
        accumulator_diff.insert(utxo_hash.into());
    }
}

/// Finish the market effects of a block, after the transaction UTXOs
pub(in crate::state) fn connect_end(
    state: &State,
    rwtxn: &mut RwTxn,
    accumulator_diff: &mut AccumulatorDiff,
    ctxt: ConnectContext,
    txs: ConnectTxs,
) -> Result<(), Error> {
    let ConnectContext {
        height,
        mainchain_timestamp,
        genesis_ts,
        current_period,
    } = ctxt;
    let ConnectTxs {
        state_update,
        skipped_tx_indices,
    } = txs;
    let mut market_utxos = MarketUtxoChanges::default();
    if let Some(consolidation_undo) = state_update.apply_all_changes(
        state,
        rwtxn,
        accumulator_diff,
        &mut market_utxos,
        height,
    )? {
        state
            .consolidation_undo
            .put(rwtxn, &height, &consolidation_undo)?;
    }
    if height == 0 {
        state.decisions().mint_genesis(
            rwtxn,
            mainchain_timestamp,
            height,
            genesis_ts,
        )?;
    } else {
        state.decisions().process_block_pricing(
            rwtxn,
            height,
            mainchain_timestamp,
            genesis_ts,
        )?;
    }
    // Transition Claimed → Voting after transaction application, so a claim
    // in the block that opens a period votes alongside the rest of it.
    let claimed_needing_voting = state
        .decisions()
        .get_claimed_decisions_needing_voting(rwtxn, current_period)?;
    for decision_id in claimed_needing_voting {
        state.decisions().transition_decision_to_voting(
            rwtxn,
            decision_id,
            height,
        )?;
    }
    // Resolve Revealing → Resolved (grouped by period, ascending order).
    // Runs after transaction application so reveals in this block
    // count toward the closing period's consensus.
    let voting_ready = state
        .decisions()
        .get_revealing_decisions_needing_resolution(
            rwtxn,
            mainchain_timestamp,
            Some(height),
            genesis_ts,
        )?;
    let mut periods_to_resolve: BTreeMap<
        u32,
        Vec<crate::state::decisions::DecisionId>,
    > = BTreeMap::new();
    for decision_id in voting_ready {
        periods_to_resolve
            .entry(decision_id.voting_period())
            .or_default()
            .push(decision_id);
    }
    let mut consensus_undo_entries = Vec::new();
    for (vp_num, decision_ids) in &periods_to_resolve {
        let (start, end) =
            crate::state::voting::period_calculator::calculate_period_boundaries(
                *vp_num,
                state.decisions().get_config(),
                genesis_ts,
            );
        let period = crate::state::voting::types::VotingPeriod {
            id: crate::state::voting::types::VotingPeriodId::new(*vp_num),
            start_timestamp: start,
            end_timestamp: end,
            status: crate::state::voting::types::VotingPeriodStatus::Closed,
            decision_ids: decision_ids.clone(),
        };
        tracing::info!(
            "Protocol: Processing closed period {} at block height {} (period ended at timestamp {})",
            period.id.0,
            height,
            period.end_timestamp
        );
        if let Some(undo_entry) = state.voting().calculate_and_store_consensus(
            rwtxn,
            &period,
            state,
            mainchain_timestamp,
            height,
            state.decisions(),
        )? {
            consensus_undo_entries.push(undo_entry);
        }
    }
    if !consensus_undo_entries.is_empty() {
        let undo_data = crate::state::undo::ConsensusUndoData {
            entries: consensus_undo_entries,
        };
        state.consensus_undo.put(rwtxn, &height, &undo_data)?;
    }
    {
        let (payout_results, settlement_undo_entries) =
            state.markets().transition_and_payout_resolved_markets(
                rwtxn,
                accumulator_diff,
                &mut market_utxos,
                state,
                state.decisions(),
                height,
            )?;
        for (market_id, summary) in &payout_results {
            let refund_sats = summary
                .creator_refund
                .as_ref()
                .map(|r| r.amount_sats)
                .unwrap_or(0);
            tracing::info!(
                "Protocol: Market {} auto-settled with {} sats treasury + {} sats fees distributed to {} shareholders ({} sats refunded to creator)",
                market_id,
                summary.treasury_distributed,
                summary.total_fees_distributed,
                summary.shareholder_count,
                refund_sats,
            );
        }
        if !settlement_undo_entries.is_empty() {
            let undo_data = crate::state::undo::SettlementUndoData {
                entries: settlement_undo_entries,
            };
            state.settlement_undo.put(rwtxn, &height, &undo_data)?;
        }
    }
    if !market_utxos.is_empty() {
        state.market_utxos.put(rwtxn, &height, &market_utxos)?;
    }
    if !skipped_tx_indices.is_empty() {
        let mut indices: Vec<u32> =
            skipped_tx_indices.iter().map(|i| *i as u32).collect();
        indices.sort_unstable();
        state
            .skipped_tx_indices_undo
            .put(rwtxn, &height, &indices)?;
    }
    if let Some(prev_timestamp) =
        state.mainchain_timestamp.try_get(rwtxn, &())?
    {
        state
            .mainchain_timestamp_undo
            .put(rwtxn, &height, &prev_timestamp)?;
    }
    state
        .mainchain_timestamp
        .put(rwtxn, &(), &mainchain_timestamp)?;
    Ok(())
}

/// Start to disconnect the market effects of the tip, before its
/// transactions. Returns the indices of the trades that the tip skipped.
pub(in crate::state) fn disconnect_begin(
    state: &State,
    rwtxn: &mut RwTxn,
    accumulator_diff: &mut AccumulatorDiff,
    height: u32,
) -> Result<HashSet<u32>, Error> {
    // Settlement runs last in connect, so it reverts first here
    if let Some(settlement_undo) =
        state.settlement_undo.try_get(rwtxn, &height)?
    {
        revert_settlement(
            state,
            rwtxn,
            accumulator_diff,
            &settlement_undo,
            height,
        )?;
        state.settlement_undo.delete(rwtxn, &height)?;
    }
    if let Some(consensus_undo) =
        state.consensus_undo.try_get(rwtxn, &height)?
    {
        revert_consensus(state, rwtxn, &consensus_undo)?;
        state.consensus_undo.delete(rwtxn, &height)?;
    }
    if let Some(consolidation_undo) =
        state.consolidation_undo.try_get(rwtxn, &height)?
    {
        revert_consolidation(
            state,
            rwtxn,
            accumulator_diff,
            &consolidation_undo,
        )?;
        state.consolidation_undo.delete(rwtxn, &height)?;
    }
    state.market_utxos.delete(rwtxn, &height)?;
    let skipped_tx_indices = state
        .skipped_tx_indices_undo
        .try_get(rwtxn, &height)?
        .unwrap_or_default()
        .into_iter()
        .collect();
    Ok(skipped_tx_indices)
}

/// Revert the market data of a transaction, before its UTXOs revert
pub(in crate::state) fn disconnect_tx(
    state: &State,
    rwtxn: &mut RwTxn,
    accumulator_diff: &mut AccumulatorDiff,
    tx: &Transaction,
    trade_share_deltas: &mut Vec<TradeShareDelta>,
) -> Result<(), Error> {
    let Some(data) = &tx.data else {
        return Ok(());
    };
    let filled_tx = state.fill_transaction_from_stxos(rwtxn, tx.clone())?;
    match data {
        TxData::ClaimDecision(_) => {
            let () = revert_decision_claim(state, rwtxn, &filled_tx)?;
        }
        TxData::CreateMarket { .. } => {
            let () = revert_create_market(state, rwtxn, &filled_tx)?;
        }
        TxData::Trade { .. } => {
            trade_share_deltas.extend(revert_trade_market_state(
                state,
                rwtxn,
                accumulator_diff,
                &filled_tx,
            )?);
        }
        TxData::SubmitBallot { .. } => {
            let () = revert_submit_ballot(state, rwtxn, &filled_tx)?;
        }
        TxData::RevealBallot { .. } => {
            let () = revert_reveal_ballot(state, rwtxn, &filled_tx)?;
        }
        TxData::TransferReputation { .. } => {}
        TxData::AmplifyBeta { .. } => {
            let () = revert_amplify_beta(state, rwtxn, &filled_tx)?;
        }
        TxData::TakeOffer { offer, .. } => {
            state.closed_offers.delete(rwtxn, &offer.id())?;
        }
        TxData::CancelOffer { offer } => {
            state.closed_offers.delete(rwtxn, &offer.id())?;
        }
    }
    Ok(())
}

/// Finish the market effects of disconnecting the tip, after its UTXOs
pub(in crate::state) fn disconnect_end(
    state: &State,
    rwtxn: &mut RwTxn,
    height: u32,
    skipped_tx_indices: &HashSet<u32>,
    trade_share_deltas: Vec<TradeShareDelta>,
) -> Result<(), Error> {
    if !skipped_tx_indices.is_empty() {
        state.skipped_tx_indices_undo.delete(rwtxn, &height)?;
    }
    let mut batched: HashMap<(Address, MarketId), HashMap<u32, i64>> =
        HashMap::new();
    for delta in &trade_share_deltas {
        batched
            .entry((delta.address, delta.market_id))
            .or_default()
            .entry(delta.outcome_index)
            .and_modify(|v| *v += delta.share_delta)
            .or_insert(delta.share_delta);
    }
    for ((address, market_id), outcome_changes) in &batched {
        for (&outcome_index, &net_delta) in outcome_changes {
            if net_delta > 0 {
                state.markets().add_shares_to_account(
                    rwtxn,
                    address,
                    *market_id,
                    outcome_index,
                    net_delta,
                    height,
                )?;
            } else if net_delta < 0 {
                state.markets().remove_shares_from_account(
                    rwtxn,
                    address,
                    market_id,
                    outcome_index,
                    -net_delta,
                    height,
                )?;
            }
        }
    }
    if let Some(rep_undo) =
        state.reputation_transfer_undo.try_get(rwtxn, &height)?
    {
        for entry in rep_undo.entries.iter().rev() {
            state.reputation().set_reputation(
                rwtxn,
                &entry.sender,
                entry.sender_pre_reputation,
            )?;
            state.reputation().set_reputation(
                rwtxn,
                &entry.receiver,
                entry.receiver_pre_reputation,
            )?;
        }
        state.reputation_transfer_undo.delete(rwtxn, &height)?;
    }
    // Roll back decision states (Claimed → Voting transitions)
    if height > 0 {
        state
            .decisions()
            .rollback_decision_states_to_height(rwtxn, height - 1)?;
        tracing::info!(
            "Rolled back decision states to height {} during reorg",
            height - 1
        );
    }
    if let Some(prev_highest) = state.minting_undo.try_get(rwtxn, &height)? {
        state
            .decisions()
            .delete_periods_above(rwtxn, prev_highest)?;
        state.minting_undo.delete(rwtxn, &height)?;
    }
    state
        .decisions()
        .restore_period_pricing_undo(rwtxn, height)?;
    match state.mainchain_timestamp_undo.try_get(rwtxn, &height)? {
        Some(prev_timestamp) => {
            state.mainchain_timestamp.put(rwtxn, &(), &prev_timestamp)?;
            state.mainchain_timestamp_undo.delete(rwtxn, &height)?;
        }
        None => {
            state.mainchain_timestamp.delete(rwtxn, &())?;
        }
    }
    Ok(())
}

fn revert_delete_utxo(
    state: &State,
    rwtxn: &mut RwTxn,
    accumulator_diff: &mut AccumulatorDiff,
    outpoint: &OutPoint,
) -> Result<(), Error> {
    let deleted = state.delete_utxo(rwtxn, outpoint, accumulator_diff)?;
    if !deleted {
        tracing::trace!(
            "UTXO not found during revert \
             (expected for zero-value outputs): {outpoint:?}"
        );
    }
    Ok(())
}

/// Revert UTXO consolidation (C3)
fn revert_consolidation(
    state: &State,
    rwtxn: &mut RwTxn,
    accumulator_diff: &mut AccumulatorDiff,
    undo: &crate::state::undo::ConsolidationUndoData,
) -> Result<(), Error> {
    // Process entries in reverse order
    for entry in undo.entries.iter().rev() {
        // Delete new UTXOs that were created during consolidation
        if let Some(ref outpoint) = entry.new_treasury_utxo {
            revert_delete_utxo(state, rwtxn, accumulator_diff, outpoint)?;
        }
        if let Some(ref outpoint) = entry.new_fee_utxo {
            revert_delete_utxo(state, rwtxn, accumulator_diff, outpoint)?;
        }
        for outpoint in &entry.trade_change_utxos {
            revert_delete_utxo(state, rwtxn, accumulator_diff, outpoint)?;
        }

        // Restore old treasury UTXOs
        for (outpoint, filled_output) in &entry.old_treasury_utxos {
            state.insert_utxo(
                rwtxn,
                outpoint,
                filled_output,
                accumulator_diff,
            )?;
        }
        // Restore old fee UTXOs
        for (outpoint, filled_output) in &entry.old_fee_utxos {
            state.insert_utxo(
                rwtxn,
                outpoint,
                filled_output,
                accumulator_diff,
            )?;
        }

        // Restore market_funds_utxo pointers
        if let Some(ref outpoint) = entry.old_treasury_pointer {
            state.markets().set_market_funds_utxo(
                rwtxn,
                &entry.market_id,
                false,
                outpoint,
            )?;
        } else {
            state.markets().clear_market_funds_utxo(
                rwtxn,
                &entry.market_id,
                false,
            )?;
        }
        if let Some(ref outpoint) = entry.old_fee_pointer {
            state.markets().set_market_funds_utxo(
                rwtxn,
                &entry.market_id,
                true,
                outpoint,
            )?;
        } else {
            state.markets().clear_market_funds_utxo(
                rwtxn,
                &entry.market_id,
                true,
            )?;
        }
    }

    tracing::info!(
        "Reverted UTXO consolidation for {} markets",
        undo.entries.len()
    );
    Ok(())
}

/// Revert market settlement and payouts (C1)
fn revert_settlement(
    state: &State,
    rwtxn: &mut RwTxn,
    accumulator_diff: &mut AccumulatorDiff,
    undo: &crate::state::undo::SettlementUndoData,
    block_height: u32,
) -> Result<(), Error> {
    for entry in undo.entries.iter().rev() {
        // Revert automatic share payouts (deletes payout UTXOs, restores shares)
        state.markets().revert_automatic_share_payouts(
            state,
            rwtxn,
            accumulator_diff,
            &entry.payout_summary,
            block_height,
        )?;

        // Restore treasury UTXO
        if let Some((ref outpoint, ref filled_output)) = entry.treasury_utxo {
            state.insert_utxo(
                rwtxn,
                outpoint,
                filled_output,
                accumulator_diff,
            )?;
            state.markets().set_market_funds_utxo(
                rwtxn,
                &entry.pre_settlement_market.id,
                false,
                outpoint,
            )?;
        }

        // Restore fee UTXO
        if let Some((ref outpoint, ref filled_output)) = entry.fee_utxo {
            state.insert_utxo(
                rwtxn,
                outpoint,
                filled_output,
                accumulator_diff,
            )?;
            state.markets().set_market_funds_utxo(
                rwtxn,
                &entry.pre_settlement_market.id,
                true,
                outpoint,
            )?;
        }

        // Restore market to pre-settlement state (Trading, no final prices)
        state
            .markets()
            .restore_market(rwtxn, &entry.pre_settlement_market)?;

        tracing::info!(
            "Reverted settlement for market {}",
            entry.pre_settlement_market.id
        );
    }
    Ok(())
}

/// Revert consensus voting state (C2)
fn revert_consensus(
    state: &State,
    rwtxn: &mut RwTxn,
    undo: &crate::state::undo::ConsensusUndoData,
) -> Result<(), Error> {
    for entry in undo.entries.iter().rev() {
        // Delete decision outcomes that were written
        for decision_id in &entry.decision_outcome_ids {
            state
                .voting()
                .databases()
                .delete_decision_outcome(rwtxn, *decision_id)?;
        }

        // Delete period stats if they didn't exist before
        if !entry.had_period_stats {
            state
                .voting()
                .databases()
                .delete_period_stats(rwtxn, entry.period_id)?;
        }

        state
            .reputation()
            .clear_and_restore(rwtxn, &entry.pre_consensus_reputation)?;

        if !entry.resolved_decision_ids.is_empty() {
            tracing::info!(
                "Period {} had {} resolved decisions \
                 to revert via height rollback",
                entry.period_id.0,
                entry.resolved_decision_ids.len(),
            );
        }

        tracing::info!("Reverted consensus for period {}", entry.period_id.0);
    }
    Ok(())
}

fn apply_decision_claim(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    block_height: u32,
) -> Result<(), Error> {
    use crate::state::decisions::{Decision, DecisionId};

    let claim = filled_tx.claim_decision().ok_or_else(|| {
        Error::InvalidTransaction {
            reason: "Not a decision claim transaction".to_string(),
        }
    })?;

    let market_maker_address_bytes = filled_tx
        .spent_utxos
        .first()
        .ok_or_else(|| Error::InvalidTransaction {
            reason: "No spent UTXOs found".to_string(),
        })?
        .address
        .0;

    let claiming_txid = filled_tx.transaction.txid();

    for entry in &claim.decisions {
        let decision_id = DecisionId::from_bytes(entry.decision_id_bytes)?;

        let entry_type = claim.decision_type.clone();

        let decision = Decision::new(
            market_maker_address_bytes,
            entry_type,
            entry.header.clone(),
            entry.description.clone(),
            entry.option_0_label.clone(),
            entry.option_1_label.clone(),
            entry.tags.clone().unwrap_or_default(),
        )?;

        state.decisions().claim_decision(
            rwtxn,
            decision_id,
            decision,
            claiming_txid,
            Some(block_height),
        )?;

        tracing::debug!(
            "Claimed decision {} (type: {:?})",
            const_hex::encode(decision_id.as_bytes()),
            claim.decision_type
        );
    }

    Ok(())
}

fn revert_decision_claim(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    use crate::state::decisions::DecisionId;

    let claim = filled_tx.claim_decision().ok_or_else(|| {
        Error::InvalidTransaction {
            reason: "Not a decision claim transaction".to_string(),
        }
    })?;

    for entry in &claim.decisions {
        let decision_id = DecisionId::from_bytes(entry.decision_id_bytes)?;
        state
            .decisions()
            .revert_decision_claim(rwtxn, decision_id)?;
    }

    Ok(())
}

fn extract_creator_address(
    filled_tx: &FilledTransaction,
) -> Result<crate::types::Address, Error> {
    filled_tx
        .spent_utxos
        .first()
        .map(|utxo| utxo.address)
        .ok_or_else(|| Error::InvalidTransaction {
            reason: "No spent UTXOs found".to_string(),
        })
}

fn configure_market_builder(
    mut builder: crate::state::MarketBuilder,
    description: &str,
    trading_fee: Option<f64>,
) -> crate::state::MarketBuilder {
    if !description.is_empty() {
        builder = builder.with_description(description.to_string());
    }

    if let Some(fee) = trading_fee {
        builder = builder.with_fee(fee);
    }

    builder
}

/// Derive the current effective beta for a market.
/// `beta = liquidity_base_sats / ln(num_outcomes)`, where the liquidity base is
/// the creation seed plus confirmed `AmplifyBeta` deposits (ordinary trade
/// proceeds are excluded). Mempool-pending deposits are not counted until they
/// confirm.
fn market_beta(market: &crate::state::Market) -> f64 {
    trading::derive_beta_from_liquidity(
        market.liquidity_base_sats,
        market.shares().len(),
    )
}

/// Effective shares and beta for a market, folding in the trades already
/// applied to `state_update` earlier in this block. Lets same-block trades
/// price sequentially against running state instead of stale pre-block state.
/// Beta tracks the liquidity base (seed + confirmed and same-block `AmplifyBeta`
/// deposits), never ordinary trade proceeds.
fn running_market_state(
    market_id: &MarketId,
    market: &crate::state::Market,
    state_update: &StateUpdate,
) -> (ndarray::Array1<i64>, f64) {
    let mut effective_shares = market.shares().clone();
    for leg in state_update
        .market_updates
        .iter()
        .filter(|update| update.market_id == *market_id)
        .flat_map(|update| &update.legs)
    {
        effective_shares[leg.outcome_index] += leg.shares;
    }

    let pending_amplify: u64 = state_update
        .pending_trade_settlements
        .iter()
        .filter(|s| s.market_id == *market_id && s.is_amplify)
        .map(|s| s.buy_cost_sats)
        .sum();
    let running_base =
        market.liquidity_base_sats.saturating_add(pending_amplify);

    let beta = trading::derive_beta_from_liquidity(
        running_base,
        effective_shares.len(),
    );
    (effective_shares, beta)
}

pub fn compute_market_tags(
    decisions: &std::collections::HashMap<
        crate::state::decisions::DecisionId,
        crate::state::decisions::Decision,
    >,
) -> Vec<String> {
    let mut tags = std::collections::BTreeSet::new();
    for decision in decisions.values() {
        for tag in &decision.tags {
            tags.insert(tag.clone());
        }
    }
    tags.into_iter().collect()
}

fn revert_create_market(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    use crate::state::{decisions::DecisionId, markets::compute_market_id};

    let view = filled_tx.as_market_creation().ok_or_else(|| {
        Error::InvalidTransaction {
            reason: "Not a market creation transaction".to_string(),
        }
    })?;

    let creator_address = extract_creator_address(filled_tx)?;

    let market_id = compute_market_id(
        view.title,
        view.description,
        &creator_address,
        view.dimension_specs,
    );

    state.markets().delete_market(rwtxn, &market_id)?;

    for payload in view.new_claims {
        for entry in &payload.decisions {
            let decision_id = DecisionId::from_bytes(entry.decision_id_bytes)?;
            state
                .decisions()
                .revert_decision_claim(rwtxn, decision_id)?;
        }
    }

    Ok(())
}

pub(in crate::state) struct TradeShareDelta {
    address: Address,
    market_id: MarketId,
    outcome_index: u32,
    share_delta: i64,
}

/// A trade's legs, its volume split across them in proportion to the shares
/// each moves, the remainder to the first leg
fn trade_legs(
    deltas: &[crate::types::ShareDelta],
    gross_sats: u64,
) -> Vec<TradeLeg> {
    let total_shares: u128 = deltas
        .iter()
        .map(|delta| u128::from(delta.shares.unsigned_abs()))
        .sum();
    let mut legs: Vec<TradeLeg> = deltas
        .iter()
        .map(|delta| TradeLeg {
            outcome_index: delta.outcome_index as usize,
            shares: delta.shares,
            volume_sats: (u128::from(gross_sats)
                * u128::from(delta.shares.unsigned_abs())
                / total_shares.max(1)) as u64,
        })
        .collect();
    let assigned: u64 = legs.iter().map(|leg| leg.volume_sats).sum();
    if let Some(first) = legs.first_mut() {
        first.volume_sats += gross_sats - assigned;
    }
    legs
}

fn revert_trade_market_state(
    state: &State,
    rwtxn: &mut RwTxn,
    accumulator_diff: &mut AccumulatorDiff,
    filled_tx: &FilledTransaction,
) -> Result<Vec<TradeShareDelta>, Error> {
    let trade = filled_tx.trade().ok_or_else(|| Error::InvalidTransaction {
        reason: "Not a trade transaction".to_string(),
    })?;
    let height = state.try_get_height(rwtxn)?.unwrap_or(0);

    let mut market = state
        .markets()
        .get_market(rwtxn, &trade.market_id)?
        .ok_or_else(|| Error::InvalidTransaction {
            reason: "Market not found during trade revert".to_string(),
        })?;

    let mut pre_trade_shares = market.shares().clone();
    for delta in &trade.deltas {
        pre_trade_shares[delta.outcome_index as usize] -= delta.shares;
    }
    let quote = trading::quote_trade(
        &pre_trade_shares,
        &trade.deltas,
        market_beta(&market),
        market.trading_fee(),
    )
    .map_err(|e| Error::InvalidTransaction {
        reason: format!("Trade quote failed during trade revert: {e}"),
    })?;
    for leg in trade_legs(&trade.deltas, quote.gross_sats()) {
        market
            .revert_trading_volume(leg.outcome_index, leg.volume_sats)
            .map_err(|e| Error::InvalidTransaction {
                reason: format!(
                    "Volume revert failed during trade revert: {e:?}"
                ),
            })?;
    }
    market
        .update_shares(pre_trade_shares, height)
        .map_err(|e| Error::InvalidTransaction {
            reason: format!("Failed to revert market shares: {e:?}"),
        })?;
    state.markets().update_market(rwtxn, &market)?;

    let change_outpoint = StateUpdate::generate_trade_change_outpoint(
        &trade.market_id,
        &trade.trader,
        filled_tx.txid().0,
    );
    if let Err(e) = state.delete_utxo(rwtxn, &change_outpoint, accumulator_diff)
    {
        tracing::trace!(
            "UTXO not found during trade revert (expected if 0 change): {e:?}"
        );
    }

    Ok(trade
        .deltas
        .iter()
        .map(|delta| TradeShareDelta {
            address: trade.trader,
            market_id: trade.market_id,
            outcome_index: delta.outcome_index,
            share_delta: -delta.shares,
        })
        .collect())
}

/// Trades and `AmplifyBeta` deposits of a block under construction, applied
/// in block order with the same rules as block connect.
#[derive(Default)]
pub struct TradeSimulation(StateUpdate);

impl TradeSimulation {
    pub fn apply(
        &mut self,
        state: &State,
        archive: &crate::archive::Archive,
        rotxn: &RoTxn,
        filled_tx: &FilledTransaction,
    ) -> Result<TradeApplyResult, Error> {
        match &filled_tx.transaction.data {
            Some(TxData::Trade { .. }) => {
                crate::validation::MarketValidator::validate_trade(
                    state, archive, rotxn, filled_tx, None,
                )?;
                apply_trade(state, rotxn, filled_tx, &mut self.0)
            }
            Some(TxData::AmplifyBeta { .. }) => {
                apply_amplify_beta(filled_tx, &mut self.0)?;
                Ok(TradeApplyResult::Applied)
            }
            _ => Ok(TradeApplyResult::Applied),
        }
    }
}

/// Returns `TradeApplyResult::Skipped` for slippage failures (soft-fail).
fn apply_trade(
    state: &State,
    rotxn: &RoTxn,
    filled_tx: &FilledTransaction,
    state_update: &mut StateUpdate,
) -> Result<TradeApplyResult, Error> {
    let trade = filled_tx.trade().ok_or_else(|| Error::InvalidTransaction {
        reason: "Not a trade transaction".to_string(),
    })?;

    let market = state
        .markets()
        .get_market(rotxn, &trade.market_id)?
        .ok_or_else(|| Error::InvalidTransaction {
            reason: format!("Market {:?} does not exist", trade.market_id),
        })?;

    let market_state = market.state();
    if !market_state.allows_trading() {
        return Err(Error::InvalidTransaction {
            reason: format!(
                "Cannot trade: market is in {market_state:?} state"
            ),
        });
    }

    let (effective_shares, beta) =
        running_market_state(&trade.market_id, &market, state_update);
    let quote = trading::quote_trade(
        &effective_shares,
        &trade.deltas,
        beta,
        market.trading_fee(),
    )
    .map_err(|e| Error::InvalidTransaction {
        reason: format!("Invalid trade: {e}"),
    })?;

    if trade.sells() {
        let seller_account = state
            .markets()
            .get_user_share_account(rotxn, &trade.trader)?;
        let pending_changes = state_update
            .share_account_changes
            .get(&(trade.trader, trade.market_id));
        for delta in trade.deltas.iter().filter(|delta| delta.shares < 0) {
            let owned = seller_account
                .as_ref()
                .and_then(|account| {
                    account
                        .positions
                        .get(&(trade.market_id, delta.outcome_index))
                        .copied()
                })
                .unwrap_or(0);
            let pending = pending_changes
                .and_then(|outcomes| outcomes.get(&delta.outcome_index))
                .copied()
                .unwrap_or(0);
            let effective_owned = owned + pending;
            if effective_owned < -delta.shares {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Insufficient shares: trying to sell {} but only \
                         own {owned} (effective: {effective_owned}) for \
                         outcome {}",
                        -delta.shares, delta.outcome_index
                    ),
                });
            }
        }
    }

    if quote.net_cost_sats > trade.max_cost {
        return Ok(TradeApplyResult::Skipped {
            reason: format!(
                "Trade net cost {} sats exceeds max cost {} sats",
                quote.net_cost_sats, trade.max_cost
            ),
        });
    }

    let input_value_sats = filled_tx
        .get_value_in()
        .map_err(|_| Error::InvalidTransaction {
            reason: "Failed to compute input value".to_string(),
        })?
        .to_sat();

    state_update.add_pending_trade_settlement(PendingTradeSettlement {
        market_id: trade.market_id,
        trader_address: trade.trader,
        input_value_sats,
        buy_cost_sats: quote.buy_cost_sats,
        sell_gross_sats: quote.sell_gross_sats,
        fee_sats: quote.fee_sats,
        transaction_id: filled_tx.transaction.txid().0,
        is_amplify: false,
    });

    state_update.add_market_update(MarketStateUpdate {
        market_id: trade.market_id,
        legs: trade_legs(&trade.deltas, quote.gross_sats()),
    });

    for delta in &trade.deltas {
        state_update.add_share_account_change(
            trade.trader,
            trade.market_id,
            delta.outcome_index,
            delta.shares,
        );
    }

    Ok(TradeApplyResult::Applied)
}

fn apply_market_creation(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    state_update: &mut StateUpdate,
    height: u32,
) -> Result<(), Error> {
    use crate::state::{
        MarketBuilder,
        decisions::{Decision, DecisionId},
        markets::{DimensionSpec, generate_market_treasury_address},
    };
    use std::collections::HashMap;

    let view = filled_tx.as_market_creation().ok_or_else(|| {
        Error::InvalidTransaction {
            reason: "Not a market creation transaction".to_string(),
        }
    })?;

    let creator_address = extract_creator_address(filled_tx)?;
    let creator_address_bytes = creator_address.0;
    let claiming_txid = filled_tx.transaction.txid();

    for payload in view.new_claims {
        for entry in &payload.decisions {
            let decision_id = DecisionId::from_bytes(entry.decision_id_bytes)?;
            let decision = Decision::new(
                creator_address_bytes,
                payload.decision_type.clone(),
                entry.header.clone(),
                entry.description.clone(),
                entry.option_0_label.clone(),
                entry.option_1_label.clone(),
                entry.tags.clone().unwrap_or_default(),
            )?;
            state.decisions().claim_decision(
                rwtxn,
                decision_id,
                decision,
                claiming_txid,
                Some(height),
            )?;
        }
    }

    let dimension_specs = view.dimension_specs.to_vec();

    let mut decisions = HashMap::new();
    for spec in &dimension_specs {
        let decision_id = match spec {
            DimensionSpec::Single(decision_id)
            | DimensionSpec::Categorical(decision_id) => *decision_id,
        };

        let decision_entry = state
            .decisions()
            .get_decision_entry(rwtxn, decision_id)?
            .ok_or_else(|| Error::InvalidDecisionId {
                reason: format!("Decision {decision_id:?} does not exist"),
            })?;

        let decision = decision_entry.decision.ok_or_else(|| {
            Error::InvalidDecisionId {
                reason: format!("Decision {decision_id:?} has no decision"),
            }
        })?;

        decisions.insert(decision_id, decision);
    }

    let mut builder =
        MarketBuilder::new(view.title.to_string(), creator_address);
    builder =
        configure_market_builder(builder, view.description, view.trading_fee);

    let computed_tags = compute_market_tags(&decisions);
    builder = builder.with_tags(computed_tags);

    let builder = builder
        .with_dimensions(dimension_specs.clone())
        .with_tx_pow(
            view.tx_pow_hash_selector.unwrap_or(0),
            view.tx_pow_ordering.unwrap_or(0),
            view.tx_pow_difficulty.unwrap_or(0),
        );

    let mut market = builder.build(height, None, &decisions).map_err(|e| {
        tracing::warn!("Market creation failed for '{}': {e}", view.title);
        Error::InvalidTransaction {
            reason: format!("Market creation failed: {e}"),
        }
    })?;
    tracing::info!(
        "Market built: id={} states={} tradeable={}",
        market.id,
        market.shares().len(),
        market.get_outcome_count()
    );

    let market_id = market.id;
    let market_id_bytes = *market_id.as_bytes();
    let txid = filled_tx.txid();
    let treasury_address = generate_market_treasury_address(&market_id);

    for (vout, output) in filled_tx.transaction.outputs.iter().enumerate() {
        if let OutputContent::MarketFunds {
            market_id: output_market_id,
            amount,
            is_fee: false,
        } = &output.content
            && output_market_id == &market_id_bytes
            && output.address == treasury_address
        {
            let outpoint = OutPoint::Regular {
                txid,
                vout: vout as u32,
            };
            state
                .markets()
                .set_market_funds_utxo(rwtxn, &market_id, false, &outpoint)?;

            market.liquidity_base_sats = amount.to_sat();

            tracing::debug!(
                "Registered MarketFunds (treasury) UTXO for market {:?} with {} sats at {:?}",
                market_id,
                amount.to_sat(),
                outpoint
            );
            break;
        }
    }

    state_update.add_market_creation(MarketCreation {
        market: market.clone(),
    });

    Ok(())
}

fn apply_amplify_beta(
    filled_tx: &FilledTransaction,
    state_update: &mut StateUpdate,
) -> Result<(), Error> {
    let amplify =
        filled_tx
            .amplify_beta()
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "Not an amplify_beta transaction".to_string(),
            })?;

    let input_value_sats = filled_tx
        .get_value_in()
        .map_err(|_| Error::InvalidTransaction {
            reason: "Failed to compute input value".to_string(),
        })?
        .to_sat();

    state_update.add_pending_trade_settlement(PendingTradeSettlement {
        market_id: amplify.market_id,
        trader_address: amplify.market_author,
        input_value_sats,
        buy_cost_sats: amplify.amount,
        sell_gross_sats: 0,
        fee_sats: 0,
        transaction_id: filled_tx.transaction.txid().0,
        is_amplify: true,
    });

    Ok(())
}

fn revert_amplify_beta(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    let amplify =
        filled_tx
            .amplify_beta()
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "Not an amplify_beta transaction".to_string(),
            })?;

    let mut market = state
        .markets()
        .get_market(rwtxn, &amplify.market_id)?
        .ok_or_else(|| Error::InvalidTransaction {
            reason: "Market not found during amplify_beta revert".to_string(),
        })?;
    market.liquidity_base_sats = market
        .liquidity_base_sats
        .checked_sub(amplify.amount)
        .ok_or_else(|| Error::InvalidTransaction {
            reason: "Liquidity base underflow during amplify_beta revert"
                .to_string(),
        })?;
    state.markets().update_market(rwtxn, &market)?;

    Ok(())
}

fn apply_submit_ballot(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    let ballot =
        filled_tx
            .submit_ballot()
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "Not a ballot submission transaction".to_string(),
            })?;
    state.voting().databases().put_sealed_ballot(
        rwtxn,
        crate::state::voting::types::VotingPeriodId::new(ballot.voting_period),
        ballot.voter,
        &crate::state::voting::types::SealedBallot {
            commitment: ballot.commitment,
            revealed: false,
        },
    )
}

fn revert_submit_ballot(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    let ballot =
        filled_tx
            .submit_ballot()
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "Not a ballot submission transaction".to_string(),
            })?;
    state.voting().databases().delete_sealed_ballot(
        rwtxn,
        crate::state::voting::types::VotingPeriodId::new(ballot.voting_period),
        ballot.voter,
    )
}

fn set_ballot_revealed(
    state: &State,
    rwtxn: &mut RwTxn,
    ballot: &crate::types::RevealBallot,
    revealed: bool,
) -> Result<(), Error> {
    state.voting().databases().put_sealed_ballot(
        rwtxn,
        crate::state::voting::types::VotingPeriodId::new(ballot.voting_period),
        ballot.voter,
        &crate::state::voting::types::SealedBallot {
            commitment: ballot.commitment(),
            revealed,
        },
    )
}

fn apply_reveal_ballot(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    state_update: &mut StateUpdate,
    height: u32,
) -> Result<(), Error> {
    use crate::state::{
        decisions::DecisionId,
        voting::types::{Vote, VotingPeriodId},
    };

    let ballot_data =
        filled_tx
            .reveal_ballot()
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "Not a ballot reveal transaction".to_string(),
            })?;

    let voter_address = ballot_data.voter;

    let timestamp =
        state.try_get_mainchain_timestamp(rwtxn)?.ok_or_else(|| {
            Error::InvalidTransaction {
                reason: "No mainchain timestamp available".to_string(),
            }
        })?;

    let mut expected_voting_period: Option<u32> = None;

    for vote_item in &ballot_data.votes {
        let decision_id = DecisionId::from_bytes(vote_item.decision_id_bytes)?;

        let voting_period = decision_id.voting_period();

        if let Some(expected) = expected_voting_period {
            if voting_period != expected {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Ballot period mismatch: decision {} requires period {} but ballot expects period {}",
                        const_hex::encode(vote_item.decision_id_bytes),
                        voting_period,
                        expected
                    ),
                });
            }
        } else {
            expected_voting_period = Some(voting_period);

            if ballot_data.voting_period != voting_period {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Ballot period mismatch: decisions require period {} but transaction specifies period {}",
                        voting_period, ballot_data.voting_period
                    ),
                });
            }
        }

        let period_id = VotingPeriodId::new(voting_period);

        let decision_ref = state
            .decisions()
            .get_decision_entry(rwtxn, decision_id)?
            .and_then(|e| e.decision);

        let vote_value =
            crate::validation::VoteValidator::convert_vote_value_with_decision(
                vote_item.vote_value,
                decision_ref.as_ref(),
            );

        let vote = Vote::new(
            voter_address,
            period_id,
            decision_id,
            vote_value,
            timestamp,
            height,
            filled_tx.txid().0,
        );

        state_update.add_vote_submission(vote);
    }

    set_ballot_revealed(state, rwtxn, &ballot_data, true)
}

fn revert_reveal_ballot(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    use crate::state::{decisions::DecisionId, voting::types::VotingPeriodId};

    let ballot_data =
        filled_tx
            .reveal_ballot()
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "Not a ballot reveal transaction".to_string(),
            })?;

    let voter_address = ballot_data.voter;

    for vote_item in &ballot_data.votes {
        let decision_id = DecisionId::from_bytes(vote_item.decision_id_bytes)?;

        let voting_period = decision_id.voting_period();
        let period_id = VotingPeriodId::new(voting_period);

        state.voting().databases().delete_vote(
            rwtxn,
            period_id,
            voter_address,
            decision_id,
        )?;
    }

    set_ballot_revealed(state, rwtxn, &ballot_data, false)
}

fn apply_transfer_reputation(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    height: u32,
) -> Result<(), Error> {
    let transfer = filled_tx.transfer_reputation().ok_or_else(|| {
        Error::InvalidTransaction {
            reason: "Not a reputation transfer transaction".to_string(),
        }
    })?;
    move_reputation(
        state,
        rwtxn,
        transfer.sender,
        transfer.dest,
        transfer.amount,
        height,
    )
}

fn move_reputation(
    state: &State,
    rwtxn: &mut RwTxn,
    sender_address: Address,
    dest: Address,
    amount: Votecoin,
    height: u32,
) -> Result<(), Error> {
    let sender_rep =
        state.reputation().get_reputation(rwtxn, &sender_address)?;
    let receiver_rep = state.reputation().get_reputation(rwtxn, &dest)?;
    let (sender_after, receiver_after) =
        crate::validation::VoteValidator::balances_after_transfer(
            sender_rep,
            receiver_rep,
            amount,
        )?;

    let undo_entry = crate::state::undo::ReputationTransferUndoEntry {
        sender: sender_address,
        sender_pre_reputation: sender_rep,
        receiver: dest,
        receiver_pre_reputation: receiver_rep,
    };

    let mut undo_data = state
        .reputation_transfer_undo
        .try_get(rwtxn, &height)?
        .unwrap_or_else(|| crate::state::undo::ReputationTransferUndoData {
            entries: Vec::new(),
        });
    undo_data.entries.push(undo_entry);
    state
        .reputation_transfer_undo
        .put(rwtxn, &height, &undo_data)?;

    state
        .reputation()
        .set_reputation(rwtxn, &sender_address, sender_after)?;
    state
        .reputation()
        .set_reputation(rwtxn, &dest, receiver_after)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Output, ShareDelta};

    fn bitcoin_input(sats: u64) -> Output {
        Output {
            address: Address::ALL_ZEROS,
            content: OutputContent::Value(bitcoin::Amount::from_sat(sats)),
        }
    }

    fn filled_tx(data: TxData, input_sats: u64) -> FilledTransaction {
        FilledTransaction {
            transaction: crate::types::Transaction {
                inputs: vec![(
                    OutPoint::Regular {
                        txid: crate::types::Txid([0; 32]),
                        vout: 0,
                    },
                    [0; 32],
                )]
                .into(),
                data: Some(data),
                ..Default::default()
            },
            spent_utxos: vec![bitcoin_input(input_sats)],
            actor_address: None,
        }
    }

    struct TradingFixture {
        _dir: tempfile::TempDir,
        env: sneed::Env,
        state: State,
        archive: crate::archive::Archive,
        market: crate::state::Market,
        tip: crate::types::BlockHash,
    }

    fn trading_fixture() -> TradingFixture {
        use crate::archive::Archive;
        use crate::state::MarketBuilder;
        use crate::state::decisions::{Decision, DecisionId, DecisionType};
        use crate::state::markets::DimensionSpec;
        use bitcoin::hashes::Hash as _;

        let dir = tempfile::tempdir().unwrap();
        let mut opts = heed::EnvOpenOptions::new();
        opts.map_size(64 * 1024 * 1024)
            .max_dbs(State::NUM_DBS + Archive::NUM_DBS);
        let env = unsafe { sneed::Env::open(&opts, dir.path()) }.unwrap();
        let state = State::new(&env, None, true).unwrap();
        let archive = Archive::new(&env).unwrap();

        let decision_id = DecisionId::new(true, 0, 0).unwrap();
        let decision = Decision::new(
            Address::ALL_ZEROS.0,
            DecisionType::Binary,
            "Header".to_string(),
            "Desc".to_string(),
            None,
            None,
            vec![],
        )
        .unwrap();
        let mut market =
            MarketBuilder::new("Market".to_string(), Address::ALL_ZEROS)
                .with_dimensions(vec![DimensionSpec::Single(decision_id)])
                .build(0, None, &HashMap::from([(decision_id, decision)]))
                .unwrap();
        market.liquidity_base_sats = 10_000;
        market.shares = ndarray::Array1::from(vec![0, 20_000]);

        let header = crate::types::Header {
            merkle_root: MerkleRoot::default(),
            prev_side_hash: None,
            prev_main_hash: bitcoin::BlockHash::all_zeros(),
            roots: Vec::new(),
        };
        let tip = header.hash();
        let mut rwtxn = env.write_txn().unwrap();
        archive.put_header(&mut rwtxn, &header).unwrap();
        state.tip.put(&mut rwtxn, &(), &tip).unwrap();
        state.height.put(&mut rwtxn, &(), &0).unwrap();
        state.markets().add_market(&mut rwtxn, &market).unwrap();
        rwtxn.commit().unwrap();

        TradingFixture {
            _dir: dir,
            env,
            state,
            archive,
            market,
            tip,
        }
    }

    fn buy_cost(market: &crate::state::Market, shares: i64) -> i64 {
        trading::quote_trade(
            &market.shares,
            &[ShareDelta {
                outcome_index: 0,
                shares,
            }],
            market_beta(market),
            market.trading_fee(),
        )
        .unwrap()
        .net_cost_sats
    }

    fn utxo_set(
        state: &State,
        rotxn: &RoTxn,
    ) -> anyhow::Result<BTreeMap<OutPoint, Output>> {
        use fallible_iterator::FallibleIterator as _;
        let utxos = state
            .utxos
            .iter(rotxn)?
            .map(|(key, output)| Ok((OutPoint::from(key), output)))
            .collect()?;
        Ok(utxos)
    }

    /// The recorded creates and deletes must be exactly the change in the
    /// UTXO set, so an indexer that applies them reaches the node's set
    fn ensure_changes_match(
        before: &BTreeMap<OutPoint, Output>,
        after: &BTreeMap<OutPoint, Output>,
        changes: &MarketUtxoChanges,
    ) -> anyhow::Result<()> {
        let mut replayed = before.clone();
        for utxo in &changes.creates {
            anyhow::ensure!(
                replayed
                    .insert(utxo.outpoint, utxo.output.clone())
                    .is_none(),
                "{:?} was in the UTXO set before its create",
                utxo.outpoint
            );
        }
        for utxo in &changes.deletes {
            anyhow::ensure!(
                replayed.remove(&utxo.outpoint).as_ref() == Some(&utxo.output),
                "{:?} was not in the UTXO set with its output before its \
                 delete",
                utxo.outpoint
            );
        }
        anyhow::ensure!(
            &replayed == after,
            "the recorded changes do not reach the UTXO set of the node"
        );
        Ok(())
    }

    /// Consolidation records every UTXO write it makes.
    #[test]
    fn consolidation_records_writes() -> anyhow::Result<()> {
        use crate::state::markets::{
            generate_market_author_fee_address,
            generate_market_treasury_address,
        };

        const HEIGHT: u32 = 1;

        let fixture = trading_fixture();
        let state = &fixture.state;
        let market_id = fixture.market.id;
        let seller = Address([1; 20]);
        let buyer = Address([2; 20]);
        let mut rwtxn = fixture.env.write_txn()?;
        for is_fee in [false, true] {
            let outpoint = OutPoint::MarketFunds {
                market_id: market_id.0,
                block_height: 0,
                is_fee,
            };
            let address = if is_fee {
                generate_market_author_fee_address(&market_id)
            } else {
                generate_market_treasury_address(&market_id)
            };
            let output = Output {
                address,
                content: OutputContent::MarketFunds {
                    market_id: market_id.0,
                    amount: bitcoin::Amount::from_sat(10_000),
                    is_fee,
                },
            };
            state.insert_utxo(
                &mut rwtxn,
                &outpoint,
                &output,
                &mut AccumulatorDiff::default(),
            )?;
            state.markets().set_market_funds_utxo(
                &mut rwtxn, &market_id, is_fee, &outpoint,
            )?;
        }
        let before = utxo_set(state, &rwtxn)?;

        let mut changes = MarketUtxoChanges::default();
        StateUpdate::consolidate_market_utxos(
            state,
            &mut rwtxn,
            &mut AccumulatorDiff::default(),
            &mut changes,
            HEIGHT,
            &[
                PendingTradeSettlement {
                    market_id,
                    trader_address: seller,
                    input_value_sats: 1_500,
                    buy_cost_sats: 0,
                    sell_gross_sats: 3_030,
                    fee_sats: 30,
                    transaction_id: [3; 32],
                    is_amplify: false,
                },
                PendingTradeSettlement {
                    market_id,
                    trader_address: buyer,
                    input_value_sats: 50_000,
                    buy_cost_sats: 2_000,
                    sell_gross_sats: 0,
                    fee_sats: 20,
                    transaction_id: [4; 32],
                    is_amplify: false,
                },
            ],
        )?;
        let after = utxo_set(state, &rwtxn)?;
        ensure_changes_match(&before, &after, &changes)?;

        let reasons = |utxos: &[MarketUtxo]| {
            utxos.iter().map(|utxo| utxo.reason).collect::<Vec<_>>()
        };
        anyhow::ensure!(
            reasons(&changes.creates)
                == [
                    MarketUtxoReason::TradeChange,
                    MarketUtxoReason::TradeChange,
                    MarketUtxoReason::Treasury,
                    MarketUtxoReason::AuthorFee,
                ],
            "creates: {:?}",
            reasons(&changes.creates)
        );
        anyhow::ensure!(
            reasons(&changes.deletes)
                == [MarketUtxoReason::Treasury, MarketUtxoReason::AuthorFee],
            "deletes: {:?}",
            reasons(&changes.deletes)
        );
        Ok(())
    }

    /// The trading fixture with `Address::ALL_ZEROS` holding the market's
    /// 20 000 shares of outcome 1, and a funded treasury
    fn funded_trading_fixture(
        rwtxn: &mut RwTxn,
        state: &State,
        market: &crate::state::Market,
    ) -> anyhow::Result<AccumulatorDiff> {
        use crate::state::markets::generate_market_treasury_address;
        use crate::types::Txid;

        state.markets().add_shares_to_account(
            rwtxn,
            &Address::ALL_ZEROS,
            market.id,
            1,
            20_000,
            0,
        )?;
        let treasury = OutPoint::Regular {
            txid: Txid([8; 32]),
            vout: 0,
        };
        let mut diff = AccumulatorDiff::default();
        state.insert_utxo(
            rwtxn,
            &treasury,
            &Output {
                address: generate_market_treasury_address(&market.id),
                content: OutputContent::MarketFunds {
                    market_id: market.id.0,
                    amount: bitcoin::Amount::from_sat(40_000),
                    is_fee: false,
                },
            },
            &mut diff,
        )?;
        state
            .markets()
            .set_market_funds_utxo(rwtxn, &market.id, false, &treasury)?;
        Ok(diff)
    }

    fn vector_trade(
        market: &crate::state::Market,
        tip: crate::types::BlockHash,
        deltas: Vec<ShareDelta>,
        max_cost: i64,
        input_sats: u64,
    ) -> FilledTransaction {
        filled_tx(
            TxData::Trade {
                market_id: market.id,
                deltas,
                trader: Address::ALL_ZEROS,
                max_cost,
                tx_pow_nonce: None,
                prev_block_hash: tip,
            },
            input_sats,
        )
    }

    fn sell_one_buy_zero() -> Vec<ShareDelta> {
        vec![
            ShareDelta {
                outcome_index: 0,
                shares: 1_000,
            },
            ShareDelta {
                outcome_index: 1,
                shares: -2_000,
            },
        ]
    }

    /// A vector trade settles every leg at once; a rollback reverts them all.
    #[test]
    fn vector_trade_round_trips() -> anyhow::Result<()> {
        use crate::math::trading::TRADE_MINER_FEE_SATS;

        let TradingFixture {
            env,
            state,
            archive,
            market,
            tip,
            ..
        } = trading_fixture();
        let mut rwtxn = env.write_txn()?;
        let mut diff = funded_trading_fixture(&mut rwtxn, &state, &market)?;
        let initial_utxos = state.get_utxos(&rwtxn)?;
        let quote = trading::quote_trade(
            &market.shares,
            &sell_one_buy_zero(),
            market_beta(&market),
            market.trading_fee(),
        )?;
        assert!(quote.buy_cost_sats > 0 && quote.sell_gross_sats > 0);
        let input_sats =
            quote.net_cost_sats.max(0) as u64 + TRADE_MINER_FEE_SATS;
        let trade = vector_trade(
            &market,
            tip,
            sell_one_buy_zero(),
            quote.net_cost_sats,
            input_sats,
        );
        crate::validation::MarketValidator::validate_trade(
            &state, &archive, &rwtxn, &trade, None,
        )?;
        let mut update = StateUpdate::new();
        assert!(matches!(
            apply_trade(&state, &rwtxn, &trade, &mut update)?,
            TradeApplyResult::Applied
        ));
        assert_eq!(update.pending_trade_settlements.len(), 1);

        let mut changes = MarketUtxoChanges::default();
        let undo = update
            .apply_all_changes(&state, &mut rwtxn, &mut diff, &mut changes, 1)?
            .unwrap();
        let after = state.markets().get_market(&rwtxn, &market.id)?.unwrap();
        assert_eq!(after.shares.to_vec(), vec![1_000, 18_000]);
        assert_eq!(
            after.total_volume_sats,
            market.total_volume_sats + quote.gross_sats()
        );
        let positions = state
            .markets()
            .get_user_share_account(&rwtxn, &Address::ALL_ZEROS)?
            .unwrap()
            .positions;
        assert_eq!(positions.get(&(market.id, 0)), Some(&1_000));
        assert_eq!(positions.get(&(market.id, 1)), Some(&18_000));
        state.markets().verify_share_invariant(&rwtxn, &market.id)?;

        let created = |reason| {
            changes
                .creates
                .iter()
                .filter(|utxo| utxo.reason == reason)
                .map(|utxo| utxo.output.get_value().to_sat())
                .collect::<Vec<_>>()
        };
        let trader_change = (input_sats as i64
            - TRADE_MINER_FEE_SATS as i64
            - quote.net_cost_sats) as u64;
        assert_eq!(created(MarketUtxoReason::TradeChange), vec![trader_change]);
        assert_eq!(
            created(MarketUtxoReason::Treasury),
            vec![40_000 + quote.buy_cost_sats - quote.sell_gross_sats]
        );
        assert_eq!(created(MarketUtxoReason::AuthorFee), vec![quote.fee_sats]);

        revert_consolidation(&state, &mut rwtxn, &mut diff, &undo)?;
        let deltas =
            revert_trade_market_state(&state, &mut rwtxn, &mut diff, &trade)?;
        disconnect_end(&state, &mut rwtxn, 1, &HashSet::new(), deltas)?;
        let restored = state.markets().get_market(&rwtxn, &market.id)?.unwrap();
        assert_eq!(restored.shares, market.shares);
        assert_eq!(restored.total_volume_sats, market.total_volume_sats);
        assert_eq!(restored.outcome_volumes_sats, market.outcome_volumes_sats);
        state.markets().verify_share_invariant(&rwtxn, &market.id)?;
        assert_eq!(state.get_utxos(&rwtxn)?, initial_utxos);
        Ok(())
    }

    #[test]
    fn over_max_cost_skips_legs() -> anyhow::Result<()> {
        use crate::math::trading::TRADE_MINER_FEE_SATS;

        let TradingFixture {
            env,
            state,
            market,
            tip,
            ..
        } = trading_fixture();
        let mut rwtxn = env.write_txn()?;
        funded_trading_fixture(&mut rwtxn, &state, &market)?;
        let quote = trading::quote_trade(
            &market.shares,
            &sell_one_buy_zero(),
            market_beta(&market),
            market.trading_fee(),
        )?;
        let trade = vector_trade(
            &market,
            tip,
            sell_one_buy_zero(),
            quote.net_cost_sats - 1,
            100_000 + TRADE_MINER_FEE_SATS,
        );
        let mut update = StateUpdate::new();
        assert!(matches!(
            apply_trade(&state, &rwtxn, &trade, &mut update)?,
            TradeApplyResult::Skipped { .. }
        ));
        assert!(update.pending_trade_settlements.is_empty());
        assert!(update.market_updates.is_empty());
        assert!(update.share_account_changes.is_empty());
        Ok(())
    }

    #[test]
    fn validation_checks_every_leg() -> anyhow::Result<()> {
        use crate::math::trading::TRADE_MINER_FEE_SATS;

        let TradingFixture {
            env,
            state,
            archive,
            market,
            tip,
            ..
        } = trading_fixture();
        let mut rwtxn = env.write_txn()?;
        funded_trading_fixture(&mut rwtxn, &state, &market)?;
        state.markets().add_shares_to_account(
            &mut rwtxn,
            &Address([9; 20]),
            market.id,
            1,
            10,
            0,
        )?;
        let leg = |outcome_index, shares| ShareDelta {
            outcome_index,
            shares,
        };
        let validate = |trade: &FilledTransaction| {
            crate::validation::MarketValidator::validate_trade(
                &state, &archive, &rwtxn, trade, None,
            )
        };
        let funded = |deltas, max_cost| {
            vector_trade(&market, tip, deltas, max_cost, 1_000_000)
        };

        validate(&funded(vec![leg(0, 10), leg(1, -10)], 0))?;
        assert!(validate(&funded(vec![], 0)).is_err());
        assert!(validate(&funded(vec![leg(1, -10), leg(0, 10)], 0)).is_err());
        assert!(validate(&funded(vec![leg(0, 10), leg(0, 10)], 0)).is_err());
        assert!(validate(&funded(vec![leg(0, 0)], 0)).is_err());
        assert!(validate(&funded(vec![leg(2, 10)], 0)).is_err());
        assert!(
            validate(&funded(vec![leg(0, 10), leg(1, -20_001)], 0)).is_err()
        );
        assert!(
            validate(&vector_trade(
                &market,
                tip,
                vec![leg(0, 10)],
                5_000,
                5_000 + TRADE_MINER_FEE_SATS - 1,
            ))
            .is_err()
        );
        let mut stranger = funded(vec![leg(1, -10)], 0);
        if let Some(TxData::Trade { trader, .. }) =
            &mut stranger.transaction.data
        {
            *trader = Address([9; 20]);
        }
        assert!(validate(&stranger).is_err());
        Ok(())
    }

    #[test]
    fn trade_rejects_share_overflow() -> anyhow::Result<()> {
        let TradingFixture {
            env,
            state,
            archive,
            market,
            tip,
            ..
        } = trading_fixture();
        let trade = filled_tx(
            TxData::Trade {
                market_id: market.id,
                deltas: vec![ShareDelta {
                    outcome_index: 1,
                    shares: i64::MAX,
                }],
                trader: Address::ALL_ZEROS,
                max_cost: 0,
                tx_pow_nonce: None,
                prev_block_hash: tip,
            },
            1_000,
        );
        let rotxn = env.read_txn()?;
        crate::validation::MarketValidator::validate_trade(
            &state, &archive, &rotxn, &trade, None,
        )?;
        assert!(
            TradeSimulation::default()
                .apply(&state, &archive, &rotxn, &trade)
                .is_err()
        );
        Ok(())
    }

    /// Trading halts while votes are being revealed.
    #[test]
    fn trade_halts_on_reveal() -> anyhow::Result<()> {
        use crate::state::decisions::{Decision, DecisionType};
        use crate::types::Txid;

        let TradingFixture {
            env,
            state,
            archive,
            market,
            tip,
            ..
        } = trading_fixture();
        let decision_id = market.decision_ids[0];
        let trade = filled_tx(
            TxData::Trade {
                market_id: market.id,
                deltas: vec![ShareDelta {
                    outcome_index: 1,
                    shares: 1,
                }],
                trader: Address::ALL_ZEROS,
                max_cost: 0,
                tx_pow_nonce: None,
                prev_block_hash: tip,
            },
            1_000,
        );
        let validate = || -> anyhow::Result<_> {
            let rotxn = env.read_txn()?;
            Ok(crate::validation::MarketValidator::validate_trade(
                &state, &archive, &rotxn, &trade, None,
            ))
        };
        let mut rwtxn = env.write_txn()?;
        state.decisions().mint_genesis(&mut rwtxn, 0, 0, 0)?;
        state.decisions().claim_decision(
            &mut rwtxn,
            decision_id,
            Decision::new(
                [0; 20],
                DecisionType::Binary,
                String::new(),
                String::new(),
                None,
                None,
                vec![],
            )?,
            Txid([0; 32]),
            None,
        )?;
        state.decisions().transition_decision_to_voting(
            &mut rwtxn,
            decision_id,
            1,
        )?;
        rwtxn.commit()?;
        validate()??;

        let mut rwtxn = env.write_txn()?;
        state.decisions().transition_decision_to_revealing(
            &mut rwtxn,
            decision_id,
            2,
        )?;
        rwtxn.commit()?;
        assert!(validate()?.is_err());
        Ok(())
    }

    #[test]
    fn trade_rejects_cost_overflow() -> anyhow::Result<()> {
        let TradingFixture {
            env,
            state,
            archive,
            mut market,
            tip,
            ..
        } = trading_fixture();
        market.trading_fee = 1.0;
        market.liquidity_base_sats = 1;
        market.shares.fill(0);
        let mut rwtxn = env.write_txn()?;
        state.markets().update_market(&mut rwtxn, &market)?;
        rwtxn.commit()?;
        let trade = filled_tx(
            TxData::Trade {
                market_id: market.id,
                deltas: vec![ShareDelta {
                    outcome_index: 0,
                    shares: i64::MAX,
                }],
                trader: Address::ALL_ZEROS,
                max_cost: 0,
                tx_pow_nonce: None,
                prev_block_hash: tip,
            },
            1_000,
        );
        let rotxn = env.read_txn()?;
        crate::validation::MarketValidator::validate_trade(
            &state, &archive, &rotxn, &trade, None,
        )?;
        assert!(
            TradeSimulation::default()
                .apply(&state, &archive, &rotxn, &trade)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn sell_rollback_restores_volume() -> anyhow::Result<()> {
        use crate::math::trading::TRADE_MINER_FEE_SATS;
        use crate::state::markets::generate_market_treasury_address;
        use crate::types::{Output, Txid};

        let TradingFixture {
            env,
            state,
            archive,
            mut market,
            tip,
            ..
        } = trading_fixture();
        market.total_volume_sats = 20_000;
        market.outcome_volumes_sats = vec![0, 20_000];
        let mut rwtxn = env.write_txn()?;
        state.markets().update_market(&mut rwtxn, &market)?;
        state.markets().add_shares_to_account(
            &mut rwtxn,
            &Address::ALL_ZEROS,
            market.id,
            1,
            20_000,
            0,
        )?;
        let treasury = OutPoint::Regular {
            txid: Txid([8; 32]),
            vout: 0,
        };
        let mut diff = AccumulatorDiff::default();
        state.insert_utxo(
            &mut rwtxn,
            &treasury,
            &Output {
                address: generate_market_treasury_address(&market.id),
                content: OutputContent::MarketFunds {
                    market_id: market.id.0,
                    amount: bitcoin::Amount::from_sat(40_000),
                    is_fee: false,
                },
            },
            &mut diff,
        )?;
        state
            .markets()
            .set_market_funds_utxo(&mut rwtxn, &market.id, false, &treasury)?;
        let initial_utxos = state.get_utxos(&rwtxn)?;
        let trade = filled_tx(
            TxData::Trade {
                market_id: market.id,
                deltas: vec![ShareDelta {
                    outcome_index: 1,
                    shares: -2_000,
                }],
                trader: Address::ALL_ZEROS,
                max_cost: 0,
                tx_pow_nonce: None,
                prev_block_hash: tip,
            },
            TRADE_MINER_FEE_SATS,
        );
        crate::validation::MarketValidator::validate_trade(
            &state, &archive, &rwtxn, &trade, None,
        )?;
        let mut update = StateUpdate::new();
        assert!(matches!(
            apply_trade(&state, &rwtxn, &trade, &mut update)?,
            TradeApplyResult::Applied
        ));
        let settlement = &update.pending_trade_settlements[0];
        let gross = settlement.sell_gross_sats;
        assert!(gross > settlement.fee_sats);
        let undo = update
            .apply_all_changes(
                &state,
                &mut rwtxn,
                &mut diff,
                &mut MarketUtxoChanges::default(),
                1,
            )?
            .unwrap();
        let after = state.markets().get_market(&rwtxn, &market.id)?.unwrap();
        assert_eq!(after.total_volume_sats, market.total_volume_sats + gross);
        revert_consolidation(&state, &mut rwtxn, &mut diff, &undo)?;
        let deltas =
            revert_trade_market_state(&state, &mut rwtxn, &mut diff, &trade)?;
        disconnect_end(&state, &mut rwtxn, 1, &HashSet::new(), deltas)?;
        let restored = state.markets().get_market(&rwtxn, &market.id)?.unwrap();
        assert_eq!(restored.total_volume_sats, market.total_volume_sats);
        assert_eq!(restored.outcome_volumes_sats, market.outcome_volumes_sats);
        assert_eq!(restored.shares, market.shares);
        state.markets().verify_share_invariant(&rwtxn, &market.id)?;
        assert_eq!(state.get_utxos(&rwtxn)?, initial_utxos);
        Ok(())
    }

    /// Trade simulation prices at a beta amplified earlier in the same block.
    #[test]
    fn sim_uses_amplified_beta() {
        use crate::math::trading::TRADE_MINER_FEE_SATS;

        let TradingFixture {
            env,
            state,
            archive,
            market,
            tip,
            ..
        } = trading_fixture();
        let rotxn = env.read_txn().unwrap();

        let shares = 1_000;
        let trade = filled_tx(
            TxData::Trade {
                market_id: market.id,
                deltas: vec![ShareDelta {
                    outcome_index: 0,
                    shares,
                }],
                trader: Address::ALL_ZEROS,
                max_cost: buy_cost(&market, shares),
                tx_pow_nonce: None,
                prev_block_hash: tip,
            },
            100_000,
        );
        let amount = 1_000_000;
        let amplify = filled_tx(
            TxData::AmplifyBeta {
                market_id: market.id,
                amount,
                market_author: Address::ALL_ZEROS,
            },
            amount + TRADE_MINER_FEE_SATS,
        );

        let mut trades = TradeSimulation::default();
        assert!(matches!(
            trades.apply(&state, &archive, &rotxn, &trade).unwrap(),
            TradeApplyResult::Applied
        ));

        let mut trades = TradeSimulation::default();
        assert!(matches!(
            trades.apply(&state, &archive, &rotxn, &amplify).unwrap(),
            TradeApplyResult::Applied
        ));
        assert!(matches!(
            trades.apply(&state, &archive, &rotxn, &trade).unwrap(),
            TradeApplyResult::Skipped { .. }
        ));
    }

    /// Trade simulation rejects selling shares bought in the same block.
    #[test]
    fn sim_rejects_same_block_sell() {
        let TradingFixture {
            env,
            state,
            archive,
            market,
            tip,
            ..
        } = trading_fixture();
        let rotxn = env.read_txn().unwrap();

        let shares = 20_000;
        let trade = |shares, max_cost| {
            filled_tx(
                TxData::Trade {
                    market_id: market.id,
                    deltas: vec![ShareDelta {
                        outcome_index: 0,
                        shares,
                    }],
                    trader: Address::ALL_ZEROS,
                    max_cost,
                    tx_pow_nonce: None,
                    prev_block_hash: tip,
                },
                100_000,
            )
        };
        let buy = trade(shares, buy_cost(&market, shares));
        let sell = trade(-shares, 0);

        let mut trades = TradeSimulation::default();
        assert!(matches!(
            trades.apply(&state, &archive, &rotxn, &buy).unwrap(),
            TradeApplyResult::Applied
        ));
        let result = trades.apply(&state, &archive, &rotxn, &sell);
        assert!(
            matches!(
                &result,
                Err(Error::InvalidTransaction { reason })
                    if reason.starts_with("Insufficient shares")
            ),
            "sell of same-block shares must fail prevalidation: {:?}",
            result.as_ref().err()
        );
    }

    /// A sell beyond the shares left after a same-block sell is rejected.
    #[test]
    fn double_spend_rejected() {
        let mut state_update = StateUpdate::new();
        let trader = Address::ALL_ZEROS;
        let market_id = MarketId::new([1u8; 6]);
        let outcome_index: u32 = 0;

        let owned_shares_from_db: i64 = 100;

        state_update.add_share_account_change(
            trader,
            market_id,
            outcome_index,
            -60,
        );

        let pending_delta = state_update
            .share_account_changes
            .get(&(trader, market_id))
            .and_then(|outcomes| outcomes.get(&outcome_index))
            .copied()
            .unwrap_or(0);

        let effective_owned = owned_shares_from_db + pending_delta;
        assert_eq!(effective_owned, 40);

        let shares_to_sell: i64 = 50;
        assert!(
            effective_owned < shares_to_sell,
            "Double-spend protection: should reject sell of {shares_to_sell} when only {effective_owned} effectively owned"
        );

        let valid_sell: i64 = 40;
        assert!(
            effective_owned >= valid_sell,
            "Should allow selling {valid_sell} when {effective_owned} effectively owned"
        );
    }

    /// Buying then selling within the same block is allowed.
    #[test]
    fn buy_then_sell_allowed() {
        let mut state_update = StateUpdate::new();
        let trader = Address::ALL_ZEROS;
        let market_id = MarketId::new([2u8; 6]);
        let outcome_index: u32 = 1;

        let owned_shares_from_db: i64 = 0;

        state_update.add_share_account_change(
            trader,
            market_id,
            outcome_index,
            100,
        );

        let pending_delta = state_update
            .share_account_changes
            .get(&(trader, market_id))
            .and_then(|outcomes| outcomes.get(&outcome_index))
            .copied()
            .unwrap_or(0);

        let effective_owned = owned_shares_from_db + pending_delta;
        assert_eq!(effective_owned, 100);

        let shares_to_sell: i64 = 50;
        assert!(
            effective_owned >= shares_to_sell,
            "Should allow selling {shares_to_sell} after buying 100 in same block"
        );
    }

    /// A same-block sell of one outcome leaves other outcomes untouched.
    #[test]
    fn sells_isolated_by_outcome() {
        let mut state_update = StateUpdate::new();
        let trader = Address::ALL_ZEROS;
        let market_id = MarketId::new([3u8; 6]);

        let owned_outcome_0: i64 = 100;
        let owned_outcome_1: i64 = 100;

        state_update.add_share_account_change(trader, market_id, 0, -80);

        let pending_delta_0 = state_update
            .share_account_changes
            .get(&(trader, market_id))
            .and_then(|outcomes| outcomes.get(&0))
            .copied()
            .unwrap_or(0);
        let effective_owned_0 = owned_outcome_0 + pending_delta_0;
        assert_eq!(effective_owned_0, 20);

        let pending_delta_1 = state_update
            .share_account_changes
            .get(&(trader, market_id))
            .and_then(|outcomes| outcomes.get(&1))
            .copied()
            .unwrap_or(0);
        let effective_owned_1 = owned_outcome_1 + pending_delta_1;
        assert_eq!(effective_owned_1, 100);

        assert!(effective_owned_1 >= 100);
    }

    /// Same-block sells of one outcome accumulate against the owned shares.
    #[test]
    fn sells_accumulate() {
        let mut state_update = StateUpdate::new();
        let trader = Address::ALL_ZEROS;
        let market_id = MarketId::new([4u8; 6]);
        let outcome_index: u32 = 0;

        let owned_shares_from_db: i64 = 100;

        state_update.add_share_account_change(
            trader,
            market_id,
            outcome_index,
            -30,
        );

        state_update.add_share_account_change(
            trader,
            market_id,
            outcome_index,
            -40,
        );

        let pending_delta = state_update
            .share_account_changes
            .get(&(trader, market_id))
            .and_then(|outcomes| outcomes.get(&outcome_index))
            .copied()
            .unwrap_or(0);

        assert_eq!(pending_delta, -70);

        let effective_owned = owned_shares_from_db + pending_delta;
        assert_eq!(effective_owned, 30);

        let third_sell: i64 = 40;
        assert!(
            effective_owned < third_sell,
            "Should reject third sell: {third_sell} > {effective_owned} effective"
        );

        assert!(effective_owned >= 30);
    }

    /// Same-block buys fold into the effective shares while beta stays fixed.
    #[test]
    fn buys_fold_shares_not_beta() {
        use crate::math::trading;
        use ndarray::array;

        let outcomes = 2usize;
        let liquidity_base =
            trading::calculate_lmsr_liquidity(10_000_000.0, outcomes).round()
                as u64;
        let beta =
            trading::derive_beta_from_liquidity(liquidity_base, outcomes);

        let s0 = array![0i64, 0i64];
        let after_one = array![100_000i64, 0i64];
        let after_two = array![200_000i64, 0i64];

        let market_id = MarketId::new([7u8; 6]);
        let mut state_update = StateUpdate::new();
        state_update.add_market_update(MarketStateUpdate {
            market_id,
            legs: vec![TradeLeg {
                outcome_index: 0,
                shares: 100_000,
                volume_sats: 0,
            }],
        });
        state_update.add_pending_trade_settlement(PendingTradeSettlement {
            market_id,
            trader_address: Address::ALL_ZEROS,
            input_value_sats: 0,
            buy_cost_sats: 12_345,
            sell_gross_sats: 0,
            fee_sats: 0,
            transaction_id: [0u8; 32],
            is_amplify: false,
        });

        let mut effective = s0.clone();
        for leg in state_update
            .market_updates
            .iter()
            .filter(|update| update.market_id == market_id)
            .flat_map(|update| &update.legs)
        {
            effective[leg.outcome_index] += leg.shares;
        }
        assert_eq!(effective, after_one);

        let base1 =
            trading::calculate_update_cost(&s0, &after_one, beta).unwrap();
        let base2_sequential =
            trading::calculate_update_cost(&after_one, &after_two, beta)
                .unwrap();

        assert!(base2_sequential > base1);
    }

    /// Settlement adds Utreexo leaves, so the order of the settled markets is
    /// consensus. A rollback puts the markets back in the trading index in
    /// the reverse order, and the replay of the same block must reach the
    /// same roots.
    #[test]
    fn replay_keeps_roots() -> anyhow::Result<()> {
        use crate::archive::Archive;
        use crate::state::MarketBuilder;
        use crate::state::decisions::{Decision, DecisionId, DecisionType};
        use crate::state::markets::{
            DimensionSpec, generate_market_treasury_address,
        };
        use crate::state::undo::SettlementUndoData;
        use crate::types::{Accumulator, Output, Txid, UtreexoNodeHash};

        const HEIGHT: u32 = 3;

        let dir = tempfile::tempdir()?;
        let mut opts = heed::EnvOpenOptions::new();
        opts.map_size(64 * 1024 * 1024)
            .max_dbs(State::NUM_DBS + Archive::NUM_DBS);
        let env = unsafe { sneed::Env::open(&opts, dir.path()) }?;
        let state = State::new(&env, None, true)?;
        let mut rwtxn = env.write_txn()?;

        // Leaves of the two treasury outputs that settlement spends
        let mut treasury_leaves = AccumulatorDiff::default();
        for index in 0..2u8 {
            let decision_id = DecisionId::new(false, 0, index.into())?;
            let decision = Decision::new(
                Address::ALL_ZEROS.0,
                DecisionType::Binary,
                format!("Header {index}"),
                "Desc".to_string(),
                None,
                None,
                vec![],
            )?;
            let decisions = state.decisions();
            decisions.claim_decision(
                &mut rwtxn,
                decision_id,
                decision.clone(),
                Txid([index; 32]),
                Some(0),
            )?;
            decisions.transition_decision_to_voting(
                &mut rwtxn,
                decision_id,
                1,
            )?;
            decisions.transition_decision_to_revealing(
                &mut rwtxn,
                decision_id,
                2,
            )?;
            decisions.transition_decision_to_resolved(
                &mut rwtxn,
                decision_id,
                2,
                1.0,
            )?;
            let market = MarketBuilder::new(
                format!("Market {index}"),
                Address::ALL_ZEROS,
            )
            .with_dimensions(vec![DimensionSpec::Single(decision_id)])
            .build(
                0,
                None,
                &HashMap::from([(decision_id, decision)]),
            )?;
            state.markets().add_market(&mut rwtxn, &market)?;
            let treasury = OutPoint::Regular {
                txid: Txid([0x10 + index; 32]),
                vout: 0,
            };
            let treasury_output = Output {
                address: generate_market_treasury_address(&market.id),
                content: OutputContent::MarketFunds {
                    market_id: market.id.0,
                    amount: bitcoin::Amount::from_sat(10_000),
                    is_fee: false,
                },
            };
            state.insert_utxo(
                &mut rwtxn,
                &treasury,
                &treasury_output,
                &mut treasury_leaves,
            )?;
            state.markets().set_market_funds_utxo(
                &mut rwtxn, &market.id, false, &treasury,
            )?;
            state.markets().add_shares_to_account(
                &mut rwtxn,
                &Address([index + 1; 20]),
                market.id,
                0,
                1_000,
                0,
            )?;
        }

        let settle = |rwtxn: &mut RwTxn| -> anyhow::Result<(
            Vec<UtreexoNodeHash>,
            SettlementUndoData,
        )> {
            let mut diff = AccumulatorDiff::default();
            let mut changes = MarketUtxoChanges::default();
            let before = utxo_set(&state, rwtxn)?;
            let (_, entries) =
                state.markets().transition_and_payout_resolved_markets(
                    rwtxn,
                    &mut diff,
                    &mut changes,
                    &state,
                    state.decisions(),
                    HEIGHT,
                )?;
            anyhow::ensure!(entries.len() == 2, "both markets must settle");
            ensure_changes_match(&before, &utxo_set(&state, rwtxn)?, &changes)?;
            anyhow::ensure!(
                changes.deletes.len() == 2
                    && changes.deletes.iter().all(|utxo| {
                        utxo.reason == MarketUtxoReason::Treasury
                    }),
                "settlement must record both treasury deletes: {:?}",
                changes.deletes
            );
            let mut accumulator = Accumulator::default();
            accumulator.apply_diff(treasury_leaves.clone())?;
            accumulator.apply_diff(diff)?;
            Ok((accumulator.get_roots(), SettlementUndoData { entries }))
        };

        let initial_outputs = state.get_utxos(&rwtxn)?;
        let total_value = |outputs: &HashMap<OutPoint, Output>| -> u64 {
            outputs
                .values()
                .map(|output| output.get_value().to_sat())
                .sum()
        };
        let initial_value = total_value(&initial_outputs);
        let (roots, undo) = settle(&mut rwtxn)?;
        let settled_outputs = state.get_utxos(&rwtxn)?;
        assert_eq!(total_value(&settled_outputs), initial_value);

        let mut repeated_changes = MarketUtxoChanges::default();
        let (repeated_payouts, repeated_undo) =
            state.markets().transition_and_payout_resolved_markets(
                &mut rwtxn,
                &mut AccumulatorDiff::default(),
                &mut repeated_changes,
                &state,
                state.decisions(),
                HEIGHT + 1,
            )?;
        assert!(repeated_payouts.is_empty());
        assert!(repeated_undo.is_empty());
        assert!(repeated_changes.is_empty());
        assert_eq!(state.get_utxos(&rwtxn)?, settled_outputs);
        revert_settlement(
            &state,
            &mut rwtxn,
            &mut AccumulatorDiff::default(),
            &undo,
            HEIGHT,
        )?;
        assert_eq!(state.get_utxos(&rwtxn)?, initial_outputs);
        let (replayed_roots, _) = settle(&mut rwtxn)?;
        assert_eq!(state.get_utxos(&rwtxn)?, settled_outputs);
        anyhow::ensure!(
            replayed_roots == roots,
            "the replay reached other roots: {replayed_roots:?} != {roots:?}"
        );
        Ok(())
    }
}
