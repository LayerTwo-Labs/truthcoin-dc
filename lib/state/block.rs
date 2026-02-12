use std::collections::{HashMap, HashSet};

use sneed::{RoTxn, RwTxn};

use crate::{
    math::trading,
    state::{Error, State, UtxoManager, error, markets::MarketId},
    types::{
        Address, AmountOverflowError, Authorization, Body, FilledOutput,
        FilledOutputContent, FilledTransaction, GetAddress as _,
        GetBitcoinValue as _, Header, InPoint, OutPoint, OutputContent,
        SpentOutput, TxData, Verify as _,
    },
};

struct StateUpdate {
    market_updates: Vec<MarketStateUpdate>,
    market_creations: Vec<MarketCreation>,
    share_account_changes: HashMap<(Address, MarketId), HashMap<u32, i64>>,
    slot_changes: Vec<SlotStateChange>,
    vote_submissions: Vec<VoteSubmission>,
    voter_registrations: Vec<VoterRegistration>,
    reputation_updates: Vec<ReputationUpdate>,
    pending_sell_payouts: Vec<PendingSellPayout>,
    pending_buy_settlements: Vec<PendingBuySettlement>,
    pending_sell_input_changes: Vec<(Address, u64, [u8; 32])>,
}

struct MarketStateUpdate {
    market_id: MarketId,
    /// Share delta: (outcome_index, shares_to_add)
    share_delta: Option<(usize, i64)>,
    new_beta: Option<f64>,
    transaction_id: Option<[u8; 32]>,
    volume_sats: Option<u64>,
    /// Trading fee collected for the market author (in satoshis)
    fee_sats: Option<u64>,
}

struct SlotStateChange {
    slot_id: crate::state::slots::SlotId,
    new_decision: Option<crate::state::slots::Decision>,
    period_transition: Option<u32>,
}

struct MarketCreation {
    market: crate::state::Market,
}

struct VoteSubmission {
    vote: crate::state::voting::types::Vote,
}

struct VoterRegistration {
    initial_reputation: crate::state::voting::types::VoterReputation,
}

struct ReputationUpdate {
    updated_reputation: crate::state::voting::types::VoterReputation,
}

pub struct PendingSellPayout {
    pub market_id: MarketId,
    pub seller_address: Address,
    pub payout_sats: u64,
    pub fee_sats: u64,
    pub outcome_index: u32,
    pub transaction_id: [u8; 32],
}

pub struct PendingBuySettlement {
    pub market_id: MarketId,
    pub trader_address: Address,
    pub input_value_sats: u64,
    pub lmsr_cost_sats: u64,
    pub market_fee_sats: u64,
    pub transaction_id: [u8; 32],
}

enum TradeApplyResult {
    Applied,
    Skipped { reason: String },
}

impl StateUpdate {
    fn new() -> Self {
        Self {
            market_updates: Vec::new(),
            market_creations: Vec::new(),
            share_account_changes: HashMap::new(),
            slot_changes: Vec::new(),
            vote_submissions: Vec::new(),
            voter_registrations: Vec::new(),
            reputation_updates: Vec::new(),
            pending_sell_payouts: Vec::new(),
            pending_buy_settlements: Vec::new(),
            pending_sell_input_changes: Vec::new(),
        }
    }

    fn verify_internal_consistency(&self) -> Result<(), Error> {
        let mut created_market_ids = std::collections::HashSet::new();
        for creation in &self.market_creations {
            if !created_market_ids.insert(creation.market.id.clone()) {
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
            // Validate beta if provided (share deltas don't need LMSR validation here)
            if let Some(beta) = update.new_beta
                && beta <= 0.0
            {
                return Err(Error::InvalidTransaction {
                    reason: "Beta must be positive".to_string(),
                });
            }

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

            crate::validation::MarketValidator::validate_lmsr_parameters(
                creation.market.b(),
                creation.market.shares(),
            )?;
        }

        for slot_change in &self.slot_changes {
            if slot_change.new_decision.is_some()
                && state
                    .slots()
                    .get_slot(rotxn, slot_change.slot_id)?
                    .is_none()
            {
                return Err(Error::InvalidSlotId {
                    reason: format!(
                        "Slot {:?} does not exist",
                        slot_change.slot_id
                    ),
                });
            }
        }

        Ok(())
    }

    fn apply_all_changes(
        &self,
        state: &State,
        rwtxn: &mut RwTxn,
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

        let mut aggregated_deltas: super::type_aliases::AggregatedDeltas =
            std::collections::HashMap::new();

        for update in &self.market_updates {
            if let Some((outcome_index, delta)) = update.share_delta {
                aggregated_deltas
                    .entry(update.market_id.clone())
                    .or_default()
                    .push((
                        outcome_index,
                        delta,
                        update.volume_sats,
                        update.fee_sats,
                        update.transaction_id,
                    ));
            }
        }

        for (market_id, deltas) in aggregated_deltas {
            let mut market = state
                .markets()
                .get_market(rwtxn, &market_id)?
                .ok_or_else(|| Error::InvalidTransaction {
                    reason: format!("Market {market_id:?} not found"),
                })?;

            let mut new_shares = market.shares().clone();

            for (outcome_index, delta, volume_sats, _fee_sats, _txid) in &deltas
            {
                new_shares[*outcome_index] += *delta;
                if let Some(vol) = volume_sats {
                    market
                        .update_trading_volume(*outcome_index, *vol)
                        .map_err(|e| Error::InvalidTransaction {
                            reason: format!("Failed to update volume: {e:?}"),
                        })?;
                }
            }

            market
                .update_state(height as u64, None, None, Some(new_shares), None)
                .map_err(|e| Error::InvalidTransaction {
                    reason: format!("Failed to update market state: {e:?}"),
                })?;

            state.markets().update_market(rwtxn, &market)?;
            state.clear_mempool_shares(rwtxn, &market_id)?;
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
                            market_id.clone(),
                            outcome_index,
                            share_delta,
                            height as u64,
                        )?;
                    } else {
                        state.markets().remove_shares_from_account(
                            rwtxn,
                            address,
                            market_id,
                            outcome_index,
                            -share_delta,
                            height as u64,
                        )?;
                    }
                }
            }
        }

        for slot_change in &self.slot_changes {
            if let Some(ref _decision) = slot_change.new_decision
                && let Some(slot) =
                    state.slots().get_slot(rwtxn, slot_change.slot_id)?
                && slot.decision.is_some()
            {
                return Err(Error::InvalidSlotId {
                    reason: format!(
                        "Slot {:?} already has a decision",
                        slot_change.slot_id
                    ),
                });
            }

            if let Some(new_period) = slot_change.period_transition {
                let current_period = slot_change.slot_id.period_index();
                if new_period <= current_period {
                    return Err(Error::InvalidSlotId {
                        reason: format!(
                            "Invalid period transition from {current_period} to {new_period}"
                        ),
                    });
                }
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

        for registration in &self.voter_registrations {
            state.voting().databases().put_voter_reputation(
                rwtxn,
                &registration.initial_reputation,
            )?;
        }

        for update in &self.reputation_updates {
            state
                .voting()
                .databases()
                .put_voter_reputation(rwtxn, &update.updated_reputation)?;
        }

        let consolidation_undo = Self::consolidate_market_utxos(
            state,
            rwtxn,
            height,
            &self.pending_sell_payouts,
            &self.pending_buy_settlements,
            &self.pending_sell_input_changes,
        )?;

        Ok(consolidation_undo)
    }

    pub fn generate_sell_payout_outpoint(
        market_id: &MarketId,
        seller_address: &Address,
        transaction_id: [u8; 32],
    ) -> OutPoint {
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(b"SELL_PAYOUT");
        hasher.update(&market_id.0);
        hasher.update(&seller_address.0);
        hasher.update(&transaction_id);

        let hash = hasher.finalize();
        let merkle_root = crate::types::MerkleRoot::from(*hash.as_bytes());

        OutPoint::Coinbase {
            merkle_root,
            vout: 0,
        }
    }

    pub fn generate_buy_change_outpoint(
        market_id: &MarketId,
        trader_address: &Address,
        transaction_id: [u8; 32],
    ) -> OutPoint {
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(b"BUY_CHANGE");
        hasher.update(&market_id.0);
        hasher.update(&trader_address.0);
        hasher.update(&transaction_id);

        let hash = hasher.finalize();
        let merkle_root = crate::types::MerkleRoot::from(*hash.as_bytes());

        OutPoint::Coinbase {
            merkle_root,
            vout: 0,
        }
    }

    pub fn generate_sell_input_change_outpoint(
        trader_address: &Address,
        transaction_id: [u8; 32],
    ) -> OutPoint {
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(b"SELL_INPUT_CHANGE");
        hasher.update(&trader_address.0);
        hasher.update(&transaction_id);

        let hash = hasher.finalize();
        let merkle_root = crate::types::MerkleRoot::from(*hash.as_bytes());

        OutPoint::Coinbase {
            merkle_root,
            vout: 0,
        }
    }

    fn consolidate_market_utxos(
        state: &State,
        rwtxn: &mut RwTxn,
        height: u32,
        pending_sell_payouts: &[PendingSellPayout],
        pending_buy_settlements: &[PendingBuySettlement],
        pending_sell_input_changes: &[(Address, u64, [u8; 32])],
    ) -> Result<Option<crate::state::undo::ConsolidationUndoData>, Error> {
        use crate::math::trading::TRADE_MINER_FEE_SATS;
        use crate::state::markets::{
            generate_market_author_fee_address,
            generate_market_treasury_address,
        };
        use crate::types::{BitcoinOutputContent, FilledOutput, OutPoint};
        use std::collections::HashSet;

        let sell_payout_markets: HashSet<[u8; 6]> =
            pending_sell_payouts.iter().map(|p| p.market_id.0).collect();
        let buy_settlement_markets: HashSet<[u8; 6]> = pending_buy_settlements
            .iter()
            .map(|s| s.market_id.0)
            .collect();

        let mut markets_to_consolidate: HashSet<[u8; 6]> = HashSet::new();
        markets_to_consolidate.extend(sell_payout_markets);
        markets_to_consolidate.extend(buy_settlement_markets);

        let pending_utxo_markets =
            state.markets().get_markets_with_pending_utxos(rwtxn)?;
        markets_to_consolidate.extend(pending_utxo_markets);

        if markets_to_consolidate.is_empty()
            && pending_sell_input_changes.is_empty()
        {
            return Ok(None);
        }

        let mut undo_entries = Vec::new();

        for market_id_bytes in &markets_to_consolidate {
            let market_id = MarketId::new(*market_id_bytes);

            let mut treasury_total = 0u64;
            let mut treasury_utxos_to_consume = Vec::new();
            let mut fee_total = 0u64;
            let mut fee_utxos_to_consume = Vec::new();

            // Capture pre-consolidation state for undo
            let old_treasury_pointer = state
                .markets()
                .get_market_funds_utxo(rwtxn, &market_id, false)?;
            let old_fee_pointer = state
                .markets()
                .get_market_funds_utxo(rwtxn, &market_id, true)?;
            let old_pending_utxos = state
                .markets()
                .get_pending_market_funds_utxos(rwtxn, &market_id)?;

            // Capture old treasury UTXOs with their filled outputs
            let mut old_treasury_utxos_with_outputs = Vec::new();
            let mut old_fee_utxos_with_outputs = Vec::new();

            if let Some(existing_outpoint) = old_treasury_pointer
                && let Some(utxo) =
                    state.utxos.try_get(rwtxn, &existing_outpoint)?
            {
                treasury_total += utxo.get_bitcoin_value().to_sat();
                old_treasury_utxos_with_outputs.push((existing_outpoint, utxo));
                treasury_utxos_to_consume.push(existing_outpoint);
            }

            if let Some(existing_outpoint) = old_fee_pointer
                && let Some(utxo) =
                    state.utxos.try_get(rwtxn, &existing_outpoint)?
            {
                fee_total += utxo.get_bitcoin_value().to_sat();
                old_fee_utxos_with_outputs.push((existing_outpoint, utxo));
                fee_utxos_to_consume.push(existing_outpoint);
            }

            for (outpoint, is_fee) in &old_pending_utxos {
                if let Some(utxo) = state.utxos.try_get(rwtxn, outpoint)? {
                    if *is_fee {
                        fee_total += utxo.get_bitcoin_value().to_sat();
                        old_fee_utxos_with_outputs.push((*outpoint, utxo));
                        fee_utxos_to_consume.push(*outpoint);
                    } else {
                        treasury_total += utxo.get_bitcoin_value().to_sat();
                        old_treasury_utxos_with_outputs.push((*outpoint, utxo));
                        treasury_utxos_to_consume.push(*outpoint);
                    }
                }
            }

            let market_buy_settlements: Vec<&PendingBuySettlement> =
                pending_buy_settlements
                    .iter()
                    .filter(|s| s.market_id == market_id)
                    .collect();

            for settlement in &market_buy_settlements {
                treasury_total += settlement.lmsr_cost_sats;
                fee_total += settlement.market_fee_sats;
            }

            let market_sell_payouts: Vec<&PendingSellPayout> =
                pending_sell_payouts
                    .iter()
                    .filter(|p| p.market_id == market_id)
                    .collect();
            let total_sell_payouts: u64 =
                market_sell_payouts.iter().map(|p| p.payout_sats).sum();
            let total_sell_fees: u64 =
                market_sell_payouts.iter().map(|p| p.fee_sats).sum();
            fee_total += total_sell_fees;

            if total_sell_payouts > treasury_total {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Treasury underflow: sell payouts {total_sell_payouts} exceed treasury {treasury_total} for market {market_id:?}",
                    ),
                });
            }

            let mut new_treasury_utxo = None;
            let mut new_fee_utxo = None;
            let mut sell_payout_utxos = Vec::new();
            let mut buy_change_utxos = Vec::new();

            let has_treasury_work = !treasury_utxos_to_consume.is_empty()
                || !market_sell_payouts.is_empty()
                || !market_buy_settlements.is_empty();

            if has_treasury_work
                && (treasury_total > 0 || !market_sell_payouts.is_empty())
            {
                for outpoint in &treasury_utxos_to_consume {
                    state.delete_utxo_with_address_index(rwtxn, outpoint)?;
                }
                state
                    .markets()
                    .clear_market_funds_utxo(rwtxn, &market_id, false)?;

                for payout in market_sell_payouts.iter() {
                    let payout_outpoint = Self::generate_sell_payout_outpoint(
                        &market_id,
                        &payout.seller_address,
                        payout.transaction_id,
                    );
                    let payout_output = FilledOutput {
                        address: payout.seller_address,
                        content: FilledOutputContent::Bitcoin(
                            BitcoinOutputContent(bitcoin::Amount::from_sat(
                                payout.payout_sats,
                            )),
                        ),
                        memo: vec![],
                    };
                    state.insert_utxo_with_address_index(
                        rwtxn,
                        &payout_outpoint,
                        &payout_output,
                    )?;
                    sell_payout_utxos.push(payout_outpoint);
                }

                for settlement in &market_buy_settlements {
                    let change = settlement
                        .input_value_sats
                        .saturating_sub(TRADE_MINER_FEE_SATS)
                        .saturating_sub(settlement.lmsr_cost_sats)
                        .saturating_sub(settlement.market_fee_sats);
                    if change > 0 {
                        let change_outpoint =
                            Self::generate_buy_change_outpoint(
                                &market_id,
                                &settlement.trader_address,
                                settlement.transaction_id,
                            );
                        let change_output = FilledOutput {
                            address: settlement.trader_address,
                            content: FilledOutputContent::Bitcoin(
                                BitcoinOutputContent(
                                    bitcoin::Amount::from_sat(change),
                                ),
                            ),
                            memo: vec![],
                        };
                        state.insert_utxo_with_address_index(
                            rwtxn,
                            &change_outpoint,
                            &change_output,
                        )?;
                        buy_change_utxos.push(change_outpoint);
                    }
                }

                let remaining_treasury =
                    treasury_total.saturating_sub(total_sell_payouts);

                if remaining_treasury > 0 {
                    let treasury_address =
                        generate_market_treasury_address(&market_id);
                    let new_outpoint = OutPoint::MarketFunds {
                        market_id: *market_id_bytes,
                        block_height: height,
                        is_fee: false,
                    };
                    let new_output = FilledOutput::new(
                        treasury_address,
                        FilledOutputContent::MarketFunds {
                            market_id: *market_id_bytes,
                            amount: BitcoinOutputContent(
                                bitcoin::Amount::from_sat(remaining_treasury),
                            ),
                            is_fee: false,
                        },
                    );
                    state.insert_utxo_with_address_index(
                        rwtxn,
                        &new_outpoint,
                        &new_output,
                    )?;
                    state.markets().set_market_funds_utxo(
                        rwtxn,
                        &market_id,
                        false,
                        &new_outpoint,
                    )?;
                    new_treasury_utxo = Some(new_outpoint);
                }
            }

            let has_fee_work = !fee_utxos_to_consume.is_empty()
                || total_sell_fees > 0
                || !market_buy_settlements.is_empty();

            if has_fee_work && fee_total > 0 {
                for outpoint in &fee_utxos_to_consume {
                    state.delete_utxo_with_address_index(rwtxn, outpoint)?;
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
                let new_output = FilledOutput::new(
                    fee_address,
                    FilledOutputContent::MarketFunds {
                        market_id: *market_id_bytes,
                        amount: BitcoinOutputContent(
                            bitcoin::Amount::from_sat(fee_total),
                        ),
                        is_fee: true,
                    },
                );
                state.insert_utxo_with_address_index(
                    rwtxn,
                    &new_outpoint,
                    &new_output,
                )?;
                state.markets().set_market_funds_utxo(
                    rwtxn,
                    &market_id,
                    true,
                    &new_outpoint,
                )?;
                new_fee_utxo = Some(new_outpoint);
            }

            state
                .markets()
                .clear_pending_market_funds_utxos(rwtxn, &market_id)?;

            undo_entries.push(crate::state::undo::ConsolidationUndoEntry {
                market_id,
                old_treasury_utxos: old_treasury_utxos_with_outputs,
                old_fee_utxos: old_fee_utxos_with_outputs,
                old_treasury_pointer,
                old_fee_pointer,
                old_pending_utxos,
                new_treasury_utxo,
                new_fee_utxo,
                sell_payout_utxos,
                buy_change_utxos,
            });
        }

        let mut sell_input_change_utxos = Vec::new();
        for (address, change_sats, tx_id) in pending_sell_input_changes {
            if *change_sats > 0 {
                let change_outpoint =
                    Self::generate_sell_input_change_outpoint(address, *tx_id);
                let change_output = FilledOutput {
                    address: *address,
                    content: FilledOutputContent::Bitcoin(
                        BitcoinOutputContent(bitcoin::Amount::from_sat(
                            *change_sats,
                        )),
                    ),
                    memo: vec![],
                };
                state.insert_utxo_with_address_index(
                    rwtxn,
                    &change_outpoint,
                    &change_output,
                )?;
                sell_input_change_utxos.push(change_outpoint);
            }
        }

        Ok(Some(crate::state::undo::ConsolidationUndoData {
            entries: undo_entries,
            sell_input_change_utxos,
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

    fn add_pending_sell_payout(&mut self, payout: PendingSellPayout) {
        self.pending_sell_payouts.push(payout);
    }

    fn add_pending_buy_settlement(&mut self, settlement: PendingBuySettlement) {
        self.pending_buy_settlements.push(settlement);
    }

    fn add_pending_sell_input_change(
        &mut self,
        address: Address,
        change_sats: u64,
        tx_id: [u8; 32],
    ) {
        self.pending_sell_input_changes
            .push((address, change_sats, tx_id));
    }
}
use crate::types::MerkleRoot;

pub fn validate(
    state: &State,
    rotxn: &RoTxn,
    header: &Header,
    body: &Body,
) -> Result<(bitcoin::Amount, Vec<FilledTransaction>, MerkleRoot), Error> {
    let tip_hash = state.try_get_tip(rotxn)?;
    if header.prev_side_hash != tip_hash {
        let err = error::InvalidHeader::PrevSideHash {
            expected: tip_hash,
            received: header.prev_side_hash,
        };
        return Err(Error::InvalidHeader(err));
    };
    let merkle_root = body.compute_merkle_root();
    if merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: header.merkle_root,
            computed: merkle_root,
        };
        return Err(err);
    }

    let future_height =
        state.try_get_height(rotxn)?.map_or(0, |height| height + 1);

    let mut coinbase_value = bitcoin::Amount::ZERO;
    for output in &body.coinbase {
        coinbase_value = coinbase_value
            .checked_add(output.get_bitcoin_value())
            .ok_or(AmountOverflowError)?;
    }
    let mut spent_utxos = HashSet::new();
    let filled_txs: Vec<_> = body
        .transactions
        .iter()
        .map(|t| state.fill_transaction(rotxn, t))
        .collect::<Result<_, _>>()?;

    for filled_tx in filled_txs.iter() {
        for input in &filled_tx.transaction.inputs {
            if spent_utxos.contains(input) {
                return Err(Error::UtxoDoubleSpent);
            }
            spent_utxos.insert(*input);
        }
        let _fee = state.validate_filled_transaction(
            rotxn,
            filled_tx,
            Some(future_height),
        )?;
    }
    let spent_utxos = filled_txs.iter().flat_map(|t| t.spent_utxos.iter());
    for (authorization, spent_utxo) in
        body.authorizations.iter().zip(spent_utxos)
    {
        if authorization.get_address() != spent_utxo.address {
            return Err(Error::WrongPubKeyForAddress);
        }
    }
    if Authorization::verify_body(body).is_err() {
        return Err(Error::AuthorizationError);
    }
    Ok((coinbase_value, filled_txs, merkle_root))
}

pub fn connect(
    state: &State,
    rwtxn: &mut RwTxn,
    header: &Header,
    body: &Body,
    mainchain_timestamp: u64,
    filled_txs: Vec<FilledTransaction>,
    pre_computed_merkle_root: MerkleRoot,
) -> Result<(), Error> {
    let height = state.try_get_height(rwtxn)?.map_or(0, |height| height + 1);
    let tip_hash = state.try_get_tip(rwtxn)?;
    if tip_hash != header.prev_side_hash {
        let err = error::InvalidHeader::PrevSideHash {
            expected: tip_hash,
            received: header.prev_side_hash,
        };
        return Err(Error::InvalidHeader(err));
    }
    if pre_computed_merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: pre_computed_merkle_root,
            computed: header.merkle_root,
        };
        return Err(err);
    }

    if height == 0 {
        state
            .genesis_timestamp
            .put(rwtxn, &(), &mainchain_timestamp)?;
    }
    let genesis_ts = state.try_get_genesis_timestamp(rwtxn)?.unwrap_or(0);

    if height == 0 {
        state.slots().mint_genesis(
            rwtxn,
            mainchain_timestamp,
            height,
            genesis_ts,
        )?;
    } else {
        state.slots().mint_up_to(
            rwtxn,
            mainchain_timestamp,
            height,
            genesis_ts,
        )?;
    }

    let current_period =
        crate::state::voting::period_calculator::get_current_period(
            mainchain_timestamp,
            Some(height),
            genesis_ts,
            state.slots().get_config(),
        )?;

    // Transition Claimed → Voting (all transitions before any resolution)
    let claimed_needing_voting = state
        .slots()
        .get_claimed_slots_needing_voting(rwtxn, current_period)?;
    for slot_id in claimed_needing_voting {
        state.slots().transition_slot_to_voting(
            rwtxn,
            slot_id,
            height as u64,
            mainchain_timestamp,
        )?;
    }

    // Resolve Voting → Resolved (grouped by period, ascending order)
    let voting_ready = state
        .slots()
        .get_voting_slots_needing_resolution(rwtxn, current_period)?;
    let mut periods_to_resolve: std::collections::BTreeMap<
        u32,
        Vec<crate::state::slots::SlotId>,
    > = std::collections::BTreeMap::new();
    for slot_id in voting_ready {
        periods_to_resolve
            .entry(slot_id.voting_period())
            .or_default()
            .push(slot_id);
    }

    let mut consensus_undo_entries = Vec::new();
    for (vp_num, decision_slots) in &periods_to_resolve {
        let (start, end) =
            crate::state::voting::period_calculator::calculate_period_boundaries(
                *vp_num,
                state.slots().get_config(),
                genesis_ts,
            );
        let period = crate::state::voting::types::VotingPeriod {
            id: crate::state::voting::types::VotingPeriodId::new(*vp_num),
            start_timestamp: start,
            end_timestamp: end,
            status: crate::state::voting::types::VotingPeriodStatus::Closed,
            decision_slots: decision_slots.clone(),
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
            height as u64,
            state.slots(),
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
        let (payout_results, ossification_undo_entries) =
            state.markets().transition_and_payout_resolved_markets(
                rwtxn,
                state,
                state.slots(),
                height,
            )?;

        if !payout_results.is_empty() {
            for (market_id, summary) in &payout_results {
                tracing::info!(
                    "Protocol: Market {} auto-ossified with {} sats treasury + {} sats fees distributed to {} shareholders",
                    market_id,
                    summary.treasury_distributed,
                    summary.author_fees_distributed,
                    summary.shareholder_count
                );
            }
        }

        if !ossification_undo_entries.is_empty() {
            let undo_data = crate::state::undo::OssificationUndoData {
                entries: ossification_undo_entries,
            };
            state.ossification_undo.put(rwtxn, &height, &undo_data)?;
        }
    }

    for (vout, output) in body.coinbase.iter().enumerate() {
        let outpoint = OutPoint::Coinbase {
            merkle_root: header.merkle_root,
            vout: vout as u32,
        };
        let filled_content = match output.content.clone() {
            OutputContent::Bitcoin(value) => {
                FilledOutputContent::Bitcoin(value)
            }
            OutputContent::Withdrawal(withdrawal) => {
                FilledOutputContent::BitcoinWithdrawal(withdrawal)
            }
            OutputContent::Votecoin(amount) => {
                if height == 0 {
                    FilledOutputContent::Votecoin(amount)
                } else {
                    return Err(Error::BadCoinbaseOutputContent);
                }
            }
            OutputContent::MarketFunds { .. } => {
                return Err(Error::BadCoinbaseOutputContent);
            }
        };
        let filled_output = FilledOutput {
            address: output.address,
            content: filled_content,
            memo: output.memo.clone(),
        };
        state.insert_utxo_with_address_index(
            rwtxn,
            &outpoint,
            &filled_output,
        )?;
    }
    let mut state_update = StateUpdate::new();
    let mut skipped_tx_indices: HashSet<usize> = HashSet::new();

    for (idx, filled_tx) in filled_txs.iter().enumerate() {
        match &filled_tx.transaction.data {
            Some(TxData::Trade { .. }) => {
                match apply_trade(
                    state,
                    rwtxn,
                    filled_tx,
                    &mut state_update,
                    height,
                )? {
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
            Some(TxData::ClaimDecisionSlot { .. }) => {
                apply_slot_claim(
                    state,
                    rwtxn,
                    filled_tx,
                    &mut state_update,
                    height,
                    mainchain_timestamp,
                )?;
            }
            Some(TxData::SubmitVote { .. }) => {
                apply_submit_vote(
                    state,
                    rwtxn,
                    filled_tx,
                    &mut state_update,
                    height,
                )?;
            }
            Some(TxData::SubmitVoteBatch { .. }) => {
                apply_submit_vote_batch(
                    state,
                    rwtxn,
                    filled_tx,
                    &mut state_update,
                    height,
                )?;
            }
            Some(TxData::ClaimCategorySlots { .. }) => {
                apply_claim_category_slots(
                    state,
                    rwtxn,
                    filled_tx,
                    mainchain_timestamp,
                    height,
                )?;
            }
            Some(TxData::RegisterVoter { .. }) => {}
            None => {}
        }
    }

    state_update.validate_all_changes(state, rwtxn)?;

    {
        use crate::math::trading::TRADE_MINER_FEE_SATS;

        let mut actual_total_fees = bitcoin::Amount::ZERO;
        for (idx, filled_tx) in filled_txs.iter().enumerate() {
            if skipped_tx_indices.contains(&idx) {
                continue;
            }
            let tx_fee = if filled_tx.is_trade() {
                bitcoin::Amount::from_sat(TRADE_MINER_FEE_SATS)
            } else {
                filled_tx.bitcoin_fee()?.ok_or(Error::NotEnoughValueIn)?
            };
            actual_total_fees = actual_total_fees
                .checked_add(tx_fee)
                .ok_or(AmountOverflowError)?;
        }

        let mut coinbase_value = bitcoin::Amount::ZERO;
        for output in &body.coinbase {
            coinbase_value = coinbase_value
                .checked_add(output.get_bitcoin_value())
                .ok_or(AmountOverflowError)?;
        }

        if coinbase_value > actual_total_fees {
            return Err(Error::NotEnoughFees);
        }
    }

    for (idx, filled_tx) in filled_txs.iter().enumerate() {
        if skipped_tx_indices.contains(&idx) {
            continue;
        }
        apply_utxo_changes(state, rwtxn, filled_tx)?;
    }

    if let Some(consolidation_undo) =
        state_update.apply_all_changes(state, rwtxn, height)?
    {
        state
            .consolidation_undo
            .put(rwtxn, &height, &consolidation_undo)?;
    }

    let block_hash = header.hash();
    state.tip.put(rwtxn, &(), &block_hash)?;
    state.height.put(rwtxn, &(), &height)?;
    state
        .mainchain_timestamp
        .put(rwtxn, &(), &mainchain_timestamp)?;

    Ok(())
}

pub fn disconnect_tip(
    state: &State,
    rwtxn: &mut RwTxn,
    header: &Header,
    body: &Body,
) -> Result<(), Error> {
    // 1. Verify tip hash matches
    let tip_hash = state.tip.try_get(rwtxn, &())?.ok_or(Error::NoTip)?;
    if tip_hash != header.hash() {
        let err = error::InvalidHeader::BlockHash {
            expected: tip_hash,
            computed: header.hash(),
        };
        return Err(Error::InvalidHeader(err));
    }
    let merkle_root = body.compute_merkle_root();
    if merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: header.merkle_root,
            computed: merkle_root,
        };
        return Err(err);
    }
    let height = state
        .try_get_height(rwtxn)?
        .expect("Height should not be None");

    // 2. Revert UTXO consolidation (C3)
    // Consolidation is applied AFTER transactions during connect, so revert BEFORE transactions
    if let Some(consolidation_undo) =
        state.consolidation_undo.try_get(rwtxn, &height)?
    {
        revert_consolidation(state, rwtxn, &consolidation_undo)?;
        state.consolidation_undo.delete(rwtxn, &height)?;
    }

    // 3. Revert transaction-level UTXOs and tx-specific state (existing)
    body.transactions.iter().rev().try_for_each(|tx| {
        let txid = tx.txid();
        let filled_tx = state.fill_transaction_from_stxos(rwtxn, tx.clone())?;
        match &tx.data {
            None => (),
            Some(TxData::ClaimDecisionSlot { .. }) => {
                let () = revert_claim_decision_slot(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::CreateMarket { .. }) => {
                let () = revert_create_market(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::Trade { .. }) => {
                let () = revert_trade(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::SubmitVote { .. }) => {
                let () = revert_submit_vote(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::SubmitVoteBatch { .. }) => {
                let () = revert_submit_vote_batch(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::ClaimCategorySlots { .. }) => {
                let () = revert_claim_category_slots(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::RegisterVoter { .. }) => {}
        }

        tx.outputs.iter().enumerate().rev().try_for_each(
            |(vout, _output)| {
                let outpoint = OutPoint::Regular {
                    txid,
                    vout: vout as u32,
                };
                if state.delete_utxo_with_address_index(rwtxn, &outpoint)? {
                    Ok(())
                } else {
                    Err(Error::NoUtxo { outpoint })
                }
            },
        )?;
        tx.inputs.iter().rev().try_for_each(|outpoint| {
            if let Some(spent_output) = state.stxos.try_get(rwtxn, outpoint)? {
                state.stxos.delete(rwtxn, outpoint)?;
                state.insert_utxo_with_address_index(
                    rwtxn,
                    outpoint,
                    &spent_output.output,
                )?;
                Ok(())
            } else {
                Err(Error::NoStxo {
                    outpoint: *outpoint,
                })
            }
        })
    })?;

    // 4. Revert coinbase UTXOs (existing)
    body.coinbase.iter().enumerate().rev().try_for_each(
        |(vout, _output)| {
            let outpoint = OutPoint::Coinbase {
                merkle_root: header.merkle_root,
                vout: vout as u32,
            };
            if state.delete_utxo_with_address_index(rwtxn, &outpoint)? {
                Ok(())
            } else {
                Err(Error::NoUtxo { outpoint })
            }
        },
    )?;

    // 5. Revert market ossification/payouts (C1)
    if let Some(ossification_undo) =
        state.ossification_undo.try_get(rwtxn, &height)?
    {
        revert_ossification(state, rwtxn, &ossification_undo, height as u64)?;
        state.ossification_undo.delete(rwtxn, &height)?;
    }

    // 6. Revert consensus voting state (C2)
    if let Some(consensus_undo) =
        state.consensus_undo.try_get(rwtxn, &height)?
    {
        revert_consensus(state, rwtxn, &consensus_undo)?;
        state.consensus_undo.delete(rwtxn, &height)?;
    }

    // 7. Rollback slot states (existing)
    if height > 0 {
        state
            .slots()
            .rollback_slot_states_to_height(rwtxn, (height - 1) as u64)?;

        tracing::info!(
            "Rolled back slot states to height {} during reorg",
            height - 1
        );
    }

    // 8. Update tip/height to previous (existing)
    match (header.prev_side_hash, height) {
        (None, 0) => {
            state.tip.delete(rwtxn, &())?;
            state.height.delete(rwtxn, &())?;
        }
        (None, _) | (_, 0) => return Err(Error::NoTip),
        (Some(prev_side_hash), height) => {
            state.tip.put(rwtxn, &(), &prev_side_hash)?;
            state.height.put(rwtxn, &(), &(height - 1))?;
        }
    }
    Ok(())
}

/// Revert UTXO consolidation (C3)
fn revert_consolidation(
    state: &State,
    rwtxn: &mut RwTxn,
    undo: &crate::state::undo::ConsolidationUndoData,
) -> Result<(), Error> {
    // Process entries in reverse order
    for entry in undo.entries.iter().rev() {
        // Delete new UTXOs that were created during consolidation
        if let Some(ref outpoint) = entry.new_treasury_utxo
            && let Err(e) =
                state.delete_utxo_with_address_index(rwtxn, outpoint)
        {
            tracing::trace!("UTXO cleanup during revert: {e:?}");
        }
        if let Some(ref outpoint) = entry.new_fee_utxo
            && let Err(e) =
                state.delete_utxo_with_address_index(rwtxn, outpoint)
        {
            tracing::trace!("UTXO cleanup during revert: {e:?}");
        }
        for outpoint in &entry.sell_payout_utxos {
            if let Err(e) =
                state.delete_utxo_with_address_index(rwtxn, outpoint)
            {
                tracing::trace!("UTXO cleanup during revert: {e:?}");
            }
        }
        for outpoint in &entry.buy_change_utxos {
            if let Err(e) =
                state.delete_utxo_with_address_index(rwtxn, outpoint)
            {
                tracing::trace!("UTXO cleanup during revert: {e:?}");
            }
        }

        // Restore old treasury UTXOs
        for (outpoint, filled_output) in &entry.old_treasury_utxos {
            state.insert_utxo_with_address_index(
                rwtxn,
                outpoint,
                filled_output,
            )?;
        }
        // Restore old fee UTXOs
        for (outpoint, filled_output) in &entry.old_fee_utxos {
            state.insert_utxo_with_address_index(
                rwtxn,
                outpoint,
                filled_output,
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

        // Restore pending market funds UTXOs
        state.markets().restore_pending_market_funds_utxos(
            rwtxn,
            &entry.market_id,
            &entry.old_pending_utxos,
        )?;
    }

    // Delete sell input change UTXOs
    for outpoint in &undo.sell_input_change_utxos {
        if let Err(e) = state.delete_utxo_with_address_index(rwtxn, outpoint) {
            tracing::trace!("UTXO cleanup during revert: {e:?}");
        }
    }

    tracing::info!(
        "Reverted UTXO consolidation for {} markets",
        undo.entries.len()
    );
    Ok(())
}

/// Revert market ossification and payouts (C1)
fn revert_ossification(
    state: &State,
    rwtxn: &mut RwTxn,
    undo: &crate::state::undo::OssificationUndoData,
    block_height: u64,
) -> Result<(), Error> {
    for entry in undo.entries.iter().rev() {
        // Revert automatic share payouts (deletes payout UTXOs, restores shares)
        state.markets().revert_automatic_share_payouts(
            state,
            rwtxn,
            &entry.payout_summary,
            &entry.pre_ossification_market,
            block_height,
        )?;

        // Restore treasury UTXO
        if let Some((ref outpoint, ref filled_output)) = entry.treasury_utxo {
            state.insert_utxo_with_address_index(
                rwtxn,
                outpoint,
                filled_output,
            )?;
            state.markets().set_market_funds_utxo(
                rwtxn,
                &entry.pre_ossification_market.id,
                false,
                outpoint,
            )?;
        }

        // Restore fee UTXO
        if let Some((ref outpoint, ref filled_output)) = entry.fee_utxo {
            state.insert_utxo_with_address_index(
                rwtxn,
                outpoint,
                filled_output,
            )?;
            state.markets().set_market_funds_utxo(
                rwtxn,
                &entry.pre_ossification_market.id,
                true,
                outpoint,
            )?;
        }

        // Restore market to pre-ossification state (Trading, no final prices)
        state
            .markets()
            .update_market(rwtxn, &entry.pre_ossification_market)?;

        tracing::info!(
            "Reverted ossification for market {}",
            entry.pre_ossification_market.id
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
        for slot_id in &entry.decision_outcome_slot_ids {
            state
                .voting()
                .databases()
                .delete_decision_outcome(rwtxn, *slot_id)?;
        }

        // Delete period stats if they didn't exist before
        if !entry.had_period_stats {
            state
                .voting()
                .databases()
                .delete_period_stats(rwtxn, entry.period_id)?;
        }

        // Revert voter reputations to previous values
        for (address, prev_reputation) in &entry.previous_voter_reputations {
            if let Some(prev) = prev_reputation {
                state
                    .voting()
                    .databases()
                    .put_voter_reputation(rwtxn, prev)?;
            } else {
                state
                    .voting()
                    .databases()
                    .delete_voter_reputation(rwtxn, *address)?;
            }
        }

        // Revert votecoin redistribution: delete created UTXOs, restore consumed ones
        for outpoint in &entry.created_votecoin_utxos {
            if let Err(e) = state.delete_utxo_supply_neutral(rwtxn, outpoint) {
                tracing::trace!("Votecoin UTXO cleanup during revert: {e:?}");
            }
        }
        for (outpoint, filled_output) in &entry.consumed_votecoin_utxos {
            state.insert_utxo_supply_neutral(rwtxn, outpoint, filled_output)?;
        }

        // Revert pending redistribution record
        if entry.had_pending_redistribution {
            if let Some(ref prev) = entry.previous_pending_redistribution {
                state
                    .voting()
                    .databases()
                    .put_pending_redistribution(rwtxn, prev)?;
            } else {
                state
                    .voting()
                    .databases()
                    .delete_pending_redistribution(rwtxn, entry.period_id)?;
            }
        }

        tracing::info!("Reverted consensus for period {}", entry.period_id.0);
    }
    Ok(())
}

fn apply_claim_decision_slot(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    mainchain_timestamp: u64,
    block_height: u32,
) -> Result<(), Error> {
    use crate::state::slots::{Decision, SlotId};

    let claim = filled_tx.claim_decision_slot().ok_or_else(|| {
        Error::InvalidTransaction {
            reason: "Not a decision slot claim transaction".to_string(),
        }
    })?;

    let slot_id = SlotId::from_bytes(claim.slot_id_bytes)?;

    let market_maker_address_bytes = filled_tx
        .spent_utxos
        .first()
        .ok_or_else(|| Error::InvalidTransaction {
            reason: "No spent UTXOs found".to_string(),
        })?
        .address
        .0;

    let decision = Decision::new(
        market_maker_address_bytes,
        claim.slot_id_bytes,
        claim.is_standard,
        claim.is_scaled,
        claim.question.clone(),
        claim.min,
        claim.max,
        claim.option_0_label.clone(),
        claim.option_1_label.clone(),
    )?;

    let claiming_txid = filled_tx.transaction.txid();

    let genesis_ts = state.try_get_genesis_timestamp(rwtxn)?.unwrap_or(0);

    state.slots().claim_slot(
        rwtxn,
        slot_id,
        decision,
        claiming_txid,
        mainchain_timestamp,
        Some(block_height),
        genesis_ts,
    )?;

    let slot_period = slot_id.period_index();
    let voting_period = slot_id.voting_period();

    tracing::debug!(
        "Claimed slot {} from period {} (votes in period {})",
        hex::encode(slot_id.as_bytes()),
        slot_period,
        voting_period
    );

    Ok(())
}

fn revert_claim_decision_slot(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    use crate::state::slots::SlotId;

    let claim = filled_tx.claim_decision_slot().ok_or_else(|| {
        Error::InvalidTransaction {
            reason: "Not a decision slot claim transaction".to_string(),
        }
    })?;

    let slot_id = SlotId::from_bytes(claim.slot_id_bytes)?;

    state.slots().revert_claim_slot(rwtxn, slot_id)?;

    Ok(())
}

/// Apply a category slots claim transaction.
/// Claims multiple slots atomically with the same txid as the category identifier.
fn apply_claim_category_slots(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    mainchain_timestamp: u64,
    block_height: u32,
) -> Result<(), Error> {
    use crate::state::slots::{Decision, SlotId};

    let category_claim = filled_tx.claim_category_slots().ok_or_else(|| {
        Error::InvalidTransaction {
            reason: "Not a category slots claim transaction".to_string(),
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
    let genesis_ts = state.try_get_genesis_timestamp(rwtxn)?.unwrap_or(0);

    for (slot_id_bytes, question) in &category_claim.slots {
        let slot_id = SlotId::from_bytes(*slot_id_bytes)?;

        let decision = Decision::new(
            market_maker_address_bytes,
            *slot_id_bytes,
            category_claim.is_standard,
            false, // is_scaled = false for category slots
            question.clone(),
            None, // min = None for binary
            None, // max = None for binary
            None, // option_0_label - default for category slots
            None, // option_1_label - default for category slots
        )?;

        state.slots().claim_slot(
            rwtxn,
            slot_id,
            decision,
            claiming_txid,
            mainchain_timestamp,
            Some(block_height),
            genesis_ts,
        )?;

        tracing::debug!(
            "Claimed category slot {} with category txid {}",
            hex::encode(slot_id.as_bytes()),
            hex::encode(claiming_txid.0)
        );
    }

    tracing::info!(
        "Claimed {} slots as category with txid {}",
        category_claim.slots.len(),
        hex::encode(claiming_txid.0)
    );

    Ok(())
}

/// Revert a category slots claim transaction.
fn revert_claim_category_slots(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    use crate::state::slots::SlotId;

    let category_claim = filled_tx.claim_category_slots().ok_or_else(|| {
        Error::InvalidTransaction {
            reason: "Not a category slots claim transaction".to_string(),
        }
    })?;

    for (slot_id_bytes, _question) in &category_claim.slots {
        let slot_id = SlotId::from_bytes(*slot_id_bytes)?;
        state.slots().revert_claim_slot(rwtxn, slot_id)?;
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
    tags: &Option<Vec<String>>,
    b: f64,
    trading_fee: Option<f64>,
) -> crate::state::MarketBuilder {
    if !description.is_empty() {
        builder = builder.with_description(description.to_string());
    }

    if let Some(tags) = tags.as_ref() {
        builder = builder.with_tags(tags.clone());
    }

    builder = builder.with_beta(b);

    if let Some(fee) = trading_fee {
        builder = builder.with_fee(fee);
    }

    builder
}

fn revert_create_market(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    use crate::state::{MarketBuilder, markets::DimensionSpec};
    use std::collections::HashMap;

    let market_data =
        filled_tx
            .create_market()
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "Not a market creation transaction".to_string(),
            })?;

    let creator_address = extract_creator_address(filled_tx)?;
    let dimension_specs = market_data.dimension_specs.clone();

    let mut decisions = HashMap::new();

    for spec in &dimension_specs {
        match spec {
            DimensionSpec::Single(slot_id) => {
                if let Some(slot) = state.slots.get_slot(rwtxn, *slot_id)?
                    && let Some(decision) = slot.decision
                {
                    decisions.insert(*slot_id, decision);
                }
            }
            DimensionSpec::Categorical(slot_ids) => {
                for slot_id in slot_ids {
                    if let Some(slot) = state.slots.get_slot(rwtxn, *slot_id)?
                        && let Some(decision) = slot.decision
                    {
                        decisions.insert(*slot_id, decision);
                    }
                }
            }
        }
    }

    let mut builder =
        MarketBuilder::new(market_data.title.clone(), creator_address);
    builder = configure_market_builder(
        builder,
        &market_data.description,
        &market_data.tags,
        market_data.b,
        market_data.trading_fee,
    );

    let builder = builder
        .with_dimensions(dimension_specs)
        .with_residual_names(market_data.residual_names.clone());

    // Reconstruct the market to get its ID (height doesn't affect ID)
    let market = builder.build(0, None, &decisions).map_err(|e| {
        Error::InvalidTransaction {
            reason: format!("Market reconstruction failed: {e}"),
        }
    })?;

    let market_id = market.id.clone();

    if let Some(outpoint) = state
        .markets()
        .get_market_funds_utxo(rwtxn, &market_id, false)?
    {
        state.delete_utxo_with_address_index(rwtxn, &outpoint)?;
    }

    state.markets().delete_market(rwtxn, &market_id)?;

    Ok(())
}

fn revert_trade(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    let trade = filled_tx.trade().ok_or_else(|| Error::InvalidTransaction {
        reason: "Not a trade transaction".to_string(),
    })?;

    let is_buy = trade.is_buy();
    let height = state.try_get_height(rwtxn)?.unwrap_or(0);

    // Revert trading volume and market shares
    let mut market = state
        .markets()
        .get_market(rwtxn, &trade.market_id)?
        .ok_or_else(|| Error::InvalidTransaction {
            reason: "Market not found during trade revert".to_string(),
        })?;

    let shares_delta = trade.shares_abs() as i64;
    let outcome = trade.outcome_index as usize;

    if is_buy {
        // During connect: new_shares[outcome] = old_shares[outcome] + delta
        // So old_shares = current_shares - delta
        let mut pre_trade_shares = market.shares().clone();
        pre_trade_shares[outcome] -= shares_delta;

        if let Ok(base_cost) = trading::calculate_update_cost(
            &pre_trade_shares,
            market.shares(),
            market.b(),
        ) && let Ok(buy_cost) =
            trading::calculate_buy_cost(base_cost, market.trading_fee())
            && let Err(e) =
                market.revert_trading_volume(outcome, buy_cost.total_cost_sats)
        {
            tracing::warn!("Volume revert failed during trade revert: {e:?}");
        }

        market
            .update_shares(pre_trade_shares, height as u64)
            .map_err(|e| Error::InvalidTransaction {
                reason: format!("Failed to revert market shares: {e:?}"),
            })?;
    } else {
        // During connect: new_shares[outcome] = old_shares[outcome] - delta
        // So old_shares = current_shares + delta
        let mut pre_trade_shares = market.shares().clone();
        pre_trade_shares[outcome] += shares_delta;

        if let Ok(base_cost) = trading::calculate_update_cost(
            market.shares(),
            &pre_trade_shares,
            market.b(),
        ) && let Ok(sell_proceeds) =
            trading::calculate_sell_proceeds(base_cost, market.trading_fee())
            && let Err(e) = market.revert_trading_volume(
                outcome,
                sell_proceeds.gross_proceeds_sats,
            )
        {
            tracing::warn!("Volume revert failed during trade revert: {e:?}");
        }

        market
            .update_shares(pre_trade_shares, height as u64)
            .map_err(|e| Error::InvalidTransaction {
                reason: format!("Failed to revert market shares: {e:?}"),
            })?;
    }

    state.markets().update_market(rwtxn, &market)?;

    if is_buy {
        state.markets().revert_share_trade(
            rwtxn,
            &trade.trader,
            trade.market_id.clone(),
            trade.outcome_index,
            trade.shares_abs() as i64,
            height as u64,
        )?;

        let change_outpoint = StateUpdate::generate_buy_change_outpoint(
            &trade.market_id,
            &trade.trader,
            filled_tx.txid().0,
        );
        if let Err(e) =
            state.delete_utxo_with_address_index(rwtxn, &change_outpoint)
        {
            tracing::trace!(
                "UTXO not found during trade revert (expected if 0 change): {e:?}"
            );
        }
    } else {
        state.markets().add_shares_to_account(
            rwtxn,
            &trade.trader,
            trade.market_id.clone(),
            trade.outcome_index,
            trade.shares_abs() as i64,
            height as u64,
        )?;

        let payout_outpoint = StateUpdate::generate_sell_payout_outpoint(
            &trade.market_id,
            &trade.trader,
            filled_tx.txid().0,
        );
        if let Err(e) =
            state.delete_utxo_with_address_index(rwtxn, &payout_outpoint)
        {
            tracing::trace!(
                "UTXO not found during trade revert (expected if 0 change): {e:?}"
            );
        }

        let change_outpoint = StateUpdate::generate_sell_input_change_outpoint(
            &trade.trader,
            filled_tx.txid().0,
        );
        if let Err(e) =
            state.delete_utxo_with_address_index(rwtxn, &change_outpoint)
        {
            tracing::trace!(
                "UTXO not found during trade revert (expected if 0 change): {e:?}"
            );
        }
    }

    Ok(())
}

fn apply_utxo_changes(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    let txid = filled_tx.txid();

    for (vin, input) in filled_tx.inputs().iter().enumerate() {
        let spent_output = state
            .utxos
            .try_get(rwtxn, input)?
            .ok_or(Error::NoUtxo { outpoint: *input })?;

        let spent_output = SpentOutput {
            output: spent_output,
            inpoint: InPoint::Regular {
                txid,
                vin: vin as u32,
            },
        };
        state.delete_utxo_with_address_index(rwtxn, input)?;
        state.stxos.put(rwtxn, input, &spent_output)?;
    }

    let Some(filled_outputs) = filled_tx.filled_outputs() else {
        let err = error::FillTxOutputContents(Box::new(filled_tx.clone()));
        return Err(err.into());
    };

    for (vout, filled_output) in filled_outputs.iter().enumerate() {
        let outpoint = OutPoint::Regular {
            txid,
            vout: vout as u32,
        };

        state.insert_utxo_with_address_index(
            rwtxn,
            &outpoint,
            filled_output,
        )?;
    }

    Ok(())
}

/// Returns `TradeApplyResult::Skipped` for slippage failures (soft-fail).
fn apply_trade(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    state_update: &mut StateUpdate,
    _height: u32,
) -> Result<TradeApplyResult, Error> {
    use crate::math::trading::TRADE_MINER_FEE_SATS;

    let trade = filled_tx.trade().ok_or_else(|| Error::InvalidTransaction {
        reason: "Not a trade transaction".to_string(),
    })?;

    let is_buy = trade.is_buy();
    let shares_abs = trade.shares_abs();
    let outcome_index = trade.outcome_index as usize;

    let market = state
        .markets()
        .get_market(rwtxn, &trade.market_id)?
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

    let mut new_shares = market.shares().clone();
    new_shares[outcome_index] += trade.shares;

    if new_shares[outcome_index] < 0 {
        return Err(Error::InvalidTransaction {
            reason: format!(
                "Trade would result in negative market shares: {} for outcome {}",
                new_shares[outcome_index], outcome_index
            ),
        });
    }

    let input_value_sats = filled_tx
        .spent_bitcoin_value()
        .map_err(|_| Error::InvalidTransaction {
            reason: "Failed to compute input value".to_string(),
        })?
        .to_sat();

    let (volume_sats, fee_sats) = if is_buy {
        // Buy: cost = LMSR(current -> new)
        let base_cost = trading::calculate_update_cost(
            market.shares(),
            &new_shares,
            market.b(),
        )
        .map_err(|e| Error::InvalidTransaction {
            reason: format!("Failed to calculate trade cost: {e:?}"),
        })?;

        let buy_cost =
            trading::calculate_buy_cost(base_cost, market.trading_fee())
                .map_err(|e| Error::InvalidTransaction {
                    reason: format!("Buy cost calculation failed: {e}"),
                })?;

        let total_trade_cost = buy_cost.total_cost_sats;

        if total_trade_cost + TRADE_MINER_FEE_SATS > trade.limit_sats {
            return Ok(TradeApplyResult::Skipped {
                reason: format!(
                    "Buy cost {} sats + miner fee {} sats exceeds max cost {} sats",
                    total_trade_cost, TRADE_MINER_FEE_SATS, trade.limit_sats
                ),
            });
        }

        state_update.add_pending_buy_settlement(PendingBuySettlement {
            market_id: trade.market_id.clone(),
            trader_address: trade.trader,
            input_value_sats,
            lmsr_cost_sats: buy_cost.base_cost_sats,
            market_fee_sats: buy_cost.trading_fee_sats,
            transaction_id: filled_tx.transaction.txid().0,
        });

        (buy_cost.total_cost_sats, buy_cost.trading_fee_sats)
    } else {
        let seller_account = state
            .markets()
            .get_user_share_account(rwtxn, &trade.trader)?;

        let owned_shares = seller_account
            .as_ref()
            .and_then(|account| {
                account
                    .positions
                    .get(&(trade.market_id.clone(), trade.outcome_index))
                    .copied()
            })
            .unwrap_or(0);

        // Check pending share changes from earlier transactions in this block
        let pending_delta = state_update
            .share_account_changes
            .get(&(trade.trader, trade.market_id.clone()))
            .and_then(|outcomes| outcomes.get(&trade.outcome_index))
            .copied()
            .unwrap_or(0);

        let effective_owned = owned_shares + pending_delta;

        if effective_owned < shares_abs as i64 {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Insufficient shares: trying to sell {} but only own {} (effective: {}) for outcome {}",
                    shares_abs,
                    owned_shares,
                    effective_owned,
                    trade.outcome_index
                ),
            });
        }

        // Sell: proceeds = LMSR(new -> current) since shares decreased
        let proceeds = trading::calculate_update_cost(
            &new_shares,
            market.shares(),
            market.b(),
        )
        .map_err(|e| Error::InvalidTransaction {
            reason: format!("Failed to calculate sell proceeds: {e:?}"),
        })?;

        let sell_proceeds =
            trading::calculate_sell_proceeds(proceeds, market.trading_fee())
                .map_err(|e| Error::InvalidTransaction {
                    reason: format!("Sell proceeds calculation failed: {e}"),
                })?;

        let net_proceeds_sats = sell_proceeds.net_proceeds_sats;
        let fee_sats = sell_proceeds.trading_fee_sats;

        if net_proceeds_sats < trade.limit_sats {
            return Ok(TradeApplyResult::Skipped {
                reason: format!(
                    "Sell proceeds {} sats below minimum {} sats",
                    net_proceeds_sats, trade.limit_sats
                ),
            });
        }

        state_update.add_pending_sell_payout(PendingSellPayout {
            market_id: trade.market_id.clone(),
            seller_address: trade.trader,
            payout_sats: net_proceeds_sats,
            fee_sats,
            outcome_index: trade.outcome_index,
            transaction_id: filled_tx.transaction.txid().0,
        });

        // Record sell input change (input_value - miner_fee)
        let sell_input_change =
            input_value_sats.saturating_sub(TRADE_MINER_FEE_SATS);
        if sell_input_change > 0 {
            state_update.add_pending_sell_input_change(
                trade.trader,
                sell_input_change,
                filled_tx.transaction.txid().0,
            );
        }

        (net_proceeds_sats, fee_sats)
    };

    state_update.add_market_update(MarketStateUpdate {
        market_id: trade.market_id.clone(),
        share_delta: Some((outcome_index, trade.shares)),
        new_beta: None,
        transaction_id: Some(filled_tx.transaction.txid().0),
        volume_sats: Some(volume_sats),
        fee_sats: Some(fee_sats),
    });

    state_update.add_share_account_change(
        trade.trader,
        trade.market_id,
        trade.outcome_index,
        trade.shares,
    );

    Ok(TradeApplyResult::Applied)
}

fn apply_market_creation(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    state_update: &mut StateUpdate,
    height: u32,
) -> Result<(), Error> {
    use crate::state::{MarketBuilder, markets::DimensionSpec};
    use std::collections::HashMap;

    let market_data =
        filled_tx
            .create_market()
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "Not a market creation transaction".to_string(),
            })?;

    let creator_address = extract_creator_address(filled_tx)?;
    let dimension_specs = market_data.dimension_specs.clone();

    let mut decisions = HashMap::new();
    for spec in &dimension_specs {
        let slot_ids = match spec {
            DimensionSpec::Single(slot_id) => vec![*slot_id],
            DimensionSpec::Categorical(slot_ids) => slot_ids.clone(),
        };

        for slot_id in slot_ids {
            let slot =
                state.slots.get_slot(rwtxn, slot_id)?.ok_or_else(|| {
                    Error::InvalidSlotId {
                        reason: format!("Slot {slot_id:?} does not exist"),
                    }
                })?;

            let decision =
                slot.decision.ok_or_else(|| Error::InvalidSlotId {
                    reason: format!("Slot {slot_id:?} has no decision"),
                })?;

            decisions.insert(slot_id, decision);
        }
    }

    let mut builder =
        MarketBuilder::new(market_data.title.clone(), creator_address);
    builder = configure_market_builder(
        builder,
        &market_data.description,
        &market_data.tags,
        market_data.b,
        market_data.trading_fee,
    );

    let builder = builder
        .with_dimensions(dimension_specs.clone())
        .with_residual_names(market_data.residual_names.clone());

    let market =
        builder
            .build(height as u64, None, &decisions)
            .map_err(|e| Error::InvalidTransaction {
                reason: format!("Market creation failed: {e}"),
            })?;

    let market_id = market.id.clone();
    let market_id_bytes = *market_id.as_bytes();
    let txid = filled_tx.txid();

    for (vout, output) in filled_tx.outputs().iter().enumerate() {
        if let OutputContent::MarketFunds {
            market_id: output_market_id,
            amount,
            is_fee: false,
        } = &output.content
            && output_market_id == &market_id_bytes
        {
            let outpoint = OutPoint::Regular {
                txid,
                vout: vout as u32,
            };
            state
                .markets()
                .set_market_funds_utxo(rwtxn, &market_id, false, &outpoint)?;

            tracing::debug!(
                "Registered MarketFunds (treasury) UTXO for market {:?} with {} sats at {:?}",
                market_id,
                amount.0.to_sat(),
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
fn apply_slot_claim(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    _state_update: &mut StateUpdate,
    height: u32,
    mainchain_timestamp: u64,
) -> Result<(), Error> {
    apply_claim_decision_slot(
        state,
        rwtxn,
        filled_tx,
        mainchain_timestamp,
        height,
    )
}

fn apply_submit_vote(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    state_update: &mut StateUpdate,
    height: u32,
) -> Result<(), Error> {
    use crate::state::{
        slots::SlotId,
        voting::types::{Vote, VotingPeriodId},
    };

    let vote_data =
        filled_tx
            .submit_vote()
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "Not a vote submission transaction".to_string(),
            })?;

    let voter_address = filled_tx
        .spent_utxos
        .first()
        .ok_or_else(|| Error::InvalidTransaction {
            reason: "Vote transaction must have inputs".to_string(),
        })?
        .address;

    let decision_id = SlotId::from_bytes(vote_data.slot_id_bytes)?;

    let slot_claim_period = decision_id.period_index();
    let voting_period = decision_id.voting_period();
    let period_id = VotingPeriodId::new(voting_period);

    if vote_data.voting_period != voting_period {
        return Err(Error::InvalidTransaction {
            reason: format!(
                "Vote period mismatch: slot {} was claimed in period {} and must be voted on in period {}, but transaction specifies period {}",
                hex::encode(vote_data.slot_id_bytes),
                slot_claim_period,
                voting_period,
                vote_data.voting_period
            ),
        });
    }

    let timestamp =
        state.try_get_mainchain_timestamp(rwtxn)?.ok_or_else(|| {
            Error::InvalidTransaction {
                reason: "No mainchain timestamp available".to_string(),
            }
        })?;

    let vote_value = crate::validation::VoteValidator::convert_vote_value(
        vote_data.vote_value,
    );

    let vote = Vote::new(
        voter_address,
        period_id,
        decision_id,
        vote_value,
        timestamp,
        height as u64,
        filled_tx.txid().0,
    );

    state_update.add_vote_submission(vote);

    Ok(())
}

fn revert_submit_vote(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    use crate::state::{slots::SlotId, voting::types::VotingPeriodId};

    let vote_data =
        filled_tx
            .submit_vote()
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "Not a vote submission transaction".to_string(),
            })?;

    let voter_address = filled_tx
        .spent_utxos
        .first()
        .ok_or_else(|| Error::InvalidTransaction {
            reason: "Vote transaction must have inputs".to_string(),
        })?
        .address;

    let decision_id = SlotId::from_bytes(vote_data.slot_id_bytes)?;

    let voting_period = decision_id.voting_period();
    let period_id = VotingPeriodId::new(voting_period);

    state.voting().databases().delete_vote(
        rwtxn,
        period_id,
        voter_address,
        decision_id,
    )?;

    Ok(())
}

fn apply_submit_vote_batch(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    state_update: &mut StateUpdate,
    height: u32,
) -> Result<(), Error> {
    use crate::state::{
        slots::SlotId,
        voting::types::{Vote, VotingPeriodId},
    };

    let batch_data = filled_tx.submit_vote_batch().ok_or_else(|| {
        Error::InvalidTransaction {
            reason: "Not a vote batch submission transaction".to_string(),
        }
    })?;

    let voter_address = filled_tx
        .spent_utxos
        .first()
        .ok_or_else(|| Error::InvalidTransaction {
            reason: "Vote batch transaction must have inputs".to_string(),
        })?
        .address;

    let timestamp =
        state.try_get_mainchain_timestamp(rwtxn)?.ok_or_else(|| {
            Error::InvalidTransaction {
                reason: "No mainchain timestamp available".to_string(),
            }
        })?;

    let mut expected_voting_period: Option<u32> = None;

    for vote_item in &batch_data.votes {
        let decision_id = SlotId::from_bytes(vote_item.slot_id_bytes)?;

        let voting_period = decision_id.voting_period();

        if let Some(expected) = expected_voting_period {
            if voting_period != expected {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Batch vote period mismatch: slot {} requires period {} but batch expects period {}",
                        hex::encode(vote_item.slot_id_bytes),
                        voting_period,
                        expected
                    ),
                });
            }
        } else {
            expected_voting_period = Some(voting_period);

            if batch_data.voting_period != voting_period {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Batch vote period mismatch: slots require period {} but transaction specifies period {}",
                        voting_period, batch_data.voting_period
                    ),
                });
            }
        }

        let period_id = VotingPeriodId::new(voting_period);

        let vote_value = crate::validation::VoteValidator::convert_vote_value(
            vote_item.vote_value,
        );

        let vote = Vote::new(
            voter_address,
            period_id,
            decision_id,
            vote_value,
            timestamp,
            height as u64,
            filled_tx.txid().0,
        );

        state_update.add_vote_submission(vote);
    }

    Ok(())
}

fn revert_submit_vote_batch(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    use crate::state::{slots::SlotId, voting::types::VotingPeriodId};

    let batch_data = filled_tx.submit_vote_batch().ok_or_else(|| {
        Error::InvalidTransaction {
            reason: "Not a vote batch submission transaction".to_string(),
        }
    })?;

    let voter_address = filled_tx
        .spent_utxos
        .first()
        .ok_or_else(|| Error::InvalidTransaction {
            reason: "Vote batch transaction must have inputs".to_string(),
        })?
        .address;

    for vote_item in &batch_data.votes {
        let decision_id = SlotId::from_bytes(vote_item.slot_id_bytes)?;

        let voting_period = decision_id.voting_period();
        let period_id = VotingPeriodId::new(voting_period);

        state.voting().databases().delete_vote(
            rwtxn,
            period_id,
            voter_address,
            decision_id,
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_spend_protection_same_block() {
        let mut state_update = StateUpdate::new();
        let trader = Address::ALL_ZEROS;
        let market_id = MarketId::new([1u8; 6]);
        let outcome_index: u32 = 0;

        let owned_shares_from_db: i64 = 100;

        state_update.add_share_account_change(
            trader,
            market_id.clone(),
            outcome_index,
            -60,
        );

        let pending_delta = state_update
            .share_account_changes
            .get(&(trader, market_id.clone()))
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

    #[test]
    fn test_buy_then_sell_same_block_allowed() {
        let mut state_update = StateUpdate::new();
        let trader = Address::ALL_ZEROS;
        let market_id = MarketId::new([2u8; 6]);
        let outcome_index: u32 = 1;

        let owned_shares_from_db: i64 = 0;

        state_update.add_share_account_change(
            trader,
            market_id.clone(),
            outcome_index,
            100,
        );

        let pending_delta = state_update
            .share_account_changes
            .get(&(trader, market_id.clone()))
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

    #[test]
    fn test_multiple_sells_different_outcomes_same_block() {
        let mut state_update = StateUpdate::new();
        let trader = Address::ALL_ZEROS;
        let market_id = MarketId::new([3u8; 6]);

        let owned_outcome_0: i64 = 100;
        let owned_outcome_1: i64 = 100;

        state_update.add_share_account_change(
            trader,
            market_id.clone(),
            0,
            -80,
        );

        let pending_delta_0 = state_update
            .share_account_changes
            .get(&(trader, market_id.clone()))
            .and_then(|outcomes| outcomes.get(&0))
            .copied()
            .unwrap_or(0);
        let effective_owned_0 = owned_outcome_0 + pending_delta_0;
        assert_eq!(effective_owned_0, 20);

        let pending_delta_1 = state_update
            .share_account_changes
            .get(&(trader, market_id.clone()))
            .and_then(|outcomes| outcomes.get(&1))
            .copied()
            .unwrap_or(0);
        let effective_owned_1 = owned_outcome_1 + pending_delta_1;
        assert_eq!(effective_owned_1, 100);

        assert!(effective_owned_1 >= 100);
    }

    #[test]
    fn test_cumulative_sells_same_outcome_same_block() {
        let mut state_update = StateUpdate::new();
        let trader = Address::ALL_ZEROS;
        let market_id = MarketId::new([4u8; 6]);
        let outcome_index: u32 = 0;

        let owned_shares_from_db: i64 = 100;

        state_update.add_share_account_change(
            trader,
            market_id.clone(),
            outcome_index,
            -30,
        );

        state_update.add_share_account_change(
            trader,
            market_id.clone(),
            outcome_index,
            -40,
        );

        let pending_delta = state_update
            .share_account_changes
            .get(&(trader, market_id.clone()))
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
}
