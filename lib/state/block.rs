use std::collections::{HashMap, HashSet};

use sneed::{RoTxn, RwTxn};

use crate::{
    math::trading,
    state::{Error, State, UtxoManager, error, markets::MarketId},
    types::{
        Address, Body, FilledOutput, FilledOutputContent, FilledTransaction,
        GetBitcoinValue as _, Header, InPoint, MerkleRoot, OutPoint,
        OutPointKey, OutputContent, SpentOutput, TxData,
    },
};

struct StateUpdate {
    market_updates: Vec<MarketStateUpdate>,
    market_creations: Vec<MarketCreation>,
    share_account_changes: HashMap<(Address, MarketId), HashMap<u32, i64>>,
    vote_submissions: Vec<VoteSubmission>,
    pending_sell_payouts: Vec<PendingSellPayout>,
    pending_buy_settlements: Vec<PendingBuySettlement>,
    pending_sell_input_changes: Vec<(Address, u64, [u8; 32])>,
}

struct MarketStateUpdate {
    market_id: MarketId,
    share_delta: Option<(usize, i64)>,
    new_beta: Option<f64>,
    transaction_id: Option<[u8; 32]>,
    volume_sats: Option<u64>,
    fee_sats: Option<u64>,
}

struct MarketCreation {
    market: crate::state::Market,
}

struct VoteSubmission {
    vote: crate::state::voting::types::Vote,
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
            vote_submissions: Vec::new(),
            pending_sell_payouts: Vec::new(),
            pending_buy_settlements: Vec::new(),
            pending_sell_input_changes: Vec::new(),
        }
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
                    .entry(update.market_id)
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
                .update_state(height, None, Some(new_shares), None)
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
        let merkle_root = MerkleRoot::from(*hash.as_bytes());

        OutPoint::Payout {
            hash: merkle_root,
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
        let merkle_root = MerkleRoot::from(*hash.as_bytes());

        OutPoint::Payout {
            hash: merkle_root,
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
        let merkle_root = MerkleRoot::from(*hash.as_bytes());

        OutPoint::Payout {
            hash: merkle_root,
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
                && let Some(utxo) = state.utxos.try_get(
                    rwtxn,
                    &OutPointKey::from_outpoint(&existing_outpoint),
                )?
            {
                treasury_total += utxo.get_bitcoin_value().to_sat();
                old_treasury_utxos_with_outputs.push((existing_outpoint, utxo));
                treasury_utxos_to_consume.push(existing_outpoint);
            }

            if let Some(existing_outpoint) = old_fee_pointer
                && let Some(utxo) = state.utxos.try_get(
                    rwtxn,
                    &OutPointKey::from_outpoint(&existing_outpoint),
                )?
            {
                fee_total += utxo.get_bitcoin_value().to_sat();
                old_fee_utxos_with_outputs.push((existing_outpoint, utxo));
                fee_utxos_to_consume.push(existing_outpoint);
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

            let total_sell_gross = total_sell_payouts + total_sell_fees;
            if total_sell_gross > treasury_total {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Treasury underflow: sell gross {total_sell_gross} \
                         (payouts {total_sell_payouts} + fees \
                         {total_sell_fees}) exceed treasury \
                         {treasury_total} for market {market_id:?}",
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
                    state.delete_utxo(rwtxn, outpoint)?;
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
                    state.insert_utxo(
                        rwtxn,
                        &payout_outpoint,
                        &payout_output,
                    )?;
                    sell_payout_utxos.push(payout_outpoint);
                }

                for settlement in &market_buy_settlements {
                    let change = settlement
                        .input_value_sats
                        .checked_sub(TRADE_MINER_FEE_SATS)
                        .and_then(|v| v.checked_sub(settlement.lmsr_cost_sats))
                        .and_then(|v| v.checked_sub(settlement.market_fee_sats))
                        .ok_or_else(|| Error::InvalidTransaction {
                            reason: format!(
                                "Buy change underflow: input {} < \
                                 fees {} + cost {} + market_fee {}",
                                settlement.input_value_sats,
                                TRADE_MINER_FEE_SATS,
                                settlement.lmsr_cost_sats,
                                settlement.market_fee_sats,
                            ),
                        })?;
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
                        state.insert_utxo(
                            rwtxn,
                            &change_outpoint,
                            &change_output,
                        )?;
                        buy_change_utxos.push(change_outpoint);
                    }
                }

                let remaining_treasury = treasury_total
                    .checked_sub(total_sell_payouts)
                    .and_then(|v| v.checked_sub(total_sell_fees))
                    .ok_or_else(|| Error::InvalidTransaction {
                        reason: format!(
                            "Treasury remainder underflow: \
                             treasury {treasury_total} < \
                             payouts {total_sell_payouts} + \
                             fees {total_sell_fees}"
                        ),
                    })?;

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
                    state.insert_utxo(rwtxn, &new_outpoint, &new_output)?;
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
                    state.delete_utxo(rwtxn, outpoint)?;
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
                state.insert_utxo(rwtxn, &new_outpoint, &new_output)?;
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
                state.insert_utxo(rwtxn, &change_outpoint, &change_output)?;
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

pub fn connect_prevalidated(
    state: &State,
    rwtxn: &mut RwTxn,
    header: &Header,
    body: &Body,
    mainchain_timestamp: u64,
    prevalidated: super::PrevalidatedBlock,
) -> Result<(), Error> {
    // Use precomputed values — validation already done in prevalidate
    let height = prevalidated.next_height;
    let filled_txs = prevalidated.filled_transactions;
    let validated_coinbase = prevalidated.coinbase_value;

    if height == 0 {
        state
            .genesis_timestamp
            .put(rwtxn, &(), &mainchain_timestamp)?;
        if let Some(first_coinbase) = body.coinbase.first() {
            state.reputation().set_reputation(
                rwtxn,
                &first_coinbase.address,
                1.0,
            )?;
            tracing::info!(
                "Genesis block: Initialized reputation for {}",
                first_coinbase.address.as_base58()
            );
        }
    }
    let genesis_ts = state.try_get_genesis_timestamp(rwtxn)?.unwrap_or(0);

    let prev_highest_minted = state
        .decisions()
        .get_highest_minted_period(rwtxn)?
        .unwrap_or(0);

    if height == 0 {
        state.decisions().mint_genesis(
            rwtxn,
            mainchain_timestamp,
            height,
            genesis_ts,
        )?;
    } else {
        state.decisions().mint_up_to(
            rwtxn,
            mainchain_timestamp,
            height,
            genesis_ts,
        )?;
    }

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

    // Transition Claimed → Voting (all transitions before any resolution)
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

    crate::validation::BlockValidator::validate_coinbase_outputs(
        &body.coinbase,
        height,
    )?;

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
            OutputContent::MarketFunds { .. } => {
                unreachable!(
                    "validated by BlockValidator::validate_coinbase_outputs"
                )
            }
        };
        let filled_output = FilledOutput {
            address: output.address,
            content: filled_content,
            memo: output.memo.clone(),
        };
        state.insert_utxo(rwtxn, &outpoint, &filled_output)?;
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
            Some(TxData::ClaimDecision { .. }) => {
                apply_decision_claim(state, rwtxn, filled_tx, height)?;
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
            Some(TxData::SubmitBallot { .. }) => {
                apply_submit_ballot(
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
                apply_amplify_beta(state, rwtxn, filled_tx, &mut state_update)?;
            }
            None => {}
        }
    }

    state_update.validate_all_changes(state, rwtxn)?;

    crate::validation::BlockValidator::validate_fees(
        validated_coinbase,
        &filled_txs,
        &skipped_tx_indices,
    )?;

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

    // Resolve Voting → Resolved (grouped by period, ascending order).
    // Runs after transaction application so votes submitted in this block
    // count toward the closing period's consensus.
    let voting_ready = state
        .decisions()
        .get_voting_decisions_needing_resolution(rwtxn, current_period)?;
    let mut periods_to_resolve: std::collections::BTreeMap<
        u32,
        Vec<crate::state::decisions::DecisionId>,
    > = std::collections::BTreeMap::new();
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
        let (payout_results, ossification_undo_entries) =
            state.markets().transition_and_payout_resolved_markets(
                rwtxn,
                state,
                state.decisions(),
                height,
            )?;

        if !payout_results.is_empty() {
            for (market_id, summary) in &payout_results {
                let refund_sats = summary
                    .creator_refund
                    .as_ref()
                    .map(|r| r.amount_sats)
                    .unwrap_or(0);
                tracing::info!(
                    "Protocol: Market {} auto-ossified with {} sats treasury + {} sats fees distributed to {} shareholders ({} sats refunded to creator)",
                    market_id,
                    summary.treasury_distributed,
                    summary.total_fees_distributed,
                    summary.shareholder_count,
                    refund_sats,
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
    let merkle_root =
        Body::compute_merkle_root(&body.coinbase, &body.transactions);
    if merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: header.merkle_root,
            computed: merkle_root,
        };
        return Err(err);
    }
    let height = state.try_get_height(rwtxn)?.ok_or(Error::NoTip)?;

    // 2. Revert market ossification/payouts (runs last in connect, so first here)
    if let Some(ossification_undo) =
        state.ossification_undo.try_get(rwtxn, &height)?
    {
        revert_ossification(state, rwtxn, &ossification_undo, height)?;
        state.ossification_undo.delete(rwtxn, &height)?;
    }

    // 3. Revert consensus voting state
    if let Some(consensus_undo) =
        state.consensus_undo.try_get(rwtxn, &height)?
    {
        revert_consensus(state, rwtxn, &consensus_undo)?;
        state.consensus_undo.delete(rwtxn, &height)?;
    }

    // 4. Revert UTXO consolidation (applied during apply_all_changes)
    if let Some(consolidation_undo) =
        state.consolidation_undo.try_get(rwtxn, &height)?
    {
        revert_consolidation(state, rwtxn, &consolidation_undo)?;
        state.consolidation_undo.delete(rwtxn, &height)?;
    }

    // 5. Revert transaction-level UTXOs and tx-specific state
    let mut trade_share_deltas: Vec<TradeShareDelta> = Vec::new();

    for tx in body.transactions.iter().rev() {
        let txid = tx.txid();
        let filled_tx = state.fill_transaction_from_stxos(rwtxn, tx.clone())?;
        match &tx.data {
            None => (),
            Some(TxData::ClaimDecision { .. }) => {
                let () = revert_decision_claim(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::CreateMarket { .. }) => {
                let () = revert_create_market(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::Trade { .. }) => {
                let delta =
                    revert_trade_market_state(state, rwtxn, &filled_tx)?;
                trade_share_deltas.push(delta);
            }
            Some(TxData::SubmitVote { .. }) => {
                let () = revert_submit_vote(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::SubmitBallot { .. }) => {
                let () = revert_submit_ballot(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::TransferReputation { .. }) => {}
            // AmplifyBeta only deposits bitcoin to the treasury; the
            // consolidation undo step already reverts the treasury UTXO,
            // so there is no transaction-level state to revert here.
            Some(TxData::AmplifyBeta { .. }) => {}
        }

        tx.outputs.iter().enumerate().rev().try_for_each(
            |(vout, _output)| {
                let outpoint = OutPoint::Regular {
                    txid,
                    vout: vout as u32,
                };
                if state.delete_utxo(rwtxn, &outpoint)? {
                    Ok(())
                } else {
                    Err(Error::NoUtxo { outpoint })
                }
            },
        )?;
        tx.inputs.iter().rev().try_for_each(|outpoint| {
            let outpoint_key = OutPointKey::from_outpoint(outpoint);
            if let Some(spent_output) =
                state.stxos.try_get(rwtxn, &outpoint_key)?
            {
                state.stxos.delete(rwtxn, &outpoint_key)?;
                state.insert_utxo(rwtxn, outpoint, &spent_output.output)?;
                Ok(())
            } else {
                Err(Error::NoStxo {
                    outpoint: *outpoint,
                })
            }
        })?;
    }

    // 3b. Apply batched share account changes from trade reverts
    if !trade_share_deltas.is_empty() {
        let mut batched: std::collections::HashMap<
            (Address, MarketId),
            std::collections::HashMap<u32, i64>,
        > = std::collections::HashMap::new();

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
    }

    // 3c. Revert reputation transfers
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

    // 6. Revert coinbase UTXOs
    body.coinbase.iter().enumerate().rev().try_for_each(
        |(vout, _output)| {
            let outpoint = OutPoint::Coinbase {
                merkle_root: header.merkle_root,
                vout: vout as u32,
            };
            if state.delete_utxo(rwtxn, &outpoint)? {
                Ok(())
            } else {
                Err(Error::NoUtxo { outpoint })
            }
        },
    )?;

    // 7. Rollback decision states (Claimed → Voting transitions)
    if height > 0 {
        state
            .decisions()
            .rollback_decision_states_to_height(rwtxn, height - 1)?;

        tracing::info!(
            "Rolled back decision states to height {} during reorg",
            height - 1
        );
    }

    // 7b. Revert minted periods
    if let Some(prev_highest) = state.minting_undo.try_get(rwtxn, &height)? {
        state
            .decisions()
            .delete_periods_above(rwtxn, prev_highest)?;
        state.minting_undo.delete(rwtxn, &height)?;
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

fn revert_delete_utxo(
    state: &State,
    rwtxn: &mut RwTxn,
    outpoint: &OutPoint,
) -> Result<(), Error> {
    let deleted = state.delete_utxo(rwtxn, outpoint)?;
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
    undo: &crate::state::undo::ConsolidationUndoData,
) -> Result<(), Error> {
    // Process entries in reverse order
    for entry in undo.entries.iter().rev() {
        // Delete new UTXOs that were created during consolidation
        if let Some(ref outpoint) = entry.new_treasury_utxo {
            revert_delete_utxo(state, rwtxn, outpoint)?;
        }
        if let Some(ref outpoint) = entry.new_fee_utxo {
            revert_delete_utxo(state, rwtxn, outpoint)?;
        }
        for outpoint in &entry.sell_payout_utxos {
            revert_delete_utxo(state, rwtxn, outpoint)?;
        }
        for outpoint in &entry.buy_change_utxos {
            revert_delete_utxo(state, rwtxn, outpoint)?;
        }

        // Restore old treasury UTXOs
        for (outpoint, filled_output) in &entry.old_treasury_utxos {
            state.insert_utxo(rwtxn, outpoint, filled_output)?;
        }
        // Restore old fee UTXOs
        for (outpoint, filled_output) in &entry.old_fee_utxos {
            state.insert_utxo(rwtxn, outpoint, filled_output)?;
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

    // Delete sell input change UTXOs
    for outpoint in &undo.sell_input_change_utxos {
        revert_delete_utxo(state, rwtxn, outpoint)?;
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
    block_height: u32,
) -> Result<(), Error> {
    for entry in undo.entries.iter().rev() {
        // Revert automatic share payouts (deletes payout UTXOs, restores shares)
        state.markets().revert_automatic_share_payouts(
            state,
            rwtxn,
            &entry.payout_summary,
            block_height,
        )?;

        // Restore treasury UTXO
        if let Some((ref outpoint, ref filled_output)) = entry.treasury_utxo {
            state.insert_utxo(rwtxn, outpoint, filled_output)?;
            state.markets().set_market_funds_utxo(
                rwtxn,
                &entry.pre_ossification_market.id,
                false,
                outpoint,
            )?;
        }

        // Restore fee UTXO
        if let Some((ref outpoint, ref filled_output)) = entry.fee_utxo {
            state.insert_utxo(rwtxn, outpoint, filled_output)?;
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
            .restore_market(rwtxn, &entry.pre_ossification_market)?;

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
            hex::encode(decision_id.as_bytes()),
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
/// Beta is `treasury / ln(num_outcomes)`, where treasury is the confirmed
/// treasury UTXO value plus any pending mempool amplify_beta deposits.
fn market_beta(
    state: &State,
    rotxn: &RoTxn,
    market_id: &MarketId,
    market: &crate::state::Market,
) -> Result<f64, Error> {
    let confirmed = state
        .markets()
        .get_market_funds_sats(rotxn, state, market_id, false)?;
    let pending = state.get_mempool_treasury_delta(rotxn, market_id)?;
    let total = confirmed.saturating_add(pending);
    Ok(trading::derive_beta_from_liquidity(
        total,
        market.shares().len(),
    ))
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
            DimensionSpec::Single(decision_id) => {
                if let Some(entry) =
                    state.decisions().get_decision_entry(rwtxn, *decision_id)?
                    && let Some(decision) = entry.decision
                {
                    decisions.insert(*decision_id, decision);
                }
            }
            DimensionSpec::Categorical(decision_id) => {
                if let Some(entry) =
                    state.decisions().get_decision_entry(rwtxn, *decision_id)?
                    && let Some(decision) = entry.decision
                {
                    decisions.insert(*decision_id, decision);
                }
            }
        }
    }

    let mut builder =
        MarketBuilder::new(market_data.title.clone(), creator_address);
    builder = configure_market_builder(
        builder,
        &market_data.description,
        market_data.trading_fee,
    );

    let computed_tags = compute_market_tags(&decisions);
    builder = builder.with_tags(computed_tags);

    let builder = builder.with_dimensions(dimension_specs);

    // Reconstruct the market to get its ID (height doesn't affect ID)
    let market = builder.build(0, None, &decisions).map_err(|e| {
        Error::InvalidTransaction {
            reason: format!("Market reconstruction failed: {e}"),
        }
    })?;

    let market_id = market.id;

    if let Some(outpoint) = state
        .markets()
        .get_market_funds_utxo(rwtxn, &market_id, false)?
    {
        state.delete_utxo(rwtxn, &outpoint)?;
    }

    state.markets().delete_market(rwtxn, &market_id)?;

    Ok(())
}

struct TradeShareDelta {
    address: Address,
    market_id: MarketId,
    outcome_index: u32,
    share_delta: i64,
}

fn revert_trade_market_state(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<TradeShareDelta, Error> {
    let trade = filled_tx.trade().ok_or_else(|| Error::InvalidTransaction {
        reason: "Not a trade transaction".to_string(),
    })?;

    let is_buy = trade.is_buy();
    let height = state.try_get_height(rwtxn)?.unwrap_or(0);

    let mut market = state
        .markets()
        .get_market(rwtxn, &trade.market_id)?
        .ok_or_else(|| Error::InvalidTransaction {
            reason: "Market not found during trade revert".to_string(),
        })?;

    let shares_delta = trade.shares_abs() as i64;
    let outcome = trade.outcome_index as usize;

    let beta = market_beta(state, rwtxn, &trade.market_id, &market)?;

    if is_buy {
        let mut pre_trade_shares = market.shares().clone();
        pre_trade_shares[outcome] -= shares_delta;

        let base_cost = trading::calculate_update_cost(
            &pre_trade_shares,
            market.shares(),
            beta,
        )
        .map_err(|e| Error::InvalidTransaction {
            reason: format!("LMSR calc failed during buy trade revert: {e:?}"),
        })?;
        let buy_cost =
            trading::calculate_buy_cost(base_cost, market.trading_fee())
                .map_err(|e| Error::InvalidTransaction {
                    reason: format!(
                        "Buy cost calc failed during trade revert: {e}"
                    ),
                })?;
        market
            .revert_trading_volume(outcome, buy_cost.total_cost_sats)
            .map_err(|e| Error::InvalidTransaction {
                reason: format!(
                    "Volume revert failed during buy trade revert: \
                     {e:?}"
                ),
            })?;

        market
            .update_shares(pre_trade_shares, height)
            .map_err(|e| Error::InvalidTransaction {
                reason: format!("Failed to revert market shares: {e:?}"),
            })?;
    } else {
        let mut pre_trade_shares = market.shares().clone();
        pre_trade_shares[outcome] += shares_delta;

        let base_cost = trading::calculate_update_cost(
            market.shares(),
            &pre_trade_shares,
            beta,
        )
        .map_err(|e| Error::InvalidTransaction {
            reason: format!("LMSR calc failed during sell trade revert: {e:?}"),
        })?;
        let sell_proceeds =
            trading::calculate_sell_proceeds(base_cost, market.trading_fee())
                .map_err(|e| Error::InvalidTransaction {
                reason: format!(
                    "Sell proceeds calc failed during trade \
                         revert: {e}"
                ),
            })?;
        market
            .revert_trading_volume(outcome, sell_proceeds.gross_proceeds_sats)
            .map_err(|e| Error::InvalidTransaction {
                reason: format!(
                    "Volume revert failed during sell trade revert: \
                     {e:?}"
                ),
            })?;

        market
            .update_shares(pre_trade_shares, height)
            .map_err(|e| Error::InvalidTransaction {
                reason: format!("Failed to revert market shares: {e:?}"),
            })?;
    }

    state.markets().update_market(rwtxn, &market)?;

    if is_buy {
        let change_outpoint = StateUpdate::generate_buy_change_outpoint(
            &trade.market_id,
            &trade.trader,
            filled_tx.txid().0,
        );
        if let Err(e) = state.delete_utxo(rwtxn, &change_outpoint) {
            tracing::trace!(
                "UTXO not found during trade revert (expected if 0 change): {e:?}"
            );
        }
    } else {
        let payout_outpoint = StateUpdate::generate_sell_payout_outpoint(
            &trade.market_id,
            &trade.trader,
            filled_tx.txid().0,
        );
        if let Err(e) = state.delete_utxo(rwtxn, &payout_outpoint) {
            tracing::trace!(
                "UTXO not found during trade revert (expected if 0 change): {e:?}"
            );
        }

        let change_outpoint = StateUpdate::generate_sell_input_change_outpoint(
            &trade.trader,
            filled_tx.txid().0,
        );
        if let Err(e) = state.delete_utxo(rwtxn, &change_outpoint) {
            tracing::trace!(
                "UTXO not found during trade revert (expected if 0 change): {e:?}"
            );
        }
    }

    let account_delta = if is_buy { -shares_delta } else { shares_delta };

    Ok(TradeShareDelta {
        address: trade.trader,
        market_id: trade.market_id,
        outcome_index: trade.outcome_index,
        share_delta: account_delta,
    })
}

fn apply_utxo_changes(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    let txid = filled_tx.txid();

    for (vin, input) in filled_tx.inputs().iter().enumerate() {
        let input_key = OutPointKey::from_outpoint(input);
        let spent_output = state
            .utxos
            .try_get(rwtxn, &input_key)?
            .ok_or(Error::NoUtxo { outpoint: *input })?;

        let spent_output = SpentOutput {
            output: spent_output,
            inpoint: InPoint::Regular {
                txid,
                vin: vin as u32,
            },
        };
        state.delete_utxo(rwtxn, input)?;
        state.stxos.put(rwtxn, &input_key, &spent_output)?;
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

        state.insert_utxo(rwtxn, &outpoint, filled_output)?;
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

    let beta = market_beta(state, rwtxn, &trade.market_id, &market)?;

    let (volume_sats, fee_sats) = if is_buy {
        // Buy: cost = LMSR(current -> new)
        let base_cost =
            trading::calculate_update_cost(market.shares(), &new_shares, beta)
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
            market_id: trade.market_id,
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
                    .get(&(trade.market_id, trade.outcome_index))
                    .copied()
            })
            .unwrap_or(0);

        // Check pending share changes from earlier transactions in this block
        let pending_delta = state_update
            .share_account_changes
            .get(&(trade.trader, trade.market_id))
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
        let proceeds =
            trading::calculate_update_cost(&new_shares, market.shares(), beta)
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
            market_id: trade.market_id,
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
        market_id: trade.market_id,
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
        let decision_ids = match spec {
            DimensionSpec::Single(decision_id) => vec![*decision_id],
            DimensionSpec::Categorical(decision_id) => vec![*decision_id],
        };

        for decision_id in decision_ids {
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
    }

    let mut builder =
        MarketBuilder::new(market_data.title.clone(), creator_address);
    builder = configure_market_builder(
        builder,
        &market_data.description,
        market_data.trading_fee,
    );

    let computed_tags = compute_market_tags(&decisions);
    builder = builder.with_tags(computed_tags);

    let builder = builder
        .with_dimensions(dimension_specs.clone())
        .with_tx_pow(
            market_data.tx_pow_hash_selector.unwrap_or(0),
            market_data.tx_pow_ordering.unwrap_or(0),
            market_data.tx_pow_difficulty.unwrap_or(0),
        );

    let market = builder.build(height, None, &decisions).map_err(|e| {
        tracing::warn!(
            "Market creation failed for '{}': {e}",
            market_data.title
        );
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

fn apply_amplify_beta(
    state: &State,
    rwtxn: &mut RwTxn,
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
        .spent_bitcoin_value()
        .map_err(|_| Error::InvalidTransaction {
            reason: "Failed to compute input value".to_string(),
        })?
        .to_sat();

    state_update.add_pending_buy_settlement(PendingBuySettlement {
        market_id: amplify.market_id,
        trader_address: amplify.market_author,
        input_value_sats,
        lmsr_cost_sats: amplify.amount,
        market_fee_sats: 0,
        transaction_id: filled_tx.transaction.txid().0,
    });

    let pending_delta =
        state.get_mempool_treasury_delta(rwtxn, &amplify.market_id)?;
    let remaining_delta = match pending_delta.checked_sub(amplify.amount) {
        Some(remaining) => remaining,
        None => {
            tracing::warn!(
                "apply_amplify_beta: mempool treasury delta {} for market \
                 {:?} is less than applied amount {}; resetting to 0",
                pending_delta,
                amplify.market_id,
                amplify.amount
            );
            0
        }
    };
    if remaining_delta == 0 {
        state.clear_mempool_treasury_delta(rwtxn, &amplify.market_id)?;
    } else {
        state.put_mempool_treasury_delta(
            rwtxn,
            &amplify.market_id,
            remaining_delta,
        )?;
    }

    Ok(())
}

fn apply_submit_vote(
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

    let vote_data =
        filled_tx
            .submit_vote()
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "Not a vote submission transaction".to_string(),
            })?;

    let voter_address = vote_data.voter;

    let decision_id = DecisionId::from_bytes(vote_data.decision_id_bytes)?;

    let decision_claim_period = decision_id.period_index();
    let voting_period = decision_id.voting_period();
    let period_id = VotingPeriodId::new(voting_period);

    if vote_data.voting_period != voting_period {
        return Err(Error::InvalidTransaction {
            reason: format!(
                "Vote period mismatch: decision {} was claimed in period {} and must be voted on in period {}, but transaction specifies period {}",
                hex::encode(vote_data.decision_id_bytes),
                decision_claim_period,
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

    let decision_ref = state
        .decisions()
        .get_decision_entry(rwtxn, decision_id)?
        .and_then(|e| e.decision);

    let vote_value =
        crate::validation::VoteValidator::convert_vote_value_with_decision(
            vote_data.vote_value,
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

    Ok(())
}

fn revert_submit_vote(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    use crate::state::{decisions::DecisionId, voting::types::VotingPeriodId};

    let vote_data =
        filled_tx
            .submit_vote()
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "Not a vote submission transaction".to_string(),
            })?;

    let voter_address = vote_data.voter;

    let decision_id = DecisionId::from_bytes(vote_data.decision_id_bytes)?;

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

fn apply_submit_ballot(
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
            .submit_ballot()
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "Not a ballot submission transaction".to_string(),
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
                        hex::encode(vote_item.decision_id_bytes),
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

    Ok(())
}

fn revert_submit_ballot(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    use crate::state::{decisions::DecisionId, voting::types::VotingPeriodId};

    let ballot_data =
        filled_tx
            .submit_ballot()
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "Not a ballot submission transaction".to_string(),
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

    Ok(())
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

    let sender_address = transfer.sender;

    let sender_rep =
        state.reputation().get_reputation(rwtxn, &sender_address)?;
    if sender_rep < transfer.amount {
        return Err(Error::InvalidTransaction {
            reason: format!(
                "Insufficient reputation at apply: have {sender_rep}, \
                 need {}",
                transfer.amount
            ),
        });
    }
    let receiver_rep =
        state.reputation().get_reputation(rwtxn, &transfer.dest)?;

    let undo_entry = crate::state::undo::ReputationTransferUndoEntry {
        sender: sender_address,
        sender_pre_reputation: sender_rep,
        receiver: transfer.dest,
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

    state.reputation().set_reputation(
        rwtxn,
        &sender_address,
        sender_rep - transfer.amount,
    )?;
    state.reputation().set_reputation(
        rwtxn,
        &transfer.dest,
        receiver_rep + transfer.amount,
    )?;

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

    #[test]
    fn test_buy_then_sell_same_block_allowed() {
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

    #[test]
    fn test_multiple_sells_different_outcomes_same_block() {
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

    #[test]
    fn test_cumulative_sells_same_outcome_same_block() {
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
}
