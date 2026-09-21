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
    pub is_amplify: bool,
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
        let mut mutated_market_ids = std::collections::BTreeSet::new();
        mutated_market_ids
            .extend(self.market_updates.iter().map(|update| update.market_id));
        mutated_market_ids.extend(
            self.pending_buy_settlements
                .iter()
                .filter(|settlement| settlement.is_amplify)
                .map(|settlement| settlement.market_id),
        );
        if !mutated_market_ids.is_empty() {
            let mut entries = Vec::with_capacity(mutated_market_ids.len());
            for market_id in mutated_market_ids {
                let market = state
                    .markets()
                    .get_market(rwtxn, &market_id)?
                    .ok_or_else(|| Error::InvalidTransaction {
                        reason: format!(
                            "Market {market_id:?} not found before mutation"
                        ),
                    })?;
                entries.push(market);
            }
            let touched_addresses: std::collections::BTreeSet<_> = self
                .share_account_changes
                .keys()
                .map(|(address, _)| *address)
                .collect();
            let mut share_accounts =
                Vec::with_capacity(touched_addresses.len());
            for address in touched_addresses {
                share_accounts.push((
                    address,
                    state.markets().get_user_share_account(rwtxn, &address)?,
                ));
            }
            state.market_transition_undo.put(
                rwtxn,
                &height,
                &crate::state::undo::MarketTransitionUndoData {
                    entries,
                    share_accounts,
                },
            )?;
        }

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

        let mut amplify_by_market: std::collections::HashMap<MarketId, u64> =
            std::collections::HashMap::new();
        for settlement in &self.pending_buy_settlements {
            if settlement.is_amplify {
                *amplify_by_market.entry(settlement.market_id).or_default() +=
                    settlement.lmsr_cost_sats;
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
            state
                .native()
                .capture_account(state, rwtxn, height, *address)?;
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
    let parent_height = prevalidated.parent_height;
    let filled_txs = prevalidated.filled_transactions;
    let validated_coinbase = prevalidated.coinbase_value;

    // The current mainchain timestamp is consensus state used by Decision and
    // market transitions. Preserve its exact predecessor before overwriting it
    // so disconnect_tip is a true inverse, including at genesis.
    let previous_mainchain_timestamp =
        state.try_get_mainchain_timestamp(rwtxn)?;
    state.mainchain_timestamp_undo.put(
        rwtxn,
        &height,
        &previous_mainchain_timestamp,
    )?;

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
            Some(TxData::NativeOperation(operation)) => {
                if !apply_native_operation(
                    state,
                    rwtxn,
                    filled_tx,
                    operation,
                    &mut state_update,
                    height,
                    parent_height,
                )? {
                    skipped_tx_indices.insert(idx);
                }
            }
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
            Some(TxData::ClaimDecision(_)) => {
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
                apply_amplify_beta(filled_tx, &mut state_update)?;
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
        let (payout_results, settlement_undo_entries) =
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
                    "Protocol: Market {} auto-settled with {} sats treasury + {} sats fees distributed to {} shareholders ({} sats refunded to creator)",
                    market_id,
                    summary.treasury_distributed,
                    summary.total_fees_distributed,
                    summary.shareholder_count,
                    refund_sats,
                );
            }
        }

        if !settlement_undo_entries.is_empty() {
            let undo_data = crate::state::undo::SettlementUndoData {
                entries: settlement_undo_entries,
            };
            state.settlement_undo.put(rwtxn, &height, &undo_data)?;
        }
    }

    if !skipped_tx_indices.is_empty() {
        let mut indices: Vec<u32> =
            skipped_tx_indices.iter().map(|i| *i as u32).collect();
        indices.sort_unstable();
        state
            .skipped_tx_indices_undo
            .put(rwtxn, &height, &indices)?;
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

    // 2. Revert market settlement/payouts (runs last in connect, so first here)
    if let Some(settlement_undo) =
        state.settlement_undo.try_get(rwtxn, &height)?
    {
        revert_settlement(state, rwtxn, &settlement_undo, height)?;
        state.settlement_undo.delete(rwtxn, &height)?;
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

    let skipped_tx_indices: HashSet<u32> = state
        .skipped_tx_indices_undo
        .try_get(rwtxn, &height)?
        .unwrap_or_default()
        .into_iter()
        .collect();

    for (idx, tx) in body.transactions.iter().enumerate().rev() {
        if skipped_tx_indices.contains(&(idx as u32)) {
            continue;
        }
        let txid = tx.txid();
        let filled_tx = state.fill_transaction_from_stxos(rwtxn, tx.clone())?;
        match &tx.data {
            None => (),
            Some(TxData::ClaimDecision(_)) => {
                let () = revert_decision_claim(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::CreateMarket { .. }) => {
                let () = revert_create_market(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::NativeOperation(operation)) => {
                use crate::types::native::{
                    EscrowAssetV1, NativeEffectKindV1, NativeOperationV1,
                };
                if matches!(operation, NativeOperationV1::BuyForIntent { .. }) {
                    trade_share_deltas.push(revert_trade_market_state(
                        state, rwtxn, &filled_tx,
                    )?);
                } else if let Some(effect) =
                    state.native().get_effect(rwtxn, txid.0)?
                {
                    if matches!(
                        effect.kind,
                        NativeEffectKindV1::Transferred
                            | NativeEffectKindV1::Claimed
                            | NativeEffectKindV1::Refunded
                    ) && effect.asset == EscrowAssetV1::Shares
                        && effect.owner != effect.recipient
                    {
                        trade_share_deltas.push(TradeShareDelta {
                            address: effect.owner,
                            market_id: effect.market_id,
                            outcome_index: effect.outcome_index,
                            share_delta: effect.shares,
                        });
                        trade_share_deltas.push(TradeShareDelta {
                            address: effect.recipient,
                            market_id: effect.market_id,
                            outcome_index: effect.outcome_index,
                            share_delta: -effect.shares,
                        });
                    }
                } else {
                    return Err(super::native::invalid(
                        "missing native effect during disconnect",
                    ));
                }
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
            Some(TxData::AmplifyBeta { .. }) => {
                let () = revert_amplify_beta(state, rwtxn, &filled_tx)?;
            }
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

    if !skipped_tx_indices.is_empty() {
        state.skipped_tx_indices_undo.delete(rwtxn, &height)?;
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

    if let Some(market_undo) =
        state.market_transition_undo.try_get(rwtxn, &height)?
    {
        for market in market_undo.entries {
            state.markets().update_market(rwtxn, &market)?;
        }
        for (address, account) in market_undo.share_accounts {
            state.markets().restore_share_account(
                rwtxn,
                &address,
                account.as_ref(),
            )?;
        }
        state.market_transition_undo.delete(rwtxn, &height)?;
    }
    state.native().restore(state, rwtxn, height)?;

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

    // 7c. Restore period_pricing from undo log
    state
        .decisions()
        .restore_period_pricing_undo(rwtxn, height)?;

    // 8. Restore consensus time and then update tip/height to the predecessor.
    let previous_mainchain_timestamp = state
        .mainchain_timestamp_undo
        .try_get(rwtxn, &height)?
        .ok_or_else(|| {
            Error::DatabaseError(format!(
                "missing mainchain timestamp undo at height {height}"
            ))
        })?;
    match previous_mainchain_timestamp {
        Some(timestamp) => {
            state.mainchain_timestamp.put(rwtxn, &(), &timestamp)?;
        }
        None => {
            state.mainchain_timestamp.delete(rwtxn, &())?;
        }
    }
    state.mainchain_timestamp_undo.delete(rwtxn, &height)?;
    if height == 0 {
        state.genesis_timestamp.delete(rwtxn, &())?;
    }

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

/// Revert market settlement and payouts (C1)
fn revert_settlement(
    state: &State,
    rwtxn: &mut RwTxn,
    undo: &crate::state::undo::SettlementUndoData,
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
        for (address, account) in &entry.pre_settlement_share_accounts {
            state.markets().restore_share_account(
                rwtxn,
                address,
                Some(account),
            )?;
        }

        // Restore treasury UTXO
        if let Some((ref outpoint, ref filled_output)) = entry.treasury_utxo {
            state.insert_utxo(rwtxn, outpoint, filled_output)?;
            state.markets().set_market_funds_utxo(
                rwtxn,
                &entry.pre_settlement_market.id,
                false,
                outpoint,
            )?;
        }

        // Restore fee UTXO
        if let Some((ref outpoint, ref filled_output)) = entry.fee_utxo {
            state.insert_utxo(rwtxn, outpoint, filled_output)?;
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
    for update in &state_update.market_updates {
        if update.market_id == *market_id
            && let Some((outcome_index, delta)) = update.share_delta
        {
            effective_shares[outcome_index] += delta;
        }
    }

    let pending_amplify: u64 = state_update
        .pending_buy_settlements
        .iter()
        .filter(|s| s.market_id == *market_id && s.is_amplify)
        .map(|s| s.lmsr_cost_sats)
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

    let beta = market_beta(&market);

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

#[allow(clippy::too_many_arguments)]
fn apply_native_operation(
    state: &State,
    txn: &mut RwTxn,
    filled: &FilledTransaction,
    operation: &crate::types::native::NativeOperationV1,
    update: &mut StateUpdate,
    height: u32,
    parent_height: u32,
) -> Result<bool, Error> {
    use super::native::{check_deadline, invalid, validate_terminal};
    use crate::types::native::*;
    check_deadline(operation, parent_height)?;
    if height < super::native::NATIVE_OPERATIONS_ACTIVATION_HEIGHT {
        return Err(invalid("native operations not activated"));
    }
    if let Some(key) = operation.nonce() {
        if state.native().nonces.try_get(txn, &key)?.is_some() {
            return Err(invalid("native operation nonce already consumed"));
        }
    }
    let txid = filled.txid().0;
    let effect = match operation {
        NativeOperationV1::BuyForIntent { intent, .. } => {
            if matches!(
                apply_trade(state, txn, filled, update, height)?,
                TradeApplyResult::Skipped { .. }
            ) {
                return Ok(false);
            }
            NativeEffectV1 {
                transaction_id: txid,
                sidechain_height: height,
                parent_height,
                reference: intent.reference,
                kind: NativeEffectKindV1::Bought,
                escrow_id: None,
                owner: intent.recipient,
                recipient: intent.recipient,
                market_id: intent.market_id,
                outcome_index: intent.outcome_index,
                shares: intent.shares,
                asset: EscrowAssetV1::Shares,
                escrow_snapshot: None,
            }
        }
        NativeOperationV1::TransferShares {
            owner,
            recipient,
            market_id,
            outcome_index,
            shares,
            reference,
            ..
        } => {
            require_available(
                state,
                txn,
                update,
                *owner,
                *market_id,
                *outcome_index,
                *shares,
            )?;
            native_share_transfer(
                state,
                txn,
                update,
                height,
                *owner,
                *recipient,
                *market_id,
                *outcome_index,
                *shares,
            )?;
            NativeEffectV1 {
                transaction_id: txid,
                sidechain_height: height,
                parent_height,
                reference: *reference,
                kind: NativeEffectKindV1::Transferred,
                escrow_id: None,
                owner: *owner,
                recipient: *recipient,
                market_id: *market_id,
                outcome_index: *outcome_index,
                shares: *shares,
                asset: EscrowAssetV1::Shares,
                escrow_snapshot: None,
            }
        }
        NativeOperationV1::LockShares {
            owner,
            claim_address,
            refund_address,
            market_id,
            outcome_index,
            shares,
            hashlock,
            claim_before_parent,
            reference,
            mutable_rights,
            ..
        } => {
            require_available(
                state,
                txn,
                update,
                *owner,
                *market_id,
                *outcome_index,
                *shares,
            )?;
            let escrow_id = escrow_id(txid);
            let escrow = ShareEscrowV1 {
                escrow_id,
                owner: *owner,
                claim_address: *claim_address,
                refund_address: *refund_address,
                market_id: *market_id,
                outcome_index: *outcome_index,
                shares: *shares,
                hashlock: *hashlock,
                claim_before_parent: *claim_before_parent,
                reference: *reference,
                asset: EscrowAssetV1::Shares,
                status: EscrowStatusV1::Locked,
                mutable_rights: *mutable_rights,
            };
            state.native().create_escrow(txn, height, &escrow)?;
            NativeEffectV1 {
                transaction_id: txid,
                sidechain_height: height,
                parent_height,
                reference: *reference,
                kind: NativeEffectKindV1::Locked,
                escrow_id: Some(escrow_id),
                owner: *owner,
                recipient: *claim_address,
                market_id: *market_id,
                outcome_index: *outcome_index,
                shares: *shares,
                asset: EscrowAssetV1::Shares,
                escrow_snapshot: Some(escrow),
            }
        }
        NativeOperationV1::MutateEscrow { .. } => state
            .native()
            .mutate_escrow(txn, height, parent_height, txid, operation)?,
        NativeOperationV1::ClaimEscrow { escrow_id, .. }
        | NativeOperationV1::RefundEscrow { escrow_id } => {
            let mut escrow = state
                .native()
                .get_escrow(txn, *escrow_id)?
                .ok_or_else(|| invalid("unknown native escrow"))?;
            let preimage = match operation {
                NativeOperationV1::ClaimEscrow { preimage, .. } => {
                    Some(preimage)
                }
                _ => None,
            };
            validate_terminal(&escrow, parent_height, preimage)?;
            let refund = preimage.is_none();
            let recipient = if refund {
                escrow.refund_address
            } else {
                escrow.claim_address
            };
            if escrow.asset == EscrowAssetV1::Shares {
                // This operation spends its own reservation, not the unreserved balance.
                native_share_transfer(
                    state,
                    txn,
                    update,
                    height,
                    escrow.owner,
                    recipient,
                    escrow.market_id,
                    escrow.outcome_index,
                    escrow.shares,
                )?;
            }
            state.native().terminate(
                state,
                txn,
                height,
                &mut escrow,
                txid,
                recipient,
                refund,
            )?;
            NativeEffectV1 {
                transaction_id: txid,
                sidechain_height: height,
                parent_height,
                reference: escrow.reference,
                kind: if refund {
                    NativeEffectKindV1::Refunded
                } else {
                    NativeEffectKindV1::Claimed
                },
                escrow_id: Some(*escrow_id),
                owner: escrow.owner,
                recipient,
                market_id: escrow.market_id,
                outcome_index: escrow.outcome_index,
                shares: escrow.shares,
                asset: escrow.asset.clone(),
                escrow_snapshot: Some(escrow),
            }
        }
    };
    state.native().record(txn, operation, &effect)?;
    Ok(true)
}

fn effective_native_shares(
    state: &State,
    txn: &sneed::RoTxn,
    update: &StateUpdate,
    owner: Address,
    market: MarketId,
    outcome: u32,
) -> Result<i64, Error> {
    let owned = state
        .markets()
        .get_user_share_account(txn, &owner)?
        .and_then(|account| account.positions.get(&(market, outcome)).copied())
        .unwrap_or(0);
    let pending = update
        .share_account_changes
        .get(&(owner, market))
        .and_then(|outcomes| outcomes.get(&outcome))
        .copied()
        .unwrap_or(0);
    owned
        .checked_add(pending)
        .ok_or_else(|| super::native::invalid("native share balance overflow"))
}

fn require_available(
    state: &State,
    txn: &sneed::RoTxn,
    update: &StateUpdate,
    owner: Address,
    market: MarketId,
    outcome: u32,
    quantity: i64,
) -> Result<(), Error> {
    let effective =
        effective_native_shares(state, txn, update, owner, market, outcome)?;
    let reserved = state
        .native()
        .reserved_shares(txn, owner, market, outcome)?;
    if quantity <= 0
        || effective
            .checked_sub(reserved)
            .is_none_or(|available| available < quantity)
    {
        return Err(super::native::invalid(
            "insufficient unreserved native shares",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn native_share_transfer(
    state: &State,
    txn: &mut RwTxn,
    update: &mut StateUpdate,
    height: u32,
    owner: Address,
    recipient: Address,
    market: MarketId,
    outcome: u32,
    quantity: i64,
) -> Result<(), Error> {
    if quantity <= 0
        || effective_native_shares(state, txn, update, owner, market, outcome)?
            < quantity
    {
        return Err(super::native::invalid("insufficient native shares"));
    }
    if owner == recipient {
        return Ok(());
    }
    effective_native_shares(state, txn, update, recipient, market, outcome)?
        .checked_add(quantity)
        .ok_or_else(|| {
            super::native::invalid("recipient native share overflow")
        })?;
    state.native().capture_account(state, txn, height, owner)?;
    state
        .native()
        .capture_account(state, txn, height, recipient)?;
    for (address, delta) in [(owner, -quantity), (recipient, quantity)] {
        let entry = update
            .share_account_changes
            .entry((address, market))
            .or_default()
            .entry(outcome)
            .or_default();
        *entry = entry.checked_add(delta).ok_or_else(|| {
            super::native::invalid("pending native share overflow")
        })?;
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

    let (effective_shares, beta) =
        running_market_state(&trade.market_id, &market, state_update);

    let mut new_shares = effective_shares.clone();
    new_shares[outcome_index] = new_shares[outcome_index]
        .checked_add(trade.shares)
        .ok_or_else(|| super::native::invalid("market share overflow"))?;
    effective_native_shares(
        state,
        rwtxn,
        state_update,
        trade.trader,
        trade.market_id,
        trade.outcome_index,
    )?
    .checked_add(trade.shares)
    .ok_or_else(|| super::native::invalid("account share overflow"))?;

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
            &effective_shares,
            &new_shares,
            beta,
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

        if total_trade_cost
            .checked_add(TRADE_MINER_FEE_SATS)
            .is_none_or(|cost| cost > trade.limit_sats)
        {
            return Ok(TradeApplyResult::Skipped {
                reason: format!(
                    "Buy cost {} sats + miner fee {} sats exceeds max cost {} sats",
                    total_trade_cost, TRADE_MINER_FEE_SATS, trade.limit_sats
                ),
            });
        }

        state_update.add_pending_buy_settlement(PendingBuySettlement {
            market_id: trade.market_id,
            trader_address: match &filled_tx.transaction.data {
                Some(TxData::NativeOperation(
                    crate::types::native::NativeOperationV1::BuyForIntent {
                        change_address,
                        ..
                    },
                )) => *change_address,
                _ => trade.trader,
            },
            input_value_sats,
            lmsr_cost_sats: buy_cost.base_cost_sats,
            market_fee_sats: buy_cost.trading_fee_sats,
            transaction_id: filled_tx.transaction.txid().0,
            is_amplify: false,
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

        let reserved = state.native().reserved_shares(
            rwtxn,
            trade.trader,
            trade.market_id,
            trade.outcome_index,
        )?;
        let effective_owned = owned_shares
            .checked_add(pending_delta)
            .and_then(|q| q.checked_sub(reserved))
            .ok_or_else(|| super::native::invalid("share balance overflow"))?;

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
            &effective_shares,
            beta,
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
    use crate::state::{
        MarketBuilder,
        decisions::{Decision, DecisionId},
        markets::DimensionSpec,
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

            market.liquidity_base_sats = amount.0.to_sat();

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

    #[test]
    fn same_block_buys_fold_shares_not_beta() {
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
            share_delta: Some((0, 100_000)),
            transaction_id: None,
            volume_sats: None,
            fee_sats: None,
        });
        state_update.add_pending_buy_settlement(PendingBuySettlement {
            market_id,
            trader_address: Address::ALL_ZEROS,
            input_value_sats: 0,
            lmsr_cost_sats: 12_345,
            market_fee_sats: 0,
            transaction_id: [0u8; 32],
            is_amplify: false,
        });

        let mut effective = s0.clone();
        for update in &state_update.market_updates {
            if update.market_id == market_id
                && let Some((oi, d)) = update.share_delta
            {
                effective[oi] += d;
            }
        }
        assert_eq!(effective, after_one);

        let base1 = trading::calculate_buy_cost(
            trading::calculate_update_cost(&s0, &after_one, beta).unwrap(),
            0.0,
        )
        .unwrap()
        .base_cost_sats;

        let base2_sequential = trading::calculate_buy_cost(
            trading::calculate_update_cost(&after_one, &after_two, beta)
                .unwrap(),
            0.0,
        )
        .unwrap()
        .base_cost_sats;

        assert!(base2_sequential > base1);
    }
}

#[cfg(test)]
mod native_integration_tests {
    use super::*;
    use crate::state::markets::{
        Market, MarketState,
        types::{MarketPayoutSummary, SharePayoutRecord},
    };
    use crate::types::{
        BitcoinOutputContent, FilledOutputContent, Transaction, native::*,
    };
    use bitcoin::hashes::Hash;
    use sha2::{Digest, Sha256};

    fn fixture() -> (sneed::Env, State, tempfile::TempDir, MarketId, Address) {
        let dir = tempfile::tempdir().unwrap();
        let mut options = heed::EnvOpenOptions::new();
        options.map_size(64 * 1024 * 1024).max_dbs(State::NUM_DBS);
        let env = unsafe { sneed::Env::open(&options, dir.path()) }.unwrap();
        let state = State::new(&env, None).unwrap();
        let owner = Address([1; 20]);
        let market_id = MarketId::new([1; 6]);
        let market = Market {
            id: market_id,
            title: "native escrow test".into(),
            description: String::new(),
            tags: vec![],
            creator_address: owner,
            dimension_specs: vec![],
            decision_ids: vec![],
            state_combos: vec![vec![0], vec![1]],
            created_at_height: 0,
            expires_at_height: None,
            tau_from_now: 0,
            storage_fee_sats: 0,
            market_state: MarketState::Trading,
            trading_fee: 0.005,
            liquidity_base_sats: 100,
            shares: ndarray::array![100, 0],
            final_prices: ndarray::array![0.0, 0.0],
            version: 0,
            last_updated_height: 0,
            total_volume_sats: 0,
            outcome_volumes_sats: vec![0, 0],
            tx_pow_hash_selector: 0,
            tx_pow_ordering: 0,
            tx_pow_difficulty: 0,
        };
        let mut txn = env.write_txn().unwrap();
        state.markets().add_market(&mut txn, &market).unwrap();
        state
            .markets()
            .add_shares_to_account(&mut txn, &owner, market_id, 0, 100, 0)
            .unwrap();
        txn.commit().unwrap();
        (env, state, dir, market_id, owner)
    }
    fn filled(operation: NativeOperationV1) -> FilledTransaction {
        FilledTransaction {
            actor_address: operation.actor(),
            spent_utxos: vec![],
            transaction: Transaction {
                inputs: vec![],
                outputs: vec![],
                memo: vec![],
                data: Some(TxData::NativeOperation(operation)),
            },
        }
    }
    fn execute(
        state: &State,
        txn: &mut RwTxn,
        update: &mut StateUpdate,
        tx: &FilledTransaction,
        height: u32,
        parent: u32,
    ) -> Result<bool, Error> {
        let Some(TxData::NativeOperation(op)) = &tx.transaction.data else {
            unreachable!()
        };
        apply_native_operation(state, txn, tx, op, update, height, parent)
    }
    fn lock(
        owner: Address,
        market_id: MarketId,
        nonce: u8,
        shares: i64,
    ) -> FilledTransaction {
        filled(NativeOperationV1::LockShares {
            owner,
            claim_address: Address([2; 20]),
            refund_address: Address([3; 20]),
            market_id,
            outcome_index: 0,
            shares,
            hashlock: Sha256::digest([9; 32]).into(),
            claim_before_parent: 10,
            nonce: [nonce; 32],
            reference: [7; 32],
            mutable_rights: true,
        })
    }
    fn balance(
        state: &State,
        txn: &sneed::RoTxn,
        owner: Address,
        market: MarketId,
    ) -> i64 {
        state
            .markets()
            .get_user_share_account(txn, &owner)
            .unwrap()
            .and_then(|a| a.positions.get(&(market, 0)).copied())
            .unwrap_or(0)
    }
    fn rights_key(n: u8) -> ed25519_dalek::SigningKey {
        ed25519_dalek::SigningKey::from_bytes(&[n; 32])
    }
    fn rights_address(key: &ed25519_dalek::SigningKey) -> Address {
        crate::authorization::get_address(&key.verifying_key().into())
    }
    fn mutation(
        mutation: EscrowMutationV1,
        nonce: u8,
        claim: &ed25519_dalek::SigningKey,
        refund: Option<&ed25519_dalek::SigningKey>,
    ) -> FilledTransaction {
        let intent = EscrowMutationIntentV1 {
            genesis_hash: [1; 32].into(),
            nonce: [nonce; 32],
            mutation,
            valid_from_parent: 0,
            valid_before_parent: 1000,
        };
        let sign =
            |key: &ed25519_dalek::SigningKey| crate::types::Authorization {
                verifying_key: key.verifying_key().into(),
                signature: crate::authorization::sign(
                    key,
                    crate::authorization::Dst::NativeEscrowMutation,
                    &intent.signing_bytes().unwrap(),
                ),
            };
        let claim_authorization = sign(claim);
        let refund_authorization = refund.map(sign);
        filled(NativeOperationV1::MutateEscrow {
            intent,
            claim_authorization,
            refund_authorization,
        })
    }
    #[test]
    fn sealed_escrow_rejects_jointly_signed_mutation_and_keeps_claim_refund() {
        for cash in [None, Some(0), Some(19)] {
            let (env, state, _dir, market, owner) = fixture();
            let a = rights_key(11);
            let b = rights_key(12);
            let c = rights_key(13);
            let mut tx = lock(owner, market, 1, 70);
            if let Some(TxData::NativeOperation(
                NativeOperationV1::LockShares {
                    claim_address,
                    refund_address,
                    mutable_rights,
                    ..
                },
            )) = &mut tx.transaction.data
            {
                *claim_address = rights_address(&a);
                *refund_address = rights_address(&b);
                *mutable_rights = false;
            }
            let mut json = serde_json::to_value(&tx.transaction.data).unwrap();
            json["NativeOperation"]["LockShares"]
                .as_object_mut()
                .unwrap()
                .remove("mutable_rights");
            let decoded: Option<TxData> = serde_json::from_value(json).unwrap();
            assert!(matches!(
                decoded,
                Some(TxData::NativeOperation(NativeOperationV1::LockShares {
                    mutable_rights: false,
                    ..
                }))
            ));
            let id = escrow_id(tx.txid().0);
            let mut txn = env.write_txn().unwrap();
            let mut update = StateUpdate::new();
            execute(&state, &mut txn, &mut update, &tx, 1, 5).unwrap();
            if let Some(value) = cash {
                state
                    .native()
                    .settle_payout(
                        &state,
                        &mut txn,
                        &SharePayoutRecord {
                            market_id: market,
                            address: owner,
                            outcome_index: 0,
                            shares_redeemed: 100,
                            final_price: 0.19,
                            payout_sats: value,
                        },
                        1,
                    )
                    .unwrap();
            }
            let original =
                state.native().get_escrow(&txn, id).unwrap().unwrap();
            for spec in [EscrowMutationV1::Assign {
                escrow_id: id,
                new_claim_address: rights_address(&c),
                new_refund_address: rights_address(&c),
                reference: [23; 32],
            }] {
                let mutate = mutation(spec, 2, &a, Some(&b));
                assert!(
                    execute(&state, &mut txn, &mut update, &mutate, 1, 5)
                        .is_err()
                );
                assert_eq!(
                    state.native().get_escrow(&txn, id).unwrap(),
                    Some(original.clone())
                );
                assert!(
                    state
                        .native()
                        .get_effect(&txn, mutate.txid().0)
                        .unwrap()
                        .is_none()
                );
            }
            let refund =
                filled(NativeOperationV1::RefundEscrow { escrow_id: id });
            assert!(
                execute(&state, &mut txn, &mut update, &refund, 1, 9).is_err()
            );
            execute(&state, &mut txn, &mut update, &refund, 1, 10).unwrap();
            let terminal =
                state.native().get_escrow(&txn, id).unwrap().unwrap();
            assert!(!terminal.mutable_rights);
            assert_eq!(terminal.refund_address, rights_address(&b));
        }
    }

    #[test]
    fn mutation_expiry_is_signed_and_atomic_at_accepting_parent_boundary() {
        for accepted_parent in [5, 6, 9, 10, 11] {
            let (env, state, _dir, market, owner) = fixture();
            let a = rights_key(11);
            let b = rights_key(12);
            let c = rights_key(13);
            let mut lock_tx = lock(owner, market, 1, 70);
            if let Some(TxData::NativeOperation(
                NativeOperationV1::LockShares {
                    claim_address,
                    refund_address,
                    ..
                },
            )) = &mut lock_tx.transaction.data
            {
                *claim_address = rights_address(&a);
                *refund_address = rights_address(&b);
            }
            let id = escrow_id(lock_tx.txid().0);
            let mut txn = env.write_txn().unwrap();
            let mut update = StateUpdate::new();
            execute(&state, &mut txn, &mut update, &lock_tx, 1, 5).unwrap();
            let prior = state.native().get_escrow(&txn, id).unwrap();
            let mut tx = mutation(
                EscrowMutationV1::Assign {
                    escrow_id: id,
                    new_claim_address: rights_address(&c),
                    new_refund_address: rights_address(&c),
                    reference: [23; 32],
                },
                2,
                &a,
                Some(&b),
            );
            if let Some(TxData::NativeOperation(
                NativeOperationV1::MutateEscrow {
                    intent,
                    claim_authorization,
                    refund_authorization,
                },
            )) = &mut tx.transaction.data
            {
                intent.valid_from_parent = 6;
                intent.valid_before_parent = 10;
                assert!(!intent.verify(claim_authorization));
                let sign = |key: &ed25519_dalek::SigningKey| {
                    crate::types::Authorization {
                        verifying_key: key.verifying_key().into(),
                        signature: crate::authorization::sign(
                            key,
                            crate::authorization::Dst::NativeEscrowMutation,
                            &intent.signing_bytes().unwrap(),
                        ),
                    }
                };
                *claim_authorization = sign(&a);
                *refund_authorization = Some(sign(&b));
            }
            let result =
                execute(&state, &mut txn, &mut update, &tx, 2, accepted_parent);
            if (6..10).contains(&accepted_parent) {
                result.unwrap();
                let effect = state
                    .native()
                    .get_effect(&txn, tx.txid().0)
                    .unwrap()
                    .unwrap();
                assert!(matches!(
                    effect.kind,
                    NativeEffectKindV1::Assigned {
                        valid_from_parent: 6,
                        valid_before_parent: 10,
                        ..
                    }
                ));
                assert_eq!(effect.parent_height, accepted_parent);
                state.native().restore(&state, &mut txn, 2).unwrap();
                assert_eq!(state.native().get_escrow(&txn, id).unwrap(), prior);
                assert!(
                    state
                        .native()
                        .get_effect(&txn, tx.txid().0)
                        .unwrap()
                        .is_none()
                );
            } else {
                assert!(result.is_err());
                assert_eq!(state.native().get_escrow(&txn, id).unwrap(), prior);
                assert!(
                    state
                        .native()
                        .get_effect(&txn, tx.txid().0)
                        .unwrap()
                        .is_none()
                );
                assert!(
                    state
                        .native()
                        .nonces
                        .try_get(&txn, &(rights_address(&a), [2; 32]))
                        .unwrap()
                        .is_none()
                );
            }
        }
    }

    #[test]
    fn native_assign_and_claim_ordered_atomic_rights_and_undo() {
        let (env, state, _dir, market, owner) = fixture();
        let a = rights_key(11);
        let b = rights_key(12);
        let c = rights_key(13);
        let mut lock_tx = lock(owner, market, 1, 70);
        if let Some(TxData::NativeOperation(NativeOperationV1::LockShares {
            claim_address,
            refund_address,
            ..
        })) = &mut lock_tx.transaction.data
        {
            *claim_address = rights_address(&a);
            *refund_address = rights_address(&b);
        }
        let id = escrow_id(lock_tx.txid().0);
        let mut txn = env.write_txn().unwrap();
        let mut update = StateUpdate::new();
        execute(&state, &mut txn, &mut update, &lock_tx, 1, 5).unwrap();
        let assign = mutation(
            EscrowMutationV1::Assign {
                escrow_id: id,
                new_claim_address: rights_address(&c),
                new_refund_address: rights_address(&c),
                reference: [23; 32],
            },
            3,
            &a,
            Some(&b),
        );
        let spec = EscrowMutationV1::Assign {
            escrow_id: id,
            new_claim_address: rights_address(&c),
            new_refund_address: rights_address(&c),
            reference: [23; 32],
        };
        for invalid in [
            mutation(spec.clone(), 3, &a, None),
            mutation(spec, 3, &a, Some(&c)),
        ] {
            assert!(
                execute(&state, &mut txn, &mut update, &invalid, 1, 5).is_err()
            );
        }
        execute(&state, &mut txn, &mut update, &assign, 1, 5).unwrap();
        let receipt = state
            .native()
            .get_effect(&txn, assign.txid().0)
            .unwrap()
            .unwrap();
        assert_eq!(
            receipt.escrow_snapshot.as_ref().unwrap().status,
            EscrowStatusV1::Locked
        );
        assert_eq!(
            receipt.kind,
            NativeEffectKindV1::Assigned {
                previous_claim_address: rights_address(&a),
                previous_refund_address: rights_address(&b),
                valid_from_parent: 0,
                valid_before_parent: 1000,
            }
        );
        let stale = mutation(
            EscrowMutationV1::Assign {
                escrow_id: id,
                new_claim_address: rights_address(&a),
                new_refund_address: rights_address(&b),
                reference: [24; 32],
            },
            4,
            &a,
            Some(&b),
        );
        assert!(execute(&state, &mut txn, &mut update, &stale, 1, 5).is_err());
        let back = mutation(
            EscrowMutationV1::Assign {
                escrow_id: id,
                new_claim_address: rights_address(&a),
                new_refund_address: rights_address(&b),
                reference: [24; 32],
            },
            4,
            &c,
            None,
        );
        execute(&state, &mut txn, &mut update, &back, 1, 5).unwrap();
        assert!(execute(&state, &mut txn, &mut update, &assign, 1, 5).is_err()); // consumed nonce even after rights return
        let claim = filled(NativeOperationV1::ClaimEscrow {
            escrow_id: id,
            preimage: [9; 32],
        });
        execute(&state, &mut txn, &mut update, &claim, 1, 5).unwrap();
        let terminal = mutation(
            EscrowMutationV1::Assign {
                escrow_id: id,
                new_claim_address: rights_address(&c),
                new_refund_address: rights_address(&c),
                reference: [25; 32],
            },
            5,
            &a,
            Some(&b),
        );
        assert!(
            execute(&state, &mut txn, &mut update, &terminal, 1, 5).is_err()
        );
        assert_eq!(
            state
                .native()
                .reserved_shares(&txn, owner, market, 0)
                .unwrap(),
            0
        );
        // Historical assignment still describes exactly what was delivered.
        assert_eq!(
            state
                .native()
                .get_effect(&txn, assign.txid().0)
                .unwrap()
                .unwrap(),
            receipt
        );
        update.apply_all_changes(&state, &mut txn, 1).unwrap();
        assert_eq!(balance(&state, &txn, rights_address(&a), market), 70);
        state.native().restore(&state, &mut txn, 1).unwrap();
        assert!(state.native().get_escrow(&txn, id).unwrap().is_none());
        assert_eq!(balance(&state, &txn, owner, market), 100);
        assert_eq!(
            state
                .native()
                .reserved_shares(&txn, owner, market, 0)
                .unwrap(),
            0
        );
        assert!(
            state
                .native()
                .get_effect(&txn, assign.txid().0)
                .unwrap()
                .is_none()
        );
    }
    #[test]
    fn native_cash_assignment_preserves_value_deadline_and_undo() {
        for total in [0, 1, u64::MAX] {
            let (env, state, _dir, market, owner) = fixture();
            let key = rights_key(20);
            let address = rights_address(&key);
            let escrow = ShareEscrowV1 {
                escrow_id: [31; 32],
                owner,
                claim_address: address,
                refund_address: address,
                market_id: market,
                outcome_index: 0,
                shares: 7,
                hashlock: Sha256::digest([9; 32]).into(),
                claim_before_parent: 10,
                reference: [32; 32],
                asset: EscrowAssetV1::NativeCash(total),
                status: EscrowStatusV1::Locked,
                mutable_rights: true,
            };
            let mut txn = env.write_txn().unwrap();
            state.native().create_escrow(&mut txn, 1, &escrow).unwrap();
            let destination = rights_address(&rights_key(21));
            let assign = mutation(
                EscrowMutationV1::Assign {
                    escrow_id: escrow.escrow_id,
                    new_claim_address: destination,
                    new_refund_address: destination,
                    reference: [33; 32],
                },
                6,
                &key,
                None,
            );
            let mut update = StateUpdate::new();
            execute(&state, &mut txn, &mut update, &assign, 2, 20).unwrap();
            let assigned = state
                .native()
                .get_escrow(&txn, escrow.escrow_id)
                .unwrap()
                .unwrap();
            assert_eq!(assigned.asset, EscrowAssetV1::NativeCash(total));
            assert_eq!(assigned.shares, escrow.shares);
            assert_eq!(
                assigned.claim_before_parent,
                escrow.claim_before_parent
            );
            assert_eq!(assigned.hashlock, escrow.hashlock);
            assert_eq!(assigned.claim_address, destination);
            assert_eq!(assigned.refund_address, destination);
            assert_eq!(state.native().cash_liability(&txn).unwrap(), total);
            assert_eq!(
                state
                    .native()
                    .reserved_shares(&txn, owner, market, 0)
                    .unwrap(),
                0
            );
            state.native().restore(&state, &mut txn, 2).unwrap();
            assert_eq!(
                state.native().get_escrow(&txn, escrow.escrow_id).unwrap(),
                Some(escrow)
            );
            assert_eq!(state.native().cash_liability(&txn).unwrap(), total);
        }
    }

    #[test]
    fn native_lock_blocks_sell_double_lock_and_transfer_then_claim_is_final() {
        let (env, state, _dir, market, owner) = fixture();
        let mut txn = env.write_txn().unwrap();
        let mut update = StateUpdate::new();
        let lock_tx = lock(owner, market, 1, 70);
        let id = escrow_id(lock_tx.txid().0);
        assert!(
            execute(&state, &mut txn, &mut update, &lock_tx, 1, 5).unwrap()
        );
        assert_eq!(
            state
                .native()
                .reserved_shares(&txn, owner, market, 0)
                .unwrap(),
            70
        );
        assert!(
            execute(
                &state,
                &mut txn,
                &mut update,
                &lock(owner, market, 2, 31),
                1,
                5
            )
            .is_err()
        );
        let transfer = filled(NativeOperationV1::TransferShares {
            owner,
            recipient: Address([4; 20]),
            market_id: market,
            outcome_index: 0,
            shares: 31,
            nonce: [3; 32],
            reference: [4; 32],
            valid_from_parent: 0,
            valid_before_parent: 10,
        });
        assert!(
            execute(&state, &mut txn, &mut update, &transfer, 1, 5).is_err()
        );
        let sell = FilledTransaction {
            transaction: Transaction {
                inputs: vec![],
                outputs: vec![],
                memo: vec![],
                data: Some(TxData::Trade {
                    market_id: market,
                    outcome_index: 0,
                    shares: -31,
                    trader: owner,
                    limit_sats: 0,
                    tx_pow_nonce: None,
                    prev_block_hash: [0; 32].into(),
                }),
            },
            spent_utxos: vec![],
            actor_address: Some(owner),
        };
        assert!(apply_trade(&state, &mut txn, &sell, &mut update, 1).is_err());
        let early_refund =
            filled(NativeOperationV1::RefundEscrow { escrow_id: id });
        assert!(
            execute(&state, &mut txn, &mut update, &early_refund, 1, 9)
                .is_err()
        );
        let bad_claim = filled(NativeOperationV1::ClaimEscrow {
            escrow_id: id,
            preimage: [8; 32],
        });
        assert!(
            execute(&state, &mut txn, &mut update, &bad_claim, 1, 9).is_err()
        );
        let claim = filled(NativeOperationV1::ClaimEscrow {
            escrow_id: id,
            preimage: [9; 32],
        });
        assert!(execute(&state, &mut txn, &mut update, &claim, 1, 9).unwrap());
        assert!(
            execute(&state, &mut txn, &mut update, &early_refund, 1, 10)
                .is_err()
        );
        update.apply_all_changes(&state, &mut txn, 1).unwrap();
        assert_eq!(balance(&state, &txn, owner, market), 30);
        assert_eq!(balance(&state, &txn, Address([2; 20]), market), 70);
        assert_eq!(
            state
                .native()
                .reserved_shares(&txn, owner, market, 0)
                .unwrap(),
            0
        );
        assert_eq!(
            state
                .native()
                .get_effect(&txn, claim.txid().0)
                .unwrap()
                .unwrap()
                .kind,
            NativeEffectKindV1::Claimed
        );
        state.native().restore(&state, &mut txn, 1).unwrap();
        assert_eq!(balance(&state, &txn, owner, market), 100);
        assert!(
            state
                .markets()
                .get_user_share_account(&txn, &Address([2; 20]))
                .unwrap()
                .is_none()
        );
        assert!(state.native().get_escrow(&txn, id).unwrap().is_none());
        assert!(
            state
                .native()
                .get_effect(&txn, claim.txid().0)
                .unwrap()
                .is_none()
        );
        assert!(
            state
                .native()
                .nonces
                .try_get(&txn, &(owner, [1; 32]))
                .unwrap()
                .is_none()
        );
    }
    #[test]
    fn native_settlement_keeps_hashlock_and_cash_refund_and_undo() {
        for payout in [0, 51] {
            let (env, state, _dir, market, owner) = fixture();
            let mut txn = env.write_txn().unwrap();
            let original = bincode::serialize(
                &state
                    .markets()
                    .get_user_share_account(&txn, &owner)
                    .unwrap(),
            )
            .unwrap();
            let mut update = StateUpdate::new();
            let lock_tx = lock(owner, market, 1, 70);
            let id = escrow_id(lock_tx.txid().0);
            execute(&state, &mut txn, &mut update, &lock_tx, 1, 5).unwrap();
            update.apply_all_changes(&state, &mut txn, 1).unwrap();
            let summary = MarketPayoutSummary {
                market_id: market,
                treasury_distributed: payout,
                total_fees_distributed: 0,
                shareholder_count: 1,
                payouts: vec![SharePayoutRecord {
                    market_id: market,
                    address: owner,
                    outcome_index: 0,
                    shares_redeemed: 100,
                    final_price: 0.51,
                    payout_sats: payout,
                }],
                fee_payouts: vec![],
                creator_refund: None,
                block_height: 2,
            };
            state
                .markets()
                .apply_automatic_share_payouts(&state, &mut txn, &summary, 2)
                .unwrap();
            let settled = state.native().get_escrow(&txn, id).unwrap().unwrap();
            let EscrowAssetV1::NativeCash(cash) = settled.asset else {
                panic!("escrow did not settle")
            };
            assert_eq!(cash, if payout == 0 { 0 } else { 36 });
            assert_eq!(settled.hashlock, Sha256::digest([9; 32]).as_slice());
            assert_eq!(
                state
                    .native()
                    .reserved_shares(&txn, owner, market, 0)
                    .unwrap(),
                0
            );
            assert_eq!(balance(&state, &txn, owner, market), 0);
            let claim = filled(NativeOperationV1::ClaimEscrow {
                escrow_id: id,
                preimage: [9; 32],
            });
            assert!(
                execute(
                    &state,
                    &mut txn,
                    &mut StateUpdate::new(),
                    &claim,
                    3,
                    10
                )
                .is_err()
            );
            let refund =
                filled(NativeOperationV1::RefundEscrow { escrow_id: id });
            execute(&state, &mut txn, &mut StateUpdate::new(), &refund, 3, 10)
                .unwrap();
            let output = state
                .utxos
                .try_get(
                    &txn,
                    &super::super::native::cash_outpoint(refund.txid().0)
                        .into(),
                )
                .unwrap();
            assert_eq!(output.is_some(), cash > 0);
            if let Some(output) = output {
                assert_eq!(output.address, Address([3; 20]));
                assert_eq!(output.get_bitcoin_value().to_sat(), cash);
            }
            state.native().restore(&state, &mut txn, 3).unwrap();
            assert_eq!(
                state.native().get_escrow(&txn, id).unwrap().unwrap(),
                settled
            );
            state
                .markets()
                .revert_automatic_share_payouts(&state, &mut txn, &summary, 2)
                .unwrap();
            state.native().restore(&state, &mut txn, 2).unwrap();
            assert_eq!(
                state.native().get_escrow(&txn, id).unwrap().unwrap().asset,
                EscrowAssetV1::Shares
            );
            assert_eq!(
                state
                    .native()
                    .reserved_shares(&txn, owner, market, 0)
                    .unwrap(),
                70
            );
            assert_eq!(
                bincode::serialize(
                    &state
                        .markets()
                        .get_user_share_account(&txn, &owner)
                        .unwrap()
                )
                .unwrap(),
                original
            );
        }
    }
    #[test]
    fn native_connected_blocks_disconnect_and_replay_exactly() {
        let (env, state, _dir, market, owner) = fixture();
        let mut txn = env.write_txn().unwrap();
        let original = bincode::serialize(
            &state
                .markets()
                .get_user_share_account(&txn, &owner)
                .unwrap(),
        )
        .unwrap();
        let genesis_body = Body::new(
            vec![],
            vec![crate::types::Output {
                address: owner,
                content: crate::types::OutputContent::Bitcoin(
                    BitcoinOutputContent(bitcoin::Amount::ZERO),
                ),
                memo: vec![],
            }],
        );
        let genesis = Header {
            merkle_root: Body::compute_merkle_root(
                &genesis_body.coinbase,
                &genesis_body.transactions,
            ),
            prev_side_hash: None,
            prev_main_hash: bitcoin::BlockHash::from_byte_array([1; 32]),
        };
        connect_prevalidated(
            &state,
            &mut txn,
            &genesis,
            &genesis_body,
            100,
            crate::state::PrevalidatedBlock {
                filled_transactions: vec![],
                computed_merkle_root: genesis.merkle_root,
                coinbase_value: bitcoin::Amount::ZERO,
                next_height: 0,
                parent_height: 4,
            },
        )
        .unwrap();
        let lock_tx = lock(owner, market, 1, 70);
        let id = escrow_id(lock_tx.txid().0);
        let body = Body {
            coinbase: vec![],
            transactions: vec![lock_tx.transaction.clone()],
            authorizations: vec![],
            actor_proofs: vec![None],
        };
        let header = Header {
            merkle_root: Body::compute_merkle_root(
                &body.coinbase,
                &body.transactions,
            ),
            prev_side_hash: Some(genesis.hash()),
            prev_main_hash: bitcoin::BlockHash::from_byte_array([2; 32]),
        };
        for _ in 0..2 {
            connect_prevalidated(
                &state,
                &mut txn,
                &header,
                &body,
                101,
                crate::state::PrevalidatedBlock {
                    filled_transactions: vec![lock_tx.clone()],
                    computed_merkle_root: header.merkle_root,
                    coinbase_value: bitcoin::Amount::ZERO,
                    next_height: 1,
                    parent_height: 5,
                },
            )
            .unwrap();
            let claim = filled(NativeOperationV1::ClaimEscrow {
                escrow_id: id,
                preimage: [9; 32],
            });
            let claim_body = Body {
                coinbase: vec![],
                transactions: vec![claim.transaction.clone()],
                authorizations: vec![],
                actor_proofs: vec![None],
            };
            let claim_header = Header {
                merkle_root: Body::compute_merkle_root(
                    &claim_body.coinbase,
                    &claim_body.transactions,
                ),
                prev_side_hash: Some(header.hash()),
                prev_main_hash: bitcoin::BlockHash::from_byte_array([3; 32]),
            };
            connect_prevalidated(
                &state,
                &mut txn,
                &claim_header,
                &claim_body,
                102,
                crate::state::PrevalidatedBlock {
                    filled_transactions: vec![claim],
                    computed_merkle_root: claim_header.merkle_root,
                    coinbase_value: bitcoin::Amount::ZERO,
                    next_height: 2,
                    parent_height: 6,
                },
            )
            .unwrap();
            assert_eq!(balance(&state, &txn, Address([2; 20]), market), 70);
            disconnect_tip(&state, &mut txn, &claim_header, &claim_body)
                .unwrap();
            assert_eq!(
                state
                    .native()
                    .reserved_shares(&txn, owner, market, 0)
                    .unwrap(),
                70
            );
            assert_eq!(
                state.native().get_escrow(&txn, id).unwrap().unwrap().status,
                EscrowStatusV1::Locked
            );
            disconnect_tip(&state, &mut txn, &header, &body).unwrap();
            assert!(state.native().get_escrow(&txn, id).unwrap().is_none());
            assert_eq!(
                state.try_get_mainchain_timestamp(&txn).unwrap(),
                Some(100)
            );
            assert_eq!(
                bincode::serialize(
                    &state
                        .markets()
                        .get_user_share_account(&txn, &owner)
                        .unwrap()
                )
                .unwrap(),
                original
            );
            assert!(
                state
                    .markets()
                    .get_user_share_account(&txn, &Address([2; 20]))
                    .unwrap()
                    .is_none()
            );
        }
    }
    #[test]
    fn native_sponsored_soft_skip_has_no_receipt_or_nonce() {
        let (env, state, _dir, market, owner) = fixture();
        let mut txn = env.write_txn().unwrap();
        let key = ed25519_dalek::SigningKey::from_bytes(&[3; 32]);
        let intent = BuyIntentV1 {
            genesis_hash: [0; 32].into(),
            recipient: owner,
            market_id: market,
            outcome_index: 0,
            shares: 10,
            nonce: [5; 32],
            reference: [6; 32],
            valid_from_parent: 0,
            valid_before_parent: 10,
        };
        let authorization = crate::authorization::Authorization {
            verifying_key: key.verifying_key().into(),
            signature: crate::authorization::sign(
                &key,
                crate::authorization::Dst::NativeIntent,
                &intent.signing_bytes().unwrap(),
            ),
        };
        let mut buy = filled(NativeOperationV1::BuyForIntent {
            intent,
            authorization,
            change_address: Address([4; 20]),
            limit_sats: 1,
            tx_pow_nonce: None,
            prev_block_hash: [0; 32].into(),
        });
        buy.spent_utxos.push(FilledOutput {
            address: Address([4; 20]),
            content: FilledOutputContent::Bitcoin(BitcoinOutputContent(
                bitcoin::Amount::from_sat(100),
            )),
            memo: vec![],
        });
        assert!(
            !execute(&state, &mut txn, &mut StateUpdate::new(), &buy, 1, 5)
                .unwrap()
        );
        assert!(
            state
                .native()
                .get_effect(&txn, buy.txid().0)
                .unwrap()
                .is_none()
        );
        assert!(
            state
                .native()
                .nonces
                .try_get(&txn, &(owner, [5; 32]))
                .unwrap()
                .is_none()
        );
    }
}
