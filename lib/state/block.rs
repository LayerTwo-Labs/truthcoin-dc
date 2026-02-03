use ndarray::{Array, Ix1};
use std::collections::{HashMap, HashSet};

use sneed::{RoTxn, RwTxn};

use crate::{
    math::{
        lmsr::{LmsrError, LmsrService},
        trading,
    },
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
    share_account_changes: HashMap<(Address, MarketId), HashMap<u32, f64>>,
    slot_changes: Vec<SlotStateChange>,
    vote_submissions: Vec<VoteSubmission>,
    voter_registrations: Vec<VoterRegistration>,
    reputation_updates: Vec<ReputationUpdate>,
    pending_sell_payouts: Vec<PendingSellPayout>,
}

struct MarketStateUpdate {
    market_id: MarketId,
    /// Share delta: (outcome_index, shares_to_add)
    share_delta: Option<(usize, f64)>,
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
    pub shares_sold: f64,
    pub outcome_index: u32,
    pub transaction_id: [u8; 32],
}

/// Used to implement soft-fail behavior for slippage failures
enum TradeApplyResult {
    /// Trade was successfully applied
    Applied,
    /// Trade was skipped due to slippage - transaction stays in mempool
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

        for ((address, market_id), outcome_changes) in
            &self.share_account_changes
        {
            for &delta in outcome_changes.values() {
                if !delta.is_finite() {
                    return Err(Error::InvalidTransaction {
                        reason: format!(
                            "Invalid share delta for address {address:?} market {market_id:?}: {delta}"
                        ),
                    });
                }
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
    ) -> Result<(), Error> {
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
                new_shares[*outcome_index] += delta;
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
                if share_delta != 0.0 {
                    if share_delta > 0.0 {
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

        Self::consolidate_market_utxos(
            state,
            rwtxn,
            height,
            &self.pending_sell_payouts,
        )?;

        Ok(())
    }

    /// Generate a deterministic outpoint for sell payouts.
    /// Uses transaction_id to make the outpoint deterministically reconstructible
    /// during revert operations (instead of block_height + sequence which is lost).
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

    fn consolidate_market_utxos(
        state: &State,
        rwtxn: &mut RwTxn,
        height: u32,
        pending_sell_payouts: &[PendingSellPayout],
    ) -> Result<(), Error> {
        use crate::state::markets::{
            generate_market_author_fee_address,
            generate_market_treasury_address,
        };
        use crate::types::{BitcoinOutputContent, FilledOutput, OutPoint};
        use std::collections::HashSet;

        let pending_utxo_markets =
            state.markets().get_markets_with_pending_utxos(rwtxn)?;

        let sell_payout_markets: HashSet<[u8; 6]> =
            pending_sell_payouts.iter().map(|p| p.market_id.0).collect();

        let mut markets_to_consolidate: HashSet<[u8; 6]> =
            pending_utxo_markets.into_iter().collect();
        markets_to_consolidate.extend(sell_payout_markets);

        if markets_to_consolidate.is_empty() {
            return Ok(());
        }

        for market_id_bytes in markets_to_consolidate {
            let market_id = MarketId::new(market_id_bytes);

            let mut treasury_total = 0u64;
            let mut treasury_utxos_to_consume = Vec::new();
            let mut fee_total = 0u64;
            let mut fee_utxos_to_consume = Vec::new();

            if let Some(existing_outpoint) = state
                .markets()
                .get_market_funds_utxo(rwtxn, &market_id, false)?
                && let Some(utxo) =
                    state.utxos.try_get(rwtxn, &existing_outpoint)?
            {
                treasury_total += utxo.get_bitcoin_value().to_sat();
                treasury_utxos_to_consume.push(existing_outpoint);
            }

            if let Some(existing_outpoint) = state
                .markets()
                .get_market_funds_utxo(rwtxn, &market_id, true)?
                && let Some(utxo) =
                    state.utxos.try_get(rwtxn, &existing_outpoint)?
            {
                fee_total += utxo.get_bitcoin_value().to_sat();
                fee_utxos_to_consume.push(existing_outpoint);
            }

            for (outpoint, is_fee) in state
                .markets()
                .get_pending_market_funds_utxos(rwtxn, &market_id)?
            {
                if let Some(utxo) = state.utxos.try_get(rwtxn, &outpoint)? {
                    if is_fee {
                        fee_total += utxo.get_bitcoin_value().to_sat();
                        fee_utxos_to_consume.push(outpoint);
                    } else {
                        treasury_total += utxo.get_bitcoin_value().to_sat();
                        treasury_utxos_to_consume.push(outpoint);
                    }
                }
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

            let has_treasury_work = !treasury_utxos_to_consume.is_empty()
                || !market_sell_payouts.is_empty();

            if has_treasury_work && treasury_total > 0 {
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
                }

                let remaining_treasury =
                    treasury_total.saturating_sub(total_sell_payouts);

                if remaining_treasury > 0 {
                    let treasury_address =
                        generate_market_treasury_address(&market_id);
                    let new_outpoint = OutPoint::MarketFunds {
                        market_id: market_id_bytes,
                        block_height: height,
                        is_fee: false,
                    };
                    let new_output = FilledOutput::new(
                        treasury_address,
                        FilledOutputContent::MarketFunds {
                            market_id: market_id_bytes,
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
                }
            }

            let has_fee_work =
                !fee_utxos_to_consume.is_empty() || total_sell_fees > 0;

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
                    market_id: market_id_bytes,
                    block_height: height,
                    is_fee: true,
                };
                let new_output = FilledOutput::new(
                    fee_address,
                    FilledOutputContent::MarketFunds {
                        market_id: market_id_bytes,
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
            }

            state
                .markets()
                .clear_pending_market_funds_utxos(rwtxn, &market_id)?;
        }

        Ok(())
    }

    fn add_market_update(&mut self, update: MarketStateUpdate) {
        self.market_updates.push(update);
    }
    fn add_share_account_change(
        &mut self,
        address: Address,
        market_id: MarketId,
        outcome: u32,
        delta: f64,
    ) {
        *self
            .share_account_changes
            .entry((address, market_id))
            .or_default()
            .entry(outcome)
            .or_insert(0.0) += delta;
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
}
fn query_update_cost(
    current_shares: &Array<f64, Ix1>,
    new_shares: &Array<f64, Ix1>,
    beta: f64,
) -> Result<f64, LmsrError> {
    LmsrService::calculate_update_cost(current_shares, new_shares, beta)
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
    let mut total_fees = bitcoin::Amount::ZERO;
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
        total_fees = total_fees
            .checked_add(state.validate_filled_transaction(
                rotxn,
                filled_tx,
                Some(future_height),
            )?)
            .ok_or(AmountOverflowError)?;
    }
    if coinbase_value > total_fees {
        return Err(Error::NotEnoughFees);
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
    Ok((total_fees, filled_txs, merkle_root))
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
            .slots()
            .mint_genesis(rwtxn, mainchain_timestamp, height)?;
    } else {
        state
            .slots()
            .mint_up_to(rwtxn, mainchain_timestamp, height)?;
    }

    for claimed_slot in state.slots().get_all_claimed_slots(rwtxn)? {
        let slot_id = claimed_slot.slot_id;
        let current_state =
            state.slots().get_slot_current_state(rwtxn, slot_id)?;

        if current_state == crate::state::slots::SlotState::Claimed {
            let voting_period = slot_id.voting_period();
            let period_info = crate::state::voting::period_calculator::calculate_voting_period(
                rwtxn,
                crate::state::voting::types::VotingPeriodId(voting_period),
                height,
                mainchain_timestamp,
                state.slots().get_config(),
                state.slots(),
                false,
            )?;

            if matches!(
                period_info.status,
                crate::state::voting::types::VotingPeriodStatus::Active
                    | crate::state::voting::types::VotingPeriodStatus::Closed
            ) {
                state.slots().transition_slot_to_voting(
                    rwtxn,
                    slot_id,
                    height as u64,
                    mainchain_timestamp,
                )?;
            }
        }
    }

    let all_periods = state.voting().get_all_periods(
        rwtxn,
        mainchain_timestamp,
        height,
        state.slots().get_config(),
        state.slots(),
    )?;

    for (period_id, period) in all_periods {
        if period.status
            == crate::state::voting::types::VotingPeriodStatus::Closed
        {
            let votes = state
                .voting()
                .databases()
                .get_votes_for_period(rwtxn, period_id)?;

            if !votes.is_empty() {
                let existing_outcomes = state
                    .voting()
                    .databases()
                    .get_consensus_outcomes_for_period(rwtxn, period_id)?;

                if existing_outcomes.is_empty() {
                    tracing::info!(
                        "Protocol: Automatically calculating consensus for period {} at block height {} (period ended at timestamp {})",
                        period_id.0,
                        height,
                        period.end_timestamp
                    );

                    state.voting().calculate_and_store_consensus(
                        rwtxn,
                        period_id,
                        state,
                        mainchain_timestamp,
                        height as u64,
                        state.slots(),
                    )?;

                    tracing::info!(
                        "Protocol: Successfully calculated consensus for period {}",
                        period_id.0
                    );
                }
            }
        }
    }

    {
        let payout_results =
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

    for (idx, filled_tx) in filled_txs.iter().enumerate() {
        if skipped_tx_indices.contains(&idx) {
            continue;
        }
        apply_utxo_changes(state, rwtxn, filled_tx)?;
    }

    state_update.apply_all_changes(state, rwtxn, height)?;

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

    if height > 0 {
        state
            .slots()
            .rollback_slot_states_to_height(rwtxn, (height - 1) as u64)?;

        tracing::info!(
            "Rolled back slot states to height {} during reorg",
            height - 1
        );
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
    )?;

    let claiming_txid = filled_tx.transaction.txid();

    state.slots().claim_slot(
        rwtxn,
        slot_id,
        decision,
        claiming_txid,
        mainchain_timestamp,
        Some(block_height),
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
        )?;

        state.slots().claim_slot(
            rwtxn,
            slot_id,
            decision,
            claiming_txid,
            mainchain_timestamp,
            Some(block_height),
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

/// Revert a trade transaction (buy or sell).
///
/// Reverting a trade means:
/// - For buy: Remove shares from trader, remove pending treasury/fee UTXOs
/// - For sell: Add shares back to trader, delete payout UTXO
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
    let market_id_bytes = *trade.market_id.as_bytes();

    if is_buy {
        state.markets().revert_share_trade(
            rwtxn,
            &trade.trader,
            trade.market_id.clone(),
            trade.outcome_index,
            trade.shares_abs(),
            height as u64,
        )?;

        let txid = filled_tx.txid();
        for (vout, output) in filled_tx.outputs().iter().enumerate() {
            if let OutputContent::MarketFunds {
                market_id, is_fee, ..
            } = &output.content
                && *market_id == market_id_bytes
            {
                let outpoint = OutPoint::Regular {
                    txid,
                    vout: vout as u32,
                };
                state.markets().remove_pending_market_funds_utxo(
                    rwtxn,
                    &trade.market_id,
                    &outpoint,
                    *is_fee,
                )?;
            }
        }
    } else {
        state.markets().add_shares_to_account(
            rwtxn,
            &trade.trader,
            trade.market_id.clone(),
            trade.outcome_index,
            trade.shares_abs(),
            height as u64,
        )?;

        let payout_outpoint = StateUpdate::generate_sell_payout_outpoint(
            &trade.market_id,
            &trade.trader,
            filled_tx.txid().0,
        );
        // Ignore result - UTXO might not exist if consolidation failed
        drop(state.delete_utxo_with_address_index(rwtxn, &payout_outpoint));
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

        if let FilledOutputContent::MarketFunds {
            market_id, is_fee, ..
        } = &filled_output.content
        {
            let market_id = MarketId::new(*market_id);
            state.markets().add_pending_market_funds_utxo(
                rwtxn, &market_id, &outpoint, *is_fee,
            )?;
        }
    }

    Ok(())
}

/// Apply a trade transaction (buy or sell shares) to state update.
///
/// This unified function replaces the separate apply_market_buy and apply_market_sell.
/// The sign of shares determines direction: positive = buy, negative = sell.
///
/// For buys: Treasury and fee values are tracked via explicit transaction outputs
/// For sells: Creates pending payout (to be created during UTXO consolidation)
///
/// Returns `TradeApplyResult::Skipped` for slippage failures (soft-fail behavior).
/// The transaction remains in mempool for retry in future blocks.
fn apply_trade(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    state_update: &mut StateUpdate,
    _height: u32,
) -> Result<TradeApplyResult, Error> {
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

    if new_shares[outcome_index] < 0.0 {
        return Err(Error::InvalidTransaction {
            reason: format!(
                "Trade would result in negative market shares: {} for outcome {}",
                new_shares[outcome_index], outcome_index
            ),
        });
    }

    let (volume_sats, fee_sats) = if is_buy {
        // Buy: cost = LMSR(current -> new)
        let base_cost =
            query_update_cost(market.shares(), &new_shares, market.b())
                .map_err(|e| Error::InvalidTransaction {
                    reason: format!("Failed to calculate trade cost: {e:?}"),
                })?;

        let buy_cost =
            trading::calculate_buy_cost(base_cost, market.trading_fee())
                .map_err(|e| Error::InvalidTransaction {
                    reason: format!("Buy cost calculation failed: {e}"),
                })?;

        // Slippage protection - soft-fail: skip transaction, don't fail the block
        if buy_cost.total_cost_sats > trade.limit_sats {
            return Ok(TradeApplyResult::Skipped {
                reason: format!(
                    "Buy cost {} sats (base: {}, fee: {}) exceeds max cost {} sats",
                    buy_cost.total_cost_sats,
                    buy_cost.base_cost_sats,
                    buy_cost.trading_fee_sats,
                    trade.limit_sats
                ),
            });
        }

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
            .unwrap_or(0.0);

        // Check pending share changes from earlier transactions in this block
        // to prevent double-spend of shares within the same block
        let pending_delta = state_update
            .share_account_changes
            .get(&(trade.trader, trade.market_id.clone()))
            .and_then(|outcomes| outcomes.get(&trade.outcome_index))
            .copied()
            .unwrap_or(0.0);

        // pending_delta is negative for pending sells, positive for pending buys
        let effective_owned = owned_shares + pending_delta;

        if effective_owned < shares_abs {
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
            query_update_cost(&new_shares, market.shares(), market.b())
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

        // Slippage protection - soft-fail: skip transaction, don't fail the block
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
            shares_sold: shares_abs,
            outcome_index: trade.outcome_index,
            transaction_id: filled_tx.transaction.txid().0,
        });

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
        // This test verifies that the effective ownership calculation
        // correctly accounts for pending share changes in the same block,
        // preventing double-spend of shares.

        let mut state_update = StateUpdate::new();
        let trader = Address::ALL_ZEROS;
        let market_id = MarketId::new([1u8; 6]);
        let outcome_index: u32 = 0;

        // Simulate: trader owns 100 shares in database
        let owned_shares_from_db = 100.0;

        // Simulate: first sell of 60 shares already recorded in state_update
        // (negative delta because selling reduces shares)
        state_update.add_share_account_change(
            trader,
            market_id.clone(),
            outcome_index,
            -60.0, // Sold 60 shares
        );

        // Calculate effective ownership (same logic as in apply_trade)
        let pending_delta = state_update
            .share_account_changes
            .get(&(trader, market_id.clone()))
            .and_then(|outcomes| outcomes.get(&outcome_index))
            .copied()
            .unwrap_or(0.0);

        let effective_owned = owned_shares_from_db + pending_delta;

        // Effective ownership should be 100 - 60 = 40
        assert_eq!(effective_owned, 40.0);

        // Attempting to sell 50 more shares should fail
        // because effective_owned (40) < shares_to_sell (50)
        let shares_to_sell = 50.0;
        assert!(
            effective_owned < shares_to_sell,
            "Double-spend protection: should reject sell of {} when only {} effectively owned",
            shares_to_sell,
            effective_owned
        );

        // But selling 40 or less should be allowed
        let valid_sell = 40.0;
        assert!(
            effective_owned >= valid_sell,
            "Should allow selling {} when {} effectively owned",
            valid_sell,
            effective_owned
        );
    }

    #[test]
    fn test_buy_then_sell_same_block_allowed() {
        // Verify that buying shares and then selling some of them
        // in the same block works correctly.

        let mut state_update = StateUpdate::new();
        let trader = Address::ALL_ZEROS;
        let market_id = MarketId::new([2u8; 6]);
        let outcome_index: u32 = 1;

        // Trader starts with 0 shares in database
        let owned_shares_from_db = 0.0;

        // First transaction: buy 100 shares (positive delta)
        state_update.add_share_account_change(
            trader,
            market_id.clone(),
            outcome_index,
            100.0,
        );

        // Calculate effective ownership after buy
        let pending_delta = state_update
            .share_account_changes
            .get(&(trader, market_id.clone()))
            .and_then(|outcomes| outcomes.get(&outcome_index))
            .copied()
            .unwrap_or(0.0);

        let effective_owned = owned_shares_from_db + pending_delta;
        assert_eq!(effective_owned, 100.0);

        // Should be able to sell up to 100 shares
        let shares_to_sell = 50.0;
        assert!(
            effective_owned >= shares_to_sell,
            "Should allow selling {} after buying 100 in same block",
            shares_to_sell
        );
    }

    #[test]
    fn test_multiple_sells_different_outcomes_same_block() {
        // Verify that selling shares of different outcomes
        // in the same block works independently.

        let mut state_update = StateUpdate::new();
        let trader = Address::ALL_ZEROS;
        let market_id = MarketId::new([3u8; 6]);

        // Trader owns 100 shares of outcome 0 and 100 shares of outcome 1
        let owned_outcome_0 = 100.0;
        let owned_outcome_1 = 100.0;

        // Sell 80 shares of outcome 0
        state_update.add_share_account_change(
            trader,
            market_id.clone(),
            0,
            -80.0,
        );

        // Check effective ownership for outcome 0
        let pending_delta_0 = state_update
            .share_account_changes
            .get(&(trader, market_id.clone()))
            .and_then(|outcomes| outcomes.get(&0))
            .copied()
            .unwrap_or(0.0);
        let effective_owned_0 = owned_outcome_0 + pending_delta_0;
        assert_eq!(effective_owned_0, 20.0);

        // Check effective ownership for outcome 1 (should be unaffected)
        let pending_delta_1 = state_update
            .share_account_changes
            .get(&(trader, market_id.clone()))
            .and_then(|outcomes| outcomes.get(&1))
            .copied()
            .unwrap_or(0.0);
        let effective_owned_1 = owned_outcome_1 + pending_delta_1;
        assert_eq!(effective_owned_1, 100.0);

        // Should still be able to sell 100 shares of outcome 1
        assert!(effective_owned_1 >= 100.0);
    }

    #[test]
    fn test_cumulative_sells_same_outcome_same_block() {
        // Verify that multiple sells of the same outcome accumulate correctly.

        let mut state_update = StateUpdate::new();
        let trader = Address::ALL_ZEROS;
        let market_id = MarketId::new([4u8; 6]);
        let outcome_index: u32 = 0;

        // Trader owns 100 shares
        let owned_shares_from_db = 100.0;

        // First sell: 30 shares
        state_update.add_share_account_change(
            trader,
            market_id.clone(),
            outcome_index,
            -30.0,
        );

        // Second sell: 40 shares
        state_update.add_share_account_change(
            trader,
            market_id.clone(),
            outcome_index,
            -40.0,
        );

        // Total pending delta should be -70
        let pending_delta = state_update
            .share_account_changes
            .get(&(trader, market_id.clone()))
            .and_then(|outcomes| outcomes.get(&outcome_index))
            .copied()
            .unwrap_or(0.0);

        assert_eq!(pending_delta, -70.0);

        let effective_owned = owned_shares_from_db + pending_delta;
        assert_eq!(effective_owned, 30.0);

        // Third sell of 40 should fail (only 30 effectively owned)
        let third_sell = 40.0;
        assert!(
            effective_owned < third_sell,
            "Should reject third sell: {} > {} effective",
            third_sell,
            effective_owned
        );

        // But selling 30 should work
        assert!(effective_owned >= 30.0);
    }
}
