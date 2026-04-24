use fallible_iterator::FallibleIterator;
use heed::types::SerdeBincode;
use ndarray::{Array, Ix1};
use sneed::{DatabaseUnique, Env, RoTxn, RwTxn};
use std::collections::{HashMap, HashSet};

use crate::state::Error;
use crate::state::UtxoManager;
use crate::state::decisions::{Decision, DecisionId};
use crate::types::{Address, GetBitcoinValue, OutPoint, OutPointKey};

use super::market::Market;
use super::payouts::{
    calculate_fee_distribution, generate_share_payout_outpoint,
};
use super::types::{
    MarketError, MarketId, MarketPayoutSummary, MarketState, ShareAccount,
};

#[derive(Clone)]
#[allow(clippy::type_complexity)]
pub struct MarketsDatabase {
    markets: DatabaseUnique<SerdeBincode<[u8; 6]>, SerdeBincode<Market>>,
    state_index:
        DatabaseUnique<SerdeBincode<MarketState>, SerdeBincode<Vec<MarketId>>>,
    expiry_index:
        DatabaseUnique<SerdeBincode<u32>, SerdeBincode<Vec<MarketId>>>,
    decision_index:
        DatabaseUnique<SerdeBincode<DecisionId>, SerdeBincode<Vec<MarketId>>>,
    share_accounts:
        DatabaseUnique<SerdeBincode<Address>, SerdeBincode<ShareAccount>>,
    /// is_fee=false for treasury, is_fee=true for author fees
    market_funds_utxos:
        DatabaseUnique<SerdeBincode<([u8; 6], bool)>, SerdeBincode<OutPoint>>,
    /// Cumulative pending treasury deposits from unconfirmed amplify_beta
    /// transactions. Since beta = treasury / ln(num_outcomes), this delta
    /// is sufficient to derive the effective beta for pricing on reads.
    /// Cleared when the corresponding block is applied.
    market_mempool_treasury_delta:
        DatabaseUnique<SerdeBincode<[u8; 6]>, SerdeBincode<u64>>,
    /// Pending per-outcome share totals from unconfirmed trades.
    /// Cleared when the corresponding block is applied.
    mempool_shares:
        DatabaseUnique<SerdeBincode<[u8; 6]>, SerdeBincode<Vec<i64>>>,
}

impl MarketsDatabase {
    pub const NUM_DBS: u32 = 8;

    pub fn new(env: &Env, rwtxn: &mut RwTxn) -> Result<Self, Error> {
        let markets = DatabaseUnique::create(env, rwtxn, "markets")?;
        let state_index =
            DatabaseUnique::create(env, rwtxn, "markets_by_state")?;
        let expiry_index =
            DatabaseUnique::create(env, rwtxn, "markets_by_expiry")?;
        let decision_index =
            DatabaseUnique::create(env, rwtxn, "markets_by_decision")?;
        let share_accounts =
            DatabaseUnique::create(env, rwtxn, "share_accounts")?;
        let market_funds_utxos =
            DatabaseUnique::create(env, rwtxn, "market_funds_utxos")?;
        let market_mempool_treasury_delta = DatabaseUnique::create(
            env,
            rwtxn,
            "market_mempool_treasury_delta",
        )?;
        let mempool_shares =
            DatabaseUnique::create(env, rwtxn, "mempool_shares")?;

        Ok(MarketsDatabase {
            markets,
            state_index,
            expiry_index,
            decision_index,
            share_accounts,
            market_funds_utxos,
            market_mempool_treasury_delta,
            mempool_shares,
        })
    }

    pub fn add_market(
        &self,
        txn: &mut RwTxn,
        market: &Market,
    ) -> Result<(), Error> {
        self.markets.put(txn, market.id.as_bytes(), market)?;

        self.update_state_index(txn, &market.id, None, Some(market.state()))?;

        if let Some(expires_at) = market.expires_at_height {
            self.update_expiry_index(txn, &market.id, None, Some(expires_at))?;
        }

        for &decision_id in &market.decision_ids {
            self.update_decision_index(txn, &market.id, decision_id, true)?;
        }

        Ok(())
    }

    pub fn delete_market(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
    ) -> Result<bool, Error> {
        // Get the market first to know what indexes to clean up
        let Some(market) = self.markets.try_get(txn, market_id.as_bytes())?
        else {
            return Ok(false);
        };

        self.update_state_index(txn, market_id, Some(market.state()), None)?;

        if let Some(expires_at) = market.expires_at_height {
            self.update_expiry_index(txn, market_id, Some(expires_at), None)?;
        }

        for &decision_id in &market.decision_ids {
            self.update_decision_index(txn, market_id, decision_id, false)?;
        }

        self.market_funds_utxos
            .delete(txn, &(*market_id.as_bytes(), false))?;
        self.market_funds_utxos
            .delete(txn, &(*market_id.as_bytes(), true))?;

        self.markets.delete(txn, market_id.as_bytes())?;

        Ok(true)
    }

    pub fn get_market(
        &self,
        txn: &RoTxn,
        market_id: &MarketId,
    ) -> Result<Option<Market>, Error> {
        Ok(self.markets.try_get(txn, market_id.as_bytes())?)
    }

    pub fn get_all_markets(&self, txn: &RoTxn) -> Result<Vec<Market>, Error> {
        let markets = self
            .markets
            .iter(txn)?
            .map(|(_, market)| Ok(market))
            .collect()?;
        Ok(markets)
    }

    pub fn get_markets_batch(
        &self,
        txn: &RoTxn,
        market_ids: &[MarketId],
    ) -> Result<HashMap<MarketId, Market>, Error> {
        if market_ids.is_empty() {
            return Ok(HashMap::new());
        }

        const BATCH_THRESHOLD: usize = 3;
        if market_ids.len() < BATCH_THRESHOLD {
            let mut markets = HashMap::with_capacity(market_ids.len());

            for market_id in market_ids {
                if let Some(market) = self.get_market(txn, market_id)? {
                    markets.insert(*market_id, market);
                }
            }

            return Ok(markets);
        }

        let market_id_set: HashSet<_> = market_ids.iter().collect();
        let mut markets = HashMap::with_capacity(market_ids.len());
        let mut found_count = 0;

        let market_iter = self.markets.iter(txn).map_err(|e| {
            Error::DatabaseError(format!("Market batch iteration failed: {e}"))
        })?;

        let mut market_iter = market_iter;
        while let Some(item) = market_iter.next().map_err(|e| {
            Error::DatabaseError(format!(
                "Market batch iteration item failed: {e}"
            ))
        })? {
            let (market_id_bytes, market) = item;

            let market_id = MarketId::new(market_id_bytes);

            if market_id_set.contains(&market_id) {
                markets.insert(market_id, market);
                found_count += 1;

                if found_count >= market_ids.len() {
                    break;
                }
            }
        }

        tracing::debug!(
            "Batch loaded {}/{} requested markets using optimized iteration",
            found_count,
            market_ids.len()
        );

        Ok(markets)
    }

    pub fn get_markets_by_state(
        &self,
        txn: &RoTxn,
        state: MarketState,
    ) -> Result<Vec<Market>, Error> {
        let market_ids =
            self.state_index.try_get(txn, &state)?.unwrap_or_default();

        let mut markets = Vec::with_capacity(market_ids.len());
        for market_id in market_ids {
            if let Some(market) = self.get_market(txn, &market_id)? {
                markets.push(market);
            }
        }

        Ok(markets)
    }

    pub fn update_market(
        &self,
        txn: &mut RwTxn,
        market: &Market,
    ) -> Result<(), Error> {
        let old_market = self.get_market(txn, &market.id)?;
        if let Some(ref old) = old_market {
            crate::validation::MarketStateValidator::validate_market_state_transition(
                old.state(),
                market.state(),
            )?;
        }
        self.write_market_and_indexes(txn, market, old_market)
    }

    /// Like `update_market` but bypasses the state-machine transition check.
    /// Used only on reorg to restore a market's pre-ossification state, where
    /// the forward-direction transition (Ossified → Trading) is illegal but
    /// legitimate in undo.
    pub fn restore_market(
        &self,
        txn: &mut RwTxn,
        market: &Market,
    ) -> Result<(), Error> {
        let old_market = self.get_market(txn, &market.id)?;
        self.write_market_and_indexes(txn, market, old_market)
    }

    fn write_market_and_indexes(
        &self,
        txn: &mut RwTxn,
        market: &Market,
        old_market: Option<Market>,
    ) -> Result<(), Error> {
        let old_decisions: HashSet<_> = old_market
            .as_ref()
            .map(|m| m.decision_ids.iter().cloned().collect())
            .unwrap_or_default();

        self.markets
            .put(txn, market.id.as_bytes(), market)
            .map_err(|e| {
                tracing::error!(
                    "Failed to update primary market storage for market {}: {}",
                    market.id,
                    e
                );
                Error::DatabaseError(format!(
                    "Primary market update failed: {e}"
                ))
            })?;

        if let Some(old) = old_market {
            if old.state() != market.state() {
                self.update_state_index(
                    txn,
                    &market.id,
                    Some(old.state()),
                    Some(market.state()),
                )
                .map_err(|e| {
                    tracing::error!(
                        "Failed to update state index for market {}: {}",
                        market.id,
                        e
                    );
                    Error::DatabaseError(format!(
                        "State index update failed: {e}"
                    ))
                })?;
            }

            if old.expires_at_height != market.expires_at_height {
                self.update_expiry_index(
                    txn,
                    &market.id,
                    old.expires_at_height,
                    market.expires_at_height,
                )
                .map_err(|e| {
                    tracing::error!(
                        "Failed to update expiry index for market {}: {}",
                        market.id,
                        e
                    );
                    Error::DatabaseError(format!(
                        "Expiry index update failed: {e}"
                    ))
                })?;
            }

            let new_entrys: HashSet<_> =
                market.decision_ids.iter().cloned().collect();

            for decision_id in old_decisions.difference(&new_entrys) {
                self.update_decision_index(
                    txn,
                    &market.id,
                    *decision_id,
                    false,
                )
                .map_err(|e| {
                    tracing::error!(
                        "Failed to remove market {} from decision index {}: {}",
                        market.id,
                        hex::encode(decision_id.as_bytes()),
                        e
                    );
                    Error::DatabaseError(format!(
                        "Decision index removal failed: {e}"
                    ))
                })?;
            }

            for decision_id in new_entrys.difference(&old_decisions) {
                self.update_decision_index(txn, &market.id, *decision_id, true)
                    .map_err(|e| {
                        tracing::error!(
                            "Failed to add market {} to decision index {}: {}",
                            market.id,
                            hex::encode(decision_id.as_bytes()),
                            e
                        );
                        Error::DatabaseError(format!(
                            "Decision index addition failed: {e}"
                        ))
                    })?;
            }
        }

        tracing::debug!(
            "Successfully updated market {} with all indexes",
            market.id
        );
        Ok(())
    }

    /// Transition resolved markets directly from Trading -> Ossified with automatic share payouts.
    /// Called every block from connect_body. Checks ALL trading markets to see if
    /// their decisions are now Resolved in the DecisionStateHistory database.
    #[allow(clippy::type_complexity)]
    pub fn transition_and_payout_resolved_markets(
        &self,
        txn: &mut RwTxn,
        state: &crate::state::State,
        decisions_db: &crate::state::decisions::Dbs,
        current_height: u32,
    ) -> Result<
        (
            Vec<(MarketId, MarketPayoutSummary)>,
            Vec<crate::state::undo::OssificationUndoEntry>,
        ),
        Error,
    > {
        let mut results = Vec::new();
        let mut undo_entries = Vec::new();
        let trading_markets =
            self.get_markets_by_state(txn, MarketState::Trading)?;

        for mut market in trading_markets {
            if market.decision_ids.is_empty() {
                continue;
            }

            let mut all_decisions_resolved = true;
            let mut decision_outcomes: std::collections::HashMap<
                DecisionId,
                f64,
            > = std::collections::HashMap::new();
            let mut decisions: HashMap<DecisionId, Decision> = HashMap::new();

            for decision_id in &market.decision_ids {
                let decision_state = decisions_db
                    .get_decision_current_state(txn, *decision_id)?;

                if decision_state
                    != crate::state::decisions::DecisionState::Resolved
                {
                    all_decisions_resolved = false;
                    break;
                }

                let voting_period_id =
                    crate::state::voting::types::VotingPeriodId::new(
                        decision_id.voting_period(),
                    );
                if let Some(outcome) =
                    state.voting().databases().get_consensus_outcome(
                        txn,
                        voting_period_id,
                        *decision_id,
                    )?
                {
                    decision_outcomes.insert(*decision_id, outcome);
                }

                if let Some(entry) =
                    decisions_db.get_decision_entry(txn, *decision_id)?
                    && let Some(decision) = entry.decision
                {
                    decisions.insert(*decision_id, decision);
                }
            }

            if all_decisions_resolved {
                let market_id = market.id;

                // Capture pre-ossification state for undo
                let pre_ossification_market = market.clone();

                // Capture treasury and fee UTXOs before they are consumed
                let treasury_utxo = if let Some(outpoint) =
                    self.get_market_funds_utxo(txn, &market_id, false)?
                {
                    state
                        .utxos
                        .try_get(txn, &OutPointKey::from_outpoint(&outpoint))?
                        .map(|output| (outpoint, output))
                } else {
                    None
                };
                let fee_utxo = if let Some(outpoint) =
                    self.get_market_funds_utxo(txn, &market_id, true)?
                {
                    state
                        .utxos
                        .try_get(txn, &OutPointKey::from_outpoint(&outpoint))?
                        .map(|output| (outpoint, output))
                } else {
                    None
                };

                let final_prices = market
                    .calculate_final_prices(&decision_outcomes, &decisions)
                    .map_err(|e| {
                        Error::DatabaseError(format!(
                            "Failed to calculate final prices: {e:?}"
                        ))
                    })?;

                market
                    .update_state(
                        current_height,
                        None,
                        None,
                        Some(final_prices),
                    )
                    .map_err(|e| {
                        Error::DatabaseError(format!(
                            "Failed to set final prices: {e:?}"
                        ))
                    })?;

                let payout_summary = self.calculate_share_payouts(
                    txn,
                    state,
                    &market,
                    current_height,
                    &decision_outcomes,
                    &decisions,
                )?;

                // Apply payouts
                self.apply_automatic_share_payouts(
                    state,
                    txn,
                    &payout_summary,
                    current_height,
                )?;

                market
                    .update_state(
                        current_height,
                        Some(MarketState::Ossified),
                        None,
                        None,
                    )
                    .map_err(|e| {
                        Error::DatabaseError(format!(
                            "Failed to ossify market: {e:?}"
                        ))
                    })?;

                self.update_market(txn, &market)?;

                undo_entries.push(crate::state::undo::OssificationUndoEntry {
                    pre_ossification_market,
                    payout_summary: payout_summary.clone(),
                    treasury_utxo,
                    fee_utxo,
                });

                results.push((market_id, payout_summary));
            }
        }

        Ok((results, undo_entries))
    }

    pub fn cancel_market(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
    ) -> Result<(), MarketError> {
        let mut market = self
            .get_market(txn, market_id)
            .map_err(|e| MarketError::DatabaseError(e.to_string()))?
            .ok_or(MarketError::MarketNotFound { id: *market_id })?;
        market.cancel_market(None, 0)?;
        self.update_market(txn, &market)
            .map_err(|e| MarketError::DatabaseError(e.to_string()))
    }

    pub fn invalidate_market(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
    ) -> Result<(), MarketError> {
        let mut market = self
            .get_market(txn, market_id)
            .map_err(|e| MarketError::DatabaseError(e.to_string()))?
            .ok_or(MarketError::MarketNotFound { id: *market_id })?;
        market.invalidate_market(None, 0)?;
        self.update_market(txn, &market)
            .map_err(|e| MarketError::DatabaseError(e.to_string()))
    }

    pub fn get_mempool_shares(
        &self,
        rotxn: &RoTxn,
        market_id: &MarketId,
    ) -> Result<Option<Array<i64, Ix1>>, Error> {
        match self.mempool_shares.try_get(rotxn, &market_id.0)? {
            Some(shares) => Ok(Some(Array::from_vec(shares))),
            None => Ok(None),
        }
    }

    pub fn put_mempool_shares(
        &self,
        rwtxn: &mut RwTxn,
        market_id: &MarketId,
        shares: &Array<i64, Ix1>,
    ) -> Result<(), Error> {
        self.mempool_shares
            .put(rwtxn, &market_id.0, &shares.to_vec())?;
        Ok(())
    }

    pub fn clear_mempool_shares(
        &self,
        rwtxn: &mut RwTxn,
        market_id: &MarketId,
    ) -> Result<(), Error> {
        self.mempool_shares.delete(rwtxn, &market_id.0)?;
        Ok(())
    }

    pub fn get_mempool_treasury_delta(
        &self,
        rotxn: &RoTxn,
        market_id: &MarketId,
    ) -> Result<u64, Error> {
        Ok(self
            .market_mempool_treasury_delta
            .try_get(rotxn, market_id.as_bytes())?
            .unwrap_or(0))
    }

    pub fn put_mempool_treasury_delta(
        &self,
        rwtxn: &mut RwTxn,
        market_id: &MarketId,
        delta: u64,
    ) -> Result<(), Error> {
        self.market_mempool_treasury_delta.put(
            rwtxn,
            market_id.as_bytes(),
            &delta,
        )?;
        Ok(())
    }

    pub fn clear_mempool_treasury_delta(
        &self,
        rwtxn: &mut RwTxn,
        market_id: &MarketId,
    ) -> Result<(), Error> {
        self.market_mempool_treasury_delta
            .delete(rwtxn, market_id.as_bytes())?;
        Ok(())
    }

    fn update_state_index(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
        old_state: Option<MarketState>,
        new_state: Option<MarketState>,
    ) -> Result<(), Error> {
        if let Some(old) = old_state {
            let mut market_ids =
                self.state_index.try_get(txn, &old)?.unwrap_or_default();
            market_ids.retain(|id| id != market_id);
            if market_ids.is_empty() {
                self.state_index.delete(txn, &old)?;
            } else {
                self.state_index.put(txn, &old, &market_ids)?;
            }
        }

        if let Some(new) = new_state {
            let mut market_ids =
                self.state_index.try_get(txn, &new)?.unwrap_or_default();
            if !market_ids.contains(market_id) {
                market_ids.push(*market_id);
                self.state_index.put(txn, &new, &market_ids)?;
            }
        }

        Ok(())
    }

    fn update_expiry_index(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
        old_expiry: Option<u32>,
        new_expiry: Option<u32>,
    ) -> Result<(), Error> {
        if let Some(old) = old_expiry {
            let mut market_ids =
                self.expiry_index.try_get(txn, &old)?.unwrap_or_default();
            market_ids.retain(|id| id != market_id);
            if market_ids.is_empty() {
                self.expiry_index.delete(txn, &old)?;
            } else {
                self.expiry_index.put(txn, &old, &market_ids)?;
            }
        }

        if let Some(new) = new_expiry {
            let mut market_ids =
                self.expiry_index.try_get(txn, &new)?.unwrap_or_default();
            if !market_ids.contains(market_id) {
                market_ids.push(*market_id);
                self.expiry_index.put(txn, &new, &market_ids)?;
            }
        }

        Ok(())
    }

    fn update_decision_index(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
        decision_id: DecisionId,
        add: bool,
    ) -> Result<(), Error> {
        let mut market_ids = self
            .decision_index
            .try_get(txn, &decision_id)?
            .unwrap_or_default();

        if add {
            if !market_ids.contains(market_id) {
                market_ids.push(*market_id);
                self.decision_index.put(txn, &decision_id, &market_ids)?;
            }
        } else {
            market_ids.retain(|id| id != market_id);
            if market_ids.is_empty() {
                self.decision_index.delete(txn, &decision_id)?;
            } else {
                self.decision_index.put(txn, &decision_id, &market_ids)?;
            }
        }

        Ok(())
    }

    pub fn add_shares_to_account(
        &self,
        txn: &mut RwTxn,
        address: &Address,
        market_id: MarketId,
        outcome_index: u32,
        shares: i64,
        height: u32,
    ) -> Result<(), Error> {
        let mut account = self
            .share_accounts
            .try_get(txn, address)?
            .unwrap_or_else(ShareAccount::new);

        account.add_shares(market_id, outcome_index, shares, height);

        self.share_accounts.put(txn, address, &account)?;

        Ok(())
    }

    pub fn remove_shares_from_account(
        &self,
        txn: &mut RwTxn,
        address: &Address,
        market_id: &MarketId,
        outcome_index: u32,
        shares: i64,
        height: u32,
    ) -> Result<(), Error> {
        let mut account = self
            .share_accounts
            .try_get(txn, address)?
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "No share account found for address".to_string(),
            })?;

        account
            .remove_shares(market_id, outcome_index, shares, height)
            .map_err(|_| Error::InvalidTransaction {
                reason: "Insufficient shares for sell transaction".to_string(),
            })?;

        if account.positions.is_empty() {
            self.share_accounts.delete(txn, address)?;
        } else {
            self.share_accounts.put(txn, address, &account)?;
        }

        Ok(())
    }

    pub fn get_user_share_account(
        &self,
        txn: &RoTxn,
        address: &Address,
    ) -> Result<Option<ShareAccount>, Error> {
        Ok(self.share_accounts.try_get(txn, address)?)
    }

    /// Get all share accounts from the database (for debugging)
    pub fn get_all_share_accounts(
        &self,
        txn: &RoTxn,
    ) -> Result<super::super::type_aliases::AllShareAccounts, Error> {
        let mut result = Vec::new();
        let mut iter = self.share_accounts.iter(txn)?;
        while let Some((address, account)) = iter.next()? {
            let positions: Vec<(MarketId, u32, i64)> = account
                .positions
                .into_iter()
                .map(|((market_id, outcome_index), shares)| {
                    (market_id, outcome_index, shares)
                })
                .collect();
            if !positions.is_empty() {
                result.push((address, positions));
            }
        }
        Ok(result)
    }

    pub fn get_user_share_positions(
        &self,
        txn: &RoTxn,
        address: &Address,
    ) -> Result<Vec<(MarketId, u32, i64)>, Error> {
        if let Some(account) = self.get_user_share_account(txn, address)? {
            Ok(account
                .positions
                .into_iter()
                .map(|((market_id, outcome_index), shares)| {
                    (market_id, outcome_index, shares)
                })
                .collect())
        } else {
            Ok(Vec::new())
        }
    }

    pub fn get_market_user_positions(
        &self,
        txn: &RoTxn,
        address: &Address,
        market_id: &MarketId,
    ) -> Result<Vec<(u32, i64)>, Error> {
        if let Some(account) = self.get_user_share_account(txn, address)? {
            Ok(account
                .positions
                .into_iter()
                .filter(|((pos_market_id, _), _)| pos_market_id == market_id)
                .map(|((_, outcome_index), shares)| (outcome_index, shares))
                .collect())
        } else {
            Ok(Vec::new())
        }
    }

    pub fn get_wallet_positions_for_market_outcome(
        &self,
        txn: &RoTxn,
        addresses: &std::collections::HashSet<Address>,
        market_id: &MarketId,
        outcome_index: u32,
    ) -> Result<std::collections::HashMap<Address, i64>, Error> {
        let mut result = std::collections::HashMap::new();
        for address in addresses {
            if let Some(account) = self.get_user_share_account(txn, address)?
                && let Some(&shares) =
                    account.positions.get(&(*market_id, outcome_index))
                && shares > 0
            {
                result.insert(*address, shares);
            }
        }
        Ok(result)
    }

    pub fn revert_share_trade(
        &self,
        txn: &mut RwTxn,
        address: &Address,
        market_id: MarketId,
        outcome_index: u32,
        shares_traded: i64,
        height: u32,
    ) -> Result<(), Error> {
        self.remove_shares_from_account(
            txn,
            address,
            &market_id,
            outcome_index,
            shares_traded,
            height,
        )
    }

    pub fn get_account_nonce(
        &self,
        txn: &RoTxn,
        address: &Address,
    ) -> Result<u64, Error> {
        if let Some(account) = self.share_accounts.try_get(txn, address)? {
            Ok(account.nonce)
        } else {
            Ok(0)
        }
    }

    pub fn get_account_nonces(
        &self,
        txn: &RoTxn,
        address: &Address,
    ) -> Result<(u64, u64), Error> {
        if let Some(account) = self.share_accounts.try_get(txn, address)? {
            Ok((account.nonce, account.trade_nonce))
        } else {
            Ok((0, 0))
        }
    }

    pub fn verify_share_invariant(
        &self,
        txn: &RoTxn,
        market_id: &MarketId,
    ) -> Result<(), Error> {
        let market = self.get_market(txn, market_id)?.ok_or_else(|| {
            Error::InvalidTransaction {
                reason: format!(
                    "Market {market_id:?} not found for invariant check"
                ),
            }
        })?;

        let share_count = market.shares().len();
        let mut account_totals = vec![0i64; share_count];

        let mut iter = self.share_accounts.iter(txn)?;
        while let Some((_address, account)) = iter.next()? {
            for ((mid, outcome_index), shares) in &account.positions {
                if mid == market_id {
                    let idx = *outcome_index as usize;
                    if idx < share_count {
                        account_totals[idx] += shares;
                    }
                }
            }
        }

        for (i, (market_shares, account_shares)) in market
            .shares()
            .iter()
            .zip(account_totals.iter())
            .enumerate()
        {
            if market_shares != account_shares {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Share invariant violation for market \
                         {market_id:?} outcome {i}: \
                         market.shares={market_shares}, \
                         sum(accounts)={account_shares}"
                    ),
                });
            }
        }

        Ok(())
    }

    pub fn get_shareholders_for_market(
        &self,
        txn: &RoTxn,
        market_id: &MarketId,
    ) -> Result<super::super::type_aliases::MarketShareholders, Error> {
        let mut shareholders = Vec::new();
        let mut iter = self.share_accounts.iter(txn)?;

        while let Some((address, account)) = iter.next()? {
            let positions_for_market: Vec<(u32, i64)> = account
                .positions
                .iter()
                .filter(|((mid, _), _)| mid == market_id)
                .map(|((_, outcome_index), shares)| (*outcome_index, *shares))
                .collect();

            if !positions_for_market.is_empty() {
                shareholders.push((address, positions_for_market));
            }
        }

        Ok(shareholders)
    }

    /// Payout formula: payout_i = (shares_i * final_price_i / total_weighted_shares) * treasury
    pub fn calculate_share_payouts(
        &self,
        txn: &RoTxn,
        state: &crate::state::State,
        market: &Market,
        block_height: u32,
        decision_outcomes: &HashMap<DecisionId, f64>,
        decisions: &HashMap<DecisionId, Decision>,
    ) -> Result<MarketPayoutSummary, Error> {
        let treasury_sats =
            self.get_market_funds_sats(txn, state, &market.id, false)?;
        let final_prices = market.final_prices();

        let shareholders = self.get_shareholders_for_market(txn, &market.id)?;

        // Calculate total weighted shares for normalization
        let total_weighted_shares: f64 = shareholders
            .iter()
            .flat_map(|(_, positions)| positions.iter())
            .map(|(outcome_index, shares)| {
                let final_price = final_prices[*outcome_index as usize];
                *shares as f64 * final_price
            })
            .sum();

        // If no winning positions, refund treasury to the market creator
        // rather than burning it.
        if total_weighted_shares <= 0.0 {
            let creator_refund =
                (treasury_sats > 0).then_some(super::types::CreatorRefund {
                    address: market.creator_address,
                    amount_sats: treasury_sats,
                });
            return Ok(MarketPayoutSummary {
                market_id: market.id,
                treasury_distributed: 0,
                total_fees_distributed: 0,
                shareholder_count: 0,
                payouts: Vec::new(),
                fee_payouts: Vec::new(),
                creator_refund,
                block_height,
            });
        }

        let participants: Vec<((Address, u32, i64, f64), f64)> = shareholders
            .into_iter()
            .flat_map(|(address, positions)| {
                positions.into_iter().map(move |(outcome_index, shares)| {
                    let final_price = final_prices[outcome_index as usize];
                    let weighted_shares = shares as f64 * final_price;
                    (
                        (address, outcome_index, shares, final_price),
                        weighted_shares,
                    )
                })
            })
            .collect();

        let alloc_result =
            crate::math::allocation::allocate_proportionally_u64(
                participants,
                treasury_sats,
            )
            .map_err(|e| Error::InvalidTransaction {
                reason: format!("Payout allocation failed: {e}"),
            })?;

        let payouts: Vec<super::types::SharePayoutRecord> = alloc_result
            .allocations
            .into_iter()
            .map(
                |(
                    (address, outcome_index, shares, final_price),
                    payout_sats,
                )| {
                    super::types::SharePayoutRecord {
                        market_id: market.id,
                        address,
                        outcome_index,
                        shares_redeemed: shares,
                        final_price,
                        payout_sats,
                    }
                },
            )
            .collect();

        let total_distributed = alloc_result.total_allocated;

        let fee_sats =
            self.get_market_funds_sats(txn, state, &market.id, true)?;
        let fee_payouts = calculate_fee_distribution(
            txn,
            state,
            market,
            fee_sats,
            decision_outcomes,
            decisions,
        )?;
        let total_fees_distributed: u64 =
            fee_payouts.iter().map(|p| p.amount_sats).sum();

        Ok(MarketPayoutSummary {
            market_id: market.id,
            treasury_distributed: total_distributed,
            total_fees_distributed,
            shareholder_count: payouts.len() as u32,
            payouts,
            fee_payouts,
            creator_refund: None,
            block_height,
        })
    }

    pub fn apply_automatic_share_payouts(
        &self,
        state: &crate::state::State,
        txn: &mut RwTxn,
        payout_summary: &MarketPayoutSummary,
        block_height: u32,
    ) -> Result<(), Error> {
        use crate::types::{
            BitcoinOutputContent, FilledOutput, FilledOutputContent,
        };

        let mut sequence = 0u32;

        for payout in &payout_summary.payouts {
            let outpoint = generate_share_payout_outpoint(
                &payout.market_id,
                &payout.address,
                block_height,
                sequence,
            );

            let output = FilledOutput {
                address: payout.address,
                content: FilledOutputContent::Bitcoin(BitcoinOutputContent(
                    bitcoin::Amount::from_sat(payout.payout_sats),
                )),
                memo: vec![],
            };

            state.insert_utxo(txn, &outpoint, &output)?;

            self.remove_shares_from_account(
                txn,
                &payout.address,
                &payout.market_id,
                payout.outcome_index,
                payout.shares_redeemed,
                block_height,
            )?;

            sequence += 1;
        }

        for fee_payout in &payout_summary.fee_payouts {
            let fee_outpoint = generate_share_payout_outpoint(
                &payout_summary.market_id,
                &fee_payout.address,
                block_height,
                sequence,
            );

            let fee_output = FilledOutput {
                address: fee_payout.address,
                content: FilledOutputContent::Bitcoin(BitcoinOutputContent(
                    bitcoin::Amount::from_sat(fee_payout.amount_sats),
                )),
                memo: vec![],
            };

            state.insert_utxo(txn, &fee_outpoint, &fee_output)?;
            sequence += 1;
        }

        if let Some(refund) = &payout_summary.creator_refund {
            let refund_outpoint = generate_share_payout_outpoint(
                &payout_summary.market_id,
                &refund.address,
                block_height,
                sequence,
            );
            let refund_output = FilledOutput {
                address: refund.address,
                content: FilledOutputContent::Bitcoin(BitcoinOutputContent(
                    bitcoin::Amount::from_sat(refund.amount_sats),
                )),
                memo: vec![],
            };
            state.insert_utxo(txn, &refund_outpoint, &refund_output)?;
        }

        // Consume the Market UTXO (treasury is now distributed to shareholders)
        if let Some(market_utxo) =
            self.get_market_funds_utxo(txn, &payout_summary.market_id, false)?
        {
            state.delete_utxo(txn, &market_utxo)?;
            self.clear_market_funds_utxo(
                txn,
                &payout_summary.market_id,
                false,
            )?;
        }

        // Consume the Author Fee UTXO (fees now paid to market creator)
        if let Some(fee_utxo) =
            self.get_market_funds_utxo(txn, &payout_summary.market_id, true)?
        {
            state.delete_utxo(txn, &fee_utxo)?;
            self.clear_market_funds_utxo(txn, &payout_summary.market_id, true)?;
        }

        Ok(())
    }

    pub fn revert_automatic_share_payouts(
        &self,
        state: &crate::state::State,
        txn: &mut RwTxn,
        payout_summary: &MarketPayoutSummary,
        block_height: u32,
    ) -> Result<(), Error> {
        let mut sequence = 0u32;

        for payout in &payout_summary.payouts {
            let outpoint = generate_share_payout_outpoint(
                &payout.market_id,
                &payout.address,
                block_height,
                sequence,
            );

            state.delete_utxo(txn, &outpoint)?;

            // Restore shares to account
            self.add_shares_to_account(
                txn,
                &payout.address,
                payout.market_id,
                payout.outcome_index,
                payout.shares_redeemed,
                block_height,
            )?;

            sequence += 1;
        }

        for fee_payout in &payout_summary.fee_payouts {
            let fee_outpoint = generate_share_payout_outpoint(
                &payout_summary.market_id,
                &fee_payout.address,
                block_height,
                sequence,
            );

            state.delete_utxo(txn, &fee_outpoint)?;
            sequence += 1;
        }

        if let Some(refund) = &payout_summary.creator_refund {
            let refund_outpoint = generate_share_payout_outpoint(
                &payout_summary.market_id,
                &refund.address,
                block_height,
                sequence,
            );
            state.delete_utxo(txn, &refund_outpoint)?;
        }

        tracing::info!(
            "Reverted automatic share payouts for market {}: {} sats treasury + {} sats fees",
            payout_summary.market_id,
            payout_summary.treasury_distributed,
            payout_summary.total_fees_distributed,
        );

        Ok(())
    }

    pub fn get_market_funds_utxo(
        &self,
        txn: &RoTxn,
        market_id: &MarketId,
        is_fee: bool,
    ) -> Result<Option<OutPoint>, Error> {
        Ok(self
            .market_funds_utxos
            .try_get(txn, &(*market_id.as_bytes(), is_fee))?)
    }

    pub fn set_market_funds_utxo(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
        is_fee: bool,
        outpoint: &OutPoint,
    ) -> Result<(), Error> {
        self.market_funds_utxos.put(
            txn,
            &(*market_id.as_bytes(), is_fee),
            outpoint,
        )?;
        Ok(())
    }

    pub fn clear_market_funds_utxo(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
        is_fee: bool,
    ) -> Result<(), Error> {
        self.market_funds_utxos
            .delete(txn, &(*market_id.as_bytes(), is_fee))?;
        Ok(())
    }

    pub fn get_market_funds_sats(
        &self,
        txn: &RoTxn,
        state: &crate::state::State,
        market_id: &MarketId,
        is_fee: bool,
    ) -> Result<u64, Error> {
        match self.get_market_funds_utxo(txn, market_id, is_fee)? {
            Some(outpoint) => {
                let utxo = state
                    .utxos
                    .try_get(txn, &OutPointKey::from_outpoint(&outpoint))?
                    .ok_or(Error::NoUtxo { outpoint })?;
                Ok(utxo.get_bitcoin_value().to_sat())
            }
            None => Ok(0),
        }
    }
}
