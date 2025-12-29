use crate::math::safe_math::BasisPoints;
use crate::math::voting::constants::round_reputation;
use crate::state::{
    Error, State, UtxoManager, slots::SlotId, voting::types::VotingPeriodId,
};
use crate::types::{Address, FilledOutput, FilledOutputContent, OutPoint};
use sneed::RwTxn;
use std::collections::HashMap;
use tracing::{debug, info, warn};

fn generate_redistribution_outpoint(
    period_id: VotingPeriodId,
    voter_address: &Address,
    block_height: u64,
    sequence: u32,
) -> OutPoint {
    use blake3::Hasher;

    let mut hasher = Hasher::new();

    hasher.update(b"VOTECOIN_REDISTRIBUTION");
    hasher.update(&period_id.0.to_le_bytes());
    hasher.update(&block_height.to_le_bytes());
    hasher.update(&voter_address.0);
    hasher.update(&sequence.to_le_bytes());

    let hash = hasher.finalize();
    let merkle_root = crate::types::MerkleRoot::from(*hash.as_bytes());

    OutPoint::Coinbase {
        merkle_root,
        vout: sequence,
    }
}

#[derive(
    Clone, Debug, serde::Serialize, serde::Deserialize, utoipa::ToSchema,
)]
pub struct RedistributionTransfer {
    pub period_id: VotingPeriodId,
    pub from_address: Address,
    pub to_address: Address,
    pub amount: i64,
    pub reputation: f64,
    pub timestamp: u64,
    pub block_height: u64,
}

#[derive(
    Clone, Debug, serde::Serialize, serde::Deserialize, utoipa::ToSchema,
)]
pub struct RedistributionSummary {
    pub period_id: VotingPeriodId,
    pub total_redistributed: u64,
    pub losers_count: u32,
    pub winners_count: u32,
    pub unchanged_count: u32,
    pub transfers: Vec<RedistributionTransfer>,
    pub executed_at: u64,
    pub block_height: u64,
    pub conservation_check: i64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PeriodRedistribution {
    pub period_id: VotingPeriodId,
    pub slots_pending_redistribution: Vec<SlotId>,
    pub redistribution_summary: RedistributionSummary,
    pub applied: bool,
    pub calculated_at_height: u64,
    pub applied_at_height: Option<u64>,
}

impl PeriodRedistribution {
    pub fn new(
        period_id: VotingPeriodId,
        slots: Vec<SlotId>,
        summary: RedistributionSummary,
        height: u64,
    ) -> Self {
        Self {
            period_id,
            slots_pending_redistribution: slots,
            redistribution_summary: summary,
            applied: false,
            calculated_at_height: height,
            applied_at_height: None,
        }
    }

    pub fn mark_applied(&mut self, height: u64) {
        self.applied = true;
        self.applied_at_height = Some(height);
    }

    pub fn is_pending(&self) -> bool {
        !self.applied
    }

    pub fn get_slots_to_resolve(&self) -> &[SlotId] {
        &self.slots_pending_redistribution
    }
}

fn calculate_votecoin_deltas(
    reputation_changes: &HashMap<Address, (f64, f64)>,
    votecoin_at_stake: u64,
) -> Result<HashMap<Address, i64>, Error> {
    let mut deltas = HashMap::new();

    // Calculate reputation deltas
    let mut reputation_deltas: HashMap<Address, f64> = HashMap::new();
    let mut total_reputation_change = 0.0;

    for (&address, &(old_rep, new_rep)) in reputation_changes.iter() {
        let delta = new_rep - old_rep;
        reputation_deltas.insert(address, delta);
        total_reputation_change += delta.abs();

        debug!(
            "Reputation delta for address {}: {:.6} -> {:.6} (delta: {:+.6})",
            address.as_base58(),
            old_rep,
            new_rep,
            delta
        );
    }

    debug!(
        "Total reputation change magnitude: {:.6} (threshold: 1e-10)",
        total_reputation_change
    );

    // Early exit: no significant reputation changes
    if total_reputation_change < 1e-10 {
        debug!(
            "No reputation changes detected (below threshold), skipping VoteCoin redistribution"
        );
        for &address in reputation_changes.keys() {
            deltas.insert(address, 0);
        }
        return Ok(deltas);
    }

    // Calculate redistribution pool using integer arithmetic via BasisPoints
    let redistribution_pool =
        BasisPoints::REDISTRIBUTION_RATE.apply_to_u64(votecoin_at_stake) as i64;

    if redistribution_pool == 0 {
        debug!("Redistribution pool is zero, skipping VoteCoin redistribution");
        for &address in reputation_changes.keys() {
            deltas.insert(address, 0);
        }
        return Ok(deltas);
    }

    // Separate winners and losers
    let mut winners: Vec<(Address, f64)> = Vec::new();
    let mut losers: Vec<(Address, f64)> = Vec::new();
    let mut total_win = 0.0;
    let mut total_loss = 0.0;

    for (&address, &rep_delta) in reputation_deltas.iter() {
        if rep_delta > 0.0 {
            winners.push((address, rep_delta));
            total_win += rep_delta;
        } else if rep_delta < 0.0 {
            losers.push((address, rep_delta.abs()));
            total_loss += rep_delta.abs();
        }
    }

    debug!(
        "Winners: {} voters with total gain: {:.6}",
        winners.len(),
        total_win
    );
    debug!(
        "Losers: {} voters with total loss: {:.6}",
        losers.len(),
        total_loss
    );

    // Calculate amount to redistribute
    let effective_change = total_win.min(total_loss);
    let amount_to_redistribute =
        ((redistribution_pool as f64) * effective_change).floor() as i64;

    debug!(
        "Effective change: {:.6}, Amount to redistribute: {} (votecoin_at_stake: {}, rate: {})",
        effective_change,
        amount_to_redistribute,
        votecoin_at_stake,
        BasisPoints::REDISTRIBUTION_RATE
    );

    if amount_to_redistribute == 0 {
        debug!(
            "Amount to redistribute is zero, skipping VoteCoin redistribution"
        );
        for &address in reputation_changes.keys() {
            deltas.insert(address, 0);
        }
        return Ok(deltas);
    }

    winners.sort_by_key(|(addr, _)| *addr);
    losers.sort_by_key(|(addr, _)| *addr);

    // Calculate winner allocations using floor, track remainder
    let mut total_assigned_gain = 0i64;
    let mut winner_deltas: Vec<(Address, i64)> = Vec::new();

    for (address, rep_gain) in &winners {
        let proportion = rep_gain / total_win;
        let gain = (proportion * amount_to_redistribute as f64).floor() as i64;
        winner_deltas.push((*address, gain));
        total_assigned_gain += gain;
    }

    // Calculate loser allocations using floor, track remainder
    let mut total_assigned_loss = 0i64;
    let mut loser_deltas: Vec<(Address, i64)> = Vec::new();

    for (address, rep_loss) in &losers {
        let proportion = rep_loss / total_loss;
        let loss = (proportion * amount_to_redistribute as f64).floor() as i64;
        loser_deltas.push((*address, -loss));
        total_assigned_loss += loss;
    }

    let gain_remainder = amount_to_redistribute - total_assigned_gain;
    let loss_remainder = amount_to_redistribute - total_assigned_loss;

    if gain_remainder > 0 && !winner_deltas.is_empty() {
        winner_deltas[0].1 += gain_remainder;
    }
    if loss_remainder > 0 && !loser_deltas.is_empty() {
        loser_deltas[0].1 -= loss_remainder;
    }

    for (address, delta) in winner_deltas {
        deltas.insert(address, delta);
    }
    for (address, delta) in loser_deltas {
        deltas.insert(address, delta);
    }
    for &address in reputation_changes.keys() {
        deltas.entry(address).or_insert(0);
    }

    // Verify conservation - this should always pass with the floor-then-remainder algorithm
    let final_sum: i64 = deltas.values().sum();
    debug_assert_eq!(
        final_sum, 0,
        "Conservation invariant violated - this is a bug"
    );

    if final_sum != 0 {
        return Err(Error::InvalidTransaction {
            reason: format!(
                "VoteCoin redistribution conservation violated: sum = {final_sum} (expected 0)"
            ),
        });
    }

    Ok(deltas)
}

fn generate_transfer_records(
    votecoin_deltas: &HashMap<Address, i64>,
    reputation_changes: &HashMap<Address, (f64, f64)>,
    period_id: VotingPeriodId,
    timestamp: u64,
    block_height: u64,
) -> Result<Vec<RedistributionTransfer>, Error> {
    let mut transfers = Vec::new();

    for (&address, &delta) in votecoin_deltas.iter() {
        if delta != 0 {
            let reputation = reputation_changes
                .get(&address)
                .map(|(_, new_rep)| round_reputation(*new_rep))
                .unwrap_or(0.0);

            transfers.push(RedistributionTransfer {
                period_id,
                from_address: address,
                to_address: address,
                amount: delta,
                reputation,
                timestamp,
                block_height,
            });
        }
    }

    Ok(transfers)
}

fn adjust_deltas_for_insufficient_balances(
    state: &State,
    rwtxn: &mut RwTxn,
    votecoin_deltas: &mut HashMap<Address, i64>,
) -> Result<u64, Error> {
    let mut total_deficit = 0u64;
    let mut adjustments = Vec::new();

    for (address, &delta) in votecoin_deltas.iter() {
        let current_balance = state.get_votecoin_balance(rwtxn, address)?;

        debug!(
            "Voter {}: current balance = {} (aggregated), delta = {}",
            address, current_balance, delta
        );

        if delta < 0 {
            let amount_to_lose = (-delta) as u32;

            debug!(
                "Checking VoteCoin balance for voter {}: has {}, needs to lose {}",
                address, current_balance, amount_to_lose
            );

            if current_balance < amount_to_lose {
                let deficit = amount_to_lose - current_balance;
                total_deficit += deficit as u64;

                adjustments.push((
                    *address,
                    -(current_balance as i64),
                    deficit,
                ));

                warn!(
                    "Voter {} has insufficient VoteCoin: has {}, needs {}, deficit {}",
                    address, current_balance, amount_to_lose, deficit
                );
            }
        }
    }

    if total_deficit > 0 {
        warn!(
            "Total VoteCoin deficit: {}. Adjusting deltas to maintain conservation.",
            total_deficit
        );

        for (address, new_delta, deficit) in adjustments {
            votecoin_deltas.insert(address, new_delta);

            info!(
                "Adjusted voter {} delta to {} (deficit: {})",
                address, new_delta, deficit
            );
        }

        let mut winners: Vec<(Address, i64)> = votecoin_deltas
            .iter()
            .filter(|(_, delta)| **delta > 0)
            .map(|(id, delta)| (*id, *delta))
            .collect();
        winners.sort_by_key(|(addr, _)| *addr);

        if !winners.is_empty() {
            let total_winnings: i64 = winners.iter().map(|(_, d)| d).sum();

            if total_winnings > 0 {
                let deficit_i64 = total_deficit as i64;

                let mut total_reduced = 0i64;
                let mut reductions: Vec<(Address, i64, i64)> = Vec::new();

                for (address, original_gain) in &winners {
                    let proportion =
                        *original_gain as f64 / total_winnings as f64;
                    let reduction =
                        (deficit_i64 as f64 * proportion).floor() as i64;
                    reductions.push((*address, *original_gain, reduction));
                    total_reduced += reduction;
                }

                let remainder = deficit_i64 - total_reduced;
                if remainder > 0 && !reductions.is_empty() {
                    reductions[0].2 += remainder;
                }

                // Apply reductions
                for (address, original_gain, reduction) in reductions {
                    let new_gain = original_gain - reduction;
                    votecoin_deltas.insert(address, new_gain);

                    debug!(
                        "Reduced winner {} gain from {} to {} to cover deficit",
                        address, original_gain, new_gain
                    );
                }
            }
        }
    }

    Ok(total_deficit)
}

/// Calculates votecoin redistribution deltas based on reputation changes.
///
/// # Locking Requirements
///
/// Like `apply_votecoin_redistribution`, this must be called under
/// the consensus lock to ensure balance consistency.
///
/// # Conservation Invariant
///
/// The sum of all deltas MUST be zero (votecoin is neither created nor destroyed).
/// This is verified via `conservation_check` in the returned summary.
pub fn redistribute_votecoin_after_consensus(
    state: &State,
    rwtxn: &mut RwTxn,
    period_id: VotingPeriodId,
    reputation_changes: &HashMap<Address, (f64, f64)>,
    current_timestamp: u64,
    current_height: u64,
) -> Result<RedistributionSummary, Error> {
    info!(
        "Starting VoteCoin redistribution for period {} with {} voters",
        period_id.0,
        reputation_changes.len()
    );

    if reputation_changes.is_empty() {
        debug!(
            "No voters in period {}, skipping redistribution",
            period_id.0
        );
        return Ok(RedistributionSummary {
            period_id,
            total_redistributed: 0,
            losers_count: 0,
            winners_count: 0,
            unchanged_count: 0,
            transfers: Vec::new(),
            executed_at: current_timestamp,
            block_height: current_height,
            conservation_check: 0,
        });
    }

    let total_votecoin_supply = state
        .get_total_votecoin_supply(rwtxn)
        .map_err(|e| Error::InvalidTransaction {
            reason: format!("Failed to query total VoteCoin supply: {e}"),
        })?;

    let votecoin_at_stake = total_votecoin_supply as u64;

    if votecoin_at_stake == 0 {
        warn!(
            "No VoteCoin in circulation, skipping redistribution for period {}",
            period_id.0
        );
        return Ok(RedistributionSummary {
            period_id,
            total_redistributed: 0,
            winners_count: 0,
            losers_count: 0,
            unchanged_count: reputation_changes.len() as u32,
            transfers: vec![],
            executed_at: current_timestamp,
            block_height: current_height,
            conservation_check: 0,
        });
    }

    debug!(
        "Total VoteCoin supply for redistribution calculation: {} (period {})",
        votecoin_at_stake, period_id.0
    );

    let mut votecoin_deltas =
        calculate_votecoin_deltas(reputation_changes, votecoin_at_stake)?;

    let total_deficit = adjust_deltas_for_insufficient_balances(
        state,
        rwtxn,
        &mut votecoin_deltas,
    )?;

    if total_deficit > 0 {
        info!(
            "Redistribution adjusted for {} VoteCoin deficit in period {}",
            total_deficit, period_id.0
        );
    }

    let transfers = generate_transfer_records(
        &votecoin_deltas,
        reputation_changes,
        period_id,
        current_timestamp,
        current_height,
    )?;

    let mut losers_count = 0u32;
    let mut winners_count = 0u32;
    let mut unchanged_count = 0u32;
    let mut total_redistributed = 0u64;

    for &delta in votecoin_deltas.values() {
        if delta < 0 {
            losers_count += 1;
            total_redistributed += (-delta) as u64;
        } else if delta > 0 {
            winners_count += 1;
        } else {
            unchanged_count += 1;
        }
    }

    info!(
        "VoteCoin redistribution for period {}: {} transfers, {} total redistributed, {} winners, {} losers, {} unchanged",
        period_id.0,
        transfers.len(),
        total_redistributed,
        winners_count,
        losers_count,
        unchanged_count
    );

    let conservation_check: i64 = votecoin_deltas.values().sum();

    let summary = RedistributionSummary {
        period_id,
        total_redistributed,
        losers_count,
        winners_count,
        unchanged_count,
        transfers,
        executed_at: current_timestamp,
        block_height: current_height,
        conservation_check,
    };

    Ok(summary)
}

/// Applies votecoin redistribution after consensus is calculated.
///
/// # Locking Requirements
///
/// This function MUST be called while holding the consensus lock
/// (`VotingSystem::consensus_lock`). The consensus lock ensures:
///
/// 1. No concurrent consensus calculations can modify votecoin balances
/// 2. Balance reads during delta calculation match balances at application time
/// 3. No race conditions between `adjust_deltas_for_insufficient_balances()`
///    and actual balance modifications
///
/// The caller (`commit_consensus_result`) acquires this lock via
/// `calculate_and_store_consensus()`.
///
/// # Atomicity
///
/// All UTXO modifications use the same `rwtxn` transaction, ensuring
/// atomicity. Either all transfers complete or none do.
pub fn apply_votecoin_redistribution(
    state: &State,
    rwtxn: &mut RwTxn,
    redistribution: &RedistributionSummary,
    block_height: u64,
) -> Result<(), Error> {
    info!(
        "Applying VoteCoin redistribution for period {} at block height {}",
        redistribution.period_id.0, block_height
    );

    if redistribution.transfers.is_empty() {
        debug!("No redistribution transfers to apply");
        return Ok(());
    }

    let mut voter_net_changes = HashMap::new();

    for transfer in &redistribution.transfers {
        let address = transfer.from_address;
        *voter_net_changes.entry(address).or_insert(0i64) += transfer.amount;
    }

    let total_delta: i64 = voter_net_changes.values().sum();

    if total_delta != 0 {
        return Err(Error::InvalidTransaction {
            reason: format!(
                "VoteCoin redistribution conservation violation: total delta = {total_delta} (expected 0)"
            ),
        });
    }

    let mut sorted_voters: Vec<(Address, i64)> =
        voter_net_changes.into_iter().collect();
    sorted_voters.sort_by_key(|(address, _)| *address);

    let voter_count = sorted_voters.len();

    let mut global_sequence = 0u32;

    for (address, net_change) in sorted_voters {
        if net_change == 0 {
            continue;
        }

        debug!(
            "Processing redistribution for address {}: net_change = {}",
            address.as_base58(),
            net_change,
        );

        if net_change < 0 {
            let amount_to_remove = (-net_change) as u32;

            global_sequence = remove_votecoin_from_voter(
                state,
                rwtxn,
                &address,
                amount_to_remove,
                redistribution,
                global_sequence,
            )?;
        } else {
            let amount_to_add = net_change as u32;

            add_votecoin_to_voter(
                state,
                rwtxn,
                &address,
                amount_to_add,
                redistribution.period_id,
                block_height,
                global_sequence,
            )?;

            global_sequence += 1;
        }
    }

    info!(
        "Successfully applied VoteCoin redistribution for period {}: {} transfers affecting {} voters",
        redistribution.period_id.0,
        redistribution.transfers.len(),
        voter_count
    );

    Ok(())
}

fn remove_votecoin_from_voter(
    state: &State,
    rwtxn: &mut RwTxn,
    voter_address: &Address,
    amount: u32,
    redistribution: &RedistributionSummary,
    change_sequence_start: u32,
) -> Result<u32, Error> {
    let mut voter_addresses = std::collections::HashSet::new();
    voter_addresses.insert(*voter_address);

    let utxos = state.get_utxos_by_addresses(rwtxn, &voter_addresses)?;

    let mut votecoin_utxos: Vec<(OutPoint, u32, Address)> = utxos
        .into_iter()
        .filter_map(|(outpoint, filled_output)| {
            if let FilledOutputContent::Votecoin(votecoin_amount) =
                filled_output.content
            {
                Some((outpoint, votecoin_amount, filled_output.address))
            } else {
                None
            }
        })
        .collect();

    votecoin_utxos.sort_by_key(|(outpoint, _, _)| *outpoint);

    let total_available: u32 =
        votecoin_utxos.iter().map(|(_, amt, _)| amt).sum();

    if total_available < amount {
        return Err(Error::InvalidTransaction {
            reason: format!(
                "Insufficient VoteCoin for redistribution: address has {total_available}, needs {amount}"
            ),
        });
    }

    let mut removed = 0u32;
    let mut remaining_amount = amount;
    let mut change_sequence_counter = change_sequence_start;

    for (outpoint, votecoin_amount, original_address) in votecoin_utxos {
        if remaining_amount == 0 {
            break;
        }

        if votecoin_amount <= remaining_amount {
            state.delete_utxo_supply_neutral(rwtxn, &outpoint)?;
            removed += votecoin_amount;
            remaining_amount -= votecoin_amount;

            debug!(
                "Removed VoteCoin UTXO {:?} with {} VoteCoin from address {}",
                outpoint,
                votecoin_amount,
                original_address.as_base58()
            );
        } else {
            state.delete_utxo_supply_neutral(rwtxn, &outpoint)?;

            let change_amount = votecoin_amount - remaining_amount;
            let change_output = FilledOutput {
                address: original_address,
                content: FilledOutputContent::Votecoin(change_amount),
                memo: vec![],
            };

            let change_outpoint = generate_redistribution_outpoint(
                redistribution.period_id,
                &original_address,
                redistribution.block_height,
                change_sequence_counter,
            );
            change_sequence_counter += 1;

            state.insert_utxo_supply_neutral(
                rwtxn,
                &change_outpoint,
                &change_output,
            )?;

            removed += remaining_amount;
            remaining_amount = 0;

            debug!(
                "Partially removed {} VoteCoin from UTXO {:?}, created change UTXO {:?} with {} VoteCoin for address {}",
                removed,
                outpoint,
                change_outpoint,
                change_amount,
                voter_address.as_base58()
            );
        }
    }

    if removed != amount {
        return Err(Error::InvalidTransaction {
            reason: format!(
                "VoteCoin removal mismatch: removed {removed}, expected {amount}"
            ),
        });
    }

    Ok(change_sequence_counter)
}

fn add_votecoin_to_voter(
    state: &State,
    rwtxn: &mut RwTxn,
    voter_address: &Address,
    amount: u32,
    period_id: VotingPeriodId,
    block_height: u64,
    sequence: u32,
) -> Result<(), Error> {
    let outpoint = generate_redistribution_outpoint(
        period_id,
        voter_address,
        block_height,
        sequence,
    );

    let output = FilledOutput {
        address: *voter_address,
        content: FilledOutputContent::Votecoin(amount),
        memo: vec![],
    };

    state.insert_utxo_supply_neutral(rwtxn, &outpoint, &output)?;

    debug!(
        "Created VoteCoin UTXO {:?} with {} VoteCoin for voter {:?} in period {} (sequence {})",
        outpoint, amount, voter_address, period_id.0, sequence
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_address(id: u8) -> Address {
        let mut bytes = [0u8; 20];
        bytes[0] = id;
        Address(bytes)
    }

    #[test]
    fn test_calculate_votecoin_deltas_conservation() {
        let mut reputation_changes = HashMap::new();
        reputation_changes.insert(make_address(1), (0.4, 0.6));
        reputation_changes.insert(make_address(2), (0.6, 0.4));
        reputation_changes.insert(make_address(3), (0.5, 0.5));

        let votecoin_at_stake = 100000u64;

        let deltas =
            calculate_votecoin_deltas(&reputation_changes, votecoin_at_stake)
                .unwrap();

        let total: i64 = deltas.values().sum();
        assert_eq!(total, 0, "VoteCoin deltas must sum to zero (conservation)");

        assert!(deltas[&make_address(1)] > 0, "Winner should gain VoteCoin");
        assert!(deltas[&make_address(2)] < 0, "Loser should lose VoteCoin");
        assert_eq!(
            deltas[&make_address(3)],
            0,
            "Unchanged voter should have zero delta"
        );

        assert_eq!(
            deltas[&make_address(1)],
            -deltas[&make_address(2)],
            "Symmetric reputation changes should yield symmetric VoteCoin changes"
        );
    }

    #[test]
    fn test_calculate_votecoin_deltas_no_change() {
        let mut reputation_changes = HashMap::new();
        reputation_changes.insert(make_address(1), (0.5, 0.5));
        reputation_changes.insert(make_address(2), (0.5, 0.5));
        reputation_changes.insert(make_address(3), (0.5, 0.5));

        let votecoin_at_stake = 100000u64;

        let deltas =
            calculate_votecoin_deltas(&reputation_changes, votecoin_at_stake)
                .unwrap();

        for &delta in deltas.values() {
            assert_eq!(
                delta, 0,
                "No reputation change should yield zero VoteCoin delta"
            );
        }
    }

    #[test]
    fn test_generate_transfer_records() {
        let mut votecoin_deltas = HashMap::new();
        votecoin_deltas.insert(make_address(1), -100);
        votecoin_deltas.insert(make_address(2), 60);
        votecoin_deltas.insert(make_address(3), 40);

        let period_id = VotingPeriodId::new(1);
        let timestamp = 1000u64;
        let block_height = 100u64;

        let reputation_changes = HashMap::new();
        let transfers = generate_transfer_records(
            &votecoin_deltas,
            &reputation_changes,
            period_id,
            timestamp,
            block_height,
        )
        .unwrap();

        assert_eq!(transfers.len(), 3, "Should create 3 transfer records");

        let total_delta: i64 = transfers.iter().map(|t| t.amount).sum();
        assert_eq!(total_delta, 0, "Total delta should be zero (conservation)");

        for transfer in &transfers {
            let address = transfer.from_address;
            let expected_delta = votecoin_deltas.get(&address).unwrap();
            assert_eq!(
                transfer.amount, *expected_delta,
                "Transfer amount should match delta"
            );
        }
    }

    #[test]
    fn test_generate_redistribution_outpoint_determinism() {
        let period_id = VotingPeriodId::new(1);
        let voter_address = make_address(42);
        let block_height = 12345u64;
        let sequence = 7u32;

        let outpoint1 = generate_redistribution_outpoint(
            period_id,
            &voter_address,
            block_height,
            sequence,
        );
        let outpoint2 = generate_redistribution_outpoint(
            period_id,
            &voter_address,
            block_height,
            sequence,
        );
        let outpoint3 = generate_redistribution_outpoint(
            period_id,
            &voter_address,
            block_height,
            sequence,
        );

        assert_eq!(outpoint1, outpoint2);
        assert_eq!(outpoint2, outpoint3);
        assert_eq!(outpoint1, outpoint3);
    }

    #[test]
    fn test_generate_redistribution_outpoint_uniqueness() {
        let period_id1 = VotingPeriodId::new(1);
        let period_id2 = VotingPeriodId::new(2);
        let voter1 = make_address(1);
        let voter2 = make_address(2);
        let block_height1 = 100u64;
        let block_height2 = 200u64;

        let base_outpoint = generate_redistribution_outpoint(
            period_id1,
            &voter1,
            block_height1,
            0,
        );

        let different_period = generate_redistribution_outpoint(
            period_id2,
            &voter1,
            block_height1,
            0,
        );
        assert_ne!(base_outpoint, different_period);

        let different_voter = generate_redistribution_outpoint(
            period_id1,
            &voter2,
            block_height1,
            0,
        );
        assert_ne!(base_outpoint, different_voter);

        let different_height = generate_redistribution_outpoint(
            period_id1,
            &voter1,
            block_height2,
            0,
        );
        assert_ne!(base_outpoint, different_height);

        let different_sequence = generate_redistribution_outpoint(
            period_id1,
            &voter1,
            block_height1,
            1,
        );
        assert_ne!(base_outpoint, different_sequence);
    }
}
