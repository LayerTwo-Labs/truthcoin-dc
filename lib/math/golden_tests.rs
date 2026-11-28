use std::collections::{HashMap, HashSet};

use crate::math::{lmsr, trading, voting};
use crate::state::State;
use crate::state::decisions::DecisionId;
use crate::types::Address;

fn hash_f64s(values: impl IntoIterator<Item = f64>) -> String {
    let mut hasher = blake3::Hasher::new();
    for value in values {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

fn next_value(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 33
}

fn consensus_fixture(
    num_voters: usize,
    num_decisions: usize,
) -> voting::DetailedConsensusResult {
    let voters: Vec<Address> = (0..num_voters)
        .map(|i| {
            let mut bytes = [0u8; 20];
            bytes[0] = i as u8;
            Address(bytes)
        })
        .collect();
    let decisions: Vec<DecisionId> = (0..num_decisions)
        .map(|j| DecisionId::new(true, 1, j as u32).unwrap())
        .collect();
    let scaled: HashSet<DecisionId> =
        decisions.iter().copied().skip(1).step_by(3).collect();

    let mut seed = 0x5eed_u64;
    let mut matrix =
        voting::SparseVoteMatrix::new(voters.clone(), decisions.clone());
    let mut reveal_heights = HashMap::new();
    for (i, &voter) in voters.iter().enumerate() {
        for &decision in &decisions {
            let draw = next_value(&mut seed);
            if draw.is_multiple_of(11) {
                continue;
            }
            reveal_heights.insert(voter, 1 + i as u32 % 3);
            let value = if scaled.contains(&decision) {
                (draw % 1001) as f64 / 1000.0
            } else if draw.is_multiple_of(7) {
                0.5
            } else {
                (draw % 2) as f64
            };
            matrix.set_vote(voter, decision, value).unwrap();
        }
    }

    let mut weights = voting::VotingWeightVector::new();
    for &voter in &voters {
        let draw = next_value(&mut seed);
        weights.set_reputation(voter, (1 + draw % 97) as f64 / 1000.0);
    }

    voting::calculate_consensus(
        &matrix,
        &weights,
        &scaled,
        &HashMap::new(),
        &reveal_heights,
    )
    .unwrap()
}

fn consensus_bits(result: &voting::DetailedConsensusResult) -> Vec<f64> {
    let mut outcomes: Vec<_> = result.outcomes.iter().collect();
    outcomes.sort_by_key(|(decision, _)| **decision);
    let mut bits = result.first_loading.clone();
    bits.push(result.certainty);
    bits.extend(result.updated_reputations.values().copied());
    bits.extend(
        outcomes
            .into_iter()
            .map(|(_, outcome)| outcome.unwrap_or(f64::NAN)),
    );
    bits
}

#[test]
fn consensus_golden_hash() {
    let result = consensus_fixture(9, 7);
    assert_eq!(result.first_loading.len(), 7);
    assert_eq!(
        hash_f64s(consensus_bits(&result)),
        "2ffe781695bac3d611897a12593984a44f7dd32ae2525248fc45e21d5011bb89"
    );
}

fn reveal_timing_fixture(
    reveal_heights: &HashMap<Address, u32>,
) -> voting::DetailedConsensusResult {
    let voters: Vec<Address> = (0..12u8).map(|i| Address([i; 20])).collect();
    let decisions: Vec<DecisionId> = (0..5)
        .map(|j| DecisionId::new(true, 1, j).unwrap())
        .collect();
    let truth = [0.42, 1.0, 0.0, 1.0, 2.0];
    let mut matrix =
        voting::SparseVoteMatrix::new(voters.clone(), decisions.clone());
    let mut weights = voting::VotingWeightVector::new();
    let mut seed = 0x7e11_u64;
    for (i, &voter) in voters.iter().enumerate() {
        weights.set_reputation(
            voter,
            (5 + next_value(&mut seed) % 95) as f64 / 1000.0,
        );
        if i == 11 {
            continue;
        }
        for (j, &decision) in decisions.iter().enumerate() {
            let value = match (i, j) {
                (3, 0) => 0.421,
                (7 | 8, 2) => 1.0,
                (9, 0) => 0.9,
                (9, 4) => 0.0,
                (9, _) => 1.0 - truth[j],
                (10, 1) => 0.5,
                _ => truth[j],
            };
            matrix.set_vote(voter, decision, value).unwrap();
        }
    }
    let scaled: HashSet<DecisionId> = [decisions[0]].into_iter().collect();
    let categorical: HashMap<DecisionId, u16> =
        [(decisions[4], 3)].into_iter().collect();
    voting::calculate_consensus(
        &matrix,
        &weights,
        &scaled,
        &categorical,
        reveal_heights,
    )
    .unwrap()
}

#[test]
fn reveal_timing_golden_hash() {
    let heights: HashMap<Address, u32> = (0..11u8)
        .map(|i| (Address([i; 20]), 1 + u32::from(i) % 4))
        .collect();
    assert_eq!(
        hash_f64s(consensus_bits(&reveal_timing_fixture(&heights))),
        "1895668e665618436fad20884a637695db211706e56fea435d7d6047db0368b9"
    );
}

#[test]
fn lmsr_golden_hash() {
    let share_sets: [&[f64]; 4] = [
        &[0.0, 0.0],
        &[125.0, -40.5, 310.25],
        &[1.0e6, 2.5e6, 0.0, 7.0e5, 3.3e6],
        &[17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 900.0],
    ];
    let betas = [7.0, 1000.0, 1.0e6];
    let mut values = Vec::new();
    for shares in share_sets {
        for beta in betas {
            values.push(lmsr::calculate_cost(shares, beta).unwrap());
            values.extend(lmsr::calculate_prices(shares, beta).unwrap());
        }
        values.push(trading::calculate_lmsr_liquidity(1.0e6, shares.len()));
        values.push(trading::derive_beta_from_liquidity(
            123_456_789,
            shares.len(),
        ));
    }
    assert_eq!(
        hash_f64s(values),
        "147119f26e628491ff8a857bc6eba676e25d21983fc0bd49d76bc6bfa42787c9"
    );
}

#[test]
fn block_limits_golden_hash() {
    let mut hasher = blake3::Hasher::new();
    for month in 0..=121u32 {
        let height = month * 6 * 24 * 30;
        hasher.update(&(State::body_size_limit(height) as u64).to_le_bytes());
        hasher.update(&(State::body_sigops_limit(height) as u64).to_le_bytes());
    }
    assert_eq!(
        hasher.finalize().to_hex().to_string(),
        "a8dfdbd17f9c2b25a8a162d275de3e2aa6fb44e11255ed02b79602afc2cec669"
    );
}
