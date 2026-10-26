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
    for &voter in &voters {
        for &decision in &decisions {
            let draw = next_value(&mut seed);
            if draw.is_multiple_of(11) {
                continue;
            }
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

    voting::calculate_consensus(&matrix, &weights, &scaled, &HashMap::new())
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
fn consensus_output_bits_match_golden_hash() {
    let result = consensus_fixture(9, 7);
    assert_eq!(result.first_loading.len(), 7);
    assert_eq!(
        hash_f64s(consensus_bits(&result)),
        "1b7edc4fce3cbc2b01c6d77814c9c24aa23ef6fafa61b0772459c40b46d2098a"
    );
}

#[test]
fn lmsr_output_bits_match_golden_hash() {
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
fn block_limits_match_golden_hash() {
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
