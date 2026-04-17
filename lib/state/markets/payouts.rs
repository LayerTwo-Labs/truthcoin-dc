use sneed::RoTxn;
use std::collections::{HashMap, HashSet};

use crate::math::allocation;
use crate::state::Error;
use crate::state::decisions::{Decision, DecisionId};
use crate::types::{Address, MerkleRoot, OutPoint};

use super::market::Market;
use super::types::{FeePayoutRecord, FeeRole, MarketId};

fn is_vote_correct(
    vote: &crate::state::voting::types::VoteValue,
    consensus: f64,
    decision: &Decision,
) -> bool {
    use crate::state::voting::types::VoteValue;
    match vote {
        VoteValue::Abstain => false,
        VoteValue::Binary(b) => {
            let vote_f = if *b { 1.0 } else { 0.0 };
            vote_f == consensus
        }
        VoteValue::Scalar(v) => {
            if decision.is_scaled() {
                use crate::math::safe_math::round_to_precision;
                round_to_precision((v - consensus).abs(), 6) <= 0.05
            } else {
                *v == consensus
            }
        }
        VoteValue::Categorical(idx) => {
            if *idx as f64 == consensus {
                return true;
            }
            consensus == 0.5 && decision.inconclusive_index() == Some(*idx)
        }
    }
}

fn find_correct_voters(
    txn: &RoTxn,
    state: &crate::state::State,
    decision_ids: &[DecisionId],
    decision_outcomes: &HashMap<DecisionId, f64>,
    decisions: &HashMap<DecisionId, Decision>,
) -> Result<HashMap<DecisionId, Vec<(Address, f64)>>, Error> {
    use crate::state::voting::types::VotingPeriodId;
    let mut result: HashMap<DecisionId, Vec<(Address, f64)>> = HashMap::new();

    for decision_id in decision_ids {
        let consensus = match decision_outcomes.get(decision_id) {
            Some(v) => *v,
            None => {
                result.insert(*decision_id, Vec::new());
                continue;
            }
        };

        let decision = match decisions.get(decision_id) {
            Some(d) => d,
            None => {
                result.insert(*decision_id, Vec::new());
                continue;
            }
        };

        let period_id = VotingPeriodId::new(decision_id.voting_period());
        let votes = state
            .voting()
            .databases()
            .get_votes_for_period(txn, period_id)?;

        let mut correct_voters: Vec<(Address, f64)> = Vec::new();
        for (key, entry) in &votes {
            if key.decision_id != *decision_id {
                continue;
            }
            if !is_vote_correct(&entry.value, consensus, decision) {
                continue;
            }
            let reputation =
                state.reputation().get_reputation(txn, &key.voter_address)?;
            if reputation > 0.0 {
                correct_voters.push((key.voter_address, reputation));
            }
        }

        correct_voters.sort_by_key(|(addr, _)| *addr);
        result.insert(*decision_id, correct_voters);
    }

    Ok(result)
}

pub fn calculate_fee_distribution(
    txn: &RoTxn,
    state: &crate::state::State,
    market: &Market,
    fee_sats: u64,
    decision_outcomes: &HashMap<DecisionId, f64>,
    decisions: &HashMap<DecisionId, Decision>,
) -> Result<Vec<FeePayoutRecord>, Error> {
    if fee_sats == 0 {
        return Ok(Vec::new());
    }

    let pool_split = allocation::allocate_proportionally_u64(
        vec![
            ("voter", 2.0),
            ("decision_author", 1.0),
            ("market_author", 1.0),
        ],
        fee_sats,
    )
    .map_err(|e| Error::InvalidTransaction {
        reason: format!("Fee pool split failed: {e}"),
    })?;

    let mut voter_pool = 0u64;
    let mut decision_author_pool = 0u64;
    let mut market_author_pool = 0u64;
    for (key, amount) in &pool_split.allocations {
        match *key {
            "voter" => voter_pool = *amount,
            "decision_author" => decision_author_pool = *amount,
            "market_author" => market_author_pool = *amount,
            _ => {}
        }
    }

    let mut fee_payouts: Vec<FeePayoutRecord> = Vec::new();

    let correct_voters = find_correct_voters(
        txn,
        state,
        &market.decision_ids,
        decision_outcomes,
        decisions,
    )?;

    if voter_pool > 0 {
        let decision_weights: Vec<(&DecisionId, f64)> =
            market.decision_ids.iter().map(|d| (d, 1.0)).collect();
        let decision_pool_split = allocation::allocate_proportionally_u64(
            decision_weights,
            voter_pool,
        )
        .map_err(|e| Error::InvalidTransaction {
            reason: format!("Voter decision pool split failed: {e}"),
        })?;

        for (decision_id, sub_pool) in decision_pool_split.allocations {
            let voters =
                correct_voters.get(decision_id).cloned().unwrap_or_default();

            if voters.is_empty() {
                market_author_pool += sub_pool;
                continue;
            }

            let voter_alloc =
                allocation::allocate_proportionally_u64(voters, sub_pool)
                    .map_err(|e| Error::InvalidTransaction {
                        reason: format!("Voter payout allocation failed: {e}"),
                    })?;

            for (addr, amount) in voter_alloc.allocations {
                fee_payouts.push(FeePayoutRecord {
                    address: addr,
                    amount_sats: amount,
                    fee_role: FeeRole::CorrectVoter,
                });
            }
        }
    }

    if decision_author_pool > 0 {
        let mut unique_authors: Vec<Address> = market
            .decision_ids
            .iter()
            .filter_map(|did| {
                decisions
                    .get(did)
                    .map(|d| Address(d.market_maker_pubkey_hash))
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        unique_authors.sort();

        let author_weights: Vec<(Address, f64)> =
            unique_authors.into_iter().map(|a| (a, 1.0)).collect();
        let author_alloc = allocation::allocate_proportionally_u64(
            author_weights,
            decision_author_pool,
        )
        .map_err(|e| Error::InvalidTransaction {
            reason: format!("Decision author allocation failed: {e}"),
        })?;

        for (addr, amount) in author_alloc.allocations {
            fee_payouts.push(FeePayoutRecord {
                address: addr,
                amount_sats: amount,
                fee_role: FeeRole::DecisionAuthor,
            });
        }
    }

    if market_author_pool > 0 {
        fee_payouts.push(FeePayoutRecord {
            address: market.creator_address,
            amount_sats: market_author_pool,
            fee_role: FeeRole::MarketAuthor,
        });
    }

    fee_payouts.sort_by(|a, b| {
        let role_order = |r: &FeeRole| -> u8 {
            match r {
                FeeRole::CorrectVoter => 0,
                FeeRole::DecisionAuthor => 1,
                FeeRole::MarketAuthor => 2,
            }
        };
        role_order(&a.fee_role)
            .cmp(&role_order(&b.fee_role))
            .then(a.address.cmp(&b.address))
    });

    Ok(fee_payouts)
}

fn generate_market_address(market_id: &MarketId, domain: &[u8]) -> Address {
    use blake3::Hasher;
    let mut hasher = Hasher::new();
    hasher.update(domain);
    hasher.update(&market_id.0);
    let hash = hasher.finalize();
    let mut address_bytes = [0u8; 20];
    address_bytes.copy_from_slice(&hash.as_bytes()[0..20]);
    Address(address_bytes)
}

pub fn generate_market_treasury_address(market_id: &MarketId) -> Address {
    generate_market_address(market_id, b"MARKET_TREASURY_ADDRESS")
}

pub fn generate_market_author_fee_address(market_id: &MarketId) -> Address {
    generate_market_address(market_id, b"MARKET_AUTHOR_FEE_ADDRESS")
}

/// Generate a deterministic outpoint for share payouts
pub fn generate_share_payout_outpoint(
    market_id: &MarketId,
    shareholder_address: &Address,
    block_height: u32,
    sequence: u32,
) -> OutPoint {
    use blake3::Hasher;

    let mut hasher = Hasher::new();
    hasher.update(b"SHARE_PAYOUT");
    hasher.update(&market_id.0);
    hasher.update(&shareholder_address.0);
    hasher.update(&block_height.to_le_bytes());
    hasher.update(&sequence.to_le_bytes());

    let hash = hasher.finalize();
    let merkle_root = MerkleRoot::from(*hash.as_bytes());

    OutPoint::Payout {
        hash: merkle_root,
        vout: sequence,
    }
}

#[cfg(test)]
#[allow(clippy::print_stdout, clippy::uninlined_format_args)]
mod payout_tests {
    use super::*;

    #[test]
    fn test_is_vote_correct_binary() {
        use crate::state::decisions::{Decision, DecisionType};
        use crate::state::voting::types::VoteValue;

        let decision = Decision::new(
            [0u8; 20],
            DecisionType::Binary,
            "Test".to_string(),
            "Test decision".to_string(),
            None,
            None,
            vec![],
        )
        .unwrap();

        assert!(is_vote_correct(&VoteValue::Binary(true), 1.0, &decision));
        assert!(!is_vote_correct(&VoteValue::Binary(false), 1.0, &decision));

        assert!(is_vote_correct(&VoteValue::Binary(false), 0.0, &decision));
        assert!(!is_vote_correct(&VoteValue::Binary(true), 0.0, &decision));

        assert!(!is_vote_correct(&VoteValue::Binary(true), 0.5, &decision));
        assert!(!is_vote_correct(&VoteValue::Binary(false), 0.5, &decision));

        assert!(!is_vote_correct(&VoteValue::Abstain, 1.0, &decision));
    }

    #[test]
    fn test_is_vote_correct_categorical_real_winner() {
        use crate::state::decisions::{Decision, DecisionType};
        use crate::state::voting::types::VoteValue;

        let decision = Decision::new(
            [0u8; 20],
            DecisionType::Category {
                options: vec!["A".into(), "B".into(), "C".into()],
            },
            "Test".to_string(),
            String::new(),
            None,
            None,
            vec![],
        )
        .unwrap();

        assert!(is_vote_correct(&VoteValue::Categorical(1), 1.0, &decision));
        assert!(!is_vote_correct(&VoteValue::Categorical(0), 1.0, &decision));
        assert!(!is_vote_correct(&VoteValue::Categorical(3), 1.0, &decision));
    }

    #[test]
    fn test_is_vote_correct_categorical_inconclusive() {
        use crate::state::decisions::{Decision, DecisionType};
        use crate::state::voting::types::VoteValue;

        let decision = Decision::new(
            [0u8; 20],
            DecisionType::Category {
                options: vec!["A".into(), "B".into(), "C".into()],
            },
            "Test".to_string(),
            String::new(),
            None,
            None,
            vec![],
        )
        .unwrap();

        assert!(is_vote_correct(&VoteValue::Categorical(3), 0.5, &decision));
        assert!(!is_vote_correct(&VoteValue::Categorical(0), 0.5, &decision));
        assert!(!is_vote_correct(&VoteValue::Categorical(2), 0.5, &decision));
        assert!(!is_vote_correct(&VoteValue::Abstain, 0.5, &decision));
    }

    #[test]
    fn test_is_vote_correct_scaled() {
        use crate::state::decisions::{Decision, DecisionType};
        use crate::state::voting::types::VoteValue;

        let decision = Decision::new(
            [0u8; 20],
            DecisionType::Scaled { min: 0, max: 100 },
            "Test".to_string(),
            "Scaled decision".to_string(),
            None,
            None,
            vec![],
        )
        .unwrap();

        assert!(is_vote_correct(&VoteValue::Scalar(0.5), 0.5, &decision));
        assert!(is_vote_correct(&VoteValue::Scalar(0.55), 0.5, &decision));
        assert!(is_vote_correct(&VoteValue::Scalar(0.45), 0.5, &decision));
        assert!(!is_vote_correct(&VoteValue::Scalar(0.56), 0.5, &decision));
        assert!(!is_vote_correct(&VoteValue::Abstain, 0.5, &decision));
    }
}
