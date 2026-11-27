use super::*;
use crate::state::decisions::DecisionId;
use crate::types::Address;

fn create_test_voter_ids(count: usize) -> Vec<Address> {
    (0..count)
        .map(|i| {
            let mut addr_bytes = [0u8; 20];
            addr_bytes[0] = i as u8;
            Address(addr_bytes)
        })
        .collect()
}

fn create_test_decision_ids(count: usize) -> Vec<DecisionId> {
    (0..count)
        .map(|i| DecisionId::new(true, 1, i as u32).unwrap())
        .collect()
}

/// Every voter with a vote, revealed in block 1 as in production
fn revealed(matrix: &SparseVoteMatrix) -> HashMap<Address, u32> {
    matrix
        .get_voters()
        .iter()
        .filter(|voter| !matrix.get_voter_votes(**voter).is_empty())
        .map(|voter| (*voter, 1))
        .collect()
}

#[test]
fn test_sparse_vote_matrix_creation() {
    let voters = create_test_voter_ids(3);
    let decisions = create_test_decision_ids(4);

    let matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());

    assert_eq!(matrix.dimensions(), (3, 4));
    assert_eq!(matrix.num_votes(), 0);

    let retrieved_voters = matrix.get_voters();
    let retrieved_decisions = matrix.get_decisions();
    assert_eq!(retrieved_voters.len(), 3);
    assert_eq!(retrieved_decisions.len(), 4);

    for voter in &voters {
        assert!(retrieved_voters.contains(voter));
    }
    for decision in &decisions {
        assert!(retrieved_decisions.contains(decision));
    }
}

#[test]
fn test_sparse_matrix_votes() {
    let voters = create_test_voter_ids(3);
    let decisions = create_test_decision_ids(2);
    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());

    matrix.set_vote(voters[0], decisions[0], 1.0).unwrap();
    matrix.set_vote(voters[1], decisions[0], 0.0).unwrap();
    matrix.set_vote(voters[2], decisions[1], 0.75).unwrap();

    assert_eq!(matrix.get_vote(voters[0], decisions[0]), Some(1.0));
    assert_eq!(matrix.get_vote(voters[1], decisions[0]), Some(0.0));
    assert_eq!(matrix.get_vote(voters[2], decisions[1]), Some(0.75));

    assert_eq!(matrix.get_vote(voters[0], decisions[1]), None);
    assert_eq!(matrix.get_vote(voters[1], decisions[1]), None);
    assert_eq!(matrix.get_vote(voters[2], decisions[0]), None);

    assert_eq!(matrix.num_votes(), 3);
}

#[test]
fn test_sparse_vote_matrix_errors() {
    let voters = create_test_voter_ids(2);
    let decisions = create_test_decision_ids(2);
    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());

    let invalid_voter = Address([99u8; 20]);
    let result = matrix.set_vote(invalid_voter, decisions[0], 1.0);
    assert!(result.is_err());

    let invalid_decision = DecisionId::new(true, 2, 0).unwrap();
    let result = matrix.set_vote(voters[0], invalid_decision, 1.0);
    assert!(result.is_err());
}

#[test]
fn test_sparse_matrix_queries() {
    let voters = create_test_voter_ids(3);
    let decisions = create_test_decision_ids(3);
    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());

    matrix.set_vote(voters[0], decisions[0], 1.0).unwrap();
    matrix.set_vote(voters[0], decisions[1], 0.8).unwrap();
    matrix.set_vote(voters[1], decisions[0], 0.0).unwrap();
    matrix.set_vote(voters[2], decisions[2], 0.6).unwrap();

    let voter0_votes = matrix.get_voter_votes(voters[0]);
    assert_eq!(voter0_votes.len(), 2);
    assert_eq!(voter0_votes.get(&decisions[0]), Some(&1.0));
    assert_eq!(voter0_votes.get(&decisions[1]), Some(&0.8));

    let decision0_votes = matrix.get_decision_votes(decisions[0]);
    assert_eq!(decision0_votes.len(), 2);
    assert_eq!(decision0_votes.get(&voters[0]), Some(&1.0));
    assert_eq!(decision0_votes.get(&voters[1]), Some(&0.0));
}

#[test]
fn test_reputation_vector_basic() {
    let mut reputation = VotingWeightVector::new();
    let voters = create_test_voter_ids(3);

    assert!(reputation.is_empty());
    assert_eq!(reputation.len(), 0);

    reputation.set_reputation(voters[0], 0.8);
    reputation.set_reputation(voters[1], 0.6);
    reputation.set_reputation(voters[2], 0.4);

    assert!(!reputation.is_empty());
    assert_eq!(reputation.len(), 3);

    assert_eq!(reputation.get_reputation(voters[0]), 0.8);
    assert_eq!(reputation.get_reputation(voters[1]), 0.6);
    assert_eq!(reputation.get_reputation(voters[2]), 0.4);

    // Unknown voter should return 0.0
    let unknown_voter = Address([99u8; 20]);
    assert_eq!(reputation.get_reputation(unknown_voter), 0.0);
}

#[test]
fn test_reputation_vector_clamping() {
    let mut reputation = VotingWeightVector::new();
    let voter = create_test_voter_ids(1)[0];

    // Values outside [0, 1] get clamped
    reputation.set_reputation(voter, -0.5);
    assert_eq!(reputation.get_reputation(voter), 0.0);

    reputation.set_reputation(voter, 1.5);
    assert_eq!(reputation.get_reputation(voter), 1.0);
}

fn uniform_rep(num_voters: usize) -> nalgebra::DVector<f64> {
    nalgebra::DVector::from_element(num_voters, 1.0)
}

/// A high-reputation minority beats a low-reputation headcount majority.
#[test]
fn test_plurality_rep_over_headcount() {
    let mut matrix = nalgebra::DMatrix::from_element(6, 1, f64::NAN);
    matrix[(0, 0)] = 0.0;
    for row in 1..6 {
        matrix[(row, 0)] = 1.0;
    }
    let rep =
        nalgebra::DVector::from_vec(vec![0.9, 0.02, 0.02, 0.02, 0.02, 0.02]);

    let mut categorical_columns = std::collections::HashMap::new();
    categorical_columns.insert(0_usize, 3_u16);

    let results = super::oracle::resolve_categorical_plurality(
        &mut matrix,
        &categorical_columns,
        &rep,
    );

    assert_eq!(results[&0].winning_option, Some(0));
    assert_eq!(matrix[(0, 0)], 1.0);
    assert_eq!(matrix[(1, 0)], 0.0);
}

#[test]
fn test_plurality_clear_winner() {
    let mut matrix = nalgebra::DMatrix::from_element(3, 1, f64::NAN);
    matrix[(0, 0)] = 1.0;
    matrix[(1, 0)] = 1.0;
    matrix[(2, 0)] = 2.0;

    let mut categorical_columns = std::collections::HashMap::new();
    categorical_columns.insert(0_usize, 3_u16);

    let results = super::oracle::resolve_categorical_plurality(
        &mut matrix,
        &categorical_columns,
        &uniform_rep(3),
    );

    assert_eq!(results[&0].winning_option, Some(1));
    assert_eq!(matrix[(0, 0)], 1.0);
    assert_eq!(matrix[(1, 0)], 1.0);
    assert_eq!(matrix[(2, 0)], 0.0);
}

#[test]
fn test_plurality_three_way_tie() {
    let mut matrix = nalgebra::DMatrix::from_element(3, 1, f64::NAN);
    matrix[(0, 0)] = 0.0;
    matrix[(1, 0)] = 1.0;
    matrix[(2, 0)] = 2.0;

    let mut categorical_columns = std::collections::HashMap::new();
    categorical_columns.insert(0_usize, 3_u16);

    let results = super::oracle::resolve_categorical_plurality(
        &mut matrix,
        &categorical_columns,
        &uniform_rep(3),
    );

    assert_eq!(results[&0].winning_option, None);
    assert!(results[&0].tied);
    assert_eq!(matrix[(0, 0)], 0.0);
    assert_eq!(matrix[(1, 0)], 1.0);
    assert_eq!(matrix[(2, 0)], 2.0);
}

#[test]
fn test_plurality_two_way_tie() {
    let mut matrix = nalgebra::DMatrix::from_element(4, 1, f64::NAN);
    matrix[(0, 0)] = 0.0;
    matrix[(1, 0)] = 1.0;
    matrix[(2, 0)] = 0.0;
    matrix[(3, 0)] = 1.0;

    let mut categorical_columns = std::collections::HashMap::new();
    categorical_columns.insert(0_usize, 4_u16);

    let results = super::oracle::resolve_categorical_plurality(
        &mut matrix,
        &categorical_columns,
        &uniform_rep(4),
    );

    assert_eq!(results[&0].winning_option, None);
    assert!(results[&0].tied);
}

#[test]
fn test_plurality_inconclusive_wins() {
    let mut matrix = nalgebra::DMatrix::from_element(5, 1, f64::NAN);
    matrix[(0, 0)] = 0.0;
    matrix[(1, 0)] = 1.0;
    matrix[(2, 0)] = 3.0;
    matrix[(3, 0)] = 3.0;
    matrix[(4, 0)] = 3.0;

    let mut categorical_columns = std::collections::HashMap::new();
    categorical_columns.insert(0_usize, 3_u16);

    let results = super::oracle::resolve_categorical_plurality(
        &mut matrix,
        &categorical_columns,
        &uniform_rep(5),
    );

    assert_eq!(results[&0].winning_option, Some(3));
    assert_eq!(matrix[(0, 0)], 0.0);
    assert_eq!(matrix[(1, 0)], 0.0);
    assert_eq!(matrix[(2, 0)], 1.0);
    assert_eq!(matrix[(3, 0)], 1.0);
    assert_eq!(matrix[(4, 0)], 1.0);
}

#[test]
fn test_plurality_tie_with_inconclusive() {
    let mut matrix = nalgebra::DMatrix::from_element(5, 1, f64::NAN);
    matrix[(0, 0)] = 0.0;
    matrix[(1, 0)] = 0.0;
    matrix[(2, 0)] = 1.0;
    matrix[(3, 0)] = 1.0;
    matrix[(4, 0)] = 3.0;

    let mut categorical_columns = std::collections::HashMap::new();
    categorical_columns.insert(0_usize, 3_u16);

    let results = super::oracle::resolve_categorical_plurality(
        &mut matrix,
        &categorical_columns,
        &uniform_rep(5),
    );

    assert_eq!(results[&0].winning_option, None);
    assert!(results[&0].tied);
}

/// An inconclusive categorical winner resolves to the neutral value 0.5.
#[test]
fn test_inconclusive_outcome_neutral() {
    let voters = create_test_voter_ids(4);
    let decisions = create_test_decision_ids(1);

    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());
    matrix.set_vote(voters[0], decisions[0], 0.0).unwrap();
    matrix.set_vote(voters[1], decisions[0], 1.0).unwrap();
    matrix.set_vote(voters[2], decisions[0], 3.0).unwrap();
    matrix.set_vote(voters[3], decisions[0], 3.0).unwrap();

    let mut reputation = VotingWeightVector::new();
    for voter in &voters {
        reputation.set_reputation(*voter, 0.25);
    }

    let scaled = std::collections::HashSet::new();
    let mut categorical = std::collections::HashMap::new();
    categorical.insert(decisions[0], 3_u16);

    let result = calculate_consensus(
        &matrix,
        &reputation,
        &scaled,
        &categorical,
        &revealed(&matrix),
    )
    .unwrap();

    assert_eq!(result.categorical_winners[&decisions[0]], Some(3));
    let outcome = result.outcomes[&decisions[0]].unwrap();
    assert!((outcome - 0.5).abs() < 1e-10);
}

#[test]
fn test_plurality_all_abstain() {
    let mut matrix = nalgebra::DMatrix::from_element(3, 1, f64::NAN);

    let mut categorical_columns = std::collections::HashMap::new();
    categorical_columns.insert(0_usize, 3_u16);

    let results = super::oracle::resolve_categorical_plurality(
        &mut matrix,
        &categorical_columns,
        &uniform_rep(3),
    );

    assert_eq!(results[&0].winning_option, None);
    assert!(matrix[(0, 0)].is_nan());
    assert!(matrix[(1, 0)].is_nan());
    assert!(matrix[(2, 0)].is_nan());
}

#[test]
fn test_plurality_single_voter() {
    let mut matrix = nalgebra::DMatrix::from_element(1, 1, f64::NAN);
    matrix[(0, 0)] = 2.0;

    let mut categorical_columns = std::collections::HashMap::new();
    categorical_columns.insert(0_usize, 3_u16);

    let results = super::oracle::resolve_categorical_plurality(
        &mut matrix,
        &categorical_columns,
        &uniform_rep(1),
    );

    assert_eq!(results[&0].winning_option, Some(2));
    assert_eq!(matrix[(0, 0)], 1.0);
}

#[test]
fn test_plurality_nan_entries() {
    let mut matrix = nalgebra::DMatrix::from_element(4, 1, f64::NAN);
    matrix[(0, 0)] = 0.0;
    matrix[(1, 0)] = 0.0;
    // rows 2,3 abstain (NaN)

    let mut categorical_columns = std::collections::HashMap::new();
    categorical_columns.insert(0_usize, 3_u16);

    let results = super::oracle::resolve_categorical_plurality(
        &mut matrix,
        &categorical_columns,
        &uniform_rep(4),
    );

    assert_eq!(results[&0].winning_option, Some(0));
    assert_eq!(matrix[(0, 0)], 1.0);
    assert_eq!(matrix[(1, 0)], 1.0);
    assert!(matrix[(2, 0)].is_nan());
    assert!(matrix[(3, 0)].is_nan());
}

#[test]
fn test_mixed_binary_categorical() {
    let voters = create_test_voter_ids(3);
    let decisions = create_test_decision_ids(2);

    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());

    matrix.set_vote(voters[0], decisions[0], 1.0).unwrap();
    matrix.set_vote(voters[1], decisions[0], 0.0).unwrap();
    matrix.set_vote(voters[2], decisions[0], 1.0).unwrap();

    matrix.set_vote(voters[0], decisions[1], 0.0).unwrap();
    matrix.set_vote(voters[1], decisions[1], 1.0).unwrap();
    matrix.set_vote(voters[2], decisions[1], 0.0).unwrap();

    let mut reputation = VotingWeightVector::new();
    reputation.set_reputation(voters[0], 0.4);
    reputation.set_reputation(voters[1], 0.3);
    reputation.set_reputation(voters[2], 0.3);

    let scaled = std::collections::HashSet::new();
    let mut categorical = std::collections::HashMap::new();
    categorical.insert(decisions[1], 3_u16);

    let result = calculate_consensus(
        &matrix,
        &reputation,
        &scaled,
        &categorical,
        &revealed(&matrix),
    )
    .unwrap();

    assert!(result.outcomes.contains_key(&decisions[0]));
    assert!(result.outcomes.contains_key(&decisions[1]));

    assert!(result.categorical_winners.contains_key(&decisions[1]));
    assert_eq!(result.categorical_winners[&decisions[1]], Some(0));
}

#[test]
fn test_categorical_clear_winner() {
    let voters = create_test_voter_ids(5);
    let decisions = create_test_decision_ids(1);

    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());

    matrix.set_vote(voters[0], decisions[0], 1.0).unwrap();
    matrix.set_vote(voters[1], decisions[0], 1.0).unwrap();
    matrix.set_vote(voters[2], decisions[0], 1.0).unwrap();
    matrix.set_vote(voters[3], decisions[0], 2.0).unwrap();
    matrix.set_vote(voters[4], decisions[0], 0.0).unwrap();

    let mut reputation = VotingWeightVector::new();
    for voter in &voters {
        reputation.set_reputation(*voter, 0.2);
    }

    let scaled = std::collections::HashSet::new();
    let mut categorical = std::collections::HashMap::new();
    categorical.insert(decisions[0], 3_u16);

    let result = calculate_consensus(
        &matrix,
        &reputation,
        &scaled,
        &categorical,
        &revealed(&matrix),
    )
    .unwrap();

    assert_eq!(result.categorical_winners[&decisions[0]], Some(1));

    let outcome = result.outcomes[&decisions[0]].unwrap();
    assert!((outcome - 1.0).abs() < 0.01);
}

/// Whitepaper figure 5: reputation is conserved and the dissenter loses.
#[test]
fn test_figure5_conservation() {
    let voters = create_test_voter_ids(7);
    let decisions = create_test_decision_ids(4);
    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());

    let vote_rows: Vec<Vec<f64>> = vec![
        vec![1.0, 0.5, 0.0, 0.0],
        vec![1.0, 0.5, 0.0, 0.0],
        vec![1.0, 1.0, 0.0, 0.0], // dissenter
        vec![1.0, 0.5, 0.0, 0.0],
        vec![1.0, 0.5, 0.0, 0.0],
        vec![1.0, 0.5, 0.0, 0.0],
        vec![1.0, 0.5, 0.0, 0.0],
    ];
    for (i, row) in vote_rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            matrix.set_vote(voters[i], decisions[j], val).unwrap();
        }
    }

    let mut reputation = VotingWeightVector::new();
    for voter in &voters {
        reputation.set_reputation(*voter, 1.0 / 7.0);
    }

    let scaled = std::collections::HashSet::new();
    let categorical = std::collections::HashMap::new();
    let result = calculate_consensus(
        &matrix,
        &reputation,
        &scaled,
        &categorical,
        &revealed(&matrix),
    )
    .unwrap();

    let original_total: f64 =
        voters.iter().map(|v| reputation.get_reputation(*v)).sum();
    let new_total: f64 = result.updated_reputations.values().sum();

    assert!(
        (original_total - new_total).abs() < 1e-10,
        "Reputation not conserved: \
         original={original_total:.15}, new={new_total:.15}, \
         diff={:.2e}",
        (original_total - new_total).abs(),
    );

    let dissenter_rep = result.updated_reputations[&voters[2]];
    let conformer_rep = result.updated_reputations[&voters[0]];
    assert!(
        dissenter_rep < conformer_rep,
        "Dissenter ({dissenter_rep:.10}) should have less \
         reputation than conformer ({conformer_rep:.10})"
    );
}

/// Reputation ranks full > partial > absent, absent below its start.
#[test]
fn test_absence_penalized() {
    let voters = create_test_voter_ids(3);
    let decisions = create_test_decision_ids(4);
    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());

    for decision in &decisions {
        matrix.set_vote(voters[0], *decision, 1.0).unwrap();
    }
    matrix.set_vote(voters[1], decisions[0], 1.0).unwrap();
    matrix.set_vote(voters[1], decisions[1], 1.0).unwrap();

    let mut reputation = VotingWeightVector::new();
    for voter in &voters {
        reputation.set_reputation(*voter, 1.0 / 3.0);
    }

    let scaled = std::collections::HashSet::new();
    let categorical = std::collections::HashMap::new();
    let result = calculate_consensus(
        &matrix,
        &reputation,
        &scaled,
        &categorical,
        &revealed(&matrix),
    )
    .unwrap();

    let rep_full = result.updated_reputations[&voters[0]];
    let rep_half = result.updated_reputations[&voters[1]];
    let rep_absent = result.updated_reputations[&voters[2]];

    assert!(
        rep_full > rep_half,
        "Full participant ({rep_full:.10}) should outrank \
         half participant ({rep_half:.10})"
    );
    assert!(
        rep_half > rep_absent,
        "Half participant ({rep_half:.10}) should outrank \
         absent voter ({rep_absent:.10})"
    );
    assert!(
        rep_absent < 1.0 / 3.0,
        "Absent voter reputation ({rep_absent:.10}) should drop \
         below starting share (0.3333333333)"
    );
}

/// Full participation: only compliance moves reputation, total conserved.
#[test]
fn test_full_participation() {
    let voters = create_test_voter_ids(5);
    let decisions = create_test_decision_ids(3);
    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());

    let rows: Vec<Vec<f64>> = vec![
        vec![1.0, 0.0, 1.0],
        vec![1.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0],
        vec![1.0, 0.0, 1.0],
    ];
    for (i, row) in rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            matrix.set_vote(voters[i], decisions[j], val).unwrap();
        }
    }

    let mut reputation = VotingWeightVector::new();
    for voter in &voters {
        reputation.set_reputation(*voter, 1.0 / 5.0);
    }

    let scaled = std::collections::HashSet::new();
    let categorical = std::collections::HashMap::new();
    let result = calculate_consensus(
        &matrix,
        &reputation,
        &scaled,
        &categorical,
        &revealed(&matrix),
    )
    .unwrap();

    let dissenter = result.updated_reputations[&voters[2]];
    let conformer = result.updated_reputations[&voters[0]];
    assert!(
        dissenter < conformer,
        "With full participation the blend is identity; dissenter \
         ({dissenter:.10}) should still trail conformer \
         ({conformer:.10})"
    );

    let original_total: f64 =
        voters.iter().map(|v| reputation.get_reputation(*v)).sum();
    let new_total: f64 = result.updated_reputations.values().sum();
    assert!(
        (original_total - new_total).abs() < 1e-9,
        "Reputation should be conserved under full participation: \
         original={original_total:.12} new={new_total:.12}"
    );
}

/// Under heavy absence a dissenting voter still outranks a fully absent one.
#[test]
fn test_high_absence() {
    let voters = create_test_voter_ids(4);
    let decisions = create_test_decision_ids(4);
    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());

    for decision in &decisions {
        matrix.set_vote(voters[0], *decision, 1.0).unwrap();
    }
    matrix.set_vote(voters[1], decisions[0], 0.0).unwrap();

    let mut reputation = VotingWeightVector::new();
    for voter in &voters {
        reputation.set_reputation(*voter, 0.25);
    }

    let scaled = std::collections::HashSet::new();
    let categorical = std::collections::HashMap::new();
    let result = calculate_consensus(
        &matrix,
        &reputation,
        &scaled,
        &categorical,
        &revealed(&matrix),
    )
    .unwrap();

    let rep_voted_wrong = result.updated_reputations[&voters[1]];
    let rep_fully_absent = result.updated_reputations[&voters[2]];

    assert!(
        rep_voted_wrong > rep_fully_absent,
        "Under heavy absence the participation term dominates: \
         a voter who showed up and disagreed ({rep_voted_wrong:.10}) \
         should still outrank a fully-absent voter \
         ({rep_fully_absent:.10})"
    );
}

/// Dust addresses gain no participation share over an equal honest voter.
#[test]
fn test_dust_no_participation_gain() {
    let voters = create_test_voter_ids(12);
    let decisions = create_test_decision_ids(4);
    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());
    let honest = voters[0];
    let dust = &voters[2..];
    for voter in std::iter::once(&honest).chain(dust) {
        for (j, decision) in decisions.iter().enumerate() {
            matrix.set_vote(*voter, *decision, (j % 2) as f64).unwrap();
        }
    }

    let mut reputation = VotingWeightVector::new();
    reputation.set_reputation(honest, 0.5);
    reputation.set_reputation(voters[1], 0.4);
    for voter in dust {
        reputation.set_reputation(*voter, 0.01);
    }

    let scaled = std::collections::HashSet::new();
    let categorical = std::collections::HashMap::new();
    let result = calculate_consensus(
        &matrix,
        &reputation,
        &scaled,
        &categorical,
        &revealed(&matrix),
    )
    .unwrap();

    let dust_total: f64 =
        dust.iter().map(|v| result.updated_reputations[v]).sum();
    let ratio = dust_total / result.updated_reputations[&honest];
    assert!(
        ratio <= 0.1 / 0.5 + 1e-6,
        "dust grew relative to an identical honest voter: {ratio:.10}"
    );
}

/// When no voter casts a vote, reputation is unchanged
#[test]
fn test_no_votes_unchanged() {
    let voters = create_test_voter_ids(3);
    let decisions = create_test_decision_ids(2);
    let matrix = SparseVoteMatrix::new(voters.clone(), decisions);

    let mut reputation = VotingWeightVector::new();
    for (voter, rep) in voters.iter().zip([0.6, 0.3, 0.1]) {
        reputation.set_reputation(*voter, rep);
    }

    let scaled = std::collections::HashSet::new();
    let categorical = std::collections::HashMap::new();
    let result = calculate_consensus(
        &matrix,
        &reputation,
        &scaled,
        &categorical,
        &revealed(&matrix),
    )
    .unwrap();

    for voter in &voters {
        assert!(
            (result.updated_reputations[voter]
                - reputation.get_reputation(*voter))
            .abs()
                < 1e-9
        );
    }
}

/// Non-voting dust addresses do not shield a dissenter from its penalty.
#[test]
fn test_dust_keeps_dissent_penalty() {
    let voters = create_test_voter_ids(203);
    let decisions = create_test_decision_ids(4);
    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());
    let dissenter = voters[2];
    for decision in &decisions {
        matrix.set_vote(voters[0], *decision, 1.0).unwrap();
        matrix.set_vote(voters[1], *decision, 1.0).unwrap();
        matrix.set_vote(dissenter, *decision, 0.0).unwrap();
    }

    let mut reputation = VotingWeightVector::new();
    reputation.set_reputation(voters[0], 0.45);
    reputation.set_reputation(voters[1], 0.45);
    reputation.set_reputation(dissenter, 0.08);
    for voter in &voters[3..] {
        reputation.set_reputation(*voter, 0.0001);
    }

    let scaled = std::collections::HashSet::new();
    let categorical = std::collections::HashMap::new();
    let result = calculate_consensus(
        &matrix,
        &reputation,
        &scaled,
        &categorical,
        &revealed(&matrix),
    )
    .unwrap();

    let after = result.updated_reputations[&dissenter];
    assert!(
        after < 0.07,
        "non-voting dust shielded the dissenter: {after:.10}"
    );
}

fn categorical_consensus(
    votes_and_reps: &[(f64, f64)],
) -> (Vec<Address>, VotingWeightVector, DetailedConsensusResult) {
    let voters = create_test_voter_ids(votes_and_reps.len());
    let decisions = create_test_decision_ids(1);
    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());
    let mut reputation = VotingWeightVector::new();
    for (voter, &(vote, rep)) in voters.iter().zip(votes_and_reps) {
        matrix.set_vote(*voter, decisions[0], vote).unwrap();
        reputation.set_reputation(*voter, rep);
    }
    let scaled = std::collections::HashSet::new();
    let mut categorical = std::collections::HashMap::new();
    categorical.insert(decisions[0], 3_u16);
    let result = calculate_consensus(
        &matrix,
        &reputation,
        &scaled,
        &categorical,
        &revealed(&matrix),
    )
    .unwrap();
    (voters, reputation, result)
}

/// A plurality winner without a majority keeps reputation; losers drop.
#[test]
fn test_categorical_no_majority() {
    let (voters, reputation, result) =
        categorical_consensus(&[(0.0, 0.35), (1.0, 0.33), (2.0, 0.32)]);

    assert_eq!(result.categorical_winners.values().next(), Some(&Some(0)));
    let winner_after = result.updated_reputations[&voters[0]];
    assert!(
        winner_after >= reputation.get_reputation(voters[0]),
        "the plurality winner lost reputation: {winner_after:.10}"
    );
    assert!(
        result.updated_reputations[&voters[1]]
            < reputation.get_reputation(voters[1])
    );
}

/// An outright inconclusive win does not cost its voters reputation.
#[test]
fn test_inconclusive_win_rewards() {
    let (voters, reputation, result) =
        categorical_consensus(&[(3.0, 0.4), (0.0, 0.3), (1.0, 0.3)]);

    assert_eq!(result.categorical_winners.values().next(), Some(&Some(3)));
    assert!(
        result.updated_reputations[&voters[0]]
            >= reputation.get_reputation(voters[0])
    );
}

/// A categorical tie resolves to 0.5 and changes no reputation.
#[test]
fn test_categorical_tie_no_change() {
    let (voters, reputation, result) =
        categorical_consensus(&[(0.0, 0.5), (1.0, 0.5)]);

    assert_eq!(result.categorical_winners.values().next(), Some(&None));
    assert_eq!(result.outcomes.values().next(), Some(&Some(0.5)));
    for voter in &voters {
        assert_eq!(
            result.updated_reputations[voter],
            reputation.get_reputation(*voter)
        );
    }
}

/// resolve_unguaranteed handles a tie, an inconclusive win and a plain winner.
#[test]
fn test_unguaranteed_categorical() {
    use super::oracle::resolve_unguaranteed;

    assert_eq!(
        resolve_unguaranteed(&[(0.0, 0.5), (1.0, 0.5)], false, Some(3))
            .unwrap(),
        Some((0.5, None))
    );
    assert_eq!(
        resolve_unguaranteed(&[(3.0, 0.6), (1.0, 0.4)], false, Some(3))
            .unwrap(),
        Some((0.5, Some(3)))
    );
    assert_eq!(
        resolve_unguaranteed(&[(1.0, 0.6), (0.0, 0.4)], false, Some(3))
            .unwrap(),
        Some((1.0, Some(1)))
    );
}

#[derive(Clone, Copy)]
enum Kind {
    Binary,
    Scaled,
    Categorical(u16),
}

type Column = (Kind, &'static [Option<f64>]);

struct CampResult {
    outcomes: Vec<Option<f64>>,
    reputations: Vec<Vec<f64>>,
}

impl CampResult {
    fn total(&self, camp: usize) -> f64 {
        self.reputations[camp].iter().sum()
    }
}

fn camp_consensus(camps: &[&[f64]], columns: &[Column]) -> CampResult {
    let num_voters = camps.iter().map(|camp| camp.len()).sum();
    let voters = create_test_voter_ids(num_voters);
    let decisions = create_test_decision_ids(columns.len());
    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());
    let mut reputation = VotingWeightVector::new();
    let mut scaled = HashSet::new();
    let mut categorical = HashMap::new();
    for (decision, (kind, _)) in decisions.iter().zip(columns) {
        match kind {
            Kind::Binary => {}
            Kind::Scaled => {
                scaled.insert(*decision);
            }
            Kind::Categorical(option_count) => {
                categorical.insert(*decision, *option_count);
            }
        }
    }
    let mut camp_voters = Vec::new();
    let mut next = voters.iter();
    for (camp_idx, camp) in camps.iter().enumerate() {
        let members: Vec<Address> =
            next.by_ref().take(camp.len()).copied().collect();
        for (voter, rep) in members.iter().zip(camp.iter()) {
            reputation.set_reputation(*voter, *rep);
            for (decision, (_, votes)) in decisions.iter().zip(columns) {
                if let Some(vote) = votes[camp_idx] {
                    matrix.set_vote(*voter, *decision, vote).unwrap();
                }
            }
        }
        camp_voters.push(members);
    }
    let result = calculate_consensus(
        &matrix,
        &reputation,
        &scaled,
        &categorical,
        &revealed(&matrix),
    )
    .unwrap();
    CampResult {
        outcomes: decisions.iter().map(|d| result.outcomes[d]).collect(),
        reputations: camp_voters
            .iter()
            .map(|members| {
                members
                    .iter()
                    .map(|voter| result.updated_reputations[voter])
                    .collect()
            })
            .collect(),
    }
}

fn assert_split_invariant(
    unsplit: &[&[f64]],
    split: &[&[f64]],
    columns: &[Column],
) {
    let whole = camp_consensus(unsplit, columns);
    let pieces = camp_consensus(split, columns);
    for (a, b) in whole.outcomes.iter().zip(&pieces.outcomes) {
        match (a, b) {
            (Some(a), Some(b)) => assert!((a - b).abs() < 1e-8, "{a} vs {b}"),
            (a, b) => assert_eq!(a, b),
        }
    }
    for camp in 0..unsplit.len() {
        let (a, b) = (whole.total(camp), pieces.total(camp));
        assert!((a - b).abs() < 1e-5, "camp {camp}: {a} vs {b}");
    }
}

fn assert_reputation_unchanged(camps: &[&[f64]], result: &CampResult) {
    for (inputs, updated) in camps.iter().zip(&result.reputations) {
        for (input, output) in inputs.iter().zip(updated) {
            assert!((input - output).abs() < 1e-6, "{input} vs {output}");
        }
    }
}

const BINARY_STRAGGLER: [Column; 3] = [
    (Kind::Binary, &[Some(0.0), Some(1.0), Some(0.75)]),
    (Kind::Binary, &[Some(0.0), Some(1.0), Some(0.75)]),
    (Kind::Binary, &[Some(0.0), Some(1.0), Some(0.0)]),
];

/// 50/45/5 binary camps: splitting the 45% camp 5 ways changes nothing.
#[test]
fn split_minority_binary() {
    assert_split_invariant(
        &[&[0.5], &[0.45], &[0.05]],
        &[&[0.5], &[0.09; 5], &[0.05]],
        &BINARY_STRAGGLER,
    );
}

/// 50/45/5 binary camps: splitting the 50% camp 5 ways changes nothing.
#[test]
fn split_majority_binary() {
    assert_split_invariant(
        &[&[0.5], &[0.45], &[0.05]],
        &[&[0.1; 5], &[0.45], &[0.05]],
        &BINARY_STRAGGLER,
    );
}

/// Both camps split 3 and 7 ways match the unsplit result.
#[test]
fn split_both_camps() {
    assert_split_invariant(
        &[&[0.51], &[0.42], &[0.07]],
        &[&[0.17; 3], &[0.06; 7], &[0.07]],
        &BINARY_STRAGGLER,
    );
}

/// The 45% camp split into unequal pieces changes nothing.
#[test]
fn split_uneven_pieces() {
    assert_split_invariant(
        &[&[0.5], &[0.45], &[0.05]],
        &[&[0.5], &[0.2, 0.15, 0.1], &[0.05]],
        &BINARY_STRAGGLER,
    );
}

/// A 20-way split of the 40% camp changes nothing.
#[test]
fn split_twenty_ways() {
    assert_split_invariant(
        &[&[0.5], &[0.4], &[0.1]],
        &[&[0.5], &[0.02; 20], &[0.1]],
        &BINARY_STRAGGLER,
    );
}

/// Scaled 0.2 vs 0.8 at 50/45/5: splitting the 45% camp changes nothing.
#[test]
fn split_scaled_votes() {
    assert_split_invariant(
        &[&[0.5], &[0.45], &[0.05]],
        &[&[0.5], &[0.09; 5], &[0.05]],
        &[
            (Kind::Scaled, &[Some(0.2), Some(0.8), Some(0.65)]),
            (Kind::Scaled, &[Some(0.2), Some(0.8), Some(0.65)]),
            (Kind::Scaled, &[Some(0.2), Some(0.8), Some(0.2)]),
        ],
    );
}

/// Scaled 0.0 vs 1.0 at 50/45/5: splitting the 45% camp changes nothing.
#[test]
fn split_scaled_extremes() {
    assert_split_invariant(
        &[&[0.5], &[0.45], &[0.05]],
        &[&[0.5], &[0.09; 5], &[0.05]],
        &[
            (Kind::Scaled, &[Some(0.0), Some(1.0), Some(0.75)]),
            (Kind::Scaled, &[Some(0.0), Some(1.0), Some(0.75)]),
            (Kind::Scaled, &[Some(0.0), Some(1.0), Some(0.0)]),
        ],
    );
}

/// Binary, scaled and a decided categorical in one period: splitting
/// changes nothing.
#[test]
fn split_mixed_period() {
    assert_split_invariant(
        &[&[0.5], &[0.45], &[0.05]],
        &[&[0.5], &[0.09; 5], &[0.05]],
        &[
            (Kind::Binary, &[Some(0.0), Some(1.0), Some(0.75)]),
            (Kind::Scaled, &[Some(0.2), Some(0.8), Some(0.65)]),
            (Kind::Categorical(2), &[Some(0.0), Some(1.0), Some(0.0)]),
        ],
    );
}

/// Camp B abstains on one contested decision: splitting B changes nothing.
#[test]
fn split_with_abstentions() {
    assert_split_invariant(
        &[&[0.5], &[0.45], &[0.05]],
        &[&[0.5], &[0.09; 5], &[0.05]],
        &[
            (Kind::Binary, &[Some(0.0), Some(1.0), Some(0.75)]),
            (Kind::Binary, &[Some(0.0), None, Some(0.75)]),
            (Kind::Binary, &[Some(0.0), Some(1.0), Some(0.0)]),
        ],
    );
}

/// 70/30 with the minority split 10 ways: the majority still gains
/// reputation and wins every contested decision.
#[test]
fn majority_still_rewarded() {
    let result = camp_consensus(
        &[&[0.7], &[0.03; 10]],
        &[
            (Kind::Binary, &[Some(0.0), Some(1.0)]),
            (Kind::Binary, &[Some(0.0), Some(1.0)]),
            (Kind::Scaled, &[Some(0.2), Some(0.8)]),
        ],
    );
    assert!(result.total(0) > 0.7);
    assert_eq!(result.outcomes, vec![Some(0.0), Some(0.0), Some(0.2)]);
}

/// 50/50 binary camps with B split 5 ways: no reputation moves and the
/// contested decisions resolve 0.5.
#[test]
fn deadlock_binary_only() {
    let camps: [&[f64]; 2] = [&[0.5], &[0.1; 5]];
    let result = camp_consensus(
        &camps,
        &[
            (Kind::Binary, &[Some(0.0), Some(1.0)]),
            (Kind::Binary, &[Some(0.0), Some(1.0)]),
        ],
    );
    assert_reputation_unchanged(&camps, &result);
    assert_eq!(result.outcomes, vec![Some(0.5), Some(0.5)]);
}

/// 50/50 at scaled 0.25 vs 0.75 with B split 5 ways: no reputation moves
/// and the outcome is the old-reputation median midpoint.
#[test]
fn deadlock_scaled_exact() {
    let camps: [&[f64]; 2] = [&[0.5], &[0.1; 5]];
    let result = camp_consensus(
        &camps,
        &[
            (Kind::Scaled, &[Some(0.25), Some(0.75)]),
            (Kind::Scaled, &[Some(0.25), Some(0.75)]),
        ],
    );
    assert_reputation_unchanged(&camps, &result);
    assert_eq!(result.outcomes, vec![Some(0.5), Some(0.5)]);
}

/// Binary plus scaled 0.2 vs 0.8, whose distances differ by one ULP, with
/// A split 5 ways: still a deadlock.
#[test]
fn deadlock_rounding_noise() {
    let camps: [&[f64]; 2] = [&[0.1; 5], &[0.5]];
    let result = camp_consensus(
        &camps,
        &[
            (Kind::Binary, &[Some(0.0), Some(1.0)]),
            (Kind::Scaled, &[Some(0.2), Some(0.8)]),
            (Kind::Scaled, &[Some(0.2), Some(0.8)]),
        ],
    );
    assert_reputation_unchanged(&camps, &result);
    assert_eq!(result.outcomes, vec![Some(0.5), Some(0.5), Some(0.5)]);
}

/// 55/45 binary camps: the reference still catches to 0.5, so no
/// reputation moves and the decisions resolve 0.5.
#[test]
fn deadlock_unequal_reps() {
    let camps: [&[f64]; 2] = [&[0.55], &[0.45]];
    let result = camp_consensus(
        &camps,
        &[
            (Kind::Binary, &[Some(0.0), Some(1.0)]),
            (Kind::Binary, &[Some(0.0), Some(1.0)]),
        ],
    );
    assert_reputation_unchanged(&camps, &result);
    assert_eq!(result.outcomes, vec![Some(0.5), Some(0.5)]);
}

/// A binary deadlock plus a tied categorical: the categorical is skipped
/// and nothing moves.
#[test]
fn deadlock_categorical_tie() {
    let camps: [&[f64]; 2] = [&[0.5], &[0.1; 5]];
    let result = camp_consensus(
        &camps,
        &[
            (Kind::Binary, &[Some(0.0), Some(1.0)]),
            (Kind::Binary, &[Some(0.0), Some(1.0)]),
            (Kind::Categorical(2), &[Some(0.0), Some(1.0)]),
        ],
    );
    assert_reputation_unchanged(&camps, &result);
    assert_eq!(result.outcomes, vec![Some(0.5), Some(0.5), Some(0.5)]);
}

/// A binary deadlock gives the same result with B in 1 or 9 addresses.
#[test]
fn deadlock_split_irrelevant() {
    let columns: [Column; 2] = [
        (Kind::Binary, &[Some(0.0), Some(1.0)]),
        (Kind::Binary, &[Some(0.0), Some(1.0)]),
    ];
    assert_split_invariant(
        &[&[0.5], &[0.5]],
        &[&[0.5], &[0.5 / 9.0; 9]],
        &columns,
    );
}

/// 65/35 binary camps sit outside the catch band, so A is rewarded.
#[test]
fn no_deadlock_outside_band() {
    let result = camp_consensus(
        &[&[0.65], &[0.35]],
        &[
            (Kind::Binary, &[Some(0.0), Some(1.0)]),
            (Kind::Binary, &[Some(0.0), Some(1.0)]),
        ],
    );
    assert!(result.total(0) > 0.65);
}

/// A 51/49 binary deadlock plus a categorical A wins: the pinned column
/// breaks the deadlock and A is rewarded.
#[test]
fn decided_categorical_anchors() {
    let result = camp_consensus(
        &[&[0.51], &[0.049; 10]],
        &[
            (Kind::Binary, &[Some(0.0), Some(1.0)]),
            (Kind::Binary, &[Some(0.0), Some(1.0)]),
            (Kind::Categorical(2), &[Some(0.0), Some(1.0)]),
        ],
    );
    assert!(result.total(0) > 0.51);
    assert_eq!(result.outcomes, vec![Some(0.0), Some(0.0), Some(0.0)]);
}
