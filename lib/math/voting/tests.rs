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
fn test_sparse_vote_matrix_vote_operations() {
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

#[test]
fn test_resolve_categorical_plurality_clear_winner() {
    let mut matrix = nalgebra::DMatrix::from_element(3, 1, f64::NAN);
    matrix[(0, 0)] = 1.0;
    matrix[(1, 0)] = 1.0;
    matrix[(2, 0)] = 2.0;

    let mut categorical_columns = std::collections::HashMap::new();
    categorical_columns.insert(0_usize, 3_u16);

    let results = super::oracle::resolve_categorical_plurality(
        &mut matrix,
        &categorical_columns,
    );

    assert_eq!(results[&0].winning_option, Some(1));
    assert_eq!(matrix[(0, 0)], 1.0);
    assert_eq!(matrix[(1, 0)], 1.0);
    assert_eq!(matrix[(2, 0)], 0.0);
}

#[test]
fn test_resolve_categorical_plurality_three_way_tie() {
    let mut matrix = nalgebra::DMatrix::from_element(3, 1, f64::NAN);
    matrix[(0, 0)] = 0.0;
    matrix[(1, 0)] = 1.0;
    matrix[(2, 0)] = 2.0;

    let mut categorical_columns = std::collections::HashMap::new();
    categorical_columns.insert(0_usize, 3_u16);

    let results = super::oracle::resolve_categorical_plurality(
        &mut matrix,
        &categorical_columns,
    );

    assert_eq!(results[&0].winning_option, Some(3));
    assert_eq!(matrix[(0, 0)], 0.0);
    assert_eq!(matrix[(1, 0)], 0.0);
    assert_eq!(matrix[(2, 0)], 0.0);
}

#[test]
fn test_resolve_categorical_plurality_two_way_tie() {
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
    );

    assert_eq!(results[&0].winning_option, Some(4));
    for row in 0..4 {
        assert_eq!(matrix[(row, 0)], 0.0);
    }
}

#[test]
fn test_resolve_categorical_inconclusive_wins_outright() {
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
    );

    assert_eq!(results[&0].winning_option, Some(3));
    assert_eq!(matrix[(0, 0)], 0.0);
    assert_eq!(matrix[(1, 0)], 0.0);
    assert_eq!(matrix[(2, 0)], 1.0);
    assert_eq!(matrix[(3, 0)], 1.0);
    assert_eq!(matrix[(4, 0)], 1.0);
}

#[test]
fn test_resolve_categorical_tie_with_inconclusive_voters() {
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
    );

    assert_eq!(results[&0].winning_option, Some(3));
    assert_eq!(matrix[(0, 0)], 0.0);
    assert_eq!(matrix[(1, 0)], 0.0);
    assert_eq!(matrix[(2, 0)], 0.0);
    assert_eq!(matrix[(3, 0)], 0.0);
    assert_eq!(matrix[(4, 0)], 1.0);
}

#[test]
fn test_resolve_categorical_inconclusive_outcome_is_neutral_value() {
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

    let result =
        calculate_consensus(&matrix, &reputation, &scaled, &categorical)
            .unwrap();

    assert_eq!(result.categorical_winners[&decisions[0]], Some(3));
    let outcome = result.outcomes[&decisions[0]].unwrap();
    assert!((outcome - 0.5).abs() < 1e-10);
}

#[test]
fn test_resolve_categorical_plurality_all_abstain() {
    let mut matrix = nalgebra::DMatrix::from_element(3, 1, f64::NAN);

    let mut categorical_columns = std::collections::HashMap::new();
    categorical_columns.insert(0_usize, 3_u16);

    let results = super::oracle::resolve_categorical_plurality(
        &mut matrix,
        &categorical_columns,
    );

    assert_eq!(results[&0].winning_option, None);
    assert!(matrix[(0, 0)].is_nan());
    assert!(matrix[(1, 0)].is_nan());
    assert!(matrix[(2, 0)].is_nan());
}

#[test]
fn test_resolve_categorical_plurality_single_voter() {
    let mut matrix = nalgebra::DMatrix::from_element(1, 1, f64::NAN);
    matrix[(0, 0)] = 2.0;

    let mut categorical_columns = std::collections::HashMap::new();
    categorical_columns.insert(0_usize, 3_u16);

    let results = super::oracle::resolve_categorical_plurality(
        &mut matrix,
        &categorical_columns,
    );

    assert_eq!(results[&0].winning_option, Some(2));
    assert_eq!(matrix[(0, 0)], 1.0);
}

#[test]
fn test_resolve_categorical_with_nan_entries() {
    let mut matrix = nalgebra::DMatrix::from_element(4, 1, f64::NAN);
    matrix[(0, 0)] = 0.0;
    matrix[(1, 0)] = 0.0;
    // rows 2,3 abstain (NaN)

    let mut categorical_columns = std::collections::HashMap::new();
    categorical_columns.insert(0_usize, 3_u16);

    let results = super::oracle::resolve_categorical_plurality(
        &mut matrix,
        &categorical_columns,
    );

    assert_eq!(results[&0].winning_option, Some(0));
    assert_eq!(matrix[(0, 0)], 1.0);
    assert_eq!(matrix[(1, 0)], 1.0);
    assert!(matrix[(2, 0)].is_nan());
    assert!(matrix[(3, 0)].is_nan());
}

#[test]
fn test_mixed_binary_and_categorical_columns() {
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

    let result =
        calculate_consensus(&matrix, &reputation, &scaled, &categorical)
            .unwrap();

    assert!(result.outcomes.contains_key(&decisions[0]));
    assert!(result.outcomes.contains_key(&decisions[1]));

    assert!(result.categorical_winners.contains_key(&decisions[1]));
    assert_eq!(result.categorical_winners[&decisions[1]], Some(0));
}

#[test]
fn test_consensus_categorical_clear_winner() {
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

    let result =
        calculate_consensus(&matrix, &reputation, &scaled, &categorical)
            .unwrap();

    assert_eq!(result.categorical_winners[&decisions[0]], Some(1));

    let outcome = result.outcomes[&decisions[0]].unwrap();
    assert!((outcome - 1.0).abs() < 0.01);
}

#[test]
fn test_reputation_conservation_whitepaper_figure5() {
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
    let result =
        calculate_consensus(&matrix, &reputation, &scaled, &categorical)
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

#[test]
fn test_absent_voter_penalized_relative_to_attendance() {
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
    let result =
        calculate_consensus(&matrix, &reputation, &scaled, &categorical)
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

#[test]
fn test_full_participation_leaves_compliance_only() {
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
    let result =
        calculate_consensus(&matrix, &reputation, &scaled, &categorical)
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

#[test]
fn test_high_absence_dominates_consensus() {
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
    let result =
        calculate_consensus(&matrix, &reputation, &scaled, &categorical)
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
