use super::*;
use crate::state::slots::SlotId;
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

fn create_test_decision_ids(count: usize) -> Vec<SlotId> {
    (0..count)
        .map(|i| SlotId::new(1, i as u32).unwrap())
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

    let invalid_decision = SlotId::new(2, 0).unwrap();
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
fn test_reputation_vector_from_voter_reputations() {
    use crate::state::voting::types::{VoterReputation, VotingPeriodId};

    let voters = create_test_voter_ids(2);
    let mut voter_reps = std::collections::HashMap::new();

    let rep0 = VoterReputation::new(voters[0], 0.7, 0, VotingPeriodId(1));
    let rep1 = VoterReputation::new(voters[1], 0.3, 0, VotingPeriodId(1));

    voter_reps.insert(voters[0], rep0);
    voter_reps.insert(voters[1], rep1);

    let reputation = VotingWeightVector::from_voter_reputations(&voter_reps);

    assert_eq!(reputation.len(), 2);
    // get_voting_weight returns reputation * votecoin_proportion
    // With default votecoin_proportion of 1.0, should match reputation
}
