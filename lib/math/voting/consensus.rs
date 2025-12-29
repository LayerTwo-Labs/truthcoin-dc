use super::constants::{
    BITCOIN_HIVEMIND_NEUTRAL_VALUE, CONSENSUS_CATCH_TOLERANCE,
    SVD_NUMERICAL_TOLERANCE, round_outcome, round_reputation,
};
use super::{
    DetailedConsensusResult, SparseVoteMatrix, VotingMathError,
    VotingWeightVector,
};
use crate::math::safe_math::safe_cmp_f64;

use nalgebra::{DMatrix, DVector, SVD};
use std::collections::HashMap;

pub fn get_weight(vec: &DVector<f64>) -> DVector<f64> {
    let mut new_vec = vec.abs();

    if new_vec.sum() == 0.0 {
        new_vec.fill(1.0);
    }

    new_vec /= new_vec.sum();
    new_vec
}

pub fn re_weight(vector_in: &DVector<f64>) -> DVector<f64> {
    let mut out = vector_in.clone();

    out.apply(|x| {
        if x.is_nan() {
            *x = 0.0;
        } else {
            *x = x.abs();
        }
    });

    if out.sum() == 0.0 {
        out.fill(1.0);
    }

    let sum = out.sum();
    out.apply(|x| *x /= sum);

    out
}

pub fn catch_tl(x: f64, tolerance: f64) -> f64 {
    if x < (BITCOIN_HIVEMIND_NEUTRAL_VALUE - tolerance / 2.0) {
        0.0
    } else if x > (BITCOIN_HIVEMIND_NEUTRAL_VALUE + tolerance / 2.0) {
        1.0
    } else {
        BITCOIN_HIVEMIND_NEUTRAL_VALUE
    }
}

/// Compute weighted mean for each column (decision) in a vote matrix.
/// Returns (outcomes, has_votes) where has_votes[j] is true if column j had any non-NaN votes.
pub fn compute_weighted_column_means(
    matrix: &DMatrix<f64>,
    weights: &DVector<f64>,
) -> (DVector<f64>, Vec<bool>) {
    let num_decisions = matrix.ncols();
    let mut outcomes = DVector::zeros(num_decisions);
    let mut has_votes = vec![false; num_decisions];

    for j in 0..num_decisions {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for i in 0..matrix.nrows() {
            let vote = matrix[(i, j)];
            if !vote.is_nan() {
                let weight = weights[i];
                weighted_sum += vote * weight;
                total_weight += weight;
                has_votes[j] = true;
            }
        }

        outcomes[j] = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            BITCOIN_HIVEMIND_NEUTRAL_VALUE
        };
    }

    (outcomes, has_votes)
}

pub fn weighted_median(
    values: &DVector<f64>,
    weights: &DVector<f64>,
) -> Result<f64, VotingMathError> {
    if values.len() != weights.len() {
        return Err(VotingMathError::DimensionMismatch {
            expected: format!("values length {}", values.len()),
            actual: format!("weights length {}", weights.len()),
        });
    }

    if values.is_empty() {
        return Err(VotingMathError::EmptyMatrix);
    }

    let mut sorted: Vec<_> = values.iter().zip(weights.iter()).collect();
    sorted.sort_by(|a, b| safe_cmp_f64(*a.0, *b.0));

    let total_weight: f64 = weights.iter().sum();
    let mut cumulative_weight = 0.0;
    let half_weight = total_weight / 2.0;

    for (i, &(value, weight)) in sorted.iter().enumerate() {
        cumulative_weight += weight;
        if cumulative_weight >= half_weight {
            if cumulative_weight == half_weight && i < sorted.len() - 1 {
                return Ok((*value + *sorted[i + 1].0) / 2.0);
            } else {
                return Ok(*value);
            }
        }
    }

    // Safe because we checked for empty above
    Ok(*sorted.last().expect("non-empty after check").0)
}

fn score_reflect(
    component: &DVector<f64>,
    previous_reputation: &DVector<f64>,
) -> Result<DVector<f64>, VotingMathError> {
    let unique_count = component.len();
    if unique_count <= 2 {
        return Ok(component.clone());
    }

    let median_factor = weighted_median(component, previous_reputation)?;
    let reflection =
        component - DVector::from_element(component.len(), median_factor);
    let excessive = reflection.map(|x| x > 0.0);

    let mut adj_prin_comp = component.clone();

    for i in 0..adj_prin_comp.len() {
        if excessive[i] {
            adj_prin_comp[i] -= reflection[i] * 0.5;
        }
    }

    Ok(adj_prin_comp)
}

pub fn weighted_prin_comp(
    x: &DMatrix<f64>,
    weights: Option<&DVector<f64>>,
) -> Result<(DVector<f64>, DVector<f64>), VotingMathError> {
    let n_rows = x.nrows();
    let _n_cols = x.ncols();

    let weights = match weights {
        Some(w) => {
            if w.len() != n_rows {
                return Err(VotingMathError::DimensionMismatch {
                    expected: format!("weights length {n_rows}"),
                    actual: format!("weights length {}", w.len()),
                });
            }
            w.clone()
        }
        None => re_weight(&DVector::from_element(n_rows, 1.0)),
    };

    let total_weight = weights.sum();
    if total_weight <= 0.0 {
        return Err(VotingMathError::InvalidReputation {
            reason: "Total weight is non-positive".to_string(),
        });
    }

    let weighted_mean = (x.transpose() * &weights) / total_weight;

    let mut centered = x.clone();
    for mut row in centered.row_iter_mut() {
        row -= &weighted_mean.transpose();
    }

    let mut weighted_centered =
        DMatrix::zeros(centered.nrows(), centered.ncols());
    for (i, row) in centered.row_iter().enumerate() {
        weighted_centered.set_row(i, &(row * weights[i]));
    }
    let wcvm = (centered.transpose() * weighted_centered) / total_weight;

    let svd = SVD::new(wcvm.clone(), true, false);

    let first_loading = svd
        .u
        .as_ref()
        .ok_or_else(|| VotingMathError::NumericalError {
            reason: "SVD failed to compute U matrix".to_string(),
        })?
        .column(0)
        .clone_owned();

    let first_score = centered * &first_loading;

    Ok((first_score, first_loading))
}

pub struct RewardWeightsResult {
    pub first_loading: DVector<f64>,
    pub old_rep: DVector<f64>,
    pub this_rep: DVector<f64>,
    pub smooth_rep: DVector<f64>,
}

pub fn get_reward_weights(
    vote_matrix: &DMatrix<f64>,
    rep: Option<&DVector<f64>>,
    alpha: f64,
) -> Result<RewardWeightsResult, VotingMathError> {
    let num_voters = vote_matrix.nrows();

    let old_rep = match rep {
        Some(r) => r.clone(),
        None => DVector::from_element(num_voters, 1.0 / num_voters as f64),
    };

    let (first_score, first_loading) =
        weighted_prin_comp(vote_matrix, Some(&old_rep))?;

    let mut new_rep = old_rep.clone();

    if first_score.abs().sum() != 0.0 {
        let score_min =
            DVector::from_element(first_score.len(), first_score.min());
        let score_max =
            DVector::from_element(first_score.len(), first_score.max());

        let option1 = &first_score + score_min.abs();
        let option2 = (&first_score - score_max).abs();

        let new_score_1 = score_reflect(&option1, &old_rep)?;
        let new_score_2 = score_reflect(&option2, &old_rep)?;

        let (raw_outcomes, _) =
            compute_weighted_column_means(vote_matrix, &old_rep);
        let old_rep_outcomes =
            raw_outcomes.map(|x| catch_tl(x, CONSENSUS_CATCH_TOLERANCE));

        let mut distance =
            DMatrix::zeros(vote_matrix.nrows(), vote_matrix.ncols());
        for i in 0..vote_matrix.nrows() {
            for j in 0..vote_matrix.ncols() {
                let vote = vote_matrix[(i, j)];
                if !vote.is_nan() && j < old_rep_outcomes.len() {
                    distance[(i, j)] = (vote - old_rep_outcomes[j]).abs();
                }
            }
        }

        let dissent = first_loading.abs();
        let mainstream_preweight =
            dissent.map(|x| if x != 0.0 { 1.0 / x } else { f64::NAN });
        let mainstream = re_weight(&mainstream_preweight);

        let mut non_compliance =
            DMatrix::zeros(vote_matrix.nrows(), vote_matrix.ncols());
        for i in 0..vote_matrix.nrows() {
            for j in 0..vote_matrix.ncols() {
                if j < mainstream.len() {
                    non_compliance[(i, j)] = distance[(i, j)] * mainstream[j];
                }
            }
        }

        let max_non_compliance = non_compliance.max();
        let compliance = non_compliance.map(|x| (x - max_non_compliance).abs());

        let mut compliance_scores = DVector::zeros(num_voters);
        for i in 0..num_voters {
            let mut sum = 0.0;
            let mut count = 0.0;
            for j in 0..vote_matrix.ncols() {
                if j < mainstream.len() {
                    sum += compliance[(i, j)];
                    count += 1.0;
                }
            }
            compliance_scores[i] = if count > 0.0 { sum / count } else { 0.0 };
        }

        let weighted_score_1 = get_weight(&new_score_1);
        let difference_1 = &weighted_score_1 - &compliance_scores;
        let choice1: f64 = difference_1.iter().map(|&x| x.powi(2)).sum();

        let weighted_score_2 = get_weight(&new_score_2);
        let difference_2 = &weighted_score_2 - &compliance_scores;
        let choice2: f64 = difference_2.iter().map(|&x| x.powi(2)).sum();

        let ref_ind = choice1 - choice2;

        let rep_mean = old_rep.mean();
        let scaled_rep = old_rep.map(|x| x / rep_mean);

        let rep1 = get_weight(&new_score_1.component_mul(&scaled_rep));
        let rep2 = get_weight(&new_score_2.component_mul(&scaled_rep));

        new_rep = if ref_ind < 0.0 { rep1 } else { rep2 };
    }

    let old_rep_normalized = re_weight(&old_rep);
    let smooth_rep = &new_rep * alpha + &old_rep_normalized * (1.0 - alpha);

    Ok(RewardWeightsResult {
        first_loading,
        old_rep,
        this_rep: new_rep,
        smooth_rep,
    })
}

pub struct FillNaResult {
    pub filled: DMatrix<f64>,
    pub has_votes: Vec<bool>,
}

pub fn fill_na(
    m_na: &DMatrix<f64>,
    rep: Option<&DVector<f64>>,
    catch_p: f64,
) -> FillNaResult {
    let mut m_new = m_na.clone();
    let num_decisions = m_na.ncols();

    // Track which decisions have at least one vote
    let has_votes;

    let has_nan = m_na.iter().any(|&x| x.is_nan());

    if has_nan {
        let num_voters = m_na.nrows();
        let rep = match rep {
            Some(r) => r.clone(),
            None => DVector::from_element(num_voters, 1.0 / num_voters as f64),
        };

        let (raw_outcomes, votes_present) =
            compute_weighted_column_means(m_na, &rep);
        has_votes = votes_present;
        let decision_outcomes = raw_outcomes.map(|x| catch_tl(x, catch_p));

        for i in 0..m_new.nrows() {
            for j in 0..m_new.ncols() {
                if m_new[(i, j)].is_nan() {
                    m_new[(i, j)] = decision_outcomes[j];
                }
            }
        }
    } else {
        // No NaN values means every cell has a vote
        has_votes = vec![true; num_decisions];
    }

    FillNaResult {
        filled: m_new,
        has_votes,
    }
}

pub struct FactoryResult {
    pub filled: DMatrix<f64>,
    pub agents: RewardWeightsResult,
    pub decisions: DVector<f64>,
    pub has_votes: Vec<bool>,
    pub certainty: f64,
    pub first_loading: DVector<f64>,
}

pub fn factory(
    m0: &DMatrix<f64>,
    rep: Option<&DVector<f64>>,
    catch_p: Option<f64>,
) -> Result<FactoryResult, VotingMathError> {
    let catch_p = catch_p.unwrap_or(CONSENSUS_CATCH_TOLERANCE);

    let fill_result = fill_na(m0, rep, catch_p);
    let filled = fill_result.filled;
    let has_votes = fill_result.has_votes;

    use super::constants::REPUTATION_SMOOTHING_ALPHA;
    let player_info =
        get_reward_weights(&filled, rep, REPUTATION_SMOOTHING_ALPHA)?;

    let (decision_outcomes, _) =
        compute_weighted_column_means(&filled, &player_info.smooth_rep);
    let decision_outcomes_final =
        decision_outcomes.map(|x| catch_tl(x, catch_p));

    let mut certainty = vec![0.0; filled.ncols()];
    for j in 0..filled.ncols() {
        let mut sum = 0.0;
        for i in 0..filled.nrows() {
            if (decision_outcomes_final[j] - filled[(i, j)]).abs()
                < SVD_NUMERICAL_TOLERANCE
            {
                sum += player_info.smooth_rep[i];
            }
        }
        certainty[j] = sum;
    }

    let avg_certainty = certainty.iter().sum::<f64>() / certainty.len() as f64;

    Ok(FactoryResult {
        filled,
        first_loading: player_info.first_loading.clone(),
        agents: player_info,
        decisions: decision_outcomes_final,
        has_votes,
        certainty: avg_certainty,
    })
}

pub fn run_consensus(
    vote_matrix: &SparseVoteMatrix,
    reputation_vector: &VotingWeightVector,
) -> Result<DetailedConsensusResult, VotingMathError> {
    let voters = vote_matrix.get_voters();
    let decisions = vote_matrix.get_decisions();
    let (num_voters, num_decisions) = vote_matrix.dimensions();

    let mut dense_matrix =
        DMatrix::from_element(num_voters, num_decisions, f64::NAN);

    for (i, voter_id) in voters.iter().enumerate() {
        for (j, decision_id) in decisions.iter().enumerate() {
            if let Some(vote) = vote_matrix.get_vote(*voter_id, *decision_id) {
                dense_matrix[(i, j)] = vote;
            }
        }
    }

    let mut rep_vector = DVector::zeros(num_voters);
    for (i, voter_id) in voters.iter().enumerate() {
        rep_vector[i] = reputation_vector.get_reputation(*voter_id);
    }

    let rep_sum = rep_vector.sum();
    if rep_sum > 0.0 {
        rep_vector /= rep_sum;
    } else {
        rep_vector = DVector::from_element(num_voters, 1.0 / num_voters as f64);
    }

    let result = factory(
        &dense_matrix,
        Some(&rep_vector),
        Some(CONSENSUS_CATCH_TOLERANCE),
    )?;

    let mut outcomes = HashMap::new();
    for (j, decision_id) in decisions.iter().enumerate() {
        let outcome = if result.has_votes[j] {
            Some(round_outcome(result.decisions[j]))
        } else {
            tracing::warn!(
                "Decision {} has no votes cast - marking as unanimous abstention",
                hex::encode(decision_id.as_bytes())
            );
            None
        };
        outcomes.insert(*decision_id, outcome);
    }

    let original_total: f64 = voters
        .iter()
        .map(|v| reputation_vector.get_reputation(*v))
        .sum();
    let consensus_total: f64 = result.agents.smooth_rep.sum();
    let scale_factor = if consensus_total > 0.0 && original_total > 0.0 {
        original_total / consensus_total
    } else {
        num_voters as f64 * BITCOIN_HIVEMIND_NEUTRAL_VALUE
    };

    let mut updated_reputations = HashMap::new();
    for (i, voter_id) in voters.iter().enumerate() {
        let scaled_reputation =
            round_reputation(result.agents.smooth_rep[i] * scale_factor);
        updated_reputations.insert(*voter_id, scaled_reputation);

        let old_rep = result.agents.old_rep[i];
        let delta = scaled_reputation - old_rep;
        if delta.abs() > 1e-6 {
            tracing::debug!(
                "Reputation change for voter {}: {:.6} -> {:.6} (delta: {:+.6})",
                i,
                old_rep,
                scaled_reputation,
                delta
            );
        }
    }

    let first_loading: Vec<f64> =
        result.first_loading.iter().cloned().collect();

    let mut outliers = Vec::new();
    for (i, voter_id) in voters.iter().enumerate() {
        let old_rep = result.agents.old_rep[i];
        let new_rep = result.agents.smooth_rep[i];
        if new_rep < old_rep * 0.5 {
            outliers.push(*voter_id);
        }
    }

    Ok(DetailedConsensusResult {
        outcomes,
        first_loading,
        certainty: result.certainty,
        updated_reputations,
        outliers,
    })
}

#[cfg(test)]
#[allow(clippy::print_stdout, clippy::uninlined_format_args)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_get_weight() {
        let vec = DVector::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let result = get_weight(&vec);
        assert_eq!(result, DVector::from_vec(vec![0.25, 0.25, 0.25, 0.25]));

        let vec2 = DVector::from_vec(vec![4.0, 5.0, 6.0, 7.0]);
        let result2 = get_weight(&vec2);
        assert!((result2[0] - 0.18181818).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_median() {
        let values = DVector::from_vec(vec![3.0, 4.0, 5.0]);
        let weights = DVector::from_vec(vec![0.2, 0.2, 0.6]);
        assert_eq!(weighted_median(&values, &weights).unwrap(), 5.0);

        let weights2 = DVector::from_vec(vec![0.2, 0.2, 0.4]);
        assert_eq!(weighted_median(&values, &weights2).unwrap(), 4.5);
    }

    #[test]
    fn test_weighted_median_empty() {
        let values = DVector::from_vec(vec![]);
        let weights = DVector::from_vec(vec![]);
        assert!(weighted_median(&values, &weights).is_err());
    }

    #[test]
    fn test_weighted_median_dimension_mismatch() {
        let values = DVector::from_vec(vec![1.0, 2.0]);
        let weights = DVector::from_vec(vec![0.5]);
        assert!(weighted_median(&values, &weights).is_err());
    }

    #[test]
    fn test_catch_tl() {
        assert_eq!(catch_tl(0.2, CONSENSUS_CATCH_TOLERANCE), 0.0);
        assert_eq!(
            catch_tl(BITCOIN_HIVEMIND_NEUTRAL_VALUE, CONSENSUS_CATCH_TOLERANCE),
            BITCOIN_HIVEMIND_NEUTRAL_VALUE
        );
        assert_eq!(catch_tl(0.8, CONSENSUS_CATCH_TOLERANCE), 1.0);
        assert_eq!(catch_tl(0.14, 0.7), 0.0);
        assert_eq!(catch_tl(0.16, 0.7), BITCOIN_HIVEMIND_NEUTRAL_VALUE);
    }

    #[test]
    fn test_weighted_prin_comp() {
        let m1 = DMatrix::from_row_slice(
            3,
            3,
            &[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        );

        let (score, loading) = weighted_prin_comp(&m1, None).unwrap();

        // Check dimensions
        assert_eq!(score.len(), 3);
        assert_eq!(loading.len(), 3);

        // Results should be normalized
        assert!(loading.norm() - 1.0 < 1e-6);
    }
}
