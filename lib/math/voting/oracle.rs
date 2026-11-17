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
use std::collections::{BTreeMap, HashMap};

fn normalize_weights(v: &DVector<f64>, zero_nan: bool) -> DVector<f64> {
    let mut out = v.clone();
    out.apply(|x| {
        if zero_nan && x.is_nan() {
            *x = 0.0;
        } else {
            *x = x.abs();
        }
    });
    let mut sum = out.sum();
    if sum == 0.0 {
        out.fill(1.0);
        sum = out.sum();
    }
    out /= sum;
    out
}

fn catch_tl(x: f64, tolerance: f64) -> f64 {
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
fn compute_weighted_column_means(
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

fn weighted_median(
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
    if total_weight <= 0.0 {
        return Err(VotingMathError::InvalidReputation {
            reason: "zero total weight in weighted_median".to_string(),
        });
    }
    let mut cumulative_weight = 0.0;
    let half_weight = total_weight / 2.0;

    for (i, &(value, weight)) in sorted.iter().enumerate() {
        cumulative_weight += weight;
        if cumulative_weight > half_weight - 1e-10 {
            // Use epsilon comparison instead of exact equality for f64
            // Exact equality (==) on f64 is unreliable due to floating-point precision
            if (cumulative_weight - half_weight).abs() < 1e-10
                && i < sorted.len() - 1
            {
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
    let mut distinct: Vec<f64> = component.iter().copied().collect();
    distinct.sort_by(|a, b| safe_cmp_f64(*a, *b));
    distinct.dedup_by(|a, b| (*a - *b).abs() <= SVD_NUMERICAL_TOLERANCE);
    if distinct.len() <= 2 {
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

fn weighted_prin_comp(
    x: &DMatrix<f64>,
    weights: Option<&DVector<f64>>,
) -> Result<(DVector<f64>, DVector<f64>), VotingMathError> {
    let n_rows = x.nrows();
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
        None => normalize_weights(&DVector::from_element(n_rows, 1.0), true),
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
    // A fixed-order loop: nalgebra sends large products to matrixmultiply,
    // whose CPU-selected kernels round differently.
    let n_cols = centered.ncols();
    let wcvm = DMatrix::from_fn(n_cols, n_cols, |a, b| {
        (0..n_rows)
            .map(|i| centered[(i, a)] * weighted_centered[(i, b)])
            .sum::<f64>()
            / total_weight
    });

    let svd = SVD::new(wcvm.clone(), true, false);

    let mut first_loading = svd
        .u
        .as_ref()
        .ok_or_else(|| VotingMathError::NumericalError {
            reason: "SVD failed to compute U matrix".to_string(),
        })?
        .column(0)
        .clone_owned();

    let canonical_sign = first_loading
        .iter()
        .copied()
        .find(|x| *x != 0.0)
        .map(f64::signum)
        .unwrap_or(1.0);
    if canonical_sign < 0.0 {
        first_loading = -first_loading;
    }

    let first_score = centered * &first_loading;

    Ok((first_score, first_loading))
}

struct RewardWeightsResult {
    pub first_loading: DVector<f64>,
    pub old_rep: DVector<f64>,
    pub smooth_rep: DVector<f64>,
}

/// The outcome each voter is measured against, per column: the pinned
/// outcome when the column has one, else the caught weighted mean
fn reference_outcomes(
    raw_outcomes: &DVector<f64>,
    catch_p: f64,
    pinned: &[Option<f64>],
) -> DVector<f64> {
    DVector::from_iterator(
        raw_outcomes.len(),
        raw_outcomes.iter().enumerate().map(|(j, &raw)| {
            pinned
                .get(j)
                .copied()
                .flatten()
                .unwrap_or_else(|| catch_tl(raw, catch_p))
        }),
    )
}

fn normalize_by_rep(v: &DVector<f64>, rep: &DVector<f64>) -> DVector<f64> {
    let total = v.dot(rep);
    if total > 0.0 {
        return v / total;
    }
    DVector::from_element(v.len(), 1.0 / rep.sum())
}

fn get_reward_weights(
    vote_matrix: &DMatrix<f64>,
    rep: Option<&DVector<f64>>,
    alpha: f64,
    voter_participation: &DVector<f64>,
    participation_total: f64,
    pinned: &[Option<f64>],
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
        let old_rep_outcomes = reference_outcomes(
            &raw_outcomes,
            CONSENSUS_CATCH_TOLERANCE,
            pinned,
        );

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
            dissent.map(|x| if x > f64::EPSILON { 1.0 / x } else { 0.0 });
        let mainstream = normalize_weights(&mainstream_preweight, true);

        // Compute per-voter non-compliance as weighted sum (matches R algorithm)
        // R: NonCompliance[i] = Σⱼ distance[i,j] × mainstream[j]
        let mut non_compliance_per_voter = DVector::zeros(num_voters);
        for i in 0..num_voters {
            let mut weighted_sum = 0.0;
            for j in 0..vote_matrix.ncols() {
                if j < mainstream.len() {
                    weighted_sum += distance[(i, j)] * mainstream[j];
                }
            }
            non_compliance_per_voter[i] = weighted_sum;
        }

        // R: Compliance = ReWeight(|NonCompliance - max(NonCompliance)|)
        let max_non_compliance = non_compliance_per_voter.max();
        let compliance_raw =
            non_compliance_per_voter.map(|x| (x - max_non_compliance).abs());
        let compliance_scores = normalize_by_rep(&compliance_raw, &old_rep);
        let compliance_spread =
            max_non_compliance - non_compliance_per_voter.min();

        let choice = |new_score: &DVector<f64>| {
            let difference =
                normalize_by_rep(new_score, &old_rep) - &compliance_scores;
            old_rep.dot(&difference.map(|x| x.powi(2)))
        };
        let ref_ind = choice(&new_score_1) - choice(&new_score_2);

        let rep_mean = old_rep.mean();
        if rep_mean > 0.0 && compliance_spread > SVD_NUMERICAL_TOLERANCE {
            let scaled_rep = old_rep.map(|x| x / rep_mean);

            let rep1 = normalize_weights(
                &new_score_1.component_mul(&scaled_rep),
                false,
            );
            let rep2 = normalize_weights(
                &new_score_2.component_mul(&scaled_rep),
                false,
            );

            new_rep = if ref_ind < 0.0 { rep1 } else { rep2 };
        }
    }

    let old_rep_normalized = normalize_weights(&old_rep, true);

    let weighted_participation = voter_participation.component_mul(&old_rep);
    let participation_sum = weighted_participation.sum();
    let p_rel = if participation_sum > f64::EPSILON {
        weighted_participation / participation_sum
    } else {
        old_rep_normalized.clone()
    };
    let r_blend =
        &new_rep * participation_total + &p_rel * (1.0 - participation_total);

    let smooth_rep = &r_blend * alpha + &old_rep_normalized * (1.0 - alpha);

    Ok(RewardWeightsResult {
        first_loading,
        old_rep,
        smooth_rep,
    })
}

struct FillNaResult {
    pub filled: DMatrix<f64>,
    pub has_votes: Vec<bool>,
    pub voter_participation: DVector<f64>,
    pub participation_total: f64,
}

fn fill_na(
    m_na: &DMatrix<f64>,
    rep: Option<&DVector<f64>>,
    catch_p: f64,
    pinned: &[Option<f64>],
) -> FillNaResult {
    let mut m_new = m_na.clone();
    let num_voters = m_na.nrows();
    let num_decisions = m_na.ncols();

    let rep = match rep {
        Some(r) => r.clone(),
        None => DVector::from_element(num_voters, 1.0 / num_voters as f64),
    };

    let mut voter_participation = DVector::zeros(num_voters);
    for i in 0..num_voters {
        let cast = (0..num_decisions)
            .filter(|&j| !m_na[(i, j)].is_nan())
            .count();
        voter_participation[i] = if num_decisions == 0 {
            0.0
        } else {
            cast as f64 / num_decisions as f64
        };
    }
    let rep_sum = rep.sum();
    let participation_total = if rep_sum > 0.0 {
        voter_participation.dot(&rep) / rep_sum
    } else {
        0.0
    };

    let has_votes;
    let has_nan = m_na.iter().any(|&x| x.is_nan());

    if has_nan {
        let (raw_outcomes, votes_present) =
            compute_weighted_column_means(m_na, &rep);
        has_votes = votes_present;
        let decision_outcomes =
            reference_outcomes(&raw_outcomes, catch_p, pinned);

        for i in 0..m_new.nrows() {
            for j in 0..m_new.ncols() {
                if m_new[(i, j)].is_nan() {
                    m_new[(i, j)] = decision_outcomes[j];
                }
            }
        }
    } else {
        has_votes = vec![true; num_decisions];
    }

    FillNaResult {
        filled: m_new,
        has_votes,
        voter_participation,
        participation_total,
    }
}

struct FactoryResult {
    pub agents: RewardWeightsResult,
    pub decisions: DVector<f64>,
    pub has_votes: Vec<bool>,
    pub certainty: f64,
    pub first_loading: DVector<f64>,
}

/// Compute consensus outcomes for each decision.
///
/// Per Bitcoin Hivemind whitepaper:
/// - Binary decisions: weighted mean + catch_tl to discretize to {0, 0.5, 1}
/// - Scaled decisions: weighted median, continuous value in [0, 1]
///
/// `scaled_indices` indicates which column indices correspond to scaled decisions.
fn factory(
    m0: &DMatrix<f64>,
    rep: Option<&DVector<f64>>,
    catch_p: Option<f64>,
    scaled_indices: Option<&[bool]>,
    pinned: &[Option<f64>],
) -> Result<FactoryResult, VotingMathError> {
    let catch_p = catch_p.unwrap_or(CONSENSUS_CATCH_TOLERANCE);
    let num_decisions = m0.ncols();

    if num_decisions == 0 || m0.nrows() == 0 {
        return Err(VotingMathError::EmptyMatrix);
    }

    // Default: all binary (no scaled)
    let scaled_flags: Vec<bool> = scaled_indices
        .map(|s| s.to_vec())
        .unwrap_or_else(|| vec![false; num_decisions]);

    let fill_result = fill_na(m0, rep, catch_p, pinned);
    let filled = fill_result.filled;
    let has_votes = fill_result.has_votes;
    let voter_participation = fill_result.voter_participation;
    let participation_total = fill_result.participation_total;

    use super::constants::REPUTATION_SMOOTHING_ALPHA;
    let player_info = get_reward_weights(
        &filled,
        rep,
        REPUTATION_SMOOTHING_ALPHA,
        &voter_participation,
        participation_total,
        pinned,
    )?;

    // Calculate outcomes differently for binary vs scaled decisions
    let mut decision_outcomes_final = DVector::zeros(num_decisions);

    for j in 0..num_decisions {
        let is_scaled = scaled_flags.get(j).copied().unwrap_or(false);

        if is_scaled {
            // Scaled decisions: use weighted median, keep continuous value
            let col_values: DVector<f64> = DVector::from_iterator(
                filled.nrows(),
                filled.column(j).iter().copied(),
            );

            // Filter out NaN values and their corresponding weights
            let mut valid_values = Vec::new();
            let mut valid_weights = Vec::new();
            for i in 0..filled.nrows() {
                let val = col_values[i];
                if !val.is_nan() {
                    valid_values.push(val);
                    valid_weights.push(player_info.smooth_rep[i]);
                }
            }

            if valid_values.is_empty() {
                decision_outcomes_final[j] = BITCOIN_HIVEMIND_NEUTRAL_VALUE;
            } else {
                let values_vec = DVector::from_vec(valid_values);
                let weights_vec = DVector::from_vec(valid_weights);
                decision_outcomes_final[j] =
                    weighted_median(&values_vec, &weights_vec)
                        .unwrap_or(BITCOIN_HIVEMIND_NEUTRAL_VALUE);
            }
        } else {
            // Binary decisions: weighted mean + catch_tl
            let mut weighted_sum = 0.0;
            let mut total_weight = 0.0;
            for i in 0..filled.nrows() {
                let vote = filled[(i, j)];
                if !vote.is_nan() {
                    let weight = player_info.smooth_rep[i];
                    weighted_sum += vote * weight;
                    total_weight += weight;
                }
            }
            let raw_outcome = if total_weight > 0.0 {
                weighted_sum / total_weight
            } else {
                BITCOIN_HIVEMIND_NEUTRAL_VALUE
            };
            decision_outcomes_final[j] = catch_tl(raw_outcome, catch_p);
        }
    }

    let mut certainty = vec![0.0; num_decisions];
    for j in 0..num_decisions {
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

    let avg_certainty = if certainty.is_empty() {
        0.0
    } else {
        certainty.iter().sum::<f64>() / certainty.len() as f64
    };

    Ok(FactoryResult {
        first_loading: player_info.first_loading.clone(),
        agents: player_info,
        decisions: decision_outcomes_final,
        has_votes,
        certainty: avg_certainty,
    })
}

/// The result of a reputation-weighted plurality vote on a categorical
/// decision
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Plurality {
    /// One option has the most reputation behind it. `option_count` is
    /// "Inconclusive".
    Winner(u16),
    /// The top options hold equal reputation
    Tie,
}

/// The plurality over the options in `0..=option_count`. `None` when no
/// vote picks one of them.
pub fn weighted_plurality(
    votes: impl IntoIterator<Item = (u16, f64)>,
    option_count: u16,
) -> Option<Plurality> {
    let mut weights: BTreeMap<u16, f64> = BTreeMap::new();
    for (option, weight) in votes {
        if option <= option_count {
            *weights.entry(option).or_insert(0.0) += weight;
        }
    }
    let max_weight = weights.values().copied().reduce(f64::max)?;
    let mut winners = weights
        .iter()
        .filter(|(_, weight)| **weight == max_weight)
        .map(|(option, _)| *option);
    match (winners.next(), winners.next()) {
        (Some(winner), None) => Some(Plurality::Winner(winner)),
        _ => Some(Plurality::Tie),
    }
}

/// Outcome of a decision that SVD leaves out, from the `(value, weight)`
/// votes cast on it, with the winning option of a categorical decision
/// (`None` on a tie). `None` when no vote carries weight.
pub fn resolve_unguaranteed(
    votes: &[(f64, f64)],
    scaled: bool,
    option_count: Option<u16>,
) -> Result<Option<(f64, Option<u16>)>, VotingMathError> {
    let total_weight: f64 = votes.iter().map(|(_, weight)| weight).sum();
    if total_weight <= 0.0 {
        return Ok(None);
    }
    if let Some(option_count) = option_count {
        let plurality = weighted_plurality(
            votes
                .iter()
                .map(|&(value, weight)| (value.round() as u16, weight)),
            option_count,
        );
        return Ok(plurality.map(|plurality| match plurality {
            Plurality::Winner(winner) if winner < option_count => {
                (winner as f64, Some(winner))
            }
            Plurality::Winner(winner) => {
                (BITCOIN_HIVEMIND_NEUTRAL_VALUE, Some(winner))
            }
            Plurality::Tie => (BITCOIN_HIVEMIND_NEUTRAL_VALUE, None),
        }));
    }
    let outcome = if scaled {
        weighted_median(
            &DVector::from_iterator(votes.len(), votes.iter().map(|v| v.0)),
            &DVector::from_iterator(votes.len(), votes.iter().map(|v| v.1)),
        )?
    } else {
        let weighted_sum: f64 =
            votes.iter().map(|(value, weight)| value * weight).sum();
        catch_tl(weighted_sum / total_weight, CONSENSUS_CATCH_TOLERANCE)
    };
    Ok(Some((round_outcome(outcome), None)))
}

/// Resolve categorical decisions by reputation-weighted plurality vote.
///
/// Each categorical decision implicitly exposes one extra option at index
/// `option_count` representing "Inconclusive". Voters may pick any index in
/// `0..=option_count`. The winner is the option with the most reputation
/// behind it, Inconclusive included. When the top options tie, no option
/// wins and the decision resolves inconclusive.
///
/// Matrix rewrite, for a winner (Inconclusive included): voters at the
/// winner -> 1.0, all other (non-NaN) voters -> 0.0. A tied column is left as
/// it is, since consensus leaves it out.
pub fn resolve_categorical_plurality(
    dense_matrix: &mut DMatrix<f64>,
    categorical_columns: &HashMap<usize, u16>,
    rep: &DVector<f64>,
) -> HashMap<usize, super::CategoricalResult> {
    let mut results = HashMap::new();

    for (&col_idx, &option_count) in categorical_columns {
        let votes = (0..dense_matrix.nrows()).filter_map(|row| {
            let val = dense_matrix[(row, col_idx)];
            (!val.is_nan()).then(|| (val.round() as u16, rep[row]))
        });
        let effective_winner = match weighted_plurality(votes, option_count) {
            Some(Plurality::Winner(winner)) => winner,
            plurality => {
                results.insert(
                    col_idx,
                    super::CategoricalResult {
                        winning_option: None,
                        tied: plurality == Some(Plurality::Tie),
                    },
                );
                continue;
            }
        };

        for row in 0..dense_matrix.nrows() {
            let val = dense_matrix[(row, col_idx)];
            if val.is_nan() {
                continue;
            }
            let voter_idx = val.round() as u16;
            dense_matrix[(row, col_idx)] = if voter_idx == effective_winner {
                1.0
            } else {
                0.0
            };
        }

        results.insert(
            col_idx,
            super::CategoricalResult {
                winning_option: Some(effective_winner),
                tied: false,
            },
        );
    }

    results
}

pub fn run_consensus(
    vote_matrix: &SparseVoteMatrix,
    reputation_vector: &VotingWeightVector,
    scaled_decisions: &std::collections::HashSet<
        crate::state::decisions::DecisionId,
    >,
    categorical_decisions: &HashMap<crate::state::decisions::DecisionId, u16>,
) -> Result<DetailedConsensusResult, VotingMathError> {
    let voters = vote_matrix.get_voters();
    let decisions = vote_matrix.get_decisions();
    let (num_voters, num_decisions) = vote_matrix.dimensions();

    let scaled_indices: Vec<bool> = decisions
        .iter()
        .map(|decision_id| scaled_decisions.contains(decision_id))
        .collect();

    let categorical_columns: HashMap<usize, u16> = decisions
        .iter()
        .enumerate()
        .filter_map(|(j, decision_id)| {
            categorical_decisions
                .get(decision_id)
                .map(|&count| (j, count))
        })
        .collect();

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

    let categorical_results = resolve_categorical_plurality(
        &mut dense_matrix,
        &categorical_columns,
        &rep_vector,
    );

    let svd_columns: Vec<usize> = (0..num_decisions)
        .filter(|j| !categorical_results.get(j).is_some_and(|r| r.tied))
        .collect();
    let svd_matrix = dense_matrix.select_columns(&svd_columns);
    let svd_scaled: Vec<bool> =
        svd_columns.iter().map(|&j| scaled_indices[j]).collect();
    let svd_pinned: Vec<Option<f64>> = svd_columns
        .iter()
        .map(|j| {
            categorical_results
                .get(j)
                .and_then(|r| r.winning_option)
                .map(|_| 1.0)
        })
        .collect();
    let svd_index: HashMap<usize, usize> = svd_columns
        .iter()
        .enumerate()
        .map(|(svd_j, &j)| (j, svd_j))
        .collect();

    let result = if svd_columns.is_empty() {
        None
    } else {
        Some(factory(
            &svd_matrix,
            Some(&rep_vector),
            Some(CONSENSUS_CATCH_TOLERANCE),
            Some(&svd_scaled),
            &svd_pinned,
        )?)
    };

    let mut categorical_winners = HashMap::new();
    let mut outcomes = HashMap::new();
    for (j, decision_id) in decisions.iter().enumerate() {
        if let Some(cat_result) = categorical_results.get(&j) {
            let option_count =
                categorical_columns.get(&j).copied().unwrap_or(0);
            let outcome_val = match cat_result.winning_option {
                Some(w) if w < option_count => w as f64,
                _ => BITCOIN_HIVEMIND_NEUTRAL_VALUE,
            };
            outcomes.insert(*decision_id, Some(outcome_val));
            categorical_winners.insert(*decision_id, cat_result.winning_option);
        } else if let (Some(result), Some(&svd_j)) =
            (&result, svd_index.get(&j))
        {
            let outcome = if result.has_votes[svd_j] {
                Some(round_outcome(result.decisions[svd_j]))
            } else {
                tracing::warn!(
                    "Decision {} has no votes cast \
                     - marking as unanimous abstention",
                    const_hex::encode(decision_id.as_bytes())
                );
                None
            };
            outcomes.insert(*decision_id, outcome);
        }
    }

    let (old_rep, smooth_rep, first_loading, certainty) = match result {
        Some(result) => (
            result.agents.old_rep,
            result.agents.smooth_rep,
            result.first_loading.iter().copied().collect(),
            result.certainty,
        ),
        None => (rep_vector.clone(), rep_vector, Vec::new(), 0.0),
    };

    let original_total: f64 = voters
        .iter()
        .map(|v| reputation_vector.get_reputation(*v))
        .sum();
    let consensus_total: f64 = smooth_rep.sum();
    let scale_factor = if consensus_total > 0.0 && original_total > 0.0 {
        original_total / consensus_total
    } else {
        num_voters as f64 * BITCOIN_HIVEMIND_NEUTRAL_VALUE
    };

    let mut updated_reputations = BTreeMap::new();
    let mut max_voter: Option<(crate::types::Address, f64)> = None;

    for (i, voter_id) in voters.iter().enumerate() {
        let scaled_reputation = round_reputation(smooth_rep[i] * scale_factor);
        updated_reputations.insert(*voter_id, scaled_reputation);

        match max_voter {
            Some((_, best)) if scaled_reputation > best => {
                max_voter = Some((*voter_id, scaled_reputation));
            }
            None => {
                max_voter = Some((*voter_id, scaled_reputation));
            }
            _ => {}
        }
    }

    let rounded_total: f64 = updated_reputations.values().sum();
    let residual = original_total - rounded_total;
    if residual.abs() > 0.0
        && let Some((adjust_addr, _)) = max_voter
        && let Some(rep) = updated_reputations.get_mut(&adjust_addr)
    {
        *rep = round_reputation(*rep + residual);
    }

    if original_total > 0.0 {
        let final_total: f64 = updated_reputations.values().sum();
        debug_assert!(
            (original_total - final_total).abs() < 1e-6,
            "reputation conservation violated: before={original_total}, \
             after={final_total}"
        );
    }

    let mut outliers = Vec::new();
    for (i, voter_id) in voters.iter().enumerate() {
        if smooth_rep[i] < old_rep[i] * 0.5 {
            outliers.push(*voter_id);
        }
    }

    Ok(DetailedConsensusResult {
        outcomes,
        first_loading,
        certainty,
        updated_reputations,
        outliers,
        categorical_winners,
    })
}

#[cfg(test)]
#[allow(clippy::print_stdout, clippy::uninlined_format_args)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_normalize_weights_no_nan() {
        let vec = DVector::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let result = normalize_weights(&vec, false);
        assert_eq!(result, DVector::from_vec(vec![0.25, 0.25, 0.25, 0.25]));

        let vec2 = DVector::from_vec(vec![4.0, 5.0, 6.0, 7.0]);
        let result2 = normalize_weights(&vec2, false);
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
    fn test_weighted_median_mismatch() {
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

    /// With zero_nan, all-zero weights normalize to uniform.
    #[test]
    fn test_normalize_zero_nan_zeros() {
        let vec = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        let result = normalize_weights(&vec, true);
        assert_eq!(result, DVector::from_vec(vec![0.25, 0.25, 0.25, 0.25]));
    }

    /// With zero_nan, all-NaN weights normalize to uniform.
    #[test]
    fn test_normalize_zero_nan_nans() {
        let vec = DVector::from_vec(vec![f64::NAN, f64::NAN, f64::NAN]);
        let result = normalize_weights(&vec, true);
        let expected = 1.0 / 3.0;
        for i in 0..3 {
            assert!(
                (result[i] - expected).abs() < 1e-10,
                "index {i}: expected {expected}, got {}",
                result[i]
            );
        }
    }

    /// With zero_nan, finite weights normalize proportionally.
    #[test]
    fn test_normalize_zero_nan_finite() {
        let vec = DVector::from_vec(vec![2.0, 3.0, 5.0]);
        let result = normalize_weights(&vec, true);
        assert!((result[0] - 0.2).abs() < 1e-10);
        assert!((result[1] - 0.3).abs() < 1e-10);
        assert!((result[2] - 0.5).abs() < 1e-10);
    }

    /// Without zero_nan, all-zero weights normalize to uniform.
    #[test]
    fn test_normalize_no_nan_zeros() {
        let vec = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let result = normalize_weights(&vec, false);
        let expected = 1.0 / 3.0;
        for i in 0..3 {
            assert!(
                (result[i] - expected).abs() < 1e-10,
                "index {i}: expected {expected}, got {}",
                result[i]
            );
        }
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

    #[test]
    fn test_weighted_median_zero_weights() {
        let values = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let weights = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        assert!(weighted_median(&values, &weights).is_err());
    }

    #[test]
    fn test_factory_empty_decisions() {
        let m = DMatrix::from_row_slice(3, 0, &[]);
        let rep = DVector::from_vec(vec![0.33, 0.33, 0.34]);
        let result = factory(&m, Some(&rep), None, None, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_factory_single_voter() {
        let m = DMatrix::from_row_slice(1, 1, &[1.0]);
        let rep = DVector::from_vec(vec![1.0]);
        let result = factory(&m, Some(&rep), None, None, &[]);
        assert!(result.is_ok());
    }

    /// Cumulative weight one ULP short of half still takes the midpoint.
    #[test]
    fn median_ulp_short() {
        let values = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let weights = DVector::from_vec(vec![0.1, 0.35, 0.05, 0.5]);
        assert_eq!(weighted_median(&values, &weights).unwrap(), 3.5);
    }

    /// Splitting the lower half's weight into pieces keeps the midpoint.
    #[test]
    fn median_split_pieces() {
        let whole = weighted_median(
            &DVector::from_vec(vec![0.2, 0.8]),
            &DVector::from_vec(vec![0.5, 0.5]),
        )
        .unwrap();
        let pieces = weighted_median(
            &DVector::from_vec(vec![0.2, 0.2, 0.2, 0.2, 0.2, 0.8]),
            &DVector::from_vec(vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.5]),
        )
        .unwrap();
        assert_eq!(whole, 0.5);
        assert_eq!(pieces, 0.5);
    }

    /// Five identical scores count as one distinct value and are returned
    /// unchanged.
    #[test]
    fn reflect_identical_scores() {
        let component = DVector::from_element(5, 0.3);
        let rep = DVector::from_element(5, 0.2);
        assert_eq!(score_reflect(&component, &rep).unwrap(), component);
    }

    /// Five addresses holding two distinct scores skip reflection.
    #[test]
    fn reflect_two_values() {
        let component = DVector::from_vec(vec![0.0, 0.0, 1.0, 1.0, 1.0]);
        let rep = DVector::from_element(5, 0.2);
        assert_eq!(score_reflect(&component, &rep).unwrap(), component);
    }

    /// Three distinct scores reflect: the above-median score is pulled
    /// halfway toward the median.
    #[test]
    fn reflect_three_values() {
        let component = DVector::from_vec(vec![0.0, 0.5, 1.0]);
        let rep = DVector::from_element(3, 1.0 / 3.0);
        assert_eq!(
            score_reflect(&component, &rep).unwrap(),
            DVector::from_vec(vec![0.0, 0.5, 0.75])
        );
    }

    /// Scores within SVD_NUMERICAL_TOLERANCE of each other count once.
    #[test]
    fn reflect_near_duplicates() {
        let component = DVector::from_vec(vec![0.0, 1.0, 1.0 + 1e-12, 1.0]);
        let rep = DVector::from_element(4, 0.25);
        assert_eq!(score_reflect(&component, &rep).unwrap(), component);
    }

    /// An all-zero vector falls back to uniform per unit of reputation.
    #[test]
    fn normalize_by_rep_zeros() {
        let rep = DVector::from_vec(vec![0.2, 0.3, 0.5]);
        let result = normalize_by_rep(&DVector::zeros(3), &rep);
        assert!(result.iter().all(|x| x.is_finite()));
        assert!((rep.dot(&result) - 1.0).abs() < 1e-12);
    }

    /// One voter split into three clones gets the same summed smooth
    /// reputation as when unsplit.
    #[test]
    fn reward_weights_split_invariant() {
        let rows = [
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.5],
            [1.0, 0.0, 0.0, 1.0],
        ];
        let whole = DMatrix::from_fn(3, 4, |i, j| rows[i][j]);
        let split = DMatrix::from_fn(5, 4, |i, j| rows[[0, 1, 1, 1, 2][i]][j]);
        let whole_rep = DVector::from_vec(vec![0.5, 0.3, 0.2]);
        let split_rep = DVector::from_vec(vec![0.5, 0.1, 0.1, 0.1, 0.2]);
        let pinned = [None; 4];

        let whole_smooth = get_reward_weights(
            &whole,
            Some(&whole_rep),
            0.3,
            &DVector::from_element(3, 1.0),
            1.0,
            &pinned,
        )
        .unwrap()
        .smooth_rep;
        let split_smooth = get_reward_weights(
            &split,
            Some(&split_rep),
            0.3,
            &DVector::from_element(5, 1.0),
            1.0,
            &pinned,
        )
        .unwrap()
        .smooth_rep;

        let clone_sum = split_smooth[1] + split_smooth[2] + split_smooth[3];
        assert!((whole_smooth[0] - split_smooth[0]).abs() < 1e-9);
        assert!((whole_smooth[1] - clone_sum).abs() < 1e-9);
        assert!((whole_smooth[2] - split_smooth[4]).abs() < 1e-9);
    }
}
