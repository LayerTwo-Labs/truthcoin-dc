use super::constants::{
    BITCOIN_HIVEMIND_NEUTRAL_VALUE, CONSENSUS_CATCH_TOLERANCE,
    LOADING_NOISE_FLOOR, REPUTATION_SMOOTHING_ALPHA, REVEAL_TIER_MULTIPLIERS,
    REVEAL_TIER_SHARE, SCALED_MATCH_TOLERANCE, SVD_NUMERICAL_TOLERANCE,
    round_outcome, round_reputation,
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

/// Compute weighted mean for each column (decision) in a vote matrix,
/// ignoring NaN cells; a column with no votes is neutral.
fn compute_weighted_column_means(
    matrix: &DMatrix<f64>,
    weights: &DVector<f64>,
) -> DVector<f64> {
    let num_decisions = matrix.ncols();
    let mut outcomes = DVector::zeros(num_decisions);

    for j in 0..num_decisions {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for i in 0..matrix.nrows() {
            let vote = matrix[(i, j)];
            if !vote.is_nan() {
                let weight = weights[i];
                weighted_sum += vote * weight;
                total_weight += weight;
            }
        }

        outcomes[j] = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            BITCOIN_HIVEMIND_NEUTRAL_VALUE
        };
    }

    outcomes
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

    let anchor = first_loading
        .iter()
        .copied()
        .reduce(|best, x| if x.abs() > best.abs() { x } else { best })
        .unwrap_or(1.0);
    if anchor < 0.0 {
        first_loading = -first_loading;
    }

    let first_score = centered * &first_loading;

    Ok((first_score, first_loading))
}

struct RewardWeightsResult {
    pub first_loading: DVector<f64>,
    pub new_rep: DVector<f64>,
    pub smooth_rep: DVector<f64>,
    pub deadlocked: Vec<bool>,
}

fn column_median(
    matrix: &DMatrix<f64>,
    weights: &DVector<f64>,
    column: usize,
) -> Option<f64> {
    let (values, column_weights): (Vec<f64>, Vec<f64>) = (0..matrix.nrows())
        .filter(|&i| !matrix[(i, column)].is_nan())
        .map(|i| (matrix[(i, column)], weights[i]))
        .unzip();
    weighted_median(
        &DVector::from_vec(values),
        &DVector::from_vec(column_weights),
    )
    .ok()
}

/// The outcome each voter is measured against, per column: the pinned
/// outcome when the column has one, the weighted median of the votes cast
/// on a scaled column, else the caught weighted mean
fn reference_outcomes(
    matrix: &DMatrix<f64>,
    weights: &DVector<f64>,
    catch_p: f64,
    pinned: &[Option<f64>],
    scaled: &[bool],
) -> DVector<f64> {
    let raw_outcomes = compute_weighted_column_means(matrix, weights);
    DVector::from_iterator(
        raw_outcomes.len(),
        raw_outcomes.iter().enumerate().map(|(j, &raw)| {
            let caught = catch_tl(raw, catch_p);
            pinned.get(j).copied().flatten().unwrap_or_else(|| {
                if scaled[j] {
                    column_median(matrix, weights, j).unwrap_or(caught)
                } else {
                    caught
                }
            })
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
    scaled: &[bool],
) -> Result<RewardWeightsResult, VotingMathError> {
    let num_voters = vote_matrix.nrows();

    let old_rep = match rep {
        Some(r) => r.clone(),
        None => DVector::from_element(num_voters, 1.0 / num_voters as f64),
    };

    let old_rep_outcomes = reference_outcomes(
        vote_matrix,
        &old_rep,
        CONSENSUS_CATCH_TOLERANCE,
        pinned,
        scaled,
    );
    let deadlocked: Vec<bool> = (0..vote_matrix.ncols())
        .map(|j| {
            !scaled[j]
                && old_rep_outcomes[j] == BITCOIN_HIVEMIND_NEUTRAL_VALUE
                && [0.0, 1.0].iter().all(|side| {
                    vote_matrix.column(j).iter().any(|vote| vote == side)
                })
        })
        .collect();
    let mut neutral = vote_matrix.clone();
    for j in (0..neutral.ncols()).filter(|&j| deadlocked[j]) {
        neutral.column_mut(j).fill(BITCOIN_HIVEMIND_NEUTRAL_VALUE);
    }

    let (first_score, first_loading) =
        weighted_prin_comp(&neutral, Some(&old_rep))?;

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

        let distance = DMatrix::from_fn(
            vote_matrix.nrows(),
            vote_matrix.ncols(),
            |i, j| {
                if deadlocked[j] {
                    0.0
                } else {
                    (vote_matrix[(i, j)] - old_rep_outcomes[j]).abs()
                }
            },
        );

        let dissent = first_loading.abs();
        let noise = dissent.max() * LOADING_NOISE_FLOOR;
        let mainstream_preweight =
            dissent.map(|x| if x > noise { 1.0 / x } else { 0.0 });
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
        if rep_mean > 0.0
            && compliance_spread > SVD_NUMERICAL_TOLERANCE
            && ref_ind.abs() > SVD_NUMERICAL_TOLERANCE
        {
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
        new_rep,
        smooth_rep,
        deadlocked,
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
    scaled: &[bool],
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
        has_votes = (0..num_decisions)
            .map(|j| m_na.column(j).iter().any(|v| !v.is_nan()))
            .collect();
        let decision_outcomes =
            reference_outcomes(m_na, &rep, catch_p, pinned, scaled);

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

    let fill_result = fill_na(m0, rep, catch_p, pinned, &scaled_flags);
    let filled = fill_result.filled;
    let has_votes = fill_result.has_votes;
    let voter_participation = fill_result.voter_participation;
    let participation_total = fill_result.participation_total;

    let player_info = get_reward_weights(
        &filled,
        rep,
        REPUTATION_SMOOTHING_ALPHA,
        &voter_participation,
        participation_total,
        pinned,
        &scaled_flags,
    )?;

    let decision_outcomes_final = reference_outcomes(
        &filled,
        &player_info.smooth_rep,
        catch_p,
        &[],
        &scaled_flags,
    );

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

/// How a voter's sealed ballot fared, with the block it was revealed in
#[derive(Clone, Copy, Debug, PartialEq)]
enum Reveal {
    Silent,
    Correct(u32),
    Incorrect(u32),
}

fn vote_matches(vote: f64, outcome: f64, scaled: bool) -> bool {
    let tolerance = if scaled { SCALED_MATCH_TOLERANCE } else { 0.0 };
    (vote - outcome).abs() <= tolerance + SVD_NUMERICAL_TOLERANCE
}

/// The outcomes some revealer matched. A decision no revealer matched is
/// left out of the correctness check, so it cannot disqualify everyone.
fn matched_outcomes(
    matrix: &DMatrix<f64>,
    outcomes: &[Option<f64>],
    scaled: &[bool],
    revealed: &[bool],
) -> Vec<Option<f64>> {
    outcomes
        .iter()
        .enumerate()
        .map(|(j, outcome)| {
            outcome.filter(|&outcome| {
                (0..matrix.nrows()).any(|i| {
                    revealed[i]
                        && vote_matches(matrix[(i, j)], outcome, scaled[j])
                })
            })
        })
        .collect()
}

/// Whether the voter's row matches every column that has an outcome.
/// `outcomes[j]` is `None` for a column left out of the check. A scaled
/// vote matches within 1% of the decision's range.
fn ballot_correct(
    matrix: &DMatrix<f64>,
    voter: usize,
    outcomes: &[Option<f64>],
    scaled: &[bool],
) -> bool {
    outcomes.iter().any(Option::is_some)
        && outcomes.iter().enumerate().all(|(j, outcome)| {
            outcome.is_none_or(|outcome| {
                vote_matches(matrix[(voter, j)], outcome, scaled[j])
            })
        })
}

/// Once anyone has revealed, each revealer smooths toward its share of
/// the revealers' SVD reputation and each silent voter toward zero. When
/// the SVD scores every revealer zero, revealers keep their share of the
/// revealers' old reputation.
fn exclude_silent(
    smooth_rep: DVector<f64>,
    old_rep: &DVector<f64>,
    new_rep: &DVector<f64>,
    reveals: &[Reveal],
) -> DVector<f64> {
    let revealed_sum = |weights: &DVector<f64>| -> f64 {
        reveals
            .iter()
            .zip(weights.iter())
            .filter(|(reveal, _)| **reveal != Reveal::Silent)
            .map(|(_, weight)| weight)
            .sum()
    };
    let new_share = revealed_sum(new_rep);
    let (weights, revealed_share) = if new_share > SVD_NUMERICAL_TOLERANCE {
        (new_rep, new_share)
    } else {
        (old_rep, revealed_sum(old_rep))
    };
    if revealed_share <= 0.0 {
        return smooth_rep;
    }
    DVector::from_iterator(
        old_rep.len(),
        reveals.iter().enumerate().map(|(i, reveal)| {
            let share = match *reveal {
                Reveal::Silent => 0.0,
                Reveal::Correct(_) | Reveal::Incorrect(_) => {
                    weights[i] / revealed_share
                }
            };
            REPUTATION_SMOOTHING_ALPHA * share
                + (1.0 - REPUTATION_SMOOTHING_ALPHA) * old_rep[i]
        }),
    )
}

/// Each `(voter, height, space)` entry's share of the two reveal tiers,
/// filled earliest block first. Entries sharing a block split it pro rata.
fn tier_slices(mut ranked: Vec<(usize, u32, f64)>) -> Vec<(usize, [f64; 2])> {
    ranked.sort_by_key(|&(i, height, _)| (height, i));
    let mut capacity = [REVEAL_TIER_SHARE; 2];
    let mut slices = Vec::new();
    for block in ranked.chunk_by(|a, b| a.1 == b.1) {
        if capacity.iter().all(|cap| *cap <= SVD_NUMERICAL_TOLERANCE) {
            break;
        }
        let block_space: f64 = block.iter().map(|&(_, _, space)| space).sum();
        if block_space <= 0.0 {
            continue;
        }
        let mut left = block_space;
        let taken = capacity.each_mut().map(|cap| {
            let take = left.min(*cap);
            *cap -= take;
            left -= take;
            take
        });
        slices.extend(block.iter().map(|&(i, _, space)| {
            (i, taken.map(|take| take / block_space * space))
        }));
    }
    slices
}

/// Payout weights for correct revealers who gained, filling the tiers by
/// reputation at 1.5/1.25; a first block holding both tiers' worth of
/// reputation therefore shares the whole pool pro rata.
fn gainer_weights(gainers: Vec<(usize, u32, f64)>) -> Vec<(usize, f64)> {
    tier_slices(gainers)
        .into_iter()
        .map(|(i, slices)| {
            (
                i,
                REVEAL_TIER_MULTIPLIERS
                    .iter()
                    .zip(slices)
                    .map(|(m, s)| m * s)
                    .sum(),
            )
        })
        .collect()
}

/// Pool weights when no correct revealer gained: the correct revealers,
/// or every revealer when none was correct, by old reputation.
fn revealer_weights(
    reveals: &[Reveal],
    rep: &DVector<f64>,
) -> Vec<(usize, f64)> {
    let correct: Vec<(usize, f64)> = reveals
        .iter()
        .enumerate()
        .filter_map(|(i, reveal)| match *reveal {
            Reveal::Correct(_) => Some((i, rep[i])),
            Reveal::Silent | Reveal::Incorrect(_) => None,
        })
        .collect();
    if correct.iter().map(|(_, weight)| weight).sum::<f64>() > 0.0 {
        return correct;
    }
    reveals
        .iter()
        .enumerate()
        .filter(|(_, reveal)| **reveal != Reveal::Silent)
        .map(|(i, _)| (i, rep[i]))
        .collect()
}

/// Early incorrect revealers and silent voters smooth at a raised alpha.
/// The extra loss goes to the earliest correct revealers who gained. When
/// none gained, only silent voters pay and the revealers share it.
fn apply_reveal_timing(
    old_rep: &DVector<f64>,
    smooth_rep: DVector<f64>,
    reveals: &[Reveal],
) -> DVector<f64> {
    let change = |i: usize| smooth_rep[i] - old_rep[i];
    let total = |weights: &[(usize, f64)]| -> f64 {
        weights.iter().map(|(_, weight)| weight).sum()
    };
    let gainers = gainer_weights(
        reveals
            .iter()
            .enumerate()
            .filter_map(|(i, reveal)| match *reveal {
                Reveal::Correct(height)
                    if change(i) > SVD_NUMERICAL_TOLERANCE =>
                {
                    Some((i, height, old_rep[i]))
                }
                Reveal::Silent | Reveal::Correct(_) | Reveal::Incorrect(_) => {
                    None
                }
            })
            .collect(),
    );
    let tiered = total(&gainers) > 0.0;
    let weights = if tiered {
        gainers
    } else {
        revealer_weights(reveals, old_rep)
    };
    let total_weight = total(&weights);
    if total_weight <= 0.0 {
        return smooth_rep;
    }

    let [tier1, _] = REVEAL_TIER_MULTIPLIERS;
    let mut extra = DVector::from_iterator(
        old_rep.len(),
        reveals.iter().enumerate().map(|(i, reveal)| match *reveal {
            Reveal::Silent => (tier1 - 1.0) * -change(i),
            Reveal::Correct(_) | Reveal::Incorrect(_) => 0.0,
        }),
    );
    let losers = if tiered {
        reveals
            .iter()
            .enumerate()
            .filter_map(|(i, reveal)| match *reveal {
                Reveal::Incorrect(height)
                    if change(i) < -SVD_NUMERICAL_TOLERANCE =>
                {
                    Some((i, height, -change(i) / REPUTATION_SMOOTHING_ALPHA))
                }
                Reveal::Silent | Reveal::Correct(_) | Reveal::Incorrect(_) => {
                    None
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    for (i, slices) in tier_slices(losers) {
        let raised: f64 = REVEAL_TIER_MULTIPLIERS
            .iter()
            .zip(slices)
            .map(|(m, s)| (m - 1.0) * s)
            .sum();
        extra[i] = REPUTATION_SMOOTHING_ALPHA * raised;
    }
    let pool = extra.sum();

    let mut adjusted = smooth_rep - extra;
    for (i, weight) in weights {
        adjusted[i] += pool * weight / total_weight;
    }
    adjusted
}

pub fn run_consensus(
    vote_matrix: &SparseVoteMatrix,
    reputation_vector: &VotingWeightVector,
    scaled_decisions: &std::collections::HashSet<
        crate::state::decisions::DecisionId,
    >,
    categorical_decisions: &HashMap<crate::state::decisions::DecisionId, u16>,
    reveal_heights: &HashMap<crate::types::Address, u32>,
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
                     - leaving its outcome to the default",
                    const_hex::encode(decision_id.as_bytes())
                );
                None
            };
            outcomes.insert(*decision_id, outcome);
        }
    }

    let (smooth_rep, first_loading, certainty) = match result {
        Some(result) => {
            let references: Vec<Option<f64>> = svd_pinned
                .iter()
                .enumerate()
                .map(|(j, pinned)| {
                    let unresolved = result.agents.deadlocked[j]
                        && result.decisions[j]
                            == BITCOIN_HIVEMIND_NEUTRAL_VALUE;
                    (result.has_votes[j] && !unresolved)
                        .then(|| pinned.unwrap_or(result.decisions[j]))
                })
                .collect();
            let heights: Vec<Option<u32>> = voters
                .iter()
                .map(|voter_id| reveal_heights.get(voter_id).copied())
                .collect();
            let revealed: Vec<bool> =
                heights.iter().map(Option::is_some).collect();
            let references = matched_outcomes(
                &svd_matrix,
                &references,
                &svd_scaled,
                &revealed,
            );
            let reveals: Vec<Reveal> = heights
                .iter()
                .enumerate()
                .map(|(i, height)| match *height {
                    None => Reveal::Silent,
                    Some(height)
                        if ballot_correct(
                            &svd_matrix,
                            i,
                            &references,
                            &svd_scaled,
                        ) =>
                    {
                        Reveal::Correct(height)
                    }
                    Some(height) => Reveal::Incorrect(height),
                })
                .collect();
            let smooth_rep = exclude_silent(
                result.agents.smooth_rep,
                &rep_vector,
                &result.agents.new_rep,
                &reveals,
            );
            let smooth_rep =
                apply_reveal_timing(&rep_vector, smooth_rep, &reveals);
            (
                smooth_rep,
                result.first_loading.iter().copied().collect(),
                result.certainty,
            )
        }
        None => (rep_vector, Vec::new(), 0.0),
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

    Ok(DetailedConsensusResult {
        outcomes,
        first_loading,
        certainty,
        updated_reputations,
        categorical_winners,
    })
}

#[cfg(test)]
#[allow(clippy::print_stdout, clippy::uninlined_format_args)]
mod tests {
    use super::*;
    use crate::state::decisions::DecisionId;
    use crate::types::Address;
    use nalgebra::{DMatrix, DVector};

    /// A voter's reputation, binary votes (empty when it cast none) and
    /// reveal height
    type Row<'a> = (f64, &'a [f64], Option<u32>);

    /// Consensus outcomes and reputations for `rows`, with the decisions
    /// at `scaled` scaled
    fn consensus(
        rows: &[Row],
        scaled: &[usize],
    ) -> (Vec<Option<f64>>, Vec<f64>) {
        let columns = rows.iter().map(|row| row.1.len()).max().unwrap_or(0);
        let voters: Vec<Address> =
            (0..rows.len() as u8).map(|i| Address([i; 20])).collect();
        let decisions: Vec<DecisionId> = (0..columns as u32)
            .map(|j| DecisionId::new(true, 1, j).unwrap())
            .collect();
        let mut matrix =
            SparseVoteMatrix::new(voters.clone(), decisions.clone());
        let mut rep = VotingWeightVector::new();
        let mut heights = HashMap::new();
        for (voter, &(share, votes, height)) in voters.iter().zip(rows) {
            rep.set_reputation(*voter, share);
            for (decision, &vote) in decisions.iter().zip(votes) {
                matrix.set_vote(*voter, *decision, vote).unwrap();
            }
            if let Some(height) = height {
                heights.insert(*voter, height);
            }
        }
        let scaled = scaled.iter().map(|&j| decisions[j]).collect();
        let result =
            run_consensus(&matrix, &rep, &scaled, &HashMap::new(), &heights)
                .unwrap();
        (
            decisions.iter().map(|d| result.outcomes[d]).collect(),
            voters
                .iter()
                .map(|v| result.updated_reputations[v])
                .collect(),
        )
    }

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

    /// An SVD tie between two mirrored voters moves no reputation
    #[test]
    fn exact_tie_neutral() {
        let m = DMatrix::from_row_slice(3, 1, &[1.0, 0.0, 0.5]);
        let rep = DVector::from_vec(vec![0.25, 0.25, 0.5]);
        let smooth_rep = get_reward_weights(
            &m,
            Some(&rep),
            0.3,
            &DVector::from_element(3, 1.0),
            1.0,
            &[None],
            &[true],
        )
        .unwrap()
        .smooth_rep;
        assert!((smooth_rep - rep).amax() < 1e-12);
    }

    /// Silent scaled cells take the votes' median, so silence can't drag
    #[test]
    fn scaled_fill_median() {
        let n = f64::NAN;
        let m = DMatrix::from_row_slice(
            5,
            2,
            &[0.30, 1.0, 0.30, 1.0, 0.31, 1.0, n, n, n, n],
        );
        let rep = DVector::from_vec(vec![0.1, 0.1, 0.1, 0.35, 0.35]);
        let result =
            factory(&m, Some(&rep), None, Some(&[true, false]), &[None; 2])
                .unwrap();
        assert_eq!(result.decisions[0], 0.30);
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
            &[false; 4],
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
            &[false; 4],
        )
        .unwrap()
        .smooth_rep;

        let clone_sum = split_smooth[1] + split_smooth[2] + split_smooth[3];
        assert!((whole_smooth[0] - split_smooth[0]).abs() < 1e-9);
        assert!((whole_smooth[1] - clone_sum).abs() < 1e-9);
        assert!((whole_smooth[2] - split_smooth[4]).abs() < 1e-9);
    }

    fn assert_close(actual: &DVector<f64>, expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < 1e-12, "index {i}: {a} != {e}");
        }
    }

    fn assert_slices(
        slices: &[(usize, [f64; 2])],
        expected: &[(usize, [f64; 2])],
    ) {
        assert_eq!(slices.len(), expected.len());
        for ((i, got), (j, want)) in slices.iter().zip(expected) {
            assert_eq!(i, j);
            assert!((got[0] - want[0]).abs() < 1e-12, "voter {i}");
            assert!((got[1] - want[1]).abs() < 1e-12, "voter {i}");
        }
    }

    /// An address straddling a tier boundary splits into t1, t2 and rest
    #[test]
    fn tier_split_straddle() {
        assert_slices(
            &tier_slices(vec![(2, 3, 0.93), (0, 1, 0.03), (1, 2, 0.04)]),
            &[(0, [0.03, 0.0]), (1, [0.02, 0.02]), (2, [0.0, 0.03])],
        );
    }

    /// Pro rata, a boundary block gives each voter the same tier fraction
    #[test]
    fn tier_same_block() {
        assert_slices(
            &tier_slices(vec![
                (0, 1, 0.02),
                (1, 2, 0.06),
                (2, 2, 0.02),
                (3, 3, 0.90),
            ]),
            &[
                (0, [0.02, 0.0]),
                (1, [0.0225, 0.0375]),
                (2, [0.0075, 0.0125]),
            ],
        );
    }

    /// A first block holding over 10% shares the whole pool pro rata
    #[test]
    fn first_block_shares() {
        let old = DVector::from_vec(vec![0.1, 0.3, 0.1, 0.5]);
        let smooth = DVector::from_vec(vec![0.07, 0.31, 0.11, 0.51]);
        let reveals = [
            Reveal::Incorrect(1),
            Reveal::Correct(1),
            Reveal::Correct(1),
            Reveal::Correct(2),
        ];
        let adjusted = apply_reveal_timing(&old, smooth, &reveals);
        assert_close(&adjusted, &[0.05875, 0.3184375, 0.1128125, 0.51]);
    }

    /// Early wrong revealers pay 1.5x/1.25x to early correct revealers
    #[test]
    fn early_wrong_penalized() {
        let old = DVector::from_vec(vec![0.04, 0.06, 0.4, 0.04, 0.06, 0.4]);
        let smooth =
            DVector::from_vec(vec![0.036, 0.054, 0.36, 0.044, 0.066, 0.44]);
        let reveals = [
            Reveal::Incorrect(1),
            Reveal::Incorrect(2),
            Reveal::Incorrect(3),
            Reveal::Correct(1),
            Reveal::Correct(2),
            Reveal::Correct(3),
        ];
        let adjusted = apply_reveal_timing(&old, smooth, &reveals);
        let third_extra =
            0.3 * (0.5 * (0.05 - 0.04 / 3.0 - 0.02) + 0.25 * 0.05);
        let pool = 0.002 + 0.003 + third_extra;
        assert_close(
            &adjusted,
            &[
                0.034,
                0.051,
                0.36 - third_extra,
                0.044 + pool * 0.06 / 0.1375,
                0.066 + pool * 0.0775 / 0.1375,
                0.44,
            ],
        );
        assert!((adjusted.sum() - 1.0).abs() < 1e-12);
    }

    /// Silence pays tier 1 untiered; an incorrect gainer is skipped
    #[test]
    fn silent_penalized_gainer_skipped() {
        let old = DVector::from_vec(vec![0.3, 0.3, 0.04, 0.36]);
        let smooth = DVector::from_vec(vec![0.21, 0.36, 0.036, 0.394]);
        let reveals = [
            Reveal::Silent,
            Reveal::Correct(5),
            Reveal::Incorrect(9),
            Reveal::Incorrect(1),
        ];
        let adjusted = apply_reveal_timing(&old, smooth, &reveals);
        assert_close(&adjusted, &[0.165, 0.407, 0.034, 0.394]);
    }

    /// A near-miss takes little tier space, leaving it to the fully wrong
    #[test]
    fn near_miss_small_space() {
        let old = DVector::from_vec(vec![0.1, 0.1, 0.3, 0.5]);
        let smooth = DVector::from_vec(vec![0.099, 0.07, 0.331, 0.5]);
        let reveals = [
            Reveal::Incorrect(1),
            Reveal::Incorrect(2),
            Reveal::Correct(1),
            Reveal::Correct(2),
        ];
        let adjusted = apply_reveal_timing(&old, smooth, &reveals);
        assert_close(&adjusted, &[0.0985, 0.05925, 0.34225, 0.5]);
    }

    /// Splitting a wrong address within a block changes no one's penalty
    #[test]
    fn punish_split_invariant() {
        let whole = apply_reveal_timing(
            &DVector::from_vec(vec![0.08, 0.08, 0.84]),
            DVector::from_vec(vec![0.056, 0.056, 0.888]),
            &[
                Reveal::Incorrect(1),
                Reveal::Incorrect(1),
                Reveal::Correct(1),
            ],
        );
        let split = apply_reveal_timing(
            &DVector::from_vec(vec![0.02, 0.02, 0.02, 0.02, 0.08, 0.84]),
            DVector::from_vec(vec![0.014, 0.014, 0.014, 0.014, 0.056, 0.888]),
            &[
                Reveal::Incorrect(1),
                Reveal::Incorrect(1),
                Reveal::Incorrect(1),
                Reveal::Incorrect(1),
                Reveal::Incorrect(1),
                Reveal::Correct(1),
            ],
        );
        let pieces: f64 = split.iter().take(4).sum();
        assert!((whole[0] - pieces).abs() < 1e-12);
        assert!((whole[1] - split[4]).abs() < 1e-12);
        assert!((whole[2] - split[5]).abs() < 1e-12);
    }

    /// A float-noise loss takes no tier space from a later wrong voter
    #[test]
    fn noise_not_loser() {
        let old = DVector::from_vec(vec![0.25, 0.1, 0.25, 0.4]);
        let smooth =
            DVector::from_vec(vec![0.25f64.next_down(), 0.07, 0.28, 0.4]);
        let reveals = [
            Reveal::Incorrect(1),
            Reveal::Incorrect(2),
            Reveal::Correct(1),
            Reveal::Correct(3),
        ];
        let adjusted = apply_reveal_timing(&old, smooth, &reveals);
        assert_close(&adjusted, &[0.25f64.next_down(), 0.05875, 0.29125, 0.4]);
    }

    /// A caught 0.5 Yes/No split moves no one; lone dissent still loses
    #[test]
    fn deadlock_rewards_neither() {
        for rows in [
            [(0.56, &[0.0][..], Some(1)), (0.43, &[1.0], Some(1))],
            [(0.56, &[1.0][..], Some(1)), (0.43, &[0.0], Some(1))],
            [(0.4, &[0.0][..], Some(1)), (0.3, &[1.0], Some(1))],
        ] {
            let middle = (1.0 - rows[0].0 - rows[1].0, &[0.5][..], Some(1));
            let rows = [rows[0], rows[1], middle];
            let (outcomes, reps) = consensus(&rows, &[]);
            assert_eq!(outcomes[0], Some(0.5));
            for (&(share, _, _), rep) in rows.iter().zip(reps) {
                assert!((rep - share).abs() < 1e-9);
            }
        }
        let (outcomes, reps) = consensus(
            &[
                (0.56, &[0.0], Some(1)),
                (0.43, &[1.0], Some(1)),
                (0.01, &[], None),
            ],
            &[],
        );
        assert_eq!(outcomes[0], Some(0.5));
        assert!(reps[0] >= 0.56 && reps[1] >= 0.43);
        assert!((reps[0] / 0.56 - reps[1] / 0.43).abs() < 1e-5);
        let (outcomes, reps) =
            consensus(&[(0.85, &[0.5], Some(1)), (0.15, &[1.0], Some(1))], &[]);
        assert_eq!(outcomes[0], Some(0.5));
        assert!(reps[0] > 0.85 && reps[1] < 0.15);
    }

    /// Outside dissent or a late Inconclusive dust neither breaks a deadlock
    #[test]
    fn deadlock_with_dissent() {
        for dust in [[0.0, 0.0], [1.0, 0.0]] {
            let (outcomes, reps) = consensus(
                &[
                    (0.52, &[1.0, 1.0], Some(1)),
                    (0.47, &[0.0, 1.0], Some(1)),
                    (0.01, &dust, Some(1)),
                ],
                &[],
            );
            assert_eq!(outcomes[0], Some(0.5));
            assert!((reps[0] / 0.52 - reps[1] / 0.47).abs() < 1e-5);
            assert!(reps[2] < 0.01);
        }
        let (outcomes, reps) = consensus(
            &[
                (0.5, &[1.0, 1.0], Some(1)),
                (0.4, &[0.0, 1.0], Some(1)),
                (0.099, &[], None),
                (0.001, &[0.5, 1.0], Some(9)),
            ],
            &[],
        );
        assert_eq!(outcomes[0], Some(0.5));
        assert!(reps[3] / 0.001 <= reps[0] / 0.5 + 1e-3);
        assert!((reps[0] / 0.5 - reps[1] / 0.4).abs() < 1e-5);
    }

    /// A deadlock that resolves to 1 pays no pool to an early 0 voter
    #[test]
    fn deadlock_judged_resolved() {
        let reps = |heights: [u32; 4]| {
            let r = |w: f64| w / 149.0;
            consensus(
                &[
                    (r(39.0), &[0.0, 0.0, 0.0, 0.0], Some(heights[0])),
                    (r(44.0), &[0.5, 1.0, 1.0, 0.5], Some(heights[1])),
                    (r(18.0), &[1.0, 0.0, 1.0, 0.0], Some(heights[2])),
                    (r(48.0), &[1.0, 0.0, 0.5, 1.0], Some(heights[3])),
                ],
                &[],
            )
        };
        let (outcomes, early) = reps([1, 2, 2, 2]);
        let (_, late) = reps([2, 1, 1, 1]);
        assert_eq!(outcomes[0], Some(1.0));
        assert_eq!(early[0], late[0]);
    }

    /// A fully wrong revealer loses the full step despite 50% silence
    #[test]
    fn liar_loses_despite_silence() {
        let (_, reps) = consensus(
            &[
                (0.2, &[1.0, 1.0, 0.0], Some(1)),
                (0.2, &[1.0, 1.0, 0.0], Some(1)),
                (0.1, &[0.0, 0.0, 1.0], Some(1)),
                (0.5, &[], None),
            ],
            &[],
        );
        assert!(reps[2] < 0.1 * 0.7 + 1e-6);
    }

    /// All-zero SVD scores fall back to revealers' old-reputation shares
    #[test]
    fn zero_score_fallback() {
        let old = DVector::from_vec(vec![0.2, 0.3, 0.5]);
        let new_rep = DVector::from_vec(vec![0.0, 0.0, 1.0]);
        let smooth = &new_rep * 0.3 + &old * 0.7;
        let reveals =
            [Reveal::Correct(1), Reveal::Incorrect(1), Reveal::Silent];
        let adjusted = exclude_silent(smooth, &old, &new_rep, &reveals);
        assert_close(&adjusted, &[0.26, 0.39, 0.35]);
    }

    /// With no revealers, silent exclusion leaves smoothing as is
    #[test]
    fn no_revealers_unchanged() {
        let old = DVector::from_vec(vec![0.5, 0.5]);
        let new_rep = DVector::from_vec(vec![0.6, 0.4]);
        let smooth = DVector::from_vec(vec![0.53, 0.47]);
        let adjusted = exclude_silent(
            smooth.clone(),
            &old,
            &new_rep,
            &[Reveal::Silent, Reveal::Silent],
        );
        assert_eq!(adjusted, smooth);
    }

    /// A correct loser is skipped, so a later correct gainer takes the pool
    #[test]
    fn correct_loser_skipped() {
        let old = DVector::from_vec(vec![0.1, 0.3, 0.6]);
        let smooth = DVector::from_vec(vec![0.07, 0.29, 0.64]);
        let reveals =
            [Reveal::Incorrect(1), Reveal::Correct(1), Reveal::Correct(2)];
        let adjusted = apply_reveal_timing(&old, smooth, &reveals);
        assert_close(&adjusted, &[0.05875, 0.29, 0.65125]);
    }

    /// With no correct revealer, the silent extra goes to the revealers
    #[test]
    fn no_correct_pays_revealers() {
        let old = DVector::from_vec(vec![0.5, 0.5]);
        let smooth = DVector::from_vec(vec![0.35, 0.65]);
        let reveals = [Reveal::Silent, Reveal::Incorrect(1)];
        let adjusted = apply_reveal_timing(&old, smooth, &reveals);
        assert_close(&adjusted, &[0.275, 0.725]);
    }

    /// With no correct gainer, silence still pays the correct revealers
    #[test]
    fn silent_pays_correct() {
        let old = DVector::from_vec(vec![0.4, 0.3, 0.3]);
        let smooth = DVector::from_vec(vec![0.28, 0.3, 0.42]);
        let reveals =
            [Reveal::Silent, Reveal::Correct(1), Reveal::Incorrect(2)];
        let adjusted = apply_reveal_timing(&old, smooth, &reveals);
        assert_close(&adjusted, &[0.22, 0.36, 0.42]);
    }

    /// A silent voter's alpha share goes to the revealers
    #[test]
    fn silent_earns_nothing() {
        let old = DVector::from_vec(vec![0.5, 0.25, 0.25]);
        let new_rep = DVector::from_vec(vec![0.2, 0.3, 0.5]);
        let smooth = &new_rep * 0.3 + &old * 0.7;
        let reveals =
            [Reveal::Silent, Reveal::Correct(1), Reveal::Incorrect(1)];
        let adjusted = exclude_silent(smooth, &old, &new_rep, &reveals);
        assert_close(&adjusted, &[0.35, 0.2875, 0.3625]);
    }

    /// A 1-ulp reputation change can't move the SVD via a noise loading
    #[test]
    fn noise_loading_stable() {
        let rows = [
            [1.0, 1.0, 0.0, 0.0],
            [0.5, 0.3, 0.0, 0.5],
            [0.7, 0.5, 1.0, 0.7],
            [1.0, 0.0, 1.0, 0.5],
        ];
        let m = DMatrix::from_fn(4, 4, |i, j| rows[i][j]);
        let reps = [126.0, 160.0, 4.0, 120.0];
        let total: f64 = reps.iter().sum();
        let rep = DVector::from_iterator(4, reps.iter().map(|r| r / total));
        let mut nudged = rep.clone();
        nudged[0] *= 1.0 + 4e-16;
        nudged /= nudged.sum();
        let weights = |rep: &DVector<f64>| {
            get_reward_weights(
                &m,
                Some(rep),
                0.3,
                &DVector::from_element(4, 1.0),
                1.0,
                &[None; 4],
                &[false; 4],
            )
            .unwrap()
            .smooth_rep
        };
        assert!((weights(&rep) - weights(&nudged)).amax() < 1e-9);
    }

    /// A scaled vote leaning off the weighted median loses reputation
    #[test]
    fn scaled_lean_loses() {
        let honest = (0.2, &[0.35, 1.0][..], Some(1));
        let (_, reps) = consensus(
            &[honest, honest, honest, honest, (0.2, &[0.34, 1.0], Some(1))],
            &[0],
        );
        assert!(reps[4] < 0.2);
        assert!(reps[0] > 0.2);
    }

    /// A missing cell on a column that has an outcome is a miss
    #[test]
    fn missing_vote_incorrect() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, f64::NAN, 1.0, 0.0]);
        let scaled = [false, false];
        let outcomes = [Some(1.0), Some(0.0)];
        assert!(!ballot_correct(&m, 0, &outcomes, &scaled));
        assert!(ballot_correct(&m, 1, &outcomes, &scaled));
        assert!(ballot_correct(&m, 0, &[Some(1.0), None], &scaled));
        assert!(!ballot_correct(&m, 0, &[None, None], &scaled));
    }

    /// A decision only silent rows or no one matched leaves the check
    #[test]
    fn unmatched_dropped() {
        let m = DMatrix::from_row_slice(
            3,
            3,
            &[1.0, 0.0, 0.4, 1.0, 1.0, 0.4, 0.5, 0.5, 0.5],
        );
        let outcomes = [Some(1.0), Some(0.5), Some(0.5)];
        let matched = matched_outcomes(
            &m,
            &outcomes,
            &[false, false, true],
            &[true, true, false],
        );
        assert_eq!(matched, vec![Some(1.0), None, None]);
    }

    /// A scaled vote matches within 1% of the range, even at the minimum
    #[test]
    fn scaled_one_percent() {
        let m = DMatrix::from_row_slice(4, 1, &[0.509, 0.511, 0.009, 0.011]);
        let scaled = [true];
        assert!(ballot_correct(&m, 0, &[Some(0.5)], &scaled));
        assert!(!ballot_correct(&m, 1, &[Some(0.5)], &scaled));
        assert!(ballot_correct(&m, 2, &[Some(0.0)], &scaled));
        assert!(!ballot_correct(&m, 3, &[Some(0.0)], &scaled));
    }

    /// Reveal heights move reputation, never outcomes; block 1 takes all
    #[test]
    fn timing_keeps_outcomes() {
        let (yes, no) = (&[1.0, 1.0, 0.0, 1.0][..], &[0.0, 0.0, 1.0, 0.0][..]);
        let rows = |heights: [Option<u32>; 4]| {
            [
                (0.2, yes, heights[0]),
                (0.2, yes, heights[1]),
                (0.2, yes, heights[2]),
                (0.2, no, heights[3]),
                (0.2, &[][..], None),
            ]
        };
        let (plain_outcomes, plain_reps) = consensus(&rows([None; 4]), &[]);
        let (timed_outcomes, timed_reps) =
            consensus(&rows([Some(1), Some(2), Some(3), Some(1)]), &[]);
        assert_eq!(plain_outcomes, timed_outcomes);
        assert!((timed_reps[4] - 0.11).abs() < 1e-6);
        assert!(timed_reps[0] > timed_reps[1]);
        assert!((timed_reps[1] - timed_reps[2]).abs() < 1e-6);
        assert!(timed_reps[3] < plain_reps[3]);
        assert!((timed_reps.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }
}
