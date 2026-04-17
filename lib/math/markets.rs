//! Market-structure math: outcome-space combinatorics and storage-fee
//! formula. Pure functions over primitive inputs — no `state/` imports.

pub const BASE_MARKET_STORAGE_COST_SATS: u64 = 1000;

pub const L2_STORAGE_RATE_SATS_PER_BYTE: u64 = 1;

pub fn cartesian_product(dimensions: &[usize]) -> Vec<Vec<usize>> {
    if dimensions.is_empty() {
        return vec![vec![]];
    }

    let expected_size: usize = dimensions.iter().product();
    let mut result = Vec::with_capacity(expected_size);
    result.push(vec![]);

    for &dim_size in dimensions {
        let mut new_result = Vec::with_capacity(result.len() * dim_size);
        for combo in result {
            for value in 0..dim_size {
                let mut new_combo = combo.clone();
                new_combo.push(value);
                new_result.push(new_combo);
            }
        }
        result = new_result;
    }

    result
}

pub fn outcome_space_size(dimensions: &[usize]) -> Option<usize> {
    dimensions
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

pub fn market_storage_fee(tradeable_outcomes: usize) -> u64 {
    let quadratic =
        (tradeable_outcomes as u64).pow(2) * L2_STORAGE_RATE_SATS_PER_BYTE;
    BASE_MARKET_STORAGE_COST_SATS + quadratic
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cartesian_product_empty() {
        let expected: Vec<Vec<usize>> = vec![vec![]];
        assert_eq!(cartesian_product(&[]), expected);
    }

    #[test]
    fn test_cartesian_product_single_dim() {
        let combos = cartesian_product(&[3]);
        assert_eq!(combos, vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn test_cartesian_product_two_dims() {
        let combos = cartesian_product(&[2, 3]);
        assert_eq!(
            combos,
            vec![
                vec![0, 0],
                vec![0, 1],
                vec![0, 2],
                vec![1, 0],
                vec![1, 1],
                vec![1, 2],
            ]
        );
    }

    #[test]
    fn test_cartesian_product_three_dims() {
        let combos = cartesian_product(&[3, 3, 3]);
        assert_eq!(combos.len(), 27);
    }

    #[test]
    fn test_outcome_space_size_basic() {
        assert_eq!(outcome_space_size(&[]), Some(1));
        assert_eq!(outcome_space_size(&[3]), Some(3));
        assert_eq!(outcome_space_size(&[3, 3, 3]), Some(27));
        assert_eq!(outcome_space_size(&[2, 5, 4]), Some(40));
    }

    #[test]
    fn test_outcome_space_size_overflow() {
        assert_eq!(outcome_space_size(&[usize::MAX, 2]), None);
    }

    #[test]
    fn test_market_storage_fee() {
        assert_eq!(market_storage_fee(0), BASE_MARKET_STORAGE_COST_SATS);
        assert_eq!(market_storage_fee(9), 1000 + 81);
        assert_eq!(market_storage_fee(27), 1000 + 729);
    }
}
