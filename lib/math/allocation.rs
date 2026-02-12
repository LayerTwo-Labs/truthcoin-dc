//! Single source of truth for proportional allocation with remainder handling.
//!
//! Algorithm:
//! 1. Calculate each participant's proportion: `weight_i / total_weight`
//! 2. Allocate proportional share: `proportion * pool`
//! 3. Convert to integer using `Rounding::Nearest`
//! 4. Handle remainder:
//!    - If under-allocated: add remainder to largest allocation
//!    - If over-allocated: subtract from smallest allocation
//! 5. Assert conservation: `sum(allocations) == pool`

use super::safe_math::{Rounding, SatoshiError, to_sats, to_sats_signed};

/// Result of a proportional allocation (unsigned).
#[derive(Debug, Clone)]
pub struct AllocationResultU64<K> {
    /// List of (key, allocated_amount) pairs
    pub allocations: Vec<(K, u64)>,
    /// Total amount allocated (should equal pool after adjustment)
    pub total_allocated: u64,
}

/// Result of a proportional allocation (signed, for votecoin deltas).
#[derive(Debug, Clone)]
pub struct AllocationResultI64<K> {
    /// List of (key, allocated_amount) pairs
    pub allocations: Vec<(K, i64)>,
    /// Total amount allocated (should equal pool after adjustment)
    pub total_allocated: i64,
}

/// Allocate a pool proportionally among participants (unsigned).
///
/// Uses `Rounding::Nearest` for fair allocation, then adjusts remainder:
/// - If under-allocated: add remainder to largest allocation
/// - If over-allocated: subtract from smallest allocation
///
/// # Arguments
/// * `participants` - List of (key, weight) pairs where weight determines proportion
/// * `total_pool` - Total amount to distribute
///
/// # Returns
/// `AllocationResultU64` where `sum(allocations) == total_pool` exactly
///
/// # Example
/// ```ignore
/// let participants = vec![
///     ("alice", 30.0),  // 30% weight
///     ("bob", 70.0),    // 70% weight
/// ];
/// let result = allocate_proportionally_u64(participants, 100)?;
/// // alice gets ~30, bob gets ~70, sum is exactly 100
/// ```
pub fn allocate_proportionally_u64<K: Clone>(
    participants: Vec<(K, f64)>,
    total_pool: u64,
) -> Result<AllocationResultU64<K>, SatoshiError> {
    if participants.is_empty() || total_pool == 0 {
        return Ok(AllocationResultU64 {
            allocations: Vec::new(),
            total_allocated: 0,
        });
    }

    let total_weight: f64 = participants.iter().map(|(_, w)| *w).sum();

    if total_weight <= 0.0 {
        return Ok(AllocationResultU64 {
            allocations: Vec::new(),
            total_allocated: 0,
        });
    }

    // Phase 1: Calculate allocations with Rounding::Nearest
    let mut allocations: Vec<(K, u64)> = Vec::with_capacity(participants.len());
    let mut total_allocated: u64 = 0;

    for (key, weight) in participants {
        if weight < f64::EPSILON {
            continue;
        }

        let proportion = weight / total_weight;
        let allocation_f64 = proportion * (total_pool as f64);
        let allocation_sats = to_sats(allocation_f64, Rounding::Nearest)?;

        if allocation_sats > 0 {
            allocations.push((key, allocation_sats));
            total_allocated += allocation_sats;
        }
    }

    // Phase 2: Adjust for rounding remainder
    if !allocations.is_empty() {
        match total_allocated.cmp(&total_pool) {
            std::cmp::Ordering::Less => {
                // Under-allocated: add remainder to largest
                let remainder = total_pool - total_allocated;
                let largest_idx = allocations
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, (_, amt))| *amt)
                    .map(|(idx, _)| idx)
                    .unwrap();
                allocations[largest_idx].1 += remainder;
                total_allocated += remainder;
            }
            std::cmp::Ordering::Greater => {
                // Over-allocated: subtract from smallest
                let excess = total_allocated - total_pool;
                let smallest_idx = allocations
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, (_, amt))| *amt)
                    .map(|(idx, _)| idx)
                    .unwrap();
                allocations[smallest_idx].1 =
                    allocations[smallest_idx].1.saturating_sub(excess);
                total_allocated -= excess;
            }
            std::cmp::Ordering::Equal => {
                // Perfect allocation, no adjustment needed
            }
        }
    }

    debug_assert_eq!(
        total_allocated, total_pool,
        "Conservation violated: allocated {total_allocated} but pool was {total_pool}"
    );

    Ok(AllocationResultU64 {
        allocations,
        total_allocated,
    })
}

/// Allocate a pool proportionally among participants (signed).
///
/// Used for votecoin redistribution where participants can have positive
/// (gain) or negative (loss) allocations.
///
/// Uses `Rounding::Nearest` for fair allocation, then adjusts remainder:
/// - If under-allocated (abs): add remainder to largest absolute allocation
/// - If over-allocated (abs): subtract from smallest absolute allocation
///
/// # Arguments
/// * `participants` - List of (key, weight) pairs where weight determines proportion
/// * `total_pool` - Total amount to distribute (can be positive or negative)
///
/// # Returns
/// `AllocationResultI64` where `sum(allocations) == total_pool` exactly
pub fn allocate_proportionally_i64<K: Clone>(
    participants: Vec<(K, f64)>,
    total_pool: i64,
) -> Result<AllocationResultI64<K>, SatoshiError> {
    if participants.is_empty() || total_pool == 0 {
        return Ok(AllocationResultI64 {
            allocations: Vec::new(),
            total_allocated: 0,
        });
    }

    let total_weight: f64 = participants.iter().map(|(_, w)| w.abs()).sum();

    if total_weight <= 0.0 {
        return Ok(AllocationResultI64 {
            allocations: Vec::new(),
            total_allocated: 0,
        });
    }

    // Phase 1: Calculate allocations with Rounding::Nearest
    let mut allocations: Vec<(K, i64)> = Vec::with_capacity(participants.len());
    let mut total_allocated: i64 = 0;

    for (key, weight) in participants {
        if weight.abs() < f64::EPSILON {
            continue;
        }

        // Use absolute weight for proportion, preserve sign in result
        let proportion = weight.abs() / total_weight;
        let allocation_f64 = proportion * (total_pool as f64);
        let allocation = to_sats_signed(allocation_f64, Rounding::Nearest)?;

        if allocation != 0 {
            allocations.push((key, allocation));
            total_allocated += allocation;
        }
    }

    // Phase 2: Adjust for rounding remainder
    if !allocations.is_empty() {
        let diff = total_pool - total_allocated;
        if diff != 0 {
            if total_pool > 0 {
                // Positive pool: add to largest positive or subtract from largest negative
                if diff > 0 {
                    // Under-allocated positive pool: add to largest
                    let largest_idx = allocations
                        .iter()
                        .enumerate()
                        .max_by_key(|(_, (_, amt))| *amt)
                        .map(|(idx, _)| idx)
                        .unwrap();
                    allocations[largest_idx].1 += diff;
                } else {
                    // Over-allocated positive pool: subtract from smallest
                    let smallest_idx = allocations
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, (_, amt))| *amt)
                        .map(|(idx, _)| idx)
                        .unwrap();
                    allocations[smallest_idx].1 += diff; // diff is negative
                }
            } else {
                // Negative pool (losses): add to most negative or subtract from least negative
                if diff < 0 {
                    // Under-allocated (more negative needed): make largest more negative
                    let most_negative_idx = allocations
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, (_, amt))| *amt)
                        .map(|(idx, _)| idx)
                        .unwrap();
                    allocations[most_negative_idx].1 += diff;
                } else {
                    // Over-allocated (less negative needed): make smallest less negative
                    let least_negative_idx = allocations
                        .iter()
                        .enumerate()
                        .max_by_key(|(_, (_, amt))| *amt)
                        .map(|(idx, _)| idx)
                        .unwrap();
                    allocations[least_negative_idx].1 += diff;
                }
            }
            total_allocated = allocations.iter().map(|(_, amt)| *amt).sum();
        }
    }

    debug_assert_eq!(
        total_allocated, total_pool,
        "Conservation violated: allocated {total_allocated} but pool was {total_pool}"
    );

    Ok(AllocationResultI64 {
        allocations,
        total_allocated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_uses_nearest_rounding() {
        // With 3 participants of equal weight and pool of 100,
        // each should get 33 or 34 (100/3 = 33.33...)
        let participants = vec![("a", 1.0), ("b", 1.0), ("c", 1.0)];
        let result = allocate_proportionally_u64(participants, 100).unwrap();

        // All allocations should be 33 or 34
        for (_, amt) in &result.allocations {
            assert!(*amt == 33 || *amt == 34);
        }
    }

    #[test]
    fn test_allocation_conservation() {
        let participants =
            vec![("alice", 30.0), ("bob", 50.0), ("charlie", 20.0)];
        let pool = 1000u64;

        let result = allocate_proportionally_u64(participants, pool).unwrap();

        // Sum must equal pool exactly
        let sum: u64 = result.allocations.iter().map(|(_, amt)| *amt).sum();
        assert_eq!(sum, pool);
        assert_eq!(result.total_allocated, pool);
    }

    #[test]
    fn test_remainder_to_largest_when_under() {
        // Create scenario where rounding leaves remainder
        // 3 equal participants, pool of 100 -> 33.33 each
        // Rounds to 33 each = 99, remainder 1 goes to one of them
        let participants = vec![("a", 1.0), ("b", 1.0), ("c", 1.0)];
        let result = allocate_proportionally_u64(participants, 100).unwrap();

        assert_eq!(result.total_allocated, 100);
        // One participant should have 34, others 33
        let counts: Vec<u64> =
            result.allocations.iter().map(|(_, a)| *a).collect();
        assert!(counts.contains(&34) || counts.iter().all(|&x| x == 33));
    }

    #[test]
    fn test_remainder_from_smallest_when_over() {
        // Create scenario where rounding over-allocates
        // This is harder to construct, but the logic is tested
        let participants = vec![("a", 0.5), ("b", 0.5)];
        let result = allocate_proportionally_u64(participants, 101).unwrap();

        assert_eq!(result.total_allocated, 101);
    }

    #[test]
    fn test_signed_allocation_conservation() {
        let participants = vec![("winner1", 60.0), ("winner2", 40.0)];
        let pool = 100i64;

        let result = allocate_proportionally_i64(participants, pool).unwrap();

        let sum: i64 = result.allocations.iter().map(|(_, amt)| *amt).sum();
        assert_eq!(sum, pool);
    }

    #[test]
    fn test_signed_allocation_negative_pool() {
        // Negative pool represents losses
        let participants = vec![("loser1", 70.0), ("loser2", 30.0)];
        let pool = -100i64;

        let result = allocate_proportionally_i64(participants, pool).unwrap();

        let sum: i64 = result.allocations.iter().map(|(_, amt)| *amt).sum();
        assert_eq!(sum, pool);

        // All allocations should be negative
        for (_, amt) in &result.allocations {
            assert!(*amt < 0);
        }
    }

    #[test]
    fn test_empty_participants() {
        let result = allocate_proportionally_u64::<&str>(vec![], 100).unwrap();
        assert!(result.allocations.is_empty());
        assert_eq!(result.total_allocated, 0);
    }

    #[test]
    fn test_zero_pool() {
        let participants = vec![("a", 50.0), ("b", 50.0)];
        let result = allocate_proportionally_u64(participants, 0).unwrap();
        assert!(result.allocations.is_empty());
        assert_eq!(result.total_allocated, 0);
    }

    #[test]
    fn test_zero_weight_participants_skipped() {
        let participants = vec![
            ("a", 100.0),
            ("b", 0.0), // Should be skipped
            ("c", 0.0), // Should be skipped
        ];
        let result = allocate_proportionally_u64(participants, 100).unwrap();

        // Only "a" should receive allocation
        assert_eq!(result.allocations.len(), 1);
        assert_eq!(result.allocations[0].0, "a");
        assert_eq!(result.allocations[0].1, 100);
    }
}
