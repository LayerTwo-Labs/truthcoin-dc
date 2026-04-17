use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(
    BorshSerialize,
    Clone,
    Copy,
    Debug,
    Default,
    Deserialize,
    Eq,
    Hash,
    PartialEq,
    Serialize,
    ToSchema,
)]
pub struct TxPowConfig {
    pub hash_selector: u8,
    pub ordering: u8,
    pub difficulty: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HashFunction {
    Sha256,
    Sha512,
    Sha3_256,
    Sha3_512,
    Sha512_256,
    Sha384,
    Blake3,
    Shake256,
}

const ALL_HASH_FUNCTIONS: [HashFunction; 8] = [
    HashFunction::Sha256,
    HashFunction::Sha512,
    HashFunction::Sha3_256,
    HashFunction::Sha3_512,
    HashFunction::Sha512_256,
    HashFunction::Sha384,
    HashFunction::Blake3,
    HashFunction::Shake256,
];

impl HashFunction {
    pub fn hash(self, data: &[u8]) -> [u8; 32] {
        match self {
            HashFunction::Sha256 => {
                use sha2::Digest as _;
                let result = sha2::Sha256::digest(data);
                let mut out = [0u8; 32];
                out.copy_from_slice(&result);
                out
            }
            HashFunction::Sha512 => {
                use sha2::Digest as _;
                let result = sha2::Sha512::digest(data);
                let mut out = [0u8; 32];
                out.copy_from_slice(&result[..32]);
                out
            }
            HashFunction::Sha3_256 => {
                use sha3::Digest as _;
                let result = sha3::Sha3_256::digest(data);
                let mut out = [0u8; 32];
                out.copy_from_slice(&result);
                out
            }
            HashFunction::Sha3_512 => {
                use sha3::Digest as _;
                let result = sha3::Sha3_512::digest(data);
                let mut out = [0u8; 32];
                out.copy_from_slice(&result[..32]);
                out
            }
            HashFunction::Sha512_256 => {
                use sha2::Digest as _;
                let result = sha2::Sha512_256::digest(data);
                let mut out = [0u8; 32];
                out.copy_from_slice(&result);
                out
            }
            HashFunction::Sha384 => {
                use sha2::Digest as _;
                let result = sha2::Sha384::digest(data);
                let mut out = [0u8; 32];
                out.copy_from_slice(&result[..32]);
                out
            }
            HashFunction::Blake3 => {
                let result = blake3::hash(data);
                *result.as_bytes()
            }
            HashFunction::Shake256 => {
                use sha3::digest::{ExtendableOutput as _, Update as _};
                let mut hasher = sha3::Shake256::default();
                hasher.update(data);
                let mut out = [0u8; 32];
                sha3::digest::XofReader::read(
                    &mut hasher.finalize_xof(),
                    &mut out,
                );
                out
            }
        }
    }
}

impl TxPowConfig {
    pub fn is_enabled(&self) -> bool {
        self.difficulty > 0
    }

    pub fn selected_functions(&self) -> Vec<HashFunction> {
        ALL_HASH_FUNCTIONS
            .iter()
            .enumerate()
            .filter(|(bit, _)| self.hash_selector & (1 << bit) != 0)
            .map(|(_, func)| *func)
            .collect()
    }

    pub fn ordered_functions(&self) -> Vec<HashFunction> {
        let selected = self.selected_functions();
        let n = selected.len();
        if n <= 1 {
            return selected;
        }
        let perm_index = (self.ordering as usize) % factorial(n);
        apply_lehmer_permutation(&selected, perm_index)
    }

    pub fn compute_hash(&self, data: &[u8]) -> [u8; 32] {
        let functions = self.ordered_functions();
        if functions.is_empty() {
            return [0u8; 32];
        }
        let mut current = functions[0].hash(data);
        for func in &functions[1..] {
            current = func.hash(&current);
        }
        current
    }

    pub fn verify(&self, data: &[u8], nonce: u64) -> bool {
        if !self.is_enabled() {
            return true;
        }
        let mut input = data.to_vec();
        input.extend_from_slice(&nonce.to_le_bytes());
        let hash = self.compute_hash(&input);
        count_leading_zero_bits(&hash) >= self.difficulty as u32
    }

    pub fn mine(&self, data: &[u8]) -> u64 {
        if !self.is_enabled() {
            return 0;
        }
        (0u64..)
            .find(|&nonce| self.verify(data, nonce))
            .expect("TX-PoW mine: u64 space exhausted without finding nonce")
    }

    pub fn validate(&self) -> bool {
        if self.difficulty == 0 {
            return true;
        }
        if self.difficulty > MAX_POW_DIFFICULTY {
            return false;
        }
        self.hash_selector != 0
    }
}

pub const MAX_POW_DIFFICULTY: u8 = 32;

pub const POW_BOUND_WINDOW_BLOCKS: u32 = 10;

pub fn serialize_trade_for_pow(
    market_id: &[u8; 6],
    outcome_index: u32,
    shares: i64,
    trader: &crate::types::Address,
    limit_sats: u64,
    prev_block_hash: &crate::types::BlockHash,
) -> Vec<u8> {
    #[derive(BorshSerialize)]
    struct TradePowInput<'a> {
        market_id: &'a [u8; 6],
        outcome_index: u32,
        shares: i64,
        trader: &'a crate::types::Address,
        limit_sats: u64,
        prev_block_hash: &'a crate::types::BlockHash,
    }
    let input = TradePowInput {
        market_id,
        outcome_index,
        shares,
        trader,
        limit_sats,
        prev_block_hash,
    };
    borsh::to_vec(&input)
        .expect("BorshSerialize should not fail for TradePowInput")
}

fn count_leading_zero_bits(data: &[u8]) -> u32 {
    let mut count = 0u32;
    for byte in data {
        if *byte == 0 {
            count += 8;
        } else {
            count += byte.leading_zeros();
            break;
        }
    }
    count
}

fn factorial(n: usize) -> usize {
    (1..=n).product()
}

fn apply_lehmer_permutation<T: Clone>(items: &[T], mut index: usize) -> Vec<T> {
    let n = items.len();
    let mut available: Vec<T> = items.to_vec();
    let mut result = Vec::with_capacity(n);

    for i in (1..=n).rev() {
        let fact = factorial(i - 1);
        let chosen = index / fact;
        index %= fact;
        result.push(available.remove(chosen));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_produces_correct_output() {
        use sha2::Digest as _;
        let data = b"hello world";
        let expected = sha2::Sha256::digest(data);
        let result = HashFunction::Sha256.hash(data);
        assert_eq!(result, expected.as_slice());
    }

    #[test]
    fn test_blake3_produces_correct_output() {
        let data = b"hello world";
        let expected = blake3::hash(data);
        let result = HashFunction::Blake3.hash(data);
        assert_eq!(result, *expected.as_bytes());
    }

    #[test]
    fn test_each_hash_function_produces_32_bytes() {
        let data = b"test data for hashing";
        for func in &ALL_HASH_FUNCTIONS {
            let result = func.hash(data);
            assert_eq!(result.len(), 32, "{func:?} did not produce 32 bytes");
        }
    }

    #[test]
    fn test_different_hash_functions_produce_different_output() {
        let data = b"unique test data";
        let sha256 = HashFunction::Sha256.hash(data);
        let sha3_256 = HashFunction::Sha3_256.hash(data);
        let blake3 = HashFunction::Blake3.hash(data);
        assert_ne!(sha256, sha3_256);
        assert_ne!(sha256, blake3);
        assert_ne!(sha3_256, blake3);
    }

    #[test]
    fn test_bitmask_selection_single() {
        let config = TxPowConfig {
            hash_selector: 0b0000_0001,
            ordering: 0,
            difficulty: 1,
        };
        let selected = config.selected_functions();
        assert_eq!(selected, vec![HashFunction::Sha256]);
    }

    #[test]
    fn test_bitmask_selection_multiple() {
        let config = TxPowConfig {
            hash_selector: 0b0100_0101,
            ordering: 0,
            difficulty: 1,
        };
        let selected = config.selected_functions();
        assert_eq!(
            selected,
            vec![
                HashFunction::Sha256,
                HashFunction::Sha3_256,
                HashFunction::Blake3
            ]
        );
    }

    #[test]
    fn test_bitmask_selection_all() {
        let config = TxPowConfig {
            hash_selector: 0xFF,
            ordering: 0,
            difficulty: 1,
        };
        let selected = config.selected_functions();
        assert_eq!(selected.len(), 8);
    }

    #[test]
    fn test_bitmask_selection_none() {
        let config = TxPowConfig {
            hash_selector: 0,
            ordering: 0,
            difficulty: 1,
        };
        let selected = config.selected_functions();
        assert!(selected.is_empty());
    }

    #[test]
    fn test_lehmer_identity_permutation() {
        let config = TxPowConfig {
            hash_selector: 0b0000_0111,
            ordering: 0,
            difficulty: 1,
        };
        let ordered = config.ordered_functions();
        assert_eq!(
            ordered,
            vec![
                HashFunction::Sha256,
                HashFunction::Sha512,
                HashFunction::Sha3_256
            ]
        );
    }

    #[test]
    fn test_lehmer_reverse_permutation() {
        let config = TxPowConfig {
            hash_selector: 0b0000_0111,
            ordering: 5,
            difficulty: 1,
        };
        let ordered = config.ordered_functions();
        assert_eq!(
            ordered,
            vec![
                HashFunction::Sha3_256,
                HashFunction::Sha512,
                HashFunction::Sha256
            ]
        );
    }

    #[test]
    fn test_lehmer_wraps_on_overflow() {
        let config_0 = TxPowConfig {
            hash_selector: 0b0000_0011,
            ordering: 0,
            difficulty: 1,
        };
        let config_2 = TxPowConfig {
            hash_selector: 0b0000_0011,
            ordering: 2,
            difficulty: 1,
        };
        assert_eq!(config_0.ordered_functions(), config_2.ordered_functions());
    }

    #[test]
    fn test_hash_chaining() {
        let data = b"chain test";
        let sha256_result = HashFunction::Sha256.hash(data);
        let blake3_of_sha256 = HashFunction::Blake3.hash(&sha256_result);

        let config = TxPowConfig {
            hash_selector: 0b0100_0001,
            ordering: 0,
            difficulty: 0,
        };
        let chained = config.compute_hash(data);
        assert_eq!(chained, blake3_of_sha256);
    }

    #[test]
    fn test_leading_zero_bits() {
        assert_eq!(count_leading_zero_bits(&[0x00, 0x00, 0x01]), 23);
        assert_eq!(count_leading_zero_bits(&[0x80, 0x00]), 0);
        assert_eq!(count_leading_zero_bits(&[0x40, 0x00]), 1);
        assert_eq!(count_leading_zero_bits(&[0x00, 0x80]), 8);
        assert_eq!(count_leading_zero_bits(&[0x00, 0x00]), 16);
        assert_eq!(count_leading_zero_bits(&[0x01]), 7);
        assert_eq!(count_leading_zero_bits(&[0xFF]), 0);
        assert_eq!(count_leading_zero_bits(&[0x00, 0x01]), 15);
    }

    #[test]
    fn test_mine_and_verify() {
        let config = TxPowConfig {
            hash_selector: 0b0000_0001,
            ordering: 0,
            difficulty: 4,
        };
        let data = b"mine test data";
        let nonce = config.mine(data);
        assert!(config.verify(data, nonce));
    }

    #[test]
    fn test_verify_rejects_invalid_nonce() {
        let config = TxPowConfig {
            hash_selector: 0b0000_0001,
            ordering: 0,
            difficulty: 16,
        };
        let data = b"reject test";
        assert!(!config.verify(data, 0));
        assert!(!config.verify(data, 1));
    }

    #[test]
    fn test_disabled_pow_always_passes() {
        let config = TxPowConfig {
            hash_selector: 0,
            ordering: 0,
            difficulty: 0,
        };
        assert!(config.verify(b"anything", 0));
        assert!(config.verify(b"anything", 42));
    }

    #[test]
    fn test_mine_disabled_returns_zero() {
        let config = TxPowConfig {
            hash_selector: 0,
            ordering: 0,
            difficulty: 0,
        };
        assert_eq!(config.mine(b"data"), 0);
    }

    #[test]
    fn test_validate_config() {
        let valid = TxPowConfig {
            hash_selector: 1,
            ordering: 0,
            difficulty: 8,
        };
        assert!(valid.validate());

        let disabled = TxPowConfig {
            hash_selector: 0,
            ordering: 0,
            difficulty: 0,
        };
        assert!(disabled.validate());

        let invalid = TxPowConfig {
            hash_selector: 0,
            ordering: 0,
            difficulty: 8,
        };
        assert!(!invalid.validate());

        let at_cap = TxPowConfig {
            hash_selector: 1,
            ordering: 0,
            difficulty: MAX_POW_DIFFICULTY,
        };
        assert!(at_cap.validate());

        let over_cap = TxPowConfig {
            hash_selector: 1,
            ordering: 0,
            difficulty: MAX_POW_DIFFICULTY + 1,
        };
        assert!(!over_cap.validate());
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(2), 2);
        assert_eq!(factorial(3), 6);
        assert_eq!(factorial(4), 24);
        assert_eq!(factorial(5), 120);
    }

    #[test]
    fn test_all_permutations_of_three() {
        let items = vec![
            HashFunction::Sha256,
            HashFunction::Sha512,
            HashFunction::Sha3_256,
        ];
        let mut perms: Vec<Vec<HashFunction>> = Vec::new();
        for i in 0..6 {
            perms.push(apply_lehmer_permutation(&items, i));
        }
        assert_eq!(perms.len(), 6);
        for i in 0..6 {
            for j in (i + 1)..6 {
                assert_ne!(perms[i], perms[j]);
            }
        }
    }

    #[test]
    fn test_serialize_trade_for_pow() {
        let addr = crate::types::Address([0u8; 20]);
        let block = crate::types::BlockHash([7u8; 32]);
        let data = serialize_trade_for_pow(
            &[1, 2, 3, 4, 5, 6],
            0,
            100,
            &addr,
            50000,
            &block,
        );
        assert!(!data.is_empty());

        let data2 = serialize_trade_for_pow(
            &[1, 2, 3, 4, 5, 6],
            1,
            100,
            &addr,
            50000,
            &block,
        );
        assert_ne!(data, data2);
    }

    #[test]
    fn test_serialize_trade_for_pow_different_prev_block() {
        let addr = crate::types::Address([0u8; 20]);
        let block_a = crate::types::BlockHash([1u8; 32]);
        let block_b = crate::types::BlockHash([2u8; 32]);
        let data_a = serialize_trade_for_pow(
            &[1, 2, 3, 4, 5, 6],
            0,
            100,
            &addr,
            50000,
            &block_a,
        );
        let data_b = serialize_trade_for_pow(
            &[1, 2, 3, 4, 5, 6],
            0,
            100,
            &addr,
            50000,
            &block_b,
        );
        assert_ne!(data_a, data_b);
    }

    #[test]
    fn test_pow_nonce_bound_to_prev_block() {
        let config = TxPowConfig {
            hash_selector: 0b0000_0001,
            ordering: 0,
            difficulty: 4,
        };
        let addr = crate::types::Address([0u8; 20]);
        let block_a = crate::types::BlockHash([1u8; 32]);
        let block_b = crate::types::BlockHash([2u8; 32]);
        let data_a = serialize_trade_for_pow(
            &[1, 2, 3, 4, 5, 6],
            0,
            100,
            &addr,
            50000,
            &block_a,
        );
        let data_b = serialize_trade_for_pow(
            &[1, 2, 3, 4, 5, 6],
            0,
            100,
            &addr,
            50000,
            &block_b,
        );
        let nonce_for_a = config.mine(&data_a);
        assert!(config.verify(&data_a, nonce_for_a));
        assert!(!config.verify(&data_b, nonce_for_a));
    }

    #[test]
    fn test_single_function_no_chaining() {
        let data = b"single function test";
        let config = TxPowConfig {
            hash_selector: 0b0000_0001,
            ordering: 0,
            difficulty: 0,
        };
        let result = config.compute_hash(data);
        let expected = HashFunction::Sha256.hash(data);
        assert_eq!(result, expected);
    }
}
