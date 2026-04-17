use super::types::{
    BASE_LISTING_FEE, Decision, DecisionId, DecisionType, calculate_listing_fee,
};

fn make_binary_decision() -> Decision {
    Decision {
        market_maker_pubkey_hash: [0u8; 20],
        header: "Test binary".to_string(),
        description: String::new(),
        decision_type: DecisionType::Binary,
        option_0_label: None,
        option_1_label: None,
        tags: Vec::new(),
    }
}

fn make_scaled_decision(min: i64, max: i64) -> Decision {
    Decision {
        market_maker_pubkey_hash: [0u8; 20],
        header: "Test scaled".to_string(),
        description: String::new(),
        decision_type: DecisionType::Scaled { min, max },
        option_0_label: None,
        option_1_label: None,
        tags: Vec::new(),
    }
}

#[test]
fn test_binary_decision_no_normalization() {
    let decision = make_binary_decision();

    // Binary decisions pass through unchanged
    assert_eq!(decision.normalize_value(0.0), 0.0);
    assert_eq!(decision.normalize_value(1.0), 1.0);
    assert_eq!(decision.normalize_value(0.5), 0.5);

    assert_eq!(decision.denormalize_value(0.0), 0.0);
    assert_eq!(decision.denormalize_value(1.0), 1.0);
    assert_eq!(decision.denormalize_value(0.5), 0.5);
}

#[test]
fn test_scaled_decision_normalization() {
    // Electoral votes: 0-538
    let decision = make_scaled_decision(0, 538);

    // 0 -> 0.0, 538 -> 1.0, 269 -> 0.5
    assert!((decision.normalize_value(0.0) - 0.0).abs() < 1e-10);
    assert!((decision.normalize_value(538.0) - 1.0).abs() < 1e-10);
    assert!((decision.normalize_value(269.0) - 0.5).abs() < 1e-10);
    assert!((decision.normalize_value(270.0) - (270.0 / 538.0)).abs() < 1e-10);
}

#[test]
fn test_scaled_decision_denormalization() {
    // Electoral votes: 0-538
    let decision = make_scaled_decision(0, 538);

    // 0.0 -> 0, 1.0 -> 538, 0.5 -> 269
    assert!((decision.denormalize_value(0.0) - 0.0).abs() < 1e-10);
    assert!((decision.denormalize_value(1.0) - 538.0).abs() < 1e-10);
    assert!((decision.denormalize_value(0.5) - 269.0).abs() < 1e-10);
}

#[test]
fn test_scaled_decision_with_offset() {
    // Score: 50 to 150 (range of 100)
    let decision = make_scaled_decision(50, 150);

    // 50 -> 0.0, 150 -> 1.0, 100 -> 0.5
    assert!((decision.normalize_value(50.0) - 0.0).abs() < 1e-10);
    assert!((decision.normalize_value(150.0) - 1.0).abs() < 1e-10);
    assert!((decision.normalize_value(100.0) - 0.5).abs() < 1e-10);

    // Inverse
    assert!((decision.denormalize_value(0.0) - 50.0).abs() < 1e-10);
    assert!((decision.denormalize_value(1.0) - 150.0).abs() < 1e-10);
    assert!((decision.denormalize_value(0.5) - 100.0).abs() < 1e-10);
}

#[test]
fn test_roundtrip_normalization() {
    let decision = make_scaled_decision(100, 200);

    for value in [100.0, 125.0, 150.0, 175.0, 200.0] {
        let normalized = decision.normalize_value(value);
        let denormalized = decision.denormalize_value(normalized);
        assert!(
            (denormalized - value).abs() < 1e-10,
            "Roundtrip failed for {value}"
        );
    }
}

#[test]
fn test_validate_and_normalize_success() {
    let decision = make_scaled_decision(0, 100);

    // Valid values
    assert!(decision.validate_and_normalize(0.0).is_ok());
    assert!(decision.validate_and_normalize(50.0).is_ok());
    assert!(decision.validate_and_normalize(100.0).is_ok());

    // Check the normalized value
    let normalized = decision.validate_and_normalize(50.0).unwrap();
    assert!((normalized - 0.5).abs() < 1e-10);
}

#[test]
fn test_validate_and_normalize_out_of_bounds() {
    let decision = make_scaled_decision(0, 100);

    // Out of bounds
    assert!(decision.validate_and_normalize(-1.0).is_err());
    assert!(decision.validate_and_normalize(101.0).is_err());
}

#[test]
fn test_validate_and_normalize_nan_abstain() {
    let decision = make_scaled_decision(0, 100);
    // NaN is valid - it means "abstain" in voting
    let result = decision.validate_and_normalize(f64::NAN);
    assert!(result.is_ok());
    assert!(result.unwrap().is_nan());
}

#[test]
fn test_get_display_range_scaled() {
    let decision = make_scaled_decision(0, 538);
    let (min, max) = decision.get_display_range();
    assert!((min - 0.0).abs() < 1e-10);
    assert!((max - 538.0).abs() < 1e-10);
}

#[test]
fn test_get_display_range_binary() {
    let decision = make_binary_decision();
    let (min, max) = decision.get_display_range();
    assert!((min - 0.0).abs() < 1e-10);
    assert!((max - 1.0).abs() < 1e-10);
}

#[test]
fn test_nan_handling() {
    let decision = make_scaled_decision(0, 100);

    // NaN input returns NaN
    assert!(decision.normalize_value(f64::NAN).is_nan());
    assert!(decision.denormalize_value(f64::NAN).is_nan());
}

#[test]
fn test_zero_range() {
    let decision = make_scaled_decision(50, 50);
    assert!((decision.normalize_value(50.0) - 0.5).abs() < 1e-10);
    assert!((decision.denormalize_value(0.5) - 50.0).abs() < 1e-10);
}

#[test]
fn test_decision_id_standard_roundtrip() {
    let id = DecisionId::new(true, 5, 42).unwrap();
    assert!(id.is_standard());
    assert_eq!(id.period_index(), 5);
    assert_eq!(id.decision_index(), 42);

    let id2 = DecisionId::from_bytes(id.as_bytes()).unwrap();
    assert!(id2.is_standard());
    assert_eq!(id2.period_index(), 5);
    assert_eq!(id2.decision_index(), 42);
}

#[test]
fn test_decision_id_nonstandard_roundtrip() {
    let id = DecisionId::new(false, 10, 1000).unwrap();
    assert!(!id.is_standard());
    assert_eq!(id.period_index(), 10);
    assert_eq!(id.decision_index(), 1000);

    let id2 = DecisionId::from_bytes(id.as_bytes()).unwrap();
    assert!(!id2.is_standard());
    assert_eq!(id2.period_index(), 10);
    assert_eq!(id2.decision_index(), 1000);
}

#[test]
fn test_decision_id_max_period() {
    let id = DecisionId::new(true, 127, 0).unwrap();
    assert_eq!(id.period_index(), 127);
    assert!(DecisionId::new(true, 128, 0).is_err());
}

#[test]
fn test_decision_id_max_index() {
    let id = DecisionId::new(true, 0, 65535).unwrap();
    assert_eq!(id.decision_index(), 65535);
    assert!(DecisionId::new(true, 0, 65536).is_err());
}

#[test]
fn test_decision_id_boundary_values() {
    let id = DecisionId::new(false, 127, 65535).unwrap();
    assert!(!id.is_standard());
    assert_eq!(id.period_index(), 127);
    assert_eq!(id.decision_index(), 65535);
}

#[test]
fn test_decision_id_hex_roundtrip() {
    let id = DecisionId::new(true, 3, 256).unwrap();
    let hex = id.to_hex();
    let id2 = DecisionId::from_hex(&hex).unwrap();
    assert_eq!(id.as_bytes(), id2.as_bytes());
    assert!(id2.is_standard());
    assert_eq!(id2.period_index(), 3);
    assert_eq!(id2.decision_index(), 256);
}

#[test]
fn test_decision_id_standard_bit_encoding() {
    let std_id = DecisionId::new(true, 1, 0).unwrap();
    let nonstd_id = DecisionId::new(false, 1, 0).unwrap();
    assert_ne!(std_id.as_bytes(), nonstd_id.as_bytes());
    assert!(std_id.is_standard());
    assert!(!nonstd_id.is_standard());
    assert_eq!(std_id.period_index(), nonstd_id.period_index());
    assert_eq!(std_id.decision_index(), nonstd_id.decision_index());
}

#[test]
fn test_listing_fee_free_tier() {
    assert_eq!(calculate_listing_fee(0, 500), Some(0));
    assert_eq!(calculate_listing_fee(50, 500), Some(0));
    assert_eq!(calculate_listing_fee(99, 500), Some(0));
}

#[test]
fn test_listing_fee_escalating_tier() {
    assert_eq!(calculate_listing_fee(100, 500), Some(BASE_LISTING_FEE));
    assert_eq!(calculate_listing_fee(101, 500), Some(2 * BASE_LISTING_FEE));
    assert_eq!(
        calculate_listing_fee(200, 500),
        Some(101 * BASE_LISTING_FEE)
    );
    assert_eq!(
        calculate_listing_fee(499, 500),
        Some(400 * BASE_LISTING_FEE)
    );
}

#[test]
fn test_listing_fee_period_full() {
    assert_eq!(calculate_listing_fee(500, 500), None);
    assert_eq!(calculate_listing_fee(501, 500), None);
}

#[test]
fn test_listing_fee_smaller_period() {
    assert_eq!(calculate_listing_fee(0, 250), Some(0));
    assert_eq!(calculate_listing_fee(49, 250), Some(0));
    assert_eq!(calculate_listing_fee(50, 250), Some(BASE_LISTING_FEE));
    assert_eq!(
        calculate_listing_fee(249, 250),
        Some(200 * BASE_LISTING_FEE)
    );
    assert_eq!(calculate_listing_fee(250, 250), None);
}

#[test]
fn test_listing_fee_zero_available() {
    assert_eq!(calculate_listing_fee(0, 0), None);
}

#[test]
fn test_listing_fee_batch_category() {
    let available = 500u64;
    let base_count = 100u64;
    let mut total_fee = 0u64;
    for i in 0..3u64 {
        let fee = calculate_listing_fee(base_count + i, available).unwrap();
        total_fee += fee;
    }
    let expected =
        BASE_LISTING_FEE + 2 * BASE_LISTING_FEE + 3 * BASE_LISTING_FEE;
    assert_eq!(total_fee, expected);
}
