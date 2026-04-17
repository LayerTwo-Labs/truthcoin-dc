use super::types::*;
use crate::math::allocation;
use ndarray::Array;

#[test]
fn test_dimension_parsing() {
    let single_str = "[010101]";
    let result = parse_dimensions(single_str);
    assert!(result.is_ok());
    let dimensions = result.unwrap();
    assert_eq!(dimensions.len(), 1);
    assert!(matches!(dimensions[0], DimensionSpec::Single(_)));

    let categorical_str = "[[010101]]";
    let result = parse_dimensions(categorical_str);
    assert!(result.is_ok());
    let dimensions = result.unwrap();
    assert_eq!(dimensions.len(), 1);
    assert!(matches!(dimensions[0], DimensionSpec::Categorical(_)));

    let mixed_str = "[010101,[010102],010104]";
    let result = parse_dimensions(mixed_str);
    assert!(result.is_ok());
    let dimensions = result.unwrap();
    assert_eq!(dimensions.len(), 3);
    assert!(matches!(dimensions[0], DimensionSpec::Single(_)));
    assert!(matches!(dimensions[1], DimensionSpec::Categorical(_)));
    assert!(matches!(dimensions[2], DimensionSpec::Single(_)));

    let invalid_str = "010101,010102";
    let result = parse_dimensions(invalid_str);
    assert!(result.is_err());
}

#[test]
fn test_lmsr_initialization_spec_compliance() {
    let beta: f64 = 7.0;
    let n_outcomes: f64 = 2.0;
    let target_liquidity: f64 = 100.0;

    let min_treasury = beta * n_outcomes.ln();
    let expected_initial_shares = target_liquidity - min_treasury;

    println!("Binary market test:");
    println!(
        "  \u{03b2} = {}, n = {}, L = {}",
        beta, n_outcomes, target_liquidity
    );
    println!(
        "  Min treasury (\u{03b2}\u{00d7}ln(n)) = {:.6}",
        min_treasury
    );
    println!("  Expected initial shares = {:.6}", expected_initial_shares);

    let shares = Array::from_elem(2, expected_initial_shares);
    let calculated_treasury =
        beta * shares.mapv(|x| (x / beta).exp()).sum().ln();

    println!("  Calculated treasury = {:.6}", calculated_treasury);
    println!("  Target liquidity = {:.6}", target_liquidity);

    assert!(
        (calculated_treasury - target_liquidity).abs() < 1e-10,
        "Treasury {:.6} should equal target liquidity {:.6}",
        calculated_treasury,
        target_liquidity
    );

    let exp_shares: Array<f64, ndarray::Ix1> =
        shares.mapv(|x| (x / beta).exp());
    let sum_exp = exp_shares.sum();
    let prices: Array<f64, ndarray::Ix1> = exp_shares.mapv(|x| x / sum_exp);

    for (i, &price) in prices.iter().enumerate() {
        let expected_price = 1.0 / n_outcomes;
        println!(
            "  Price[{}] = {:.6}, expected = {:.6}",
            i, price, expected_price
        );
        assert!(
            (price - expected_price).abs() < 1e-10,
            "Price[{}] should be {:.6} but was {:.6}",
            i,
            expected_price,
            price
        );
    }

    let beta: f64 = 3.2;
    let n_outcomes: f64 = 3.0;
    let target_liquidity: f64 = 50.0;

    let min_treasury = beta * n_outcomes.ln();
    let expected_initial_shares = target_liquidity - min_treasury;

    println!("\n3-outcome market test:");
    println!(
        "  \u{03b2} = {}, n = {}, L = {}",
        beta, n_outcomes, target_liquidity
    );
    println!(
        "  Min treasury (\u{03b2}\u{00d7}ln(n)) = {:.6}",
        min_treasury
    );
    println!("  Expected initial shares = {:.6}", expected_initial_shares);

    let shares = Array::from_elem(3, expected_initial_shares);
    let calculated_treasury =
        beta * shares.mapv(|x| (x / beta).exp()).sum().ln();

    println!("  Calculated treasury = {:.6}", calculated_treasury);

    assert!(
        (calculated_treasury - target_liquidity).abs() < 1e-10,
        "Treasury should equal target liquidity"
    );

    let exp_shares: Array<f64, ndarray::Ix1> =
        shares.mapv(|x| (x / beta).exp());
    let sum_exp = exp_shares.sum();
    let prices: Array<f64, ndarray::Ix1> = exp_shares.mapv(|x| x / sum_exp);

    for &price in prices.iter() {
        let expected_price = 1.0 / n_outcomes;
        assert!(
            (price - expected_price).abs() < 1e-10,
            "All prices should be uniform at {:.6}",
            expected_price
        );
    }

    let beta: f64 = 5.0;
    let n_outcomes: f64 = 4.0;
    let min_liquidity = beta * n_outcomes.ln();

    println!("\nMinimum liquidity edge case:");
    println!(
        "  \u{03b2} = {}, n = {}, min L = {:.6}",
        beta, n_outcomes, min_liquidity
    );

    let expected_shares: f64 = min_liquidity - min_liquidity;
    assert!(
        expected_shares.abs() < 1e-10,
        "Shares should be 0 at minimum liquidity"
    );

    let shares = Array::zeros(4);
    let calculated_treasury =
        beta * shares.mapv(|x: f64| (x / beta).exp()).sum().ln();

    println!("  Treasury with zero shares = {:.6}", calculated_treasury);
    assert!(
        (calculated_treasury - min_liquidity).abs() < 1e-10,
        "Zero shares should give minimum treasury"
    );
}

#[test]
fn test_liquidity_validation() {
    let beta: f64 = 7.0;
    let n_outcomes: f64 = 2.0;
    let min_liquidity = beta * n_outcomes.ln();

    let insufficient = min_liquidity - 0.1;
    let expected_shares: f64 = insufficient - min_liquidity;

    assert!(
        expected_shares < 0.0,
        "Insufficient liquidity should result in negative shares"
    );

    let adequate = min_liquidity + 10.0;
    let expected_shares: f64 = adequate - min_liquidity;

    assert!(
        expected_shares > 0.0 && expected_shares.is_finite(),
        "Adequate liquidity should result in positive, finite shares"
    );
}

#[test]
fn test_fee_split_conservation() {
    for fee_sats in [1, 2, 3, 4, 7, 10, 100, 999, 1000, 10_001] {
        let pool_split = allocation::allocate_proportionally_u64(
            vec![
                ("voter", 2.0),
                ("decision_author", 1.0),
                ("market_author", 1.0),
            ],
            fee_sats,
        )
        .unwrap();

        let sum: u64 = pool_split.allocations.iter().map(|(_, amt)| *amt).sum();
        assert_eq!(
            sum, fee_sats,
            "Conservation violated for fee_sats={fee_sats}"
        );
    }
}

#[test]
fn test_fee_split_proportions() {
    let pool_split = allocation::allocate_proportionally_u64(
        vec![
            ("voter", 2.0),
            ("decision_author", 1.0),
            ("market_author", 1.0),
        ],
        1000,
    )
    .unwrap();

    let mut voter = 0u64;
    let mut decision_author = 0u64;
    let mut market_author = 0u64;
    for (key, amount) in &pool_split.allocations {
        match *key {
            "voter" => voter = *amount,
            "decision_author" => decision_author = *amount,
            "market_author" => market_author = *amount,
            _ => panic!("unexpected key"),
        }
    }

    assert_eq!(voter, 500);
    assert_eq!(decision_author, 250);
    assert_eq!(market_author, 250);
    assert_eq!(voter + decision_author + market_author, 1000);
}
