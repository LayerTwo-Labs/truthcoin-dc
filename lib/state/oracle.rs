//! Commit the previous completed market state, never mempool projections.

use sneed::RoTxn;

use super::{Error, Market, MarketState, State};
use crate::{
    math::trading,
    types::{
        BlockHash, Hash, hashes,
        market::MarketId,
        oracle::{
            Commitment, MAX_MARKETS, MarketPrice, PRICE_SCALE, PriceProof, Tree,
        },
    },
};

pub struct Snapshot {
    pub commitment: Commitment,
    markets: Vec<MarketPrice>,
    hashes: Vec<Hash>,
}

impl Snapshot {
    pub fn proof(&self, id: MarketId) -> Option<PriceProof> {
        let index = self
            .markets
            .binary_search_by_key(&id, |m| m.market_id)
            .ok()?;
        let proof = Tree::build_merkle_proof(&self.hashes, &[index as u32])?;
        Some(PriceProof {
            commitment: self.commitment.clone(),
            market: self.markets[index].clone(),
            index: proof.indices()[0],
            siblings: proof.lemmas().to_vec(),
        })
    }
}

fn price_units(price: f64) -> Result<u64, Error> {
    if !price.is_finite() || !(0.0..=1.0).contains(&price) {
        return Err(Error::Oracle("invalid market price"));
    }
    Ok((price * PRICE_SCALE as f64).floor() as u64)
}

fn market_price(
    market: &Market,
    decision_hashes: Vec<Hash>,
) -> Result<MarketPrice, Error> {
    let prices = if market.state() == MarketState::Settled {
        market.final_prices().clone()
    } else {
        trading::calculate_prices(
            market.shares(),
            trading::derive_beta_from_liquidity(
                market.liquidity_base_sats,
                market.shares().len(),
            ),
        )
        .map_err(|_| Error::Oracle("cannot price market"))?
    };
    let valid_outcomes = market.get_valid_state_combos();
    if prices.len() != valid_outcomes.len() || prices.is_empty() {
        return Err(Error::Oracle("market outcome count mismatch"));
    }
    // Explicit u32 indices avoid architecture-sized encoding. Decision hashes
    // bind the definitions as well as their short native identifiers.
    let outcomes: Vec<Vec<u32>> = valid_outcomes
        .iter()
        .map(|(_, row)| {
            row.iter()
                .map(|&i| {
                    u32::try_from(i)
                        .map_err(|_| Error::Oracle("outcome index overflow"))
                })
                .collect()
        })
        .collect::<Result<_, _>>()?;
    let terms = borsh::to_vec(&(
        &market.title,
        &market.description,
        market.creator_address,
        &market.dimension_specs,
        &market.decision_ids,
        &outcomes,
        market.created_at_height,
        market.expires_at_height,
        decision_hashes,
    ))
    .map_err(Error::BorshSerialize)?;
    Ok(MarketPrice {
        market_id: market.id,
        terms_hash: hashes::hash(&(*b"TCORACLE/terms/v1", terms)),
        status: market.state() as u8,
        updated_height: market.last_updated_height,
        liquidity_base_sats: market.liquidity_base_sats,
        prices: prices
            .iter()
            .copied()
            .map(price_units)
            .collect::<Result<_, _>>()?,
    })
}

impl State {
    pub fn oracle_snapshot(&self, txn: &RoTxn) -> Result<Snapshot, Error> {
        if !self.oracle_test {
            return Err(Error::Oracle("requires oracletest network"));
        }
        let mut markets = self.markets().get_all_markets(txn)?;
        if markets.len() > MAX_MARKETS {
            return Err(Error::Oracle("too many markets"));
        }
        markets.sort_unstable_by_key(|m| m.id);
        let markets = markets
            .iter()
            .map(|market| {
                let decisions = market
                    .decision_ids
                    .iter()
                    .map(|id| {
                        let entry =
                            self.decisions().get_decision_entry(txn, *id)?;
                        let encoded = borsh::to_vec(&entry)
                            .map_err(Error::BorshSerialize)?;
                        Ok(hashes::hash(&(*b"TCORACLE/decision/v1", encoded)))
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                market_price(market, decisions)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let hashes: Vec<_> = markets.iter().map(MarketPrice::hash).collect();
        Ok(Snapshot {
            commitment: Commitment {
                previous_height: self.try_get_height(txn)?,
                previous_hash: self.try_get_tip(txn)?,
                market_count: markets.len() as u32,
                root: Tree::build_merkle_root(&hashes),
            },
            markets,
            hashes,
        })
    }

    pub fn oracle_memo(
        &self,
        txn: &RoTxn,
        expected_tip: Option<BlockHash>,
    ) -> Result<Vec<u8>, Error> {
        if !self.oracle_test {
            return Ok(Vec::new());
        }
        let snapshot = self.oracle_snapshot(txn)?;
        if snapshot.commitment.previous_hash != expected_tip {
            return Err(Error::Oracle(
                "template must extend the current executed tip",
            ));
        }
        Ok(snapshot.commitment.memo())
    }

    pub(super) fn check_oracle_memo(
        &self,
        txn: &RoTxn,
        memo: &[u8],
    ) -> Result<(), Error> {
        if self.oracle_test {
            let prefix = borsh::to_vec(&(
                crate::types::oracle::MAGIC,
                self.try_get_height(txn)?,
                self.try_get_tip(txn)?,
            ))
            .map_err(Error::BorshSerialize)?;
            // Reject malformed/stale requests before scanning the market state.
            if memo.len() != prefix.len() + 36 || !memo.starts_with(&prefix) {
                return Err(Error::Oracle(
                    "missing or stale executed-state commitment",
                ));
            }
        }
        if self.oracle_test
            && memo != self.oracle_snapshot(txn)?.commitment.memo()
        {
            return Err(Error::Oracle(
                "missing or incorrect executed-state commitment",
            ));
        }
        Ok(())
    }

    pub(super) fn check_oracle_state(&self, txn: &RoTxn) -> Result<(), Error> {
        if self.oracle_test {
            let _snapshot = self.oracle_snapshot(txn)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        archive::Archive,
        state::{
            self, MarketBuilder,
            decisions::{Decision, DecisionId, DecisionType},
            markets::DimensionSpec,
        },
        types::{
            Address, Body, Coinbase, FilledTransaction, Header, Output,
            OutputContent, authorization::BatchVerificationContext,
        },
    };
    use bitcoin::hashes::Hash as _;
    use std::collections::HashMap;

    fn market(id: u8) -> Market {
        let decision_id = DecisionId::new(true, 0, 0).unwrap();
        let decision = Decision::new(
            Address::ALL_ZEROS.0,
            DecisionType::Binary,
            "Question".into(),
            "Definition".into(),
            None,
            None,
            vec![],
        )
        .unwrap();
        let mut market =
            MarketBuilder::new(format!("Market {id}"), Address::ALL_ZEROS)
                .with_dimensions(vec![DimensionSpec::Single(decision_id)])
                .build(0, None, &HashMap::from([(decision_id, decision)]))
                .unwrap();
        market.liquidity_base_sats = 10_000;
        market.id = MarketId::new([id; 6]);
        market
    }

    #[test]
    fn oracle_all_markets_and_statuses_are_committed() -> anyhow::Result<()> {
        let (_dir, env) = state::test::temp_env("oracle-all")?;
        let state = State::new_with_oracle_test(&env, None, true)?;
        let mut txn = env.write_txn()?;
        for id in (0..65).rev() {
            let mut m = market(id);
            m.market_state = match id % 4 {
                0 => MarketState::Trading,
                1 => MarketState::Cancelled,
                2 => MarketState::Invalid,
                _ => MarketState::Settled,
            };
            m.final_prices = ndarray::array![0.25, 0.75];
            state.markets().add_market(&mut txn, &m)?;
        }
        let snapshot = state.oracle_snapshot(&txn)?;
        assert_eq!(snapshot.commitment.market_count, 65);
        state.check_oracle_memo(&txn, &snapshot.commitment.memo())?;
        let mut omitted = snapshot.commitment.clone();
        omitted.market_count -= 1;
        assert!(state.check_oracle_memo(&txn, &omitted.memo()).is_err());
        let mut fabricated = snapshot.commitment.clone();
        fabricated.root[0] ^= 1;
        assert!(state.check_oracle_memo(&txn, &fabricated.memo()).is_err());
        for id in 0..65 {
            let proof = snapshot.proof(MarketId::new([id; 6])).unwrap();
            assert!(proof.verify());
            assert_eq!(
                proof.market.prices,
                if id % 4 == 3 {
                    vec![PRICE_SCALE / 4, PRICE_SCALE * 3 / 4]
                } else {
                    vec![PRICE_SCALE / 2; 2]
                }
            );
            let mut bad = proof.clone();
            bad.market.prices[0] += 1;
            assert!(!bad.verify());
            bad = proof.clone();
            bad.market.status ^= 1;
            assert!(!bad.verify());
            bad = proof.clone();
            bad.market.terms_hash[0] ^= 1;
            assert!(!bad.verify());
            bad = proof;
            bad.index = u32::MAX;
            assert!(!bad.verify());
        }
        assert!(snapshot.proof(MarketId::new([255; 6])).is_none());
        Ok(())
    }

    #[test]
    fn oracle_snapshot_tracks_execution_not_mempool_and_rolls_back()
    -> anyhow::Result<()> {
        let (_dir, env) = state::test::temp_env("oracle-execution")?;
        let state = State::new_with_oracle_test(&env, None, true)?;
        let mut txn = env.write_txn()?;
        let original = market(1);
        state.markets().add_market(&mut txn, &original)?;
        let before = state.oracle_snapshot(&txn)?.commitment.memo();
        state.markets().put_mempool_shares(
            &mut txn,
            &original.id,
            &ndarray::array![100_000, 0],
        )?;
        assert_eq!(state.oracle_snapshot(&txn)?.commitment.memo(), before);
        let mut changed = original.clone();
        changed.shares[0] += 10_000;
        state.markets().update_market(&mut txn, &changed)?;
        assert_ne!(state.oracle_snapshot(&txn)?.commitment.memo(), before);
        state.markets().restore_market(&mut txn, &original)?;
        assert_eq!(state.oracle_snapshot(&txn)?.commitment.memo(), before);
        changed = original;
        changed.description.push('!');
        state.markets().restore_market(&mut txn, &changed)?;
        assert_ne!(state.oracle_snapshot(&txn)?.commitment.memo(), before);
        Ok(())
    }

    fn empty_block(
        state: &State,
        txn: &RoTxn,
    ) -> anyhow::Result<(Header, Body)> {
        let previous = state.try_get_tip(txn)?;
        let coinbase = Coinbase {
            memo: state.oracle_memo(txn, previous)?,
            outputs: if previous.is_none() {
                vec![Output {
                    address: Address::ALL_ZEROS,
                    content: OutputContent::Value(bitcoin::Amount::ZERO),
                }]
            } else {
                vec![]
            }
            .into(),
        };
        let txs: [FilledTransaction; 0] = [];
        let root = Body::compute_merkle_root(&coinbase, &txs)?;
        let parent = bitcoin::BlockHash::all_zeros();
        let mut accumulator = state.get_accumulator(txn)?;
        Body::modify_memforest(
            Coinbase::compute_txid(&root, &parent, previous.as_ref()),
            coinbase.outputs.as_slice(),
            &txs,
            &mut accumulator.0,
        )?;
        Ok((
            Header {
                prev_side_hash: previous,
                prev_main_hash: parent,
                merkle_root: root,
                roots: accumulator.get_roots(),
            },
            Body::new(vec![], coinbase),
        ))
    }

    #[test]
    fn oracle_genesis_height_one_all_admission_paths_and_disconnect()
    -> anyhow::Result<()> {
        let (_dir, env) = state::test::temp_env("oracle-admission")?;
        let state = State::new_with_oracle_test(&env, None, true)?;
        let archive = Archive::new(&env)?;
        let auth = BatchVerificationContext::new(&mut rand::rng());
        let mut txn = env.write_txn()?;
        let genesis_memo = state.oracle_snapshot(&txn)?.commitment.memo();
        for _ in 0..2 {
            let (header, body) = empty_block(&state, &txn)?;
            let mut bad = body.clone();
            bad.coinbase.memo.clear();
            assert!(matches!(
                state.prevalidate_block(&txn, &auth, &header, &bad, &archive),
                Err(Error::Oracle(_))
            ));
            assert!(matches!(
                state.validate_block(&txn, &auth, &header, &bad, &archive),
                Err(Error::Oracle(_))
            ));
            assert!(matches!(
                state.connect_block(&mut txn, &header, &bad, 0),
                Err(Error::Oracle(_))
            ));
            let pre = state
                .prevalidate_block(&txn, &auth, &header, &body, &archive)?;
            assert!(matches!(
                state.connect_prevalidated_block(
                    &mut txn, &header, &bad, pre, 0
                ),
                Err(Error::Oracle(_))
            ));
            state.apply_block(&mut txn, &auth, &header, &body, &archive, 0)?;
            assert!(
                state.check_oracle_memo(&txn, &body.coinbase.memo).is_err()
            );
            if state.try_get_height(&txn)? == Some(1) {
                state.disconnect_tip(&mut txn, &header, &body)?;
                assert_eq!(
                    state.oracle_snapshot(&txn)?.commitment.memo(),
                    body.coinbase.memo
                );
                state.connect_block(&mut txn, &header, &body, 0)?;
            }
        }
        assert_ne!(
            state.oracle_snapshot(&txn)?.commitment.memo(),
            genesis_memo
        );
        assert!(state.oracle_memo(&txn, None).is_err());
        Ok(())
    }

    #[test]
    fn oracle_profile_is_persistent_and_legacy_rules_unchanged()
    -> anyhow::Result<()> {
        let (_dir, env) = state::test::temp_env("oracle-profile")?;
        State::new_with_oracle_test(&env, None, true)?;
        assert!(State::new(&env, None).is_err());
        State::new_with_oracle_test(&env, None, true)?;
        let (_dir2, env2, legacy) = state::test::fresh_state("oracle-legacy")?;
        assert!(State::new_with_oracle_test(&env2, None, true).is_err());
        let txn = env2.read_txn()?;
        legacy.check_oracle_memo(&txn, b"arbitrary legacy memo")?;
        assert_eq!(legacy.oracle_memo(&txn, None)?, Vec::<u8>::new());
        assert!(legacy.oracle_snapshot(&txn).is_err());
        Ok(())
    }

    #[test]
    fn oracle_prices_reject_nan_and_out_of_range() {
        for bad in [f64::NAN, f64::INFINITY, -0.001, 1.001] {
            assert!(price_units(bad).is_err());
        }
        assert_eq!(price_units(1.0).unwrap(), PRICE_SCALE);
        assert_eq!(price_units(0.0).unwrap(), 0);
        assert_eq!(price_units(0.1234567890129).unwrap(), 123456789012);
    }
}
