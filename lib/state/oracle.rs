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
