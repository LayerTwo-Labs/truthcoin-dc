use fallible_iterator::FallibleIterator;
use ndarray::Array1;
use sneed::{DbError, RoTxn};

use crate::{
    archive::{self, Archive},
    math::{lmsr::LmsrError, trading},
    state::{self, MarketId, State},
    types::{BlockHash, Body, TxData},
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("archive error")]
    Archive(#[from] archive::Error),
    #[error("database error")]
    Db(#[from] DbError),
    #[error(
        "liquidity base of market {market_id} underflows before block {block_hash}"
    )]
    LiquidityUnderflow {
        market_id: MarketId,
        block_hash: BlockHash,
    },
    #[error("market {market_id} has no creation block in the active chain")]
    NoCreationBlock { market_id: MarketId },
    #[error(
        "block {block_hash} trades outcome {outcome_index}, which market {market_id} does not have"
    )]
    OutcomeIndex {
        market_id: MarketId,
        block_hash: BlockHash,
        outcome_index: u32,
    },
    #[error("failed to calculate market prices")]
    Prices(#[from] LmsrError),
    #[error(
        "shares of outcome {outcome_index} in market {market_id} overflow before block {block_hash}"
    )]
    ShareOverflow {
        market_id: MarketId,
        block_hash: BlockHash,
        outcome_index: u32,
    },
    #[error("state error")]
    State(#[source] Box<state::Error>),
}

impl From<state::Error> for Error {
    fn from(err: state::Error) -> Self {
        Self::State(Box::new(err))
    }
}

/// Outcome prices of a market after a block of the active chain.
#[derive(Clone, Debug, PartialEq)]
pub struct MarketPricePoint {
    pub height: u32,
    pub block_hash: BlockHash,
    /// Unix time of the mainchain block that holds the BMM commitment.
    pub mainchain_timestamp: u64,
    /// Price of each tradeable outcome, by outcome index.
    pub prices: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum MarketChange {
    Trade { outcome_index: u32, shares: i64 },
    AmplifyBeta { amount: u64 },
}

struct BlockMarketChanges {
    height: u32,
    block_hash: BlockHash,
    mainchain_timestamp: u64,
    changes: Vec<MarketChange>,
}

struct MarketSnapshot {
    shares: Array1<i64>,
    liquidity_base_sats: u64,
}

impl MarketSnapshot {
    fn prices(&self) -> Result<Vec<f64>, LmsrError> {
        let beta = trading::derive_beta_from_liquidity(
            self.liquidity_base_sats,
            self.shares.len(),
        );
        Ok(trading::calculate_prices(&self.shares, beta)?.to_vec())
    }

    fn revert_block(
        &mut self,
        market_id: MarketId,
        block_hash: BlockHash,
        changes: &[MarketChange],
    ) -> Result<(), Error> {
        for change in changes {
            match *change {
                MarketChange::Trade {
                    outcome_index,
                    shares,
                } => {
                    let outcome_shares = self
                        .shares
                        .get_mut(outcome_index as usize)
                        .ok_or(Error::OutcomeIndex {
                            market_id,
                            block_hash,
                            outcome_index,
                        })?;
                    *outcome_shares = outcome_shares
                        .checked_sub(shares)
                        .ok_or(Error::ShareOverflow {
                            market_id,
                            block_hash,
                            outcome_index,
                        })?;
                }
                MarketChange::AmplifyBeta { amount } => {
                    self.liquidity_base_sats = self
                        .liquidity_base_sats
                        .checked_sub(amount)
                        .ok_or(Error::LiquidityUnderflow {
                            market_id,
                            block_hash,
                        })?;
                }
            }
        }
        Ok(())
    }
}

/// Market changes that a block applied, in block order.
fn market_changes(
    body: &Body,
    skipped_tx_indices: &[u32],
    market_id: &MarketId,
) -> Vec<MarketChange> {
    body.transactions
        .iter()
        .enumerate()
        .filter(|(idx, _)| !skipped_tx_indices.contains(&(*idx as u32)))
        .filter_map(|(_, tx)| match &tx.data {
            Some(TxData::Trade {
                market_id: trade_market_id,
                outcome_index,
                shares,
                ..
            }) if trade_market_id == market_id => Some(MarketChange::Trade {
                outcome_index: *outcome_index,
                shares: *shares,
            }),
            Some(TxData::AmplifyBeta {
                market_id: amplify_market_id,
                amount,
                ..
            }) if amplify_market_id == market_id => {
                Some(MarketChange::AmplifyBeta { amount: *amount })
            }
            _ => None,
        })
        .collect()
}

/// Walk blocks from the tip down to the creation block. Each block reverts
/// its own changes from `snapshot`, so the last point equals the tip state.
fn rebuild_price_history<Blocks>(
    market_id: MarketId,
    mut snapshot: MarketSnapshot,
    created_at_height: u32,
    mut blocks_newest_first: Blocks,
) -> Result<Vec<MarketPricePoint>, Error>
where
    Blocks: FallibleIterator<Item = BlockMarketChanges, Error = Error>,
{
    let mut points = Vec::new();
    while let Some(block) = blocks_newest_first.next()? {
        let is_creation_block = block.height == created_at_height;
        if is_creation_block || !block.changes.is_empty() {
            points.push(MarketPricePoint {
                height: block.height,
                block_hash: block.block_hash,
                mainchain_timestamp: block.mainchain_timestamp,
                prices: snapshot.prices()?,
            });
        }
        if is_creation_block {
            points.reverse();
            return Ok(points);
        }
        snapshot.revert_block(market_id, block.block_hash, &block.changes)?;
    }
    Err(Error::NoCreationBlock { market_id })
}

/// Rebuild the price history of a market from the active chain, one point
/// for the creation block and one for each block that changed its prices.
/// Returns [`None`] if the market does not exist.
pub fn try_get_market_price_history(
    state: &State,
    archive: &Archive,
    rotxn: &RoTxn,
    market_id: &MarketId,
) -> Result<Option<Vec<MarketPricePoint>>, Error> {
    let Some(market) = state.markets().get_market(rotxn, market_id)? else {
        return Ok(None);
    };
    let Some(tip) = state.try_get_tip(rotxn)? else {
        return Err(Error::NoCreationBlock {
            market_id: *market_id,
        });
    };
    let snapshot = MarketSnapshot {
        shares: market.shares().clone(),
        liquidity_base_sats: market.liquidity_base_sats,
    };
    let blocks =
        archive
            .ancestors(rotxn, tip)
            .map_err(Error::from)
            .map(|block_hash| {
                let height = archive.get_height(rotxn, block_hash)?;
                let body = archive.get_body(rotxn, block_hash)?;
                let skipped_tx_indices = state
                    .skipped_tx_indices_undo
                    .try_get(rotxn, &height)
                    .map_err(DbError::from)?
                    .unwrap_or_default();
                let bmm_main_hash =
                    archive.get_best_main_verification(rotxn, block_hash)?;
                let mainchain_timestamp = archive
                    .get_main_header_info(rotxn, &bmm_main_hash)?
                    .timestamp;
                Ok(BlockMarketChanges {
                    height,
                    block_hash,
                    mainchain_timestamp,
                    changes: market_changes(
                        &body,
                        &skipped_tx_indices,
                        market_id,
                    ),
                })
            });
    rebuild_price_history(
        *market_id,
        snapshot,
        market.created_at_height,
        blocks,
    )
    .map(Some)
}

#[cfg(test)]
mod tests {
    use fallible_iterator::IteratorExt as _;
    use ndarray::array;

    use crate::types::{Address, Transaction, hashes};

    use super::*;

    const MARKET_ID: MarketId = MarketId([1; 6]);
    const OTHER_MARKET_ID: MarketId = MarketId([2; 6]);

    fn tx(data: TxData) -> Transaction {
        Transaction {
            data: Some(data),
            ..Default::default()
        }
    }

    fn trade(market_id: MarketId, outcome_index: u32, shares: i64) -> TxData {
        TxData::Trade {
            market_id,
            outcome_index,
            shares,
            trader: Address::ALL_ZEROS,
            limit_sats: 0,
            tx_pow_nonce: None,
            prev_block_hash: hashes::BlockHash([0; 32]),
        }
    }

    fn block_hash(height: u32) -> BlockHash {
        BlockHash([height as u8; 32])
    }

    fn block(height: u32, changes: Vec<MarketChange>) -> BlockMarketChanges {
        BlockMarketChanges {
            height,
            block_hash: block_hash(height),
            mainchain_timestamp: 1_000 + u64::from(height),
            changes,
        }
    }

    fn prices(
        shares: Array1<i64>,
        liquidity_base_sats: u64,
    ) -> Result<Vec<f64>, LmsrError> {
        MarketSnapshot {
            shares,
            liquidity_base_sats,
        }
        .prices()
    }

    #[test]
    fn market_changes_drop_skipped_and_other_market_txs() {
        let body = Body {
            coinbase: Default::default(),
            transactions: vec![
                tx(trade(MARKET_ID, 1, 1_000)),
                tx(trade(OTHER_MARKET_ID, 0, 500)),
                tx(trade(MARKET_ID, 0, 5_000)),
                tx(trade(MARKET_ID, 1, -400)),
                tx(TxData::AmplifyBeta {
                    market_id: MARKET_ID,
                    amount: 5_000,
                    market_author: Address::ALL_ZEROS,
                }),
            ],
            authorizations: Vec::new(),
            actor_proofs: Vec::new(),
        };
        assert_eq!(
            market_changes(&body, &[2], &MARKET_ID),
            vec![
                MarketChange::Trade {
                    outcome_index: 1,
                    shares: 1_000
                },
                MarketChange::Trade {
                    outcome_index: 1,
                    shares: -400
                },
                MarketChange::AmplifyBeta { amount: 5_000 },
            ]
        );
    }

    #[test]
    fn rebuild_create_buy_sell_amplify() -> anyhow::Result<()> {
        let blocks_newest_first = vec![
            block(14, vec![MarketChange::AmplifyBeta { amount: 5_000 }]),
            block(
                13,
                vec![MarketChange::Trade {
                    outcome_index: 1,
                    shares: -400,
                }],
            ),
            block(12, Vec::new()),
            block(
                11,
                vec![MarketChange::Trade {
                    outcome_index: 1,
                    shares: 1_000,
                }],
            ),
            block(10, Vec::new()),
            block(9, Vec::new()),
        ];
        let tip_snapshot = MarketSnapshot {
            shares: array![0, 600],
            liquidity_base_sats: 15_000,
        };

        let points = rebuild_price_history(
            MARKET_ID,
            tip_snapshot,
            10,
            blocks_newest_first
                .into_iter()
                .map(Ok)
                .transpose_into_fallible(),
        )?;

        let expected = [
            (10, prices(array![0, 0], 10_000)?),
            (11, prices(array![0, 1_000], 10_000)?),
            (13, prices(array![0, 600], 10_000)?),
            (14, prices(array![0, 600], 15_000)?),
        ]
        .map(|(height, prices)| MarketPricePoint {
            height,
            block_hash: block_hash(height),
            mainchain_timestamp: 1_000 + u64::from(height),
            prices,
        });
        assert_eq!(points, expected);
        assert!(points[0].prices.iter().all(|p| (p - 0.5).abs() < 1e-9));
        assert!(points[1].prices[1] > points[2].prices[1]);
        assert!(points[2].prices[1] > points[3].prices[1]);
        Ok(())
    }

    #[test]
    fn rebuild_without_creation_block_fails() {
        let result = rebuild_price_history(
            MARKET_ID,
            MarketSnapshot {
                shares: array![0, 0],
                liquidity_base_sats: 10_000,
            },
            10,
            vec![block(12, Vec::new()), block(11, Vec::new())]
                .into_iter()
                .map(Ok)
                .transpose_into_fallible(),
        );
        assert!(matches!(result, Err(Error::NoCreationBlock { .. })));
    }
}
