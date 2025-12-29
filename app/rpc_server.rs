use std::net::SocketAddr;

use bitcoin::Amount;
use jsonrpsee::{
    core::{RpcResult, async_trait},
    server::{RpcServiceBuilder, Server},
    types::ErrorObject,
};

use tower_http::{
    request_id::{
        MakeRequestId, PropagateRequestIdLayer, RequestId, SetRequestIdLayer,
    },
    trace::{DefaultOnFailure, DefaultOnResponse, TraceLayer},
};
use truthcoin_dc::{
    authorization::{self, Dst, Signature},
    math::safe_math::{self, Rounding},
    net::Peer,
    node::Node,
    state::{period_to_name, timestamp_to_period},
    types::{
        Address, Authorization, Block, BlockHash, EncryptionPubKey,
        FilledOutputContent, PointedOutput, Transaction, Txid, VerifyingKey,
        WithdrawalBundle,
    },
    validation::SlotValidator,
    wallet::{Balance, CreateMarketInput, SlotClaimInput},
};
use truthcoin_dc_app_rpc_api::{
    ConsensusResults, DecisionSummary, MarketBuyRequest, MarketBuyResponse,
    ParticipationStats, PeriodStats, RegisterVoterRequest, RpcServer,
    SlotFilter, SlotListItem, SlotState, SubmitVoteBatchRequest, TxInfo,
    VoteFilter, VoteInfo, VoterInfo, VoterInfoFull, VotingPeriodFull,
};

use crate::app::App;

fn custom_err_msg(err_msg: impl Into<String>) -> ErrorObject<'static> {
    ErrorObject::owned(-1, err_msg.into(), Option::<()>::None)
}

fn custom_err<Error>(error: Error) -> ErrorObject<'static>
where
    anyhow::Error: From<Error>,
{
    let error = anyhow::Error::from(error);
    custom_err_msg(format!("{error:#}"))
}

fn parse_market_id(
    market_id: &str,
) -> RpcResult<truthcoin_dc::state::MarketId> {
    let market_id_bytes = hex::decode(market_id)
        .map_err(|_| custom_err_msg("Invalid market ID hex format"))?;

    if market_id_bytes.len() != 6 {
        return Err(custom_err_msg("Market ID must be exactly 6 bytes"));
    }

    let mut id_array = [0u8; 6];
    id_array.copy_from_slice(&market_id_bytes);
    Ok(truthcoin_dc::state::MarketId::new(id_array))
}

pub struct RpcServerImpl {
    app: App,
}

impl RpcServerImpl {
    #[inline(always)]
    fn node(&self) -> &Node {
        &self.app.node
    }

    // block_height_to_testing_period, timestamp_to_period, and period_to_name
    // now use centralized implementations from truthcoin_dc

    async fn slots_status(
        &self,
    ) -> RpcResult<truthcoin_dc_app_rpc_api::SlotStatus> {
        let is_testing_mode = self.node().is_slots_testing_mode();
        let blocks_per_period = if is_testing_mode {
            self.node().get_slots_testing_config()
        } else {
            0
        };

        let current_timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let current_period = if is_testing_mode {
            let tip_height = self
                .app
                .node
                .try_get_tip_height()
                .map_err(custom_err)?
                .unwrap_or(0);
            self.node().block_height_to_testing_period(tip_height)
        } else {
            timestamp_to_period(current_timestamp)
        };

        let current_period_name = period_to_name(current_period);

        Ok(truthcoin_dc_app_rpc_api::SlotStatus {
            is_testing_mode,
            blocks_per_period,
            current_period,
            current_period_name,
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn claim_decision_slot(
        &self,
        period_index: u32,
        slot_index: u32,
        is_standard: bool,
        is_scaled: bool,
        question: String,
        min: Option<u16>,
        max: Option<u16>,
        fee_sats: u64,
    ) -> RpcResult<Txid> {
        use truthcoin_dc::state::slots::SlotId;

        if question.len() >= 1000 {
            return Err(custom_err_msg(
                "Question must be less than 1000 bytes",
            ));
        }

        let slot_id =
            SlotId::new(period_index, slot_index).map_err(custom_err)?;
        let slot_id_bytes = slot_id.as_bytes();
        let fee = Amount::from_sat(fee_sats);
        let tx = self
            .app
            .wallet
            .claim_decision_slot(
                SlotClaimInput {
                    slot_id_bytes,
                    is_standard,
                    is_scaled,
                    question,
                    min,
                    max,
                },
                fee,
            )
            .map_err(custom_err)?;

        let txid = tx.txid();

        self.app.sign_and_send(tx).map_err(custom_err)?;

        Ok(txid)
    }

    async fn get_slot_by_id(
        &self,
        slot_id_hex: String,
    ) -> RpcResult<Option<truthcoin_dc_app_rpc_api::SlotDetails>> {
        let slot_id = SlotValidator::parse_slot_id_from_hex(&slot_id_hex)
            .map_err(custom_err)?;
        let slot_opt = self.node().get_slot(slot_id).map_err(custom_err)?;

        let result = slot_opt.map(|slot| {
            let content = match slot.decision {
                None => truthcoin_dc_app_rpc_api::SlotContentInfo::Empty,
                Some(decision) => {
                    truthcoin_dc_app_rpc_api::SlotContentInfo::Decision(
                        truthcoin_dc_app_rpc_api::DecisionInfo {
                            id: hex::encode(decision.id),
                            market_maker_pubkey_hash: hex::encode(
                                decision.market_maker_pubkey_hash,
                            ),
                            is_standard: decision.is_standard,
                            is_scaled: decision.is_scaled,
                            question: decision.question,
                            min: decision.min,
                            max: decision.max,
                        },
                    )
                }
            };

            truthcoin_dc_app_rpc_api::SlotDetails {
                slot_id_hex: slot_id.to_hex(),
                period_index: slot_id.period_index(),
                slot_index: slot_id.slot_index(),
                content,
            }
        });

        Ok(result)
    }

    async fn list_markets_impl(
        &self,
    ) -> RpcResult<Vec<truthcoin_dc_app_rpc_api::MarketSummary>> {
        let markets_with_states = self
            .app
            .node
            .get_all_markets_with_states()
            .map_err(custom_err)?;

        let market_summaries = markets_with_states
            .into_iter()
            .map(|(market, computed_state)| {
                let market_id_hex = hex::encode(market.id.as_bytes());

                let volume_btc =
                    (market.total_volume_sats as f64) / 100_000_000.0;
                truthcoin_dc_app_rpc_api::MarketSummary {
                    market_id: market_id_hex,
                    title: market.title.clone(),
                    description: if market.description.len() > 100 {
                        format!("{}...", &market.description[..97])
                    } else {
                        market.description.clone()
                    },
                    outcome_count: market.get_outcome_count(),
                    state: format!("{computed_state:?}"),
                    volume: volume_btc,
                    created_at_height: market.created_at_height,
                }
            })
            .collect();

        Ok(market_summaries)
    }

    async fn view_market_impl(
        &self,
        market_id: String,
    ) -> RpcResult<Option<truthcoin_dc_app_rpc_api::MarketData>> {
        let market_id_struct = parse_market_id(&market_id)?;

        let (market, computed_state) = match self
            .app
            .node
            .get_market_by_id_with_state(&market_id_struct)
            .map_err(custom_err)?
        {
            Some((market, computed_state)) => (market, computed_state),
            None => return Ok(None),
        };

        let decisions = self
            .app
            .node
            .get_market_decisions(&market)
            .map_err(custom_err)?;

        let mut outcomes = Vec::new();
        let valid_state_combos = market.get_valid_state_combos();

        let prices = if let Some(mempool_shares) = self
            .app
            .node
            .get_mempool_shares(&market_id_struct)
            .map_err(custom_err)?
        {
            let all_prices = market.calculate_prices(&mempool_shares);
            let valid_prices: Vec<f64> = valid_state_combos
                .iter()
                .map(|(state_idx, _)| all_prices[*state_idx])
                .collect();

            let valid_sum: f64 = valid_prices.iter().sum();
            if valid_sum > 0.0 {
                valid_prices.iter().map(|p| p / valid_sum).collect()
            } else {
                vec![1.0 / valid_prices.len() as f64; valid_prices.len()]
            }
        } else {
            market.calculate_prices_for_display()
        };

        let total_volume = market.total_volume_sats as f64;

        for (i, (state_idx, _combo)) in valid_state_combos.iter().enumerate() {
            let name = match market
                .describe_outcome_by_state(*state_idx, &decisions)
            {
                Ok(description) => description,
                Err(_) => format!("Outcome {state_idx}"),
            };

            let current_price = prices[i];
            let probability = current_price;
            let volume = if i < market.outcome_volumes_sats.len() {
                market.outcome_volumes_sats[i] as f64
            } else {
                0.0
            };

            outcomes.push(truthcoin_dc_app_rpc_api::MarketOutcome {
                name,
                current_price,
                probability,
                volume,
                index: i,
            });
        }

        let decision_slots: Vec<String> = market
            .decision_slots
            .iter()
            .map(|slot_id| slot_id.to_hex())
            .collect();

        let resolution = if computed_state
            == truthcoin_dc::state::markets::MarketState::Ossified
        {
            let final_prices = market.final_prices();
            let mut winning_outcomes = Vec::new();

            for (i, (state_idx, _combo)) in
                valid_state_combos.iter().enumerate()
            {
                let final_price = final_prices[*state_idx];
                if final_price > 0.0 {
                    let name = match market
                        .describe_outcome_by_state(*state_idx, &decisions)
                    {
                        Ok(description) => description,
                        Err(_) => format!("Outcome {state_idx}"),
                    };
                    winning_outcomes.push(
                        truthcoin_dc_app_rpc_api::WinningOutcome {
                            outcome_index: i,
                            outcome_name: name,
                            final_price,
                        },
                    );
                }
            }

            let summary = if winning_outcomes.len() == 1 {
                format!("Resolved: {}", winning_outcomes[0].outcome_name)
            } else if winning_outcomes.is_empty() {
                "No winning outcome".to_string()
            } else {
                let names: Vec<String> = winning_outcomes
                    .iter()
                    .map(|w| {
                        format!(
                            "{} ({:.1}%)",
                            w.outcome_name,
                            w.final_price * 100.0
                        )
                    })
                    .collect();
                format!("Resolved: {}", names.join(", "))
            };

            Some(truthcoin_dc_app_rpc_api::MarketResolution {
                winning_outcomes,
                summary,
            })
        } else {
            None
        };

        let treasury_sats = self
            .app
            .node
            .get_market_treasury_sats(&market_id_struct)
            .map_err(custom_err)?;
        let treasury_btc = (treasury_sats as f64) / 100_000_000.0;

        let market_data = truthcoin_dc_app_rpc_api::MarketData {
            market_id,
            title: market.title.clone(),
            description: market.description.clone(),
            outcomes,
            state: format!("{computed_state:?}"),
            market_maker: market.creator_address.to_string(),
            expires_at: market.expires_at_height,
            beta: market.b(),
            trading_fee: market.trading_fee(),
            tags: market.tags.clone(),
            created_at_height: market.created_at_height,
            treasury: treasury_btc,
            total_volume,
            liquidity: treasury_btc,
            decision_slots,
            resolution,
        };

        Ok(Some(market_data))
    }

    async fn create_market_impl(
        &self,
        request: truthcoin_dc_app_rpc_api::CreateMarketRequest,
    ) -> RpcResult<String> {
        let tx = self
            .app
            .wallet
            .create_market(
                CreateMarketInput {
                    title: request.title,
                    description: request.description,
                    market_type: request.market_type,
                    decision_slots: if request.decision_slots.is_empty() {
                        None
                    } else {
                        Some(request.decision_slots)
                    },
                    dimensions: request.dimensions,
                    has_residual: request.has_residual,
                    beta: request.beta,
                    trading_fee: request.trading_fee,
                    tags: request.tags,
                    initial_liquidity: request.initial_liquidity,
                },
                bitcoin::Amount::from_sat(request.fee_sats),
            )
            .map_err(custom_err)?;

        let txid = tx.txid();

        self.app.sign_and_send(tx).map_err(custom_err)?;

        Ok(txid.to_string())
    }

    async fn get_user_share_positions_impl(
        &self,
        address: Address,
    ) -> RpcResult<truthcoin_dc_app_rpc_api::UserHoldings> {
        let node = &self.node();

        let positions_data = node
            .get_user_share_positions(&address)
            .map_err(custom_err)?;

        let mut positions = Vec::new();
        let mut total_value = 0.0;
        let mut total_cost_basis = 0.0;
        let mut active_markets = std::collections::HashSet::new();
        let mut last_updated_height = 0u64;

        let unique_market_ids: Vec<truthcoin_dc::state::MarketId> =
            positions_data
                .iter()
                .map(|(market_id, _, _)| market_id.clone())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();

        let markets_map = node
            .get_markets_batch(&unique_market_ids)
            .map_err(custom_err)?;

        let mut prices_cache = std::collections::HashMap::new();
        for (market_id, market) in &markets_map {
            prices_cache.insert(market_id.clone(), market.get_current_prices());
        }

        for (market_id, outcome_index, position_data) in positions_data {
            if let (Some(market), Some(current_prices)) =
                (markets_map.get(&market_id), prices_cache.get(&market_id))
            {
                let outcome_price = current_prices
                    .get(outcome_index as usize)
                    .copied()
                    .unwrap_or(0.0);
                let shares_held = position_data;
                let current_value = shares_held * outcome_price;

                let outcome_name = if let Some(combo) =
                    market.state_combos.get(outcome_index as usize)
                {
                    format!("Outcome {outcome_index}: {combo:?}")
                } else {
                    format!("Outcome {outcome_index}")
                };

                positions.push(truthcoin_dc_app_rpc_api::SharePosition {
                    market_id: market_id.to_string(),
                    outcome_index: outcome_index as usize,
                    outcome_name,
                    shares_held,
                    avg_purchase_price: outcome_price,
                    current_price: outcome_price,
                    current_value,
                    unrealized_pnl: 0.0,
                    cost_basis: current_value,
                });

                total_value += current_value;
                total_cost_basis += current_value;
                active_markets.insert(market_id);
                last_updated_height = last_updated_height.max(1);
            }
        }

        let total_unrealized_pnl = total_value - total_cost_basis;

        Ok(truthcoin_dc_app_rpc_api::UserHoldings {
            address: address.to_string(),
            positions,
            total_value,
            total_cost_basis,
            total_unrealized_pnl,
            active_markets: active_markets.len(),
            last_updated_height,
        })
    }

    async fn get_market_share_positions_impl(
        &self,
        address: Address,
        market_id: String,
    ) -> RpcResult<Vec<truthcoin_dc_app_rpc_api::SharePosition>> {
        let market_id_struct = parse_market_id(&market_id)?;
        let node = &self.node();

        let positions_data = node
            .get_market_user_positions(&address, &market_id_struct)
            .map_err(custom_err)?;

        let mut positions = Vec::new();

        if let Ok(Some(market)) = node.get_market_by_id(&market_id_struct) {
            let current_prices = market.get_current_prices();

            for (outcome_index, position_data) in positions_data {
                let outcome_price = current_prices
                    .get(outcome_index as usize)
                    .copied()
                    .unwrap_or(0.0);
                let shares_held = position_data;
                let current_value = shares_held * outcome_price;

                let outcome_name = if let Some(combo) =
                    market.state_combos.get(outcome_index as usize)
                {
                    format!("Outcome {outcome_index}: {combo:?}")
                } else {
                    format!("Outcome {outcome_index}")
                };

                positions.push(truthcoin_dc_app_rpc_api::SharePosition {
                    market_id: market_id.clone(),
                    outcome_index: outcome_index as usize,
                    outcome_name,
                    shares_held,
                    avg_purchase_price: outcome_price,
                    current_price: outcome_price,
                    current_value,
                    unrealized_pnl: 0.0,
                    cost_basis: current_value,
                });
            }
        }

        Ok(positions)
    }

    async fn register_voter_impl(
        &self,
        request: RegisterVoterRequest,
    ) -> RpcResult<String> {
        let fee = bitcoin::Amount::from_sat(request.fee_sats);

        let tx = self.app.wallet.register_voter(fee).map_err(custom_err)?;

        let txid = tx.txid();

        self.app.sign_and_send(tx).map_err(custom_err)?;

        Ok(format!("{txid}"))
    }

    async fn submit_vote_batch_impl(
        &self,
        request: SubmitVoteBatchRequest,
    ) -> RpcResult<String> {
        use truthcoin_dc::types::VoteBatchItem;

        if request.votes.is_empty() {
            return Err(custom_err_msg("Batch cannot be empty"));
        }

        let mut batch_items = Vec::new();
        let mut period_id: Option<u32> = None;

        for vote in request.votes {
            let slot_id =
                SlotValidator::parse_slot_id_from_hex(&vote.decision_id)
                    .map_err(|e| {
                        custom_err_msg(format!("Invalid decision ID: {e}"))
                    })?;

            let vote_period = slot_id.voting_period();

            match period_id {
                None => period_id = Some(vote_period),
                Some(p) if p != vote_period => {
                    return Err(custom_err_msg(format!(
                        "All votes in batch must be for same period. Expected {}, got {} for decision {}",
                        p, vote_period, vote.decision_id
                    )));
                }
                _ => {}
            }

            batch_items.push(VoteBatchItem {
                slot_id_bytes: slot_id.as_bytes(),
                vote_value: vote.vote_value,
            });
        }

        let period_id = period_id.unwrap();
        let fee = bitcoin::Amount::from_sat(request.fee_sats);

        let tx = self
            .app
            .wallet
            .submit_vote_batch(batch_items, period_id, fee)
            .map_err(custom_err)?;

        let txid = tx.txid();

        self.app.sign_and_send(tx).map_err(custom_err)?;

        Ok(format!("{txid}"))
    }

    async fn list_voters_impl(&self) -> RpcResult<Vec<VoterInfo>> {
        let rotxn = self.node().read_txn().map_err(custom_err)?;

        let all_voters = self
            .app
            .node
            .voting_state()
            .databases()
            .get_all_voters(&rotxn)
            .map_err(custom_err)?;

        let mut voter_infos = Vec::new();

        for voter_address in all_voters {
            let reputation_opt = self
                .app
                .node
                .voting_state()
                .databases()
                .get_voter_reputation(&rotxn, voter_address)
                .map_err(custom_err)?;

            let Some(reputation) = reputation_opt else {
                continue;
            };

            let votes = self
                .app
                .node
                .voting_state()
                .databases()
                .get_votes_by_voter(&rotxn, voter_address)
                .map_err(custom_err)?;

            voter_infos.push(VoterInfo {
                address: voter_address.to_string(),
                reputation: reputation.reputation,
                total_votes: votes.len() as u64,
                periods_active: reputation.total_decisions as u32,
                accuracy_score: reputation.get_accuracy_rate(),
                registered_at_height: 0,
                is_active: reputation.total_decisions > 0,
            });
        }

        Ok(voter_infos)
    }

    async fn get_votecoin_balance_impl(
        &self,
        address: Address,
    ) -> RpcResult<u32> {
        let rotxn = self.node().read_txn().map_err(custom_err)?;

        let votecoin_balance = self
            .app
            .node
            .get_votecoin_balance_for(&rotxn, &address)
            .map_err(custom_err)?;

        Ok(votecoin_balance)
    }
}

#[async_trait]
impl RpcServer for RpcServerImpl {
    async fn get_block(&self, block_hash: BlockHash) -> RpcResult<Block> {
        let block = self.node().get_block(block_hash).map_err(custom_err)?;
        Ok(block)
    }

    async fn get_best_sidechain_block_hash(
        &self,
    ) -> RpcResult<Option<BlockHash>> {
        self.node().try_get_tip().map_err(custom_err)
    }

    async fn get_best_mainchain_block_hash(
        &self,
    ) -> RpcResult<Option<bitcoin::BlockHash>> {
        let Some(sidechain_hash) =
            self.node().try_get_tip().map_err(custom_err)?
        else {
            return Ok(None);
        };
        let block_hash = self
            .node()
            .get_best_main_verification(sidechain_hash)
            .map_err(custom_err)?;
        Ok(Some(block_hash))
    }

    async fn get_bmm_inclusions(
        &self,
        block_hash: truthcoin_dc::types::BlockHash,
    ) -> RpcResult<Vec<bitcoin::BlockHash>> {
        self.app
            .node
            .get_bmm_inclusions(block_hash)
            .map_err(custom_err)
    }

    async fn get_new_address(&self) -> RpcResult<Address> {
        self.app.wallet.get_new_address().map_err(custom_err)
    }

    async fn get_new_encryption_key(&self) -> RpcResult<EncryptionPubKey> {
        self.app.wallet.get_new_encryption_key().map_err(custom_err)
    }

    async fn get_new_verifying_key(&self) -> RpcResult<VerifyingKey> {
        self.app.wallet.get_new_verifying_key().map_err(custom_err)
    }

    async fn get_transaction(
        &self,
        txid: Txid,
    ) -> RpcResult<Option<Transaction>> {
        self.node().try_get_transaction(txid).map_err(custom_err)
    }

    async fn get_transaction_info(
        &self,
        txid: Txid,
    ) -> RpcResult<Option<TxInfo>> {
        let Some((filled_tx, txin)) = self
            .app
            .node
            .try_get_filled_transaction(txid)
            .map_err(custom_err)?
        else {
            return Ok(None);
        };
        let confirmations = match txin {
            Some(txin) => {
                let tip_height = self
                    .app
                    .node
                    .try_get_tip_height()
                    .map_err(custom_err)?
                    .expect("Height should exist for tip");
                let height = self
                    .app
                    .node
                    .get_height(txin.block_hash)
                    .map_err(custom_err)?;
                Some(tip_height - height)
            }
            None => None,
        };
        let fee_sats = filled_tx
            .transaction
            .bitcoin_fee()
            .map_err(custom_err)?
            .unwrap()
            .to_sat();
        let res = TxInfo {
            confirmations,
            fee_sats,
            txin,
        };
        Ok(Some(res))
    }

    async fn get_wallet_addresses(&self) -> RpcResult<Vec<Address>> {
        let addrs = self.app.wallet.get_addresses().map_err(custom_err)?;
        let mut res: Vec<_> = addrs.into_iter().collect();
        res.sort_by_key(|addr| addr.as_base58());
        Ok(res)
    }

    async fn get_wallet_utxos(
        &self,
    ) -> RpcResult<Vec<PointedOutput<FilledOutputContent>>> {
        let utxos = self.app.wallet.get_utxos().map_err(custom_err)?;
        let utxos = utxos
            .into_iter()
            .map(|(outpoint, output)| PointedOutput { outpoint, output })
            .collect();
        Ok(utxos)
    }

    async fn getblockcount(&self) -> RpcResult<u32> {
        let height = self.node().try_get_tip_height().map_err(custom_err)?;
        Ok(height.map_or(0, |h| h + 1))
    }

    async fn latest_failed_withdrawal_bundle_height(
        &self,
    ) -> RpcResult<Option<u32>> {
        let height = self
            .app
            .node
            .get_latest_failed_bundle_height()
            .map_err(custom_err)?;
        Ok(height)
    }

    async fn list_peers(&self) -> RpcResult<Vec<Peer>> {
        let peers = self.node().get_active_peers();
        Ok(peers)
    }

    async fn list_utxos(
        &self,
    ) -> RpcResult<Vec<PointedOutput<FilledOutputContent>>> {
        let utxos = self.node().get_all_utxos().map_err(custom_err)?;
        let res = utxos
            .into_iter()
            .map(|(outpoint, output)| PointedOutput { outpoint, output })
            .collect();
        Ok(res)
    }

    async fn mine(&self, fee: Option<u64>) -> RpcResult<()> {
        let fee = fee.map(bitcoin::Amount::from_sat);
        self.app
            .local_pool
            .spawn_pinned({
                let app = self.app.clone();
                move || async move { app.mine(fee).await.map_err(custom_err) }
            })
            .await
            .unwrap()
    }

    async fn my_unconfirmed_utxos(&self) -> RpcResult<Vec<PointedOutput>> {
        let addresses = self.app.wallet.get_addresses().map_err(custom_err)?;
        let utxos = self
            .app
            .node
            .get_unconfirmed_utxos_by_addresses(&addresses)
            .map_err(custom_err)?
            .into_iter()
            .map(|(outpoint, output)| PointedOutput { outpoint, output })
            .collect();
        Ok(utxos)
    }

    async fn my_utxos(
        &self,
    ) -> RpcResult<Vec<PointedOutput<FilledOutputContent>>> {
        let utxos = self
            .app
            .wallet
            .get_utxos()
            .map_err(custom_err)?
            .into_iter()
            .map(|(outpoint, output)| PointedOutput { outpoint, output })
            .collect();
        Ok(utxos)
    }

    async fn openapi_schema(&self) -> RpcResult<utoipa::openapi::OpenApi> {
        let res =
            <truthcoin_dc_app_rpc_api::RpcDoc as utoipa::OpenApi>::openapi();
        Ok(res)
    }

    async fn pending_withdrawal_bundle(
        &self,
    ) -> RpcResult<Option<WithdrawalBundle>> {
        self.app
            .node
            .get_pending_withdrawal_bundle()
            .map_err(custom_err)
    }

    async fn remove_from_mempool(&self, txid: Txid) -> RpcResult<()> {
        self.node().remove_from_mempool(txid).map_err(custom_err)
    }

    async fn set_seed_from_mnemonic(&self, mnemonic: String) -> RpcResult<()> {
        self.app
            .wallet
            .set_seed_from_mnemonic(mnemonic.as_str())
            .map_err(custom_err)
    }

    async fn sidechain_wealth_sats(&self) -> RpcResult<u64> {
        let sidechain_wealth =
            self.node().get_sidechain_wealth().map_err(custom_err)?;
        Ok(sidechain_wealth.to_sat())
    }

    async fn sign_arbitrary_msg(
        &self,
        verifying_key: VerifyingKey,
        msg: String,
    ) -> RpcResult<Signature> {
        self.app
            .wallet
            .sign_arbitrary_msg(&verifying_key, &msg)
            .map_err(custom_err)
    }

    async fn sign_arbitrary_msg_as_addr(
        &self,
        address: Address,
        msg: String,
    ) -> RpcResult<Authorization> {
        self.app
            .wallet
            .sign_arbitrary_msg_as_addr(&address, &msg)
            .map_err(custom_err)
    }

    async fn stop(&self) {
        std::process::exit(0);
    }

    async fn transfer(
        &self,
        dest: Address,
        value_sats: u64,
        fee_sats: u64,
        memo: Option<String>,
    ) -> RpcResult<Txid> {
        let memo = match memo {
            None => None,
            Some(memo) => {
                let hex = hex::decode(memo).map_err(custom_err)?;
                Some(hex)
            }
        };
        let tx = self
            .app
            .wallet
            .create_transfer(
                dest,
                Amount::from_sat(value_sats),
                Amount::from_sat(fee_sats),
                memo,
            )
            .map_err(custom_err)?;
        let txid = tx.txid();
        let () = self.app.sign_and_send(tx).map_err(custom_err)?;
        Ok(txid)
    }

    async fn transfer_votecoin(
        &self,
        dest: Address,
        amount: u32,
        fee_sats: u64,
        memo: Option<String>,
    ) -> RpcResult<Txid> {
        let memo = match memo {
            None => None,
            Some(memo) => {
                let hex = hex::decode(memo).map_err(custom_err)?;
                Some(hex)
            }
        };
        let tx = self
            .app
            .wallet
            .create_votecoin_transfer(
                dest,
                amount,
                Amount::from_sat(fee_sats),
                memo,
            )
            .map_err(custom_err)?;
        let txid = tx.txid();
        let () = self.app.sign_and_send(tx).map_err(custom_err)?;
        Ok(txid)
    }

    async fn verify_signature(
        &self,
        signature: Signature,
        verifying_key: VerifyingKey,
        dst: Dst,
        msg: String,
    ) -> RpcResult<bool> {
        let res = authorization::verify(
            signature,
            &verifying_key,
            dst,
            msg.as_bytes(),
        );
        Ok(res)
    }

    async fn withdraw(
        &self,
        mainchain_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        amount_sats: u64,
        fee_sats: u64,
        mainchain_fee_sats: u64,
    ) -> RpcResult<Txid> {
        let tx = self
            .app
            .wallet
            .create_withdrawal(
                mainchain_address,
                Amount::from_sat(amount_sats),
                Amount::from_sat(mainchain_fee_sats),
                Amount::from_sat(fee_sats),
            )
            .map_err(custom_err)?;
        let txid = tx.txid();
        self.app.sign_and_send(tx).map_err(custom_err)?;
        Ok(txid)
    }

    async fn calculate_initial_liquidity(
        &self,
        request: truthcoin_dc_app_rpc_api::CalculateInitialLiquidityRequest,
    ) -> RpcResult<truthcoin_dc_app_rpc_api::InitialLiquidityCalculation> {
        let beta = request.beta;

        if beta <= 0.0 {
            return Err(custom_err_msg(format!(
                "Beta parameter must be positive, got: {beta}",
            )));
        }
        let (num_outcomes, market_config, outcome_breakdown) = if let Some(
            num,
        ) =
            request.num_outcomes
        {
            if num < 2 {
                return Err(custom_err_msg(
                    "Number of outcomes must be at least 2".to_string(),
                ));
            }
            (
                num,
                format!("Preview: {num} outcomes"),
                format!("{num} outcomes specified"),
            )
        } else if let Some(ref decision_slots) = request.decision_slots {
            match request.market_type.as_str() {
                "independent" => {
                    let num = 2_usize.pow(decision_slots.len() as u32);
                    (
                        num,
                        format!(
                            "Independent: {} binary decisions",
                            decision_slots.len()
                        ),
                        format!(
                            "{} binary decisions = 2^{} = {} total outcome combinations",
                            decision_slots.len(),
                            decision_slots.len(),
                            num
                        ),
                    )
                }
                "categorical" => {
                    let base_outcomes = decision_slots.len();
                    let has_residual = request.has_residual.unwrap_or(false);
                    let total = if has_residual {
                        base_outcomes + 1
                    } else {
                        base_outcomes
                    };
                    (
                        total,
                        format!(
                            "Categorical: {} categories{}",
                            base_outcomes,
                            if has_residual { " + residual" } else { "" }
                        ),
                        format!(
                            "{} categories{} = {} total outcomes",
                            base_outcomes,
                            if has_residual { " + 1 residual" } else { "" },
                            total
                        ),
                    )
                }
                "dimensional" => {
                    if let Some(ref _dims) = request.dimensions {
                        let estimated_outcomes = decision_slots.len() * 2;
                        (
                            estimated_outcomes,
                            format!(
                                "Dimensional: {} dimensions",
                                decision_slots.len()
                            ),
                            format!(
                                "Estimated {} outcomes from {} dimensions (actual calculation requires dimension parsing)",
                                estimated_outcomes,
                                decision_slots.len()
                            ),
                        )
                    } else {
                        return Err(custom_err_msg(
                            "Dimensional markets require dimension specification".to_string()
                        ));
                    }
                }
                _ => {
                    return Err(custom_err_msg(format!(
                        "Unknown market type: {}",
                        request.market_type
                    )));
                }
            }
        } else {
            return Err(custom_err_msg(
                "Either decision_slots or num_outcomes must be provided"
                    .to_string(),
            ));
        };

        let initial_liquidity_f64 = beta * (num_outcomes as f64).ln();
        let initial_liquidity_sats =
            safe_math::to_sats(initial_liquidity_f64, Rounding::Up).map_err(
                |e| {
                    custom_err_msg(format!("Liquidity calculation failed: {e}"))
                },
            )?;

        Ok(truthcoin_dc_app_rpc_api::InitialLiquidityCalculation {
            beta,
            num_outcomes,
            initial_liquidity_sats,
            min_treasury_sats: initial_liquidity_sats,
            market_config,
            outcome_breakdown,
        })
    }

    async fn bitcoin_balance(&self) -> RpcResult<Balance> {
        self.app.wallet.get_bitcoin_balance().map_err(custom_err)
    }

    async fn create_deposit(
        &self,
        address: Address,
        value_sats: u64,
        fee_sats: u64,
    ) -> RpcResult<bitcoin::Txid> {
        let tx = self
            .app
            .wallet
            .create_transfer(
                address,
                bitcoin::Amount::from_sat(value_sats),
                bitcoin::Amount::from_sat(fee_sats),
                None,
            )
            .map_err(custom_err)?;

        let txid = tx.txid();
        self.app.sign_and_send(tx).map_err(custom_err)?;

        let bitcoin_txid = bitcoin::Txid::from_raw_hash(
            bitcoin::hashes::Hash::from_byte_array(txid.0),
        );
        Ok(bitcoin_txid)
    }

    async fn connect_peer(&self, addr: SocketAddr) -> RpcResult<()> {
        self.node().connect_peer(addr).map_err(custom_err)
    }

    async fn decrypt_msg(
        &self,
        encryption_pubkey: EncryptionPubKey,
        ciphertext: String,
    ) -> RpcResult<String> {
        let ciphertext_bytes = hex::decode(&ciphertext).map_err(|e| {
            ErrorObject::owned(
                -32602,
                "Invalid hex string",
                Some(e.to_string()),
            )
        })?;

        let decrypted_bytes = self
            .app
            .wallet
            .decrypt_msg(&encryption_pubkey, &ciphertext_bytes)
            .map_err(custom_err)?;

        Ok(hex::encode(decrypted_bytes))
    }

    async fn encrypt_msg(
        &self,
        _encryption_pubkey: EncryptionPubKey,
        _msg: String,
    ) -> RpcResult<String> {
        Err(ErrorObject::owned(
            -32601,
            "Encryption not implemented",
            Some("Use external encryption tools"),
        ))
    }

    async fn format_deposit_address(
        &self,
        address: Address,
    ) -> RpcResult<String> {
        Ok(format!("{address}"))
    }

    async fn generate_mnemonic(&self) -> RpcResult<String> {
        let mnemonic = bip39::Mnemonic::new(
            bip39::MnemonicType::Words12,
            bip39::Language::English,
        );
        Ok(mnemonic.to_string())
    }

    async fn refresh_wallet(&self) -> RpcResult<()> {
        self.app.update().map_err(custom_err)
    }

    async fn slot_status(
        &self,
    ) -> RpcResult<truthcoin_dc_app_rpc_api::SlotStatus> {
        self.slots_status().await
    }

    async fn slot_list(
        &self,
        filter: Option<SlotFilter>,
    ) -> RpcResult<Vec<SlotListItem>> {
        use truthcoin_dc::state::voting::types::VotingPeriodId;

        let mut results = Vec::new();

        // Get current period for determining slot states
        let current_timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let is_testing = self.node().is_slots_testing_mode();
        let current_period = if is_testing {
            let tip_height = self
                .node()
                .try_get_tip_height()
                .map_err(custom_err)?
                .unwrap_or(0);
            self.node().block_height_to_testing_period(tip_height)
        } else {
            timestamp_to_period(current_timestamp)
        };

        // Get periods to check
        let periods_to_check: Vec<u32> = if let Some(ref f) = filter {
            if let Some(p) = f.period {
                vec![p]
            } else {
                // Get all periods that have slots
                let all_slots =
                    self.node().get_all_slot_quarters().map_err(custom_err)?;
                all_slots.into_iter().map(|(p, _)| p).collect()
            }
        } else {
            let all_slots =
                self.node().get_all_slot_quarters().map_err(custom_err)?;
            all_slots.into_iter().map(|(p, _)| p).collect()
        };

        for period in periods_to_check {
            let period_id = VotingPeriodId(period);

            // Get all slots in period (available)
            let available = self
                .node()
                .get_available_slots_in_period(period_id)
                .map_err(custom_err)?;
            for slot_id in available {
                let state = SlotState::Available;

                // Apply filter
                if let Some(ref f) = filter
                    && let Some(ref status) = f.status
                    && !matches!(status, SlotState::Available)
                {
                    continue;
                }

                results.push(SlotListItem {
                    slot_id_hex: slot_id.to_hex(),
                    period_index: slot_id.period_index(),
                    slot_index: slot_id.slot_index(),
                    state,
                    decision: None,
                });
            }

            // Get claimed slots
            let claimed = self
                .node()
                .get_claimed_slots_in_period(period_id)
                .map_err(custom_err)?;
            for slot in claimed {
                // Determine state based on period
                let voting_period = slot.slot_id.voting_period();
                let state = if voting_period < current_period.saturating_sub(1)
                {
                    SlotState::Ossified
                } else if voting_period == current_period
                    || voting_period == current_period.saturating_sub(1)
                {
                    SlotState::Voting
                } else {
                    SlotState::Claimed
                };

                // Apply filter
                if let Some(ref f) = filter
                    && let Some(ref status) = f.status
                    && !matches!(
                        (status, &state),
                        (SlotState::Available, SlotState::Available)
                            | (SlotState::Claimed, SlotState::Claimed)
                            | (SlotState::Voting, SlotState::Voting)
                            | (SlotState::Ossified, SlotState::Ossified)
                    )
                {
                    continue;
                }

                let decision = slot.decision.map(|d| {
                    truthcoin_dc_app_rpc_api::DecisionInfo {
                        id: hex::encode(d.id),
                        market_maker_pubkey_hash: hex::encode(
                            d.market_maker_pubkey_hash,
                        ),
                        is_standard: d.is_standard,
                        is_scaled: d.is_scaled,
                        question: d.question,
                        min: d.min,
                        max: d.max,
                    }
                });

                results.push(SlotListItem {
                    slot_id_hex: slot.slot_id.to_hex(),
                    period_index: slot.slot_id.period_index(),
                    slot_index: slot.slot_id.slot_index(),
                    state,
                    decision,
                });
            }
        }

        Ok(results)
    }

    async fn slot_get(
        &self,
        slot_id: String,
    ) -> RpcResult<Option<truthcoin_dc_app_rpc_api::SlotDetails>> {
        self.get_slot_by_id(slot_id).await
    }

    async fn slot_claim(
        &self,
        period_index: u32,
        slot_index: u32,
        is_standard: bool,
        is_scaled: bool,
        question: String,
        min: Option<u16>,
        max: Option<u16>,
        fee_sats: u64,
    ) -> RpcResult<Txid> {
        self.claim_decision_slot(
            period_index,
            slot_index,
            is_standard,
            is_scaled,
            question,
            min,
            max,
            fee_sats,
        )
        .await
    }

    // --- MARKETS ---

    async fn market_create(
        &self,
        request: truthcoin_dc_app_rpc_api::CreateMarketRequest,
    ) -> RpcResult<String> {
        self.create_market_impl(request).await
    }

    async fn market_list(
        &self,
    ) -> RpcResult<Vec<truthcoin_dc_app_rpc_api::MarketSummary>> {
        self.list_markets_impl().await
    }

    async fn market_get(
        &self,
        market_id: String,
    ) -> RpcResult<Option<truthcoin_dc_app_rpc_api::MarketData>> {
        self.view_market_impl(market_id).await
    }

    async fn market_buy(
        &self,
        request: MarketBuyRequest,
    ) -> RpcResult<MarketBuyResponse> {
        let market_id_struct = parse_market_id(&request.market_id)?;

        // Calculate cost first
        let market = self
            .node()
            .get_market_by_id(&market_id_struct)
            .map_err(custom_err)?
            .ok_or_else(|| custom_err_msg("Market not found"))?;

        // Calculate base cost and trading fee separately for explicit outputs
        let mut new_shares = market.shares().clone();
        new_shares[request.outcome_index] += request.shares_amount;
        let trade_cost =
            market.query_update_cost(new_shares).map_err(custom_err)?;
        let fee_amount = trade_cost * market.trading_fee();
        let base_cost_sats = safe_math::to_sats(trade_cost, Rounding::Up)
            .map_err(|e| {
                custom_err_msg(format!("Cost calculation failed: {e}"))
            })?;
        let trading_fee_sats = safe_math::to_sats(fee_amount, Rounding::Up)
            .map_err(|e| {
                custom_err_msg(format!("Fee calculation failed: {e}"))
            })?;
        let cost_sats = base_cost_sats + trading_fee_sats;

        // Calculate new price after trade
        let prices = market.get_current_prices();
        let new_price =
            prices.get(request.outcome_index).copied().unwrap_or(0.0);

        // If dry_run, return cost without executing
        if request.dry_run.unwrap_or(false) {
            return Ok(MarketBuyResponse {
                txid: None,
                cost_sats,
                base_cost_sats,
                trading_fee_sats,
                new_price,
            });
        }

        // Validate required params for actual trade
        let max_cost = request.max_cost.ok_or_else(|| {
            custom_err_msg("max_cost is required when dry_run is false")
        })?;
        let fee_sats = request.fee_sats.ok_or_else(|| {
            custom_err_msg("fee_sats is required when dry_run is false")
        })?;

        // Slippage check
        if cost_sats > max_cost {
            return Err(custom_err_msg(format!(
                "Share cost {cost_sats} exceeds maximum cost {max_cost} (slippage protection)",
            )));
        }

        // Execute trade
        let tx = self
            .app
            .wallet
            .buy_shares(
                market_id_struct,
                request.outcome_index,
                request.shares_amount,
                base_cost_sats,
                trading_fee_sats,
                bitcoin::Amount::from_sat(fee_sats),
            )
            .map_err(custom_err)?;

        let txid = tx.txid();
        self.app.sign_and_send(tx).map_err(custom_err)?;

        Ok(MarketBuyResponse {
            txid: Some(txid.to_string()),
            cost_sats,
            base_cost_sats,
            trading_fee_sats,
            new_price,
        })
    }

    async fn market_positions(
        &self,
        address: Address,
        market_id: Option<String>,
    ) -> RpcResult<truthcoin_dc_app_rpc_api::UserHoldings> {
        if let Some(mid) = market_id {
            // Filter by market - get positions for specific market
            let positions =
                self.get_market_share_positions_impl(address, mid).await?;

            let total_value: f64 =
                positions.iter().map(|p| p.current_value).sum();
            let total_cost_basis: f64 =
                positions.iter().map(|p| p.cost_basis).sum();

            Ok(truthcoin_dc_app_rpc_api::UserHoldings {
                address: address.to_string(),
                positions,
                total_value,
                total_cost_basis,
                total_unrealized_pnl: total_value - total_cost_basis,
                active_markets: 1,
                last_updated_height: 0,
            })
        } else {
            // Get all positions
            self.get_user_share_positions_impl(address).await
        }
    }

    // --- VOTING ---

    async fn vote_register(
        &self,
        request: RegisterVoterRequest,
    ) -> RpcResult<String> {
        self.register_voter_impl(request).await
    }

    async fn vote_voter(
        &self,
        address: Address,
    ) -> RpcResult<Option<VoterInfoFull>> {
        // Gather config values BEFORE opening the read transaction
        let current_timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let current_height = self
            .node()
            .try_get_tip_height()
            .map_err(custom_err)?
            .unwrap_or(0);
        let config = self.node().get_slot_config();
        let slots_db = self.node().get_slots_db();

        let rotxn = self.node().read_txn().map_err(custom_err)?;

        // Get basic voter info
        let reputation_opt = self
            .app
            .node
            .voting_state()
            .databases()
            .get_voter_reputation(&rotxn, address)
            .map_err(custom_err)?;

        let is_registered = reputation_opt.is_some();

        if !is_registered {
            return Ok(Some(VoterInfoFull {
                address: address.to_string(),
                is_registered: false,
                reputation: 0.0,
                votecoin_balance: 0,
                total_votes: 0,
                periods_active: 0,
                accuracy_score: 0.0,
                registered_at_height: 0,
                is_active: false,
                current_period_participation: None,
            }));
        }

        let reputation = reputation_opt.unwrap();

        // Get votecoin balance
        let votecoin_balance = self
            .app
            .node
            .get_votecoin_balance_for(&rotxn, &address)
            .map_err(custom_err)?;

        // Get vote count
        let votes = self
            .app
            .node
            .voting_state()
            .databases()
            .get_votes_by_voter(&rotxn, address)
            .map_err(custom_err)?;

        // Get current period participation
        let active_period_opt = self
            .app
            .node
            .voting_state()
            .get_active_period(
                &rotxn,
                current_timestamp,
                current_height,
                config,
                slots_db,
            )
            .map_err(custom_err)?;

        let current_period_participation =
            if let Some(period) = active_period_opt {
                let period_votes: Vec<_> = votes
                    .iter()
                    .filter(|(key, _)| key.period_id == period.id)
                    .collect();

                let votes_cast = period_votes.len() as u32;
                let decisions_available = period.decision_slots.len() as u32;
                let participation_rate = if decisions_available > 0 {
                    votes_cast as f64 / decisions_available as f64
                } else {
                    0.0
                };

                Some(ParticipationStats {
                    period_id: period.id.as_u32(),
                    votes_cast,
                    decisions_available,
                    participation_rate,
                })
            } else {
                None
            };

        Ok(Some(VoterInfoFull {
            address: address.to_string(),
            is_registered: true,
            reputation: reputation.reputation,
            votecoin_balance,
            total_votes: votes.len() as u64,
            periods_active: reputation.total_decisions as u32,
            accuracy_score: reputation.get_accuracy_rate(),
            registered_at_height: 0,
            is_active: reputation.total_decisions > 0,
            current_period_participation,
        }))
    }

    async fn vote_voters(&self) -> RpcResult<Vec<VoterInfo>> {
        self.list_voters_impl().await
    }

    async fn vote_submit(
        &self,
        votes: Vec<truthcoin_dc_app_rpc_api::VoteBatchItem>,
        fee_sats: u64,
    ) -> RpcResult<String> {
        let request = SubmitVoteBatchRequest { votes, fee_sats };
        self.submit_vote_batch_impl(request).await
    }

    async fn vote_list(&self, filter: VoteFilter) -> RpcResult<Vec<VoteInfo>> {
        use truthcoin_dc::state::voting::types::VotingPeriodId;

        let rotxn = self.node().read_txn().map_err(custom_err)?;
        let mut vote_infos = Vec::new();

        // If filtering by voter
        if let Some(voter_address) = filter.voter {
            let all_votes = self
                .app
                .node
                .voting_state()
                .databases()
                .get_votes_by_voter(&rotxn, voter_address)
                .map_err(custom_err)?;

            for (vote_key, vote_entry) in all_votes {
                // Apply period filter
                if let Some(pid) = filter.period_id
                    && vote_key.period_id != VotingPeriodId::new(pid)
                {
                    continue;
                }

                // Apply decision filter
                if let Some(ref did) = filter.decision_id
                    && vote_key.decision_id.to_hex() != *did
                {
                    continue;
                }

                // Skip abstentions - they don't have a meaningful vote value
                if let Some(vote_value) = vote_entry.to_float_opt() {
                    vote_infos.push(VoteInfo {
                        voter_address: voter_address.to_string(),
                        decision_id: vote_key.decision_id.to_hex(),
                        vote_value,
                        period_id: vote_key.period_id.as_u32(),
                        block_height: vote_entry.block_height,
                        txid: String::from(""),
                        is_batch_vote: false,
                    });
                }
            }
        } else if let Some(ref decision_id) = filter.decision_id {
            // If filtering by decision
            let slot_id = SlotValidator::parse_slot_id_from_hex(decision_id)
                .map_err(|e| {
                    custom_err_msg(format!("Invalid decision ID: {e}"))
                })?;

            let votes = self
                .app
                .node
                .voting_state()
                .databases()
                .get_votes_for_decision(&rotxn, slot_id)
                .map_err(custom_err)?;

            for (vote_key, vote_entry) in votes {
                if let Some(pid) = filter.period_id
                    && vote_key.period_id != VotingPeriodId::new(pid)
                {
                    continue;
                }

                // Skip abstentions - they don't have a meaningful vote value
                if let Some(vote_value) = vote_entry.to_float_opt() {
                    vote_infos.push(VoteInfo {
                        voter_address: vote_key.voter_address.to_string(),
                        decision_id: decision_id.clone(),
                        vote_value,
                        period_id: vote_key.period_id.as_u32(),
                        block_height: vote_entry.block_height,
                        txid: String::from(""),
                        is_batch_vote: false,
                    });
                }
            }
        } else if let Some(period_id) = filter.period_id {
            // If filtering by period only
            let voting_period_id = VotingPeriodId::new(period_id);
            let votes = self
                .app
                .node
                .voting_state()
                .databases()
                .get_votes_for_period(&rotxn, voting_period_id)
                .map_err(custom_err)?;

            for (vote_key, vote_entry) in votes {
                // Skip abstentions - they don't have a meaningful vote value
                if let Some(vote_value) = vote_entry.to_float_opt() {
                    vote_infos.push(VoteInfo {
                        voter_address: vote_key.voter_address.to_string(),
                        decision_id: vote_key.decision_id.to_hex(),
                        vote_value,
                        period_id: vote_key.period_id.as_u32(),
                        block_height: vote_entry.block_height,
                        txid: String::from(""),
                        is_batch_vote: false,
                    });
                }
            }
        }

        Ok(vote_infos)
    }

    async fn vote_period(
        &self,
        period_id: Option<u32>,
    ) -> RpcResult<Option<VotingPeriodFull>> {
        use truthcoin_dc::state::voting::types::VotingPeriodId;

        // Gather config values BEFORE opening the main read transaction
        let current_height = self
            .node()
            .try_get_tip_height()
            .map_err(custom_err)?
            .unwrap_or(0);
        let current_timestamp_for_period =
            self.node().get_last_block_timestamp().map_err(custom_err)?;
        let config = self.node().get_slot_config();
        let slots_db = self.node().get_slots_db();

        // Get current period if none specified
        let period_id = if let Some(pid) = period_id {
            pid
        } else {
            let rotxn = self.node().read_txn().map_err(custom_err)?;
            let current_timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| custom_err_msg(format!("System time error: {e}")))?
                .as_secs();

            let active_period_opt = self
                .app
                .node
                .voting_state()
                .get_active_period(
                    &rotxn,
                    current_timestamp,
                    current_height,
                    config,
                    slots_db,
                )
                .map_err(custom_err)?;

            match active_period_opt {
                Some(period) => period.id.as_u32(),
                None => return Ok(None),
            }
        };

        let voting_period_id = VotingPeriodId::new(period_id);

        // Collect slot data BEFORE opening the main read transaction
        // to avoid nested transactions
        let rotxn = self.node().read_txn().map_err(custom_err)?;

        let has_consensus = self
            .app
            .node
            .voting_state()
            .databases()
            .has_consensus(&rotxn, voting_period_id)
            .map_err(custom_err)?;

        let period = match truthcoin_dc::state::voting::period_calculator::calculate_voting_period(
            &rotxn,
            voting_period_id,
            current_height,
            current_timestamp_for_period,
            config,
            slots_db,
            has_consensus,
        ) {
            Ok(p) => p,
            Err(_) => return Ok(None),
        };

        // Collect decision slot IDs to fetch outside the transaction
        let decision_slot_ids: Vec<_> = period.decision_slots.clone();

        // Drop the read transaction before fetching slots
        drop(rotxn);

        // Get decisions - each get_slot call can safely use its own transaction now
        let decisions: Vec<DecisionSummary> = decision_slot_ids
            .iter()
            .map(|slot_id| {
                let slot_opt = self.node().get_slot(*slot_id).ok().flatten();
                let (question, is_standard, is_scaled) = slot_opt
                    .and_then(|s| s.decision)
                    .map(|d| (d.question, d.is_standard, d.is_scaled))
                    .unwrap_or(("".to_string(), false, false));

                DecisionSummary {
                    slot_id_hex: slot_id.to_hex(),
                    question,
                    is_standard,
                    is_scaled,
                }
            })
            .collect();

        // Re-open transaction for remaining operations
        let rotxn = self.node().read_txn().map_err(custom_err)?;

        // Get stats
        let (total_voters, total_votes, _) = self
            .app
            .node
            .voting_state()
            .get_participation_stats(&rotxn, voting_period_id, config, slots_db)
            .map_err(custom_err)?;

        let votes = self
            .app
            .node
            .voting_state()
            .databases()
            .get_votes_for_period(&rotxn, voting_period_id)
            .map_err(custom_err)?;

        let active_voters: std::collections::HashSet<_> =
            votes.keys().map(|k| k.voter_address).collect();

        let participation_rate = if total_voters > 0 {
            active_voters.len() as f64 / total_voters as f64
        } else {
            0.0
        };

        let stats = PeriodStats {
            total_voters,
            active_voters: active_voters.len() as u64,
            total_votes,
            participation_rate,
        };

        // Get consensus if resolved
        let consensus = if period.status
            == truthcoin_dc::state::voting::types::VotingPeriodStatus::Resolved
        {
            let outcomes_map = self
                .app
                .node
                .voting_state()
                .databases()
                .get_consensus_outcomes_for_period(&rotxn, voting_period_id)
                .map_err(custom_err)?;

            let mut outcomes = std::collections::HashMap::new();
            for (slot_id, outcome) in outcomes_map {
                outcomes.insert(slot_id.to_hex(), outcome);
            }

            let period_stats = self
                .app
                .node
                .voting_state()
                .databases()
                .get_period_stats(&rotxn, voting_period_id)
                .map_err(custom_err)?;

            let mut reputation_updates = std::collections::HashMap::new();
            let (first_loading, certainty) = if let Some(ref ps) = period_stats
            {
                if let Some(ref rep_changes) = ps.reputation_changes {
                    for (voter_id, (old_rep, new_rep)) in rep_changes {
                        if let Some(reputation) = self
                            .node()
                            .voting_state()
                            .databases()
                            .get_voter_reputation(&rotxn, *voter_id)
                            .map_err(custom_err)?
                        {
                            let rep_update =
                                truthcoin_dc_app_rpc_api::ReputationUpdate {
                                    old_reputation: *old_rep,
                                    new_reputation: *new_rep,
                                    votecoin_proportion: reputation
                                        .votecoin_proportion,
                                    compliance_score: 0.0,
                                };
                            reputation_updates
                                .insert(voter_id.to_string(), rep_update);
                        }
                    }
                }
                (
                    ps.first_loading.clone().unwrap_or_default(),
                    ps.certainty.unwrap_or(0.0),
                )
            } else {
                (Vec::new(), 0.0)
            };

            Some(ConsensusResults {
                outcomes,
                first_loading,
                certainty,
                reputation_updates,
                outliers: Vec::new(),
                vote_matrix_dimensions: (active_voters.len(), decisions.len()),
                algorithm_version: "SVD-PCA-v1.0".to_string(),
            })
        } else {
            None
        };

        // Get redistribution if available
        let redistribution = self
            .app
            .node
            .voting_state()
            .databases()
            .get_pending_redistribution(&rotxn, voting_period_id)
            .map_err(custom_err)?
            .map(|redist| truthcoin_dc_app_rpc_api::RedistributionInfo {
                period_id,
                total_redistributed: redist
                    .redistribution_summary
                    .total_redistributed,
                winners_count: redist.redistribution_summary.winners_count,
                losers_count: redist.redistribution_summary.losers_count,
                unchanged_count: redist.redistribution_summary.unchanged_count,
                conservation_check: redist
                    .redistribution_summary
                    .conservation_check,
                block_height: redist.calculated_at_height,
                is_applied: redist.applied,
                slots_affected: redist
                    .slots_pending_redistribution
                    .iter()
                    .map(|s| s.to_hex())
                    .collect(),
            });

        Ok(Some(VotingPeriodFull {
            period_id,
            status: format!("{:?}", period.status),
            start_height: 0, // Not stored
            end_height: 0,   // Not stored
            start_time: period.start_timestamp,
            end_time: period.end_timestamp,
            decisions,
            stats,
            consensus,
            redistribution,
        }))
    }

    // --- VOTECOIN ---

    async fn votecoin_transfer(
        &self,
        dest: Address,
        amount: u32,
        fee_sats: u64,
        memo: Option<String>,
    ) -> RpcResult<Txid> {
        self.transfer_votecoin(dest, amount, fee_sats, memo).await
    }

    async fn votecoin_balance(&self, address: Address) -> RpcResult<u32> {
        self.get_votecoin_balance_impl(address).await
    }
}

#[derive(Clone, Debug)]
struct RequestIdMaker;

impl MakeRequestId for RequestIdMaker {
    fn make_request_id<B>(
        &mut self,
        _: &http::Request<B>,
    ) -> Option<RequestId> {
        use uuid::Uuid;
        let id = Uuid::new_v4();
        let id = id.as_simple();
        let id = format!("req_{id}");

        let Ok(header_value) = http::HeaderValue::from_str(&id) else {
            return None;
        };

        Some(RequestId::new(header_value))
    }
}

pub async fn run_server(
    app: App,
    rpc_url: url::Url,
) -> anyhow::Result<SocketAddr> {
    const REQUEST_ID_HEADER: &str = "x-request-id";

    let tracer = tower::ServiceBuilder::new()
        .layer(SetRequestIdLayer::new(
            http::HeaderName::from_static(REQUEST_ID_HEADER),
            RequestIdMaker,
        ))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(move |request: &http::Request<_>| {
                    let request_id = request
                        .headers()
                        .get(http::HeaderName::from_static(REQUEST_ID_HEADER))
                        .and_then(|h| h.to_str().ok())
                        .filter(|s| !s.is_empty());

                    tracing::span!(
                        tracing::Level::DEBUG,
                        "request",
                        method = %request.method(),
                        uri = %request.uri(),
                        request_id,
                    )
                })
                .on_request(())
                .on_eos(())
                .on_response(
                    DefaultOnResponse::new().level(tracing::Level::INFO),
                )
                .on_failure(
                    DefaultOnFailure::new().level(tracing::Level::ERROR),
                ),
        )
        .layer(PropagateRequestIdLayer::new(http::HeaderName::from_static(
            REQUEST_ID_HEADER,
        )))
        .into_inner();

    let http_middleware = tower::ServiceBuilder::new().layer(tracer);
    let rpc_middleware = RpcServiceBuilder::new().rpc_logger(1024);

    let server = Server::builder()
        .set_http_middleware(http_middleware)
        .set_rpc_middleware(rpc_middleware)
        .build(rpc_url.socket_addrs(|| None)?.as_slice())
        .await?;

    let addr = server.local_addr()?;
    let handle = server.start(RpcServerImpl { app }.into_rpc());

    tokio::spawn(handle.stopped());

    Ok(addr)
}
