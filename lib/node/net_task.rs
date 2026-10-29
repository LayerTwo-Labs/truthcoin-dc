//! Task to manage peers and their responses

use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant},
};

use error_fatality::{Fatality as _, Nested as _, Split};
use fallible_iterator::{FallibleIterator, IteratorExt};
use futures::{
    StreamExt,
    channel::{
        mpsc::{self, UnboundedReceiver, UnboundedSender},
        oneshot,
    },
    stream,
};
use nonempty::NonEmpty;
use rand::{RngExt as _, SeedableRng, rngs::ChaCha20Rng};
use sneed::{DbError, EnvError, RwTxn, RwTxnError};
use tokio::task::{self, JoinHandle};
use tokio_stream::StreamNotifyClose;

use crate::{
    archive::{self, Archive},
    mempool::{self, MemPool},
    net::{
        self, Net, PeerConnectionInfo, PeerConnectionMessage, PeerInfoRx,
        PeerRequest, PeerResponse, PeerStateId, error::peer::Recoverable as _,
        peer_message,
    },
    node::{
        FatalErrorTx,
        error::net_task::{self as error, Error},
        mainchain_task::{self, MainchainTaskHandle},
        supervise,
    },
    state::{self, State},
    types::{
        BmmResult, Body, Header, MerkleRoot, Tip,
        authorization::BatchVerificationContext,
        net::{PeerAddress, ResolvedPeerAddress},
        proto::mainchain::{self, Event as MainchainBlockEvent},
    },
    util::{ErrorChain, join_set},
};

/// Reconnect delays for one peer address, so that a connection that keeps
/// failing is retried less and less often instead of every 10s forever.
#[derive(Debug, Default)]
struct ReconnectBackoff {
    /// Reconnects in a row that did not stay up for [`Self::STABLE`]
    failures: u32,
    /// When the last reconnect was scheduled to happen
    last_dial: Option<Instant>,
}

impl ReconnectBackoff {
    const BASE: Duration = Duration::from_secs(10);
    const MAX: Duration = Duration::from_secs(10 * 60);
    /// A connection that stayed up at least this long resets the backoff
    const STABLE: Duration = Duration::from_secs(5 * 60);

    /// Delay before the next reconnect: [`Self::BASE`], doubled for each
    /// failure in a row, capped at [`Self::MAX`], plus up to 25% jitter so
    /// that two peers do not keep retrying in step.
    fn next_delay<R>(&mut self, now: Instant, mut rng: R) -> Duration
    where
        R: rand::CryptoRng,
    {
        if self.last_dial.is_none_or(|last_dial| {
            now.saturating_duration_since(last_dial) >= Self::STABLE
        }) {
            self.failures = 0;
        }
        let factor = 1u32.checked_shl(self.failures).unwrap_or(u32::MAX);
        let delay = Self::BASE.saturating_mul(factor).min(Self::MAX);
        let jitter = rng.random_range(0f64..=0.25);
        let delay = delay.mul_f64(1.0 + jitter);
        self.failures = self.failures.saturating_add(1);
        self.last_dial = Some(now + delay);
        delay
    }

    /// [`Self::BASE`] plus up to 25% jitter, for a dial that failed before
    /// the connection was usable (a timeout, or "already connected" from two
    /// nodes dialing each other at once): retried promptly as before, but no
    /// longer in step with the other node.
    fn jittered<R>(mut rng: R) -> Duration
    where
        R: rand::CryptoRng,
    {
        Self::BASE.mul_f64(1.0 + rng.random_range(0f64..=0.25))
    }
}

#[allow(clippy::too_many_arguments)]
fn connect_tip_(
    rwtxn: &mut RwTxn<'_>,
    archive: &Archive,
    batch_verification_ctxt: &BatchVerificationContext,
    mempool: &MemPool,
    state: &State,
    header: &Header,
    body: &Body,
    two_way_peg_data: &mainchain::TwoWayPegData,
) -> Result<(), Error> {
    let block_hash = header.hash();
    let prev_main_height =
        archive.get_main_height(rwtxn, header.prev_main_hash)?;
    // 2WPD events are connected BEFORE the block's transactions, so
    // that txs can spend eg. recreated unsuccessful-withdrawal outputs.
    let () = state.connect_two_way_peg_data(rwtxn, two_way_peg_data)?;
    let prevalidated = state.prevalidate_block(
        rwtxn,
        batch_verification_ctxt,
        header,
        prev_main_height,
        body,
        archive,
    )?;
    let mainchain_timestamp = {
        let main_block_hash =
            archive.get_best_main_verification(rwtxn, block_hash)?;
        archive
            .get_main_header_info(rwtxn, &main_block_hash)?
            .timestamp
    };
    if tracing::enabled!(tracing::Level::DEBUG) {
        let height = state.try_get_height(rwtxn)?;
        let merkle_root = state.connect_prevalidated_block(
            rwtxn,
            header,
            body,
            prevalidated,
            mainchain_timestamp,
        )?;
        tracing::debug!(?height, %merkle_root, %block_hash, "connected body")
    } else {
        let _: MerkleRoot = state.connect_prevalidated_block(
            rwtxn,
            header,
            body,
            prevalidated,
            mainchain_timestamp,
        )?;
    }
    let () = state.connect_header(rwtxn, header)?;
    // Withdrawal bundles are constructed AFTER the block's
    // transactions, so that txs in this block can still cancel
    // withdrawal outputs before they are selected.
    let () = state.maybe_create_withdrawal_bundle(rwtxn)?;
    let accumulator = state.get_accumulator(rwtxn)?;
    // TODO: are these needed?
    {
        let () = archive.put_header(rwtxn, header)?;
        let () = archive.put_body(rwtxn, block_hash, body)?;
    }
    let () = archive.put_accumulator(rwtxn, block_hash, &accumulator)?;
    // A mempool transaction that spends an input this block spends under
    // another txid is a double spend, so it goes with its children. The
    // block's own transactions leave their children behind, because a child of
    // a confirmed parent spends a confirmed output.
    for transaction in &body.transactions {
        let _evicted: Vec<_> =
            mempool.evict_decision_claim_conflicts(rwtxn, transaction)?;
        let txid = transaction.txid();
        for (outpoint, _) in &transaction.inputs {
            match mempool.spender(rwtxn, outpoint)? {
                Some(spender) if spender != txid => {
                    let () = mempool.delete(rwtxn, spender)?;
                }
                Some(_) | None => (),
            }
        }
    }
    for transaction in &body.transactions {
        let () = mempool.delete_confirmed(rwtxn, transaction.txid())?;
    }
    let () = mempool.regenerate_proofs(rwtxn, &accumulator)?;
    Ok(())
}

pub(in crate::node) fn disconnect_tip_(
    rwtxn: &mut RwTxn<'_>,
    archive: &Archive,
    mempool: &MemPool,
    state: &State,
) -> Result<(), Error> {
    let tip_block_hash =
        state.try_get_tip(rwtxn)?.ok_or(state::Error::NoTip)?;
    let tip_header = archive.get_header(rwtxn, tip_block_hash)?;
    let tip_body = archive.get_body(rwtxn, tip_block_hash)?;
    let height = state.try_get_height(rwtxn)?.ok_or(state::Error::NoTip)?;
    let two_way_peg_data = {
        let last_applied_deposit_block = state
            .deposit_blocks
            .rev_iter(rwtxn)
            .map_err(DbError::from)?
            .find_map(|(_, (block_hash, applied_height))| {
                if applied_height < height {
                    Ok(Some((block_hash, applied_height)))
                } else {
                    Ok(None)
                }
            })
            .map_err(DbError::from)?;
        let last_applied_withdrawal_bundle_event_block = state
            .withdrawal_bundle_event_blocks
            .rev_iter(rwtxn)
            .map_err(DbError::from)?
            .find_map(|(_, (block_hash, applied_height))| {
                if applied_height < height {
                    Ok(Some((block_hash, applied_height)))
                } else {
                    Ok(None)
                }
            })
            .map_err(DbError::from)?;
        let start_block_hash = match (
            last_applied_deposit_block,
            last_applied_withdrawal_bundle_event_block,
        ) {
            (None, None) => None,
            (Some((block_hash, _)), None) | (None, Some((block_hash, _))) => {
                Some(block_hash)
            }
            (
                Some((deposit_block, deposit_block_applied_height)),
                Some((
                    withdrawal_event_block,
                    withdrawal_event_block_applied_height,
                )),
            ) => {
                match deposit_block_applied_height
                    .cmp(&withdrawal_event_block_applied_height)
                {
                    Ordering::Less => Some(withdrawal_event_block),
                    Ordering::Greater => Some(deposit_block),
                    Ordering::Equal => {
                        if archive.is_main_descendant(
                            rwtxn,
                            withdrawal_event_block,
                            deposit_block,
                        )? {
                            Some(withdrawal_event_block)
                        } else {
                            assert!(archive.is_main_descendant(
                                rwtxn,
                                deposit_block,
                                withdrawal_event_block
                            )?);
                            Some(deposit_block)
                        }
                    }
                }
            }
        };
        let block_infos: Vec<_> = archive
            .main_ancestors(rwtxn, tip_header.prev_main_hash)
            .take_while(|ancestor| {
                Ok(Some(ancestor) != start_block_hash.as_ref())
            })
            .filter_map(|ancestor| {
                let block_info =
                    archive.get_main_block_info(rwtxn, &ancestor)?;
                if block_info.events.is_empty() {
                    Ok(None)
                } else {
                    let height = archive.get_main_height(rwtxn, ancestor)?;
                    let block_info = mainchain::two_way_peg_data::BlockInfo {
                        height,
                        inner: block_info,
                    };
                    Ok(Some((ancestor, block_info)))
                }
            })
            .collect()?;
        mainchain::TwoWayPegData {
            block_info: block_infos.into_iter().rev().collect(),
        }
    };
    let () = state.disconnect_withdrawal_bundle_creation(rwtxn)?;
    let () = state.disconnect_tip_body(rwtxn, &tip_header, &tip_body)?;
    let () = state.disconnect_two_way_peg_data(rwtxn, &two_way_peg_data)?;
    let () = state.disconnect_tip_header(rwtxn, &tip_header)?;
    // TODO: revert accumulator only necessary because rustreexo does not
    // support undo yet
    {
        match state.try_get_tip(rwtxn)? {
            Some(new_tip) => {
                let accumulator = archive.get_accumulator(rwtxn, new_tip)?;
                let () = state
                    .utreexo_accumulator
                    .put(rwtxn, &(), &accumulator)
                    .map_err(DbError::from)?;
            }
            None => {
                state
                    .utreexo_accumulator
                    .delete(rwtxn, &())
                    .map_err(DbError::from)?;
            }
        };
    }
    for transaction in tip_body.authorized_transactions().iter().rev() {
        match mempool.put_disconnected(rwtxn, transaction) {
            Ok(()) => (),
            Err(mempool::Error::DecisionAlreadyClaimedInMempool(_)) => {
                let _evicted: Vec<_> = mempool.evict_decision_claim_conflicts(
                    rwtxn,
                    &transaction.transaction,
                )?;
                mempool.put_disconnected(rwtxn, transaction)?;
            }
            Err(err) => return Err(err.into()),
        }
    }
    let accumulator = state.get_accumulator(rwtxn)?;
    mempool.regenerate_proofs(rwtxn, &accumulator)?;
    Ok(())
}

/// A peer sends a transaction this node refuses. The peer is at fault, so the
/// net task must stay up. A database error belongs to this node, and it must
/// stop the task.
fn is_fatal_peer_tx_error(err: &Error) -> bool {
    match err {
        // The transaction spends an output no source holds, so no proof exists
        // for it. A mempool transaction that drops out leaves its children in
        // that state.
        Error::State(_) => false,
        Error::MemPool(err) => !matches!(
            err,
            mempool::Error::DecisionAlreadyClaimedInMempool(_)
                | mempool::Error::TooManyAncestors { .. }
                | mempool::Error::UtxoDoubleSpent
        ),
        _ => true,
    }
}

/// Re-org to the specified tip, if it is better than the current tip.
/// The new tip block and all ancestor blocks must exist in the node's archive.
/// A result of `Ok(true)` indicates a successful re-org.
/// A result of `Ok(false)` indicates that no re-org was attempted.
// a state error means a peer sent an invalid block; it must not be fatal
fn reorg_to_tip<ThreadLocalStorage>(
    env: &sneed::Env<ThreadLocalStorage>,
    archive: &Archive,
    batch_verification_ctxt: &BatchVerificationContext,
    mempool: &MemPool,
    state: &State,
    new_tip: Tip,
) -> Result<bool, Error> {
    let mut rwtxn = env.write_txn().map_err(EnvError::from)?;
    let tip_height = state.try_get_height(&rwtxn)?;
    let tip = state
        .try_get_tip(&rwtxn)?
        .map(|tip_hash| {
            let bmm_verification =
                archive.get_best_main_verification(&rwtxn, tip_hash)?;
            Ok::<_, Error>(Tip {
                block_hash: tip_hash,
                main_block_hash: bmm_verification,
            })
        })
        .transpose()?;
    if let Some(tip) = tip {
        // check that new tip is better than current tip
        if archive.better_tip(&rwtxn, tip, new_tip)? != Some(new_tip) {
            tracing::debug!(
                ?tip,
                ?new_tip,
                "New tip is not better than current tip"
            );
            return Ok(false);
        }
    }
    let common_ancestor = if let Some(tip) = tip {
        archive.last_common_ancestor(
            &rwtxn,
            tip.block_hash,
            new_tip.block_hash,
        )?
    } else {
        None
    };
    // Check that all necessary bodies exist before disconnecting tip
    let blocks_to_apply: NonEmpty<(Header, Body)> = {
        let header = archive.get_header(&rwtxn, new_tip.block_hash)?;
        let body = archive.get_body(&rwtxn, new_tip.block_hash)?;
        let ancestors = if let Some(prev_side_hash) = header.prev_side_hash {
            archive
                .ancestors(&rwtxn, prev_side_hash)
                .take_while(|block_hash| {
                    Ok(common_ancestor.is_none_or(|common_ancestor| {
                        *block_hash != common_ancestor
                    }))
                })
                .map(|block_hash| {
                    let header = archive.get_header(&rwtxn, block_hash)?;
                    let body = archive.get_body(&rwtxn, block_hash)?;
                    Ok((header, body))
                })
                .collect()?
        } else {
            Vec::new()
        };
        NonEmpty {
            head: (header, body),
            tail: ancestors,
        }
    };
    // Disconnect tip until common ancestor is reached
    if let Some(tip_height) = tip_height {
        let common_ancestor_height =
            if let Some(common_ancestor) = common_ancestor {
                Some(archive.get_height(&rwtxn, common_ancestor)?)
            } else {
                None
            };
        tracing::debug!(
            ?tip,
            ?tip_height,
            ?common_ancestor,
            ?common_ancestor_height,
            "Disconnecting tip until common ancestor is reached"
        );
        let disconnects =
            if let Some(common_ancestor_height) = common_ancestor_height {
                tip_height - common_ancestor_height
            } else {
                tip_height + 1
            };
        for _ in 0..disconnects {
            let () = disconnect_tip_(&mut rwtxn, archive, mempool, state)?;
        }
    }
    {
        let tip_hash = state.try_get_tip(&rwtxn)?;
        assert_eq!(tip_hash, common_ancestor);
    }
    let mut two_way_peg_data_batch: Vec<_> = {
        let common_ancestor_header =
            if let Some(common_ancestor) = common_ancestor {
                Some(archive.get_header(&rwtxn, common_ancestor)?)
            } else {
                None
            };
        let common_ancestor_prev_main_hash =
            common_ancestor_header.map(|header| header.prev_main_hash);
        archive
            .main_ancestors(&rwtxn, blocks_to_apply.head.0.prev_main_hash)
            .take_while(|ancestor| {
                Ok(Some(ancestor) != common_ancestor_prev_main_hash.as_ref())
            })
            .map(|ancestor| {
                let height = archive.get_main_height(&rwtxn, ancestor)?;
                let block_info =
                    archive.get_main_block_info(&rwtxn, &ancestor)?;
                let block_info = mainchain::two_way_peg_data::BlockInfo {
                    height,
                    inner: block_info,
                };
                Ok((ancestor, block_info))
            })
            .collect()?
    };
    // Apply blocks until new tip is reached
    for (header, body) in blocks_to_apply.into_iter().rev() {
        let two_way_peg_data = {
            let mut two_way_peg_data = mainchain::TwoWayPegData::default();
            'fill_2wpd: while let Some((block_hash, block_info)) =
                two_way_peg_data_batch.pop()
            {
                two_way_peg_data.block_info.replace(block_hash, block_info);
                if block_hash == header.prev_main_hash {
                    break 'fill_2wpd;
                }
            }
            two_way_peg_data
        };
        let () = match connect_tip_(
            &mut rwtxn,
            archive,
            batch_verification_ctxt,
            mempool,
            state,
            &header,
            &body,
            &two_way_peg_data,
        ) {
            Ok(()) => (),
            Err(err) => {
                if !err.is_fatal() {
                    // The stored body for this block failed validation (e.g. a peer
                    // supplied a body whose contents do not match the header's merkle
                    // root). Abort the reorg and discard the invalid body from the
                    // archive so that the block is reported missing again and the real
                    // body is re-requested, instead of the archive staying poisoned.
                    drop(rwtxn);
                    let mut rwtxn = env.write_txn()?;
                    let () = archive.delete_body(
                        &mut rwtxn,
                        header.hash(),
                        &body,
                    )?;
                    rwtxn.commit()?;
                }
                return Err(err);
            }
        };
        let new_tip_hash = state.try_get_tip(&rwtxn)?.unwrap();
        let bmm_verification =
            archive.get_best_main_verification(&rwtxn, new_tip_hash)?;
        let new_tip = Tip {
            block_hash: new_tip_hash,
            main_block_hash: bmm_verification,
        };
        if let Some(tip) = tip
            && archive.better_tip(&rwtxn, tip, new_tip)? != Some(new_tip)
        {
            continue;
        }
        rwtxn.commit().map_err(RwTxnError::from)?;
        tracing::info!("synced to tip: {}", new_tip.block_hash);
        rwtxn = env.write_txn().map_err(EnvError::from)?;
    }
    let tip = state.try_get_tip(&rwtxn)?;
    assert_eq!(tip, Some(new_tip.block_hash));
    rwtxn.commit().map_err(RwTxnError::from)?;
    tracing::info!("synced to tip: {}", new_tip.block_hash);
    Ok(true)
}

#[derive(Clone)]
struct NetTaskContext {
    env: sneed::Env<heed::WithoutTls>,
    archive: Archive,
    mainchain_task: MainchainTaskHandle,
    mempool: MemPool,
    net: Net,
    state: State,
}

/// Message indicating a tip that is ready to reorg to, with the address of the
/// peer connection that caused the request, if it originated from a peer.
/// If the request originates from this node, then the socket address is
/// None.
/// An optional oneshot sender can be used receive the result of attempting
/// to reorg to the new tip, on the corresponding oneshot receiver.
type NewTipReadyMessage =
    (Tip, Option<SocketAddr>, Option<oneshot::Sender<bool>>);

struct NetTask {
    ctxt: NetTaskContext,
    /// Receive a request to forward to the mainchain task, with the address of
    /// the peer connection that caused the request, and the peer state ID of
    /// the request
    forward_mainchain_task_request_rx:
        UnboundedReceiver<(mainchain_task::Request, SocketAddr, PeerStateId)>,
    /// Push a request to forward to the mainchain task, with the address of
    /// the peer connection that caused the request, and the peer state ID of
    /// the request
    forward_mainchain_task_request_tx:
        UnboundedSender<(mainchain_task::Request, SocketAddr, PeerStateId)>,
    mainchain_task_event_rx: UnboundedReceiver<mainchain_task::Event>,
    /// Receive a tip that is ready to reorg to, with the address of the peer
    /// connection that caused the request, if it originated from a peer.
    /// If the request originates from this node, then the socket address is
    /// None.
    /// An optional oneshot sender can be used receive the result of attempting
    /// to reorg to the new tip, on the corresponding oneshot receiver.
    new_tip_ready_rx: UnboundedReceiver<NewTipReadyMessage>,
    /// Push a tip that is ready to reorg to, with the address of the peer
    /// connection that caused the request, if it originated from a peer.
    /// If the request originates from this node, then the socket address is
    /// None.
    /// An optional oneshot sender can be used receive the result of attempting
    /// to reorg to the new tip, on the corresponding oneshot receiver.
    new_tip_ready_tx: UnboundedSender<NewTipReadyMessage>,
    peer_info_rx: PeerInfoRx,
    rng: ChaCha20Rng,
}

impl NetTask {
    fn handle_response(
        ctxt: &NetTaskContext,
        // Attempt to switch to a descendant tip once a body has been
        // stored, if all other ancestor bodies are available.
        // Each descendant tip maps to the peers that sent that tip.
        descendant_tips: &mut HashMap<
            crate::types::BlockHash,
            HashMap<Tip, HashSet<SocketAddr>>,
        >,
        new_tip_ready_tx: &UnboundedSender<NewTipReadyMessage>,
        addr: SocketAddr,
        resp: PeerResponse,
        req: PeerRequest,
    ) -> Result<(), Error> {
        tracing::debug!(?req, ?resp, "starting response handler");
        match (req, resp) {
            (
                PeerRequest::GetBlock(
                    req @ peer_message::GetBlockRequest {
                        block_hash,
                        descendant_tip: Some(descendant_tip),
                        ancestor,
                        height: _,
                        peer_state_id: Some(peer_state_id),
                    },
                ),
                ref resp @ PeerResponse::Block(ref block),
            ) => {
                if block.header.hash() != block_hash {
                    // Invalid response
                    tracing::warn!(
                        %addr,
                        ?req,
                        ?resp,
                        "Invalid response from peer; unexpected block hash"
                    );
                    let () = ctxt.net.remove_active_peer(addr);
                    return Ok::<_, Error>(());
                }
                {
                    let mut rwtxn =
                        ctxt.env.write_txn().map_err(EnvError::from)?;
                    let () = ctxt.archive.put_body(
                        &mut rwtxn,
                        block_hash,
                        &block.body,
                    )?;
                    rwtxn.commit().map_err(RwTxnError::from)?;
                }
                // Notify the peer connection if all requested block bodies are
                // now available
                {
                    let rotxn = ctxt.env.read_txn().map_err(EnvError::from)?;
                    let ancestor_height = if let Some(ancestor) = ancestor {
                        Some(ctxt.archive.get_height(&rotxn, ancestor)?)
                    } else {
                        None
                    };
                    let earliest_missing_body = ctxt
                        .archive
                        .iter_missing_bodies(
                            &rotxn,
                            block_hash,
                            ancestor_height.map_or(0, |height| height + 1),
                        )
                        .next()?;
                    if let Some(earliest_missing_body) = earliest_missing_body {
                        descendant_tips
                            .entry(earliest_missing_body)
                            .or_default()
                            .entry(descendant_tip)
                            .or_default()
                            .insert(addr);
                    } else {
                        let message = PeerConnectionMessage::BodiesAvailable(
                            peer_state_id,
                        );
                        let _: bool =
                            ctxt.net.push_internal_message(message, addr);
                    }
                }
                // Check if any new tips can be applied,
                // and send new tip ready if so
                {
                    let rotxn = ctxt.env.read_txn().map_err(EnvError::from)?;
                    let tip = ctxt
                        .state
                        .try_get_tip(&rotxn)?
                        .map(|tip_hash| {
                            let bmm_verification = ctxt
                                .archive
                                .get_best_main_verification(&rotxn, tip_hash)?;
                            Ok::<_, Error>(Tip {
                                block_hash: tip_hash,
                                main_block_hash: bmm_verification,
                            })
                        })
                        .transpose()?;
                    // Find the BMM verification that is an ancestor of
                    // `main_descendant_tip`
                    let main_block_hash = ctxt
                        .archive
                        .get_bmm_results(&rotxn, block_hash)?
                        .into_iter()
                        .map(Result::<_, Error>::Ok)
                        .transpose_into_fallible()
                        .find_map(|(main_block_hash, bmm_result)| {
                            match bmm_result {
                                BmmResult::Failed => Ok(None),
                                BmmResult::Verified => {
                                    if ctxt.archive.is_main_descendant(
                                        &rotxn,
                                        main_block_hash,
                                        descendant_tip.main_block_hash,
                                    )? {
                                        Ok(Some(main_block_hash))
                                    } else {
                                        Ok(None)
                                    }
                                }
                            }
                        })?
                        .unwrap();
                    let block_tip = Tip {
                        block_hash,
                        main_block_hash,
                    };

                    if block.header.prev_side_hash
                        == tip.map(|tip| tip.block_hash)
                    {
                        tracing::trace!(
                            ?block_tip,
                            %addr,
                            "sending new tip ready, originating from peer"
                        );

                        let () = new_tip_ready_tx
                            .unbounded_send((block_tip, Some(addr), None))
                            .map_err(|err| {
                                Error::SendNewTipReady(err.into_send_error())
                            })?;
                    }
                    let Some(block_descendant_tips) =
                        descendant_tips.remove(&block_hash)
                    else {
                        return Ok(());
                    };
                    for (descendant_tip, sources) in block_descendant_tips {
                        let common_ancestor_height = if let Some(tip) = tip
                            && let Some(common_ancestor) =
                                ctxt.archive.last_common_ancestor(
                                    &rotxn,
                                    descendant_tip.block_hash,
                                    tip.block_hash,
                                )? {
                            Some(
                                ctxt.archive
                                    .get_height(&rotxn, common_ancestor)?,
                            )
                        } else {
                            None
                        };
                        let earliest_missing_body = ctxt
                            .archive
                            .iter_missing_bodies(
                                &rotxn,
                                descendant_tip.block_hash,
                                common_ancestor_height
                                    .map_or(0, |height| height + 1),
                            )
                            .next()?;
                        // If a better tip is ready, send a notification
                        'better_tip: {
                            let next_tip = if let Some(earliest_missing_body) =
                                earliest_missing_body
                            {
                                descendant_tips
                                    .entry(earliest_missing_body)
                                    .or_default()
                                    .entry(descendant_tip)
                                    .or_default()
                                    .extend(sources.iter().cloned());

                                // Parent of the earlist missing body
                                ctxt.archive
                                    .get_header(&rotxn, earliest_missing_body)?
                                    .prev_side_hash
                                    .map(|tip_hash| {
                                        let bmm_verification = ctxt
                                            .archive
                                            .get_best_main_verification(
                                                &rotxn, tip_hash,
                                            )?;
                                        Ok::<_, Error>(Tip {
                                            block_hash: tip_hash,
                                            main_block_hash: bmm_verification,
                                        })
                                    })
                                    .transpose()?
                            } else {
                                Some(descendant_tip)
                            };
                            let Some(next_tip) = next_tip else {
                                break 'better_tip;
                            };
                            if let Some(tip) = tip
                                && ctxt
                                    .archive
                                    .better_tip(&rotxn, tip, next_tip)?
                                    != Some(next_tip)
                            {
                                break 'better_tip;
                            } else {
                                tracing::debug!(
                                    new_tip = ?next_tip,
                                    "sending new tip ready to sources"
                                );
                                for addr in sources {
                                    tracing::trace!(%addr, new_tip = ?next_tip, "sending new tip ready");
                                    let () = new_tip_ready_tx
                                        .unbounded_send((
                                            next_tip,
                                            Some(addr),
                                            None,
                                        ))
                                        .map_err(|err| {
                                            Error::SendNewTipReady(
                                                err.into_send_error(),
                                            )
                                        })?;
                                }
                            }
                        }
                    }
                }
                Ok(())
            }
            (
                PeerRequest::GetBlock(peer_message::GetBlockRequest {
                    block_hash: req_block_hash,
                    descendant_tip: Some(_),
                    ancestor: _,
                    height: _,
                    peer_state_id: Some(_),
                }),
                PeerResponse::NoBlock {
                    block_hash: resp_block_hash,
                },
            ) if req_block_hash == resp_block_hash => Ok(()),
            (
                PeerRequest::GetHeaders(
                    ref req @ peer_message::GetHeadersRequest {
                        ref start,
                        end,
                        height: Some(height),
                        peer_state_id: Some(peer_state_id),
                    },
                ),
                PeerResponse::Headers(headers),
            ) => {
                // check that the end header is as requested
                let Some(end_header) = headers.last() else {
                    tracing::warn!(%addr, ?req, "Invalid response from peer; missing end header");
                    let () = ctxt.net.remove_active_peer(addr);
                    return Ok(());
                };
                let end_header_hash = end_header.hash();
                if end_header_hash != end {
                    tracing::warn!(%addr, ?req, ?end_header,"Invalid response from peer; unexpected end header");
                    let () = ctxt.net.remove_active_peer(addr);
                    return Ok(());
                }
                // Must be at least one header due to previous check
                let start_hash = headers.first().unwrap().prev_side_hash;
                // check that the first header is after a start block
                if let Some(start_hash) = start_hash
                    && !start.contains(&start_hash)
                {
                    tracing::warn!(%addr, ?req, %start_hash, "Invalid response from peer; invalid start hash");
                    let () = ctxt.net.remove_active_peer(addr);
                    return Ok(());
                }
                // check that the end header height is as expected
                {
                    let rotxn = ctxt.env.read_txn().map_err(EnvError::from)?;
                    let start_height = if let Some(start_hash) = start_hash {
                        Some(ctxt.archive.get_height(&rotxn, start_hash)?)
                    } else {
                        None
                    };
                    let end_height = match start_height {
                        Some(start_height) => {
                            start_height + headers.len() as u32
                        }
                        None => headers.len() as u32 - 1,
                    };
                    if end_height != height {
                        tracing::warn!(%addr, ?req, ?start_hash, "Invalid response from peer; invalid end height");
                        let () = ctxt.net.remove_active_peer(addr);
                        return Ok(());
                    }
                }
                // check that headers are sequential based on prev_side_hash,
                // and no header builds on an invalidated block.
                {
                    let rotxn = ctxt.env.read_txn().map_err(EnvError::from)?;
                    let mut prev_side_hash = start_hash;
                    for header in &headers {
                        if header.prev_side_hash != prev_side_hash {
                            tracing::warn!(%addr, ?req, ?headers,"Invalid response from peer; non-sequential headers");
                            let () = ctxt.net.remove_active_peer(addr);
                            return Ok(());
                        }
                        if ctxt
                            .archive
                            .invalidated_block(&rotxn, &header.hash())?
                        {
                            tracing::warn!(%addr, ?req, ?headers,"Invalid response from peer; invalidated block header");
                            let () = ctxt.net.remove_active_peer(addr);
                            return Ok(());
                        }
                        prev_side_hash = Some(header.hash());
                    }
                }
                // Store new headers
                let () = tokio::task::block_in_place(|| {
                    let mut rwtxn =
                        ctxt.env.write_txn().map_err(EnvError::from)?;
                    for header in &headers {
                        let block_hash = header.hash();
                        if ctxt
                            .archive
                            .try_get_header(&rwtxn, block_hash)?
                            .is_none()
                        {
                            if let Some(parent) = header.prev_side_hash
                                && ctxt
                                    .archive
                                    .try_get_header(&rwtxn, parent)?
                                    .is_none()
                            {
                                break;
                            } else {
                                ctxt.archive.put_header(&mut rwtxn, header)?;
                            }
                        }
                    }
                    rwtxn.commit().map_err(RwTxnError::from)?;
                    Ok::<_, Error>(())
                })?;
                // Notify peer connection that headers are available
                let message = PeerConnectionMessage::Headers(peer_state_id);
                let _: bool = ctxt.net.push_internal_message(message, addr);
                Ok(())
            }
            (
                PeerRequest::GetHeaders(peer_message::GetHeadersRequest {
                    start: _,
                    end,
                    height: _,
                    peer_state_id: _,
                }),
                PeerResponse::NoHeader { block_hash },
            ) if end == block_hash => Ok(()),
            (
                PeerRequest::PushTransaction(
                    peer_message::PushTransactionRequest { transaction: _ },
                ),
                PeerResponse::TransactionAccepted(_),
            ) => Ok(()),
            (
                PeerRequest::PushTransaction(
                    peer_message::PushTransactionRequest { transaction: _ },
                ),
                PeerResponse::TransactionRejected(_),
            ) => Ok(()),
            (
                req @ (PeerRequest::GetBlock { .. }
                | PeerRequest::GetHeaders { .. }
                | PeerRequest::PushTransaction { .. }),
                resp,
            ) => {
                // Invalid response
                tracing::warn!(%addr, ?req, ?resp,"Invalid response from peer");
                let () = ctxt.net.remove_active_peer(addr);
                Ok(())
            }
        }
    }

    fn handle_mainchain_block_event(
        ctxt: &NetTaskContext,
        _event: MainchainBlockEvent,
    ) -> Result<(), Error> {
        let mut rwtxn = ctxt.env.write_txn().map_err(EnvError::from)?;
        while let Some(state_tip) = ctxt.state.try_get_tip(&rwtxn)?
            && !ctxt
                .archive
                .side_tips()
                .sidechain_tips()
                .contains_key(&rwtxn, &state_tip)
                .map_err(archive::Error::from)?
        {
            // The full disconnect also reverts the 2WPD and the accumulator
            let () = disconnect_tip_(
                &mut rwtxn,
                &ctxt.archive,
                &ctxt.mempool,
                &ctxt.state,
            )?;
        }
        let best_side_tip = ctxt
            .archive
            .side_tips()
            .best_side_tip(&rwtxn)
            .map_err(archive::Error::from)?;
        rwtxn.commit()?;
        if let Some(best_side_tip) = best_side_tip {
            let best_side_tip = Tip {
                block_hash: best_side_tip.block_hash,
                main_block_hash: best_side_tip.info.main_block_hash,
            };
            match reorg_to_tip(
                &ctxt.env,
                &ctxt.archive,
                &ctxt.net.batch_verification_ctxt,
                &ctxt.mempool,
                &ctxt.state,
                best_side_tip,
            )
            .into_nested()?
            {
                Ok(completed_reorg) => {
                    let _: bool = completed_reorg;
                }
                Err(err) => tracing::warn!(
                    ?best_side_tip,
                    err = format!("{:#}", ErrorChain::new(&err)),
                    "rejecting invalid best side tip"
                ),
            }
        }
        Ok(())
    }

    fn handle_mainchain_task_response(
        ctxt: &NetTaskContext,
        mainchain_task_request_sources: &mut HashMap<
            mainchain_task::Request,
            HashSet<(SocketAddr, PeerStateId)>,
        >,
        response: mainchain_task::Response,
    ) -> Result<(), Error> {
        let request = (&response).into();
        match response {
            mainchain_task::Response::AncestorInfos(block_hash, res) => {
                let Some(sources) =
                    mainchain_task_request_sources.remove(&request)
                else {
                    return Ok(());
                };
                let res = res.map_err(Arc::new);
                for (addr, peer_state_id) in sources {
                    let message = match res {
                        Ok(true) => PeerConnectionMessage::MainchainAncestors(
                            peer_state_id,
                        ),
                        Ok(false) => {
                            PeerConnectionMessage::MainchainAncestorsError(
                                error::MainchainAncestors::BlockNotAvailable {
                                    block_hash,
                                },
                            )
                        }
                        Err(ref err) => {
                            PeerConnectionMessage::MainchainAncestorsError(
                                err.clone().into(),
                            )
                        }
                    };
                    let _: bool = ctxt.net.push_internal_message(message, addr);
                }
                Ok(())
            }
        }
    }

    #[inline]
    fn handle_mainchain_task_event(
        ctxt: &NetTaskContext,
        mainchain_task_request_sources: &mut HashMap<
            mainchain_task::Request,
            HashSet<(SocketAddr, PeerStateId)>,
        >,
        event: mainchain_task::Event,
    ) -> Result<(), Error> {
        match event {
            mainchain_task::Event::Block(event) => {
                Self::handle_mainchain_block_event(ctxt, event)
            }
            mainchain_task::Event::Response(resp) => {
                Self::handle_mainchain_task_response(
                    ctxt,
                    mainchain_task_request_sources,
                    resp,
                )
            }
        }
    }

    async fn run(mut self) -> Result<(), Error> {
        tracing::debug!("starting net task");
        #[derive(Debug)]
        enum MailboxItem {
            AcceptConnection(
                Result<
                    Option<SocketAddr>,
                    <net::error::AcceptConnection as Split>::Fatal,
                >,
            ),
            // Forward a mainchain task request, along with the peer that
            // caused the request, and the peer state ID of the request
            ForwardMainchainTaskRequest(
                mainchain_task::Request,
                SocketAddr,
                PeerStateId,
            ),
            MainchainTaskEvent(mainchain_task::Event),
            // Apply new tip from peer or self.
            // An optional oneshot sender can be used receive the result of
            // attempting to reorg to the new tip, on the corresponding oneshot
            // receiver.
            NewTipReady(Tip, Option<SocketAddr>, Option<oneshot::Sender<bool>>),
            PeerInfo(Option<(SocketAddr, Option<PeerConnectionInfo>)>),
            // Signal to reconnect to a peer
            ReconnectPeer(ResolvedPeerAddress),
            // The loop that dials known peers stopped on an error
            RedialKnownPeers(Box<net::Error>),
        }
        let accept_connections = stream::try_unfold((), |()| {
            let env = self.ctxt.env.clone();
            let net = self.ctxt.net.clone();
            let fut = async move {
                let maybe_socket_addr =
                    net.accept_incoming(env).await.into_nested()?;

                // / Return:
                // - The value to yield (maybe_socket_addr)
                // - The rng for the next iteration
                // Wrapped in Result and Option
                Result::<_, _>::Ok(Some((maybe_socket_addr, ())))
            };
            Box::pin(fut)
        })
        .filter_map(async |item| match item {
            Ok(Ok(maybe_socket_addr)) => Some(Ok(maybe_socket_addr)),
            Ok(Err(non_fatal_err)) => {
                // type the error explicitly
                let non_fatal_err:
                    <net::error::AcceptConnection as Split>::Jfyi =
                    non_fatal_err;
                tracing::error!(
                    "Failed to accept connection: {:#}",
                    ErrorChain::new(&non_fatal_err)
                );
                None
            }
            Err(fatal_err) => Some(Err(fatal_err)),
        })
        .map(MailboxItem::AcceptConnection);
        let forward_request_stream = self
            .forward_mainchain_task_request_rx
            .map(|(request, addr, peer_state_id)| {
                MailboxItem::ForwardMainchainTaskRequest(
                    request,
                    addr,
                    peer_state_id,
                )
            });
        let mainchain_task_event_stream = self
            .mainchain_task_event_rx
            .map(MailboxItem::MainchainTaskEvent);
        let new_tip_ready_stream =
            self.new_tip_ready_rx.map(|(block_hash, addr, resp_tx)| {
                MailboxItem::NewTipReady(block_hash, addr, resp_tx)
            });
        let peer_info_stream = StreamNotifyClose::new(self.peer_info_rx)
            .map(MailboxItem::PeerInfo);
        let (reconnect_peer_spawner, reconnect_peer_rx) = join_set::new();
        let reconnect_peer_stream = reconnect_peer_rx
            .map(|addr| MailboxItem::ReconnectPeer(addr.unwrap()));
        let redial_known_peers_stream = {
            const MIN_DELAY: Duration = Duration::from_secs(60);
            const MAX_DELAY: Duration = Duration::from_secs(600);
            let env = self.ctxt.env.clone();
            let net = self.ctxt.net.clone();
            stream::once(async move {
                net.redial_known_peers(env, MIN_DELAY, MAX_DELAY).await
            })
            .filter_map(async |res| res.err().map(Box::new))
            .map(MailboxItem::RedialKnownPeers)
        };
        let mut mailbox_stream = stream::select_all([
            accept_connections.boxed(),
            forward_request_stream.boxed(),
            mainchain_task_event_stream.boxed(),
            new_tip_ready_stream.boxed(),
            peer_info_stream.boxed(),
            reconnect_peer_stream.boxed(),
            redial_known_peers_stream.boxed(),
        ]);
        // Attempt to switch to a descendant tip once a body has been
        // stored, if all other ancestor bodies are available.
        // Each descendant tip maps to the peers that sent that tip.
        let mut descendant_tips = HashMap::<
            crate::types::BlockHash,
            HashMap<Tip, HashSet<SocketAddr>>,
        >::new();
        // Map associating mainchain task requests with the peer(s) that
        // caused the request, and the request peer state ID
        let mut mainchain_task_request_sources = HashMap::<
            mainchain_task::Request,
            HashSet<(SocketAddr, PeerStateId)>,
        >::new();
        // Backoff state for peers that this node dialed
        let mut reconnect_backoff =
            HashMap::<PeerAddress, ReconnectBackoff>::new();
        while let Some(mailbox_item) = mailbox_stream.next().await {
            tracing::trace!(?mailbox_item, "received new mailbox item");
            match mailbox_item {
                MailboxItem::AcceptConnection(res) => match res {
                    // We received a connection new incoming network connection, but no peer
                    // was added
                    Ok(None) => {
                        continue;
                    }
                    Ok(Some(addr)) => {
                        tracing::trace!(%addr, "accepted new incoming connection");
                    }
                    Err(fatal_err) => {
                        // explicitly type error
                        let fatal_err: <net::error::AcceptConnection as Split>::Fatal =
                            fatal_err;
                        tracing::error!(
                            "failed to accept connection: {:#}",
                            ErrorChain::new(&fatal_err)
                        );
                    }
                },
                MailboxItem::ForwardMainchainTaskRequest(
                    request,
                    peer,
                    peer_state_id,
                ) => {
                    if self.ctxt.mainchain_task.request(request).is_err() {
                        tracing::warn!(
                            ?request,
                            %peer,
                            "the mainchain task took no request"
                        );
                        continue;
                    }
                    mainchain_task_request_sources
                        .entry(request)
                        .or_default()
                        .insert((peer, peer_state_id));
                }
                MailboxItem::MainchainTaskEvent(event) => {
                    let () = Self::handle_mainchain_task_event(
                        &self.ctxt,
                        &mut mainchain_task_request_sources,
                        event,
                    )?;
                }
                MailboxItem::NewTipReady(new_tip, addr, resp_tx) => {
                    let reorg_result = task::block_in_place(|| {
                        {
                            let rotxn = self
                                .ctxt
                                .env
                                .read_txn()
                                .map_err(|err| Error::DbEnv(err.into()))?;
                            if !self
                                .ctxt
                                .archive
                                .side_tips()
                                .sidechain_tips()
                                .contains_key(&rotxn, &new_tip.block_hash)
                                .map_err(archive::Error::from)?
                            {
                                return Ok(false);
                            }
                            let side_tips_tip = self
                                .ctxt
                                .archive
                                .side_tips()
                                .get_mainchain_tip(&rotxn)
                                .map_err(archive::Error::from)?;
                            if !self.ctxt.archive.is_main_descendant(
                                &rotxn,
                                new_tip.main_block_hash,
                                side_tips_tip.block_hash(),
                            )? {
                                return Ok(false);
                            }
                        }
                        reorg_to_tip(
                            &self.ctxt.env,
                            &self.ctxt.archive,
                            &self.ctxt.net.batch_verification_ctxt,
                            &self.ctxt.mempool,
                            &self.ctxt.state,
                            new_tip,
                        )
                    })
                    .into_nested()?;
                    let reorg_applied = match reorg_result {
                        Ok(applied) => applied,
                        // an invalid block must not kill the net task; drop the
                        // peer and keep running
                        Err(err) => {
                            tracing::warn!(
                                ?new_tip,
                                ?addr,
                                err = format!("{:#}", ErrorChain::new(&err)),
                                "rejecting invalid tip from peer"
                            );
                            if let Some(addr) = addr {
                                let () = self.ctxt.net.remove_active_peer(addr);
                            }
                            false
                        }
                    };
                    if let Some(resp_tx) = resp_tx {
                        let () = resp_tx
                            .send(reorg_applied)
                            .map_err(|_| Error::SendReorgResultOneshot)?;
                    }
                }
                MailboxItem::PeerInfo(None) => {
                    return Err(Error::PeerInfoRxClosed);
                }
                MailboxItem::PeerInfo(Some((addr, None))) => {
                    // peer connection is closed, remove it
                    tracing::warn!(%addr, "Connection to peer closed");
                    let () = self.ctxt.net.remove_active_peer(addr);
                }
                MailboxItem::PeerInfo(Some((addr, Some(peer_info)))) => {
                    tracing::trace!(%addr, ?peer_info, "mailbox item: received PeerInfo");
                    match peer_info {
                        PeerConnectionInfo::Error {
                            err,
                            resolved_peer_addr,
                        } => {
                            let err_msg =
                                format!("{:#}", ErrorChain::new(&err));
                            tracing::error!(
                                %addr,
                                err = err_msg,
                                "Peer connection error",
                            );
                            let conn_info =
                                self.ctxt.net.try_with_active_peer_connection(
                                    addr,
                                    |conn_handle| {
                                        (
                                            conn_handle
                                                .received_msg_successfully(),
                                            conn_handle.is_outbound(),
                                        )
                                    },
                                );
                            let () = self.ctxt.net.remove_active_peer(addr);
                            if err.is_ban() {
                                let mut rwtxn = self
                                    .ctxt
                                    .env
                                    .write_txn()
                                    .map_err(EnvError::from)?;
                                let () =
                                    self.ctxt.net.ban_peer(&mut rwtxn, addr)?;
                                rwtxn.commit().map_err(RwTxnError::from)?;
                                continue;
                            }
                            // A peer on another network never becomes useful,
                            // so it must not survive into the next start.
                            if err.is_bad_magic() {
                                let peer_address = resolved_peer_addr
                                    .as_peer_address()
                                    .to_owned();
                                let mut rwtxn = self
                                    .ctxt
                                    .env
                                    .write_txn()
                                    .map_err(EnvError::from)?;
                                let forgotten = self
                                    .ctxt
                                    .net
                                    .forget_peer(&mut rwtxn, &peer_address)?;
                                rwtxn.commit().map_err(RwTxnError::from)?;
                                if forgotten {
                                    tracing::warn!(
                                        %peer_address,
                                        "forgot peer: it runs another network"
                                    );
                                }
                                continue;
                            }
                            let Some((received_msg_successfully, outbound)) =
                                conn_info
                            else {
                                continue;
                            };
                            // Leave reconnecting to the side that dialed.
                            // The address of an inbound peer is its source
                            // address; dialing it back collides with the
                            // peer's own reconnect.
                            if !outbound {
                                tracing::debug!(
                                    %addr,
                                    "inbound peer disconnected, not reconnecting"
                                );
                                continue;
                            }
                            // A working connection that failed reconnects
                            // with backoff; a dial that failed before it was
                            // usable keeps the prompt retry, with jitter so
                            // that two nodes dialing each other at once
                            // ("already connected") stop retrying in step.
                            let (reconnect_addr, delay) =
                                if received_msg_successfully
                                    && err.may_reconnect()
                                {
                                    // Forget peers whose last reconnect is long
                                    // past, so the map stays bounded by the peers
                                    // that are failing now
                                    let now = Instant::now();
                                    reconnect_backoff.retain(|_, backoff| {
                                        backoff.last_dial.is_some_and(
                                            |last_dial| {
                                                now.saturating_duration_since(
                                                    last_dial,
                                                ) < ReconnectBackoff::STABLE
                                            },
                                        )
                                    });
                                    let delay = reconnect_backoff
                                        .entry(
                                            resolved_peer_addr
                                                .as_peer_address()
                                                .to_owned(),
                                        )
                                        .or_default()
                                        .next_delay(now, &mut self.rng);
                                    (resolved_peer_addr, delay)
                                } else if err.is_duplicate_connection()
                                    && err.may_reconnect()
                                {
                                    (
                                        resolved_peer_addr,
                                        ReconnectBackoff::jittered(
                                            &mut self.rng,
                                        ),
                                    )
                                } else if let (_, Some(next_addr)) =
                                    resolved_peer_addr
                                        .clone()
                                        .pop_first_ip_addr()
                                {
                                    (
                                        next_addr,
                                        ReconnectBackoff::jittered(
                                            &mut self.rng,
                                        ),
                                    )
                                } else if err.is_connect_timeout() {
                                    (
                                        resolved_peer_addr,
                                        ReconnectBackoff::jittered(
                                            &mut self.rng,
                                        ),
                                    )
                                } else {
                                    continue;
                                };
                            tracing::info!(
                                %addr,
                                ?delay,
                                "reconnecting to peer after delay"
                            );
                            reconnect_peer_spawner.spawn(async move {
                                tokio::time::sleep(delay).await;
                                reconnect_addr
                            });
                        }
                        PeerConnectionInfo::NeedMainchainAncestors {
                            main_hash,
                            peer_state_id,
                        } => {
                            let request =
                                mainchain_task::Request::AncestorInfos(
                                    main_hash,
                                );
                            let () = self
                                .forward_mainchain_task_request_tx
                                .unbounded_send((request, addr, peer_state_id))
                                .map_err(|_| {
                                    Error::ForwardMainchainTaskRequest
                                })?;
                        }
                        PeerConnectionInfo::NewTipReady(new_tip) => {
                            tracing::debug!(
                                ?new_tip,
                                %addr,
                                "mailbox item: received NewTipReady from peer, sending on channel"
                            );
                            self.new_tip_ready_tx
                                .unbounded_send((new_tip, Some(addr), None))
                                .map_err(|err| {
                                    Error::SendNewTipReady(
                                        err.into_send_error(),
                                    )
                                })?;
                        }
                        PeerConnectionInfo::NewTransaction(mut new_tx) => {
                            let mut rwtxn = self
                                .ctxt
                                .env
                                .write_txn()
                                .map_err(EnvError::from)?;
                            let unconfirmed =
                                self.ctxt.mempool.unconfirmed_outputs(
                                    &rwtxn,
                                    &new_tx.transaction,
                                )?;
                            let accepted =
                                match self.ctxt.state.regenerate_proof(
                                    &rwtxn,
                                    &unconfirmed,
                                    &mut new_tx.transaction,
                                ) {
                                    Ok(()) => self
                                        .ctxt
                                        .mempool
                                        .put(&mut rwtxn, &new_tx)
                                        .map_err(Error::from),
                                    Err(err) => Err(Error::from(err)),
                                };
                            match accepted {
                                Ok(()) => {
                                    rwtxn.commit().map_err(RwTxnError::from)?;
                                    // broadcast
                                    let () = self.ctxt.net.push_tx(
                                        HashSet::from_iter([addr]),
                                        &new_tx,
                                    );
                                }
                                Err(err) if !is_fatal_peer_tx_error(&err) => {
                                    // Drop the write, so the rows that `put`
                                    // wrote before it refused never land.
                                    drop(rwtxn);
                                    // A peer that relays a transaction back is
                                    // normal traffic, not a fault.
                                    tracing::debug!(
                                        %addr,
                                        "this node refuses a peer's \
                                         transaction: {err}"
                                    );
                                }
                                Err(err) => return Err(err),
                            }
                        }
                        PeerConnectionInfo::Response(boxed) => {
                            let (resp, req) = *boxed;
                            tracing::trace!(
                                resp = format!("{resp:#?}"),
                                req = format!("{req:#?}"),
                                "mail box: received PeerConnectionInfo::Response"
                            );
                            let () = tokio::task::block_in_place(|| {
                                Self::handle_response(
                                    &self.ctxt,
                                    &mut descendant_tips,
                                    &self.new_tip_ready_tx,
                                    addr,
                                    resp,
                                    req,
                                )
                            })?;
                        }
                    }
                }
                MailboxItem::ReconnectPeer(resolved_peer_address) => {
                    let peer_address =
                        resolved_peer_address.as_peer_address().to_owned();
                    match self.ctxt.net.connect_peer(
                        self.ctxt.env.clone(),
                        resolved_peer_address,
                    ) {
                        Ok(()) => (),
                        Err(err) => {
                            tracing::error!(
                                %peer_address,
                                "Failed to connect to peer: {:#}",
                                ErrorChain::new(&err)
                            )
                        }
                    }
                }
                MailboxItem::RedialKnownPeers(err) => {
                    return Err(Error::Net(err));
                }
            }
        }
        Ok(())
    }
}

/// Handle to the net task.
/// Task is aborted on drop.
#[derive(Clone)]
pub(super) struct NetTaskHandle {
    task: Arc<JoinHandle<()>>,
    /// Push a tip that is ready to reorg to, with the address of the peer
    /// connection that caused the request, if it originated from a peer.
    /// If the request originates from this node, then the socket address is
    /// None.
    /// An optional oneshot sender can be used receive the result of attempting
    /// to reorg to the new tip, on the corresponding oneshot receiver.
    new_tip_ready_tx: UnboundedSender<NewTipReadyMessage>,
}

impl NetTaskHandle {
    #[allow(clippy::too_many_arguments)]
    pub fn new<R>(
        runtime: &tokio::runtime::Runtime,
        env: sneed::Env<heed::WithoutTls>,
        archive: Archive,
        mainchain_task: MainchainTaskHandle,
        mainchain_task_event_rx: UnboundedReceiver<mainchain_task::Event>,
        mempool: MemPool,
        net: Net,
        peer_info_rx: PeerInfoRx,
        mut rng: R,
        state: State,
        fatal_error_tx: FatalErrorTx,
    ) -> Self
    where
        R: rand::CryptoRng,
    {
        let ctxt = NetTaskContext {
            env,
            archive,
            mainchain_task,
            mempool,
            net,
            state,
        };
        let (
            forward_mainchain_task_request_tx,
            forward_mainchain_task_request_rx,
        ) = mpsc::unbounded();
        let (new_tip_ready_tx, new_tip_ready_rx) = mpsc::unbounded();
        let task = NetTask {
            ctxt,
            forward_mainchain_task_request_tx,
            forward_mainchain_task_request_rx,
            mainchain_task_event_rx,
            new_tip_ready_tx: new_tip_ready_tx.clone(),
            new_tip_ready_rx,
            peer_info_rx,
            rng: ChaCha20Rng::from_rng(&mut rng),
        };
        let task = runtime.spawn(supervise("Net", task.run(), fatal_error_tx));
        NetTaskHandle {
            task: Arc::new(task),
            new_tip_ready_tx,
        }
    }

    /// Push a tip that is ready to reorg to, and await successful application.
    /// A result of Ok(true) indicates that the tip was applied and reorged
    /// to successfully.
    /// A result of Ok(false) indicates that the tip was not reorged to.
    pub async fn new_tip_ready_confirm(
        &self,
        new_tip: Tip,
    ) -> Result<bool, Error> {
        tracing::debug!(?new_tip, "sending new tip ready confirm");

        let (oneshot_tx, oneshot_rx) = oneshot::channel();
        let () = self
            .new_tip_ready_tx
            .unbounded_send((new_tip, None, Some(oneshot_tx)))
            .map_err(|err| Error::SendNewTipReady(err.into_send_error()))?;
        oneshot_rx.await.map_err(Error::ReceiveReorgResultOneshot)
    }
}

impl Drop for NetTaskHandle {
    // If only one reference exists (ie. within self), abort the net task.
    fn drop(&mut self) {
        // use `Arc::get_mut` since `Arc::into_inner` requires ownership of the
        // Arc, and cloning would increase the reference count
        if let Some(task) = Arc::get_mut(&mut self.task) {
            tracing::debug!("dropping net task handle, aborting task");
            task.abort()
        }
    }
}

#[cfg(test)]
mod reconnect_backoff_test {
    use std::time::{Duration, Instant};

    use rand::{SeedableRng, rngs::ChaCha20Rng};

    use super::ReconnectBackoff;

    // the delay doubles per failure in a row, stays within +25% jitter and is
    // capped; a connection that stayed up for STABLE starts over at BASE
    #[test]
    fn delay_doubles_caps_and_resets() {
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let mut backoff = ReconnectBackoff::default();
        let mut now = Instant::now();
        let mut base = ReconnectBackoff::BASE;
        for _ in 0..10 {
            let delay = backoff.next_delay(now, &mut rng);
            let want = base.min(ReconnectBackoff::MAX);
            assert!(
                delay >= want && delay <= want.mul_f64(1.25),
                "{delay:?} vs {want:?}"
            );
            // the next failure comes right after the reconnect it scheduled
            now += delay;
            base = base.saturating_mul(2);
        }
        let delay = backoff.next_delay(
            now + ReconnectBackoff::STABLE + Duration::from_secs(1),
            &mut rng,
        );
        assert!(
            delay <= ReconnectBackoff::BASE.mul_f64(1.25),
            "no reset: {delay:?}"
        );
    }

    // a failed dial is retried after BASE..=BASE+25%, never in a fixed step
    #[test]
    fn jittered_stays_within_a_quarter() {
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let delays: Vec<_> = (0..20)
            .map(|_| ReconnectBackoff::jittered(&mut rng))
            .collect();
        for d in &delays {
            assert!(
                *d >= ReconnectBackoff::BASE
                    && *d <= ReconnectBackoff::BASE.mul_f64(1.25)
            );
        }
        assert!(delays.windows(2).any(|w| w[0] != w[1]), "no jitter");
    }
}

#[cfg(test)]
mod test {
    use std::{
        collections::{HashMap, HashSet},
        net::Ipv4Addr,
        time::Duration,
    };
    use truthcoin_dc_types::authorization::BatchVerificationContext;

    use anyhow::Context;
    use bitcoin::hashes::Hash as _;
    use error_fatality::Fatality as _;
    use futures::channel::mpsc;
    use rand::SeedableRng as _;

    use crate::{
        archive::side_tips::BmmCommitment,
        net::{PeerConnectionInfo, make_server_endpoint},
        node::{
            Config, Node,
            net_task::{Error, NetTask, NetTaskContext},
        },
        state,
        types::{
            Accumulator, AccumulatorDiff, Body, Coinbase, Header, MerkleRoot,
            Network, OutPoint, OutPointKey, Output, OutputContent,
            PointedOutput, Transaction, hash,
            net::PeerConnectionStatus,
            proto::mainchain::{
                BlockHeaderInfo, BlockInfo, Event as MainchainBlockEvent,
                ValidatorClient,
            },
        },
        wallet::Wallet,
    };

    fn temp_node(
        runtime: &tokio::runtime::Runtime,
    ) -> anyhow::Result<(temp_dir::TempDir, Node)> {
        let temp_dir = temp_dir::TempDir::new()?;
        let channel =
            tonic::transport::Endpoint::from_static("http://127.0.0.1:1")
                .connect_lazy();
        let (fatal_error_tx, _fatal_error_rx) = mpsc::unbounded();
        let node = Node::new(
            Config {
                datadir: temp_dir.path().to_owned(),
                bind_addr: (Ipv4Addr::LOCALHOST, 0).into(),
                magic_bytes_override: None,
                network: Network::Regtest,
                add_peers: HashSet::new(),
                server_names: HashSet::new(),
                decision_config_testing: None,
            },
            ValidatorClient::new(channel),
            None,
            &mut rand::rng(),
            runtime,
            fatal_error_tx,
        )?;
        Ok((temp_dir, node))
    }

    /// Connect a mainchain tip, so that block selection finds its height
    fn connect_main_tip(node: &Node) -> anyhow::Result<()> {
        let header_info = BlockHeaderInfo {
            block_hash: bitcoin::BlockHash::from_byte_array([0xaa; 32]),
            prev_block_hash: bitcoin::BlockHash::all_zeros(),
            height: 0,
            work: bitcoin::Target::MAX.to_work(),
            timestamp: 0,
        };
        let mut rwtxn = node.env.write_txn()?;
        node.archive
            .put_main_header_info(&mut rwtxn, &header_info)?;
        node.archive.put_main_block_info(
            &mut rwtxn,
            header_info.block_hash,
            &BlockInfo::default(),
        )?;
        node.archive.side_tips().connect_mainchain_tip(
            &mut rwtxn,
            header_info,
            None,
        )?;
        rwtxn.commit()?;
        Ok(())
    }

    fn check_reputation_selection(
        start_reputation: f64,
        amounts: &[f64],
        expected_count: usize,
    ) -> anyhow::Result<()> {
        use crate::types::{
            Address, Body, Coinbase, Header, TransactionData,
            authorization::{SigningKey, authorize, get_address},
        };
        use bitcoin::hashes::Hash as _;

        let runtime = tokio::runtime::Runtime::new()?;
        runtime.block_on(async {
            let (_temp_dir, node) = temp_node(&runtime)?;
            let () = connect_main_tip(&node)?;
            node.task_handles.net.task.abort();
            while !node.task_handles.net.task.is_finished() {
                tokio::task::yield_now().await;
            }
            let key = SigningKey::new(&mut rand::rng());
            let sender = get_address(&(&key).into());
            let receiver = Address([3; 20]);
            let mut accumulator = Accumulator::default();
            let mut diff = AccumulatorDiff::default();
            let mut inputs = Vec::new();
            let mut rwtxn = node.env.write_txn()?;
            node.state.reputation().set_reputation(
                &mut rwtxn,
                &sender,
                start_reputation,
            )?;
            for index in 0..amounts.len() {
                let pointed = PointedOutput {
                    outpoint: OutPoint::Regular {
                        txid: [index as u8; 32].into(),
                        vout: 0,
                    },
                    output: Output {
                        address: sender,
                        content: OutputContent::Value(
                            bitcoin::Amount::from_sat(1_000),
                        ),
                    },
                };
                node.state.utxos.put(
                    &mut rwtxn,
                    &OutPointKey::from(&pointed.outpoint),
                    &pointed.output,
                )?;
                diff.insert((&pointed).into());
                inputs.push((pointed.outpoint, hash(&pointed)));
            }
            accumulator.apply_diff(diff)?;
            node.state.utreexo_accumulator.put(
                &mut rwtxn,
                &(),
                &accumulator,
            )?;
            rwtxn.commit()?;

            for (input, &amount) in inputs.into_iter().zip(amounts) {
                let tx = Transaction {
                    inputs: vec![input].into(),
                    outputs: vec![Output {
                        address: sender,
                        content: OutputContent::Value(
                            bitcoin::Amount::from_sat(900),
                        ),
                    }]
                    .into(),
                    data: Some(TransactionData::TransferReputation {
                        sender,
                        dest: receiver,
                        amount,
                    }),
                    ..Default::default()
                };
                let tx = authorize(rand::rng(), &[(sender, &key)], tx)?;
                node.submit_transaction(tx)?;
            }

            let (selected, fee) = node.get_transactions(100)?;
            let selected_count = selected.len();
            let transferred: f64 = selected
                .iter()
                .map(|tx| tx.transaction.transfer_reputation().unwrap().amount)
                .sum();
            let body = Body::new(
                selected.iter().cloned().map(Into::into).collect(),
                Coinbase::default(),
            );
            let merkle_root = Body::compute_merkle_root(
                &body.coinbase,
                &selected,
                &body.authorizations,
                &body.actor_proofs,
            )?;
            let mut header = Header {
                merkle_root,
                prev_side_hash: None,
                prev_main_hash: bitcoin::BlockHash::all_zeros(),
                roots: Vec::new(),
            };
            Body::modify_memforest(
                header.compute_coinbase_txid(),
                &[],
                &selected,
                &mut accumulator.0,
            )?;
            header.roots = accumulator.get_roots();
            let mut rwtxn = node.env.write_txn()?;
            node.state.apply_block(
                &mut rwtxn,
                &node.batch_verification_ctxt,
                &header,
                0,
                &body,
                &node.archive,
                1_700_000_000,
            )?;
            assert_eq!(selected_count, expected_count);
            assert_eq!(fee.to_sat(), 100 * expected_count as u64);
            assert_eq!(
                node.state.reputation().get_reputation(&rwtxn, &sender)?,
                start_reputation - transferred
            );
            assert_eq!(
                node.state.reputation().get_reputation(&rwtxn, &receiver)?,
                transferred
            );
            Ok(())
        })
    }

    #[test]
    fn block_selection_keeps_reputation_balance() -> anyhow::Result<()> {
        check_reputation_selection(1.0, &[0.75, 0.75], 1)
    }

    #[test]
    fn block_selection_keeps_funded_reputation_transfers() -> anyhow::Result<()>
    {
        check_reputation_selection(1.0, &[0.25, 0.25], 2)
    }

    #[test]
    fn block_selection_matches_decimal_reputation() -> anyhow::Result<()> {
        check_reputation_selection(0.3, &[0.1, 0.2], 1)
    }

    #[test]
    fn block_selection_skips_share_overflow() -> anyhow::Result<()> {
        check_trade_selection(false)
    }

    #[test]
    fn block_selection_skips_cost_overflow() -> anyhow::Result<()> {
        check_trade_selection(true)
    }

    fn check_trade_selection(cost_overflow: bool) -> anyhow::Result<()> {
        use crate::{
            state::{
                MarketBuilder,
                decisions::{Decision, DecisionId, DecisionType},
                markets::DimensionSpec,
            },
            types::{
                Authorization, AuthorizedTransaction, Body, Coinbase, Header,
                TxData,
                authorization::{self, SigningKey},
            },
        };
        use bitcoin::hashes::Hash as _;

        let runtime = tokio::runtime::Runtime::new()?;
        runtime.block_on(async {
            let (_temp_dir, node) = temp_node(&runtime)?;
            let () = connect_main_tip(&node)?;
            node.task_handles.net.task.abort();
            while !node.task_handles.net.task.is_finished() {
                tokio::task::yield_now().await;
            }
            let mut rng = rand::rng();
            let key = SigningKey::new(&mut rng);
            let trader = authorization::get_address(&(&key).into());
            let decision_id = DecisionId::new(true, 0, 0)?;
            let decision = Decision::new(
                trader.0,
                DecisionType::Binary,
                String::new(),
                String::new(),
                None,
                None,
                vec![],
            )?;
            let mut market = MarketBuilder::new("Market".into(), trader)
                .with_dimensions(vec![DimensionSpec::Single(decision_id)])
                .build(0, None, &HashMap::from([(decision_id, decision)]))?;
            if cost_overflow {
                market.liquidity_base_sats = 1;
                market.trading_fee = 1.0;
                market.shares.fill(0);
            } else {
                market.liquidity_base_sats = 10_000;
                market.shares = ndarray::Array1::from(vec![0, 20_000]);
            }
            let body = Body::new(vec![], Coinbase::default());
            let header = Header {
                merkle_root: Body::compute_merkle_root(
                    &body.coinbase,
                    &Vec::<crate::types::FilledTransaction>::new(),
                    &body.authorizations,
                    &body.actor_proofs,
                )?,
                prev_side_hash: None,
                prev_main_hash: bitcoin::BlockHash::all_zeros(),
                roots: Vec::new(),
            };
            let tip = header.hash();
            let mut rwtxn = node.env.write_txn()?;
            node.archive.put_header(&mut rwtxn, &header)?;
            node.state.connect_block(&mut rwtxn, &header, &body, 1000)?;
            node.state.markets().add_market(&mut rwtxn, &market)?;
            if !cost_overflow {
                node.state.markets().add_shares_to_account(
                    &mut rwtxn,
                    &crate::types::Address([9; 20]),
                    market.id,
                    1,
                    20_000,
                    0,
                )?;
            }
            let mut diff = AccumulatorDiff::default();
            let mut transactions = Vec::new();
            for (index, (shares, input_sats)) in
                [(i64::MAX, 1_000), (1, 10_000)].into_iter().enumerate()
            {
                let pointed = PointedOutput {
                    outpoint: OutPoint::Regular {
                        txid: hash(&index).into(),
                        vout: 0,
                    },
                    output: Output {
                        address: trader,
                        content: OutputContent::Value(
                            bitcoin::Amount::from_sat(input_sats),
                        ),
                    },
                };
                node.state.utxos.put(
                    &mut rwtxn,
                    &OutPointKey::from(&pointed.outpoint),
                    &pointed.output,
                )?;
                diff.insert((&pointed).into());
                transactions.push(Transaction {
                    inputs: vec![(pointed.outpoint, hash(&pointed))].into(),
                    data: Some(TxData::Trade {
                        market_id: market.id,
                        outcome_index: u32::from(index == 0 && !cost_overflow),
                        shares,
                        trader,
                        limit_sats: input_sats,
                        tx_pow_nonce: None,
                        prev_block_hash: tip,
                    }),
                    ..Default::default()
                });
            }
            let mut accumulator = node.state.get_accumulator(&rwtxn)?;
            accumulator.apply_diff(diff)?;
            node.state.utreexo_accumulator.put(
                &mut rwtxn,
                &(),
                &accumulator,
            )?;
            rwtxn.commit()?;
            let valid_txid = transactions[1].txid();
            for transaction in transactions {
                let authorization = Authorization {
                    verifying_key: (&key).into(),
                    signature: authorization::sign_tx(
                        &mut rng,
                        &key,
                        &transaction,
                    )?,
                };
                let mut transaction = AuthorizedTransaction {
                    transaction,
                    authorizations: vec![authorization],
                    actor_proof: None,
                };
                let mut rwtxn = node.env.write_txn()?;
                node.state.regenerate_proof(
                    &rwtxn,
                    &HashMap::new(),
                    &mut transaction.transaction,
                )?;
                node.state.validate_transaction(
                    &rwtxn,
                    &node.batch_verification_ctxt,
                    &HashMap::new(),
                    0,
                    &transaction,
                    &node.archive,
                )?;
                node.mempool.put(&mut rwtxn, &transaction)?;
                rwtxn.commit()?;
            }
            assert_eq!(node.get_all_transactions()?.len(), 2);
            {
                let mut rwtxn = node.env.write_txn()?;
                let result = if cost_overflow {
                    node.update_mempool_buy(&mut rwtxn, market.id, 0, i64::MAX)?;
                    node.update_mempool_buy(&mut rwtxn, market.id, 0, 1)
                } else {
                    node.update_mempool_buy(&mut rwtxn, market.id, 1, i64::MAX)
                };
                assert!(matches!(result, Err(crate::node::Error::State(error)) if matches!(*error, state::Error::InvalidTransaction { .. })));
            }
            for _ in 0..2 {
                let (selected, fee) = node.get_transactions(100)?;
                assert_eq!(selected.len(), 1);
                assert_eq!(selected[0].transaction.txid(), valid_txid);
                assert_eq!(
                    fee.to_sat(),
                    crate::math::trading::TRADE_MINER_FEE_SATS
                );
                assert_eq!(node.get_all_transactions()?.len(), 2);
            }
            Ok(())
        })
    }

    #[test]
    fn block_selection_skips_conflicting_votes() -> anyhow::Result<()> {
        use crate::{
            state::decisions::{Decision, DecisionId, DecisionType},
            types::{
                Authorization, AuthorizedTransaction, BallotItem, Body,
                Coinbase, Header, TxData,
                authorization::{self, SigningKey},
            },
        };
        use bitcoin::hashes::Hash as _;

        let runtime = tokio::runtime::Runtime::new()?;
        runtime.block_on(async {
            for ballots in
                [[false, false], [false, true], [true, false], [true, true]]
            {
                let (_temp_dir, node) = temp_node(&runtime)?;
                let () = connect_main_tip(&node)?;
                node.task_handles.net.task.abort();
                while !node.task_handles.net.task.is_finished() {
                    tokio::task::yield_now().await;
                }
                let mut rng = rand::rng();
                let key = SigningKey::new(&mut rng);
                let voter = authorization::get_address(&(&key).into());
                let decisions = [
                    DecisionId::new(false, 1, 0)?,
                    DecisionId::new(false, 1, 1)?,
                ];
                let mut rwtxn = node.env.write_txn()?;
                let body = Body::new(vec![], Coinbase::default());
                let header = Header {
                    merkle_root: Body::compute_merkle_root(
                        &body.coinbase,
                        &Vec::<crate::types::FilledTransaction>::new(),
                        &body.authorizations,
                        &body.actor_proofs,
                    )?,
                    prev_side_hash: None,
                    prev_main_hash: bitcoin::BlockHash::all_zeros(),
                    roots: Vec::new(),
                };
                node.archive.put_header(&mut rwtxn, &header)?;
                let timestamp = 1000
                    + u64::from(node.state.decisions().get_config().quantity);
                node.state
                    .connect_block(&mut rwtxn, &header, &body, timestamp)?;
                node.state
                    .reputation()
                    .set_reputation(&mut rwtxn, &voter, 1.0)?;
                for decision_id in decisions {
                    let decision = Decision::new(
                        voter.0,
                        DecisionType::Binary,
                        String::new(),
                        String::new(),
                        None,
                        None,
                        vec![],
                    )?;
                    node.state.decisions().claim_decision(
                        &mut rwtxn,
                        decision_id,
                        decision,
                        [1; 32].into(),
                        Some(0),
                    )?;
                    node.state.decisions().transition_decision_to_voting(
                        &mut rwtxn,
                        decision_id,
                        1,
                    )?;
                }
                let mut diff = AccumulatorDiff::default();
                let mut transactions = Vec::new();
                for (index, ballot) in
                    ballots.into_iter().chain([true]).enumerate()
                {
                    let pointed = PointedOutput {
                        outpoint: OutPoint::Regular {
                            txid: hash(&index).into(),
                            vout: 0,
                        },
                        output: Output {
                            address: voter,
                            content: OutputContent::Value(
                                bitcoin::Amount::from_sat(1_000),
                            ),
                        },
                    };
                    node.state.utxos.put(
                        &mut rwtxn,
                        &OutPointKey::from(&pointed.outpoint),
                        &pointed.output,
                    )?;
                    diff.insert((&pointed).into());
                    let decision_id_bytes =
                        decisions[usize::from(index == 2)].as_bytes();
                    let data = if ballot {
                        TxData::SubmitBallot {
                            voter,
                            voting_period: 2,
                            votes: vec![BallotItem {
                                decision_id_bytes,
                                vote_value: 1.0,
                            }],
                        }
                    } else {
                        TxData::SubmitVote {
                            voter,
                            voting_period: 2,
                            decision_id_bytes,
                            vote_value: 0.0,
                        }
                    };
                    transactions.push(Transaction {
                        inputs: vec![(pointed.outpoint, hash(&pointed))].into(),
                        data: Some(data),
                        ..Default::default()
                    });
                }
                let mut accumulator = Accumulator::default();
                accumulator.apply_diff(diff)?;
                node.state.utreexo_accumulator.put(
                    &mut rwtxn,
                    &(),
                    &accumulator,
                )?;
                rwtxn.commit()?;
                for transaction in transactions {
                    let authorization = Authorization {
                        verifying_key: (&key).into(),
                        signature: authorization::sign_tx(
                            &mut rng,
                            &key,
                            &transaction,
                        )?,
                    };
                    node.submit_transaction(AuthorizedTransaction {
                        transaction,
                        authorizations: vec![authorization],
                        actor_proof: None,
                    })?;
                }
                let (selected, fee) = node.get_transactions(100)?;
                assert_eq!(selected.len(), 2, "ballots {ballots:?}");
                assert_eq!(fee.to_sat(), 2_000);
                let keys: HashSet<_> = selected
                    .iter()
                    .flat_map(|tx| {
                        crate::validation::vote::vote_keys(
                            &tx.transaction.transaction,
                        )
                    })
                    .collect();
                assert_eq!(keys.len(), 2);
                for decision in decisions {
                    assert!(keys.contains(&(2, voter, decision.as_bytes())));
                }
                assert_eq!(node.get_all_transactions()?.len(), 3);
            }
            Ok(())
        })
    }

    #[test]
    fn block_selection_obeys_body_size_limit() -> anyhow::Result<()> {
        use crate::types::{
            AuthorizedTransaction, Body, Coinbase,
            authorization::{self, Authorization, SigningKey},
        };

        let runtime = tokio::runtime::Runtime::new()?;
        runtime.block_on(async {
            let (_temp_dir, node) = temp_node(&runtime)?;
            let () = connect_main_tip(&node)?;
            node.task_handles.net.task.abort();
            while !node.task_handles.net.task.is_finished() {
                tokio::task::yield_now().await;
            }
            let mut rng = rand::rng();
            let key = SigningKey::new(&mut rng);
            let address = authorization::get_address(&(&key).into());
            let mut rwtxn = node.env.write_txn()?;
            let mut diff = AccumulatorDiff::default();
            let mut transactions = Vec::new();
            for index in 0_u32..100 {
                let pointed = PointedOutput {
                    outpoint: OutPoint::Regular {
                        txid: hash(&index).into(),
                        vout: 0,
                    },
                    output: Output {
                        address,
                        content: OutputContent::Value(
                            bitcoin::Amount::from_sat(1_000),
                        ),
                    },
                };
                node.state.utxos.put(
                    &mut rwtxn,
                    &OutPointKey::from(&pointed.outpoint),
                    &pointed.output,
                )?;
                diff.insert((&pointed).into());
                transactions.push(Transaction {
                    inputs: vec![(pointed.outpoint, hash(&pointed))].into(),
                    outputs: vec![
                        Output {
                            address,
                            content: OutputContent::Value(
                                bitcoin::Amount::ZERO
                            ),
                        };
                        3_000
                    ]
                    .into(),
                    ..Default::default()
                });
            }
            let mut accumulator = Accumulator::default();
            accumulator.apply_diff(diff)?;
            node.state.utreexo_accumulator.put(
                &mut rwtxn,
                &(),
                &accumulator,
            )?;
            rwtxn.commit()?;
            for (index, transaction) in transactions.into_iter().enumerate() {
                let authorization = Authorization {
                    verifying_key: (&key).into(),
                    signature: authorization::sign_tx(
                        &mut rng,
                        &key,
                        &transaction,
                    )?,
                };
                let transaction = AuthorizedTransaction {
                    transaction,
                    actor_proof: (index % 2 == 0)
                        .then(|| Box::new(authorization.clone())),
                    authorizations: vec![authorization],
                };
                assert!(bincode::serialized_size(&transaction)? < 100 * 1024);
                node.submit_transaction(transaction)?;
            }
            let all =
                Body::new(node.get_all_transactions()?, Coinbase::default());
            let limit = state::State::body_size_limit(0);
            assert_eq!(all.transactions.len(), 100);
            assert!(borsh::object_length(&all)? > limit);

            let (selected, fee) = node.get_transactions(1_000)?;
            assert!(!selected.is_empty());
            let body = Body::new(
                selected.into_iter().map(Into::into).collect(),
                Coinbase {
                    memo: Vec::new(),
                    outputs: vec![Output {
                        address,
                        content: OutputContent::Value(fee),
                    }]
                    .into(),
                },
            );
            let size = borsh::object_length(&body)?;
            assert!(size <= limit, "Body size {size} exceeds limit {limit}");
            assert!(body.transactions.len() < 100);
            assert_eq!(node.get_all_transactions()?.len(), 100);
            Ok(())
        })
    }

    #[test]
    fn block_selection_counts_actor_proofs() -> anyhow::Result<()> {
        use crate::types::{
            AuthorizedTransaction,
            authorization::{self, Authorization, SigningKey},
        };

        let runtime = tokio::runtime::Runtime::new()?;
        runtime.block_on(async {
            let (_temp_dir, node) = temp_node(&runtime)?;
            let () = connect_main_tip(&node)?;
            node.task_handles.net.task.abort();
            while !node.task_handles.net.task.is_finished() {
                tokio::task::yield_now().await;
            }
            let mut rng = rand::rng();
            let key = SigningKey::new(&mut rng);
            let address = authorization::get_address(&(&key).into());
            let mut rwtxn = node.env.write_txn()?;
            let mut diff = AccumulatorDiff::default();
            let mut transactions = Vec::new();
            for index in 0_u32..1_000 {
                let mut inputs = Vec::new();
                for vout in 0..42 {
                    let pointed = PointedOutput {
                        outpoint: OutPoint::Regular {
                            txid: hash(&index).into(),
                            vout,
                        },
                        output: Output {
                            address,
                            content: OutputContent::Value(
                                bitcoin::Amount::from_sat(1_000),
                            ),
                        },
                    };
                    node.state.utxos.put(
                        &mut rwtxn,
                        &OutPointKey::from(&pointed.outpoint),
                        &pointed.output,
                    )?;
                    diff.insert((&pointed).into());
                    inputs.push((pointed.outpoint, hash(&pointed)));
                }
                transactions.push(Transaction {
                    inputs: inputs.into(),
                    outputs: vec![Output {
                        address,
                        content: OutputContent::Value(
                            bitcoin::Amount::from_sat(41_000),
                        ),
                    }]
                    .into(),
                    ..Default::default()
                });
            }
            let mut accumulator = Accumulator::default();
            accumulator.apply_diff(diff)?;
            node.state.utreexo_accumulator.put(
                &mut rwtxn,
                &(),
                &accumulator,
            )?;
            for mut transaction in transactions {
                let targets: Vec<_> = transaction
                    .inputs
                    .iter()
                    .map(|(_, hash)| hash.into())
                    .collect();
                transaction.proof = accumulator.prove(&targets)?;
                let authorization = Authorization {
                    verifying_key: (&key).into(),
                    signature: authorization::sign_tx(
                        &mut rng,
                        &key,
                        &transaction,
                    )?,
                };
                let transaction = AuthorizedTransaction {
                    transaction,
                    authorizations: vec![authorization.clone(); 42],
                    actor_proof: Some(Box::new(authorization)),
                };
                node.mempool.put(&mut rwtxn, &transaction)?;
            }
            rwtxn.commit()?;

            let (selected, fee) = node.get_transactions(1_000)?;
            assert_eq!(selected.len(), state::State::body_sigops_limit(0) / 43);
            assert_eq!(fee.to_sat(), selected.len() as u64 * 1_000);
            let sigops: usize = selected
                .iter()
                .map(|tx| {
                    tx.authorizations.len()
                        + usize::from(tx.actor_proof.is_some())
                })
                .sum();
            assert!(sigops <= state::State::body_sigops_limit(0));
            assert!(sigops + 43 > state::State::body_sigops_limit(0));
            assert_eq!(node.get_all_transactions()?.len(), 1_000);
            Ok(())
        })
    }

    #[test]
    fn peer_transaction_conflicts_do_not_stop_net_task() -> anyhow::Result<()> {
        let runtime = tokio::runtime::Runtime::new()?;
        runtime.block_on(async {
            let (temp_dir, node) = temp_node(&runtime)?;
            node.task_handles.net.task.abort();
            while !node.task_handles.net.task.is_finished() {
                tokio::task::yield_now().await;
            }
            let wallet = Wallet::new(&temp_dir.path().join("wallet"))?;
            wallet.set_seed(&[2; 64])?;
            let address = wallet.get_new_address()?;
            let mut inputs = Vec::new();
            let mut utxos = HashMap::new();
            let mut diff = AccumulatorDiff::default();
            let mut rwtxn = node.env.write_txn()?;
            for index in 0..2 {
                let pointed = PointedOutput {
                    outpoint: OutPoint::Regular {
                        txid: [index; 32].into(),
                        vout: 0,
                    },
                    output: Output {
                        address,
                        content: OutputContent::Value(
                            bitcoin::Amount::from_sat(1_000),
                        ),
                    },
                };
                node.state.utxos.put(
                    &mut rwtxn,
                    &OutPointKey::from(&pointed.outpoint),
                    &pointed.output,
                )?;
                diff.insert((&pointed).into());
                inputs.push((pointed.outpoint, hash(&pointed)));
                utxos.insert(pointed.outpoint, pointed.output);
            }
            wallet.put_utxos(&HashMap::new(), &HashMap::new(), &utxos)?;
            let mut accumulator = Accumulator::default();
            accumulator.apply_diff(diff)?;
            node.state.utreexo_accumulator.put(
                &mut rwtxn,
                &(),
                &accumulator,
            )?;
            let make_tx = |inputs, value| -> anyhow::Result<_> {
                let mut tx = Transaction {
                    inputs,
                    outputs: vec![Output {
                        address,
                        content: OutputContent::Value(
                            bitcoin::Amount::from_sat(value),
                        ),
                    }]
                    .into(),
                    ..Default::default()
                };
                node.state.regenerate_proof(
                    &rwtxn,
                    &HashMap::new(),
                    &mut tx,
                )?;
                let tx = wallet.authorize(rand::rng(), tx)?;
                node.state.validate_transaction(
                    &rwtxn,
                    &BatchVerificationContext::new(&mut rand::rng()),
                    &HashMap::new(),
                    0,
                    &tx,
                    &node.archive,
                )?;
                Ok(tx)
            };
            let first_tx = make_tx(vec![inputs[0]].into(), 900)?;
            let conflict_tx =
                make_tx(vec![inputs[1], inputs[0]].into(), 1_800)?;
            let next_tx = make_tx(vec![inputs[1]].into(), 900)?;
            node.mempool.put(&mut rwtxn, &first_tx)?;
            rwtxn.commit()?;

            let (
                forward_mainchain_task_request_tx,
                forward_mainchain_task_request_rx,
            ) = mpsc::unbounded();
            let (_mainchain_task_event_tx, mainchain_task_event_rx) =
                mpsc::unbounded();
            let (new_tip_ready_tx, new_tip_ready_rx) = mpsc::unbounded();
            let (peer_info_tx, peer_info_rx) = mpsc::unbounded();
            let task = NetTask {
                ctxt: NetTaskContext {
                    env: node.env.clone(),
                    archive: node.archive.clone(),
                    mainchain_task: node.task_handles.mainchain.clone(),
                    mempool: node.mempool.clone(),
                    net: node.net.clone(),
                    state: node.state.clone(),
                },
                forward_mainchain_task_request_tx,
                forward_mainchain_task_request_rx,
                mainchain_task_event_rx,
                new_tip_ready_tx,
                new_tip_ready_rx,
                peer_info_rx,
                rng: rand::rngs::ChaCha20Rng::seed_from_u64(0),
            };
            for tx in [&first_tx, &conflict_tx, &next_tx] {
                peer_info_tx.unbounded_send((
                    (Ipv4Addr::LOCALHOST, 1).into(),
                    Some(PeerConnectionInfo::NewTransaction(tx.clone())),
                ))?;
            }
            drop(peer_info_tx);
            let result =
                tokio::time::timeout(Duration::from_secs(5), task.run())
                    .await?;
            assert!(
                matches!(result, Err(Error::PeerInfoRxClosed)),
                "the network task stopped before the mailbox closed: {result:?}"
            );
            let rotxn = node.env.read_txn()?;
            for tx in [&first_tx, &next_tx] {
                assert!(
                    node.mempool
                        .transactions
                        .try_get(&rotxn, &tx.transaction.txid())?
                        .is_some()
                );
            }
            assert!(
                node.mempool
                    .transactions
                    .try_get(&rotxn, &conflict_tx.transaction.txid())?
                    .is_none()
            );
            Ok(())
        })
    }

    #[test]
    fn retry_connection_timeout_before_first_message() -> anyhow::Result<()> {
        let runtime = tokio::runtime::Runtime::new()?;
        runtime.block_on(async {
            let (_temp_dir, node) = temp_node(&runtime)?;
            let silent_peer =
                tokio::net::UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).await?;
            let addr = silent_peer.local_addr()?;
            node.connect_peer(addr.into())?;
            assert_eq!(node.get_active_peers().len(), 1);
            assert_eq!(
                node.net.try_with_active_peer_connection(addr, |peer| peer
                    .received_msg_successfully(),),
                Some(false)
            );

            tokio::time::timeout(Duration::from_secs(35), async {
                while !node.get_active_peers().is_empty() {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .context("the QUIC connection did not time out")?;
            drop(silent_peer);
            let (remote, _) = make_server_endpoint(addr, HashSet::new())?;
            let retry = tokio::time::timeout(Duration::from_secs(15), async {
                remote
                    .accept()
                    .await
                    .context("the endpoint closed before the retry")?
                    .await
                    .map_err(anyhow::Error::from)
            })
            .await
            .context("the node did not retry the connection timeout")??;
            assert!(retry.close_reason().is_none());
            remote.close(0_u32.into(), b"test complete");
            Ok(())
        })
    }

    #[test]
    fn retry_duplicate_close_before_first_message() -> anyhow::Result<()> {
        let runtime = tokio::runtime::Runtime::new()?;
        runtime.block_on(async {
            let (_temp_dir, node) = temp_node(&runtime)?;
            let (remote, _) = make_server_endpoint(
                (Ipv4Addr::LOCALHOST, 0).into(),
                HashSet::new(),
            )?;
            let addr = remote.local_addr()?;
            node.connect_peer(addr.into())?;
            let mut connection =
                tokio::time::timeout(Duration::from_secs(5), remote.accept())
                    .await?
                    .context("the first connection did not arrive")?
                    .await?;
            assert_eq!(
                node.net.try_with_active_peer_connection(addr, |peer| peer
                    .received_msg_successfully(),),
                Some(false)
            );

            let remote_addr = connection.remote_address();
            for _ in 0..2 {
                let closed_at = tokio::time::Instant::now();
                connection.close(1_u32.into(), b"already connected");
                connection =
                    tokio::time::timeout(Duration::from_secs(15), async {
                        remote
                            .accept()
                            .await
                            .context("the endpoint closed before the retry")?
                            .await
                            .map_err(anyhow::Error::from)
                    })
                    .await
                    .context("the node did not retry the duplicate close")??;

                assert!(closed_at.elapsed() >= Duration::from_secs(10));
                assert_eq!(connection.remote_address(), remote_addr);
                assert_eq!(
                    node.net.try_with_active_peer_connection(addr, |peer| peer
                        .received_msg_successfully(),),
                    Some(false)
                );
            }
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    if node.get_active_peers().iter().any(|peer| {
                        peer.address == addr
                            && peer.status == PeerConnectionStatus::Connected
                    }) {
                        return;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await?;
            remote.close(0_u32.into(), b"test complete");
            Ok(())
        })
    }

    #[test]
    fn an_invalid_best_side_tip_does_not_stop_the_net_task()
    -> anyhow::Result<()> {
        let runtime = tokio::runtime::Runtime::new()?;
        runtime.block_on(async {
            let temp_dir = temp_dir::TempDir::new()?;
            let channel =
                tonic::transport::Endpoint::from_static("http://127.0.0.1:1")
                    .connect_lazy();
            let (fatal_error_tx, _fatal_error_rx) =
                futures::channel::mpsc::unbounded();
            let node = Node::new(
                Config {
                    datadir: temp_dir.path().to_owned(),
                    bind_addr: (Ipv4Addr::LOCALHOST, 0).into(),
                    magic_bytes_override: None,
                    network: Network::Regtest,
                    add_peers: HashSet::new(),
                    server_names: HashSet::new(),
                    decision_config_testing: None,
                },
                ValidatorClient::new(channel),
                None,
                &mut rand::rng(),
                &runtime,
                fatal_error_tx,
            )?;
            node.task_handles.net.task.abort();
            while !node.task_handles.net.task.is_finished() {
                tokio::task::yield_now().await;
            }
            let main_parent = BlockHeaderInfo {
                block_hash: bitcoin::BlockHash::from_byte_array([2; 32]),
                prev_block_hash: bitcoin::BlockHash::all_zeros(),
                height: 0,
                work: bitcoin::Target::MAX.to_work(),
                timestamp: 0,
            };
            let main_header = BlockHeaderInfo {
                block_hash: bitcoin::BlockHash::from_byte_array([1; 32]),
                prev_block_hash: main_parent.block_hash,
                height: main_parent.height + 1,
                work: bitcoin::Target::MAX.to_work(),
                timestamp: 0,
            };
            let header = Header {
                merkle_root: MerkleRoot::from([7; 32]),
                prev_side_hash: None,
                prev_main_hash: main_parent.block_hash,
                roots: Vec::new(),
            };
            let block_hash = header.hash();
            let body = Body {
                coinbase: Coinbase::default(),
                transactions: Vec::new(),
                authorizations: Vec::new(),
                actor_proofs: Vec::new(),
            };
            let mut rwtxn = node.env.write_txn()?;
            // Mainchain headers and block infos, including the BMM
            // commitment, MUST be stored before the sidechain header,
            // as required by `Archive::put_header`
            node.archive
                .put_main_header_info(&mut rwtxn, &main_parent)?;
            node.archive.put_main_block_info(
                &mut rwtxn,
                main_parent.block_hash,
                &BlockInfo::default(),
            )?;
            node.archive
                .put_main_header_info(&mut rwtxn, &main_header)?;
            node.archive.put_main_block_info(
                &mut rwtxn,
                main_header.block_hash,
                &BlockInfo {
                    bmm_commitment: Some(block_hash),
                    events: Vec::new(),
                },
            )?;
            node.archive.put_header(&mut rwtxn, &header)?;
            node.archive.side_tips().connect_mainchain_tip(
                &mut rwtxn,
                main_parent,
                None,
            )?;
            node.archive.side_tips().connect_mainchain_tip(
                &mut rwtxn,
                main_header,
                Some(BmmCommitment {
                    sidechain_block_hash: block_hash,
                    sidechain_header_data: (&header).into(),
                }),
            )?;
            node.archive.put_body(&mut rwtxn, block_hash, &body)?;
            rwtxn.commit()?;
            let ctxt = NetTaskContext {
                env: node.env.clone(),
                archive: node.archive.clone(),
                mainchain_task: node.task_handles.mainchain.clone(),
                mempool: node.mempool.clone(),
                net: node.net.clone(),
                state: node.state.clone(),
            };
            for _ in 0..2 {
                NetTask::handle_mainchain_block_event(
                    &ctxt,
                    MainchainBlockEvent::DisconnectBlock {
                        block_hash: main_header.block_hash,
                    },
                )?;
            }
            let rotxn = node.env.read_txn()?;
            assert!(node.archive.try_get_body(&rotxn, block_hash)?.is_none());
            assert!(node.state.try_get_tip(&rotxn)?.is_none());
            Ok(())
        })
    }

    // a peer's invalid block (value out > value in) must not be fatal
    #[test]
    fn invalid_peer_block_is_not_fatal() {
        let err = Error::State(state::Error::NotEnoughValueIn);
        assert!(!err.is_fatal());
    }

    // local infrastructure errors stay fatal
    #[test]
    fn infrastructure_error_is_fatal() {
        assert!(Error::PeerInfoRxClosed.is_fatal());
    }

    // a peer that pushes a chain past the ancestor limit, a double spend, or a
    // transaction whose parent just left the mempool, must not stop the net
    // task
    #[test]
    fn a_peer_transaction_this_node_refuses_is_not_fatal() {
        use crate::{mempool, node::net_task::is_fatal_peer_tx_error};

        assert!(!is_fatal_peer_tx_error(&Error::MemPool(
            mempool::Error::TooManyAncestors {
                count: mempool::MAX_UNCONFIRMED_ANCESTORS,
            }
        )));
        assert!(!is_fatal_peer_tx_error(&Error::MemPool(
            mempool::Error::UtxoDoubleSpent
        )));
        assert!(!is_fatal_peer_tx_error(&Error::MemPool(
            mempool::Error::DecisionAlreadyClaimedInMempool(String::new())
        )));
        assert!(!is_fatal_peer_tx_error(&Error::State(
            state::Error::NotEnoughValueIn
        )));
        assert!(is_fatal_peer_tx_error(&Error::PeerInfoRxClosed));
    }

    #[test]
    fn block_selection_drops_transactions_that_conflict_at_connect()
    -> anyhow::Result<()> {
        use crate::{
            state::{
                decisions::{Decision, DecisionId, DecisionType},
                markets::{
                    DimensionSpec, compute_market_id,
                    generate_market_treasury_address,
                },
            },
            types::{
                Address, Body, Coinbase, Header, TransactionData,
                authorization::{SigningKey, authorize, get_address},
            },
        };
        use bitcoin::hashes::Hash as _;

        let runtime = tokio::runtime::Runtime::new()?;
        runtime.block_on(async {
            let (_temp_dir, node) = temp_node(&runtime)?;
            let () = connect_main_tip(&node)?;
            node.task_handles.net.task.abort();
            while !node.task_handles.net.task.is_finished() {
                tokio::task::yield_now().await;
            }
            let key = SigningKey::new(&mut rand::rng());
            let creator = get_address(&(&key).into());
            let decision_id = DecisionId::new(false, 0, 0)?;
            let dimension_specs = vec![DimensionSpec::Single(decision_id)];
            let market_id =
                compute_market_id("title", "", &creator, &dimension_specs);
            let mut accumulator = Accumulator::default();
            let mut diff = AccumulatorDiff::default();
            let mut inputs = Vec::new();
            let mut rwtxn = node.env.write_txn()?;
            node.state.decisions().claim_decision(
                &mut rwtxn,
                decision_id,
                Decision::new(
                    creator.0,
                    DecisionType::Binary,
                    "header".to_owned(),
                    String::new(),
                    None,
                    None,
                    Vec::new(),
                )?,
                [9; 32].into(),
                Some(0),
            )?;
            for index in 0..2u8 {
                let pointed = PointedOutput {
                    outpoint: OutPoint::Regular {
                        txid: [index; 32].into(),
                        vout: 0,
                    },
                    output: Output {
                        address: creator,
                        content: OutputContent::Value(
                            bitcoin::Amount::from_sat(10_000),
                        ),
                    },
                };
                node.state.utxos.put(
                    &mut rwtxn,
                    &OutPointKey::from(&pointed.outpoint),
                    &pointed.output,
                )?;
                diff.insert((&pointed).into());
                inputs.push((pointed.outpoint, hash(&pointed)));
            }
            accumulator.apply_diff(diff)?;
            node.state.utreexo_accumulator.put(
                &mut rwtxn,
                &(),
                &accumulator,
            )?;
            rwtxn.commit()?;

            let mut market_txs = Vec::new();
            for input in inputs {
                let tx = Transaction {
                    inputs: vec![input].into(),
                    outputs: vec![
                        Output {
                            address: generate_market_treasury_address(
                                &market_id,
                            ),
                            content: OutputContent::MarketFunds {
                                market_id: market_id.0,
                                amount: bitcoin::Amount::from_sat(5_000),
                                is_fee: false,
                            },
                        },
                        Output {
                            address: creator,
                            content: OutputContent::Value(
                                bitcoin::Amount::from_sat(3_900),
                            ),
                        },
                    ]
                    .into(),
                    data: Some(TransactionData::CreateMarket {
                        title: "title".to_owned(),
                        description: String::new(),
                        dimension_specs: dimension_specs.clone(),
                        new_claims: Vec::new(),
                        trading_fee: None,
                        tx_pow_hash_selector: None,
                        tx_pow_ordering: None,
                        tx_pow_difficulty: None,
                    }),
                    ..Default::default()
                };
                let tx = authorize(rand::rng(), &[(creator, &key)], tx)?;
                market_txs.push(tx.transaction.clone());
                node.submit_transaction(tx)?;
            }
            // Each market creation has a child that spends its change.
            let mut child_txids = Vec::new();
            for market_tx in &market_txs {
                let change = OutPoint::Regular {
                    txid: market_tx.txid(),
                    vout: 1,
                };
                let child = Transaction {
                    inputs: vec![(
                        change,
                        hash(&PointedOutput {
                            outpoint: change,
                            output: market_tx.outputs.as_slice()[1].clone(),
                        }),
                    )]
                    .into(),
                    outputs: vec![Output {
                        address: Address([4; 20]),
                        content: OutputContent::Value(
                            bitcoin::Amount::from_sat(3_000),
                        ),
                    }]
                    .into(),
                    ..Default::default()
                };
                let child = authorize(rand::rng(), &[(creator, &key)], child)?;
                child_txids.push(child.transaction.txid());
                node.submit_transaction(child)?;
            }

            let (selected, fee) = node.get_transactions(100)?;
            let selected_txids: Vec<_> =
                selected.iter().map(|tx| tx.transaction.txid()).collect();
            let kept = market_txs
                .iter()
                .position(|tx| selected_txids.contains(&tx.txid()))
                .expect("one market creation stays in the block");
            assert_eq!(
                selected_txids,
                vec![market_txs[kept].txid(), child_txids[kept]]
            );
            assert_eq!(fee.to_sat(), 2_000);
            let body = Body::new(
                selected.iter().cloned().map(Into::into).collect(),
                Coinbase::default(),
            );
            let merkle_root = Body::compute_merkle_root(
                &body.coinbase,
                &selected,
                &body.authorizations,
                &body.actor_proofs,
            )?;
            let mut header = Header {
                merkle_root,
                prev_side_hash: None,
                prev_main_hash: bitcoin::BlockHash::all_zeros(),
                roots: Vec::new(),
            };
            Body::modify_memforest(
                header.compute_coinbase_txid(),
                &[],
                &selected,
                &mut accumulator.0,
            )?;
            header.roots = accumulator.get_roots();
            let mut rwtxn = node.env.write_txn()?;
            node.state.apply_block(
                &mut rwtxn,
                &node.batch_verification_ctxt,
                &header,
                0,
                &body,
                &node.archive,
                1_700_000_000,
            )?;
            assert!(
                node.state
                    .markets()
                    .get_market(&rwtxn, &market_id)?
                    .is_some()
            );
            Ok(())
        })
    }
}
