use std::collections::{BTreeMap, HashMap};

use fallible_iterator::FallibleIterator;
use sneed::{RoTxn, RwTxn};

use crate::{
    state::{
        Error, State, UtxoManager, WITHDRAWAL_BUNDLE_FAILURE_GAP,
        WithdrawalBundleInfo,
        rollback::{HeightStamped, RollBack},
    },
    types::{
        AggregatedWithdrawal, AmountOverflowError, BlockIndexEvents,
        FilledOutput, FilledOutputContent, InPoint, M6id, OutPoint,
        OutPointKey, SpentOutput, WithdrawalBundle, WithdrawalBundleEvent,
        WithdrawalBundleEventStatus, WithdrawalBundleStatus,
        WithdrawalOutputContent,
        proto::mainchain::{BlockEvent, TwoWayPegData},
    },
};

fn collect_withdrawal_bundle(
    state: &State,
    txn: &RoTxn,
    block_height: u32,
) -> Result<Option<WithdrawalBundle>, Error> {
    let mut address_to_aggregated_withdrawal = HashMap::<
        bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        AggregatedWithdrawal,
    >::new();
    state.utxos.iter(txn)?.map_err(Error::from).for_each(
        |(outpoint, output)| {
            if let FilledOutputContent::BitcoinWithdrawal(
                WithdrawalOutputContent {
                    value,
                    ref main_address,
                    main_fee,
                },
            ) = output.content
            {
                let aggregated = address_to_aggregated_withdrawal
                    .entry(main_address.clone())
                    .or_insert(AggregatedWithdrawal {
                        spend_utxos: HashMap::new(),
                        main_address: main_address.clone(),
                        value: bitcoin::Amount::ZERO,
                        main_fee: bitcoin::Amount::ZERO,
                    });
                aggregated.value = aggregated
                    .value
                    .checked_add(value)
                    .ok_or(AmountOverflowError)?;
                aggregated.main_fee = aggregated
                    .main_fee
                    .checked_add(main_fee)
                    .ok_or(AmountOverflowError)?;
                aggregated
                    .spend_utxos
                    .insert(outpoint.to_outpoint(), output);
            }
            Ok(())
        },
    )?;
    if address_to_aggregated_withdrawal.is_empty() {
        return Ok(None);
    }
    let mut aggregated_withdrawals: Vec<_> =
        address_to_aggregated_withdrawal.into_values().collect();
    aggregated_withdrawals.sort_by_key(|a| std::cmp::Reverse(a.clone()));
    let mut fee = bitcoin::Amount::ZERO;
    let mut spend_utxos = BTreeMap::<OutPoint, FilledOutput>::new();
    let mut bundle_outputs = Vec::new();
    let mut bundle_txouts_size: u32 = 0;
    for aggregated in &aggregated_withdrawals {
        let script_pubkey =
            aggregated.main_address.assume_checked_ref().script_pubkey();
        let Ok(n_outputs) = u32::try_from(bundle_outputs.len() + 1) else {
            break;
        };
        let Ok(spk_size) = u32::try_from(script_pubkey.len()) else {
            // This SPK is invalid, but others might be ok
            continue;
        };
        let Some(txout_size) = WithdrawalBundle::txout_size(spk_size) else {
            // This SPK is invalid, but others might be ok
            continue;
        };
        if let Some(sum_txout_sizes) =
            bundle_txouts_size.checked_add(txout_size)
        {
            bundle_txouts_size = sum_txout_sizes;
        } else {
            break;
        };
        if WithdrawalBundle::predict_weight(n_outputs, bundle_txouts_size)
            .is_none()
        {
            break;
        }
        let bundle_output = bitcoin::TxOut {
            value: aggregated.value,
            script_pubkey,
        };
        spend_utxos.extend(aggregated.spend_utxos.clone());
        bundle_outputs.push(bundle_output);
        fee += aggregated.main_fee;
    }
    let bundle =
        WithdrawalBundle::new(block_height, fee, spend_utxos, bundle_outputs)?;
    if bundle.tx().weight().to_wu()
        > bitcoin::policy::MAX_STANDARD_TX_WEIGHT as u64
    {
        Err(Error::BundleTooHeavy {
            weight: bundle.tx().weight().to_wu(),
            max_weight: bitcoin::policy::MAX_STANDARD_TX_WEIGHT as u64,
        })?;
    }
    Ok(Some(bundle))
}

fn connect_withdrawal_bundle_submitted(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    index_events: &mut BlockIndexEvents,
    event_block_hash: &bitcoin::BlockHash,
    m6id: M6id,
) -> Result<(), Error> {
    if let Some(bundle_m6id) =
        state.pending_withdrawal_bundle.try_get(rwtxn, &())?
        && bundle_m6id == m6id
    {
        tracing::debug!(
            %block_height,
            %m6id,
            "Pending withdrawal bundle submission confirmed"
        );
        let (bundle, mut bundle_status) = state
            .withdrawal_bundles
            .try_get(rwtxn, &m6id)?
            .ok_or_else(|| {
                Error::DatabaseError(format!(
                    "pending withdrawal bundle {m6id} \
                     unknown in withdrawal_bundles"
                ))
            })?;
        let bundle = match bundle {
            WithdrawalBundleInfo::Known(bundle) => bundle,
            WithdrawalBundleInfo::Unknown
            | WithdrawalBundleInfo::UnknownConfirmed { spend_utxos: _ } => {
                return Err(Error::DatabaseError(format!(
                    "pending withdrawal bundle {m6id} is not known"
                )));
            }
        };
        for (outpoint, spend_output) in bundle.spend_utxos() {
            let outpoint_key = OutPointKey::from_outpoint(outpoint);
            if !state.delete_utxo(rwtxn, outpoint)? {
                return Err(Error::NoUtxo {
                    outpoint: *outpoint,
                });
            };
            let spent_output = SpentOutput {
                output: spend_output.clone(),
                inpoint: InPoint::Withdrawal { m6id },
            };
            state.stxos.put(rwtxn, &outpoint_key, &spent_output)?;
            index_events.bundle_spends.push((*outpoint, m6id));
        }
        if bundle_status.latest().value != WithdrawalBundleStatus::Pending {
            return Err(Error::DatabaseError(format!(
                "expected Pending status before Submitted, got {:?}",
                bundle_status.latest().value,
            )));
        }
        bundle_status
            .push(WithdrawalBundleStatus::Submitted, block_height)
            .map_err(|e| {
                Error::DatabaseError(format!(
                    "failed to push submitted bundle status: {e:?}"
                ))
            })?;
        state.withdrawal_bundles.put(
            rwtxn,
            &m6id,
            &(WithdrawalBundleInfo::Known(bundle), bundle_status),
        )?;
        state.pending_withdrawal_bundle.delete(rwtxn, &())?;
    } else if let Some((bundle, mut bundle_status)) =
        state.withdrawal_bundles.try_get(rwtxn, &m6id)?
    {
        match (&bundle, bundle_status.latest().value) {
            (_, WithdrawalBundleStatus::Confirmed) => {
                return Err(Error::DatabaseError(format!(
                    "confirmed withdrawal bundle {m6id} resubmitted in \
                     {event_block_hash}"
                )));
            }
            (
                _,
                WithdrawalBundleStatus::Submitted
                | WithdrawalBundleStatus::SubmittedUnexpected,
            ) => {
                return Err(Error::DatabaseError(format!(
                    "withdrawal bundle {m6id} submitted in {} resubmitted in \
                     {event_block_hash}",
                    bundle_status.latest().height,
                )));
            }
            (
                WithdrawalBundleInfo::Known(_),
                WithdrawalBundleStatus::Dropped,
            ) => {
                tracing::warn!(%event_block_hash, %m6id, "dropped bundle submitted");
            }
            (
                WithdrawalBundleInfo::Unknown
                | WithdrawalBundleInfo::UnknownConfirmed { spend_utxos: _ },
                WithdrawalBundleStatus::Dropped,
            ) => {
                return Err(Error::DatabaseError(format!(
                    "unknown withdrawal bundle {m6id} marked as dropped in {}",
                    bundle_status.latest().height,
                )));
            }
            (
                WithdrawalBundleInfo::Known(_),
                WithdrawalBundleStatus::Pending,
            ) => {
                return Err(Error::DatabaseError(format!(
                    "dropped withdrawal bundle {m6id} marked as pending in \
                     withdrawal_bundles"
                )));
            }
            (
                WithdrawalBundleInfo::Unknown
                | WithdrawalBundleInfo::UnknownConfirmed { spend_utxos: _ },
                WithdrawalBundleStatus::Pending,
            ) => {
                return Err(Error::DatabaseError(format!(
                    "unknown withdrawal bundle {m6id} marked as pending in {}",
                    bundle_status.latest().height,
                )));
            }
            (
                WithdrawalBundleInfo::Known(_) | WithdrawalBundleInfo::Unknown,
                WithdrawalBundleStatus::Failed,
            ) => {
                tracing::warn!(%event_block_hash, %m6id, "failed bundle resubmitted");
            }
            (
                WithdrawalBundleInfo::UnknownConfirmed { spend_utxos: _ },
                WithdrawalBundleStatus::Failed,
            ) => {
                return Err(Error::DatabaseError(format!(
                    "unknown confirmed withdrawal bundle {m6id} marked as \
                     failed in {}",
                    bundle_status.latest().height,
                )));
            }
        }
        bundle_status
            .push(WithdrawalBundleStatus::SubmittedUnexpected, block_height)
            .map_err(|e| {
                Error::DatabaseError(format!(
                    "failed to push submitted-unexpected status: {e:?}"
                ))
            })?;
        state
            .withdrawal_bundles
            .put(rwtxn, &m6id, &(bundle, bundle_status))?;
    } else {
        tracing::warn!(
            %event_block_hash,
            %m6id,
            "Unknown withdrawal bundle submitted"
        );
        state.withdrawal_bundles.put(
            rwtxn,
            &m6id,
            &(
                WithdrawalBundleInfo::Unknown,
                RollBack::<HeightStamped<_>>::new(
                    WithdrawalBundleStatus::Submitted,
                    block_height,
                ),
            ),
        )?;
    };
    Ok(())
}

fn connect_withdrawal_bundle_confirmed(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    event_block_hash: &bitcoin::BlockHash,
    m6id: M6id,
) -> Result<(), Error> {
    let (mut bundle, mut bundle_status) = state
        .withdrawal_bundles
        .try_get(rwtxn, &m6id)?
        .ok_or(Error::UnknownWithdrawalBundle { m6id })?;
    if bundle_status.latest().value == WithdrawalBundleStatus::Confirmed {
        return Ok(());
    }
    if !matches!(
        bundle_status.latest().value,
        WithdrawalBundleStatus::Submitted
            | WithdrawalBundleStatus::SubmittedUnexpected
    ) {
        return Err(Error::DatabaseError(format!(
            "expected Submitted status before Confirmed, got {:?}",
            bundle_status.latest().value,
        )));
    }
    match &bundle {
        WithdrawalBundleInfo::UnknownConfirmed { spend_utxos: _ } => {
            return Err(Error::UnknownWithdrawalBundleReconfirmed {
                event_block_hash: *event_block_hash,
                m6id,
            });
        }
        WithdrawalBundleInfo::Unknown => {
            // If an unknown bundle is confirmed, all UTXOs older than the
            // bundle submission are potentially spent. This is only accepted
            // when block height is 0, when no UTXOs could have been
            // double-spent yet; ALL UTXOs are then considered spent.
            if block_height == 0 {
                tracing::warn!(
                    %event_block_hash,
                    %m6id,
                    "Unknown withdrawal bundle confirmed, marking all UTXOs as spent"
                );
                let utxos_keys: BTreeMap<_, _> =
                    state.utxos.iter(rwtxn)?.collect()?;
                let mut utxos = BTreeMap::new();
                for (outpoint_key, output) in &utxos_keys {
                    let outpoint = outpoint_key.to_outpoint();
                    let spent_output = SpentOutput {
                        output: output.clone(),
                        inpoint: InPoint::Withdrawal { m6id },
                    };
                    state.stxos.put(rwtxn, outpoint_key, &spent_output)?;
                    utxos.insert(outpoint, output.clone());
                }
                state.clear_utxos(rwtxn)?;
                bundle = WithdrawalBundleInfo::UnknownConfirmed {
                    spend_utxos: utxos,
                };
            } else {
                return Err(Error::UnknownWithdrawalBundleConfirmed {
                    event_block_hash: *event_block_hash,
                    m6id,
                });
            }
        }
        WithdrawalBundleInfo::Known(bundle) => {
            if matches!(
                bundle_status.latest().value,
                WithdrawalBundleStatus::SubmittedUnexpected
            ) {
                // A previously dropped or failed bundle is confirmed; unless
                // all of the bundle UTXOs can be spent, the chain is
                // insolvent and cannot continue.
                tracing::warn!(
                    %event_block_hash,
                    %m6id,
                    "Unexpected withdrawal bundle confirmed, marking bundle UTXOs as spent"
                );
                for (outpoint, output) in bundle.spend_utxos() {
                    let outpoint_key = OutPointKey::from_outpoint(outpoint);
                    if !state.delete_utxo(rwtxn, outpoint)? {
                        return Err(
                            Error::UnexpectedWithdrawalBundleInsolvency {
                                event_block_hash: *event_block_hash,
                                m6id,
                                outpoint: *outpoint,
                            },
                        );
                    }
                    let spent_output = SpentOutput {
                        output: output.clone(),
                        inpoint: InPoint::Withdrawal { m6id },
                    };
                    state.stxos.put(rwtxn, &outpoint_key, &spent_output)?;
                }
            }
        }
    }
    bundle_status
        .push(WithdrawalBundleStatus::Confirmed, block_height)
        .map_err(|e| {
            Error::DatabaseError(format!(
                "failed to push confirmed bundle status: {e:?}"
            ))
        })?;
    state
        .withdrawal_bundles
        .put(rwtxn, &m6id, &(bundle, bundle_status))?;
    Ok(())
}

fn connect_withdrawal_bundle_failed(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    m6id: M6id,
) -> Result<(), Error> {
    let (bundle, mut bundle_status) = state
        .withdrawal_bundles
        .try_get(rwtxn, &m6id)?
        .ok_or_else(|| Error::UnknownWithdrawalBundle { m6id })?;
    if bundle_status.latest().value == WithdrawalBundleStatus::Failed {
        return Ok(());
    }
    if !matches!(
        bundle_status.latest().value,
        WithdrawalBundleStatus::Submitted
            | WithdrawalBundleStatus::SubmittedUnexpected
    ) {
        return Err(Error::DatabaseError(format!(
            "expected Submitted status before Failed, got {:?}",
            bundle_status.latest().value,
        )));
    }
    match &bundle {
        WithdrawalBundleInfo::Unknown
        | WithdrawalBundleInfo::UnknownConfirmed { .. } => (),
        WithdrawalBundleInfo::Known(bundle) => 'known: {
            if matches!(
                bundle_status.latest().value,
                WithdrawalBundleStatus::SubmittedUnexpected
            ) {
                break 'known;
            }
            for (outpoint, output) in bundle.spend_utxos() {
                let outpoint_key = OutPointKey::from_outpoint(outpoint);
                state.stxos.delete(rwtxn, &outpoint_key)?;
                state.insert_utxo(rwtxn, outpoint, output)?;
            }
            let latest_failed_m6id = if let Some(mut latest_failed_m6id) =
                state.latest_failed_withdrawal_bundle.try_get(rwtxn, &())?
            {
                latest_failed_m6id.push(m6id, block_height).map_err(|e| {
                    Error::DatabaseError(format!(
                        "failed to push latest failed m6id: {e:?}"
                    ))
                })?;
                latest_failed_m6id
            } else {
                RollBack::<HeightStamped<_>>::new(m6id, block_height)
            };
            state.latest_failed_withdrawal_bundle.put(
                rwtxn,
                &(),
                &latest_failed_m6id,
            )?;
        }
    }
    bundle_status
        .push(WithdrawalBundleStatus::Failed, block_height)
        .map_err(|e| {
            Error::DatabaseError(format!(
                "failed to push failed bundle status: {e:?}"
            ))
        })?;
    state
        .withdrawal_bundles
        .put(rwtxn, &m6id, &(bundle, bundle_status))?;
    Ok(())
}

fn connect_withdrawal_bundle_event(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    index_events: &mut BlockIndexEvents,
    event_block_hash: &bitcoin::BlockHash,
    event: &WithdrawalBundleEvent,
) -> Result<(), Error> {
    match event.status {
        WithdrawalBundleEventStatus::Submitted => {
            connect_withdrawal_bundle_submitted(
                state,
                rwtxn,
                block_height,
                index_events,
                event_block_hash,
                event.m6id,
            )
        }
        WithdrawalBundleEventStatus::Confirmed => {
            connect_withdrawal_bundle_confirmed(
                state,
                rwtxn,
                block_height,
                event_block_hash,
                event.m6id,
            )
        }
        WithdrawalBundleEventStatus::Failed => {
            connect_withdrawal_bundle_failed(
                state,
                rwtxn,
                block_height,
                event.m6id,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn connect_2wpd_event(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    index_events: &mut BlockIndexEvents,
    latest_deposit_block_hash: &mut Option<bitcoin::BlockHash>,
    latest_withdrawal_bundle_event_block_hash: &mut Option<bitcoin::BlockHash>,
    event_block_hash: bitcoin::BlockHash,
    event: &BlockEvent,
) -> Result<(), Error> {
    match event {
        BlockEvent::Deposit(deposit) => {
            let outpoint = OutPoint::Deposit(deposit.outpoint);
            let output = deposit.output.clone();
            state.insert_utxo(rwtxn, &outpoint, &output)?;
            index_events.deposits.push((outpoint, output));
            *latest_deposit_block_hash = Some(event_block_hash);
        }
        BlockEvent::WithdrawalBundle(withdrawal_bundle_event) => {
            let () = connect_withdrawal_bundle_event(
                state,
                rwtxn,
                block_height,
                index_events,
                &event_block_hash,
                withdrawal_bundle_event,
            )?;
            *latest_withdrawal_bundle_event_block_hash = Some(event_block_hash);
        }
    }
    Ok(())
}

pub fn connect(
    state: &State,
    rwtxn: &mut RwTxn,
    two_way_peg_data: &TwoWayPegData,
) -> Result<(), Error> {
    let block_height = state.try_get_height(rwtxn)?.ok_or(Error::NoTip)?;
    let mut index_events = BlockIndexEvents::default();
    let mut latest_deposit_block_hash = None;
    let mut latest_withdrawal_bundle_event_block_hash = None;
    for (event_block_hash, event_block_info) in &two_way_peg_data.block_info {
        for event in &event_block_info.events {
            let () = connect_2wpd_event(
                state,
                rwtxn,
                block_height,
                &mut index_events,
                &mut latest_deposit_block_hash,
                &mut latest_withdrawal_bundle_event_block_hash,
                *event_block_hash,
                event,
            )?;
        }
    }
    // Record what this block moved outside its body. An address index cannot
    // see a deposit or a bundle spend any other way.
    if !index_events.is_empty() {
        state
            .block_index_events
            .put(rwtxn, &block_height, &index_events)?;
    }
    if let Some(latest_deposit_block_hash) = latest_deposit_block_hash {
        let deposit_block_seq_idx = state
            .deposit_blocks
            .last(rwtxn)?
            .map_or(0, |(seq_idx, _)| seq_idx + 1);
        state.deposit_blocks.put(
            rwtxn,
            &deposit_block_seq_idx,
            &(latest_deposit_block_hash, block_height),
        )?;
    }
    if let Some(latest_withdrawal_bundle_event_block_hash) =
        latest_withdrawal_bundle_event_block_hash
    {
        let withdrawal_bundle_event_block_seq_idx = state
            .withdrawal_bundle_event_blocks
            .last(rwtxn)?
            .map_or(0, |(seq_idx, _)| seq_idx + 1);
        state.withdrawal_bundle_event_blocks.put(
            rwtxn,
            &withdrawal_bundle_event_block_seq_idx,
            &(latest_withdrawal_bundle_event_block_hash, block_height),
        )?;
    }
    let last_withdrawal_bundle_failure_height = state
        .get_latest_failed_withdrawal_bundle(rwtxn)?
        .map(|(height, _bundle)| height)
        .unwrap_or_default();
    if block_height - last_withdrawal_bundle_failure_height
        >= WITHDRAWAL_BUNDLE_FAILURE_GAP
        && state
            .pending_withdrawal_bundle
            .try_get(rwtxn, &())?
            .is_none()
        && let Some(bundle) =
            collect_withdrawal_bundle(state, rwtxn, block_height)?
    {
        let m6id = bundle.compute_m6id();
        state.pending_withdrawal_bundle.put(rwtxn, &(), &m6id)?;
        let bundle_status = if let Some((_bundle, mut bundle_status)) =
            state.withdrawal_bundles.try_get(rwtxn, &m6id)?
        {
            bundle_status
                .push(WithdrawalBundleStatus::Pending, block_height)
                .map_err(|e| {
                    Error::DatabaseError(format!(
                        "failed to push pending bundle status: {e:?}"
                    ))
                })?;
            bundle_status
        } else {
            RollBack::<HeightStamped<_>>::new(
                WithdrawalBundleStatus::Pending,
                block_height,
            )
        };
        state.withdrawal_bundles.put(
            rwtxn,
            &m6id,
            &(WithdrawalBundleInfo::Known(bundle), bundle_status),
        )?;
        tracing::trace!(
            %block_height,
            %m6id,
            "Stored pending withdrawal bundle"
        );
    }
    Ok(())
}

fn disconnect_withdrawal_bundle_submitted(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    m6id: M6id,
) -> Result<(), Error> {
    let Some((bundle, bundle_status)) =
        state.withdrawal_bundles.try_get(rwtxn, &m6id)?
    else {
        if let Some(pending_bundle_m6id) =
            state.pending_withdrawal_bundle.try_get(rwtxn, &())?
            && pending_bundle_m6id == m6id
        {
            return Ok(());
        } else {
            return Err(Error::UnknownWithdrawalBundle { m6id });
        }
    };
    let (bundle_status, latest_bundle_status) = bundle_status.pop();
    if !matches!(
        latest_bundle_status.value,
        WithdrawalBundleStatus::Submitted
            | WithdrawalBundleStatus::SubmittedUnexpected
    ) {
        return Err(Error::DatabaseError(format!(
            "expected Submitted status for disconnect, got {:?}",
            latest_bundle_status.value,
        )));
    }
    if latest_bundle_status.height != block_height {
        return Err(Error::DatabaseError(format!(
            "bundle height {} != block height {block_height}",
            latest_bundle_status.height,
        )));
    }
    match &bundle {
        WithdrawalBundleInfo::Unknown
        | WithdrawalBundleInfo::UnknownConfirmed { .. } => (),
        WithdrawalBundleInfo::Known(bundle) => {
            if let Some(bundle_status) = &bundle_status
                && bundle_status.latest().value
                    == WithdrawalBundleStatus::Pending
            {
                for (outpoint, output) in bundle.spend_utxos().iter().rev() {
                    let outpoint_key = OutPointKey::from_outpoint(outpoint);
                    if !state.stxos.delete(rwtxn, &outpoint_key)? {
                        return Err(Error::NoStxo {
                            outpoint: *outpoint,
                        });
                    };
                    state.insert_utxo(rwtxn, outpoint, output)?;
                }
                state.pending_withdrawal_bundle.put(rwtxn, &(), &m6id)?;
            }
        }
    }
    if let Some(bundle_status) = bundle_status {
        state
            .withdrawal_bundles
            .put(rwtxn, &m6id, &(bundle, bundle_status))?;
    } else {
        state.withdrawal_bundles.delete(rwtxn, &m6id)?;
    }
    Ok(())
}

fn disconnect_withdrawal_bundle_confirmed(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    m6id: M6id,
) -> Result<(), Error> {
    let (mut bundle, bundle_status) = state
        .withdrawal_bundles
        .try_get(rwtxn, &m6id)?
        .ok_or_else(|| Error::UnknownWithdrawalBundle { m6id })?;
    let (prev_bundle_status, latest_bundle_status) = bundle_status.pop();
    if matches!(
        latest_bundle_status.value,
        WithdrawalBundleStatus::Submitted
            | WithdrawalBundleStatus::SubmittedUnexpected
    ) {
        return Ok(());
    }
    if latest_bundle_status.value != WithdrawalBundleStatus::Confirmed {
        return Err(Error::DatabaseError(format!(
            "expected Confirmed status for disconnect, got {:?}",
            latest_bundle_status.value,
        )));
    }
    if latest_bundle_status.height != block_height {
        return Err(Error::DatabaseError(format!(
            "confirmed bundle height {} != block height {block_height}",
            latest_bundle_status.height,
        )));
    }
    let prev_bundle_status = prev_bundle_status.ok_or_else(|| {
        Error::DatabaseError(
            "confirmed bundle has no previous status".to_string(),
        )
    })?;
    if !matches!(
        prev_bundle_status.latest().value,
        WithdrawalBundleStatus::Submitted
            | WithdrawalBundleStatus::SubmittedUnexpected
    ) {
        return Err(Error::DatabaseError(format!(
            "expected Submitted status before Confirmed, got {:?}",
            prev_bundle_status.latest().value,
        )));
    }
    match &bundle {
        WithdrawalBundleInfo::Known(known) => {
            if matches!(
                prev_bundle_status.latest().value,
                WithdrawalBundleStatus::SubmittedUnexpected
            ) {
                for (outpoint, output) in known.spend_utxos() {
                    let outpoint_key = OutPointKey::from_outpoint(outpoint);
                    state.insert_utxo(rwtxn, outpoint, output)?;
                    if !state.stxos.delete(rwtxn, &outpoint_key)? {
                        return Err(Error::NoStxo {
                            outpoint: *outpoint,
                        });
                    };
                }
            }
        }
        WithdrawalBundleInfo::UnknownConfirmed { spend_utxos } => {
            for (outpoint, output) in spend_utxos {
                let outpoint_key = OutPointKey::from_outpoint(outpoint);
                state.insert_utxo(rwtxn, outpoint, output)?;
                if !state.stxos.delete(rwtxn, &outpoint_key)? {
                    return Err(Error::NoStxo {
                        outpoint: *outpoint,
                    });
                };
            }
            bundle = WithdrawalBundleInfo::Unknown;
        }
        WithdrawalBundleInfo::Unknown => (),
    }
    state.withdrawal_bundles.put(
        rwtxn,
        &m6id,
        &(bundle, prev_bundle_status),
    )?;
    Ok(())
}

fn disconnect_withdrawal_bundle_failed(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    m6id: M6id,
) -> Result<(), Error> {
    let (bundle, bundle_status) = state
        .withdrawal_bundles
        .try_get(rwtxn, &m6id)?
        .ok_or_else(|| Error::UnknownWithdrawalBundle { m6id })?;
    let (prev_bundle_status, latest_bundle_status) = bundle_status.pop();
    if latest_bundle_status.value == WithdrawalBundleStatus::Submitted {
        return Ok(());
    } else if latest_bundle_status.value != WithdrawalBundleStatus::Failed {
        return Err(Error::DatabaseError(format!(
            "expected Failed status for disconnect, got {:?}",
            latest_bundle_status.value,
        )));
    }
    if latest_bundle_status.height != block_height {
        return Err(Error::DatabaseError(format!(
            "failed bundle height {} != block height {block_height}",
            latest_bundle_status.height,
        )));
    }
    let prev_bundle_status = prev_bundle_status.ok_or_else(|| {
        Error::DatabaseError("failed bundle has no previous status".to_string())
    })?;
    if !matches!(
        prev_bundle_status.latest().value,
        WithdrawalBundleStatus::Submitted
            | WithdrawalBundleStatus::SubmittedUnexpected
    ) {
        return Err(Error::DatabaseError(format!(
            "expected Submitted status before Failed, got {:?}",
            prev_bundle_status.latest().value,
        )));
    }
    match &bundle {
        WithdrawalBundleInfo::Unknown
        | WithdrawalBundleInfo::UnknownConfirmed { .. } => (),
        WithdrawalBundleInfo::Known(bundle) => 'known: {
            if matches!(
                prev_bundle_status.latest().value,
                WithdrawalBundleStatus::SubmittedUnexpected
            ) {
                break 'known;
            }
            for (outpoint, output) in bundle.spend_utxos().iter().rev() {
                let outpoint_key = OutPointKey::from_outpoint(outpoint);
                let spent_output = SpentOutput {
                    output: output.clone(),
                    inpoint: InPoint::Withdrawal { m6id },
                };
                state.stxos.put(rwtxn, &outpoint_key, &spent_output)?;
                if !state.delete_utxo(rwtxn, outpoint)? {
                    return Err(Error::NoUtxo {
                        outpoint: *outpoint,
                    });
                };
            }
            let (prev_latest_failed_m6id, latest_failed_m6id) = state
                .latest_failed_withdrawal_bundle
                .try_get(rwtxn, &())?
                .ok_or_else(|| {
                    Error::DatabaseError(
                        "latest failed withdrawal bundle should exist"
                            .to_string(),
                    )
                })?
                .pop();
            if latest_failed_m6id.value != m6id {
                return Err(Error::DatabaseError(format!(
                    "latest failed m6id {:?} != expected {m6id:?}",
                    latest_failed_m6id.value,
                )));
            }
            if latest_failed_m6id.height != block_height {
                return Err(Error::DatabaseError(format!(
                    "latest failed height {} != block height \
                     {block_height}",
                    latest_failed_m6id.height,
                )));
            }
            if let Some(prev_latest_failed_m6id) = prev_latest_failed_m6id {
                state.latest_failed_withdrawal_bundle.put(
                    rwtxn,
                    &(),
                    &prev_latest_failed_m6id,
                )?;
            } else {
                state.latest_failed_withdrawal_bundle.delete(rwtxn, &())?;
            }
        }
    }
    state.withdrawal_bundles.put(
        rwtxn,
        &m6id,
        &(bundle, prev_bundle_status),
    )?;
    Ok(())
}

fn disconnect_withdrawal_bundle_event(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    event: &WithdrawalBundleEvent,
) -> Result<(), Error> {
    match event.status {
        WithdrawalBundleEventStatus::Submitted => {
            disconnect_withdrawal_bundle_submitted(
                state,
                rwtxn,
                block_height,
                event.m6id,
            )
        }
        WithdrawalBundleEventStatus::Confirmed => {
            disconnect_withdrawal_bundle_confirmed(
                state,
                rwtxn,
                block_height,
                event.m6id,
            )
        }
        WithdrawalBundleEventStatus::Failed => {
            disconnect_withdrawal_bundle_failed(
                state,
                rwtxn,
                block_height,
                event.m6id,
            )
        }
    }
}

fn disconnect_event(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    latest_deposit_block_hash: &mut Option<bitcoin::BlockHash>,
    latest_withdrawal_bundle_event_block_hash: &mut Option<bitcoin::BlockHash>,
    event_block_hash: bitcoin::BlockHash,
    event: &BlockEvent,
) -> Result<(), Error> {
    match event {
        BlockEvent::Deposit(deposit) => {
            let outpoint = OutPoint::Deposit(deposit.outpoint);
            if !state.delete_utxo(rwtxn, &outpoint)? {
                return Err(Error::NoUtxo { outpoint });
            }
            if latest_deposit_block_hash.is_none() {
                *latest_deposit_block_hash = Some(event_block_hash);
            }
        }
        BlockEvent::WithdrawalBundle(withdrawal_bundle_event) => {
            let () = disconnect_withdrawal_bundle_event(
                state,
                rwtxn,
                block_height,
                withdrawal_bundle_event,
            )?;
            if latest_withdrawal_bundle_event_block_hash.is_none() {
                *latest_withdrawal_bundle_event_block_hash =
                    Some(event_block_hash);
            }
        }
    }
    Ok(())
}

pub fn disconnect(
    state: &State,
    rwtxn: &mut RwTxn,
    two_way_peg_data: &TwoWayPegData,
) -> Result<(), Error> {
    let block_height = state.try_get_height(rwtxn)?.ok_or(Error::NoTip)?;
    state.block_index_events.delete(rwtxn, &block_height)?;
    let mut latest_deposit_block_hash = None;
    let mut latest_withdrawal_bundle_event_block_hash = None;
    for (event_block_hash, event_block_info) in
        two_way_peg_data.block_info.iter().rev()
    {
        for event in event_block_info.events.iter().rev() {
            let () = disconnect_event(
                state,
                rwtxn,
                block_height,
                &mut latest_deposit_block_hash,
                &mut latest_withdrawal_bundle_event_block_hash,
                *event_block_hash,
                event,
            )?;
        }
    }
    if let Some(latest_withdrawal_bundle_event_block_hash) =
        latest_withdrawal_bundle_event_block_hash
    {
        let (
            last_withdrawal_bundle_event_block_seq_idx,
            (
                last_withdrawal_bundle_event_block_hash,
                last_withdrawal_bundle_event_block_height,
            ),
        ) = state
            .withdrawal_bundle_event_blocks
            .last(rwtxn)?
            .ok_or(Error::NoWithdrawalBundleEventBlock)?;
        if latest_withdrawal_bundle_event_block_hash
            != last_withdrawal_bundle_event_block_hash
        {
            return Err(Error::DatabaseError(format!(
                "withdrawal bundle event block hash mismatch: \
                 {latest_withdrawal_bundle_event_block_hash} != \
                 {last_withdrawal_bundle_event_block_hash}"
            )));
        }
        if block_height != last_withdrawal_bundle_event_block_height {
            return Err(Error::DatabaseError(format!(
                "withdrawal bundle event block height mismatch: \
                 {block_height} != \
                 {last_withdrawal_bundle_event_block_height}"
            )));
        }
        if !state
            .withdrawal_bundle_event_blocks
            .delete(rwtxn, &last_withdrawal_bundle_event_block_seq_idx)?
        {
            return Err(Error::NoWithdrawalBundleEventBlock);
        };
    }
    let last_withdrawal_bundle_failure_height = state
        .get_latest_failed_withdrawal_bundle(rwtxn)?
        .map(|(height, _bundle)| height)
        .unwrap_or_default();
    if block_height - last_withdrawal_bundle_failure_height
        >= WITHDRAWAL_BUNDLE_FAILURE_GAP
        && let Some(bundle_m6id) =
            state.pending_withdrawal_bundle.try_get(rwtxn, &())?
        && let (bundle, bundle_status) = state
            .withdrawal_bundles
            .try_get(rwtxn, &bundle_m6id)?
            .ok_or_else(|| {
                Error::DatabaseError(format!(
                    "pending withdrawal bundle {bundle_m6id} \
                     unknown in withdrawal_bundles"
                ))
            })?
        && bundle_status.latest().height == block_height
    {
        state.pending_withdrawal_bundle.delete(rwtxn, &())?;
        if let (Some(bundle_status), _latest_bundle_status) =
            bundle_status.pop()
        {
            state.withdrawal_bundles.put(
                rwtxn,
                &bundle_m6id,
                &(bundle, bundle_status),
            )?;
        } else {
            state.withdrawal_bundles.delete(rwtxn, &bundle_m6id)?;
        }
    }
    if let Some(latest_deposit_block_hash) = latest_deposit_block_hash {
        let (
            last_deposit_block_seq_idx,
            (last_deposit_block_hash, last_deposit_block_height),
        ) = state
            .deposit_blocks
            .last(rwtxn)?
            .ok_or(Error::NoDepositBlock)?;
        if latest_deposit_block_hash != last_deposit_block_hash {
            return Err(Error::DatabaseError(format!(
                "deposit block hash mismatch: \
                 {latest_deposit_block_hash} != \
                 {last_deposit_block_hash}"
            )));
        }
        if block_height != last_deposit_block_height {
            return Err(Error::DatabaseError(format!(
                "deposit block height mismatch: {block_height} != \
                 {last_deposit_block_height}"
            )));
        }
        if !state
            .deposit_blocks
            .delete(rwtxn, &last_deposit_block_seq_idx)?
        {
            return Err(Error::NoDepositBlock);
        };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use bitcoin::{
        Network,
        hashes::Hash as _,
        secp256k1::{Secp256k1, SecretKey},
    };
    use hashlink::LinkedHashMap;

    use crate::{
        archive::Archive,
        state::{
            State, UtxoManager as _, WithdrawalBundleInfo,
            rollback::{HeightStamped, RollBack},
            two_way_peg_data::{
                collect_withdrawal_bundle, connect, disconnect,
                disconnect_withdrawal_bundle_failed,
            },
        },
        types::{
            Address, BitcoinOutputContent, BlockIndexEvents, FilledOutput,
            FilledOutputContent, Hash, InPoint, M6id, OutPoint, OutPointKey,
            Txid, WithdrawalBundle, WithdrawalBundleEvent,
            WithdrawalBundleEventStatus, WithdrawalBundleStatus,
            WithdrawalOutputContent,
            proto::mainchain::{BlockEvent, BlockInfo, TwoWayPegData},
        },
    };

    // open a fresh state-backed env in a unique temp dir
    fn temp_env() -> (sneed::Env, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let env_path = dir.path().join("data.mdb");
        std::fs::create_dir_all(&env_path).unwrap();
        let mut opts = heed::EnvOpenOptions::new();
        opts.map_size(64 * 1024 * 1024)
            .max_dbs(State::NUM_DBS + Archive::NUM_DBS);
        let env = unsafe { sneed::Env::open(&opts, &env_path) }.unwrap();
        (env, dir)
    }

    // a failed known bundle reinstates its utxos as spendable, so disconnecting
    // the failure must spend them again
    #[test]
    fn disconnect_failed_bundle_spends_reinstated_utxo() {
        let (env, _dir) = temp_env();
        let state = State::new(&env, None).unwrap();
        let outpoint = OutPoint::Regular {
            txid: Txid(Hash::from([1; 32])),
            vout: 0,
        };
        let output = FilledOutput {
            address: Address::ALL_ZEROS,
            content: FilledOutputContent::Bitcoin(BitcoinOutputContent(
                bitcoin::Amount::from_sat(1000),
            )),
            memo: Vec::new(),
        };
        let key = OutPointKey::from_outpoint(&outpoint);

        let m6id = {
            let mut spend_utxos = BTreeMap::new();
            spend_utxos.insert(outpoint, output.clone());
            let bundle = WithdrawalBundle::new(
                1,
                bitcoin::Amount::ZERO,
                spend_utxos,
                Vec::new(),
            )
            .unwrap();
            let m6id = bundle.compute_m6id();
            let mut bundle_status = RollBack::<HeightStamped<_>>::new(
                WithdrawalBundleStatus::Submitted,
                0,
            );
            bundle_status
                .push(WithdrawalBundleStatus::Failed, 1)
                .unwrap();
            let mut rwtxn = env.write_txn().unwrap();
            state
                .withdrawal_bundles
                .put(
                    &mut rwtxn,
                    &m6id,
                    &(WithdrawalBundleInfo::Known(bundle), bundle_status),
                )
                .unwrap();
            state
                .latest_failed_withdrawal_bundle
                .put(
                    &mut rwtxn,
                    &(),
                    &RollBack::<HeightStamped<_>>::new(m6id, 1),
                )
                .unwrap();
            // the failure reinstated the utxo
            state.insert_utxo(&mut rwtxn, &outpoint, &output).unwrap();
            rwtxn.commit().unwrap();
            m6id
        };

        let mut rwtxn = env.write_txn().unwrap();
        disconnect_withdrawal_bundle_failed(&state, &mut rwtxn, 1, m6id)
            .unwrap();
        assert!(state.utxos.try_get(&rwtxn, &key).unwrap().is_none());
        let stxo = state.stxos.try_get(&rwtxn, &key).unwrap().unwrap();
        assert_eq!(stxo.inpoint, InPoint::Withdrawal { m6id });
        rwtxn.commit().unwrap();
    }

    // disconnecting a withdrawal bundle event must remove its
    // withdrawal_bundle_event_blocks record, not a deposit_blocks record that
    // happens to share the same sequence index
    #[test]
    fn disconnect_withdrawal_event_block_uses_correct_db() {
        let (env, _dir) = temp_env();
        let state = State::new(&env, None).unwrap();

        let block_height = 5u32;
        let m6id = M6id(bitcoin::Txid::from_byte_array([7; 32]));
        let event_block_hash = bitcoin::BlockHash::from_byte_array([9; 32]);
        let deposit_block_hash = bitcoin::BlockHash::from_byte_array([3; 32]);

        let mut rwtxn = env.write_txn().unwrap();
        state.height.put(&mut rwtxn, &(), &block_height).unwrap();
        state
            .withdrawal_bundles
            .put(
                &mut rwtxn,
                &m6id,
                &(
                    WithdrawalBundleInfo::Unknown,
                    RollBack::<HeightStamped<_>>::new(
                        WithdrawalBundleStatus::Submitted,
                        block_height,
                    ),
                ),
            )
            .unwrap();
        state
            .withdrawal_bundle_event_blocks
            .put(&mut rwtxn, &0, &(event_block_hash, block_height))
            .unwrap();
        // a deposit record at the same sequence index that must survive
        state
            .deposit_blocks
            .put(&mut rwtxn, &0, &(deposit_block_hash, block_height))
            .unwrap();
        rwtxn.commit().unwrap();

        let two_way_peg_data = {
            let mut block_info = LinkedHashMap::new();
            block_info.insert(
                event_block_hash,
                BlockInfo {
                    bmm_commitment: None,
                    events: vec![BlockEvent::WithdrawalBundle(
                        WithdrawalBundleEvent {
                            m6id,
                            status: WithdrawalBundleEventStatus::Submitted,
                        },
                    )],
                },
            );
            TwoWayPegData { block_info }
        };

        let mut rwtxn = env.write_txn().unwrap();
        disconnect(&state, &mut rwtxn, &two_way_peg_data).unwrap();
        assert!(
            state
                .withdrawal_bundle_event_blocks
                .try_get(&rwtxn, &0)
                .unwrap()
                .is_none()
        );
        assert!(state.deposit_blocks.try_get(&rwtxn, &0).unwrap().is_some());
        rwtxn.commit().unwrap();
    }

    fn seeded_public_key(idx: u32) -> bitcoin::CompressedPublicKey {
        let secp = Secp256k1::new();
        let mut key_bytes = [0_u8; 32];
        key_bytes[28..].copy_from_slice(&idx.to_be_bytes());
        let secret_key = SecretKey::from_slice(&key_bytes)
            .expect("small non-zero integers are valid secret keys");
        let public_key =
            bitcoin::secp256k1::PublicKey::from_secret_key(&secp, &secret_key);
        bitcoin::CompressedPublicKey(public_key)
    }

    fn regtest_p2wpkh_address(
        idx: u32,
    ) -> bitcoin::Address<bitcoin::address::NetworkUnchecked> {
        let public_key = seeded_public_key(idx);
        bitcoin::Address::p2wpkh(&public_key, Network::Regtest).into_unchecked()
    }

    fn with_state_with_withdrawals<R>(
        count: u32,
        main_address: fn(
            u32,
        ) -> bitcoin::Address<
            bitcoin::address::NetworkUnchecked,
        >,
        f: impl FnOnce(&State, &mut sneed::RwTxn<'_>) -> R,
    ) -> anyhow::Result<R> {
        let (env, _dir) = temp_env();
        let state = State::new(&env, None)?;
        let mut rwtxn = env.write_txn()?;
        state.height.put(
            &mut rwtxn,
            &(),
            &crate::state::WITHDRAWAL_BUNDLE_FAILURE_GAP,
        )?;

        for idx in 1..=count {
            let mut txid_bytes = [0_u8; 32];
            txid_bytes[28..].copy_from_slice(&idx.to_be_bytes());
            let outpoint = OutPoint::Regular {
                txid: Txid(Hash::from(txid_bytes)),
                vout: 0,
            };
            let output = FilledOutput {
                address: Address::ALL_ZEROS,
                content: FilledOutputContent::BitcoinWithdrawal(
                    WithdrawalOutputContent {
                        value: bitcoin::Amount::from_sat(1_000),
                        main_fee: bitcoin::Amount::ZERO,
                        main_address: main_address(idx),
                    },
                ),
                memo: Vec::new(),
            };
            state.utxos.put(
                &mut rwtxn,
                &OutPointKey::from_outpoint(&outpoint),
                &output,
            )?;
        }
        Ok(f(&state, &mut rwtxn))
    }

    #[test]
    fn collect_withdrawal_bundle_p2wpkh_off_by_one_does_not_exceed_weight()
    -> anyhow::Result<()> {
        const CLAIMED_MAX_BUNDLE_OUTPUTS: u32 = 3_222;

        let bundle = with_state_with_withdrawals(
            CLAIMED_MAX_BUNDLE_OUTPUTS + 1,
            regtest_p2wpkh_address,
            |state, rwtxn| collect_withdrawal_bundle(state, rwtxn, 42),
        )?;
        let bundle = match bundle {
            Ok(Some(bundle)) => bundle,
            Ok(None) => anyhow::bail!("expected a withdrawal bundle"),
            Err(err) => anyhow::bail!("unexpected collection error: {err:?}"),
        };
        let output_count = bundle.tx().output.len();
        let weight = bundle.tx().weight().to_wu();

        anyhow::ensure!(
            output_count == (CLAIMED_MAX_BUNDLE_OUTPUTS as usize + 2),
            "expected {} tx outputs including metadata, got {output_count}",
            CLAIMED_MAX_BUNDLE_OUTPUTS as usize + 2,
        );
        anyhow::ensure!(
            weight <= bitcoin::policy::MAX_STANDARD_TX_WEIGHT as u64,
            "unexpected overweight P2WPKH bundle: {weight} wu"
        );
        Ok(())
    }

    // connecting a deposit then disconnecting it on a reorg must round-trip
    #[test]
    fn deposit_reorg_round_trips() {
        use crate::types::{Body, Header, proto::mainchain::Deposit};

        let (env, _dir) = temp_env();
        let archive = Archive::new(&env).unwrap();
        let state = State::new(&env, None).unwrap();

        let empty_body = Body {
            coinbase: Vec::new(),
            transactions: Vec::new(),
            authorizations: Vec::new(),
            actor_proofs: Vec::new(),
        };
        let merkle_root = Body::compute_merkle_root(
            &empty_body.coinbase,
            &empty_body.transactions,
        );
        let main0 = bitcoin::BlockHash::from_byte_array([10; 32]);
        let main1 = bitcoin::BlockHash::from_byte_array([11; 32]);

        let genesis = Header {
            merkle_root,
            prev_side_hash: None,
            prev_main_hash: main0,
        };
        {
            let mut rwtxn = env.write_txn().unwrap();
            state
                .apply_block(&archive, &mut rwtxn, &genesis, &empty_body, 0)
                .unwrap();
            state
                .connect_two_way_peg_data(&mut rwtxn, &TwoWayPegData::default())
                .unwrap();
            rwtxn.commit().unwrap();
        }

        let block1 = Header {
            merkle_root,
            prev_side_hash: Some(genesis.hash()),
            prev_main_hash: main1,
        };
        let deposit_outpoint = bitcoin::OutPoint {
            txid: bitcoin::Txid::from_byte_array([2; 32]),
            vout: 0,
        };
        let deposit_key =
            OutPointKey::from_outpoint(&OutPoint::Deposit(deposit_outpoint));
        let deposit_twpd = {
            let mut block_info = LinkedHashMap::new();
            block_info.insert(
                main1,
                BlockInfo {
                    bmm_commitment: None,
                    events: vec![BlockEvent::Deposit(Deposit {
                        tx_index: 0,
                        outpoint: deposit_outpoint,
                        output: FilledOutput {
                            address: Address::ALL_ZEROS,
                            content: FilledOutputContent::Bitcoin(
                                BitcoinOutputContent(
                                    bitcoin::Amount::from_sat(1000),
                                ),
                            ),
                            memo: Vec::new(),
                        },
                    })],
                },
            );
            TwoWayPegData { block_info }
        };
        {
            let mut rwtxn = env.write_txn().unwrap();
            state
                .apply_block(&archive, &mut rwtxn, &block1, &empty_body, 0)
                .unwrap();
            state
                .connect_two_way_peg_data(&mut rwtxn, &deposit_twpd)
                .unwrap();
            assert!(
                state.utxos.try_get(&rwtxn, &deposit_key).unwrap().is_some()
            );
            assert!(state.deposit_blocks.last(&rwtxn).unwrap().is_some());
            rwtxn.commit().unwrap();
        }

        {
            let mut rwtxn = env.write_txn().unwrap();
            state
                .disconnect_two_way_peg_data(&mut rwtxn, &deposit_twpd)
                .unwrap();
            assert!(
                state.utxos.try_get(&rwtxn, &deposit_key).unwrap().is_none()
            );
            assert!(state.deposit_blocks.last(&rwtxn).unwrap().is_none());
            rwtxn.commit().unwrap();
        }
    }

    // A single two-way-peg batch can span multiple mainchain blocks. Connecting
    // deposits from two distinct blocks then disconnecting the batch must
    // restore the prior state. Before the fix, disconnect recomputed the latest
    // deposit block hash by reverse iteration (yielding the oldest block) and
    // failed the consistency check against the newest hash connect stored.
    #[test]
    fn disconnect_two_deposit_blocks_restores_state() -> anyhow::Result<()> {
        use crate::types::proto::mainchain::Deposit;

        fn deposit_block(salt: u8) -> (bitcoin::BlockHash, BlockInfo) {
            let dep = Deposit {
                tx_index: 0,
                outpoint: bitcoin::OutPoint {
                    txid: bitcoin::Txid::from_byte_array([salt; 32]),
                    vout: 0,
                },
                output: FilledOutput {
                    address: Address::ALL_ZEROS,
                    content: FilledOutputContent::Bitcoin(
                        BitcoinOutputContent(bitcoin::Amount::from_sat(1000)),
                    ),
                    memo: Vec::new(),
                },
            };
            (
                bitcoin::BlockHash::from_byte_array([salt; 32]),
                BlockInfo {
                    bmm_commitment: None,
                    events: vec![BlockEvent::Deposit(dep)],
                },
            )
        }
        let (env, _dir) = temp_env();
        let state = State::new(&env, None)?;
        let mut rwtxn = env.write_txn()?;
        state.height.put(&mut rwtxn, &(), &10)?;

        let mut block_info = LinkedHashMap::new();
        let (h1, b1) = deposit_block(1);
        let (h2, b2) = deposit_block(2);
        block_info.insert(h1, b1);
        block_info.insert(h2, b2);
        let tdp = TwoWayPegData { block_info };

        let () = connect(&state, &mut rwtxn, &tdp)?;
        anyhow::ensure!(state.utxos.len(&rwtxn)? == 2);
        disconnect(&state, &mut rwtxn, &tdp)?;

        anyhow::ensure!(state.utxos.len(&rwtxn)? == 0);
        anyhow::ensure!(state.deposit_blocks.len(&rwtxn)? == 0);
        Ok(())
    }

    #[test]
    fn block_index_events_round_trip() -> anyhow::Result<()> {
        let (env, _dir) = temp_env();
        let state = State::new(&env, None)?;
        let deposit_outpoint = |byte: u8| {
            OutPoint::Deposit(bitcoin::OutPoint {
                txid: bitcoin::Txid::from_byte_array([byte; 32]),
                vout: 0,
            })
        };
        let events = BlockIndexEvents {
            deposits: vec![(
                deposit_outpoint(1),
                FilledOutput {
                    address: Address::ALL_ZEROS,
                    content: FilledOutputContent::Bitcoin(
                        BitcoinOutputContent(bitcoin::Amount::from_sat(5000)),
                    ),
                    memo: Vec::new(),
                },
            )],
            bundle_spends: vec![(
                deposit_outpoint(2),
                M6id(bitcoin::Txid::from_byte_array([3; 32])),
            )],
        };
        {
            let mut rwtxn = env.write_txn()?;
            state.block_index_events.put(&mut rwtxn, &7, &events)?;
            rwtxn.commit()?;
        }
        {
            let rotxn = env.read_txn()?;
            anyhow::ensure!(state.get_block_index_events(&rotxn, 7)? == events);
            // A height that moved nothing outside its body reads as empty.
            anyhow::ensure!(
                state.get_block_index_events(&rotxn, 8)?.is_empty()
            );
        }

        // A disconnect drops the events, so a reorg leaves nothing behind for
        // the block that takes the height.
        {
            let mut rwtxn = env.write_txn()?;
            state.block_index_events.delete(&mut rwtxn, &7)?;
            rwtxn.commit()?;
        }
        let rotxn = env.read_txn()?;
        anyhow::ensure!(state.get_block_index_events(&rotxn, 7)?.is_empty());

        // A height that moved nothing writes no row, so deleting it again is
        // still safe.
        {
            let mut rwtxn = env.write_txn()?;
            state.block_index_events.delete(&mut rwtxn, &8)?;
            rwtxn.commit()?;
        }
        Ok(())
    }
}
