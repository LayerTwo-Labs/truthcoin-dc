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
        AggregatedWithdrawal, AmountOverflowError, FilledOutput,
        FilledOutputContent, InPoint, M6id, OutPoint, SpentOutput,
        WithdrawalBundle, WithdrawalBundleEvent, WithdrawalBundleStatus,
        WithdrawalOutputContent,
        proto::mainchain::{BlockEvent, TwoWayPegData},
    },
};

fn collect_withdrawal_bundle(
    state: &State,
    txn: &RoTxn,
    block_height: u32,
) -> Result<Option<WithdrawalBundle>, Error> {
    const BUNDLE_0_WEIGHT: u64 = 504;
    const OUTPUT_WEIGHT: u64 = 128;
    const MAX_BUNDLE_OUTPUTS: usize =
        ((bitcoin::policy::MAX_STANDARD_TX_WEIGHT as u64 - BUNDLE_0_WEIGHT)
            / OUTPUT_WEIGHT) as usize;

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
                aggregated.spend_utxos.insert(outpoint, output);
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
    let mut bundle_outputs = vec![];
    for aggregated in &aggregated_withdrawals {
        if bundle_outputs.len() > MAX_BUNDLE_OUTPUTS {
            break;
        }
        let bundle_output = bitcoin::TxOut {
            value: aggregated.value,
            script_pubkey: aggregated
                .main_address
                .assume_checked_ref()
                .script_pubkey(),
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
    _event_block_hash: &bitcoin::BlockHash,
    m6id: M6id,
) -> Result<(), Error> {
    if let Some((bundle, bundle_block_height)) =
        state.pending_withdrawal_bundle.try_get(rwtxn, &())?
        && bundle.compute_m6id() == m6id
    {
        assert_eq!(bundle_block_height, block_height - 1);
        for (outpoint, spend_output) in bundle.spend_utxos() {
            state.delete_utxo_with_address_index(rwtxn, outpoint)?;
            let spent_output = SpentOutput {
                output: spend_output.clone(),
                inpoint: InPoint::Withdrawal { m6id },
            };
            state.stxos.put(rwtxn, outpoint, &spent_output)?;
        }
        state.withdrawal_bundles.put(
            rwtxn,
            &m6id,
            &(
                WithdrawalBundleInfo::Known(bundle),
                RollBack::<HeightStamped<_>>::new(
                    WithdrawalBundleStatus::Submitted,
                    block_height,
                ),
            ),
        )?;
        state.pending_withdrawal_bundle.delete(rwtxn, &())?;
    } else if let Some((_bundle, bundle_status)) =
        state.withdrawal_bundles.try_get(rwtxn, &m6id)?
    {
        assert_eq!(
            bundle_status.earliest().value,
            WithdrawalBundleStatus::Submitted
        );
    } else {
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
    assert_eq!(
        bundle_status.latest().value,
        WithdrawalBundleStatus::Submitted
    );
    if !bundle.is_known() {
        if block_height == 0 {
            let utxos: BTreeMap<_, _> = state.utxos.iter(rwtxn)?.collect()?;
            for (outpoint, output) in &utxos {
                let spent_output = SpentOutput {
                    output: output.clone(),
                    inpoint: InPoint::Withdrawal { m6id },
                };
                state.stxos.put(rwtxn, outpoint, &spent_output)?;
            }
            state.clear_utxos_and_address_index(rwtxn)?;
            bundle =
                WithdrawalBundleInfo::UnknownConfirmed { spend_utxos: utxos };
        } else {
            return Err(Error::UnknownWithdrawalBundleConfirmed {
                event_block_hash: *event_block_hash,
                m6id,
            });
        }
    }
    bundle_status
        .push(WithdrawalBundleStatus::Confirmed, block_height)
        .expect("Push confirmed status should be valid");
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
    assert_eq!(
        bundle_status.latest().value,
        WithdrawalBundleStatus::Submitted
    );
    bundle_status
        .push(WithdrawalBundleStatus::Failed, block_height)
        .expect("Push failed status should be valid");
    match &bundle {
        WithdrawalBundleInfo::Unknown
        | WithdrawalBundleInfo::UnknownConfirmed { .. } => (),
        WithdrawalBundleInfo::Known(bundle) => {
            for (outpoint, output) in bundle.spend_utxos() {
                state.stxos.delete(rwtxn, outpoint)?;
                state
                    .insert_utxo_with_address_index(rwtxn, outpoint, output)?;
            }
            let latest_failed_m6id = if let Some(mut latest_failed_m6id) =
                state.latest_failed_withdrawal_bundle.try_get(rwtxn, &())?
            {
                latest_failed_m6id
                    .push(m6id, block_height)
                    .expect("Push latest failed m6id should be valid");
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
    state
        .withdrawal_bundles
        .put(rwtxn, &m6id, &(bundle, bundle_status))?;
    Ok(())
}

fn connect_withdrawal_bundle_event(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    event_block_hash: &bitcoin::BlockHash,
    event: &WithdrawalBundleEvent,
) -> Result<(), Error> {
    match event.status {
        WithdrawalBundleStatus::Submitted => {
            connect_withdrawal_bundle_submitted(
                state,
                rwtxn,
                block_height,
                event_block_hash,
                event.m6id,
            )
        }
        WithdrawalBundleStatus::Confirmed => {
            connect_withdrawal_bundle_confirmed(
                state,
                rwtxn,
                block_height,
                event_block_hash,
                event.m6id,
            )
        }
        WithdrawalBundleStatus::Failed => connect_withdrawal_bundle_failed(
            state,
            rwtxn,
            block_height,
            event.m6id,
        ),
    }
}

fn connect_2wpd_event(
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
            let output = deposit.output.clone();
            state.insert_utxo_with_address_index(rwtxn, &outpoint, &output)?;
            *latest_deposit_block_hash = Some(event_block_hash);
        }
        BlockEvent::WithdrawalBundle(withdrawal_bundle_event) => {
            let () = connect_withdrawal_bundle_event(
                state,
                rwtxn,
                block_height,
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
    let mut latest_deposit_block_hash = None;
    let mut latest_withdrawal_bundle_event_block_hash = None;
    for (event_block_hash, event_block_info) in &two_way_peg_data.block_info {
        for event in &event_block_info.events {
            let () = connect_2wpd_event(
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
        state.pending_withdrawal_bundle.put(
            rwtxn,
            &(),
            &(bundle, block_height),
        )?;
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
        if let Some((bundle, _)) =
            state.pending_withdrawal_bundle.try_get(rwtxn, &())?
            && bundle.compute_m6id() == m6id
        {
            return Ok(());
        } else {
            return Err(Error::UnknownWithdrawalBundle { m6id });
        }
    };
    let bundle_status = bundle_status.latest();
    assert_eq!(bundle_status.value, WithdrawalBundleStatus::Submitted);
    assert_eq!(bundle_status.height, block_height);
    match bundle {
        WithdrawalBundleInfo::Unknown
        | WithdrawalBundleInfo::UnknownConfirmed { .. } => (),
        WithdrawalBundleInfo::Known(bundle) => {
            for (outpoint, output) in bundle.spend_utxos().iter().rev() {
                if !state.stxos.delete(rwtxn, outpoint)? {
                    return Err(Error::NoStxo {
                        outpoint: *outpoint,
                    });
                };
                state
                    .insert_utxo_with_address_index(rwtxn, outpoint, output)?;
            }
            state.pending_withdrawal_bundle.put(
                rwtxn,
                &(),
                &(bundle, bundle_status.height - 1),
            )?;
        }
    }
    state.withdrawal_bundles.delete(rwtxn, &m6id)?;
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
    if latest_bundle_status.value == WithdrawalBundleStatus::Submitted {
        return Ok(());
    }
    assert_eq!(
        latest_bundle_status.value,
        WithdrawalBundleStatus::Confirmed
    );
    assert_eq!(latest_bundle_status.height, block_height);
    let prev_bundle_status = prev_bundle_status
        .expect("Pop confirmed bundle status should be valid");
    assert_eq!(
        prev_bundle_status.latest().value,
        WithdrawalBundleStatus::Submitted
    );
    match bundle {
        WithdrawalBundleInfo::Known(_) | WithdrawalBundleInfo::Unknown => (),
        WithdrawalBundleInfo::UnknownConfirmed { spend_utxos } => {
            for (outpoint, output) in spend_utxos {
                state.insert_utxo_with_address_index(
                    rwtxn, &outpoint, &output,
                )?;
                if !state.stxos.delete(rwtxn, &outpoint)? {
                    return Err(Error::NoStxo { outpoint });
                };
            }
            bundle = WithdrawalBundleInfo::Unknown;
        }
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
    } else {
        assert_eq!(latest_bundle_status.value, WithdrawalBundleStatus::Failed);
    }
    assert_eq!(latest_bundle_status.height, block_height);
    let prev_bundle_status =
        prev_bundle_status.expect("Pop failed bundle status should be valid");
    assert_eq!(
        prev_bundle_status.latest().value,
        WithdrawalBundleStatus::Submitted
    );
    match &bundle {
        WithdrawalBundleInfo::Unknown
        | WithdrawalBundleInfo::UnknownConfirmed { .. } => (),
        WithdrawalBundleInfo::Known(bundle) => {
            for (outpoint, output) in bundle.spend_utxos().iter().rev() {
                let spent_output = SpentOutput {
                    output: output.clone(),
                    inpoint: InPoint::Withdrawal { m6id },
                };
                state.stxos.put(rwtxn, outpoint, &spent_output)?;
                if state.delete_utxo_with_address_index(rwtxn, outpoint)? {
                    return Err(Error::NoUtxo {
                        outpoint: *outpoint,
                    });
                };
            }
            let (prev_latest_failed_m6id, latest_failed_m6id) = state
                .latest_failed_withdrawal_bundle
                .try_get(rwtxn, &())?
                .expect("latest failed withdrawal bundle should exist")
                .pop();
            assert_eq!(latest_failed_m6id.value, m6id);
            assert_eq!(latest_failed_m6id.height, block_height);
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
        WithdrawalBundleStatus::Submitted => {
            disconnect_withdrawal_bundle_submitted(
                state,
                rwtxn,
                block_height,
                event.m6id,
            )
        }
        WithdrawalBundleStatus::Confirmed => {
            disconnect_withdrawal_bundle_confirmed(
                state,
                rwtxn,
                block_height,
                event.m6id,
            )
        }
        WithdrawalBundleStatus::Failed => disconnect_withdrawal_bundle_failed(
            state,
            rwtxn,
            block_height,
            event.m6id,
        ),
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
            if !state.delete_utxo_with_address_index(rwtxn, &outpoint)? {
                return Err(Error::NoUtxo { outpoint });
            }
            *latest_deposit_block_hash = Some(event_block_hash);
        }
        BlockEvent::WithdrawalBundle(withdrawal_bundle_event) => {
            let () = disconnect_withdrawal_bundle_event(
                state,
                rwtxn,
                block_height,
                withdrawal_bundle_event,
            )?;
            *latest_withdrawal_bundle_event_block_hash = Some(event_block_hash);
        }
    }
    Ok(())
}

pub fn disconnect(
    state: &State,
    rwtxn: &mut RwTxn,
    two_way_peg_data: &TwoWayPegData,
) -> Result<(), Error> {
    let block_height = state
        .try_get_height(rwtxn)?
        .expect("Height should not be None");
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
        assert_eq!(
            latest_withdrawal_bundle_event_block_hash,
            last_withdrawal_bundle_event_block_hash
        );
        assert_eq!(block_height - 1, last_withdrawal_bundle_event_block_height);
        if !state
            .deposit_blocks
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
        > WITHDRAWAL_BUNDLE_FAILURE_GAP
        && let Some((_bundle, bundle_height)) =
            state.pending_withdrawal_bundle.try_get(rwtxn, &())?
        && bundle_height == block_height - 1
    {
        state.pending_withdrawal_bundle.delete(rwtxn, &())?;
    }
    if let Some(latest_deposit_block_hash) = latest_deposit_block_hash {
        let (
            last_deposit_block_seq_idx,
            (last_deposit_block_hash, last_deposit_block_height),
        ) = state
            .deposit_blocks
            .last(rwtxn)?
            .ok_or(Error::NoDepositBlock)?;
        assert_eq!(latest_deposit_block_hash, last_deposit_block_hash);
        assert_eq!(block_height - 1, last_deposit_block_height);
        if !state
            .deposit_blocks
            .delete(rwtxn, &last_deposit_block_seq_idx)?
        {
            return Err(Error::NoDepositBlock);
        };
    }
    Ok(())
}
