//! Connect and disconnect blocks

use std::collections::HashMap;

use sneed::{RoTxn, RwTxn, db::error::Error as DbError};

use crate::{
    state::{
        Error, PrevalidatedBlock, State, error, markets, two_way_peg_data,
    },
    types::{
        AccumulatorDiff, AmountOverflowError, Body, FilledTransaction,
        GetAddress as _, GetValue as _, Header, InPoint, MerkleRoot, OutPoint,
        OutPointKey, Output, PointedOutput, SpentOutput, UtreexoNodeHash,
        authorization::{self, BatchVerificationContext},
    },
};

/// Prevalidate a block: compute and verify all read-only checks and
/// prepare data needed for fast connection.
pub fn prevalidate(
    batch_verification_ctxt: &BatchVerificationContext,
    state: &State,
    rotxn: &RoTxn,
    header: &Header,
    prev_main_height: u32,
    body: &Body,
    archive: &crate::archive::Archive,
) -> Result<PrevalidatedBlock, Error> {
    let tip_hash = state.try_get_tip(rotxn)?;
    if header.prev_side_hash != tip_hash {
        let err = error::InvalidHeader::PrevSideHash {
            expected: tip_hash,
            received: header.prev_side_hash,
        };
        return Err(Error::InvalidHeader(err));
    };
    let next_height = state.try_get_height(rotxn)?.map_or(0, |h| h + 1);
    if sigops(body) > State::body_sigops_limit(next_height) {
        return Err(Error::TooManySigops);
    }
    let body_size = State::body_size(body)?;
    if body_size > State::body_size_limit(next_height) as u64 {
        return Err(Error::BodyTooLarge);
    }

    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rotxn, &())
        .map_err(DbError::from)?
        .unwrap_or_default();

    // gather and verify transactions
    let mut total_inputs: usize = 0;
    let mut total_outputs: usize = 0;
    for tx in &body.transactions {
        total_inputs += tx.inputs.len();
        total_outputs += tx.outputs.len();
    }
    let mut all_input_keys: Vec<OutPointKey> = Vec::with_capacity(total_inputs);
    // Accumulator diff from txs. The bool value is true for insertions, false
    // for deletions.
    let mut accumulator_diff_txs =
        Vec::with_capacity(total_inputs + total_outputs);
    let mut filled_transactions: Vec<FilledTransaction> =
        Vec::with_capacity(body.transactions.len());
    let mut total_fees = bitcoin::Amount::ZERO;
    // Outputs that earlier transactions in this body made. A transaction may
    // spend one of them, and the accumulator holds no leaf for it, so it is
    // never a proof target.
    let mut body_outputs = HashMap::<OutPoint, Output>::new();
    for (tx_idx, transaction) in body.transactions.iter().enumerate() {
        let txid = transaction.txid();
        let mut spent_utxos = Vec::with_capacity(transaction.inputs.len());
        let mut spent_utxo_hashes =
            Vec::<UtreexoNodeHash>::with_capacity(transaction.inputs.len());
        for (outpoint, utxo_hash) in &transaction.inputs {
            let key = OutPointKey::from(outpoint);
            match state.utxos.try_get(rotxn, &key)? {
                Some(spent_output) => {
                    spent_utxos.push(spent_output);
                    spent_utxo_hashes.push(utxo_hash.into());
                }
                None => {
                    let spent_output = body_outputs
                        .get(outpoint)
                        .cloned()
                        .ok_or(error::NoUtxo {
                            outpoint: *outpoint,
                        })?;
                    spent_utxos.push(spent_output);
                }
            }
            all_input_keys.push(OutPointKey::from(outpoint));
            accumulator_diff_txs.push((false, utxo_hash.into()));
        }
        for (vout, output) in transaction.outputs.iter().enumerate() {
            let outpoint = OutPoint::Regular {
                txid,
                vout: vout as u32,
            };
            let pointed_output = PointedOutput {
                outpoint,
                output: output.clone(),
            };
            accumulator_diff_txs.push((true, (&pointed_output).into()));
        }
        body_outputs.extend(transaction.outputs_by_outpoint());
        if !accumulator.verify(&transaction.proof, &spent_utxo_hashes)? {
            return Err(Error::UtreexoProofFailed { txid });
        }
        let filled_tx = FilledTransaction {
            spent_utxos,
            transaction: transaction.clone(),
            actor_address: actor_address(body, tx_idx),
        };
        total_fees = total_fees
            .checked_add(state.validate_filled_transaction(
                rotxn,
                prev_main_height,
                &filled_tx,
                archive,
                Some(next_height),
            )?)
            .ok_or(AmountOverflowError)?;
        filled_transactions.push(filled_tx);
    }
    let () =
        crate::validation::check_duplicate_decision_claims(&body.transactions)?;
    crate::validation::vote::check_duplicate_votes(&body.transactions)?;
    let computed_merkle_root = Body::compute_merkle_root(
        &body.coinbase,
        filled_transactions.as_slice(),
        &body.authorizations,
        &body.actor_proofs,
    )?;
    if computed_merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: header.merkle_root,
            computed: computed_merkle_root,
        };
        return Err(err);
    }
    {
        use rayon::prelude::ParallelSliceMut;
        all_input_keys.par_sort_unstable();
        if all_input_keys.windows(2).any(|w| w[0] == w[1]) {
            return Err(Error::UtxoDoubleSpent);
        }
    }
    let coinbase_txid = header.compute_coinbase_txid();
    let mut coinbase_value = bitcoin::Amount::ZERO;
    let mut accumulator_diff = AccumulatorDiff::with_capacity(
        body.coinbase.outputs.len() + accumulator_diff_txs.len(),
    );
    for (vout, output) in body.coinbase.outputs.iter().enumerate() {
        coinbase_value = coinbase_value
            .checked_add(output.get_value())
            .ok_or(AmountOverflowError)?;
        let outpoint = OutPoint::Coinbase {
            txid: coinbase_txid,
            vout: vout as u32,
        };
        let pointed_output = PointedOutput {
            outpoint,
            output: output.clone(),
        };
        accumulator_diff.insert((&pointed_output).into());
    }
    for (insert, utxo_hash) in accumulator_diff_txs {
        if insert {
            accumulator_diff.insert(utxo_hash);
        } else {
            accumulator_diff.remove(utxo_hash);
        }
    }
    if coinbase_value > total_fees {
        return Err(Error::NotEnoughFees);
    }
    let spent_utxos = filled_transactions
        .iter()
        .flat_map(|t| t.spent_utxos.iter());
    for (authorization, spent_utxo) in
        body.authorizations.iter().zip(spent_utxos)
    {
        if authorization.get_address() != spent_utxo.address {
            return Err(Error::WrongPubKeyForAddress);
        }
    }
    let () = authorization::verify_body(batch_verification_ctxt, body)?;
    // Withdrawal bundles are constructed after the block's
    // transactions, and the header roots commit to the
    // post-construction accumulator: simulate the construction here.
    let construction_diff =
        two_way_peg_data::simulate_withdrawal_bundle_construction(
            state,
            rotxn,
            next_height,
            &filled_transactions,
            &body.coinbase,
            coinbase_txid,
        )?;
    // Check root consistency without committing to DB
    let () = accumulator.apply_diff(accumulator_diff.clone())?;
    let () = accumulator.apply_diff(construction_diff)?;
    let roots: Vec<UtreexoNodeHash> = accumulator.get_roots();
    if roots != header.roots {
        return Err(Error::UtreexoRootsMismatch);
    }

    Ok(PrevalidatedBlock {
        filled_transactions,
        computed_merkle_root,
        total_fees,
        coinbase_value,
        next_height,
        accumulator_diff,
    })
}

/// Connect a block using the provided prevalidated data.
pub fn connect_prevalidated(
    state: &State,
    rwtxn: &mut RwTxn,
    header: &Header,
    body: &Body,
    pre: PrevalidatedBlock,
    mainchain_timestamp: u64,
) -> Result<crate::types::MerkleRoot, Error> {
    let tip_hash = state.try_get_tip(rwtxn)?;
    if tip_hash != header.prev_side_hash {
        let err = error::InvalidHeader::PrevSideHash {
            expected: tip_hash,
            received: header.prev_side_hash,
        };
        return Err(Error::InvalidHeader(err));
    }
    if pre.computed_merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: pre.computed_merkle_root,
            computed: header.merkle_root,
        };
        return Err(err);
    }
    let market_ctxt = markets::block::connect_begin(
        state,
        rwtxn,
        body,
        pre.next_height,
        mainchain_timestamp,
    )?;
    let coinbase_txid = header.compute_coinbase_txid();

    // Apply UTXO set changes
    for (vout, output) in body.coinbase.outputs.iter().enumerate() {
        let outpoint = OutPoint::Coinbase {
            txid: coinbase_txid,
            vout: vout as u32,
        };
        state
            .utxos
            .put(rwtxn, &OutPointKey::from(&outpoint), output)
            .map_err(DbError::from)?;
    }

    let market_txs = markets::block::connect_txs(
        state,
        rwtxn,
        &market_ctxt,
        &pre.filled_transactions,
        pre.coinbase_value,
    )?;
    let mut accumulator_diff = pre.accumulator_diff;
    for (tx_idx, filled) in pre.filled_transactions.iter().enumerate() {
        if market_txs.skipped_tx_indices.contains(&tx_idx) {
            markets::block::unapply_skipped_tx_diff(
                &mut accumulator_diff,
                &filled.transaction,
            );
            continue;
        }
        let txid = filled.transaction.txid();
        for (vin, (outpoint, _)) in filled.transaction.inputs.iter().enumerate()
        {
            let spent_output = state
                .utxos
                .try_get(rwtxn, &OutPointKey::from(outpoint))
                .map_err(DbError::from)?
                .ok_or(error::NoUtxo {
                    outpoint: *outpoint,
                })?;
            state
                .utxos
                .delete(rwtxn, &OutPointKey::from(outpoint))
                .map_err(DbError::from)?;
            let spent_output = SpentOutput {
                output: spent_output,
                inpoint: InPoint::Regular {
                    txid,
                    vin: vin as u32,
                },
            };
            state
                .stxos
                .put(rwtxn, &OutPointKey::from(outpoint), &spent_output)
                .map_err(DbError::from)?;
        }
        for (vout, output) in filled.transaction.outputs.iter().enumerate() {
            let outpoint = OutPoint::Regular {
                txid,
                vout: vout as u32,
            };
            state
                .utxos
                .put(rwtxn, &OutPointKey::from(&outpoint), output)
                .map_err(DbError::from)?;
        }
    }

    let () = markets::block::connect_end(
        state,
        rwtxn,
        &mut accumulator_diff,
        market_ctxt,
        market_txs,
    )?;

    // NOTE: the tip, height, and coinbase height are NOT updated here;
    // that happens after the block's transactions, in `connect_header`.

    // Apply accumulator diff
    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rwtxn, &())
        .map_err(DbError::from)?
        .unwrap_or_default();
    let () = accumulator.apply_diff(accumulator_diff)?;
    state
        .utreexo_accumulator
        .put(rwtxn, &(), &accumulator)
        .map_err(DbError::from)?;

    Ok(pre.computed_merkle_root)
}

/// Connect a block header: update the tip, height, and coinbase
/// height.
/// MUST be applied AFTER the block's transactions, and BEFORE
/// withdrawal bundle construction.
pub fn connect_header(
    state: &State,
    rwtxn: &mut RwTxn,
    header: &Header,
) -> Result<(), Error> {
    let block_hash = header.hash();
    let height = state.try_get_height(rwtxn)?.map_or(0, |height| height + 1);
    state.tip.put(rwtxn, &(), &block_hash)?;
    let coinbase_txid = header.compute_coinbase_txid();
    state.coinbase_height.put(rwtxn, &coinbase_txid, &height)?;
    state.height.put(rwtxn, &(), &height)?;
    Ok(())
}

pub fn validate(
    batch_verification_context: &BatchVerificationContext,
    state: &State,
    rotxn: &RoTxn,
    header: &Header,
    prev_main_height: u32,
    body: &Body,
    archive: &crate::archive::Archive,
) -> Result<(bitcoin::Amount, MerkleRoot), Error> {
    let tip_hash = state.try_get_tip(rotxn)?;
    if header.prev_side_hash != tip_hash {
        let err = error::InvalidHeader::PrevSideHash {
            expected: tip_hash,
            received: header.prev_side_hash,
        };
        return Err(Error::InvalidHeader(err));
    };
    let height = state.try_get_height(rotxn)?.map_or(0, |height| height + 1);
    if sigops(body) > State::body_sigops_limit(height) {
        return Err(Error::TooManySigops);
    }
    let body_size = State::body_size(body)?;
    if body_size > State::body_size_limit(height) as u64 {
        return Err(Error::BodyTooLarge);
    }
    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rotxn, &())
        .map_err(DbError::from)?
        .unwrap_or_default();
    // Outputs that earlier transactions in this body made, and the proof
    // targets each transaction keeps. An input that this body answers has no
    // leaf in the accumulator, so it is never a proof target.
    let mut body_outputs = HashMap::<OutPoint, Output>::new();
    let mut proof_targets: Vec<Vec<UtreexoNodeHash>> =
        Vec::with_capacity(body.transactions.len());
    let mut filled_transactions: Vec<FilledTransaction> =
        Vec::with_capacity(body.transactions.len());
    for (tx_idx, transaction) in body.transactions.iter().enumerate() {
        let mut spent_utxos = Vec::with_capacity(transaction.inputs.len());
        let mut spent_utxo_hashes =
            Vec::<UtreexoNodeHash>::with_capacity(transaction.inputs.len());
        for (outpoint, utxo_hash) in &transaction.inputs {
            let key = OutPointKey::from(outpoint);
            match state.utxos.try_get(rotxn, &key)? {
                Some(spent_output) => {
                    spent_utxos.push(spent_output);
                    spent_utxo_hashes.push(utxo_hash.into());
                }
                None => {
                    let spent_output = body_outputs
                        .get(outpoint)
                        .cloned()
                        .ok_or(error::NoUtxo {
                            outpoint: *outpoint,
                        })?;
                    spent_utxos.push(spent_output);
                }
            }
        }
        body_outputs.extend(transaction.outputs_by_outpoint());
        proof_targets.push(spent_utxo_hashes);
        filled_transactions.push(FilledTransaction {
            spent_utxos,
            transaction: transaction.clone(),
            actor_address: actor_address(body, tx_idx),
        });
    }
    let merkle_root = Body::compute_merkle_root(
        &body.coinbase,
        filled_transactions.as_slice(),
        &body.authorizations,
        &body.actor_proofs,
    )?;
    if merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: header.merkle_root,
            computed: merkle_root,
        };
        return Err(err);
    }
    let coinbase_txid = header.compute_coinbase_txid();
    let mut accumulator_diff = AccumulatorDiff::default();
    let mut coinbase_value = bitcoin::Amount::ZERO;
    for (vout, output) in body.coinbase.outputs.iter().enumerate() {
        coinbase_value = coinbase_value
            .checked_add(output.get_value())
            .ok_or(AmountOverflowError)?;
        let outpoint = OutPoint::Coinbase {
            txid: coinbase_txid,
            vout: vout as u32,
        };
        let pointed_output = PointedOutput {
            outpoint,
            output: output.clone(),
        };
        accumulator_diff.insert((&pointed_output).into());
    }
    let mut total_fees = bitcoin::Amount::ZERO;
    // Gather all input keys to check double-spends via sort-and-scan
    let total_inputs = body.inputs_len();
    let mut all_input_keys = Vec::with_capacity(total_inputs);
    for (filled_transaction, spent_utxo_hashes) in
        filled_transactions.iter().zip(&proof_targets)
    {
        let txid = filled_transaction.transaction.txid();
        for (outpoint, utxo_hash) in &filled_transaction.transaction.inputs {
            all_input_keys.push(OutPointKey::from(outpoint));
            accumulator_diff.remove(utxo_hash.into());
        }
        for (vout, output) in
            filled_transaction.transaction.outputs.iter().enumerate()
        {
            let outpoint = OutPoint::Regular {
                txid,
                vout: vout as u32,
            };
            let pointed_output = PointedOutput {
                outpoint,
                output: output.clone(),
            };
            accumulator_diff.insert((&pointed_output).into());
        }
        total_fees = total_fees
            .checked_add(state.validate_filled_transaction(
                rotxn,
                prev_main_height,
                filled_transaction,
                archive,
                Some(height),
            )?)
            .ok_or(AmountOverflowError)?;
        // verify utreexo proof
        if !accumulator
            .verify(&filled_transaction.transaction.proof, spent_utxo_hashes)?
        {
            return Err(Error::UtreexoProofFailed { txid });
        }
    }
    // Sort and check for duplicate outpoints (double-spend detection)
    {
        use rayon::prelude::ParallelSliceMut;
        all_input_keys.par_sort_unstable();
        if all_input_keys.windows(2).any(|w| w[0] == w[1]) {
            return Err(Error::UtxoDoubleSpent);
        }
    }
    let () =
        crate::validation::check_duplicate_decision_claims(&body.transactions)?;
    crate::validation::vote::check_duplicate_votes(&body.transactions)?;
    if coinbase_value > total_fees {
        return Err(Error::NotEnoughFees);
    }
    let spent_utxos = filled_transactions
        .iter()
        .flat_map(|t| t.spent_utxos.iter());
    for (authorization, spent_utxo) in
        body.authorizations.iter().zip(spent_utxos)
    {
        if authorization.get_address() != spent_utxo.address {
            return Err(Error::WrongPubKeyForAddress);
        }
    }
    let () = authorization::verify_body(batch_verification_context, body)?;
    // Check root consistency without committing to DB
    let () = accumulator.apply_diff(accumulator_diff)?;
    let roots: Vec<UtreexoNodeHash> = accumulator.get_roots();
    if roots != header.roots {
        return Err(Error::UtreexoRootsMismatch);
    }
    Ok((total_fees, merkle_root))
}

pub fn connect(
    state: &State,
    rwtxn: &mut RwTxn,
    header: &Header,
    body: &Body,
    mainchain_timestamp: u64,
) -> Result<MerkleRoot, Error> {
    let tip_hash = state.try_get_tip(rwtxn)?;
    if tip_hash != header.prev_side_hash {
        let err = error::InvalidHeader::PrevSideHash {
            expected: tip_hash,
            received: header.prev_side_hash,
        };
        return Err(Error::InvalidHeader(err));
    }
    let height = state.try_get_height(rwtxn)?.map_or(0, |height| height + 1);
    let market_ctxt = markets::block::connect_begin(
        state,
        rwtxn,
        body,
        height,
        mainchain_timestamp,
    )?;
    let coinbase_txid = header.compute_coinbase_txid();
    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rwtxn, &())?
        .unwrap_or_default();
    let mut accumulator_diff = AccumulatorDiff::default();
    let mut coinbase_value = bitcoin::Amount::ZERO;
    for (vout, output) in body.coinbase.outputs.iter().enumerate() {
        coinbase_value = coinbase_value
            .checked_add(output.get_value())
            .ok_or(AmountOverflowError)?;
        let outpoint = OutPoint::Coinbase {
            txid: coinbase_txid,
            vout: vout as u32,
        };
        let pointed_output = PointedOutput {
            outpoint,
            output: output.clone(),
        };
        accumulator_diff.insert((&pointed_output).into());
        let key = OutPointKey::from(&outpoint);
        state.utxos.put(rwtxn, &key, output)?;
    }
    // Outputs that earlier transactions in this body made
    let mut body_outputs = HashMap::<OutPoint, Output>::new();
    let mut filled_txs: Vec<FilledTransaction> =
        Vec::with_capacity(body.transactions.len());
    for (tx_idx, t) in body.transactions.iter().enumerate() {
        let mut filled_tx = state.fill_transaction(rwtxn, &body_outputs, t)?;
        filled_tx.actor_address = actor_address(body, tx_idx);
        body_outputs.extend(t.outputs_by_outpoint());
        filled_txs.push(filled_tx);
    }
    let market_txs = markets::block::connect_txs(
        state,
        rwtxn,
        &market_ctxt,
        &filled_txs,
        coinbase_value,
    )?;
    for (tx_idx, transaction) in body.transactions.iter().enumerate() {
        if market_txs.skipped_tx_indices.contains(&tx_idx) {
            continue;
        }
        let txid = transaction.txid();
        for (vin, (outpoint, utxo_hash)) in
            transaction.inputs.iter().enumerate()
        {
            let key = OutPointKey::from(outpoint);
            let spent_output =
                state.utxos.try_get(rwtxn, &key)?.ok_or(error::NoUtxo {
                    outpoint: *outpoint,
                })?;

            accumulator_diff.remove(utxo_hash.into());
            let key = OutPointKey::from(outpoint);
            state.utxos.delete(rwtxn, &key)?;
            let spent_output = SpentOutput {
                output: spent_output,
                inpoint: InPoint::Regular {
                    txid,
                    vin: vin as u32,
                },
            };
            state
                .stxos
                .put(rwtxn, &OutPointKey::from(outpoint), &spent_output)
                .map_err(DbError::from)?;
        }
        for (vout, output) in transaction.outputs.iter().enumerate() {
            let outpoint = OutPoint::Regular {
                txid,
                vout: vout as u32,
            };
            let pointed_output = PointedOutput {
                outpoint,
                output: output.clone(),
            };
            accumulator_diff.insert((&pointed_output).into());
            let key = OutPointKey::from(&outpoint);
            state.utxos.put(rwtxn, &key, output)?;
        }
    }
    let () = markets::block::connect_end(
        state,
        rwtxn,
        &mut accumulator_diff,
        market_ctxt,
        market_txs,
    )?;
    let merkle_root = Body::compute_merkle_root(
        &body.coinbase,
        filled_txs.as_slice(),
        &body.authorizations,
        &body.actor_proofs,
    )?;
    if merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: header.merkle_root,
            computed: merkle_root,
        };
        return Err(err);
    }
    let block_hash = header.hash();
    state.tip.put(rwtxn, &(), &block_hash)?;
    state.coinbase_height.put(rwtxn, &coinbase_txid, &height)?;
    state.height.put(rwtxn, &(), &height)?;
    let () = accumulator.apply_diff(accumulator_diff)?;
    state.utreexo_accumulator.put(rwtxn, &(), &accumulator)?;
    Ok(merkle_root)
}

pub fn disconnect_tip_body(
    state: &State,
    rwtxn: &mut RwTxn,
    header: &Header,
    body: &Body,
) -> Result<(), Error> {
    let tip_hash = state
        .tip
        .try_get(rwtxn, &())
        .map_err(DbError::from)?
        .ok_or(Error::NoTip)?;
    let block_hash = header.hash();
    if tip_hash != block_hash {
        let err = error::InvalidHeader::BlockHash {
            expected: tip_hash,
            computed: block_hash,
        };
        return Err(Error::InvalidHeader(err));
    }
    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rwtxn, &())
        .map_err(DbError::from)?
        .unwrap_or_default();
    tracing::debug!("Got acc");
    let mut accumulator_diff = AccumulatorDiff::default();
    let height = state
        .try_get_height(rwtxn)?
        .expect("Height should not be None");
    let skipped_tx_indices = markets::block::disconnect_begin(
        state,
        rwtxn,
        &mut accumulator_diff,
        height,
    )?;
    let mut trade_share_deltas = Vec::new();
    // revert txs, last-to-first
    body.transactions.iter().enumerate().rev().try_for_each(
        |(tx_idx, tx)| {
            if skipped_tx_indices.contains(&(tx_idx as u32)) {
                return Ok(());
            }
            let () = markets::block::disconnect_tx(
                state,
                rwtxn,
                &mut accumulator_diff,
                tx,
                &mut trade_share_deltas,
            )?;
            let txid = tx.txid();
            // delete UTXOs, last-to-first
            tx.outputs.iter().enumerate().rev().try_for_each(
                |(vout, output)| {
                    let outpoint = OutPoint::Regular {
                        txid,
                        vout: vout as u32,
                    };
                    let pointed_output = PointedOutput {
                        outpoint,
                        output: output.clone(),
                    };
                    accumulator_diff.remove((&pointed_output).into());
                    let key = OutPointKey::from(&outpoint);
                    if state.utxos.delete(rwtxn, &key)? {
                        Ok::<_, Error>(())
                    } else {
                        Err(error::NoUtxo { outpoint }.into())
                    }
                },
            )?;
            // unspend STXOs, last-to-first
            tx.inputs
                .iter()
                .rev()
                .try_for_each(|(outpoint, utxo_hash)| {
                    let key = OutPointKey::from(outpoint);
                    if let Some(spent_output) =
                        state.stxos.try_get(rwtxn, &key)?
                    {
                        accumulator_diff.insert(utxo_hash.into());
                        state.stxos.delete(rwtxn, &key)?;
                        state.utxos.put(rwtxn, &key, &spent_output.output)?;
                        Ok(())
                    } else {
                        Err(Error::NoStxo {
                            outpoint: *outpoint,
                        })
                    }
                })
        },
    )?;
    {
        let coinbase_txid = header.compute_coinbase_txid();
        // delete coinbase UTXOs, last-to-first
        body.coinbase
            .outputs
            .iter()
            .enumerate()
            .rev()
            .try_for_each(|(vout, output)| {
                let outpoint = OutPoint::Coinbase {
                    txid: coinbase_txid,
                    vout: vout as u32,
                };
                let pointed_output = PointedOutput {
                    outpoint,
                    output: output.clone(),
                };
                accumulator_diff.remove((&pointed_output).into());
                let key = OutPointKey::from(&outpoint);
                if state.utxos.delete(rwtxn, &key)? {
                    Ok::<_, Error>(())
                } else {
                    Err(error::NoUtxo { outpoint }.into())
                }
            })?;
    }
    let () = markets::block::disconnect_end(
        state,
        rwtxn,
        height,
        &skipped_tx_indices,
        trade_share_deltas,
    )?;
    if header.prev_side_hash.is_none()
        && height == 0
        && let Some(output) = body.coinbase.outputs.iter().next()
    {
        state
            .reputation()
            .delete_reputation(rwtxn, &output.address)?;
    }
    // NOTE: the tip, height, and coinbase height are NOT rolled back
    // here; that happens after 2WPD is disconnected, in
    // `disconnect_tip_header`.
    let () = accumulator.apply_diff(accumulator_diff)?;
    state
        .utreexo_accumulator
        .put(rwtxn, &(), &accumulator)
        .map_err(DbError::from)?;
    Ok(())
}

/// Disconnect the tip block's header: roll back the coinbase height,
/// tip, and height.
/// MUST be applied AFTER disconnecting the block's transactions and
/// 2WPD.
pub fn disconnect_tip_header(
    state: &State,
    rwtxn: &mut RwTxn,
    header: &Header,
) -> Result<(), Error> {
    let coinbase_txid = header.compute_coinbase_txid();
    state.coinbase_height.delete(rwtxn, &coinbase_txid)?;
    let height = state
        .try_get_height(rwtxn)?
        .expect("Height should not be None");
    match (header.prev_side_hash, height) {
        (None, 0) => {
            state.tip.delete(rwtxn, &()).map_err(DbError::from)?;
            state.height.delete(rwtxn, &()).map_err(DbError::from)?;
        }
        (None, _) | (_, 0) => return Err(Error::NoTip),
        (Some(prev_side_hash), height) => {
            state
                .tip
                .put(rwtxn, &(), &prev_side_hash)
                .map_err(DbError::from)?;
            state
                .height
                .put(rwtxn, &(), &(height - 1))
                .map_err(DbError::from)?;
        }
    }
    Ok(())
}

fn sigops(body: &Body) -> usize {
    body.authorizations.len() + body.actor_proofs.iter().flatten().count()
}

/// Address that the actor proof of a transaction proves, if any
fn actor_address(body: &Body, tx_idx: usize) -> Option<crate::types::Address> {
    body.actor_proofs
        .get(tx_idx)
        .and_then(|proof| proof.as_ref())
        .map(|auth| auth.get_address())
}

#[cfg(test)]
mod test {
    use crate::types::transaction::outpoint::DEPOSIT_MATURITY;
    use std::collections::HashMap;

    use truthcoin_dc_types::Coinbase;

    use crate::{
        state::test::{fresh_state, value_output},
        types::FilledTransaction,
    };

    #[test]
    fn block_actor_proofs_obey_signature_limit() -> anyhow::Result<()> {
        use bitcoin::hashes::Hash as _;

        use crate::{
            state::{Error, State},
            types::{
                Authorization, AuthorizedTransaction, Body, Header, OutPoint,
                Transaction,
                authorization::{self, BatchVerificationContext, SigningKey},
            },
        };

        let (_temp_dir, env, state) =
            fresh_state("block_actor_proofs_obey_signature_limit")?;
        let archive = crate::archive::Archive::new(&env)?;
        let mut rng = rand::rng();
        let batch = BatchVerificationContext::new(&mut rng);
        let key = SigningKey::new(&mut rng);
        let transaction = Transaction::default();
        let actor_proof = Authorization {
            verifying_key: (&key).into(),
            signature: authorization::sign_tx(&mut rng, &key, &transaction)?,
        };
        let authorized = AuthorizedTransaction {
            transaction: transaction.clone(),
            authorizations: Vec::new(),
            actor_proof: Some(Box::new(actor_proof)),
        };
        let filled = FilledTransaction {
            transaction,
            spent_utxos: Vec::new(),
            actor_address: None,
        };
        let rotxn = env.read_txn()?;
        let body = Body::new(vec![authorized.clone()], Coinbase::default());
        let header = Header {
            merkle_root: Body::compute_merkle_root(
                &body.coinbase,
                &[filled],
                &body.authorizations,
                &body.actor_proofs,
            )?,
            prev_side_hash: None,
            prev_main_hash: bitcoin::BlockHash::all_zeros(),
            roots: state.get_accumulator(&rotxn)?.get_roots(),
        };
        state.validate_block(&rotxn, &batch, &header, 0, &body, &archive)?;
        state.prevalidate_block(&rotxn, &batch, &header, 0, &body, &archive)?;

        let missing_outpoint = OutPoint::Regular {
            txid: authorized.transaction.txid(),
            vout: 0,
        };
        let limit = State::body_sigops_limit(0);
        for count in [limit, limit + 1] {
            let mut body =
                Body::new(vec![authorized.clone(); count], Coinbase::default());
            body.transactions[0].inputs =
                vec![(missing_outpoint, [0; 32])].into();
            assert!(borsh::object_length(&body)? <= State::body_size_limit(0));
            let validation = state
                .validate_block(&rotxn, &batch, &header, 0, &body, &archive);
            let prevalidation = state
                .prevalidate_block(&rotxn, &batch, &header, 0, &body, &archive);
            if count == limit {
                assert!(
                    matches!(
                        validation,
                        Err(Error::NoUtxo(error)) if error.outpoint == missing_outpoint
                    ),
                    "the exact signature limit must reach the input check"
                );
                assert!(
                    matches!(
                        prevalidation,
                        Err(Error::NoUtxo(error)) if error.outpoint == missing_outpoint
                    ),
                    "the exact signature limit must reach the input check"
                );
            } else {
                assert!(matches!(validation, Err(Error::TooManySigops)));
                assert!(matches!(prevalidation, Err(Error::TooManySigops)));
            }
        }
        Ok(())
    }

    #[test]
    fn genesis_reorg_replaces_reputation_credit() -> anyhow::Result<()> {
        use bitcoin::hashes::Hash as _;

        use crate::types::{
            AccumulatorDiff, Address, Body, Header, OutPoint, PointedOutput,
            authorization::BatchVerificationContext,
        };

        let (_temp_dir, env, state) =
            fresh_state("genesis_reorg_replaces_reputation_credit")?;
        let archive = crate::archive::Archive::new(&env)?;
        let batch = BatchVerificationContext::new(&mut rand::rng());
        let first_address = Address([1; 20]);
        let next_address = Address([2; 20]);
        let mut rwtxn = env.write_txn()?;

        for address in [first_address, next_address] {
            let output = value_output(address, 0);
            let body = Body::new(
                Vec::new(),
                Coinbase {
                    outputs: vec![output.clone()].into(),
                    ..Default::default()
                },
            );
            let mut header = Header {
                merkle_root: Body::compute_merkle_root::<FilledTransaction>(
                    &body.coinbase,
                    &[],
                    &body.authorizations,
                    &body.actor_proofs,
                )?,
                prev_side_hash: None,
                prev_main_hash: bitcoin::BlockHash::all_zeros(),
                roots: Vec::new(),
            };
            let mut accumulator = state.get_accumulator(&rwtxn)?;
            let mut diff = AccumulatorDiff::default();
            diff.insert(
                (&PointedOutput {
                    outpoint: OutPoint::Coinbase {
                        txid: header.compute_coinbase_txid(),
                        vout: 0,
                    },
                    output,
                })
                    .into(),
            );
            accumulator.apply_diff(diff)?;
            header.roots = accumulator.get_roots();
            state.apply_block(
                &mut rwtxn,
                &batch,
                &header,
                0,
                &body,
                &archive,
                1_700_000_000,
            )?;
            assert_eq!(
                state.reputation().get_reputation(&rwtxn, &address)?,
                1.0
            );
            if address == first_address {
                state.disconnect_withdrawal_bundle_creation(&mut rwtxn)?;
                state.disconnect_tip_body(&mut rwtxn, &header, &body)?;
                state.disconnect_two_way_peg_data(
                    &mut rwtxn,
                    &Default::default(),
                )?;
                state.disconnect_tip_header(&mut rwtxn, &header)?;
            }
        }

        assert_eq!(
            state.reputation().get_reputation(&rwtxn, &first_address)?,
            0.0
        );
        assert_eq!(state.reputation().count_holders(&rwtxn)?, 1);
        Ok(())
    }

    #[test]
    fn validation_rejects_outpoint_utxo_hash_mismatch() -> anyhow::Result<()> {
        use bitcoin::hashes::Hash as _;

        use crate::types::{
            Accumulator, AccumulatorDiff, Body, Header, OutPoint, OutPointKey,
            PointedOutput, Transaction, UtreexoNodeHash,
            authorization::{
                self, BatchVerificationContext, SigningKey, get_address,
            },
            hash,
        };

        let (_temp_dir, env, state) =
            fresh_state("validation_rejects_outpoint_utxo_hash_mismatch")?;
        let archive = crate::archive::Archive::new(&env)?;
        let mut rng = rand::rng();
        let batch_verification_ctxt = BatchVerificationContext::new(&mut rng);

        // Attacker key (owns A). Victim key (owns B).
        let attacker = SigningKey::new(&mut rng);
        let attacker_addr = get_address(&(&attacker).into());
        let victim = SigningKey::new(&mut rng);
        let victim_addr = get_address(&(&victim).into());

        // UTXO A (attacker, 10_000) and victim UTXO B (20_000).
        let outpoint_a = OutPoint::Deposit(bitcoin::OutPoint {
            txid: bitcoin::Txid::from_byte_array([0xAA; 32]),
            vout: 0,
        });
        let output_a = value_output(attacker_addr, 10_000);
        let outpoint_b = OutPoint::Deposit(bitcoin::OutPoint {
            txid: bitcoin::Txid::from_byte_array([0xBB; 32]),
            vout: 0,
        });
        let output_b = value_output(victim_addr, 20_000);

        // Leaf hashes for A and B (the Utreexo commitments).
        let pointed_a = PointedOutput {
            outpoint: outpoint_a,
            output: output_a.clone(),
        };
        let pointed_b = PointedOutput {
            outpoint: outpoint_b,
            output: output_b.clone(),
        };
        let leaf_a: UtreexoNodeHash = (&pointed_a).into();
        let leaf_b: UtreexoNodeHash = (&pointed_b).into();
        let hash_b: crate::types::Hash = hash(&pointed_b); // input's utxo_hash

        // Helper: build a fresh accumulator seeded with leaves A and B
        // (Accumulator is not Clone, so re-seed when a fresh copy is needed).
        let seeded_accumulator = || -> anyhow::Result<_> {
            let mut acc = Accumulator::default();
            let mut diff = AccumulatorDiff::default();
            diff.insert(leaf_a);
            diff.insert(leaf_b);
            acc.apply_diff(diff)?;
            Ok(acc)
        };

        // Seed UTXO DB with A and B; seed the accumulator with both leaves.
        let pre_accumulator = seeded_accumulator()?;
        {
            let mut rwtxn = env.write_txn()?;
            state.utxos.put(
                &mut rwtxn,
                &OutPointKey::from(&outpoint_a),
                &output_a,
            )?;
            state.utxos.put(
                &mut rwtxn,
                &OutPointKey::from(&outpoint_b),
                &output_b,
            )?;
            state
                .utreexo_accumulator
                .put(&mut rwtxn, &(), &pre_accumulator)?;
            // tip stays unset (None) so validate's prev_side_hash check
            // expects header.prev_side_hash == None.
            rwtxn.commit()?;
        }

        // Build the malicious tx: input = (outpoint_A, hash_B) + proof for B;
        // output C = 9_000 to the attacker.
        let proof_for_b = pre_accumulator.prove(&[leaf_b])?;
        let output_c = value_output(attacker_addr, 9_000);
        let tx = Transaction {
            inputs: vec![(outpoint_a, hash_b)].into(),
            proof: proof_for_b,
            outputs: vec![output_c.clone()].into(),
            data: None,
        };
        // Sign with A's key (the spender of outpoint A authorizes the tx).
        let authorized = authorization::authorize(
            &mut rng,
            &[(attacker_addr, &attacker)],
            tx,
        )?;

        // Assemble body.
        let body = Body::new(vec![authorized], Coinbase::default());

        let main_height = 9999;

        // Compute the header the validator expects:
        //   merkle_root from the filled tx, roots = post-block accumulator
        //   (B's leaf removed, C's leaf inserted) -- exactly the diff validate
        //   builds from the SUPPLIED utxo_hash.
        let filled = {
            let rotxn = env.read_txn()?;
            let filled = state.fill_transaction(
                &rotxn,
                &HashMap::new(),
                &body.transactions[0],
            )?;
            // tx validation REJECTS the outpoint/utxo_hash mismatch.
            anyhow::ensure!(
                state
                    .validate_filled_transaction(
                        &rotxn,
                        main_height,
                        &filled,
                        &archive,
                        None
                    )
                    .is_err()
            );
            filled
        };

        let merkle_root = Body::compute_merkle_root(
            &body.coinbase,
            &[filled],
            &body.authorizations,
            &body.actor_proofs,
        )?;
        let mut post_accumulator = seeded_accumulator()?;
        {
            let mut diff = AccumulatorDiff::default();
            // validate removes the SUPPLIED utxo_hash (B), inserts output C.
            diff.remove(leaf_b);
            let txid = body.transactions[0].txid();
            let pointed_c = PointedOutput {
                outpoint: OutPoint::Regular { txid, vout: 0 },
                output: output_c.clone(),
            };
            diff.insert((&pointed_c).into());
            post_accumulator.apply_diff(diff)?;
        }
        let header = Header {
            merkle_root,
            prev_side_hash: None,
            prev_main_hash: bitcoin::BlockHash::from_byte_array([0x55u8; 32]),
            roots: post_accumulator.get_roots(),
        };

        // block validation REJECTS the outpoint/utxo_hash mismatch.
        {
            let rotxn = env.read_txn()?;
            anyhow::ensure!(
                state
                    .validate_block(
                        &rotxn,
                        &batch_verification_ctxt,
                        &header,
                        main_height,
                        &body,
                        &archive,
                    )
                    .is_err(),
                "BUG: real validate_block accepts an input whose outpoint (A) \
                and utxo_hash (B) refer to different UTXOs",
            );
        }
        Ok(())
    }

    /// Build a block that carries a parent and its child, and the state that
    /// validates it. `order` names the body order.
    fn chained_block(
        test_name: &str,
        child_first: bool,
    ) -> anyhow::Result<(
        temp_dir::TempDir,
        sneed::Env,
        crate::state::State,
        crate::types::Header,
        crate::types::Body,
        crate::types::authorization::BatchVerificationContext,
    )> {
        use bitcoin::hashes::Hash as _;

        use crate::types::{
            Accumulator, AccumulatorDiff, Body, Header, OutPoint, OutPointKey,
            PointedOutput, Transaction, UtreexoNodeHash,
            authorization::{SigningKey, authorize, get_address},
            hash,
        };

        let (temp_dir, env, state) = fresh_state(test_name)?;
        let mut rng = rand::rng();
        let batch_verification_ctxt =
            crate::types::authorization::BatchVerificationContext::new(
                &mut rng,
            );
        let owner = SigningKey::new(&mut rng);
        let owner_addr = get_address(&(&owner).into());

        // One confirmed deposit funds the chain.
        let deposit_outpoint = OutPoint::Deposit(bitcoin::OutPoint {
            txid: bitcoin::Txid::from_byte_array([0xCC; 32]),
            vout: 0,
        });
        let deposit_output = value_output(owner_addr, 10_000);
        let deposit_pointed = PointedOutput {
            outpoint: deposit_outpoint,
            output: deposit_output.clone(),
        };
        let deposit_leaf: UtreexoNodeHash = (&deposit_pointed).into();
        let deposit_hash = hash(&deposit_pointed);

        let mut accumulator = Accumulator::default();
        {
            let mut diff = AccumulatorDiff::default();
            diff.insert(deposit_leaf);
            accumulator.apply_diff(diff)?;
        }
        {
            let mut rwtxn = env.write_txn()?;
            state.utxos.put(
                &mut rwtxn,
                &OutPointKey::from(&deposit_outpoint),
                &deposit_output,
            )?;
            state.deposit_mainchain_height.put(
                &mut rwtxn,
                &bitcoin::Txid::from_byte_array([0xCC; 32]),
                &0,
            )?;
            state
                .utreexo_accumulator
                .put(&mut rwtxn, &(), &accumulator)?;
            rwtxn.commit()?;
        }

        // Parent spends the deposit. Child spends the parent's output.
        let parent_output = value_output(owner_addr, 9_000);
        let parent = Transaction {
            inputs: vec![(deposit_outpoint, deposit_hash)].into(),
            proof: accumulator.prove(&[deposit_leaf])?,
            outputs: vec![parent_output.clone()].into(),
            data: None,
        };
        let parent_outpoint = OutPoint::Regular {
            txid: parent.txid(),
            vout: 0,
        };
        let parent_pointed = PointedOutput {
            outpoint: parent_outpoint,
            output: parent_output.clone(),
        };
        let child_output = value_output(owner_addr, 8_000);
        let child = Transaction {
            inputs: vec![(parent_outpoint, hash(&parent_pointed))].into(),
            // The accumulator holds no leaf for the parent's output, so the
            // child proves nothing.
            proof: accumulator.prove(&[])?,
            outputs: vec![child_output.clone()].into(),
            data: None,
        };

        let authorized_parent =
            authorize(&mut rng, &[(owner_addr, &owner)], parent)?;
        let authorized_child =
            authorize(&mut rng, &[(owner_addr, &owner)], child)?;
        let body = if child_first {
            Body::new(
                vec![authorized_child, authorized_parent],
                Coinbase::default(),
            )
        } else {
            Body::new(
                vec![authorized_parent, authorized_child],
                Coinbase::default(),
            )
        };

        // The parent's output is made and spent inside the block, so the
        // accumulator only loses the deposit and gains the child's output.
        let mut post_accumulator = Accumulator::default();
        {
            let mut diff = AccumulatorDiff::default();
            diff.insert(deposit_leaf);
            post_accumulator.apply_diff(diff)?;
            let mut diff = AccumulatorDiff::default();
            diff.remove(deposit_leaf);
            diff.insert(
                (&PointedOutput {
                    outpoint: OutPoint::Regular {
                        txid: body.transactions
                            [if child_first { 1 } else { 0 }]
                        .txid(),
                        vout: 0,
                    },
                    output: parent_output.clone(),
                })
                    .into(),
            );
            diff.remove(
                (&PointedOutput {
                    outpoint: parent_outpoint,
                    output: parent_output,
                })
                    .into(),
            );
            diff.insert(
                (&PointedOutput {
                    outpoint: OutPoint::Regular {
                        txid: body.transactions
                            [if child_first { 0 } else { 1 }]
                        .txid(),
                        vout: 0,
                    },
                    output: child_output,
                })
                    .into(),
            );
            post_accumulator.apply_diff(diff)?;
        }

        let filled = vec![
            FilledTransaction {
                spent_utxos: vec![if child_first {
                    parent_pointed.output.clone()
                } else {
                    deposit_output.clone()
                }],
                transaction: body.transactions[0].clone(),
                actor_address: None,
            },
            FilledTransaction {
                spent_utxos: vec![if child_first {
                    deposit_output
                } else {
                    parent_pointed.output
                }],
                transaction: body.transactions[1].clone(),
                actor_address: None,
            },
        ];
        let header = Header {
            merkle_root: Body::compute_merkle_root(
                &body.coinbase,
                filled.as_slice(),
                &body.authorizations,
                &body.actor_proofs,
            )?,
            prev_side_hash: None,
            prev_main_hash: bitcoin::BlockHash::from_byte_array([0u8; 32]),
            roots: post_accumulator.get_roots(),
        };
        Ok((temp_dir, env, state, header, body, batch_verification_ctxt))
    }

    #[test]
    fn a_block_carries_a_parent_and_its_child() -> anyhow::Result<()> {
        let (_temp_dir, env, state, header, body, ctxt) =
            chained_block("a_block_carries_a_parent_and_its_child", false)?;
        let archive = crate::archive::Archive::new(&env)?;
        let rotxn = env.read_txn()?;
        let (fees, _) = state.validate_block(
            &rotxn,
            &ctxt,
            &header,
            DEPOSIT_MATURITY,
            &body,
            &archive,
        )?;
        anyhow::ensure!(fees == bitcoin::Amount::from_sat(2_000));
        let () = state
            .prevalidate_block(
                &rotxn,
                &ctxt,
                &header,
                DEPOSIT_MATURITY,
                &body,
                &archive,
            )
            .map(|_| ())?;
        Ok(())
    }

    #[test]
    fn a_child_before_its_parent_is_rejected() -> anyhow::Result<()> {
        let (_temp_dir, env, state, header, body, ctxt) =
            chained_block("a_child_before_its_parent_is_rejected", true)?;
        let archive = crate::archive::Archive::new(&env)?;
        let rotxn = env.read_txn()?;
        anyhow::ensure!(
            state
                .validate_block(
                    &rotxn,
                    &ctxt,
                    &header,
                    DEPOSIT_MATURITY,
                    &body,
                    &archive
                )
                .is_err(),
            "a body that puts a child before its parent must not validate",
        );
        anyhow::ensure!(
            state
                .prevalidate_block(
                    &rotxn,
                    &ctxt,
                    &header,
                    DEPOSIT_MATURITY,
                    &body,
                    &archive
                )
                .is_err(),
            "prevalidate must reject the same body",
        );
        Ok(())
    }

    #[test]
    fn duplicate_signed_votes_in_block_are_rejected() -> anyhow::Result<()> {
        use crate::{
            state::decisions::{Decision, DecisionId, DecisionType},
            types::{
                Authorization, AuthorizedTransaction, Body, Header,
                Transaction, TxData,
                authorization::{self, BatchVerificationContext, SigningKey},
            },
        };
        use bitcoin::hashes::Hash as _;
        let (_dir, env, state) = fresh_state("duplicate_votes")?;
        let archive = crate::archive::Archive::new(&env)?;
        let mut rng = rand::rng();
        let key = SigningKey::new(&mut rng);
        let voter = authorization::get_address(&(&key).into());
        let decision_id = DecisionId::new(false, 1, 0)?;
        let decision = Decision::new(
            [0; 20],
            DecisionType::Binary,
            String::new(),
            String::new(),
            None,
            None,
            vec![],
        )?;
        let mut txn = env.write_txn()?;
        let timestamp =
            1000 + u64::from(state.decisions().get_config().quantity);
        state.genesis_timestamp.put(&mut txn, &(), &1000)?;
        state.mainchain_timestamp.put(&mut txn, &(), &timestamp)?;
        state.height.put(&mut txn, &(), &10)?;
        state.reputation().set_reputation(&mut txn, &voter, 1.0)?;
        state
            .decisions()
            .mint_genesis(&mut txn, timestamp, 10, 1000)?;
        state.decisions().claim_decision(
            &mut txn,
            decision_id,
            decision,
            crate::types::Txid([1; 32]),
            Some(0),
        )?;
        state.decisions().transition_decision_to_voting(
            &mut txn,
            decision_id,
            1,
        )?;
        for ballots in
            [[false, false], [false, true], [true, false], [true, true]]
        {
            let mut txs = vec![];
            let mut filled = vec![];
            for (value, ballot) in [0.0, 1.0].into_iter().zip(ballots) {
                let transaction = Transaction {
                    data: Some(if ballot {
                        TxData::SubmitBallot {
                            voter,
                            voting_period: 2,
                            votes: vec![crate::types::BallotItem {
                                decision_id_bytes: decision_id.as_bytes(),
                                vote_value: value,
                            }],
                        }
                    } else {
                        TxData::SubmitVote {
                            voter,
                            decision_id_bytes: decision_id.as_bytes(),
                            vote_value: value,
                            voting_period: 2,
                        }
                    }),
                    ..Default::default()
                };
                let proof = Authorization {
                    verifying_key: (&key).into(),
                    signature: authorization::sign_tx(
                        &mut rng,
                        &key,
                        &transaction,
                    )?,
                };
                filled.push(FilledTransaction {
                    transaction: transaction.clone(),
                    spent_utxos: vec![],
                    actor_address: Some(voter),
                });
                txs.push(AuthorizedTransaction {
                    transaction,
                    authorizations: vec![],
                    actor_proof: Some(Box::new(proof)),
                });
            }
            let body = Body::new(txs, Coinbase::default());
            let header = Header {
                merkle_root: Body::compute_merkle_root(
                    &body.coinbase,
                    &filled,
                    &body.authorizations,
                    &body.actor_proofs,
                )?,
                prev_side_hash: None,
                prev_main_hash: bitcoin::BlockHash::all_zeros(),
                roots: state.get_accumulator(&txn)?.get_roots(),
            };
            let batch = BatchVerificationContext::new(&mut rng);
            let validation =
                state.validate_block(&txn, &batch, &header, 0, &body, &archive);
            let prevalidation = state
                .prevalidate_block(&txn, &batch, &header, 0, &body, &archive);
            assert!(
                matches!(validation, Err(crate::state::Error::InvalidTransaction { reason }) if reason == "Duplicate vote in block")
            );
            assert!(
                matches!(prevalidation, Err(crate::state::Error::InvalidTransaction { reason }) if reason == "Duplicate vote in block")
            );
        }
        Ok(())
    }

    #[test]
    fn body_size_counts_utreexo_proofs() -> anyhow::Result<()> {
        use bitcoin::hashes::Hash as _;

        use crate::{
            state::{Error, State},
            types::{
                AuthorizedTransaction, Body, Header, Transaction,
                UtreexoNodeHash, UtreexoProof,
                authorization::BatchVerificationContext,
            },
        };

        let (_temp_dir, env, state) =
            fresh_state("body_size_counts_utreexo_proofs")?;
        let archive = crate::archive::Archive::new(&env)?;
        let batch = BatchVerificationContext::new(&mut rand::rng());
        let hash_count = State::body_size_limit(0)
            / std::mem::size_of::<UtreexoNodeHash>()
            + 1;
        let transaction = Transaction {
            inputs: Vec::new().into(),
            proof: UtreexoProof::new(
                Vec::new(),
                vec![UtreexoNodeHash::Some([1; 32]); hash_count],
            ),
            outputs: Vec::new().into(),
            data: None,
        };
        let body = Body::new(
            vec![AuthorizedTransaction {
                transaction,
                authorizations: Vec::new(),
                actor_proof: None,
            }],
            Coinbase::default(),
        );
        anyhow::ensure!(
            borsh::object_length(&body)? < State::body_size_limit(0)
        );
        let header = Header {
            merkle_root: Body::compute_merkle_root::<FilledTransaction>(
                &body.coinbase,
                &[],
                &body.authorizations,
                &body.actor_proofs,
            )?,
            prev_side_hash: None,
            prev_main_hash: bitcoin::BlockHash::all_zeros(),
            roots: Vec::new(),
        };
        let rotxn = env.read_txn()?;
        let result =
            state.validate_block(&rotxn, &batch, &header, 0, &body, &archive);
        anyhow::ensure!(
            matches!(result, Err(Error::BodyTooLarge)),
            "expected BodyTooLarge, got {result:?}",
        );
        Ok(())
    }

    #[test]
    fn a_block_rejects_outputs_without_inputs() -> anyhow::Result<()> {
        use bitcoin::hashes::Hash as _;

        use crate::{
            state::Error,
            types::{
                Address, AuthorizedTransaction, Body, Header, Transaction,
                authorization::BatchVerificationContext,
            },
        };

        let (_temp_dir, env, state) =
            fresh_state("a_block_rejects_outputs_without_inputs")?;
        let archive = crate::archive::Archive::new(&env)?;
        let transaction = Transaction {
            outputs: vec![value_output(Address([1; 20]), 0)].into(),
            ..Transaction::default()
        };
        let filled = FilledTransaction {
            transaction: transaction.clone(),
            spent_utxos: Vec::new(),
            actor_address: None,
        };
        let body = Body::new(
            vec![AuthorizedTransaction {
                transaction,
                authorizations: Vec::new(),
                actor_proof: None,
            }],
            Coinbase::default(),
        );
        let rotxn = env.read_txn()?;
        let mut header = Header {
            merkle_root: Body::compute_merkle_root(
                &body.coinbase,
                std::slice::from_ref(&filled),
                &body.authorizations,
                &body.actor_proofs,
            )?,
            prev_side_hash: None,
            prev_main_hash: bitcoin::BlockHash::all_zeros(),
            roots: Vec::new(),
        };
        let mut accumulator = state.get_accumulator(&rotxn)?;
        Body::modify_memforest(
            header.compute_coinbase_txid(),
            &[],
            &[filled],
            &mut accumulator.0,
        )?;
        header.roots = accumulator.get_roots();
        let ctxt = BatchVerificationContext::new(&mut rand::rng());
        let validation =
            state.validate_block(&rotxn, &ctxt, &header, 0, &body, &archive);
        assert!(
            matches!(validation, Err(Error::OutputsWithoutInputs)),
            "{validation:?}"
        );
        let prevalidation =
            state.prevalidate_block(&rotxn, &ctxt, &header, 0, &body, &archive);
        assert!(matches!(prevalidation, Err(Error::OutputsWithoutInputs)));
        Ok(())
    }
}
