//! Connect and disconnect blocks

use rayon::prelude::*;
use sneed::{RoTxn, RwTxn};

use crate::{
    state::{Error, PrevalidatedBlock, State, amm, dutch_auction, error},
    types::{
        AmountOverflowError, Authorization, TruthcoinId, BlockHash, Body,
        FilledOutput, FilledOutputContent, GetAddress as _,
        GetBitcoinValue as _, Hash, Header, InPoint, OutPoint, OutPointKey,
        OutputContent, SpentOutput, TxData, Verify as _,
    },
};

/// Calculate total number of inputs across all transactions in a block body
fn calculate_total_inputs(body: &Body) -> usize {
    body.transactions.iter().map(|t| t.inputs.len()).sum()
}

/// Validate a block, returning the merkle root and fees
pub fn validate(
    state: &State,
    rotxn: &RoTxn,
    header: &Header,
    body: &Body,
) -> Result<bitcoin::Amount, Error> {
    let tip_hash = state.try_get_tip(rotxn)?;
    if header.prev_side_hash != tip_hash {
        let err = error::InvalidHeader::PrevSideHash {
            expected: tip_hash,
            received: header.prev_side_hash,
        };
        return Err(Error::InvalidHeader(err));
    };
    let merkle_root =
        Body::compute_merkle_root(&body.coinbase, &body.transactions);
    if merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: header.merkle_root,
            computed: merkle_root,
        };
        return Err(err);
    }
    let mut coinbase_value = bitcoin::Amount::ZERO;
    for output in &body.coinbase {
        coinbase_value = coinbase_value
            .checked_add(output.get_bitcoin_value())
            .ok_or(AmountOverflowError)?;
    }
    let mut total_fees = bitcoin::Amount::ZERO;
    let filled_txs: Vec<_> = body
        .transactions
        .iter()
        .map(|t| state.fill_transaction(rotxn, t))
        .collect::<Result<_, _>>()?;

    let total_inputs = calculate_total_inputs(body);

    // Collect all inputs as fixed-width keys for efficient double-spend detection via sort-and-scan
    let mut all_input_keys = Vec::with_capacity(total_inputs);
    for filled_transaction in &filled_txs {
        for input in &filled_transaction.transaction.inputs {
            all_input_keys.push(OutPointKey::from_outpoint(input));
        }
    }

    // Sort and check for duplicate outpoints (double-spend detection)
    all_input_keys.par_sort_unstable();
    if all_input_keys.windows(2).any(|w| w[0] == w[1]) {
        return Err(Error::UtxoDoubleSpent);
    }

    for filled_tx in &filled_txs {
        total_fees = total_fees
            .checked_add(state.validate_filled_transaction(rotxn, filled_tx)?)
            .ok_or(AmountOverflowError)?;
    }
    if coinbase_value > total_fees {
        return Err(Error::NotEnoughFees);
    }
    let spent_utxos = filled_txs.iter().flat_map(|t| t.spent_utxos.iter());
    for (authorization, spent_utxo) in
        body.authorizations.iter().zip(spent_utxos)
    {
        if authorization.get_address() != spent_utxo.address {
            return Err(Error::WrongPubKeyForAddress);
        }
    }
    if Authorization::verify_body(body).is_err() {
        return Err(Error::AuthorizationError);
    }
    Ok(total_fees)
}

pub fn prevalidate(
    state: &State,
    rotxn: &RoTxn,
    header: &Header,
    body: &Body,
) -> Result<PrevalidatedBlock, Error> {
    let tip_hash = state.try_get_tip(rotxn)?;
    if header.prev_side_hash != tip_hash {
        let err = error::InvalidHeader::PrevSideHash {
            expected: tip_hash,
            received: header.prev_side_hash,
        };
        return Err(Error::InvalidHeader(err));
    };

    let computed_merkle_root =
        Body::compute_merkle_root(&body.coinbase, &body.transactions);
    if computed_merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: header.merkle_root,
            computed: computed_merkle_root,
        };
        return Err(err);
    }

    let mut coinbase_value = bitcoin::Amount::ZERO;
    for output in &body.coinbase {
        coinbase_value = coinbase_value
            .checked_add(output.get_bitcoin_value())
            .ok_or(AmountOverflowError)?;
    }

    let mut total_fees = bitcoin::Amount::ZERO;
    let filled_transactions: Vec<_> = body
        .transactions
        .iter()
        .map(|t| state.fill_transaction(rotxn, t))
        .collect::<Result<_, _>>()?;

    let total_inputs = calculate_total_inputs(body);

    // Collect all inputs as fixed-width keys for efficient double-spend detection via sort-and-scan
    let mut all_input_keys = Vec::with_capacity(total_inputs);
    for filled_transaction in &filled_transactions {
        for input in &filled_transaction.transaction.inputs {
            all_input_keys.push(OutPointKey::from_outpoint(input));
        }
    }

    // Sort and check for duplicate outpoints (double-spend detection)
    all_input_keys.par_sort_unstable();
    if all_input_keys.windows(2).any(|w| w[0] == w[1]) {
        return Err(Error::UtxoDoubleSpent);
    }

    for filled_tx in &filled_transactions {
        total_fees = total_fees
            .checked_add(state.validate_filled_transaction(rotxn, filled_tx)?)
            .ok_or(AmountOverflowError)?;
    }

    if coinbase_value > total_fees {
        return Err(Error::NotEnoughFees);
    }

    let spent_utxos_iter = filled_transactions
        .iter()
        .flat_map(|t| t.spent_utxos.iter());
    for (authorization, spent_utxo) in
        body.authorizations.iter().zip(spent_utxos_iter)
    {
        if authorization.get_address() != spent_utxo.address {
            return Err(Error::WrongPubKeyForAddress);
        }
    }

    if Authorization::verify_body(body).is_err() {
        return Err(Error::AuthorizationError);
    }

    let height = state.try_get_height(rotxn)?.map_or(0, |height| height + 1);

    Ok(PrevalidatedBlock {
        filled_transactions,
        computed_merkle_root: BlockHash::from(Hash::from(computed_merkle_root)),
        total_fees,
        coinbase_value,
        next_height: height,
    })
}

pub fn connect_prevalidated(
    state: &State,
    rwtxn: &mut RwTxn,
    header: &Header,
    body: &Body,
    prevalidated: PrevalidatedBlock,
) -> Result<(), Error> {
    // Skip validation - already done in prevalidate
    // Use precomputed values to avoid redundant DB reads

    // Calculate precise capacities for optimal Vec performance
    let total_inputs: usize = prevalidated
        .filled_transactions
        .iter()
        .map(|tx| tx.transaction.inputs.len())
        .sum();
    let total_outputs: usize = prevalidated
        .filled_transactions
        .iter()
        .map(|tx| tx.transaction.outputs.len())
        .sum::<usize>()
        + body.coinbase.len();

    // Use Vec + sort_unstable instead of individual DB operations for better performance
    let mut utxo_deletes: Vec<OutPointKey> = Vec::with_capacity(total_inputs);
    let mut stxo_puts: Vec<(OutPointKey, SpentOutput)> =
        Vec::with_capacity(total_inputs);
    let mut utxo_puts: Vec<(OutPointKey, FilledOutput)> =
        Vec::with_capacity(total_outputs);

    // Collect coinbase outputs
    for (vout, output) in body.coinbase.iter().enumerate() {
        let outpoint = OutPoint::Coinbase {
            merkle_root: header.merkle_root,
            vout: vout as u32,
        };
        let filled_content = match output.content.clone() {
            OutputContent::Bitcoin(value) => {
                FilledOutputContent::Bitcoin(value)
            }
            OutputContent::Withdrawal(withdrawal) => {
                FilledOutputContent::BitcoinWithdrawal(withdrawal)
            }
            OutputContent::AmmLpToken(_)
            | OutputContent::Truthcoin(_)
            | OutputContent::TruthcoinControl
            | OutputContent::TruthcoinReservation
            | OutputContent::DutchAuctionReceipt => {
                return Err(Error::BadCoinbaseOutputContent);
            }
        };
        let filled_output = FilledOutput {
            address: output.address,
            content: filled_content,
            memo: output.memo.clone(),
        };
        utxo_puts.push((OutPointKey::from_outpoint(&outpoint), filled_output));
    }

    // Collect transaction changes and apply transaction data effects
    for (transaction, filled_tx) in body
        .transactions
        .iter()
        .zip(&prevalidated.filled_transactions)
    {
        let txid = filled_tx.txid();

        // Process inputs
        for (vin, input) in filled_tx.inputs().iter().enumerate() {
            let key = OutPointKey::from_outpoint(input);
            // Reuse prevalidated spent UTXO instead of reading from DB
            let prev_utxo = filled_tx.spent_utxos[vin].clone();
            let spent_output = SpentOutput {
                output: prev_utxo,
                inpoint: InPoint::Regular {
                    txid,
                    vin: vin as u32,
                },
            };
            utxo_deletes.push(key);
            stxo_puts.push((key, spent_output));
        }

        // Process outputs
        let Some(filled_outputs) = filled_tx.filled_outputs() else {
            let err = error::FillTxOutputContents(Box::new(filled_tx.clone()));
            return Err(err.into());
        };
        for (vout, filled_output) in filled_outputs.iter().enumerate() {
            let outpoint = OutPoint::Regular {
                txid,
                vout: vout as u32,
            };
            let key = OutPointKey::from_outpoint(&outpoint);
            utxo_puts.push((key, filled_output.clone()));
        }

        // Apply transaction data effects
        match &transaction.data {
            None => (),
            Some(TxData::AmmBurn { .. }) => {
                let () = amm::apply_burn(&state.amm_pools, rwtxn, filled_tx)?;
            }
            Some(TxData::AmmMint { .. }) => {
                let () = amm::apply_mint(&state.amm_pools, rwtxn, filled_tx)?;
            }
            Some(TxData::AmmSwap { .. }) => {
                let () = amm::apply_swap(&state.amm_pools, rwtxn, filled_tx)?;
            }
            Some(TxData::BitAssetReservation { commitment }) => {
                let () = state
                    .bitassets
                    .put_reservation(rwtxn, &txid, commitment)?;
            }
            Some(TxData::BitAssetRegistration {
                name_hash,
                revealed_nonce: _,
                bitasset_data,
                initial_supply,
            }) => {
                let () = state.bitassets.apply_registration(
                    rwtxn,
                    filled_tx,
                    *name_hash,
                    bitasset_data,
                    *initial_supply,
                    prevalidated.next_height,
                )?;
            }
            Some(TxData::BitAssetMint(mint_amount)) => {
                let () = state.bitassets.apply_mint(
                    rwtxn,
                    filled_tx,
                    *mint_amount,
                    prevalidated.next_height,
                )?;
            }
            Some(TxData::BitAssetUpdate(bitasset_updates)) => {
                let () = state.bitassets.apply_updates(
                    rwtxn,
                    filled_tx,
                    (**bitasset_updates).clone(),
                    prevalidated.next_height,
                )?;
            }
            Some(TxData::DutchAuctionBid { .. }) => {
                let () = dutch_auction::apply_bid(
                    &state.dutch_auctions,
                    rwtxn,
                    filled_tx,
                    prevalidated.next_height,
                )?;
            }
            Some(TxData::DutchAuctionCreate(dutch_auction_params)) => {
                let () = dutch_auction::apply_create(
                    &state.dutch_auctions,
                    rwtxn,
                    filled_tx,
                    *dutch_auction_params,
                    prevalidated.next_height,
                )?;
            }
            Some(TxData::DutchAuctionCollect { .. }) => {
                let () = dutch_auction::apply_collect(
                    &state.dutch_auctions,
                    rwtxn,
                    filled_tx,
                    prevalidated.next_height,
                )?;
            }
        }
    }

    // Sort all vectors in parallel for optimal cursor access
    utxo_deletes.par_sort_unstable();
    stxo_puts.par_sort_unstable_by_key(|(key, _)| *key);
    utxo_puts.par_sort_unstable_by_key(|(key, _)| *key);

    // Apply all database operations using pre-sorted keys for optimal B-tree access
    for key in &utxo_deletes {
        state.utxos.delete(rwtxn, key)?;
    }

    for (key, spent_output) in &stxo_puts {
        state.stxos.put(rwtxn, key, spent_output)?;
    }

    for (key, filled_output) in &utxo_puts {
        state.utxos.put(rwtxn, key, filled_output)?;
    }

    // Update tip and height using precomputed values
    let block_hash = header.hash();
    state.tip.put(rwtxn, &(), &block_hash)?;
    state.height.put(rwtxn, &(), &prevalidated.next_height)?;

    Ok(())
}

pub fn connect(
    state: &State,
    rwtxn: &mut RwTxn,
    header: &Header,
    body: &Body,
) -> Result<(), Error> {
    let height = state.try_get_height(rwtxn)?.map_or(0, |height| height + 1);
    let tip_hash = state.try_get_tip(rwtxn)?;
    if tip_hash != header.prev_side_hash {
        let err = error::InvalidHeader::PrevSideHash {
            expected: tip_hash,
            received: header.prev_side_hash,
        };
        return Err(Error::InvalidHeader(err));
    }
    let merkle_root =
        Body::compute_merkle_root(&body.coinbase, &body.transactions);
    if merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: merkle_root,
            computed: header.merkle_root,
        };
        return Err(err);
    }
    for (vout, output) in body.coinbase.iter().enumerate() {
        let outpoint = OutPoint::Coinbase {
            merkle_root: header.merkle_root,
            vout: vout as u32,
        };
        let outpoint_key = OutPointKey::from_outpoint(&outpoint);
        let filled_content = match output.content.clone() {
            OutputContent::Bitcoin(value) => {
                FilledOutputContent::Bitcoin(value)
            }
            OutputContent::Withdrawal(withdrawal) => {
                FilledOutputContent::BitcoinWithdrawal(withdrawal)
            }
            OutputContent::AmmLpToken(_)
            | OutputContent::BitAsset(_)
            | OutputContent::BitAssetControl
            | OutputContent::BitAssetReservation
            | OutputContent::DutchAuctionReceipt => {
                return Err(Error::BadCoinbaseOutputContent);
            }
        };
        let filled_output = FilledOutput {
            address: output.address,
            content: filled_content,
            memo: output.memo.clone(),
        };
        state.utxos.put(rwtxn, &outpoint_key, &filled_output)?;
    }
    for transaction in &body.transactions {
        let filled_tx = state.fill_transaction(rwtxn, transaction)?;
        let txid = filled_tx.txid();
        for (vin, input) in filled_tx.inputs().iter().enumerate() {
            let input_key = OutPointKey::from_outpoint(input);
            let spent_output = state
                .utxos
                .try_get(rwtxn, &input_key)?
                .ok_or(Error::NoUtxo { outpoint: *input })?;
            let spent_output = SpentOutput {
                output: spent_output,
                inpoint: InPoint::Regular {
                    txid,
                    vin: vin as u32,
                },
            };
            state.utxos.delete(rwtxn, &input_key)?;
            state.stxos.put(rwtxn, &input_key, &spent_output)?;
        }
        let Some(filled_outputs) = filled_tx.filled_outputs() else {
            let err = error::FillTxOutputContents(Box::new(filled_tx));
            return Err(err.into());
        };
        for (vout, filled_output) in filled_outputs.iter().enumerate() {
            let outpoint = OutPoint::Regular {
                txid,
                vout: vout as u32,
            };
            let outpoint_key = OutPointKey::from_outpoint(&outpoint);
            state.utxos.put(rwtxn, &outpoint_key, filled_output)?;
        }
        match &transaction.data {
            None => (),
            Some(TxData::AmmBurn { .. }) => {
                let () = amm::apply_burn(&state.amm_pools, rwtxn, &filled_tx)?;
            }
            Some(TxData::AmmMint { .. }) => {
                let () = amm::apply_mint(&state.amm_pools, rwtxn, &filled_tx)?;
            }
            Some(TxData::AmmSwap { .. }) => {
                let () = amm::apply_swap(&state.amm_pools, rwtxn, &filled_tx)?;
            }
            Some(TxData::TruthcoinReservation { commitment }) => {
                let () = state
                    .truthcoin
                    .put_reservation(rwtxn, &txid, commitment)?;
            }
            Some(TxData::TruthcoinRegistration {
                name_hash,
                revealed_nonce: _,
                truthcoin_data,
                initial_supply,
            }) => {
                let () = state.truthcoin.apply_registration(
                    rwtxn,
                    &filled_tx,
                    *name_hash,
                    truthcoin_data,
                    *initial_supply,
                    height,
                )?;
            }
            Some(TxData::TruthcoinMint(mint_amount)) => {
                let () = state.truthcoin.apply_mint(
                    rwtxn,
                    &filled_tx,
                    *mint_amount,
                    height,
                )?;
            }
            Some(TxData::TruthcoinUpdate(truthcoin_updates)) => {
                let () = state.truthcoin.apply_updates(
                    rwtxn,
                    &filled_tx,
                    (**truthcoin_updates).clone(),
                    height,
                )?;
            }
            Some(TxData::DutchAuctionBid { .. }) => {
                let () = dutch_auction::apply_bid(
                    &state.dutch_auctions,
                    rwtxn,
                    &filled_tx,
                    height,
                )?;
            }
            Some(TxData::DutchAuctionCreate(dutch_auction_params)) => {
                let () = dutch_auction::apply_create(
                    &state.dutch_auctions,
                    rwtxn,
                    &filled_tx,
                    *dutch_auction_params,
                    height,
                )?;
            }
            Some(TxData::DutchAuctionCollect { .. }) => {
                let () = dutch_auction::apply_collect(
                    &state.dutch_auctions,
                    rwtxn,
                    &filled_tx,
                    height,
                )?;
            }
        }
    }
    let block_hash = header.hash();
    state.tip.put(rwtxn, &(), &block_hash)?;
    state.height.put(rwtxn, &(), &height)?;
    Ok(())
}

pub fn disconnect_tip(
    state: &State,
    rwtxn: &mut RwTxn,
    header: &Header,
    body: &Body,
) -> Result<(), Error> {
    let tip_hash = state.tip.try_get(rwtxn, &())?.ok_or(Error::NoTip)?;
    if tip_hash != header.hash() {
        let err = error::InvalidHeader::BlockHash {
            expected: tip_hash,
            computed: header.hash(),
        };
        return Err(Error::InvalidHeader(err));
    }
    let merkle_root =
        Body::compute_merkle_root(&body.coinbase, &body.transactions);
    if merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: header.merkle_root,
            computed: merkle_root,
        };
        return Err(err);
    }
    let height = state
        .try_get_height(rwtxn)?
        .expect("Height should not be None");
    // revert txs, last-to-first
    body.transactions.iter().rev().try_for_each(|tx| {
        let txid = tx.txid();
        let filled_tx = state.fill_transaction_from_stxos(rwtxn, tx.clone())?;
        // revert transaction effects
        match &tx.data {
            None => (),
            Some(TxData::AmmBurn { .. }) => {
                let () = amm::revert_burn(&state.amm_pools, rwtxn, &filled_tx)?;
            }
            Some(TxData::AmmMint { .. }) => {
                let () = amm::revert_mint(&state.amm_pools, rwtxn, &filled_tx)?;
            }
            Some(TxData::AmmSwap { .. }) => {
                let () = amm::revert_swap(&state.amm_pools, rwtxn, &filled_tx)?;
            }
            Some(TxData::TruthcoinMint(mint_amount)) => {
                let () = state.truthcoin.revert_mint(
                    rwtxn,
                    &filled_tx,
                    *mint_amount,
                )?;
            }
            Some(TxData::TruthcoinRegistration {
                name_hash,
                revealed_nonce: _,
                truthcoin_data: _,
                initial_supply: _,
            }) => {
                let () = state.truthcoin.revert_registration(
                    rwtxn,
                    &filled_tx,
                    TruthcoinId(*name_hash),
                )?;
            }
            Some(TxData::TruthcoinReservation { commitment: _ }) => {
                if !state.truthcoin.delete_reservation(rwtxn, &txid)? {
                    let err = error::Truthcoin::MissingReservation { txid };
                    return Err(err.into());
                }
            }
            Some(TxData::TruthcoinUpdate(truthcoin_updates)) => {
                let () = state.truthcoin.revert_updates(
                    rwtxn,
                    &filled_tx,
                    (**truthcoin_updates).clone(),
                    height - 1,
                )?;
            }
            Some(TxData::DutchAuctionBid { .. }) => {
                let () = dutch_auction::revert_bid(
                    &state.dutch_auctions,
                    rwtxn,
                    &filled_tx,
                )?;
            }
            Some(TxData::DutchAuctionCollect { .. }) => {
                let () = dutch_auction::revert_collect(
                    &state.dutch_auctions,
                    rwtxn,
                    &filled_tx,
                )?;
            }
            Some(TxData::DutchAuctionCreate(_auction_params)) => {
                let () = dutch_auction::revert_create(
                    &state.dutch_auctions,
                    rwtxn,
                    &filled_tx,
                )?;
            }
        }
        // delete UTXOs, last-to-first
        tx.outputs.iter().enumerate().rev().try_for_each(
            |(vout, _output)| {
                let outpoint = OutPoint::Regular {
                    txid,
                    vout: vout as u32,
                };
                let outpoint_key = OutPointKey::from_outpoint(&outpoint);
                if state.utxos.delete(rwtxn, &outpoint_key)? {
                    Ok(())
                } else {
                    Err(Error::NoUtxo { outpoint })
                }
            },
        )?;
        // unspend STXOs, last-to-first
        tx.inputs.iter().rev().try_for_each(|outpoint| {
            let outpoint_key = OutPointKey::from_outpoint(outpoint);
            if let Some(spent_output) =
                state.stxos.try_get(rwtxn, &outpoint_key)?
            {
                state.stxos.delete(rwtxn, &outpoint_key)?;
                state
                    .utxos
                    .put(rwtxn, &outpoint_key, &spent_output.output)?;
                Ok(())
            } else {
                Err(Error::NoStxo {
                    outpoint: *outpoint,
                })
            }
        })
    })?;
    // delete coinbase UTXOs, last-to-first
    body.coinbase.iter().enumerate().rev().try_for_each(
        |(vout, _output)| {
            let outpoint = OutPoint::Coinbase {
                merkle_root: header.merkle_root,
                vout: vout as u32,
            };
            let outpoint_key = OutPointKey::from_outpoint(&outpoint);
            if state.utxos.delete(rwtxn, &outpoint_key)? {
                Ok(())
            } else {
                Err(Error::NoUtxo { outpoint })
            }
        },
    )?;
    match (header.prev_side_hash, height) {
        (None, 0) => {
            state.tip.delete(rwtxn, &())?;
            state.height.delete(rwtxn, &())?;
        }
        (None, _) | (_, 0) => return Err(Error::NoTip),
        (Some(prev_side_hash), height) => {
            state.tip.put(rwtxn, &(), &prev_side_hash)?;
            state.height.put(rwtxn, &(), &(height - 1))?;
        }
    }
    Ok(())
}
