use crate::state::{Error, PrevalidatedBlock};
use crate::types::{
    AmountOverflowError, Authorization, AuthorizedTransaction, Body,
    FilledTransaction, GetAddress as _, GetBitcoinValue as _, Header,
    OutPointKey, OutputContent, Verify as _,
};
use rayon::prelude::*;
use sneed::RoTxn;
use std::collections::HashSet;

use super::{DecisionValidator, MarketValidator, VoteValidator};

pub struct BlockValidator;

impl BlockValidator {
    /// Prevalidate a block: perform all read-only checks and return
    /// precomputed values to avoid redundant computation during connection.
    pub fn prevalidate(
        state: &crate::state::State,
        archive: &crate::archive::Archive,
        rotxn: &RoTxn,
        header: &Header,
        body: &Body,
    ) -> Result<PrevalidatedBlock, Error> {
        use crate::state::error;

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

        let next_height =
            state.try_get_height(rotxn)?.map_or(0, |height| height + 1);

        let mut coinbase_value = bitcoin::Amount::ZERO;
        for output in &body.coinbase {
            coinbase_value = coinbase_value
                .checked_add(output.get_bitcoin_value())
                .ok_or(AmountOverflowError)?;
        }

        let mut filled_txs: Vec<_> = body
            .transactions
            .iter()
            .map(|t| state.fill_transaction(rotxn, t))
            .collect::<Result<_, _>>()?;

        for (i, filled_tx) in filled_txs.iter_mut().enumerate() {
            filled_tx.actor_address = body
                .actor_proofs
                .get(i)
                .and_then(|p| p.as_ref())
                .map(|auth| auth.get_address());
        }

        // Collect all inputs as fixed-width keys for efficient
        // double-spend detection via parallel sort-and-scan
        let total_inputs: usize =
            body.transactions.iter().map(|t| t.inputs.len()).sum();
        let mut all_input_keys = Vec::with_capacity(total_inputs);
        for filled_tx in &filled_txs {
            for input in &filled_tx.transaction.inputs {
                all_input_keys.push(OutPointKey::from_outpoint(input));
            }
        }

        // Sort and check for duplicate outpoints (double-spend detection)
        all_input_keys.par_sort_unstable();
        if all_input_keys.windows(2).any(|w| w[0] == w[1]) {
            return Err(Error::UtxoDoubleSpent);
        }

        for filled_tx in &filled_txs {
            Self::validate_filled_transaction(
                state,
                archive,
                rotxn,
                filled_tx,
                Some(next_height),
            )?;
        }

        if body.authorizations.len() != total_inputs {
            return Err(Error::AuthorizationError);
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

        Ok(PrevalidatedBlock {
            filled_transactions: filled_txs,
            computed_merkle_root: merkle_root,
            coinbase_value,
            next_height,
        })
    }

    pub fn validate_filled_transaction(
        state: &crate::state::State,
        archive: &crate::archive::Archive,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        override_height: Option<u32>,
    ) -> Result<bitcoin::Amount, Error> {
        use crate::math::trading::TRADE_MINER_FEE_SATS;

        if tx.is_claim_decision() {
            DecisionValidator::validate_complete_decision_claim(
                state,
                rotxn,
                tx,
                override_height,
            )?;
        }

        if tx.is_create_market() {
            MarketValidator::validate_market_creation(
                state,
                rotxn,
                tx,
                override_height,
            )?;
        }

        if tx.is_trade() {
            MarketValidator::validate_trade(
                state,
                archive,
                rotxn,
                tx,
                override_height,
            )?;
            return Ok(bitcoin::Amount::from_sat(TRADE_MINER_FEE_SATS));
        }

        if tx.is_amplify_beta() {
            MarketValidator::validate_amplify_beta(
                state,
                rotxn,
                tx,
                override_height,
            )?;
            return Ok(bitcoin::Amount::from_sat(TRADE_MINER_FEE_SATS));
        }

        if tx.is_submit_vote() {
            VoteValidator::validate_vote_submission(
                state,
                rotxn,
                tx,
                override_height,
            )?;
        }

        if tx.is_submit_ballot() {
            VoteValidator::validate_ballot(state, rotxn, tx, override_height)?;
        }

        if tx.is_transfer_reputation() {
            VoteValidator::validate_reputation_transfer(
                state,
                rotxn,
                tx,
                override_height,
            )?;
        }

        tx.bitcoin_fee()?.ok_or(Error::NotEnoughValueIn)
    }

    pub fn validate_transaction(
        state: &crate::state::State,
        archive: &crate::archive::Archive,
        rotxn: &RoTxn,
        transaction: &AuthorizedTransaction,
    ) -> Result<bitcoin::Amount, Error> {
        let mut filled_transaction =
            state.fill_transaction(rotxn, &transaction.transaction)?;
        filled_transaction.actor_address = transaction
            .actor_proof
            .as_ref()
            .map(|auth| auth.get_address());
        if transaction.authorizations.len()
            != filled_transaction.spent_utxos.len()
        {
            return Err(Error::AuthorizationError);
        }
        for (authorization, spent_utxo) in transaction
            .authorizations
            .iter()
            .zip(filled_transaction.spent_utxos.iter())
        {
            if authorization.get_address() != spent_utxo.address {
                return Err(Error::WrongPubKeyForAddress);
            }
        }
        if Authorization::verify_transaction(transaction).is_err() {
            return Err(Error::AuthorizationError);
        }
        let fee = Self::validate_filled_transaction(
            state,
            archive,
            rotxn,
            &filled_transaction,
            None,
        )?;
        Ok(fee)
    }

    pub fn validate_coinbase_outputs(
        coinbase: &[crate::types::Output],
        _height: u32,
    ) -> Result<(), Error> {
        for output in coinbase {
            match &output.content {
                OutputContent::Bitcoin(_) | OutputContent::Withdrawal(_) => {}
                OutputContent::MarketFunds { .. } => {
                    return Err(Error::BadCoinbaseOutputContent);
                }
            }
        }
        Ok(())
    }

    pub fn validate_fees(
        coinbase_value: bitcoin::Amount,
        filled_txs: &[FilledTransaction],
        skipped_indices: &HashSet<usize>,
    ) -> Result<(), Error> {
        use crate::math::trading::TRADE_MINER_FEE_SATS;

        let mut actual_total_fees = bitcoin::Amount::ZERO;
        for (idx, filled_tx) in filled_txs.iter().enumerate() {
            if skipped_indices.contains(&idx) {
                continue;
            }
            let tx_fee = if filled_tx.is_trade() || filled_tx.is_amplify_beta()
            {
                bitcoin::Amount::from_sat(TRADE_MINER_FEE_SATS)
            } else {
                filled_tx.bitcoin_fee()?.ok_or(Error::NotEnoughValueIn)?
            };
            actual_total_fees = actual_total_fees
                .checked_add(tx_fee)
                .ok_or(AmountOverflowError)?;
        }

        if coinbase_value > actual_total_fees {
            return Err(Error::NotEnoughFees);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Address, BitcoinOutputContent, Output, OutputContent};

    fn bitcoin_output(sats: u64) -> Output {
        Output {
            address: Address::ALL_ZEROS,
            content: OutputContent::Bitcoin(BitcoinOutputContent(
                bitcoin::Amount::from_sat(sats),
            )),
            memo: vec![],
        }
    }

    #[test]
    fn coinbase_bitcoin_output_always_valid() {
        let outputs = vec![bitcoin_output(5000)];
        assert!(BlockValidator::validate_coinbase_outputs(&outputs, 0).is_ok());
        assert!(
            BlockValidator::validate_coinbase_outputs(&outputs, 100).is_ok()
        );
    }

    #[test]
    fn coinbase_market_funds_always_rejected() {
        let outputs = vec![Output {
            address: Address::ALL_ZEROS,
            content: OutputContent::MarketFunds {
                market_id: [0u8; 6],
                amount: BitcoinOutputContent(bitcoin::Amount::from_sat(1000)),
                is_fee: false,
            },
            memo: vec![],
        }];
        assert!(
            BlockValidator::validate_coinbase_outputs(&outputs, 0).is_err()
        );
    }

    #[test]
    fn validate_fees_exact_match_ok() {
        let fees = bitcoin::Amount::from_sat(0);
        let skipped = HashSet::new();
        assert!(BlockValidator::validate_fees(fees, &[], &skipped).is_ok());
    }

    #[test]
    fn validate_fees_coinbase_exceeds_fees_rejected() {
        let coinbase = bitcoin::Amount::from_sat(1000);
        let skipped = HashSet::new();
        assert!(
            BlockValidator::validate_fees(coinbase, &[], &skipped).is_err()
        );
    }
}
