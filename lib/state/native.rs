//! Native share operations. All state changes use the enclosing block's LMDB
//! transaction. External replay derives receipts from successful transitions.
use fallible_iterator::FallibleIterator;
use sha2::{Digest, Sha256};
use sneed::{RoTxn, RwTxn};

use super::{
    Error, State, UtxoManager,
    markets::{MarketId, ShareAccount},
};
use crate::types::{
    Address, BitcoinOutputContent, FilledOutput, FilledOutputContent,
    FilledTransaction, OutPoint, TransactionData,
    native::{
        EscrowAssetV1, EscrowStatusV1, NativeActionV3, NativeId,
        NativeOperationV3, ShareEscrowV1,
    },
};

/// Consensus fork parameter. Deployments must coordinate activation of this
/// release. Height zero supports fresh networks without changing old encodings.
pub const NATIVE_OPERATIONS_ACTIVATION_HEIGHT: u32 = 0;

pub(crate) fn invalid(reason: impl Into<String>) -> Error {
    Error::InvalidTransaction {
        reason: reason.into(),
    }
}

/// A borrowed view of existing share-account and consolidation-undo records.
/// This type creates no databases and owns no database handles.
pub struct NativeState<'a> {
    pub(crate) state: &'a State,
}

impl NativeState<'_> {
    fn account(
        &self,
        txn: &RoTxn,
        owner: Address,
    ) -> Result<ShareAccount, Error> {
        Ok(self
            .state
            .markets()
            .get_user_share_account(txn, &owner)?
            .unwrap_or_default())
    }
    pub fn get_escrow(
        &self,
        txn: &RoTxn,
        owner: Address,
        id: NativeId,
    ) -> Result<Option<ShareEscrowV1>, Error> {
        Ok(self.account(txn, owner)?.escrows.remove(&id))
    }
    pub fn reserved_shares(
        &self,
        txn: &RoTxn,
        owner: Address,
        market: MarketId,
        outcome: u32,
    ) -> Result<i64, Error> {
        self.account(txn, owner)?
            .escrows
            .values()
            .filter(|e| {
                e.market_id == market
                    && e.outcome_index == outcome
                    && e.asset == EscrowAssetV1::Shares
            })
            .try_fold(0i64, |sum, e| {
                sum.checked_add(e.shares)
                    .ok_or_else(|| invalid("reserved shares overflow"))
            })
    }
    fn write_escrow(
        &self,
        txn: &mut RwTxn,
        escrow: &ShareEscrowV1,
        remove: bool,
    ) -> Result<(), Error> {
        let mut account = self.account(txn, escrow.owner)?;
        if remove {
            account.escrows.remove(&escrow.escrow_id);
        } else {
            if !account.escrows.contains_key(&escrow.escrow_id)
                && account.escrows.len() >= 1024
            {
                return Err(invalid("too many live escrows in account"));
            }
            account.escrows.insert(escrow.escrow_id, escrow.clone());
        }
        self.state.markets().restore_share_account(
            txn,
            &escrow.owner,
            (!(account.positions.is_empty() && account.escrows.is_empty()))
                .then_some(&account),
        )
    }
    pub fn cash_liability(&self, txn: &RoTxn) -> Result<u64, Error> {
        let mut total = 0u64;
        let mut records = self.state.markets().share_accounts.iter(txn)?;
        while let Some((_, a)) = records.next()? {
            for e in a.escrows.values() {
                if let EscrowAssetV1::NativeCash(value) = e.asset {
                    total = total
                        .checked_add(value)
                        .ok_or_else(|| invalid("escrow cash overflow"))?;
                }
            }
        }
        Ok(total)
    }
    pub(crate) fn capture_account(
        &self,
        txn: &mut RwTxn,
        height: u32,
        address: Address,
    ) -> Result<(), Error> {
        let state = self.state;
        let mut undo = state
            .consolidation_undo
            .try_get(txn, &height)?
            .unwrap_or_default();
        if !undo.account_undo.accounts.contains_key(&address) {
            undo.account_undo.accounts.insert(
                address,
                state.markets().get_user_share_account(txn, &address)?,
            );
            state.consolidation_undo.put(txn, &height, &undo)?;
        }
        Ok(())
    }
    pub(crate) fn create_escrow(
        &self,
        txn: &mut RwTxn,
        height: u32,
        escrow: &ShareEscrowV1,
    ) -> Result<(), Error> {
        if self
            .get_escrow(txn, escrow.owner, escrow.escrow_id)?
            .is_some()
        {
            return Err(invalid("escrow already exists"));
        }
        self.capture_account(txn, height, escrow.owner)?;
        self.write_escrow(txn, escrow, false)
    }

    pub(crate) fn assign(
        &self,
        txn: &mut RwTxn,
        height: u32,
        filled: &FilledTransaction,
        owner: Address,
        id: NativeId,
        claim: crate::types::Address,
        refund: crate::types::Address,
        reference: NativeId,
    ) -> Result<(), Error> {
        let mut escrow = self
            .get_escrow(txn, owner, id)?
            .ok_or_else(|| invalid("unknown native escrow"))?;
        if escrow.status != EscrowStatusV1::Locked || !escrow.mutable_rights {
            return Err(invalid("escrow is terminal or sealed"));
        }
        require_owner_input(filled, escrow.claim_address)?;
        require_owner_input(filled, escrow.refund_address)?;
        self.capture_account(txn, height, owner)?;
        escrow.claim_address = claim;
        escrow.refund_address = refund;
        escrow.reference = reference;
        self.write_escrow(txn, &escrow, false)
    }

    pub(crate) fn terminate(
        &self,
        txn: &mut RwTxn,
        height: u32,
        escrow: &ShareEscrowV1,
        txid: NativeId,
        recipient: Address,
    ) -> Result<(), Error> {
        let state = self.state;
        self.capture_account(txn, height, escrow.owner)?;
        if let EscrowAssetV1::NativeCash(amount) = escrow.asset {
            if amount > 0 {
                let outpoint = cash_outpoint(txid);
                state.insert_utxo(
                    txn,
                    &outpoint,
                    &FilledOutput::new(
                        recipient,
                        FilledOutputContent::Bitcoin(BitcoinOutputContent(
                            bitcoin::Amount::from_sat(amount),
                        )),
                    ),
                )?;
                let mut undo = state
                    .consolidation_undo
                    .try_get(txn, &height)?
                    .unwrap_or_default();
                undo.account_undo.cash_outputs.push(outpoint);
                state.consolidation_undo.put(txn, &height, &undo)?;
            }
        }
        self.write_escrow(txn, escrow, true)
    }

    /// Divide the existing owner/outcome payout without creating cash.
    /// Zero-valued locks retain their claim/refund conditions.
    pub(crate) fn settle_payout(
        &self,
        txn: &mut RwTxn,
        payout: &super::markets::types::SharePayoutRecord,
        height: u32,
    ) -> Result<u64, Error> {
        let account = self.account(txn, payout.address)?;
        let locks: Vec<_> = account
            .escrows
            .into_iter()
            .filter(|(_, e)| {
                e.market_id == payout.market_id
                    && e.outcome_index == payout.outcome_index
                    && e.asset == EscrowAssetV1::Shares
            })
            .collect();
        if locks.is_empty() {
            return Ok(payout.payout_sats);
        }
        self.capture_account(txn, height, payout.address)?;
        let quantities: Vec<_> = locks
            .iter()
            .map(|(id, escrow)| (*id, escrow.shares))
            .collect();
        let (ordinary, allocations) = allocate_settlement(
            payout.shares_redeemed,
            payout.payout_sats,
            &quantities,
        )?;
        for ((_, mut escrow), amount) in locks.into_iter().zip(allocations) {
            escrow.asset = EscrowAssetV1::NativeCash(amount);
            self.write_escrow(txn, &escrow, false)?;
        }
        Ok(ordinary)
    }

    pub(crate) fn restore(
        &self,
        txn: &mut RwTxn,
        height: u32,
    ) -> Result<(), Error> {
        let state = self.state;
        if let Some(undo) = state.consolidation_undo.try_get(txn, &height)? {
            for point in undo.account_undo.cash_outputs {
                state.delete_utxo(txn, &point)?;
            }
            for (address, account) in undo.account_undo.accounts {
                state.markets().restore_share_account(
                    txn,
                    &address,
                    account.as_ref(),
                )?;
            }
            for (_, market) in undo.account_undo.markets {
                state.markets().restore_market(txn, &market)?;
            }
        }
        Ok(())
    }
}

/// Canonical integer apportionment, keyed by lock id; ordinary shares sort first.
fn allocate_settlement(
    total_shares: i64,
    payout: u64,
    locks: &[(NativeId, i64)],
) -> Result<(u64, Vec<u64>), Error> {
    if total_shares <= 0 {
        return Err(invalid("invalid settlement share count"));
    }
    let reserved = locks.iter().try_fold(0i64, |sum, (_, q)| {
        if *q <= 0 {
            return Err(invalid("invalid reserved share count"));
        }
        sum.checked_add(*q)
            .ok_or_else(|| invalid("reserved share overflow"))
    })?;
    let ordinary = total_shares
        .checked_sub(reserved)
        .filter(|q| *q >= 0)
        .ok_or_else(|| invalid("reservations exceed settled shares"))?;
    let mut buckets = vec![(None, ordinary)];
    buckets.extend(locks.iter().map(|(id, q)| (Some(*id), *q)));
    let denominator = total_shares as u128;
    let mut amounts = Vec::with_capacity(buckets.len());
    let mut remainders = Vec::new();
    let mut paid = 0u64;
    for (index, (id, quantity)) in buckets.iter().enumerate() {
        let numerator = (*quantity as u128) * (payout as u128);
        let amount = (numerator / denominator) as u64;
        paid = paid
            .checked_add(amount)
            .ok_or_else(|| invalid("allocation overflow"))?;
        amounts.push(amount);
        remainders.push((index, numerator % denominator, *id));
    }
    remainders.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.2.cmp(&b.2)));
    for (index, _, _) in remainders.into_iter().take((payout - paid) as usize) {
        amounts[index] += 1;
    }
    Ok((amounts.remove(0), amounts))
}

pub fn cash_outpoint(txid: NativeId) -> OutPoint {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"TRUTHCOIN_NATIVE_ESCROW_CASH_V1\0");
    hasher.update(&txid);
    OutPoint::Payout {
        hash: (*hasher.finalize().as_bytes()).into(),
        vout: 0,
    }
}

/// Structural/authorization checks are also used by mempool admission. Parent
/// deadline and ordered share availability checks run in block application.
/// Called only on filled inputs whose full transaction signatures are verified by
/// normal block authorization. Actor-only proofs never authorize native ownership.
pub(crate) fn require_owner_input(
    tx: &FilledTransaction,
    owner: crate::types::Address,
) -> Result<(), Error> {
    if !tx.spent_utxos.iter().any(|output| {
        output.address == owner
            && matches!(output.content, FilledOutputContent::Bitcoin(_))
    }) {
        return Err(invalid(
            "native action requires owner-controlled ordinary input",
        ));
    }
    Ok(())
}

pub fn validate(
    state: &State,
    archive: &crate::archive::Archive,
    txn: &RoTxn,
    tx: &FilledTransaction,
) -> Result<(), Error> {
    let Some(TransactionData::NativeOperation(op)) = &tx.transaction.data
    else {
        return Ok(());
    };
    if tx.transaction.inputs.is_empty()
        || tx.transaction.inputs.len() != tx.spent_utxos.len()
        || !tx
            .spent_utxos
            .iter()
            .any(|o| matches!(o.content, FilledOutputContent::Bitcoin(_)))
        || op.valid_from_parent >= op.valid_before_parent
    {
        return Err(invalid(
            "native action requires inputs and a valid height interval",
        ));
    }
    let tip = state
        .try_get_tip(txn)?
        .ok_or_else(|| invalid("native action requires genesis"))?;
    let height = state
        .try_get_height(txn)?
        .ok_or_else(|| invalid("missing native height"))?;
    if archive.get_nth_ancestor(txn, tip, height)? != op.genesis_hash {
        return Err(invalid("native action belongs to another chain"));
    }
    match &op.action {
        NativeActionV3::MoveShares {
            owner,
            market_id,
            outcome_index,
            shares,
            ..
        }
        | NativeActionV3::LockShares {
            owner,
            market_id,
            outcome_index,
            shares,
            ..
        } => {
            require_owner_input(tx, *owner)?;
            if *shares <= 0 {
                return Err(invalid("native share quantity must be positive"));
            }
            let market = state
                .markets()
                .get_market(txn, market_id)?
                .ok_or_else(|| invalid("unknown native share market"))?;
            if !market.state().allows_trading()
                || (*outcome_index as usize) >= market.shares().len()
            {
                return Err(invalid(
                    "native share market is closed or outcome invalid",
                ));
            }
        }
        // Assignment owners are checked during ordered execution, not at block start.
        _ => (),
    }
    Ok(())
}

pub fn check_deadline(
    op: &NativeOperationV3,
    parent: u32,
) -> Result<(), Error> {
    if parent < op.valid_from_parent || parent >= op.valid_before_parent {
        return Err(invalid("native action outside signed parent interval"));
    }
    if let NativeActionV3::LockShares {
        claim_before_parent,
        ..
    } = op.action
        && parent >= claim_before_parent
    {
        return Err(invalid("expired native lock"));
    }
    Ok(())
}

pub fn validate_terminal(
    escrow: &ShareEscrowV1,
    parent_height: u32,
    preimage: Option<&NativeId>,
) -> Result<(), Error> {
    if escrow.status != EscrowStatusV1::Locked {
        return Err(invalid("escrow already terminated"));
    }
    match preimage {
        Some(_) if parent_height >= escrow.claim_before_parent => {
            return Err(invalid("escrow claim expired"));
        }
        Some(secret)
            if <NativeId>::from(Sha256::digest(secret)) != escrow.hashlock =>
        {
            return Err(invalid("invalid escrow preimage"));
        }
        None if parent_height < escrow.claim_before_parent => {
            return Err(invalid("escrow refund too early"));
        }
        _ => (),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn apportionment_conserves_integer_payout_and_zero_successors() {
        let ids = [([1; 32], 1), ([2; 32], 1)];
        assert_eq!(allocate_settlement(3, 2, &ids).unwrap(), (1, vec![1, 0]));
        assert_eq!(allocate_settlement(3, 0, &ids).unwrap(), (0, vec![0, 0]));
        for payout in 0..100 {
            let (ordinary, locked) =
                allocate_settlement(3, payout, &ids).unwrap();
            assert_eq!(ordinary + locked.iter().sum::<u64>(), payout);
        }
        assert!(allocate_settlement(1, 2, &ids).is_err());
    }
}
