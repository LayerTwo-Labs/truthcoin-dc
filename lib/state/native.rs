//! Native share operations. All state changes use the enclosing block's LMDB
//! transaction. External replay derives receipts from successful transitions.
use std::collections::BTreeMap;

use fallible_iterator::FallibleIterator;
use heed::types::SerdeBincode;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sneed::{DatabaseUnique, RoTxn, RwTxn};

use super::{Error, ShareAccount, State, UtxoManager, markets::MarketId};
use crate::types::{
    Address, BitcoinOutputContent, FilledOutput, FilledOutputContent,
    FilledTransaction, OutPoint, TransactionData,
    native::{
        EscrowAssetV1, EscrowStatusV1, NativeActionV2, NativeId,
        NativeOperationV2, ShareEscrowV1,
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

#[derive(Clone)]
pub struct NativeDbs {
    pub reservations: DatabaseUnique<
        SerdeBincode<(Address, MarketId, u32)>,
        SerdeBincode<BTreeMap<NativeId, i64>>,
    >,
    pub escrows:
        DatabaseUnique<SerdeBincode<NativeId>, SerdeBincode<ShareEscrowV1>>,
    pub undo: DatabaseUnique<SerdeBincode<u32>, SerdeBincode<NativeUndoV1>>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct NativeUndoV1 {
    pub escrows: BTreeMap<NativeId, Option<ShareEscrowV1>>,
    pub accounts: BTreeMap<Address, Option<ShareAccount>>,
    pub cash_outputs: Vec<OutPoint>,
}

impl NativeDbs {
    pub const NUM_DBS: u32 = 3;

    pub fn new<Tls>(
        env: &sneed::Env<Tls>,
        txn: &mut RwTxn,
    ) -> Result<Self, Error> {
        Ok(Self {
            reservations: DatabaseUnique::create(
                env,
                txn,
                "native_reservations",
            )?,
            escrows: DatabaseUnique::create(env, txn, "native_escrows")?,
            undo: DatabaseUnique::create(env, txn, "native_undo")?,
        })
    }

    pub fn get_escrow(
        &self,
        txn: &RoTxn,
        id: NativeId,
    ) -> Result<Option<ShareEscrowV1>, Error> {
        Ok(self.escrows.try_get(txn, &id)?)
    }

    pub fn reserved_shares(
        &self,
        txn: &RoTxn,
        owner: Address,
        market: MarketId,
        outcome: u32,
    ) -> Result<i64, Error> {
        self.reservations
            .try_get(txn, &(owner, market, outcome))?
            .unwrap_or_default()
            .values()
            .try_fold(0i64, |total, q| {
                total
                    .checked_add(*q)
                    .ok_or_else(|| invalid("reserved shares overflow"))
            })
    }

    fn write_escrow(
        &self,
        txn: &mut RwTxn,
        id: NativeId,
        next: Option<&ShareEscrowV1>,
    ) -> Result<(), Error> {
        if let Some(previous) = self.escrows.try_get(txn, &id)? {
            if previous.status == EscrowStatusV1::Locked
                && previous.asset == EscrowAssetV1::Shares
            {
                let key = (
                    previous.owner,
                    previous.market_id,
                    previous.outcome_index,
                );
                let mut reservations =
                    self.reservations.try_get(txn, &key)?.unwrap_or_default();
                reservations.remove(&id);
                if reservations.is_empty() {
                    self.reservations.delete(txn, &key)?;
                } else {
                    self.reservations.put(txn, &key, &reservations)?;
                }
            }
        }
        if let Some(escrow) = next {
            if escrow.status == EscrowStatusV1::Locked
                && escrow.asset == EscrowAssetV1::Shares
            {
                let key =
                    (escrow.owner, escrow.market_id, escrow.outcome_index);
                let mut reservations =
                    self.reservations.try_get(txn, &key)?.unwrap_or_default();
                if reservations.len() >= 1024 {
                    return Err(invalid("too many active native reservations"));
                }
                reservations.insert(id, escrow.shares);
                self.reservations.put(txn, &key, &reservations)?;
            }
            self.escrows.put(txn, &id, escrow)?;
        } else {
            self.escrows.delete(txn, &id)?;
        }
        Ok(())
    }

    pub fn cash_liability(&self, txn: &RoTxn) -> Result<u64, Error> {
        let mut total = 0u64;
        let mut iter = self.escrows.iter(txn)?;
        while let Some((_, escrow)) = iter.next()? {
            if escrow.status == EscrowStatusV1::Locked {
                if let EscrowAssetV1::NativeCash(amount) = escrow.asset {
                    total = total
                        .checked_add(amount)
                        .ok_or_else(|| invalid("escrow cash overflow"))?;
                }
            }
        }
        Ok(total)
    }

    fn capture_escrow(
        &self,
        txn: &mut RwTxn,
        height: u32,
        id: NativeId,
    ) -> Result<(), Error> {
        let mut undo = self.undo.try_get(txn, &height)?.unwrap_or_default();
        if !undo.escrows.contains_key(&id) {
            undo.escrows.insert(id, self.escrows.try_get(txn, &id)?);
            self.undo.put(txn, &height, &undo)?;
        }
        Ok(())
    }

    pub(crate) fn capture_account(
        &self,
        state: &State,
        txn: &mut RwTxn,
        height: u32,
        address: Address,
    ) -> Result<(), Error> {
        let mut undo = self.undo.try_get(txn, &height)?.unwrap_or_default();
        if !undo.accounts.contains_key(&address) {
            undo.accounts.insert(
                address,
                state.markets().get_user_share_account(txn, &address)?,
            );
            self.undo.put(txn, &height, &undo)?;
        }
        Ok(())
    }

    pub(crate) fn create_escrow(
        &self,
        txn: &mut RwTxn,
        height: u32,
        escrow: &ShareEscrowV1,
    ) -> Result<(), Error> {
        if self.escrows.try_get(txn, &escrow.escrow_id)?.is_some() {
            return Err(invalid("escrow already exists"));
        }
        self.capture_escrow(txn, height, escrow.escrow_id)?;
        self.write_escrow(txn, escrow.escrow_id, Some(escrow))?;
        Ok(())
    }

    pub(crate) fn assign(
        &self,
        txn: &mut RwTxn,
        height: u32,
        filled: &FilledTransaction,
        id: NativeId,
        claim: crate::types::Address,
        refund: crate::types::Address,
        reference: NativeId,
    ) -> Result<(), Error> {
        let mut escrow = self
            .get_escrow(txn, id)?
            .ok_or_else(|| invalid("unknown native escrow"))?;
        if escrow.status != EscrowStatusV1::Locked || !escrow.mutable_rights {
            return Err(invalid("escrow is terminal or sealed"));
        }
        require_owner_input(filled, escrow.claim_address)?;
        require_owner_input(filled, escrow.refund_address)?;
        self.capture_escrow(txn, height, id)?;
        escrow.claim_address = claim;
        escrow.refund_address = refund;
        escrow.reference = reference;
        self.write_escrow(txn, id, Some(&escrow))
    }

    pub(crate) fn terminate(
        &self,
        state: &State,
        txn: &mut RwTxn,
        height: u32,
        escrow: &mut ShareEscrowV1,
        txid: NativeId,
        recipient: Address,
        refund: bool,
    ) -> Result<(), Error> {
        self.capture_escrow(txn, height, escrow.escrow_id)?;
        if let EscrowAssetV1::NativeCash(amount) = escrow.asset {
            if amount > 0 {
                let outpoint = cash_outpoint(txid);
                state.insert_utxo(
                    txn,
                    &outpoint,
                    &FilledOutput {
                        address: recipient,
                        content: FilledOutputContent::Bitcoin(
                            BitcoinOutputContent(bitcoin::Amount::from_sat(
                                amount,
                            )),
                        ),
                        memo: Vec::new(),
                    },
                )?;
                let mut undo =
                    self.undo.try_get(txn, &height)?.unwrap_or_default();
                undo.cash_outputs.push(outpoint);
                self.undo.put(txn, &height, &undo)?;
            }
        }
        escrow.status = if refund {
            EscrowStatusV1::Refunded {
                transaction_id: txid,
            }
        } else {
            EscrowStatusV1::Claimed {
                transaction_id: txid,
            }
        };
        self.write_escrow(txn, escrow.escrow_id, Some(escrow))?;
        Ok(())
    }

    /// Divide the existing owner/outcome payout without creating cash.
    /// Zero-valued locks retain their claim/refund conditions.
    pub(crate) fn settle_payout(
        &self,
        state: &State,
        txn: &mut RwTxn,
        payout: &super::markets::types::SharePayoutRecord,
        height: u32,
    ) -> Result<u64, Error> {
        let reservations = self
            .reservations
            .try_get(
                txn,
                &(payout.address, payout.market_id, payout.outcome_index),
            )?
            .unwrap_or_default();
        let mut locks = Vec::with_capacity(reservations.len());
        for id in reservations.keys() {
            locks.push((
                *id,
                self.escrows
                    .try_get(txn, id)?
                    .ok_or_else(|| invalid("missing reserved escrow"))?,
            ));
        }
        if locks.is_empty() {
            return Ok(payout.payout_sats);
        }
        self.capture_account(state, txn, height, payout.address)?;
        let quantities: Vec<_> = locks
            .iter()
            .map(|(id, escrow)| (*id, escrow.shares))
            .collect();
        let (ordinary, allocations) = allocate_settlement(
            payout.shares_redeemed,
            payout.payout_sats,
            &quantities,
        )?;
        for ((id, mut escrow), (_, amount)) in
            locks.into_iter().zip(allocations)
        {
            self.capture_escrow(txn, height, id)?;
            escrow.asset = EscrowAssetV1::NativeCash(amount);
            self.write_escrow(txn, id, Some(&escrow))?;
        }
        Ok(ordinary)
    }

    pub(crate) fn restore(
        &self,
        state: &State,
        txn: &mut RwTxn,
        height: u32,
    ) -> Result<(), Error> {
        if let Some(undo) = self.undo.try_get(txn, &height)? {
            for outpoint in undo.cash_outputs {
                state.delete_utxo(txn, &outpoint)?;
            }
            for (id, previous) in undo.escrows {
                self.write_escrow(txn, id, previous.as_ref())?;
            }
            for (address, account) in undo.accounts {
                state.markets().restore_share_account(
                    txn,
                    &address,
                    account.as_ref(),
                )?;
            }
            self.undo.delete(txn, &height)?;
        }
        Ok(())
    }
}

/// Canonical integer apportionment, keyed by lock id; ordinary shares sort first.
pub fn allocate_settlement(
    total_shares: i64,
    payout: u64,
    locks: &[(NativeId, i64)],
) -> Result<(u64, Vec<(NativeId, u64)>), Error> {
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
    Ok((
        amounts[0],
        locks
            .iter()
            .enumerate()
            .map(|(index, (id, _))| (*id, amounts[index + 1]))
            .collect(),
    ))
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
        NativeActionV2::MoveShares {
            owner,
            market_id,
            outcome_index,
            shares,
            ..
        }
        | NativeActionV2::LockShares {
            owner,
            market_id,
            outcome_index,
            shares,
            ..
        } => {
            require_owner_input(tx, *owner)?;
            validate_shares(state, txn, *market_id, *outcome_index, *shares)?;
        }
        // Assignment owners are checked during ordered execution, not at block start.
        _ => (),
    }
    Ok(())
}

fn validate_shares(
    state: &State,
    txn: &RoTxn,
    market: MarketId,
    outcome: u32,
    shares: i64,
) -> Result<(), Error> {
    if shares <= 0 {
        return Err(invalid("native share quantity must be positive"));
    }
    let market = state
        .markets()
        .get_market(txn, &market)?
        .ok_or_else(|| invalid("unknown native share market"))?;
    if !market.state().allows_trading()
        || (outcome as usize) >= market.shares().len()
    {
        return Err(invalid(
            "native share market is closed or outcome invalid",
        ));
    }
    Ok(())
}

pub fn check_deadline(
    op: &NativeOperationV2,
    parent: u32,
) -> Result<(), Error> {
    if parent < op.valid_from_parent || parent >= op.valid_before_parent {
        return Err(invalid("native action outside signed parent interval"));
    }
    if let NativeActionV2::LockShares {
        claim_before_parent,
        ..
    } = op.action
    {
        if parent >= claim_before_parent {
            return Err(invalid("expired native lock"));
        }
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
        Some(secret) => {
            if parent_height >= escrow.claim_before_parent {
                return Err(invalid("escrow claim expired"));
            }
            let digest: [u8; 32] = Sha256::digest(secret).into();
            if digest != escrow.hashlock {
                return Err(invalid("invalid escrow preimage"));
            }
        }
        None => {
            if parent_height < escrow.claim_before_parent {
                return Err(invalid("escrow refund too early"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn apportionment_conserves_integer_payout_and_zero_successors() {
        let ids = [([1; 32], 1), ([2; 32], 1)];
        assert_eq!(
            allocate_settlement(3, 2, &ids).unwrap(),
            (1, vec![([1; 32], 1), ([2; 32], 0)])
        );
        assert_eq!(
            allocate_settlement(3, 0, &ids).unwrap(),
            (0, vec![([1; 32], 0), ([2; 32], 0)])
        );
        for payout in 0..100 {
            let (ordinary, locked) =
                allocate_settlement(3, payout, &ids).unwrap();
            assert_eq!(
                ordinary + locked.iter().map(|(_, cash)| *cash).sum::<u64>(),
                payout
            );
        }
        assert!(allocate_settlement(1, 2, &ids).is_err());
    }
}
