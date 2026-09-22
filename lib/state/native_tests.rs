//! Native lifecycle regressions. Execution helpers model already verified inputs;
//! signature tests below separately exercise upstream full-transaction verification.
use super::*;
use crate::state::markets::{Market, MarketState, types::SharePayoutRecord};
use crate::types::{
    BitcoinOutputContent, FilledOutputContent, Transaction, native::*,
};
use bitcoin::hashes::Hash;
use sha2::{Digest, Sha256};
fn fixture() -> (sneed::Env, State, tempfile::TempDir, MarketId, Address) {
    let dir = tempfile::tempdir().unwrap();
    let mut options = heed::EnvOpenOptions::new();
    options
        .map_size(64 * 1024 * 1024)
        .max_dbs(State::NUM_DBS + crate::archive::Archive::NUM_DBS);
    let env = unsafe { sneed::Env::open(&options, dir.path()) }.unwrap();
    let state = State::new(&env, None).unwrap();
    let owner = Address([1; 20]);
    let market_id = MarketId::new([1; 6]);
    let market = Market {
        id: market_id,
        title: "native escrow test".into(),
        description: String::new(),
        tags: vec![],
        creator_address: owner,
        dimension_specs: vec![],
        decision_ids: vec![],
        state_combos: vec![vec![0], vec![1]],
        created_at_height: 0,
        expires_at_height: None,
        tau_from_now: 0,
        storage_fee_sats: 0,
        market_state: MarketState::Trading,
        trading_fee: 0.005,
        liquidity_base_sats: 100,
        shares: ndarray::array![100, 0],
        final_prices: ndarray::array![0.0, 0.0],
        version: 0,
        last_updated_height: 0,
        total_volume_sats: 0,
        outcome_volumes_sats: vec![0, 0],
        tx_pow_hash_selector: 0,
        tx_pow_ordering: 0,
        tx_pow_difficulty: 0,
    };
    let mut txn = env.write_txn().unwrap();
    state.markets().add_market(&mut txn, &market).unwrap();
    state
        .markets()
        .add_shares_to_account(&mut txn, &owner, market_id, 0, 100, 0)
        .unwrap();
    txn.commit().unwrap();
    (env, state, dir, market_id, owner)
}

fn filled(
    action: NativeActionV3,
    owners: &[Address],
    tag: u8,
) -> FilledTransaction {
    FilledTransaction {
        actor_address: None,
        spent_utxos: owners
            .iter()
            .map(|address| FilledOutput {
                address: *address,
                content: FilledOutputContent::Bitcoin(BitcoinOutputContent(
                    bitcoin::Amount::from_sat(10),
                )),
                memo: vec![],
            })
            .collect(),
        transaction: Transaction {
            inputs: owners
                .iter()
                .enumerate()
                .map(|(vout, _)| OutPoint::Regular {
                    txid: [tag; 32].into(),
                    vout: vout as u32,
                })
                .collect(),
            outputs: vec![],
            memo: vec![],
            data: Some(TxData::NativeOperation(NativeOperationV3 {
                genesis_hash: [0; 32].into(),
                valid_from_parent: 0,
                valid_before_parent: 100,
                reference: [tag; 32],
                action,
            })),
        },
    }
}
fn op(tx: &FilledTransaction) -> &NativeOperationV3 {
    let Some(TxData::NativeOperation(op)) = &tx.transaction.data else {
        panic!()
    };
    op
}
fn id(tx: &FilledTransaction) -> NativeId {
    escrow_id(op(tx).genesis_hash, tx.txid().0)
}
fn run(
    state: &State,
    txn: &mut RwTxn,
    update: &mut StateUpdate,
    tx: &FilledTransaction,
    height: u32,
    parent: u32,
) -> Result<bool, Error> {
    apply_native_operation(state, txn, tx, op(tx), update, height, parent)
}
fn lock(
    owner: Address,
    market_id: MarketId,
    tag: u8,
    shares: i64,
    mutable_rights: bool,
) -> FilledTransaction {
    filled(
        NativeActionV3::LockShares {
            owner,
            claim_address: Address([2; 20]),
            refund_address: Address([3; 20]),
            market_id,
            outcome_index: 0,
            shares,
            hashlock: Sha256::digest([9; 32]).into(),
            claim_before_parent: 10,
            mutable_rights,
        },
        &[owner],
        tag,
    )
}
fn resolve(eid: NativeId, claim: bool, tag: u8) -> FilledTransaction {
    filled(
        NativeActionV3::ResolveEscrow {
            original_owner: Address([1; 20]),
            escrow_id: eid,
            resolution: if claim {
                EscrowResolutionV3::Claim { preimage: [9; 32] }
            } else {
                EscrowResolutionV3::Refund
            },
        },
        &[Address([8; 20])],
        tag,
    )
}
fn balance(
    state: &State,
    txn: &sneed::RoTxn,
    owner: Address,
    market: MarketId,
) -> i64 {
    state
        .markets()
        .get_user_share_account(txn, &owner)
        .unwrap()
        .and_then(|a| a.positions.get(&(market, 0)).copied())
        .unwrap_or(0)
}

#[test]
fn reservations_block_moves_sells_and_overlapping_locks() {
    let (env, state, _dir, market, owner) = fixture();
    let mut txn = env.write_txn().unwrap();
    let mut update = StateUpdate::new();
    let locked = lock(owner, market, 1, 70, false);
    run(&state, &mut txn, &mut update, &locked, 1, 5).unwrap();
    assert!(
        run(
            &state,
            &mut txn,
            &mut update,
            &lock(owner, market, 2, 31, false),
            1,
            5
        )
        .is_err()
    );
    let movement = filled(
        NativeActionV3::MoveShares {
            owner,
            recipient: Address([4; 20]),
            market_id: market,
            outcome_index: 0,
            shares: 31,
        },
        &[owner],
        3,
    );
    assert!(run(&state, &mut txn, &mut update, &movement, 1, 5).is_err());
    let mut sell = movement.clone();
    sell.transaction.data = Some(TxData::Trade {
        market_id: market,
        outcome_index: 0,
        shares: -31,
        trader: owner,
        limit_sats: 0,
        tx_pow_nonce: None,
        prev_block_hash: [0; 32].into(),
    });
    assert!(apply_trade(&state, &mut txn, &sell, &mut update, 1).is_err());
    let wrong = filled(op(&movement).action.clone(), &[Address([5; 20])], 4);
    assert!(run(&state, &mut txn, &mut update, &wrong, 1, 5).is_err());
    let mut actor_only = movement.clone();
    actor_only.spent_utxos.clear();
    actor_only.transaction.inputs.clear();
    actor_only.actor_address = Some(owner);
    assert!(run(&state, &mut txn, &mut update, &actor_only, 1, 5).is_err());
    let mut available = movement.clone();
    if let Some(TxData::NativeOperation(op)) = &mut available.transaction.data {
        if let NativeActionV3::MoveShares { shares, .. } = &mut op.action {
            *shares = 30;
        }
    }
    run(&state, &mut txn, &mut update, &available, 1, 5).unwrap();
    assert!(run(&state, &mut txn, &mut update, &available, 1, 5).is_err());
    let claim = resolve(id(&locked), true, 5);
    run(&state, &mut txn, &mut update, &claim, 1, 9).unwrap();
    assert!(run(&state, &mut txn, &mut update, &claim, 1, 9).is_err());
    assert!(
        run(
            &state,
            &mut txn,
            &mut update,
            &resolve(id(&locked), false, 6),
            1,
            10
        )
        .is_err()
    );
    update.apply_all_changes(&state, &mut txn, 1).unwrap();
    assert_eq!(balance(&state, &txn, Address([2; 20]), market), 70);
    assert_eq!(balance(&state, &txn, owner, market), 0);
    assert_eq!(balance(&state, &txn, Address([4; 20]), market), 30);
    state.native().restore(&state, &mut txn, 1).unwrap();
    assert_eq!(balance(&state, &txn, owner, market), 100);
    assert!(
        state
            .markets()
            .get_user_share_account(&txn, &Address([2; 20]))
            .unwrap()
            .is_none()
    );
}

#[test]
fn assignment_requires_current_both_owners_and_preserves_sealed_rights() {
    for mutable in [false, true] {
        let (env, state, _dir, market, owner) = fixture();
        let mut txn = env.write_txn().unwrap();
        let mut update = StateUpdate::new();
        let locked = lock(owner, market, 1, 70, mutable);
        run(&state, &mut txn, &mut update, &locked, 1, 5).unwrap();
        let before = state
            .native()
            .get_escrow(&txn, owner, id(&locked))
            .unwrap()
            .unwrap();
        let action = NativeActionV3::AssignEscrow {
            original_owner: Address([1; 20]),
            escrow_id: id(&locked),
            new_claim_address: Address([4; 20]),
            new_refund_address: Address([4; 20]),
        };
        for owners in
            [vec![Address([2; 20])], vec![Address([3; 20])], vec![owner]]
        {
            assert!(
                run(
                    &state,
                    &mut txn,
                    &mut update,
                    &filled(action.clone(), &owners, 2),
                    1,
                    5
                )
                .is_err()
            );
        }
        let assigned = filled(action, &[Address([2; 20]), Address([3; 20])], 3);
        if !mutable {
            assert!(
                run(&state, &mut txn, &mut update, &assigned, 1, 5).is_err()
            );
            assert_eq!(
                state.native().get_escrow(&txn, owner, id(&locked)).unwrap(),
                Some(before)
            );
            continue;
        }
        run(&state, &mut txn, &mut update, &assigned, 1, 5).unwrap();
        assert!(run(&state, &mut txn, &mut update, &assigned, 1, 5).is_err());
        let after = state
            .native()
            .get_escrow(&txn, owner, id(&locked))
            .unwrap()
            .unwrap();
        assert_eq!(
            (after.shares, after.hashlock, after.claim_before_parent),
            (before.shares, before.hashlock, before.claim_before_parent)
        );
        let back = filled(
            NativeActionV3::AssignEscrow {
                original_owner: Address([1; 20]),
                escrow_id: id(&locked),
                new_claim_address: Address([2; 20]),
                new_refund_address: Address([3; 20]),
            },
            &[Address([4; 20])],
            4,
        );
        run(&state, &mut txn, &mut update, &back, 1, 5).unwrap();
        run(
            &state,
            &mut txn,
            &mut update,
            &resolve(id(&locked), false, 5),
            1,
            10,
        )
        .unwrap();
        assert!(run(&state, &mut txn, &mut update, &assigned, 1, 11).is_err());
    }
}

#[test]
fn deadlines_and_preimages_fail_without_mutating_escrow() {
    let (env, state, _dir, market, owner) = fixture();
    let mut txn = env.write_txn().unwrap();
    let mut update = StateUpdate::new();
    let locked = lock(owner, market, 1, 70, false);
    assert!(run(&state, &mut txn, &mut update, &locked, 1, 10).is_err());
    run(&state, &mut txn, &mut update, &locked, 1, 5).unwrap();
    let wrong = filled(
        NativeActionV3::ResolveEscrow {
            original_owner: Address([1; 20]),
            escrow_id: id(&locked),
            resolution: EscrowResolutionV3::Claim { preimage: [8; 32] },
        },
        &[owner],
        2,
    );
    assert!(run(&state, &mut txn, &mut update, &wrong, 1, 5).is_err());
    assert!(
        run(
            &state,
            &mut txn,
            &mut update,
            &resolve(id(&locked), false, 3),
            1,
            9
        )
        .is_err()
    );
    assert!(
        run(
            &state,
            &mut txn,
            &mut update,
            &resolve(id(&locked), true, 4),
            1,
            10
        )
        .is_err()
    );
    let mut expired = resolve(id(&locked), false, 5);
    if let Some(TxData::NativeOperation(op)) = &mut expired.transaction.data {
        op.valid_before_parent = 10;
    }
    assert!(run(&state, &mut txn, &mut update, &expired, 1, 10).is_err());
    assert_eq!(
        state
            .native()
            .get_escrow(&txn, owner, id(&locked))
            .unwrap()
            .unwrap()
            .status,
        EscrowStatusV1::Locked
    );
    run(
        &state,
        &mut txn,
        &mut update,
        &resolve(id(&locked), false, 6),
        1,
        10,
    )
    .unwrap();
}

#[test]
fn settlement_cash_remains_conditional_and_assignment_undo_is_exact() {
    for payout in [0, 1, 19, u64::MAX] {
        let (env, state, _dir, market, owner) = fixture();
        let mut txn = env.write_txn().unwrap();
        let mut update = StateUpdate::new();
        let locked = lock(owner, market, 1, 70, true);
        run(&state, &mut txn, &mut update, &locked, 1, 5).unwrap();
        let original = state
            .native()
            .get_escrow(&txn, owner, id(&locked))
            .unwrap()
            .unwrap();
        let ordinary = state
            .native()
            .settle_payout(
                &state,
                &mut txn,
                &SharePayoutRecord {
                    market_id: market,
                    address: owner,
                    outcome_index: 0,
                    shares_redeemed: 100,
                    final_price: 0.5,
                    payout_sats: payout,
                },
                2,
            )
            .unwrap();
        let cash = state.native().cash_liability(&txn).unwrap();
        assert_eq!(ordinary.checked_add(cash), Some(payout));
        let assigned = filled(
            NativeActionV3::AssignEscrow {
                original_owner: Address([1; 20]),
                escrow_id: id(&locked),
                new_claim_address: Address([4; 20]),
                new_refund_address: Address([4; 20]),
            },
            &[Address([2; 20]), Address([3; 20])],
            2,
        );
        run(&state, &mut txn, &mut update, &assigned, 2, 11).unwrap();
        let refund = resolve(id(&locked), false, 3);
        run(&state, &mut txn, &mut update, &refund, 2, 11).unwrap();
        assert_eq!(state.native().cash_liability(&txn).unwrap(), 0);
        let output = state
            .utxos
            .try_get(
                &txn,
                &OutPointKey::from_outpoint(
                    &crate::state::native::cash_outpoint(refund.txid().0),
                ),
            )
            .unwrap();
        if cash > 0 {
            let output = output.unwrap();
            assert_eq!(output.address, Address([4; 20]));
            assert_eq!(output.get_bitcoin_value().to_sat(), cash);
        } else {
            assert!(output.is_none());
        }
        state.native().restore(&state, &mut txn, 2).unwrap();
        assert_eq!(
            state.native().get_escrow(&txn, owner, id(&locked)).unwrap(),
            Some(original)
        );
        assert!(
            state
                .utxos
                .try_get(
                    &txn,
                    &OutPointKey::from_outpoint(
                        &crate::state::native::cash_outpoint(refund.txid().0)
                    )
                )
                .unwrap()
                .is_none()
        );
        assert_eq!(
            state
                .native()
                .reserved_shares(&txn, owner, market, 0)
                .unwrap(),
            70
        );
    }
}

#[test]
fn signatures_bind_native_payload_and_both_assignment_inputs() {
    use crate::authorization::{self, BatchVerificationContext, SigningKey};
    use crate::types::VerifyingKey;
    let key = |n| {
        SigningKey::from_scalar(curve25519_dalek::Scalar::from_bytes_mod_order(
            [n; 32],
        ))
        .unwrap()
    };
    let a = key(1);
    let b = key(2);
    let aa = authorization::get_address(&VerifyingKey::from(&a));
    let bb = authorization::get_address(&VerifyingKey::from(&b));
    let tx = filled(
        NativeActionV3::AssignEscrow {
            original_owner: Address([1; 20]),
            escrow_id: [3; 32],
            new_claim_address: aa,
            new_refund_address: bb,
        },
        &[aa, bb],
        7,
    );
    let mut rng = rand::rng();
    let context = BatchVerificationContext::new(&mut rng);
    let signed = authorization::authorize(
        &mut rng,
        &[(aa, &a), (bb, &b)],
        tx.transaction,
    )
    .unwrap();
    authorization::verify_authorized_transaction(&context, &signed).unwrap();
    for change in 0..5 {
        let mut changed = signed.clone();
        let Some(TxData::NativeOperation(op)) = &mut changed.transaction.data
        else {
            panic!()
        };
        match change {
            0 => op.genesis_hash = [1; 32].into(),
            1 => op.valid_before_parent += 1,
            2 => op.reference[0] ^= 1,
            3 => {
                op.action = NativeActionV3::AssignEscrow {
                    original_owner: Address([1; 20]),
                    escrow_id: [3; 32],
                    new_claim_address: bb,
                    new_refund_address: aa,
                }
            }
            _ => {
                changed.transaction.inputs[0] = OutPoint::Regular {
                    txid: [8; 32].into(),
                    vout: 0,
                }
            }
        }
        assert!(
            authorization::verify_authorized_transaction(&context, &changed)
                .is_err()
        );
    }
    let mut missing = signed;
    missing.authorizations.pop();
    assert!(
        authorization::verify_authorized_transaction(&context, &missing)
            .is_err()
    );
}

#[test]
fn connected_blocks_restore_inputs_accounts_and_escrows_after_reorg() {
    connected_reorg(false);
}

#[test]
fn mixed_trade_and_lock_reorg_restores_market_accounts_and_cash() {
    connected_reorg(true);
}

fn connected_reorg(with_trade: bool) {
    use fallible_iterator::FallibleIterator;
    let (env, state, _dir, market, owner) = fixture();
    let archive = crate::archive::Archive::new(&env).unwrap();
    let mut txn = env.write_txn().unwrap();
    let original = bincode::serialize(
        &state
            .markets()
            .get_user_share_account(&txn, &owner)
            .unwrap(),
    )
    .unwrap();
    let genesis_body = Body::new(
        vec![],
        vec![crate::types::Output {
            address: owner,
            content: crate::types::OutputContent::Bitcoin(
                BitcoinOutputContent(bitcoin::Amount::ZERO),
            ),
            memo: vec![],
        }],
    );
    let header = |prev, parent, body: &Body| Header {
        prev_side_hash: prev,
        prev_main_hash: bitcoin::BlockHash::from_byte_array([parent; 32]),
        merkle_root: Body::compute_merkle_root(
            &body.coinbase,
            &body.transactions,
        ),
    };
    let genesis = header(None, 1, &genesis_body);
    connect_prevalidated(
        &state,
        &mut txn,
        &genesis,
        &genesis_body,
        100,
        crate::state::PrevalidatedBlock {
            filled_transactions: vec![],
            computed_merkle_root: genesis.merkle_root,
            coinbase_value: bitcoin::Amount::ZERO,
            next_height: 0,
            parent_height: 4,
        },
    )
    .unwrap();
    let mut locked = lock(owner, market, 1, 70, false);
    if let Some(TxData::NativeOperation(op)) = &mut locked.transaction.data {
        op.genesis_hash = genesis.hash();
    }
    crate::state::native::validate(&state, &archive, &txn, &locked).unwrap();
    let mut wrong_chain = locked.clone();
    if let Some(TxData::NativeOperation(op)) = &mut wrong_chain.transaction.data
    {
        op.genesis_hash = [8; 32].into();
    }
    assert!(
        crate::state::native::validate(&state, &archive, &txn, &wrong_chain)
            .is_err()
    );
    let mut no_input = locked.clone();
    no_input.transaction.inputs.clear();
    no_input.spent_utxos.clear();
    assert!(
        crate::state::native::validate(&state, &archive, &txn, &no_input)
            .is_err()
    );
    let claim = resolve(id(&locked), true, 9);
    for t in [&locked, &claim] {
        for (point, output) in t.transaction.inputs.iter().zip(&t.spent_utxos) {
            state.insert_utxo(&mut txn, point, output).unwrap();
        }
    }
    let mut block_txs = vec![locked.clone()];
    if with_trade {
        let mut buy = lock(owner, market, 20, 1, false);
        buy.transaction.data = Some(TxData::Trade {
            market_id: market,
            outcome_index: 0,
            shares: 10,
            trader: owner,
            limit_sats: 10_000,
            tx_pow_nonce: None,
            prev_block_hash: genesis.hash(),
        });
        buy.spent_utxos[0].content = FilledOutputContent::Bitcoin(
            BitcoinOutputContent(bitcoin::Amount::from_sat(10_000)),
        );
        state
            .insert_utxo(
                &mut txn,
                &buy.transaction.inputs[0],
                &buy.spent_utxos[0],
            )
            .unwrap();
        block_txs.insert(0, buy);
    }
    let before_market =
        bincode::serialize(&state.markets().get_market(&txn, &market).unwrap())
            .unwrap();
    let before_utxos: Vec<_> =
        state.utxos.iter(&txn).unwrap().collect::<Vec<_>>().unwrap();
    let body = Body {
        coinbase: vec![],
        transactions: block_txs.iter().map(|t| t.transaction.clone()).collect(),
        authorizations: vec![],
        actor_proofs: vec![None; block_txs.len()],
    };
    let claim_body = Body {
        coinbase: vec![],
        transactions: vec![claim.transaction.clone()],
        authorizations: vec![],
        actor_proofs: vec![None],
    };
    let lock_header = header(Some(genesis.hash()), 2, &body);
    let claim_header = header(Some(lock_header.hash()), 3, &claim_body);
    for _ in 0..2 {
        connect_prevalidated(
            &state,
            &mut txn,
            &lock_header,
            &body,
            101,
            crate::state::PrevalidatedBlock {
                filled_transactions: block_txs.clone(),
                computed_merkle_root: lock_header.merkle_root,
                coinbase_value: bitcoin::Amount::ZERO,
                next_height: 1,
                parent_height: 5,
            },
        )
        .unwrap();
        if with_trade {
            assert_eq!(balance(&state, &txn, owner, market), 110);
        }
        // The native action has no nonce table: the normal consumed input prevents replay.
        assert!(state.fill_transaction(&txn, &locked.transaction).is_err());
        connect_prevalidated(
            &state,
            &mut txn,
            &claim_header,
            &claim_body,
            102,
            crate::state::PrevalidatedBlock {
                filled_transactions: vec![claim.clone()],
                computed_merkle_root: claim_header.merkle_root,
                coinbase_value: bitcoin::Amount::ZERO,
                next_height: 2,
                parent_height: 6,
            },
        )
        .unwrap();
        assert_eq!(balance(&state, &txn, Address([2; 20]), market), 70);
        disconnect_tip(&state, &mut txn, &claim_header, &claim_body).unwrap();
        assert_eq!(
            state
                .native()
                .reserved_shares(&txn, owner, market, 0)
                .unwrap(),
            70
        );
        disconnect_tip(&state, &mut txn, &lock_header, &body).unwrap();
        assert!(
            state
                .native()
                .get_escrow(&txn, owner, id(&locked))
                .unwrap()
                .is_none()
        );
        assert_eq!(state.try_get_mainchain_timestamp(&txn).unwrap(), Some(100));
        assert_eq!(
            bincode::serialize(
                &state
                    .markets()
                    .get_user_share_account(&txn, &owner)
                    .unwrap()
            )
            .unwrap(),
            original
        );
        assert!(
            state
                .markets()
                .get_user_share_account(&txn, &Address([2; 20]))
                .unwrap()
                .is_none()
        );
        state.fill_transaction(&txn, &locked.transaction).unwrap();
        assert_eq!(
            bincode::serialize(
                &state.markets().get_market(&txn, &market).unwrap()
            )
            .unwrap(),
            before_market
        );
        let after_utxos: Vec<_> =
            state.utxos.iter(&txn).unwrap().collect::<Vec<_>>().unwrap();
        assert_eq!(after_utxos, before_utxos);
    }
}

#[test]
fn automatic_settlement_retains_cash_only_account_and_undo_restores_it() {
    use crate::state::markets::types::MarketPayoutSummary;
    for payout in [0, 51] {
        let (env, state, _dir, market, owner) = fixture();
        let mut txn = env.write_txn().unwrap();
        let mut update = StateUpdate::new();
        let locked = lock(owner, market, 1, 70, true);
        run(&state, &mut txn, &mut update, &locked, 1, 5).unwrap();
        let original = state
            .markets()
            .get_user_share_account(&txn, &owner)
            .unwrap()
            .unwrap();
        let summary = MarketPayoutSummary {
            market_id: market,
            treasury_distributed: payout,
            total_fees_distributed: 0,
            shareholder_count: 1,
            payouts: vec![SharePayoutRecord {
                market_id: market,
                address: owner,
                outcome_index: 0,
                shares_redeemed: 100,
                final_price: 0.51,
                payout_sats: payout,
            }],
            fee_payouts: vec![],
            creator_refund: None,
            block_height: 2,
        };
        state
            .markets()
            .apply_automatic_share_payouts(&state, &mut txn, &summary, 2)
            .unwrap();
        let account = state
            .markets()
            .get_user_share_account(&txn, &owner)
            .unwrap()
            .unwrap();
        assert!(account.positions.is_empty());
        assert_eq!(account.escrows.len(), 1);
        let cash = state.native().cash_liability(&txn).unwrap();
        assert_eq!(cash, if payout == 0 { 0 } else { 36 });
        let wrong = filled(
            NativeActionV3::ResolveEscrow {
                original_owner: Address([9; 20]),
                escrow_id: id(&locked),
                resolution: EscrowResolutionV3::Refund,
            },
            &[owner],
            9,
        );
        assert!(run(&state, &mut txn, &mut update, &wrong, 3, 10).is_err());
        let assign = filled(
            NativeActionV3::AssignEscrow {
                original_owner: owner,
                escrow_id: id(&locked),
                new_claim_address: Address([4; 20]),
                new_refund_address: Address([4; 20]),
            },
            &[Address([2; 20]), Address([3; 20])],
            2,
        );
        run(&state, &mut txn, &mut update, &assign, 3, 10).unwrap();
        let refund = resolve(id(&locked), false, 3);
        run(&state, &mut txn, &mut update, &refund, 3, 10).unwrap();
        assert!(
            state
                .markets()
                .get_user_share_account(&txn, &owner)
                .unwrap()
                .is_none()
        );
        assert_eq!(state.native().cash_liability(&txn).unwrap(), 0);
        let out = state
            .utxos
            .try_get(
                &txn,
                &OutPointKey::from_outpoint(
                    &crate::state::native::cash_outpoint(refund.txid().0),
                ),
            )
            .unwrap();
        assert_eq!(out.is_some(), cash > 0);
        if let Some(out) = out {
            assert_eq!(out.address, Address([4; 20]));
            assert_eq!(out.get_bitcoin_value().to_sat(), cash);
        }
        state.native().restore(&state, &mut txn, 3).unwrap();
        assert_eq!(
            state
                .markets()
                .get_user_share_account(&txn, &owner)
                .unwrap(),
            Some(account)
        );
        state
            .markets()
            .revert_automatic_share_payouts(&state, &mut txn, &summary, 2)
            .unwrap();
        state.native().restore(&state, &mut txn, 2).unwrap();
        assert_eq!(
            state
                .markets()
                .get_user_share_account(&txn, &owner)
                .unwrap(),
            Some(original)
        );
    }
}

#[test]
fn escrow_limit_blocks_only_new_locks_and_does_not_require_an_index() {
    let (env, state, _dir, market, owner) = fixture();
    let mut txn = env.write_txn().unwrap();
    let mut update = StateUpdate::new();
    let locked = lock(owner, market, 1, 1, true);
    run(&state, &mut txn, &mut update, &locked, 1, 5).unwrap();
    let mut account = state
        .markets()
        .get_user_share_account(&txn, &owner)
        .unwrap()
        .unwrap();
    let template = account.escrows.values().next().unwrap().clone();
    for i in 1u32..1024 {
        let mut e = template.clone();
        e.escrow_id[..4].copy_from_slice(&i.to_le_bytes());
        account.escrows.insert(e.escrow_id, e);
    }
    assert_eq!(account.escrows.len(), 1024);
    account.positions.insert((market, 0), 2000);
    state
        .markets()
        .restore_share_account(&mut txn, &owner, Some(&account))
        .unwrap();
    assert!(
        run(
            &state,
            &mut txn,
            &mut update,
            &lock(owner, market, 2, 1, true),
            2,
            5
        )
        .is_err()
    );
    let assign = filled(
        NativeActionV3::AssignEscrow {
            original_owner: owner,
            escrow_id: id(&locked),
            new_claim_address: Address([4; 20]),
            new_refund_address: Address([4; 20]),
        },
        &[Address([2; 20]), Address([3; 20])],
        3,
    );
    run(&state, &mut txn, &mut update, &assign, 2, 5).unwrap();
    run(
        &state,
        &mut txn,
        &mut update,
        &resolve(id(&locked), false, 4),
        2,
        10,
    )
    .unwrap();
    let mut another = lock(owner, market, 5, 1, true);
    if let Some(TxData::NativeOperation(op)) = &mut another.transaction.data {
        if let NativeActionV3::LockShares {
            claim_before_parent,
            ..
        } = &mut op.action
        {
            *claim_before_parent = 20;
        }
    }
    run(&state, &mut txn, &mut update, &another, 2, 10).unwrap();
    assert_eq!(
        state
            .markets()
            .get_user_share_account(&txn, &owner)
            .unwrap()
            .unwrap()
            .escrows
            .len(),
        1024
    );
}

#[test]
fn database_inventory_is_exactly_upstream_and_legacy_schema_is_rejected() {
    let (env, state, dir, market, owner) = fixture();
    assert_eq!(State::NUM_DBS, 36);
    let mut txn = env.write_txn().unwrap();
    let mut update = StateUpdate::new();
    let locked = lock(owner, market, 1, 70, false);
    run(&state, &mut txn, &mut update, &locked, 1, 5).unwrap();
    run(
        &state,
        &mut txn,
        &mut update,
        &resolve(id(&locked), true, 2),
        1,
        6,
    )
    .unwrap();
    update.apply_all_changes(&state, &mut txn, 1).unwrap();
    state.native().restore(&state, &mut txn, 1).unwrap();
    txn.commit().unwrap();
    drop(state);
    drop(env);
    let mut opts = heed::EnvOpenOptions::new();
    opts.max_dbs(36).map_size(64 * 1024 * 1024);
    let raw = unsafe { opts.open(dir.path()) }.unwrap();
    let txn = raw.read_txn().unwrap();
    let main = raw
        .open_database::<heed::types::Str, heed::types::Bytes>(&txn, None)
        .unwrap()
        .unwrap();
    let actual: Vec<_> = main
        .iter(&txn)
        .unwrap()
        .map(|r| r.unwrap().0.to_owned())
        .collect();
    let expected = vec![
        "ballots",
        "block_index_events",
        "consensus_undo",
        "consolidation_undo",
        "decision_outcomes",
        "decision_state_histories",
        "deposit_blocks",
        "genesis_timestamp",
        "height",
        "latest_failed_withdrawal_bundle",
        "mainchain_timestamp",
        "market_funds_utxos",
        "markets",
        "markets_by_decision",
        "markets_by_expiry",
        "markets_by_state",
        "mempool_shares",
        "minting_undo",
        "pending_withdrawal_bundle",
        "period_decisions",
        "period_pricing",
        "period_pricing_undo",
        "period_stats",
        "reputation",
        "reputation_transfer_undo",
        "settlement_undo",
        "share_accounts",
        "skipped_tx_indices_undo",
        "state_version",
        "stxos",
        "tip",
        "utxos",
        "utxos_by_address",
        "votes",
        "withdrawal_bundle_event_blocks",
        "withdrawal_bundles",
    ];
    assert_eq!(actual, expected);
    drop(txn);
    // Write the prior upstream value layout into the EXISTING version record.
    let mut txn = raw.write_txn().unwrap();
    let version = raw
        .open_database::<sneed::UnitKey, heed::types::Bytes>(
            &txn,
            Some("state_version"),
        )
        .unwrap()
        .unwrap();
    version
        .put(
            &mut txn,
            &(),
            &bincode::serialize(&crate::types::Version {
                major: 0,
                minor: 18,
                patch: 0,
            })
            .unwrap(),
        )
        .unwrap();
    txn.commit().unwrap();
    drop(raw);
    let env = unsafe { sneed::Env::open(&opts, dir.path()) }.unwrap();
    assert!(State::new(&env, None).is_err());
}
