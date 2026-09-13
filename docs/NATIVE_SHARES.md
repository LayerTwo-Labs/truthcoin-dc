# Native share delivery and hash/time escrow v1

This release adds general Truthcoin operations. No transaction, database, or verifier knows another chain. Truthcoin's existing header encoding is unchanged. `TransactionData::NativeOperation` is appended to the existing enum; old transaction discriminants and signature domains remain unchanged.

This is a coordinated consensus fork, currently configured to activate at sidechain height zero for fresh deployments. Do not install it on an existing network without an agreed deployment height and replay/migration procedure. The profile includes the existing ECX Truthcoin rollback, finite-abstention, withdrawal-address, and exact account metadata corrections. It adds five named databases: `native_reservations`, `native_escrows`, `native_effects`, `native_nonces`, and `native_undo` (42 total state databases). No live network has been activated by this implementation.

## Operations

- `BuyForIntent`: a sponsor funds an ordinary LMSR buy of an exact positive number of shares for a recipient. The recipient signs the chain genesis hash, market, outcome, shares, reference, fill-once nonce, and parent-height window. The domain is `Dst::NativeIntent` plus `TRUTHCOIN_BUY_INTENT_V1\0` and the sidechain number. The sponsor's signed transaction fixes its spending limit and cash change address. A slippage skip consumes no input, nonce, or native receipt.
- `TransferShares`: the owner signs an exact quantity and fixed recipient, reference, nonce, and validity window. Existing UTXO authorization or actor proof authenticates the owner. Fee inputs may be sponsored separately.
- `LockShares`: owner-authorized reservation of existing, unreserved shares. The lock fixes the claim address, refund address, SHA256 hash of a 32-byte secret, claim deadline, reference, and nonce. Its ID is BLAKE3 of the tagged transaction ID. Reservations block ordinary sells, transfers, and overlapping locks.
- `ClaimEscrow`: any relayer can submit the 32-byte preimage. Execution sends the shares or native cash successor to the immutable claim address.
- `RefundEscrow`: any relayer can submit a refund after expiry. Execution sends the shares or native cash successor to the immutable refund address.

The consensus parent height is the height of the Bitcoin block containing the successful BMM commitment: the committed header's `prev_main_hash` height plus one. Buy and transfer validity are `[valid_from_parent, valid_before_parent)`. Lock creation and claims require height strictly below `claim_before_parent`; refunds require height greater than or equal to it. A claim and refund cannot both execute. Mempool presence, a revealed secret, an expired timer, and candidate inclusion are not executed effects.

## Settlement and rollback

Reserved shares remain part of the owner's account and market share totals. When the market settles, its existing, rounded owner/outcome payout is divided among free shares and locks by integer largest remainder, with deterministic ID tie-breaking. The exact original payout is conserved. A lock becomes `NativeCash(amount)`, including zero, while keeping every condition and destination. No cash UTXO is spendable until a terminal operation executes.

Native effects contain transaction ID, sidechain and parent heights, reference, operation kind, escrow ID when applicable, owner, recipient, market/outcome, quantity, and economic asset. Only successful execution records an effect or consumes a nonce. Terminal records remain available. Per-height undo restores exact escrow state, reservation index, nonce and effect membership, synthetic cash outputs, and share-account metadata. Tests cover connecting/disconnecting/replaying real state blocks, sell and overlap rejection, deadline boundaries, invalid secrets, settlement, zero successors, and cash refund rollback.

## Wallet and RPC

`sign_native_buy_intent` signs a recipient-owned intent. `create_native_operation` builds, funds, and authorizes a transaction and returns the existing signed-transaction hex response; ordinary submission broadcasts it. `get_native_escrow`, `get_native_effect`, and `get_native_reserved_shares` expose node-local state for inspection. IDs in getter parameters are exact 32-byte hex strings. `lib/types/native.rs` defines the JSON and consensus field names.

`create_bmm_candidate` accepts a list of signed transaction hex strings and a coinbase address, builds a candidate against the current parent, and runs the complete local block transition in an aborted write transaction. Required native operations must actually execute in this preview. `bid_bmm_candidate` revalidates the complete header/body before paying a bid, checks the parent has not moved, and accepts only a winning commitment in that parent's immediate child. A lost or stale attempt must be rebuilt for the fresh parent; the returned preview is never evidence of canonical execution. Raw parent timestamp and subsequent chain changes can still affect actual execution, which must be checked from canonical state.

A consumer on another system must authenticate the complete native execution and canonical history independently. The native RPC response and candidate preview are not cryptographic proof. Claim/refund status is provisional under reorganization until the consumer's chosen canonical finality rule is met.
