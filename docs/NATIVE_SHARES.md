# Minimal native share swaps (v3)

Truthcoin enforces native share ownership and conditional delivery. It does not know about Elements, ECX, foreign proofs, collateral pools, auctions or profit locks. Existing header bytes and transaction variants are unchanged; `TransactionData::NativeOperation(NativeOperationV3)` is appended. This is a consensus change for a coordinated fresh deployment, not live activation. The current activation constant is height zero; an existing network requires a reviewed activation/migration procedure.

## Transactions

The envelope commits the native genesis hash, an accepting-parent-height interval `[valid_from_parent, valid_before_parent)`, an opaque reference, and one action. The accepting parent height is the committed header's `prev_main_hash` height plus one. Existing full-transaction FROST/Ristretto signatures cover the entire envelope, inputs, outputs and fees. Every operation consumes an ordinary signed input; actor-only proofs are insufficient. Consumed UTXOs provide replay protection, so there is no native nonce table or separate signing domain.

| Action | Authorization and behavior |
|---|---|
| `MoveShares` | An ordinary input owned by the share owner authorizes movement of a positive, unreserved quantity to a fixed recipient. |
| `LockShares` | An owner-controlled ordinary input reserves shares under SHA-256 claim conditions, a claim/refund deadline and fixed destinations. `mutable_rights` defaults to false and cannot change. Lock ID hashes the native genesis and transaction ID. |
| `ResolveEscrow` | Any fee-paying relayer may claim with the exact preimage before the deadline, or refund at/after it. Delivery always goes to the escrow's current destination, never to the relayer. Terminal escrows cannot resolve twice. |
| `AssignEscrow` | A live mutable escrow requires ordinary inputs belonging to both distinct current claim/refund owners; one input suffices if they are the same owner. The transaction replaces both destinations and the reference. Asset, quantity, original owner, deadline and hashlock remain fixed. |

Assignment checks current owners in transaction order, including earlier assignments in the same block. An expired claim deadline does not erase the refund right; a live escrow may still be assigned within the assignment's signed validity interval. Sealed escrows reject assignment even with both owners' signatures. Invalid native operations fail the block transition rather than soft-skip. Existing withdrawal-output spending prohibitions remain in force.

There is no escrow subdivision or sponsored-buy operation. Suppliers acquire inventory through the existing ordinary market buy and then use `MoveShares`. An external system must authorize the recipient and reserve supplier consideration before delivery. It pays only after authenticated successful delivery; buying inventory or including a soft-skipped buy is insufficient. Separate locks represent separate quantities.

## State, settlement and rollback

No databases are added: the state database inventory remains the upstream 36 named databases. Each existing `ShareAccount` contains a map of live escrows. Reservations are derived from that map, bounded to 1,024 live escrows per original owner across all markets and assets. Sells and moves cannot consume reserved quantity. Market share totals remain unchanged by movement or reservation.

`ResolveEscrow` and `AssignEscrow` carry `(original_owner, escrow_id)` for direct lookup. The locator does not grant authorization or change during assignment. Terminal resolution removes the live record; absence alone proves neither a claim nor a refund. An account is removed only when both positions and escrows are empty.

On settlement, the existing rounded owner/outcome payout is apportioned among free shares and reservations with deterministic largest-remainder rounding. Total cash is conserved exactly. Each escrow becomes a conditional cash successor, including zero, with unchanged hashlock, deadline and destinations. Only terminal resolution creates spendable cash. Accounts retain conditional cash even after their last share position disappears. The existing `consolidation_undo` record carries first-touch account/market snapshots, created native cash outputs and the previous mainchain timestamp, including blocks without treasury consolidation. Disconnect restores snapshots once; it does not additionally reverse share deltas. Existing settlement and decision undo remain in use. No native reservation, receipt, nonce, escrow, market-transition or timestamp-undo database is introduced.

## Interfaces and external proofs

Build and sign the ordinary `AuthorizedTransaction` using upstream signing APIs, then submit through the existing `push_tx` path. Each required owner needs a spendable native input and signs the final full transaction; input or fee changes require new signatures. No new wallet/RPC signing endpoints or BMM APIs are introduced. Nodes expose canonical transactions through existing block interfaces; consumers may maintain an off-chain escrow index.

An external verifier must replay canonical native execution to derive delivery/assignment facts from successful transitions. Facts bind genesis, transaction and block identities, executed height, reference, quantities, assets, and historical prior/new destinations. RPC responses or transaction inclusion alone are not execution proofs. Receipt caching is external and non-authoritative. The external system must consume both a logical delivery and its obligation once, validate the authorized sender, and enforce canonical finality. Timeout/default requires complete history through the contractual horizon; missing data or a halted node is not absence evidence.

V3 changes transaction and account/undo encodings and is incompatible with earlier experimental profiles. The existing `state_version` value records native schema 3 and rejects prior layouts instead of silently defaulting fields. Existing data needs a reviewed migration or a rebuild from compatible history/fresh deployment; this PR supplies no automatic migration. Version and pin the source profile. ECX's delivery adapter must accept executed moves, derive receipts during replay, and use the v3 native authorization relation and owner-addressed escrow handles. Independent replay fixtures and proof artifacts must be regenerated before production. This native PR does not update or activate those external components.
