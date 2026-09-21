# Minimal native share swaps (v2)

Truthcoin enforces native share ownership and conditional delivery. It does not know about Elements, ECX, foreign proofs, collateral pools, auctions or profit locks. Existing header bytes and transaction variants are unchanged; `TransactionData::NativeOperation(NativeOperationV2)` is appended. This is a consensus change for a coordinated fresh deployment, not live activation. The current activation constant is height zero; an existing network requires a reviewed activation/migration procedure.

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

Three native databases hold escrows, reservations and undo. There is no native effect/receipt database. The reservation index is bounded to 1,024 live share escrows per owner/market/outcome; sells and moves cannot consume reserved quantity. Market share totals remain unchanged by movement or reservation.

On settlement, the existing rounded owner/outcome payout is apportioned among free shares and reservations with deterministic largest-remainder rounding. Total cash is conserved exactly. Each escrow becomes a conditional cash successor, including zero, with unchanged hashlock, deadline and destinations. Only terminal resolution creates spendable cash. Native undo preserves prior escrow/account records, derived reservations and created cash outputs; retained market/decision/timestamp undo makes disconnect/reconnect exact. Additional general rollback databases are distinct from the three native tables.

## Interfaces and external proofs

Build and sign the ordinary `AuthorizedTransaction` using upstream signing APIs, then submit through the existing `push_tx` path. Each required owner needs a spendable native input and signs the final full transaction; input or fee changes require new signatures. No new wallet/RPC signing endpoints or BMM APIs are introduced. Nodes expose canonical transactions through existing block interfaces; consumers may maintain an off-chain escrow index.

An external verifier must replay canonical native execution to derive delivery/assignment facts from successful transitions. Facts bind genesis, transaction and block identities, executed height, reference, quantities, assets, and historical prior/new destinations. RPC responses or transaction inclusion alone are not execution proofs. Receipt caching is external and non-authoritative. The external system must consume both a logical delivery and its obligation once, validate the authorized sender, and enforce canonical finality. Timeout/default requires complete history through the contractual horizon; missing data or a halted node is not absence evidence.

V2 is incompatible with prior experimental v1 signed intents, receipt databases and replay adapters. Do not open old experimental databases as v2: rebuild from a compatible history/fresh deployment, and version/pin the source profile. ECX's delivery adapter must accept executed moves, derive receipts during replay, and use the v2 native authorization relation. Independent replay fixtures and proof artifacts must be regenerated before production. This native PR does not update or activate those external components.
