# Executed-market commitments

`--network=oracletest` starts a separate fresh-chain profile. It has no seed
peers, distinct peer magic (`85189505`), an `oracletest/` data subdirectory,
and a persisted profile check. Existing networks and their activation rules
are unchanged. Use a separate local test parent/enforcer, not the live slot.

The genesis coinbase carries the empty oracle commitment, separating its
identity from legacy genesis blocks. From height 1, every coinbase memo must
equal the commitment to the **previous completed block's executed state**.
Validation and both connection paths recompute it before market mutations.
New state must also be priceable before connection succeeds. Existing undo
restores the source state on disconnect. Templates only extend the currently
executed tip; an ancestor template is refused until state has rolled back.

## Encoding

The memo is Borsh `(b"TCORCL01", Commitment)`: previous height `Option<u32>`,
previous sidechain hash `Option<[u8;32]>`, market count `u32`, root `[u8;32]`.
It is 82 bytes after genesis, 46 bytes at genesis. Integers are little-endian;
hashes are raw digest bytes. No extra memo suffix is accepted.

All stored markets are sorted by their six-byte ID, regardless of status or
activity. Each leaf contains the ID, terms hash, native status byte, last
update height, author-funded liquidity in sats, and the complete outcome-price
vector in native `get_valid_state_combos` (tradeable outcome) order. Prices use the existing libm-based LMSR
calculation and are floored to integer units of 10^-12. Settled markets use
their final payout vector instead. Cancelled/invalid marks are **not** payable
claims. Invalid/non-finite values fail closed. Mempool projections are excluded;
slippage-skipped trades have no state effect and therefore no price effect.

The terms hash binds title, description, creator, dimension/decision IDs,
explicit u32 outcome mappings, creation/expiry heights, and ordered decision
entry hashes. Decision entries use their native field order, including the
claiming transaction and optional definition. Hashes are BLAKE3 of the Borsh
tuples in `types/oracle.rs` and `lib/state/oracle.rs`, with distinct leaf,
branch, terms and decision domain tags. The tree reuses `merkle-cbt`; an empty
tree has its default zero root. Count is bound separately in the memo.

## Consumers

`get_oracle_price_proof(market_id)` returns a current-state leaf, CBMT heap
index, sibling hashes and commitment. Retain this proof and wait for the next
block to commit its exact memo. It is not a historical-proof endpoint. `null`
is not an authenticated absence proof.

`PriceProof::verify` checks only Merkle membership. A consumer must separately
authenticate the **entire memo** through the native coinbase/body/header
commitments and canonical parent BMM inclusion, pin the fresh-chain identity
and rules, and handle finality, reorgs, freshness, market terms and status.
Legacy Truthcoin miners may write arbitrary memos; a legacy-chain memo is not
a consensus oracle. BMM proves an endorsement, not execution independently of
the selected Truthcoin consensus rules. This PR does not add exchange pricing
policy, BIP300 history arbitration, or an end-to-end SPV verifier.

Snapshot construction currently scans all markets/outcomes. Header size and
individual proof size stay bounded/logarithmic, but validation work is not
incremental. This isolated profile needs scale and cross-platform qualification
before any production activation.

## Tests

```sh
cargo test --locked -p truthcoin_dc --lib
cargo test --locked -p truthcoin_dc_types --lib
cargo test --locked -p truthcoin_dc_app --bin truthcoin_dc_app cli::tests
```
