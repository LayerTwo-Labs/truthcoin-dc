# Truthcoin Regtest Setup

## Prerequisites

### Installation Guides

Choose the installation guide for your operating system:

| OS | Guide |
|----|-------|
| **macOS** | [macOS Installation Guide](docs/INSTALL_MACOS.md) |
| **Linux** | [Linux Installation Guide](docs/INSTALL_LINUX.md) |
| **Windows** | [Windows Installation Guide](docs/INSTALL_WINDOWS.md) |

Each guide covers:
- Installing Git, grpcurl, and Rust toolchain
- Downloading pre-built binaries
- Building electrs and truthcoin from source

## Integration Tests

Run the automated integration test suite:


The test suite requires compiled binaries at specific paths. Configure these in `integration_tests/example.env`:

```bash
BIP300301_ENFORCER='../bip300301_enforcer/bip300301_enforcer'
BITCOIND='../bitcoin-patched/bitcoind'
BITCOIN_CLI='../bitcoin-pathced/bitcoin-cli'
ELECTRS='../electrs/target/release/electrs'
TRUTHCOIN_APP='target/debug/truthcoin_dc_app'
```

```bash
TRUTHCOIN_INTEGRATION_TEST_ENV=integration_tests/example.env cargo run --example integration_tests
```

Adjust paths as needed for your local setup. All binaries must be compiled and executable before running tests.

To run a specific test:
```bash
TRUTHCOIN_INTEGRATION_TEST_ENV=integration_tests/example.env cargo run --example integration_tests -- --exact <test_name> 

roundtrip is a full coverage test of the node

```

---

## Quick Start

### 1. Create Data Directory
```bash
rm -rf /tmp/regtest-data && mkdir -p /tmp/regtest-data/{bitcoin,electrs,enforcer,truthcoin}
```

### 2. Start All Four Binaries In This Order (each in separate terminal)

**Bitcoin Core:**
```bash
../bitcoin-patched/bitcoind -acceptnonstdtxn -chain=regtest -datadir=/tmp/regtest-data/bitcoin \
-bind=127.0.0.1:18444 -rpcuser=regtest_user -rpcpassword=regtest_pass -rpcport=18443 \
-rest -server -zmqpubsequence=tcp://127.0.0.1:28332 -listenonion=0 -txindex
```

**Electrs:**
```bash
../electrs/target/release/electrs -vv --db-dir=/tmp/regtest-data/electrs \
--daemon-dir=/tmp/regtest-data/bitcoin --daemon-rpc-addr=127.0.0.1:18443 \
--electrum-rpc-addr=127.0.0.1:50001 --http-addr=127.0.0.1:3000 \
--monitoring-addr=127.0.0.1:4224 --network=regtest --cookie=regtest_user:regtest_pass --jsonrpc-import
```

**BIP300301 Enforcer:**
```bash
../bip300301_enforcer/bip300301_enforcer --data-dir=/tmp/regtest-data/enforcer \
--node-rpc-addr=127.0.0.1:18443 --node-rpc-user=regtest_user --node-rpc-pass=regtest_pass \
--enable-wallet --log-level=trace --serve-grpc-addr=127.0.0.1:50051 \
--serve-json-rpc-addr=127.0.0.1:18080 --serve-rpc-addr=127.0.0.1:18081 \
--wallet-auto-create --wallet-electrum-host=127.0.0.1 --wallet-electrum-port=50001 \
--wallet-esplora-url=http://127.0.0.1:3000 --wallet-skip-periodic-sync --enable-mempool
```

**Truthcoin App (Headless for CLI use):**
```bash
./target/debug/truthcoin_dc_app --headless --datadir=/tmp/regtest-data/truthcoin --network=regtest \
--mainchain-grpc-port=50051 --net-addr=127.0.0.1:18445 --rpc-port=18332 --zmq-addr=127.0.0.1:28333
```

**Truthcoin App (GUI):**
```bash
./target/debug/truthcoin_dc_app --datadir=/tmp/regtest-data/truthcoin --network=regtest \
--mainchain-grpc-port=50051 --net-addr=127.0.0.1:18445 --rpc-port=18332 --zmq-addr=127.0.0.1:28333
```

### 3. Activate Sidechain

**Fund enforcer and propose sidechain:**
```bash
# Generate initial blocks (funds enforcer wallet)
grpcurl -plaintext -d '{"blocks": 101}' 127.0.0.1:50051 cusf.mainchain.v1.WalletService.GenerateBlocks

# Propose sidechain (keep running - it's a stream)
grpcurl -plaintext -d '{
  "sidechain_id": 13,
  "declaration": {"v0": {"title": "Truthcoin", "description": "Truthcoin Drivechain",
    "hash_id_1": {"hex": "0000000000000000000000000000000000000000000000000000000000000000"},
    "hash_id_2": {"hex": "0000000000000000000000000000000000000000"}}}
}' 127.0.0.1:50051 cusf.mainchain.v1.WalletService.CreateSidechainProposal
```

**In another terminal, activate:**
```bash
grpcurl -plaintext -d '{"blocks": 7, "ack_all_proposals": true}' 127.0.0.1:50051 cusf.mainchain.v1.WalletService.GenerateBlocks
```

### 4. Setup Wallet and Fund Sidechain

```bash
# Create wallet
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 generate-mnemonic
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 set-seed-from-mnemonic "YOUR_MNEMONIC_HERE"
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-new-address

# Deposit BTC to sidechain (replace ADDRESS)
grpcurl -plaintext -d '{
  "sidechain_id": 13, "address": "YOUR_ADDRESS", "value_sats": 100000000, "fee_sats": 10000
}' 127.0.0.1:50051 cusf.mainchain.v1.WalletService.CreateDepositTransaction

# Mine L1 block, then L2 genesis block
grpcurl -plaintext -d '{"blocks": 1, "ack_all_proposals": true}' 127.0.0.1:50051 cusf.mainchain.v1.WalletService.GenerateBlocks
```

### Mining Helper Script

Use `./mine_blocks.sh [N]` to mine N block pairs (L1 + L2):
```bash
./mine_blocks.sh (If no [N] is specified, 1 block pair will be mined by default)
```

**It is recommended to make multiple deposits to your Truthcoin node to avoid double spending UTXO's within each block.
***If you run into UTXO double spend errors, you need to run the mining script for fresh UTXO's

---

## Creating Markets

Markets require decision slots. Slots must be claimed and confirmed before they can be used in market creation.

### 1. Claim a Slot

```bash
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 slot-claim \
--period-index 0 --slot-index 0 --is-standard true --is-scaled false \
--question "Will BTC hit $100K?" --fee-sats 1000
```

### 2. Mine Blocks to Confirm

```bash
./mine_blocks.sh 1
```

### 3. Get Slot ID for Market Creation

```bash
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 slot-list --period 0 --status claimed
```

Use the returned slot ID in your market's `--dimensions` parameter.

### 4. Create Market

```bash
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 market-create \
--title "Will BTC hit $100K?" --description "Binary prediction market" \
--dimensions "[SLOT_ID_HERE]" --beta 7.0 --fee-sats 1000
```

---

## CLI Reference

All commands: `./target/debug/truthcoin_dc_app_cli --rpc-port 18332 <COMMAND>`

### System
```
status          Node status
stop            Shutdown node
mine [--fee-sats N]         Mine sidechain block (default fee: 1000)
openapi-schema  Show API schema
```

### Wallet
```
balance                     Get BTC balance
get-new-address             Generate new address
get-wallet-addresses        Get all wallet addresses
transfer <DEST> --value-sats N [--fee-sats N]
withdraw <ADDR> --amount-sats N [--fee-sats N] [--mainchain-fee-sats N]
create-deposit <ADDR> --value-sats N [--fee-sats N]
format-deposit-address <ADDR>   Format deposit address
my-utxos                    List owned UTXOs
my-unconfirmed-utxos        List unconfirmed owned UTXOs
get-wallet-utxos            Get wallet UTXOs
list-utxos                  List all UTXOs
generate-mnemonic           Generate seed phrase
set-seed-from-mnemonic "<PHRASE>"
sidechain-wealth            Get total sidechain wealth
```

### Blockchain
```
get-block-count             Current height
get-block <HASH>            Get block by hash
get-best-mainchain-block-hash   Get best mainchain block hash
get-best-sidechain-block-hash   Get best sidechain block hash
get-bmm-inclusions <HASH>   Get mainchain BMM inclusions
get-transaction <TXID>      Get transaction
get-transaction-info <TXID> Get transaction info
pending-withdrawal-bundle   Get pending withdrawal bundle
latest-failed-withdrawal-bundle-height
remove-from-mempool <TXID>  Remove transaction from mempool
list-peers                  Connected peers
connect-peer <ADDR>         Connect to peer
```

### Cryptography
```
get-new-encryption-key      Get new encryption key
get-new-verifying-key       Get new verifying key
encrypt-msg --encryption-pubkey KEY --msg "MSG"
decrypt-msg --encryption-pubkey KEY --msg "MSG" [--utf8]
sign-arbitrary-msg --verifying-key KEY --msg "MSG"
sign-arbitrary-msg-as-addr --address ADDR --msg "MSG"
verify-signature --signature SIG --verifying-key KEY --dst DST --msg "MSG"
```

### slot_* (Decision Slots)
```
slot-status                 Slot system status
slot-list [--period N] [--status STATUS]
                            List slots (status: available, claimed, voting, ossified)
slot-get <SLOT_ID>          Get slot details
slot-claim --period-index N --slot-index N --is-standard BOOL --is-scaled BOOL \
           --question "<Q>" [--min N] [--max N] [--fee-sats N]
```

### market_* (Prediction Markets)
```
market-list                 List all markets
market-get <MARKET_ID>      Get market details
market-buy --market-id ID --outcome-index N --shares-amount N --max-cost N [--fee-sats N]
market-positions [--address ADDR] [--market-id ID]
market-create --title "T" --description "D" --dimensions "SPEC" \
              [--beta N] [--trading-fee N] [--tags "t1,t2"] [--fee-sats N]
calculate-share-cost --market-id ID --outcome-index N --shares-amount N
calculate-initial-liquidity --beta N [--market-type TYPE] [--num-outcomes N] \
                            [--decision-slots "s1,s2"] [--has-residual BOOL] [--dimensions "SPEC"]
```

### vote_* (Voting System)
```
vote-register [--reputation-bond-sats N] [--fee-sats N]
vote-voter <ADDRESS>        Get voter info
vote-voters                 List all voters
vote-submit --decision-id ID --vote-value N [--fee-sats N]
vote-submit --votes "id1:val1,id2:val2" [--fee-sats N]   # Batch voting
vote-list [--voter ADDR] [--decision-id ID] [--period-id N]
vote-period [--period-id N]   # Omit for current period
```

### votecoin_*
```
votecoin-transfer <DEST> --amount N [--fee-sats N]
votecoin-balance <ADDRESS>
```

---

## Market Dimensions Specification

The `--dimensions` parameter uses bracket notation to define market structures:

### Single Binary (Yes/No)
```bash
# Single decision: Will X happen?
market-create --title "Will BTC hit $100K?" --description "..." \
--dimensions "[abc123]" --beta 7.0 --fee-sats 1000
# Outcomes: Yes, No (2 outcomes)
```

### Multiple Independent Binary
```bash
# Independent decisions that can all be true/false independently
market-create --title "2024 Predictions" --description "..." \
--dimensions "[abc123,def456]" --beta 7.0 --fee-sats 1000
# Outcomes: 4 combinations (Yes-Yes, Yes-No, No-Yes, No-No)
```

### Categorical (Mutually Exclusive)
```bash
# One of N outcomes (double brackets)
market-create --title "Who wins election?" --description "..." \
--dimensions "[[abc123,def456,ghi789]]" --beta 7.0 --fee-sats 1000
# Outcomes: Candidate A wins, Candidate B wins, Candidate C wins (mutually exclusive)
```

### Mixed Dimensional
```bash
# Combine independent and categorical
market-create --title "Sports + Weather" --description "..." \
--dimensions "[abc123,[def456,ghi789]]" --beta 7.0 --fee-sats 1000
# Outcomes: 2 × 2 = 4 (independent binary × categorical)
```

### Examples by Use Case

**Election Market (Categorical):**
```bash
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 market-create \
--title "2024 Presidential Election" \
--description "Who will win the 2024 US Presidential Election?" \
--dimensions "[[slot_dem,slot_rep,slot_other]]" \
--beta 10.0 --fee-sats 1000
```

**Sports Parlay (Multiple Independent):**
```bash
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 market-create \
--title "Weekend Games Parlay" \
--description "Outcomes for multiple independent games" \
--dimensions "[game1_slot,game2_slot,game3_slot]" \
--beta 5.0 --fee-sats 1000
```

**Simple Yes/No:**
```bash
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 market-create \
--title "Will it rain tomorrow?" \
--description "Binary prediction on tomorrow's weather" \
--dimensions "[weather_slot]" \
--beta 7.0 --fee-sats 1000
```

---

## JSON-RPC

```bash
curl -X POST http://127.0.0.1:18332 -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"METHOD","params":PARAMS,"id":1}'
```

Methods mirror CLI: `mine`, `bitcoin_balance`, `get_new_address`, `slot_list`, `slot_get`, `slot_claim`, `market_list`, `market_get`, `market_buy`, `market_create`, `vote_register`, `vote_submit`, `vote_period`, `vote_voter`, `votecoin_balance`

---

## Port Reference

| Service | Port | Purpose |
|---------|------|---------|
| Bitcoin Core RPC | 18443 | Mainchain RPC |
| Electrs HTTP | 3000 | Block explorer |
| Enforcer gRPC | 50051 | Sidechain operations |
| Truthcoin RPC | 18332 | Sidechain RPC |

## Cleanup

```bash
rm -rf /tmp/regtest-data
```
