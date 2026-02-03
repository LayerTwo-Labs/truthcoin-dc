# Linux Installation Guide

## Prerequisites

### 1. Install Git

**Debian/Ubuntu:**
```bash
sudo apt install git
```

**Fedora:**
```bash
sudo dnf install git
```

**Arch Linux:**
```bash
sudo pacman -S git
```

Verify installation:
```bash
git --version
```

### 2. Install grpcurl

grpcurl is required for interacting with the BIP300301 enforcer gRPC service.

```bash
# Download latest release (adjust version as needed)
curl -sSL https://github.com/fullstorydev/grpcurl/releases/download/v1.9.1/grpcurl_1.9.1_linux_x86_64.tar.gz | tar -xz
sudo mv grpcurl /usr/local/bin/
```

Verify installation:
```bash
grpcurl --version
```

### 3. Install Rust, Cargo, and Rustup

Rust and Cargo are required to build truthcoin and electrs from source.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

Install nightly toolchain (required):
```bash
rustup install nightly
rustup default nightly
```

Verify installation:
```bash
rustc --version
cargo --version
```

## Download and Setup Binaries

### 1. Create Directory Structure

```bash
mkdir truthcoin-binaries
cd truthcoin-binaries
```

### 2. Download Pre-built Binaries

Download pre-built binaries from [releases.drivechain.info](https://releases.drivechain.info):
- `L1-bitcoin-patched-latest-x86_64-unknown-linux-gnu.zip`
- `bip300301-enforcer-latest-x86_64-unknown-linux-gnu.zip`

### 3. Rename and Organize

```bash
mv ~/Downloads/L1-bitcoin-patched-latest-x86_64-unknown-linux-gnu ./bitcoin-patched
mv ~/Downloads/bip300301-enforcer-latest-x86_64-unknown-linux-gnu ./bip300301_enforcer
```

Rename the enforcer binary:
```bash
mv ./bip300301_enforcer/bip300301-enforcer-latest-x86_64-unknown-linux-gnu ./bip300301_enforcer/bip300301_enforcer
```

### 4. Make Binaries Executable

```bash
chmod +x ./bip300301_enforcer/bip300301_enforcer
chmod +x ./bitcoin-patched/bitcoind
chmod +x ./bitcoin-patched/bitcoin-cli
```

### 5. Build from Source

> **Note:** If building Bitcoin from source, install ZMQ first or you'll get `MethodNotFound` errors for `getzmqnotifications`:
> ```bash
> sudo apt install libzmq3-dev  # Debian/Ubuntu
> sudo dnf install zeromq-devel  # Fedora
> ```

```bash
# Electrs (Blockstream fork with HTTP/REST API)
git clone https://github.com/blockstream/electrs.git
cd electrs
cargo build --release
cd ..

# Truthcoin
git clone https://github.com/LayerTwo-Labs/truthcoin-dc.git
cd truthcoin-dc
git submodule update --init --recursive
cargo build
```

---

## Integration Tests

Run the automated integration test suite to verify your setup.

### Configure Test Environment

Create or edit `integration_tests/example.env` with paths to your binaries:

```bash
BIP300301_ENFORCER='../bip300301_enforcer/bip300301_enforcer'
BITCOIND='../bitcoin-patched/bitcoind'
BITCOIN_CLI='../bitcoin-patched/bitcoin-cli'
ELECTRS='../electrs/target/release/electrs'
TRUTHCOIN_APP='./target/debug/truthcoin_dc_app'
```

Adjust paths as needed for your local setup. All binaries must be compiled and executable.

### Run All Tests

```bash
TRUTHCOIN_INTEGRATION_TEST_ENV=integration_tests/example.env cargo run --example integration_tests
```

### Run a Specific Test

```bash
TRUTHCOIN_INTEGRATION_TEST_ENV=integration_tests/example.env cargo run --example integration_tests -- --exact <test_name>
```

The `roundtrip` test provides full coverage of node functionality.

---

## Regtest Quick Start

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
  "sidechain_id": 13, "address": "449UgDABnAFtFapBjLRMzp2wRmJP", "value_sats": 100000000, "fee_sats": 10000
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

## Cleanup

```bash
rm -rf /tmp/regtest-data
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
              [--beta N] [--trading-fee N] [--tags "t1,t2"] \
              [--category-txids "txid1,txid2"] [--residual-names "name1,name2"] \
              [--fee-sats N]
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

## Decision Types and Market Examples

Markets are built from decision slots. Each slot can be **binary** (Yes/No) or **scalar** (numeric range).

### Decision Slot Types

**Binary Decision** (is_scaled=false): Resolves to Yes (1.0) or No (0.0)
```bash
# Claim a binary slot
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 slot-claim \
--period-index 0 --slot-index 0 --is-standard true --is-scaled false \
--question "Will BTC exceed $100K by end of 2025?" --fee-sats 1000
```

**Scalar Decision** (is_scaled=true): Resolves to a value within min/max range
```bash
# Claim a scalar slot (e.g., electoral votes 0-538)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 slot-claim \
--period-index 0 --slot-index 1 --is-standard true --is-scaled true \
--question "How many electoral votes will the Republican candidate receive?" \
--min 0 --max 538 --fee-sats 1000
```

---

## Market Dimensions Specification

The `--dimensions` parameter uses bracket notation to define market structures.

### 1. Single Binary Market (Yes/No)

Simple Yes/No prediction using one binary slot.

```bash
# Claim the slot
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 slot-claim \
--period-index 0 --slot-index 0 --is-standard true --is-scaled false \
--question "Will BTC hit $100K?" --fee-sats 1000

# Mine to confirm
./mine_blocks.sh 1

# Get the slot ID
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 slot-list --period 0 --status claimed
# Returns: slot_id = "000000"

# Create market
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 market-create \
--title "Will BTC hit $100K?" \
--description "Resolves Yes if Bitcoin reaches $100,000 USD" \
--dimensions "[000000]" --beta 7.0 --fee-sats 1000

# Outcomes: Yes, No (2 outcomes)
```

### 2. Single Scalar Market (Numeric Range)

Predict a numeric value within a defined range.

```bash
# Claim a scalar slot
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 slot-claim \
--period-index 0 --slot-index 1 --is-standard true --is-scaled true \
--question "BTC price on Dec 31, 2025 (in thousands USD)" \
--min 0 --max 500 --fee-sats 1000

# Mine and get slot ID "000001"

# Create market
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 market-create \
--title "BTC Year-End Price Prediction" \
--description "What will BTC price be on Dec 31, 2025? (in $1000s)" \
--dimensions "[000001]" --beta 7.0 --fee-sats 1000

# Outcomes: Under, Over (2 outcomes based on scalar midpoint)
```

### 3. Multiple Independent Binary (Parlay)

Combine independent Yes/No decisions. All combinations are tradeable.

```bash
# Claim multiple binary slots
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 slot-claim \
--period-index 0 --slot-index 2 --is-standard true --is-scaled false \
--question "Will ETH hit $10K?" --fee-sats 1000

./target/debug/truthcoin_dc_app_cli --rpc-port 18332 slot-claim \
--period-index 0 --slot-index 3 --is-standard true --is-scaled false \
--question "Will SOL hit $500?" --fee-sats 1000

# Mine and get slot IDs "000002" and "000003"

# Create parlay market
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 market-create \
--title "Crypto Price Parlay" \
--description "Combined predictions on ETH and SOL prices" \
--dimensions "[000002,000003]" --beta 7.0 --fee-sats 1000

# Outcomes (4 total):
#   0: ETH-Yes + SOL-Yes
#   1: ETH-Yes + SOL-No
#   2: ETH-No + SOL-Yes
#   3: ETH-No + SOL-No
```

### 4. Categorical Market (Mutually Exclusive)

One winner from multiple options. Use double brackets `[[...]]` and claim slots as a category.

```bash
# Claim category slots (atomic transaction)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 slot-claim-category \
--slot-ids "000004,000005,000006" \
--questions "Will Candidate A win?,Will Candidate B win?,Will Candidate C win?" \
--is-standard true --fee-sats 1000

# Mine and note the category txid returned (e.g., "abc123def456...")
./mine_blocks.sh 1

# Create categorical market with --category-txids referencing the claim transaction
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 market-create \
--title "2028 Presidential Election Winner" \
--description "Which candidate will win the 2028 US Presidential Election?" \
--dimensions "[[000004,000005,000006]]" \
--category-txids "abc123def456..." \
--beta 7.0 --fee-sats 1000

# Outcomes (3 mutually exclusive):
#   0: Candidate A wins
#   1: Candidate B wins
#   2: Candidate C wins
```

### 4b. Categorical Market with Residual Outcome

Add a "catch-all" residual outcome for options not explicitly listed.

```bash
# Claim category slots for known candidates
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 slot-claim-category \
--slot-ids "000004,000005,000006" \
--questions "Will Candidate A win?,Will Candidate B win?,Will Candidate C win?" \
--is-standard true --fee-sats 1000

# Mine and note the category txid
./mine_blocks.sh 1

# Create market with residual outcome for "Other candidate"
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 market-create \
--title "2028 Presidential Election Winner" \
--description "Which candidate will win the 2028 US Presidential Election?" \
--dimensions "[[000004,000005,000006]]" \
--category-txids "abc123def456..." \
--residual-names "Other" \
--beta 7.0 --fee-sats 1000

# Outcomes (4 total: 3 explicit + 1 residual):
#   0: Candidate A wins
#   1: Candidate B wins
#   2: Candidate C wins
#   3: Other (residual)
```

### 5. Binary + Categorical Combination

Combine an independent binary decision with a categorical group.

```bash
# Binary slot: "Will there be a recession?"
# Slot ID: "000007"

# Claim category slots for sector leadership
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 slot-claim-category \
--slot-ids "000008,000009,00000a" \
--questions "Tech sector leads?,Finance sector leads?,Energy sector leads?" \
--is-standard true --fee-sats 1000
# Returns category txid: "sector123..."

./mine_blocks.sh 1

# Create combined market with --category-txids
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 market-create \
--title "Economy & Sector Performance" \
--description "Recession prediction combined with leading sector" \
--dimensions "[000007,[000008,000009,00000a]]" \
--category-txids "sector123..." \
--beta 7.0 --fee-sats 1000

# Outcomes (2 x 3 = 6 total):
#   0: Recession-Yes + Tech leads
#   1: Recession-Yes + Finance leads
#   2: Recession-Yes + Energy leads
#   3: Recession-No + Tech leads
#   4: Recession-No + Finance leads
#   5: Recession-No + Energy leads
```

### 6. Multiple Binary (3+ Independent)

Three or more independent binary decisions.

```bash
# Slots for: "Team A wins?", "Team B wins?", "Team C wins?"
# Slot IDs: "00000b", "00000c", "00000d"

./target/debug/truthcoin_dc_app_cli --rpc-port 18332 market-create \
--title "Weekend Sports Parlay" \
--description "Independent game outcomes" \
--dimensions "[00000b,00000c,00000d]" --beta 5.0 --fee-sats 1000

# Outcomes (2^3 = 8 total combinations)
```

### 7. Two Categorical Groups

Combine two separate categorical decisions.

```bash
# Claim Category 1: Conference winner (East, West)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 slot-claim-category \
--slot-ids "00000e,00000f" \
--questions "East wins conference?,West wins conference?" \
--is-standard true --fee-sats 1000
# Returns: "conf123..."

# Claim Category 2: MVP winner (Player A, Player B, Player C)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 slot-claim-category \
--slot-ids "000010,000011,000012" \
--questions "Player A MVP?,Player B MVP?,Player C MVP?" \
--is-standard true --fee-sats 1000
# Returns: "mvp456..."

./mine_blocks.sh 1

# Create market with both category txids (comma-separated, in dimension order)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 market-create \
--title "NBA Season Predictions" \
--description "Conference champion and MVP predictions" \
--dimensions "[[00000e,00000f],[000010,000011,000012]]" \
--category-txids "conf123...,mvp456..." \
--beta 7.0 --fee-sats 1000

# Outcomes (2 x 3 = 6 total):
#   0: East wins + Player A MVP
#   1: East wins + Player B MVP
#   2: East wins + Player C MVP
#   3: West wins + Player A MVP
#   4: West wins + Player B MVP
#   5: West wins + Player C MVP
```

### 8. Scalar + Binary Combination

Combine a numeric prediction with a Yes/No decision.

```bash
# Scalar slot: "What will BTC dominance % be?" (min=30, max=80)
# Slot ID: "000013"

# Binary slot: "Will ETH flip BTC in market cap?"
# Slot ID: "000014"

./target/debug/truthcoin_dc_app_cli --rpc-port 18332 market-create \
--title "BTC Dominance & Flippening" \
--description "BTC dominance level and ETH flippening prediction" \
--dimensions "[000013,000014]" --beta 7.0 --fee-sats 1000

# Outcomes (2 x 2 = 4 total):
#   0: BTC dominance Under + Flippening Yes
#   1: BTC dominance Under + Flippening No
#   2: BTC dominance Over + Flippening Yes
#   3: BTC dominance Over + Flippening No
```

### 9. Two Scalar Decisions

Combine two numeric predictions.

```bash
# Scalar slot 1: "Fed funds rate end of year" (min=0, max=10)
# Slot ID: "000015"

# Scalar slot 2: "Inflation rate %" (min=0, max=15)
# Slot ID: "000016"

./target/debug/truthcoin_dc_app_cli --rpc-port 18332 market-create \
--title "Fed Rate vs Inflation" \
--description "Interest rate and inflation predictions" \
--dimensions "[000015,000016]" --beta 7.0 --fee-sats 1000

# Outcomes (2 x 2 = 4 total):
#   0: Rate Under + Inflation Under
#   1: Rate Under + Inflation Over
#   2: Rate Over + Inflation Under
#   3: Rate Over + Inflation Over
```

### 10. Scalar + Categorical Combination

Combine a numeric prediction with mutually exclusive options.

```bash
# Scalar slot: "S&P 500 year-end value" (min=3000, max=6000)
# Slot ID: "000017"

# Claim category slots for sector leadership
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 slot-claim-category \
--slot-ids "000018,000019,00001a" \
--questions "Tech leads?,Finance leads?,Healthcare leads?" \
--is-standard true --fee-sats 1000
# Returns: "sector789..."

./mine_blocks.sh 1

./target/debug/truthcoin_dc_app_cli --rpc-port 18332 market-create \
--title "Market Performance & Sector Leader" \
--description "S&P 500 level combined with leading sector" \
--dimensions "[000017,[000018,000019,00001a]]" \
--category-txids "sector789..." \
--beta 7.0 --fee-sats 1000

# Outcomes (2 x 3 = 6 total):
#   0: S&P Under + Tech leads
#   1: S&P Under + Finance leads
#   2: S&P Under + Healthcare leads
#   3: S&P Over + Tech leads
#   4: S&P Over + Finance leads
#   5: S&P Over + Healthcare leads
```

### 11. Complex: Scalar + Binary + Categorical

Three different decision types combined.

```bash
# Scalar slot: "BTC year-end price ($K)" (min=20, max=200)
# Slot ID: "00001b"

# Binary slot: "Will there be a spot BTC ETF approval?"
# Slot ID: "00001c"

# Claim category slots for market regime
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 slot-claim-category \
--slot-ids "00001d,00001e,00001f" \
--questions "Bull market?,Bear market?,Sideways?" \
--is-standard true --fee-sats 1000
# Returns: "regime123..."

./mine_blocks.sh 1

./target/debug/truthcoin_dc_app_cli --rpc-port 18332 market-create \
--title "Crypto Market Comprehensive" \
--description "BTC price, ETF approval, and market regime" \
--dimensions "[00001b,00001c,[00001d,00001e,00001f]]" \
--category-txids "regime123..." \
--beta 10.0 --fee-sats 1000

# Outcomes (2 x 2 x 3 = 12 total):
#   BTC price (Under/Over) x ETF (Yes/No) x Market regime (Bull/Bear/Sideways)
```

### 12. Multiple Scalars with Binary

Three or more scalars with a binary decision.

```bash
# Scalar slots for economic indicators:
# "GDP growth %" (min=-5, max=10) - Slot ID: "000020"
# "Unemployment %" (min=0, max=15) - Slot ID: "000021"
# "CPI inflation %" (min=-2, max=15) - Slot ID: "000022"

# Binary slot: "Will there be a recession?"
# Slot ID: "000023"

./target/debug/truthcoin_dc_app_cli --rpc-port 18332 market-create \
--title "Economic Indicators Bundle" \
--description "GDP, unemployment, inflation, and recession prediction" \
--dimensions "[000020,000021,000022,000023]" --beta 10.0 --fee-sats 1000

# Outcomes (2^4 = 16 total combinations)
```

### Dimensions Syntax Summary

| Pattern | Type | Outcomes | Example |
|---------|------|----------|---------|
| `[A]` | Single binary | 2 | Yes/No |
| `[A]` | Single scalar | 2 | Under/Over |
| `[A,B]` | 2 independent (any type) | 4 | All combinations |
| `[A,B,C]` | 3 independent (any type) | 8 | All combinations |
| `[[A,B,C]]` | Categorical (3 options) | 3 | Mutually exclusive |
| `[A,[B,C,D]]` | Independent + Categorical | 6 | 2 x 3 |
| `[[A,B],[C,D,E]]` | Two categoricals | 6 | 2 x 3 |
| `[A,B,[C,D]]` | 2 Independent + Categorical | 8 | 2 x 2 x 2 |
| `[A,[B,C],[D,E,F]]` | Independent + 2 Categoricals | 12 | 2 x 2 x 3 |

**Note:** Independent slots (binary or scalar) use single brackets `[A,B]`. Categorical groups use double brackets `[[A,B,C]]`. Mix them freely within the outer brackets.

---

## Port Reference

| Service | Port | Purpose |
|---------|------|---------|
| Bitcoin Core RPC | 18443 | Mainchain RPC |
| Electrs HTTP | 3000 | Block explorer |
| Enforcer gRPC | 50051 | Sidechain operations |
| Truthcoin RPC | 18332 | Sidechain RPC |

---

[Back to main README](../README.md)
