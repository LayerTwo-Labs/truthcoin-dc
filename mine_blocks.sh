#!/bin/bash
BLOCKS="${1:-1}"
pkill -f "truthcoin_dc_app_cli.*mine" 2>/dev/null || true
sleep 1.0
INITIAL_BLOCKS=$(./target/debug/truthcoin_dc_app_cli --rpc-port 18332 status | grep "Block Count:" | awk '{print $3}')
for ((i=1; i<=BLOCKS; i++)); do
    echo "Mining block pair $i/$BLOCKS..."
    ./target/debug/truthcoin_dc_app_cli --rpc-port 18332 mine > /dev/null &
    sleep 1.5
    grpcurl -plaintext -d '{"blocks": 1, "ack_all_proposals": true}' 127.0.0.1:50051 cusf.mainchain.v1.WalletService.GenerateBlocks > /dev/null
    sleep 0.5
    echo "  ✓ Block pair $i mined"
done
FINAL_BLOCKS=$(./target/debug/truthcoin_dc_app_cli --rpc-port 18332 status | grep "Block Count:" | awk '{print $3}')
MINED_BLOCKS=$((FINAL_BLOCKS - INITIAL_BLOCKS))
echo "All $((MINED_BLOCKS * 2)) blocks mined successfully (L1: $MINED_BLOCKS, L2: $MINED_BLOCKS)"
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 status