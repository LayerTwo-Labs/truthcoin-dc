#!/usr/bin/env bash
# Set up integration-test dependencies and write integrationtests.env.
# Idempotent. Re-running re-uses cached artifacts.
#
# bitcoind, electrs and the other test dependencies are the enforcer's
# dependencies, so they are fetched by its own setup script rather than
# duplicated here. Set
# ENFORCER_ENV to an existing enforcer integrationtests.env to reuse ones
# another checkout already fetched, skipping the downloads.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENFORCER_DIR="$REPO_ROOT/bip300301_enforcer"
ENV_FILE="$REPO_ROOT/integrationtests.env"

if [ ! -f "$ENFORCER_DIR/Cargo.toml" ]; then
    echo "bip300301_enforcer submodule is not checked out." >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi

# --- Dependencies (bitcoind, electrs) ---
if [ -n "${ENFORCER_ENV:-}" ]; then
    if [ ! -f "$ENFORCER_ENV" ]; then
        echo "ENFORCER_ENV is set but '$ENFORCER_ENV' does not exist" >&2
        exit 1
    fi
    echo "Reusing enforcer dependencies from $ENFORCER_ENV"
    DEPS_ENV="$ENFORCER_ENV"
else
    echo "Fetching enforcer dependencies (first run downloads and builds; slow)..."
    "$ENFORCER_DIR/scripts/setup_integration_tests.sh"
    DEPS_ENV="$ENFORCER_DIR/integrationtests.env"
fi

# --- Binaries under test ---
# Built from the pinned submodule, so it matches the enforcer library truthcoin
# compiles against.
echo "Building bip300301_enforcer (from the pinned submodule)..."
cargo build --manifest-path "$ENFORCER_DIR/Cargo.toml" --bin bip300301_enforcer

echo "Building truthcoin_dc_app..."
cargo build --manifest-path "$REPO_ROOT/Cargo.toml" --bin truthcoin_dc_app

# --- Env file ---
# The variables that the enforcer harness reads change with the submodule pin,
# so copy all of them, and replace only the binaries under test. Every path is
# absolute, so the tests do not care what the working directory is.
{
    grep -vE '^(BIP300301_ENFORCER|TRUTHCOIN_APP)=' "$DEPS_ENV"
    echo "BIP300301_ENFORCER='$ENFORCER_DIR/target/debug/bip300301_enforcer'"
    echo "TRUTHCOIN_APP='$REPO_ROOT/target/debug/truthcoin_dc_app'"
} > "$ENV_FILE"

echo "Wrote $ENV_FILE"
echo
echo "Run integration tests with:"
echo "  cargo run --example integration_tests [-- <test-name>]"
