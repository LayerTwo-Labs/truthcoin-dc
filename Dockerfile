# Stable Rust version, as of January 2025. 
FROM rust:1.84-slim-bookworm AS builder
WORKDIR /workspace
COPY . .

RUN cargo build --locked --release

# Runtime stage
FROM debian:bookworm-slim

COPY --from=builder /workspace/target/release/truthcoin_dc_app /bin/truthcoin_dc_app
COPY --from=builder /workspace/target/release/truthcoin_dc_app_cli /bin/truthcoin_dc_app_cli

# Verify we placed the binaries in the right place, 
# and that it's executable.
RUN truthcoin_dc_app --help
RUN truthcoin_dc_app_cli --help

ENTRYPOINT ["truthcoin_dc_app"]

