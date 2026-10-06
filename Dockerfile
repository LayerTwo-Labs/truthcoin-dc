# The Rust version is NOT pinned by this tag — it comes from
# rust-toolchain.toml, which rustup in this image reads. The tag only
# bootstraps rustup, so it deliberately floats on the latest 1.x.
FROM rust:1-slim-bookworm AS builder
WORKDIR /workspace

# Install the pinned toolchain before copying the source, so the download
# is cached in its own layer and is not invalidated by code changes.
COPY rust-toolchain.toml .
RUN rustup toolchain install

COPY . .

RUN cargo build --locked --release

# Runtime stage
FROM debian:bookworm-slim

COPY --from=builder /workspace/target/release/truthcoin_dc_app /bin/truthcoin_dc_app
COPY --from=builder /workspace/target/release/truthcoin_dc_app_cli /bin/truthcoin_dc_app_cli

# Verify we placed the binary in the right place, 
# and that it's executable.
RUN truthcoin_dc_app --help

ENTRYPOINT ["truthcoin_dc_app"]

