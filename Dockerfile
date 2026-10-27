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

# Run as a non-root user so a compromised container cannot write host-mapped
# paths as root, and so wallet / cookie files created at runtime are not owned
# by uid 0.
RUN groupadd --system --gid 1000 truthcoin \
    && useradd --system --uid 1000 --gid truthcoin --home-dir /var/lib/truthcoin \
       --shell /usr/sbin/nologin truthcoin \
    && mkdir -p /var/lib/truthcoin \
    && chown -R truthcoin:truthcoin /var/lib/truthcoin

COPY --from=builder /workspace/target/release/truthcoin_dc_app /bin/truthcoin_dc_app
COPY --from=builder /workspace/target/release/truthcoin_dc_app_cli /bin/truthcoin_dc_app_cli

# Verify we placed the binary in the right place,
# and that it's executable.
RUN truthcoin_dc_app --help

USER truthcoin
WORKDIR /var/lib/truthcoin
ENV HOME=/var/lib/truthcoin

ENTRYPOINT ["truthcoin_dc_app"]
