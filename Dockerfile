# Multi-stage Dockerfile for Realm
FROM rust:1.75-slim as builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install wasm-pack for WASM builds
RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Add WASM target
RUN rustup target add wasm32-unknown-unknown

# Set working directory
WORKDIR /build

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY rust-toolchain.toml ./

# Copy source code
COPY crates ./crates
COPY cli ./cli
COPY server ./server
COPY examples ./examples

# Build the release binary
RUN cargo build --release --bin realm

# Build WASM module
RUN cd crates/realm-wasm && wasm-pack build --target web

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 realm

# Copy binary from builder
COPY --from=builder /build/target/release/realm /usr/local/bin/realm

# Copy WASM module
COPY --from=builder /build/crates/realm-wasm/pkg /opt/realm/wasm/

# Create model directory
RUN mkdir -p /models && chown realm:realm /models

# Switch to non-root user
USER realm

# Set working directory
WORKDIR /home/realm

# Expose ports
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD realm info || exit 1

# Default command
ENTRYPOINT ["realm"]
CMD ["--help"]
