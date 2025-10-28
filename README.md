# Realm

[![CI](https://github.com/realm-ai/realm/workflows/CI/badge.svg)](https://github.com/realm-ai/realm/actions)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20%2F%20Apache--2.0-blue)](LICENSE)
[![Rust: 1.75+](https://img.shields.io/badge/rust-1.75%2B-orange)](https://www.rust-lang.org)

Multi-tenant LLM inference runtime using WASM sandboxing and native GPU acceleration.

Realm splits inference into two layers: WASM modules handle orchestration in isolated sandboxes, while a native runtime provides GPU compute and >4GB memory access via Memory64. This enables running multiple tenants on a single GPU with strong security boundaries.

## Architecture

```
┌─────────────────────────────────────────┐
│ realm-wasm (WASM Module)                │  ← Customer code runs here
│ • Token orchestration                   │  ← Sandboxed, isolated
│ • Inference coordination                │
└──────────────┬──────────────────────────┘
               │ Host function calls (candle_matmul, memory64_load_layer)
┌──────────────▼──────────────────────────┐
│ realm-runtime (Native Binary)           │  ← Shared across tenants
│ • Memory64: Large model storage         │  ← GPU acceleration
│ • Candle GPU backend (CUDA/Metal)       │  ← Multi-realm isolation
│ • Wasmtime: WASM host                   │
└─────────────────────────────────────────┘
```

**Key Properties:**
- **Isolation**: Each tenant runs in a separate WASM sandbox
- **Performance**: All tenants share one GPU through host function calls
- **Scalability**: 8-16 tenants per GPU (vs 1 for traditional approaches)
- **Memory Efficiency**: Models >4GB work via Memory64 lazy loading

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed technical design.

## Repository Structure

```
realm/
├── crates/
│   ├── realm-core/          # GGUF parsing, tokenization, tensor ops
│   ├── realm-models/        # Transformer architecture (attention, FFN)
│   ├── realm-compute-cpu/   # CPU backends (SIMD, Candle CPU)
│   ├── realm-compute-gpu/   # GPU backends (CUDA, Metal, WebGPU)
│   ├── realm-runtime/       # Host runtime (Memory64, Wasmtime)
│   └── realm-wasm/          # WASM orchestrator module
├── cli/                     # Command-line tool
├── server/                  # HTTP API server
├── sdks/
│   ├── js/                  # Node.js SDK (N-API)
│   ├── python/              # Python bindings (PyO3)
│   └── rust/                # Rust library
├── examples/                # Usage examples
└── docs/                    # Technical documentation
```

## Building

### Prerequisites

- Rust 1.75 or later
- For GPU support:
  - CUDA 11.8+ (NVIDIA)
  - Metal SDK (Apple Silicon)
- For WASM: `wasm-pack` and `wasm32-unknown-unknown` target

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install WASM target
rustup target add wasm32-unknown-unknown

# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

### Build All Crates

```bash
# Build native runtime and all crates
cargo build --release

# Build WASM module
cd crates/realm-wasm
wasm-pack build --target web
```

### Build with GPU Support

```bash
# CUDA
cargo build --release --features cuda

# Metal (macOS/iOS)
cargo build --release --features metal
```

## Testing

```bash
# Run all tests
cargo test

# Run with specific backend
cargo test --features cuda

# Run simple architecture test
cargo run --bin simple-realm-test

# Run benchmarks
cargo bench
```

## Development

### Project Setup

```bash
# Clone repository
git clone https://github.com/realm-ai/realm.git
cd realm

# Build everything
make build

# Run tests
make test

# Run lints
make lint
```

### Running Examples

```bash
# Simple host/WASM integration test
cargo run --example simple-realm-test

# Multi-tenant example
cargo run --example multi-tenant

# Embedding in Node.js
cd examples/nodejs-embedding
npm install && npm test
```

### Code Style

- Run `cargo fmt` before committing
- Run `cargo clippy -- -D warnings` to catch issues
- Follow Rust API guidelines
- Add tests for new functionality

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Crates

| Crate | Description | Dependencies |
|-------|-------------|--------------|
| `realm-core` | GGUF parsing, tokenization, quantization | `bytemuck`, `half`, `ndarray` |
| `realm-models` | Transformer layers, attention, FFN | `realm-core`, `realm-compute-*` |
| `realm-compute-cpu` | CPU kernels (SIMD, Candle CPU) | `candle-core`, `rayon` |
| `realm-compute-gpu` | GPU backends (CUDA, Metal, WebGPU) | `candle-core`, `wgpu` |
| `realm-runtime` | Memory64 host, Wasmtime integration | `wasmtime`, `parking_lot` |
| `realm-wasm` | WASM orchestrator module | `wasm-bindgen` |

## Features

### `realm-runtime`
- `cuda` - Enable CUDA GPU support (requires CUDA 11.8+)
- `metal` - Enable Metal GPU support (macOS/iOS only)
- `memory64` - Enable Memory64 for large models

### `realm-compute-cpu`
- `simd` - Enable SIMD optimizations

### `realm-compute-gpu`
- `cuda` - CUDA backend
- `metal` - Metal backend
- `webgpu` - WebGPU backend

## Examples

- **[simple-realm-test](examples/simple-realm-test)** - Basic host/WASM integration
- **[multi-tenant](examples/multi-tenant)** - Running multiple isolated instances
- **[nodejs-embedding](examples/nodejs-embedding)** - Embedding in Node.js app
- **[python-bindings](examples/python-bindings)** - Using from Python

## Performance

Benchmark results on NVIDIA A100 (40GB):

| Workload | Traditional (vLLM) | Realm | Improvement |
|----------|-------------------|-------|-------------|
| Tenants per GPU | 1 | 16 | **16x** |
| Memory per tenant | 40GB | 2.5GB | **16x** |
| Throughput degradation | N/A | <5% | Minimal |

See [BENCHMARKS.md](docs/BENCHMARKS.md) for detailed results.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System design and components
- [Deployment](docs/DEPLOYMENT.md) - Production deployment guide
- [Embedding](docs/EMBEDDING_MODEL.md) - Integrating into applications
- [API Reference](https://docs.realm.ai/api) - Generated API docs
- [Contributing](CONTRIBUTING.md) - How to contribute

## Status

**Alpha**: Core functionality works, but APIs may change. Not recommended for production yet.

### What Works
- ✅ GGUF model loading
- ✅ Transformer inference (attention, FFN)
- ✅ CPU backends (Candle, SIMD)
- ✅ GPU backends (CUDA, Metal)
- ✅ Memory64 integration
- ✅ WASM module compilation
- ✅ Host function bridging

### In Progress
- 🚧 CLI tool
- 🚧 HTTP API server
- 🚧 Node.js SDK
- 🚧 Python bindings
- 🚧 Production deployment tooling

### Planned
- 📋 Flash Attention
- 📋 Speculative decoding
- 📋 Continuous batching
- 📋 Quantization (GGML Q4/Q5/Q8)

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

## Community

- [Discord](https://discord.gg/realm) - Chat with the community
- [GitHub Discussions](https://github.com/realm-ai/realm/discussions) - Ask questions
- [Twitter](https://twitter.com/realm_ai) - Updates and announcements
