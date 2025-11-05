# Paris Example Outputs

This directory contains the output logs from running all Paris generation examples.

## Status

Each example attempts to answer: **"What is the capital of France?"** and should produce **"Paris"**.

## Log Files

- `native.log` - Native Rust example (direct API)
- `paris-generation.log` - Paris generation example (old example)
- `nodejs-wasm.log` - Node.js WASM example (local, no server)
- `nodejs-sdk.log` - Node.js WebSocket SDK example (requires server)
- `python-sdk.log` - Python WebSocket SDK example (requires server)
- `server.log` - Realm server log (for SDK examples)

## Running Examples

### Native Examples (No Server Required)

```bash
# Native Rust
cd examples/paris/native
cargo run --release -- /path/to/model.gguf > ../outputs/logs/native.log 2>&1

# Paris Generation
cd examples/paris-generation
cargo run --release -- /path/to/model.gguf > ../paris/outputs/logs/paris-generation.log 2>&1
```

### SDK Examples (Require Server)

1. Start server:
```bash
cd /home/puneet/realm
cargo run --release -p realm-cli -- serve \
    --host 127.0.0.1 \
    --port 8080 \
    --model /path/to/model.gguf \
    --wasm target/wasm32-unknown-unknown/release/realm_wasm.wasm
```

2. Run client examples:
```bash
# Node.js SDK
cd examples/paris/nodejs-sdk
REALM_URL=ws://localhost:8080 REALM_MODEL=tinyllama-1.1b.Q4_K_M.gguf node index.js > ../outputs/logs/nodejs-sdk.log 2>&1

# Python SDK
cd examples/paris/python-sdk
REALM_URL=ws://localhost:8080 REALM_MODEL=tinyllama-1.1b.Q4_K_M.gguf python main.py > ../outputs/logs/python-sdk.log 2>&1
```

## Verification

To verify an example produced "Paris":
```bash
grep -i "paris" native.log
```

Expected output should contain:
- "Paris" or "The capital of France is Paris"
- "✅ SUCCESS: Model correctly identified Paris"

