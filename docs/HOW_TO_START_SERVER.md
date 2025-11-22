# How to Start Server Locally with Logging

## Quick Start (With Debug Logs)

```bash
# Set log level to debug
export RUST_LOG=debug

# Start server
./target/release/realm serve \
  --wasm crates/realm-wasm/pkg-server/realm_wasm_bg.wasm \
  --model models/tinyllama-1.1b.Q4_K_M.gguf \
  --port 3000 \
  --host 127.0.0.1
```

## One-Liner (All in One Command)

```bash
RUST_LOG=debug ./target/release/realm serve \
  --wasm crates/realm-wasm/pkg-server/realm_wasm_bg.wasm \
  --model models/tinyllama-1.1b.Q4_K_M.gguf \
  --port 3000
```

## Log Levels

- `RUST_LOG=error` - Only errors
- `RUST_LOG=warn` - Warnings and errors
- `RUST_LOG=info` - Info, warnings, errors (default)
- `RUST_LOG=debug` - Debug, info, warnings, errors (most verbose)
- `RUST_LOG=trace` - Everything (very verbose)

## Test the Server

Once server is running, test it:

```bash
# Health check
curl http://localhost:3000/health

# Generate text
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 20
  }'
```

## Save Logs to File

```bash
RUST_LOG=debug ./target/release/realm serve \
  --wasm crates/realm-wasm/pkg-server/realm_wasm_bg.wasm \
  --model models/tinyllama-1.1b.Q4_K_M.gguf \
  --port 3000 \
  2>&1 | tee server.log
```

## Background Mode (Like E2E Tests)

```bash
RUST_LOG=debug ./target/release/realm serve \
  --wasm crates/realm-wasm/pkg-server/realm_wasm_bg.wasm \
  --model models/tinyllama-1.1b.Q4_K_M.gguf \
  --port 3000 \
  > /tmp/realm-server.log 2>&1 &

echo $! > /tmp/realm-server.pid

# View logs
tail -f /tmp/realm-server.log

# Stop server
kill $(cat /tmp/realm-server.pid)
```

## What to Look For in Logs

1. **Server startup**:
   - `Starting Realm server`
   - `WASM module loaded`
   - `Model loaded`

2. **Request handling**:
   - `Generating for tenant`
   - `generate() WASM entry`
   - `realm_host_generate CALLED`

3. **Errors**:
   - `ERROR:` - Critical errors
   - `Failed to` - Operation failures
   - `return -X` - Error codes from host functions



