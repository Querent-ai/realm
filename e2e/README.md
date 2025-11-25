# Realm E2E Tests

End-to-end tests for Realm WASM runtime with server integration.

## Overview

This directory contains comprehensive end-to-end tests that verify:
- ✅ Paris generation verification (basic inference)
- ✅ LoRA adapter integration readiness
- ✅ Speculative decoding integration readiness
- ✅ Continuous batching integration
- ✅ Real token-by-token streaming
- ✅ SDK-WASM integration

## Setup

1. **Build the server and WASM:**
   ```bash
   cd <project-root>
   cargo build --release --bin realm
   cd crates/realm-wasm
   wasm-pack build --target nodejs --out-dir pkg-server
   ```

2. **Install dependencies:**
   ```bash
   cd e2e
   npm install
   ```

3. **Set environment variables (optional):**
   ```bash
   export REALM_SERVER_URL=http://localhost:3000
   export REALM_WASM_PATH=../crates/realm-wasm/pkg-server/realm_wasm_bg.wasm
   export REALM_MODEL_PATH=~/.realm/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
   ```

## Running Tests

### Automatic (Recommended)

The test runner will start the server, run all tests, and stop the server:

```bash
npm test
# or
npm run test:all
```

### Manual (Server Already Running)

If you have a server running separately:

```bash
# In one terminal, start the server:
cd <project-root>
cargo run --bin realm -- serve \
  --wasm crates/realm-wasm/pkg-server/realm_wasm_bg.wasm \
  --model ~/.realm/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --port 3000 \
  --http

# In another terminal, run tests:
cd e2e
npm run test:manual
```

### Individual Tests

```bash
# Paris generation verification
npm run test:paris

# LoRA adapter tests
npm run test:lora

# Speculative decoding tests
npm run test:speculative

# Continuous batching tests
npm run test:batching
```

## Test Structure

- `test-runner.js` - Automatic test runner (starts/stops server)
- `test-paris.js` - Basic inference tests, verifies "capital of France" → "Paris"
- `test-lora.js` - LoRA adapter integration readiness tests
- `test-speculative.js` - Speculative decoding integration readiness tests
- `test-batching.js` - Continuous batching throughput tests

## Expected Results

All tests should pass when:
- Server is running and accessible
- Model is loaded (default model or via configuration)
- WASM runtime is properly initialized
- HTTP endpoints are available

## SDK-WASM Integration

The tests verify that:
1. **Server-WASM Integration**: ✅ Working
   - Server loads WASM module correctly
   - WASM orchestrates inference via host functions
   - Host functions perform computation on HOST side

2. **HTTP API**: ✅ Working
   - `/health` endpoint responds
   - `/v1/chat/completions` generates text
   - Streaming works via SSE

3. **SDK Integration**: ✅ Ready
   - Node.js SDK uses WebSocket client
   - Python SDK uses WebSocket client
   - Both connect to server (not directly to WASM)

## Architecture

```
┌─────────────────────────────────────────┐
│  E2E Tests (Node.js)                    │
│  ┌───────────────────────────────────┐  │
│  │ HTTP Client (fetch)               │  │
│  │  ↓                                 │  │
│  │ HTTP/SSE Server (realm-server)    │  │
│  │  ↓                                 │  │
│  │ RuntimeManager                     │  │
│  │  ↓                                 │  │
│  │ WASM Runtime (wasmtime)           │  │
│  │  ↓                                 │  │
│  │ Host Functions (HOST computation) │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## Troubleshooting

1. **Server not responding:**
   - Check that server is running: `curl http://localhost:3000/health`
   - Verify `REALM_SERVER_URL` environment variable
   - Check server logs for errors

2. **Model not loaded:**
   - Check server logs for model loading errors
   - Verify model path in server configuration
   - Ensure model file exists and is readable

3. **WASM errors:**
   - Check that the WASM module is built (see setup above)
   - Verify WASM path in server configuration
   - Check server logs for WASM instantiation errors

4. **Tests timeout:**
   - Increase `TIMEOUT` in test files if generation is slow
   - Check server performance
   - Verify model is loaded correctly

## Test Results

When all tests pass:
- ✅ Server-WASM integration working
- ✅ HTTP API functional
- ✅ Generation working
- ✅ Streaming working
- ✅ Batching working
- ✅ LoRA framework ready
- ✅ Speculative decoding framework ready

## Next Steps

After E2E tests pass:
1. Add HTTP endpoints for LoRA adapter configuration
2. Add HTTP endpoints for speculative decoding configuration
3. Add more comprehensive LoRA tests with actual adapters
4. Add more comprehensive speculative decoding tests with draft models
