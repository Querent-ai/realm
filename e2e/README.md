# Realm E2E Tests

End-to-end tests for Realm WASM runtime with server integration.

## Overview

This directory contains comprehensive end-to-end tests that verify:
- ✅ Paris generation verification (basic inference)
- ✅ LoRA adapter integration
- ✅ Speculative decoding integration
- ✅ Continuous batching integration
- ✅ Real token-by-token streaming

## Setup

1. **Start the Realm server:**
   ```bash
   cd /home/puneet/realm
   cargo run --bin realm-server
   ```

2. **Install dependencies:**
   ```bash
   cd e2e
   npm install
   ```

3. **Set server URL (optional):**
   ```bash
   export REALM_SERVER_URL=http://localhost:3000
   ```

## Running Tests

### All Tests
```bash
npm run test:all
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

- `test-paris.js` - Basic inference tests, verifies "capital of France" → "Paris"
- `test-lora.js` - LoRA adapter loading and application tests
- `test-speculative.js` - Speculative decoding with draft model tests
- `test-batching.js` - Continuous batching throughput tests

## Expected Results

All tests should pass when:
- Server is running and accessible
- Model is loaded (default model or via configuration)
- WASM runtime is properly initialized

## Troubleshooting

1. **Server not responding:**
   - Check that server is running: `curl http://localhost:3000/health`
   - Verify `REALM_SERVER_URL` environment variable

2. **Model not loaded:**
   - Check server logs for model loading errors
   - Verify model path in server configuration

3. **WASM errors:**
   - Check that WASM module is built: `cargo build --target wasm32-unknown-unknown`
   - Verify WASM path in server configuration

