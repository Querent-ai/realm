# Next Steps & Missing Items

## ✅ Completed (This PR)

1. **HTTP/SSE Server** ✅
   - OpenAI-compatible `/v1/chat/completions` endpoint
   - Server-Sent Events streaming
   - Health and metrics endpoints
   - Tenant authentication
   - CLI integration (`--http` flag)

2. **WASM Tests with Mocked Host Functions** ✅
   - 7 unit tests in `crates/realm-wasm/tests/host_function_mocks.rs`
   - Tests tokenization, generation config, host function patterns, KV cache tracking
   - All tests passing

3. **Integration Tests** ✅
   - HTTP server integration tests in `crates/realm-server/tests/integration.rs`
   - Tests health, metrics, chat completions validation and structure
   - Gracefully skips when WASM file not available

4. **Code Quality** ✅
   - All clippy warnings fixed
   - Code formatted
   - Copilot review issues addressed

## ❌ Missing & Next Items

### High Priority (Production Readiness)

#### 1. **Load Tests** 🔴
**Status**: Not implemented  
**Location**: Should be in `crates/realm-server/tests/load_tests.rs` or `tests/load/`  
**What's Needed**:
- Concurrent request testing (10, 50, 100+ concurrent clients)
- Throughput measurement (requests/second)
- Latency percentiles (p50, p95, p99)
- Memory usage under load
- Connection pool exhaustion testing

**Tools**: 
- Use `criterion` for Rust benchmarks
- Or `k6`/`locust` for HTTP load testing
- Or `tokio::spawn` with many concurrent test clients

**Example Structure**:
```rust
#[tokio::test]
async fn test_concurrent_requests() {
    // Spawn 100 concurrent requests
    // Measure latency, throughput
}

#[tokio::test]
async fn test_sustained_load() {
    // Run for 60 seconds
    // Monitor memory, CPU
}
```

#### 2. **Real WASM Streaming Integration** 🟡
**Status**: Currently simulated  
**Location**: `crates/realm-server/src/http.rs:236`  
**What's Needed**:
- Integrate `generate_with_callback` from WASM layer
- Real token-by-token streaming via SSE
- Handle callback returning `false` to stop generation
- Test streaming with actual WASM module

**Current**: Generates full response then chunks it  
**Needed**: Call WASM callback for each token

#### 3. **Token Counting** 🟡
**Status**: Hardcoded to 0  
**Location**: `crates/realm-server/src/http.rs:219-220`  
**What's Needed**:
- Count prompt tokens (from input messages)
- Count completion tokens (from generated text)
- Use tokenizer from RuntimeManager
- Return accurate usage stats

#### 4. **Metrics Integration** 🟡
**Status**: Stub implementation  
**Location**: `crates/realm-server/src/http.rs:110-114`  
**What's Needed**:
- Integrate with `realm-metrics` crate
- Expose Prometheus metrics:
  - Request count, latency
  - Token generation rate
  - Error rates
  - Active connections
  - Model inference time

### Medium Priority (Enhancements)

#### 5. **End-to-End Integration Tests** 🟡
**Status**: Basic tests exist, need full flow  
**Location**: `crates/realm-server/tests/integration.rs`  
**What's Needed**:
- Full flow: HTTP request → RuntimeManager → WASM → Host Functions → Response
- Test with actual model file (when available)
- Test streaming end-to-end
- Test error handling (invalid requests, model not found, etc.)
- Test authentication flow

**Current**: Tests skip when WASM not available  
**Needed**: Optional tests that run when WASM + model available

#### 6. **WASM Generation Tests** 🟡
**Status**: Stubs exist  
**Location**: `crates/realm-wasm/tests/generation_tests.rs`  
**What's Needed**:
- Implement actual test bodies
- Test full generation loop
- Test KV cache management
- Test logits processing
- Test error handling
- Test multi-token generation

**Current**: Just placeholder comments  
**Needed**: Real test implementations

#### 7. **SDK Load Tests** 🟢
**Status**: Not implemented  
**Location**: `sdks/nodejs-ws/tests/` or `sdks/python-ws/tests/`  
**What's Needed**:
- Test SDKs under concurrent load
- Test WebSocket connection pooling
- Test HTTP client connection reuse
- Test streaming performance

### Low Priority (Future Enhancements)

#### 8. **Distributed Inference Testing** 🟢
**Status**: Framework exists, untested  
**Location**: `crates/realm-compute-gpu/src/distributed.rs`  
**What's Needed**:
- Integration tests for tensor/pipeline/data parallelism
- Simulation tests (don't need actual multi-GPU)
- Test model sharding
- Test communication patterns

#### 9. **Continuous Batching Integration** 🟢
**Status**: Framework exists, not integrated  
**Location**: `crates/realm-runtime/src/batching.rs`  
**What's Needed**:
- Wire into HTTP/SSE server
- Test batching multiple requests
- Measure throughput improvement
- Test with different batch sizes

#### 10. **LoRA Adapter Integration** 🟢
**Status**: Framework exists, not integrated  
**Location**: `crates/realm-runtime/src/lora.rs`  
**What's Needed**:
- Test per-tenant LoRA adapters
- Test adapter loading/unloading
- Test adapter switching during inference

#### 11. **Speculative Decoding Integration** 🟢
**Status**: Framework exists, not integrated  
**Location**: `crates/realm-runtime/src/speculative.rs`  
**What's Needed**:
- Test draft model + target model flow
- Measure speedup
- Test rejection sampling

## Test Coverage Summary

### Current Coverage
- ✅ Unit tests: WASM host function mocks (7 tests)
- ✅ Integration tests: HTTP endpoints (4 tests, skip if WASM missing)
- ✅ GPU tests: WebGPU dequantization (comprehensive)
- ✅ Core library tests: Quantization, tokenization, etc.

### Missing Coverage
- ❌ Load tests: 0%
- ❌ End-to-end tests: Partial (skip when WASM missing)
- ❌ Streaming tests: Simulated only
- ❌ SDK tests: Basic only, no load tests
- ❌ Error handling tests: Partial
- ❌ Authentication tests: Basic only

## Recommended Next Actions

### Week 1: Critical Tests
1. **Load Tests** (2-3 days)
   - Implement concurrent request testing
   - Measure throughput and latency
   - Test under sustained load

2. **Real Streaming Integration** (1-2 days)
   - Wire up WASM callbacks
   - Test token-by-token streaming
   - Verify SSE format correctness

3. **Token Counting** (1 day)
   - Implement prompt/completion token counting
   - Test with various inputs

### Week 2: Integration & Metrics
4. **Metrics Integration** (1-2 days)
   - Wire up realm-metrics
   - Expose Prometheus endpoints
   - Test metrics collection

5. **End-to-End Tests** (2-3 days)
   - Full flow tests with WASM + model
   - Error scenario testing
   - Authentication flow testing

### Week 3: Enhancements
6. **WASM Generation Tests** (2-3 days)
   - Implement full test suite
   - Test all generation scenarios

7. **SDK Load Tests** (1-2 days)
   - Test Node.js and Python SDKs under load

## Quick Wins (Can Do Now)

1. **Token Counting** - Straightforward, just need tokenizer
2. **Metrics Integration** - realm-metrics crate exists, just wire it up
3. **More Integration Tests** - Add tests for error cases, edge cases
4. **Documentation** - Add examples for load testing, streaming

## Testing Tools to Consider

- **Load Testing**: `criterion`, `k6`, `locust`, or custom with `tokio::spawn`
- **HTTP Testing**: `axum-test` (already used), `reqwest` for client
- **WebSocket Testing**: `tokio-tungstenite` test client
- **Benchmarking**: `criterion` for Rust, `time` for Python
- **Mocking**: Custom mocks (already done for host functions)

## Notes

- Most missing items are enhancements, not blockers
- Core functionality is tested and working
- Load tests are the biggest gap for production readiness
- Streaming integration is the next logical step after load tests
