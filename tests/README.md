# Integration Tests

Integration tests for Realm are located within individual crates, not at the workspace level.

## Available Integration Tests

### Metrics Integration Tests
**Location**: `crates/realm-metrics/tests/integration_tests.rs`
**Run**: `cargo test --package realm-metrics --test integration_tests`

Tests comprehensive metrics collection, export formats, multi-tenant isolation, and end-to-end inference tracking.

### Runtime Integration Tests
**Location**: `crates/realm-runtime/tests/host_storage_integration.rs`
**Run**: `cargo test --package realm-runtime --test host_storage_integration`

Tests HOST-side model storage and multi-threaded access patterns.

## Why No Workspace-Level Tests?

The workspace root has no `src/` directory (it's a virtual workspace), so tests in `tests/` cannot be compiled. All integration tests are properly organized within their respective crates.

## Running All Integration Tests

```bash
# Run all integration tests
cargo test --workspace --test '*'

# Or specifically
cargo test --package realm-metrics --test integration_tests
cargo test --package realm-runtime --test host_storage_integration
```
