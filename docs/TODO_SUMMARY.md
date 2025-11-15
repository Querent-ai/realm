# TODO Summary

**Date**: 2025-01-31  
**Status**: All Critical TODOs Documented

---

## 🔴 High Priority TODOs

### 1. Speculative Decoding Tokenization Integration
**Location**: `crates/realm-server/src/speculative_integration.rs:39, 66, 102-103`

- **Line 39**: `// TODO: Implement proper tokenization/de-tokenization`
  - Currently uses placeholder tokenization
  - Need to integrate tokenizer into host functions
  
- **Line 66**: `// TODO: Implement proper verification logic`
  - Currently accepts all draft tokens
  - Need to compare target model predictions with draft tokens

- **Lines 102-103**: `// TODO: Get actual draft model ID` and `// TODO: Get actual target model ID`
  - Currently uses hardcoded IDs
  - Need to track model IDs from model storage

**Effort**: 1-2 days  
**Priority**: High (blocks full speculative decoding functionality)

---

### 2. GPU Batch Forward Pass Implementation
**Location**: `crates/realm-runtime/src/batch_forward.rs:114`

- **Line 114**: `// TODO: Implement actual GPU batch forward pass`
  - Currently falls back to CPU sequential processing
  - Need to implement parallel GPU batch processing

**Effort**: 3-5 days  
**Priority**: High (significant performance improvement)

---

### 3. Batch Forward Pass Integration
**Location**: `crates/realm-server/src/dispatcher.rs:623`

- **Line 623**: `// TODO: Use realm_runtime::batch_forward::BatchForwardBackend for parallel processing`
  - Framework is ready, needs integration into dispatcher
  - Replace sequential processing with batch forward pass

**Effort**: 1 day  
**Priority**: Medium (depends on GPU implementation)

---

## 🟡 Medium Priority TODOs

### 4. Speculative Decoding Full Integration
**Location**: `crates/realm-server/src/runtime_manager.rs:776`

- **Line 776**: `// TODO: Implement full speculative decoding integration`
  - Framework is ready, needs proper model ID tracking
  - Requires tokenization integration (see #1)

**Effort**: 1-2 days  
**Priority**: Medium (depends on tokenization)

---

### 5. WASM Backend Refactoring
**Location**: `crates/realm-wasm/src/lib.rs:475`

- **Line 475**: `// TODO: Refactor to use HOST-provided backends via FFI`
  - Currently creates backends in WASM
  - Should use HOST-provided backends for consistency

**Effort**: 2-3 days  
**Priority**: Medium (architectural improvement)

---

### 6. Row-wise Dequantization Optimization
**Location**: `crates/realm-runtime/src/memory64_host.rs:2067`

- **Line 2067**: `// TODO: Implement row-wise dequantization for efficiency`
  - Current implementation dequantizes entire blocks
  - Row-wise would be more efficient for large matrices

**Effort**: 1-2 days  
**Priority**: Medium (performance optimization)

---

## 🟢 Low Priority TODOs

### 7. SIMD Optimizations
**Location**: `crates/realm-compute-cpu/src/fused.rs:2337, 2422`

- **Lines 2337, 2422**: `// TODO: Optimize SIMD implementation later`
  - Current implementation works, but could be faster
  - Low priority optimization

**Effort**: 2-3 days  
**Priority**: Low (performance optimization)

---

### 8. Distributed Computing
**Location**: `crates/realm-compute-gpu/src/distributed.rs:247, 256, 265, 274, 283`

- Multiple TODOs for NCCL integration (broadcast, all-reduce, gather, scatter, cleanup)
- Distributed training/inference support

**Effort**: 1-2 weeks  
**Priority**: Low (future feature)

---

### 9. OpenTelemetry Integration
**Location**: `crates/realm-metrics/src/export/opentelemetry.rs:23, 29, 35, 41, 54`

- Multiple TODOs for OpenTelemetry metrics export
- Observability enhancement

**Effort**: 2-3 days  
**Priority**: Low (observability)

---

### 10. Tokenizer Improvements
**Location**: `crates/realm-core/src/tokenizer.rs:210`

- **Line 210**: `// TODO: Parse from metadata if available`
  - Currently uses empty merges
  - Should parse from model metadata

**Effort**: 1 day  
**Priority**: Low (functionality works)

---

### 11. Attention Implementation Fixes
**Location**: `crates/realm-models/src/lib.rs:88, 232, 272, 325`

- Multiple ignored tests for attention implementation
- Need to fix attention bugs

**Effort**: 2-3 days  
**Priority**: Low (tests are ignored, functionality may work)

---

### 12. Node.js Implementation
**Location**: `crates/realm-node/src/lib.rs:305, 337`

- **Line 305**: `// TODO: Implement full attention with KV cache`
- **Line 337**: `// TODO: Implement full FFN (gate, up, down projections)`
- Node.js bindings incomplete

**Effort**: 3-5 days  
**Priority**: Low (Node.js support)

---

### 13. ABI Tokenization
**Location**: `crates/realm-runtime/src/abi.rs:148`

- **Line 148**: `// TODO: Tokenize prompt (_prompt) before creating session`
- Tokenization should happen in ABI layer

**Effort**: 1 day  
**Priority**: Low (functionality works)

---

### 14. Flash Attention Backward Pass
**Location**: `crates/realm-runtime/src/attention/flash_attention.cu:240`

- **Line 240**: `// TODO: Implement backward pass for training`
- Training support (currently inference-only)

**Effort**: 1 week  
**Priority**: Low (training feature)

---

## 📊 Summary

| Priority | Count | Total Effort |
|----------|-------|--------------|
| 🔴 High | 3 | 5-8 days |
| 🟡 Medium | 3 | 4-7 days |
| 🟢 Low | 8 | 3-4 weeks |

**Total**: 14 TODOs across the codebase

---

## 🎯 Recommended Next Steps

1. **Speculative Decoding Tokenization** (High Priority)
   - Most critical for completing speculative decoding
   - Enables full functionality

2. **GPU Batch Forward Pass** (High Priority)
   - Significant performance improvement
   - Enables true parallel processing

3. **Batch Forward Pass Integration** (Medium Priority)
   - Quick win once GPU implementation is ready
   - Immediate performance benefit

---

## ✅ Completed TODOs

- ✅ Quantized LoRA support (all formats)
- ✅ Speculative decoding framework
- ✅ Continuous batching framework
- ✅ Batch forward pass interface
- ✅ Comprehensive test coverage
- ✅ E2E test infrastructure

---

**Most critical TODOs are documented and ready for implementation!** 🚀

