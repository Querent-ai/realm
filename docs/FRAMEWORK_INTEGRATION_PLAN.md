# Framework Integration Completion Plan

**Date**: 2025-01-31  
**Status**: Ready to Complete  
**Effort**: 1-2 days total

---

## 🎯 Why Complete These?

### 1. LoRA Integration
- **Value**: Per-tenant fine-tuning without full model copies
- **Status**: 90% complete - framework ready, just needs weight application
- **Impact**: HIGH - Enables customization per customer

### 2. Speculative Decoding
- **Value**: 2-3x speedup for generation
- **Status**: 85% complete - framework integrated, needs draft model loading
- **Impact**: HIGH - Major performance improvement

### 3. Continuous Batching
- **Value**: Better GPU utilization, higher throughput
- **Status**: 70% complete - framework exists, needs batch forward pass
- **Impact**: MEDIUM - Improves efficiency for multi-tenant workloads

---

## 📋 Integration Tasks

### 1. LoRA Runtime Integration (4-6 hours)

**What's Missing**:
- LoRA weights are loaded and stored, but not applied during model forward pass
- `lora_adapter_id` is set but weights aren't used in inference

**What Needs to Happen**:
1. Apply LoRA weights during model loading or forward pass
2. Hook into weight loading in `realm-models` or apply in `realm-runtime`
3. Apply LoRA delta: `W_final = W_base + (alpha/rank) * (B @ A)`

**Files to Modify**:
- `crates/realm-runtime/src/memory64_model.rs` - Apply LoRA after loading weights
- `crates/realm-models/src/model.rs` - Apply LoRA delta to layer weights
- Or: Apply during forward pass (more flexible but slower)

**Implementation Approach**:
```rust
// In memory64_model.rs or model.rs
if let Some(adapter_id) = &runtime.lora_adapter_id {
    if let Some(adapter) = lora_manager.get_adapter(adapter_id) {
        // Apply LoRA to each layer
        for (layer_name, base_weights) in &mut layer_weights {
            if let Some(lora_a) = adapter.lora_a.get(layer_name) {
                if let Some(lora_b) = adapter.lora_b.get(layer_name) {
                    // Compute delta: delta = (alpha/r) * (B @ A)
                    let delta = compute_lora_delta(lora_a, lora_b, adapter.alpha, adapter.rank);
                    // Apply: W_final = W_base + delta
                    apply_lora_delta(base_weights, &delta);
                }
            }
        }
    }
}
```

**Testing**:
- Load a LoRA adapter file
- Verify weights are modified
- Test inference with and without LoRA

---

### 2. Speculative Decoding Integration (3-4 hours)

**What's Missing**:
- Draft model loading in `RuntimeManager`
- Passing draft model to `InferenceSession::next_token_with_model()`
- Configuration for draft/target model pairing

**What Needs to Happen**:
1. Add draft model loading to `RuntimeManager`
2. Store draft model alongside target model per tenant
3. Pass draft model to inference session
4. Add configuration option to enable speculative decoding

**Files to Modify**:
- `crates/realm-server/src/runtime_manager.rs` - Add draft model storage
- `crates/realm-server/src/dispatcher.rs` - Pass draft model to inference
- `crates/realm-server/src/runtime_manager.rs` - Add `draft_model_path` to `ModelConfig`

**Implementation Approach**:
```rust
// In RuntimeManager
pub struct TenantRuntime {
    // ... existing fields ...
    draft_model: Option<Model>, // Add draft model
}

// In get_or_create_runtime_with_model
if let Some(draft_model_path) = config.draft_model_path {
    let draft_model = load_model(draft_model_path)?;
    runtime.draft_model = Some(draft_model);
}

// In dispatcher, when calling inference
let draft_model = runtime.draft_model.as_mut();
let token = session.next_token_with_model(&mut model, draft_model)?;
```

**Testing**:
- Load TinyLlama as draft, Llama-2 as target
- Verify speculative decoding is used
- Measure speedup (should be 2-3x)

---

### 3. Continuous Batching Integration (6-8 hours)

**What's Missing**:
- Actual batch forward pass (currently processes sequentially)
- Dynamic batch processing (triggered by time or size)
- Batch result distribution back to requests

**What Needs to Happen**:
1. Implement batch forward pass in model
2. Add batch processing trigger (time-based or size-based)
3. Distribute results back to individual requests
4. Handle variable sequence lengths in batch

**Files to Modify**:
- `crates/realm-server/src/dispatcher.rs` - Implement batch processing
- `crates/realm-models/src/model.rs` - Add batch forward pass (if needed)
- `crates/realm-runtime/src/batching.rs` - Add batch processing logic

**Implementation Approach**:
```rust
// In dispatcher::handle_generate_with_batching
async fn process_batch(batch: Vec<BatchedRequest>) -> Result<Vec<GenerationResult>> {
    // 1. Pad sequences to same length
    let max_len = batch.iter().map(|r| r.current_seq_len()).max().unwrap_or(0);
    let padded_tokens = batch.iter().map(|r| pad_to(r.all_tokens(), max_len)).collect();
    
    // 2. Batch forward pass
    let batch_logits = model.batch_forward(&padded_tokens)?;
    
    // 3. Sample tokens for each request
    let results: Vec<GenerationResult> = batch_logits
        .iter()
        .zip(batch.iter())
        .map(|(logits, request)| {
            let token = sample_token(logits);
            // Update request with new token
            // Return result
        })
        .collect();
    
    Ok(results)
}
```

**Challenges**:
- Variable sequence lengths (need padding)
- Different max_tokens per request
- Early completion (some requests finish before others)

**Testing**:
- Send multiple concurrent requests
- Verify batch processing happens
- Measure throughput improvement

---

## 🚀 Implementation Order

### Day 1 Morning: LoRA Integration (4-6 hours)
1. ✅ Framework already complete
2. Add LoRA weight application to model loading
3. Test with real LoRA adapter

### Day 1 Afternoon: Speculative Decoding (3-4 hours)
1. ✅ Framework already integrated
2. Add draft model loading to RuntimeManager
3. Connect to inference path
4. Test with draft + target models

### Day 2: Continuous Batching (6-8 hours)
1. ✅ Framework already exists
2. Implement batch forward pass
3. Add batch processing trigger
4. Test with concurrent requests

**Total**: 13-18 hours (1.5-2 days)

---

## ✅ Success Criteria

### LoRA
- [ ] LoRA adapter loads successfully
- [ ] Weights are applied to model layers
- [ ] Inference produces different results with LoRA
- [ ] Per-tenant LoRA adapters work

### Speculative Decoding
- [ ] Draft model loads alongside target model
- [ ] Speculative decoding is used when enabled
- [ ] 2-3x speedup measured vs standard inference
- [ ] Token quality maintained

### Continuous Batching
- [ ] Multiple requests batched together
- [ ] Batch forward pass processes all requests
- [ ] Results distributed correctly
- [ ] Throughput improvement measured

---

## 🎯 Why This Order?

1. **LoRA First** - Simplest, most straightforward integration
2. **Speculative Second** - Medium complexity, high impact
3. **Batching Last** - Most complex, needs careful testing

---

## 💡 Key Insights

1. **All frameworks are ready** - Just need connection
2. **High value additions** - All three provide real benefits
3. **Low risk** - Frameworks are tested, just need integration
4. **No REST API needed** - WebSocket architecture is superior

---

## 📝 Notes

- **REST API is NOT needed** - Your WebSocket + function dispatch architecture is better
- **SDKs work perfectly** - Node.js and Python SDKs call functions over WebSocket
- **WASM connects to host** - Host functions bridge WASM to GPU/CPU
- **Focus on integrations** - These provide real value with minimal effort

