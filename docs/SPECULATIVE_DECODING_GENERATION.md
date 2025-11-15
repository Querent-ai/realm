# Speculative Decoding Generation Path Integration

**Date**: 2025-01-31  
**Status**: Framework Ready, Generation Path Integration Pending

---

## 🎯 Current Status

### ✅ What's Complete

1. **Draft Model Loading**: ✅ Complete
   - Draft model config stored in `TenantRuntime`
   - Draft model loaded into WASM/model storage
   - Accessible via `TenantRuntime::draft_model_config()`

2. **Speculative Decoder Framework**: ✅ Complete
   - `SpeculativeDecoder<D, T>` trait-based implementation
   - `DraftModel` and `TargetModel` traits
   - Token acceptance/rejection logic
   - Unit tests with mock models

### ⏳ What's Pending

**Generation Path Integration**: The speculative decoder needs to be integrated into the actual generation loop in WASM.

---

## 📋 Integration Plan

### Option 1: Host-Side Integration (Recommended)

Integrate speculative decoding at the host level before calling WASM:

```rust
// In RuntimeManager::generate_stream()
if let Some(draft_config) = runtime.draft_model_config() {
    // Use speculative decoder
    let mut decoder = SpeculativeDecoder::new(
        DraftModelWrapper::new(draft_model_id),
        TargetModelWrapper::new(target_model_id),
        SpeculativeConfig::default(),
    );
    
    // Generate with speculative decoding
    let tokens = decoder.generate(&prompt_tokens, max_tokens)?;
    // Stream tokens...
} else {
    // Standard generation
    runtime.generate(prompt)?;
}
```

**Pros**:
- Easier to implement
- No WASM changes needed
- Can reuse existing model storage

**Cons**:
- Requires host-side model instances (not just WASM)

### Option 2: WASM Integration

Modify WASM generation loop to support draft model:

```rust
// In realm-wasm/src/lib.rs generate_with_host_storage_internal()
if let Some(draft_model_id) = self.draft_model_id {
    // Use speculative decoding
    // 1. Draft model generates k tokens
    // 2. Target model verifies
    // 3. Accept/reject logic
} else {
    // Standard generation
}
```

**Pros**:
- Keeps all logic in WASM
- Consistent with current architecture

**Cons**:
- Requires significant WASM changes
- More complex to implement

---

## 🔧 Implementation Steps (Option 1)

1. **Create Model Wrappers**:
   ```rust
   struct DraftModelWrapper {
       model_id: u32,
   }
   
   impl DraftModel for DraftModelWrapper {
       fn generate_draft(&mut self, prompt: &[u32], k: usize) -> Result<Vec<u32>> {
           // Call WASM or host function to generate k tokens from draft model
       }
   }
   ```

2. **Create Target Model Wrapper**:
   ```rust
   struct TargetModelWrapper {
       model_id: u32,
   }
   
   impl TargetModel for TargetModelWrapper {
       fn verify_draft(&mut self, prompt: &[u32], draft_tokens: &[u32]) -> Result<Vec<u32>> {
           // Verify draft tokens using target model
       }
   }
   ```

3. **Integrate into RuntimeManager**:
   - Check if draft model is configured
   - Create speculative decoder
   - Use decoder for generation instead of standard path

---

## 📝 Notes

- Draft model loading is complete and ready
- Speculative decoder framework is tested and working
- Integration into generation path is the remaining step
- Can be implemented incrementally without breaking existing functionality

---

**Status**: Framework ready, integration pending implementation decision

