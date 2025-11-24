# How Inference Works - Implementation Guide

## Overview

The inference system uses `InferenceSession` from `crates/realm-runtime/src/inference.rs` to manage token generation with proper state tracking, streaming support, and speculative decoding.

## Architecture

```
Prompt (string)
    ↓
Tokenizer.encode() → prompt_tokens: Vec<u32>
    ↓
InferenceSession::new(model_id, prompt_tokens, options)
    ↓
Loop: InferenceSession::next_token_with_model(model)
    ↓
    model.forward(input_tokens) → logits: Vec<f32>
    ↓
    LogitsProcessor.sample(logits) → token_id: u32
    ↓
    Check stop tokens, max tokens
    ↓
    Add to generated_tokens
    ↓
Tokenizer.decode(generated_tokens) → Generated Text
```

## Key Components

### 1. InferenceSession

**Location**: `crates/realm-runtime/src/inference.rs`

**Purpose**: Manages the complete inference lifecycle:
- Tracks prompt tokens and generated tokens
- Manages generation state (Ready → Generating → Complete/Stopped)
- Handles token buffering for streaming
- Supports speculative decoding (draft model verification)

**Key Methods**:
- `new(model_id, prompt_tokens, options)` - Create session
- `next_token_with_model(model, draft_model?)` - Generate next token
- `generated_tokens()` - Get all generated tokens so far
- `is_complete()` - Check if generation finished

### 2. GenOptions

**Structure**:
```rust
pub struct GenOptions {
    pub max_tokens: u32,           // Maximum tokens to generate
    pub temperature: f32,           // Sampling temperature
    pub top_p: f32,                // Nucleus sampling
    pub top_k: u32,                // Top-k sampling
    pub repetition_penalty: f32,   // Penalty for repetition
    pub seed: u32,                  // Random seed
    pub stop_token_count: u8,      // Number of stop tokens
    pub stop_tokens_ptr: u32,      // Pointer to stop tokens array
}
```

### 3. Model.forward()

**Location**: `crates/realm-models/src/model.rs`

**Purpose**: Performs the actual transformer forward pass:
1. Embeds tokens → hidden states
2. Forward through transformer layers
3. Final normalization
4. Project to vocabulary (LM head) → logits

**Signature**:
```rust
pub fn forward(&mut self, token_ids: &[u32], position: usize) -> Result<Vec<f32>>
```

Returns logits of shape `[seq_len * vocab_size]`, where the last `vocab_size` elements are the logits for the last token.

### 4. LogitsProcessor

**Location**: `crates/realm-runtime/src/sampling.rs`

**Purpose**: Samples next token from logits using configured strategy:
- Temperature sampling
- Top-k filtering
- Top-p (nucleus) sampling
- Repetition penalty

## Implementation Flow

### Step 1: Tokenize Prompt
```rust
let tokenizer = model.tokenizer().unwrap();
let prompt_tokens = tokenizer.encode(prompt, true)?; // true = add BOS token
```

### Step 2: Create InferenceSession
```rust
let options = GenOptions {
    max_tokens: 512,
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40,
    repetition_penalty: 1.1,
    seed: 42,
    stop_token_count: 0,
    stop_tokens_ptr: 0,
};

let mut session = InferenceSession::new(model_id, prompt_tokens, options);
```

### Step 3: Generate Tokens
```rust
let mut generated_tokens = Vec::new();

while !session.is_complete() {
    if let Some(token_id) = session.next_token_with_model(&mut model, None)? {
        generated_tokens.push(token_id);
    } else {
        break; // Generation complete
    }
}
```

### Step 4: Decode Result
```rust
let generated_text = tokenizer.decode(&generated_tokens, true)?; // true = skip special tokens
```

## Integration with realm_host_generate

The `realm_host_generate` function in `memory64_host.rs` needs to:

1. **Get tokenizer from storage**:
   ```rust
   let storage = get_global_model_storage().lock();
   let stored_model = storage.get_model(model_id)?;
   let tokenizer = stored_model.tokenizer().ok_or("No tokenizer")?;
   ```

2. **Tokenize prompt**:
   ```rust
   let prompt_tokens = tokenizer.encode(&prompt, true)?;
   ```

3. **Create InferenceSession**:
   ```rust
   let options = GenOptions::default(); // Or use custom options
   let mut session = InferenceSession::new(model_id, prompt_tokens, options);
   ```

4. **Get Model instance**:
   - **Challenge**: `StoredModel` only has quantized tensors, but `next_token_with_model()` needs a `realm_models::Model` instance
   - **Solution Options**:
     a. Load Model from GGUF bytes (store bytes in storage)
     b. Cache Model instances per model_id
     c. Create Model wrapper that uses host functions

5. **Generate tokens**:
   ```rust
   let mut all_tokens = Vec::new();
   while !session.is_complete() {
       if let Some(token) = session.next_token_with_model(&mut model, None)? {
           all_tokens.push(token);
       } else {
           break;
       }
   }
   ```

6. **Decode result**:
   ```rust
   let result = tokenizer.decode(&all_tokens, true)?;
   ```

## Current Status

✅ **Implemented**:
- `InferenceSession` with full token generation logic
- Tokenization and decoding via host functions
- Model storage with tokenizer

⚠️ **TODO**:
- Create `Model` instance from `StoredModel` for inference
- Options:
  1. Store GGUF bytes in `StoredModel` and load on-demand
  2. Add Model cache to storage (one per model_id)
  3. Create lightweight Model wrapper using host functions

## Next Steps

1. **Add Model Loading**: Implement model loading from storage (either from GGUF bytes or file path)
2. **Add Model Cache**: Cache loaded Model instances to avoid reloading on every request
3. **Integrate**: Replace echo in `realm_host_generate` with actual inference using `InferenceSession`

