# Paris Generation - Native Rust

This example demonstrates **native Rust inference** using Realm's direct API:

- **Question**: "What is the capital of France?"
- **Expected Answer**: "Paris"

## What This Shows

- ✅ Direct Rust API (no WASM, no server)
- ✅ GGUF model loading
- ✅ Complete transformer inference
- ✅ Tokenization and generation

## Run

```bash
cd examples/paris/native

# With model path
cargo run --release -- /path/to/model.gguf

# Or use default path
cargo run --release
```

## Expected Output

```
Realm Paris Generation - Native Rust
====================================

Loading model: ../../models/tinyllama-1.1b.Q4_K_M.gguf
Model header parsed
Config loaded: 22 layers, 32 heads
Model created
Tokenizer loaded: 32000 tokens
Weights loaded

Prompt: "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"

Generating response...

Response: The capital of France is Paris.

✅ SUCCESS: Model correctly identified Paris as the capital of France!
```

## Architecture

```
┌─────────────────────────┐
│  Native Rust Code       │
│  (this example)         │
└──────────┬──────────────┘
           │
┌──────────▼──────────────┐
│  realm-models          │
│  (Model, Transformer)   │
└──────────┬──────────────┘
           │
┌──────────▼──────────────┐
│  realm-core             │
│  (GGUF, Tokenizer)     │
└──────────┬──────────────┘
           │
┌──────────▼──────────────┐
│  realm-compute-cpu      │
│  (Candle CPU)          │
└─────────────────────────┘
```

## Notes

- This is the **simplest** way to use Realm
- No WASM, no server, no network
- Direct Rust API calls
- Perfect for embedding Realm in Rust applications

