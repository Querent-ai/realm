# Paris Examples Output Files

This directory contains output files from all Paris generation examples.

## Files

- `native-paris-output.txt` - Native Rust example output
- `wasm-paris-output.txt` - WASM example output  
- `nodejs-wasm-paris-output.txt` - Node.js WASM example output
- `nodejs-sdk-paris-output.txt` - Node.js SDK example output
- `python-sdk-paris-output.txt` - Python SDK example output
- `server-paris-output.txt` - Server example output

## Note

These examples require a model file (e.g., TinyLlama Q4_K_M) to generate actual "Paris" output.
The output files show:
1. Compilation status
2. Expected output format
3. How to run with a model

## Running with Model

To get actual "Paris" output, you need:

1. **Model file**: Download TinyLlama or similar GGUF model
2. **Run example**: `cargo run --release -- /path/to/model.gguf`
3. **Expected**: Response containing "Paris" when asked "What is the capital of France?"

## Verification

All examples are designed to produce "Paris" when asked:
> "What is the capital of France?"

This validates:
- ✅ Model loading works
- ✅ Inference pipeline works  
- ✅ Tokenization works
- ✅ Generation works

