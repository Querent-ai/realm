# GPU Testing Guide

This guide explains how to test GPU backends when GPU hardware is available.

---

## Prerequisites

### CUDA (NVIDIA GPU)
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- `nvcc` in PATH
- Rust toolchain with `cuda` feature enabled

### Metal (Apple Silicon)
- Apple Silicon Mac (M1, M2, M3, etc.)
- macOS 11.0 or later
- Rust toolchain with `metal` feature enabled

### WebGPU
- WebGPU-compatible browser or runtime
- Rust toolchain with `webgpu` feature enabled

---

## Building with GPU Support

### CUDA

```bash
# Build with CUDA support
cargo build --features cuda --release

# Or install with CUDA
cargo install --path . --features cuda
```

### Metal

```bash
# Build with Metal support
cargo build --features metal --release

# Or install with Metal
cargo install --path . --features metal
```

### WebGPU

```bash
# Build with WebGPU support
cargo build --features webgpu --release

# Or install with WebGPU
cargo install --path . --features webgpu
```

---

## Running GPU Tests

### CUDA Tests

```bash
# Run all GPU backend tests
cargo test -p realm-compute-gpu --features cuda --lib

# Run specific test
cargo test -p realm-compute-gpu --features cuda --lib test_matmul

# Run with verbose output
RUST_LOG=debug cargo test -p realm-compute-gpu --features cuda --lib
```

### Metal Tests

```bash
# Run all GPU backend tests
cargo test -p realm-compute-gpu --features metal --lib

# Run specific test
cargo test -p realm-compute-gpu --features metal --lib test_matmul
```

### WebGPU Tests

```bash
# Run all GPU backend tests
cargo test -p realm-compute-gpu --features webgpu --lib

# Note: WebGPU tests may require browser environment
```

---

## Running Inference with GPU

### Check GPU Availability

```bash
# CUDA
cargo run --features cuda --bin simple-realm-test -- --check-gpu

# Metal
cargo run --features metal --bin simple-realm-test -- --check-gpu
```

### Run Inference

```bash
# CUDA
RUST_LOG=info cargo run --features cuda --release --bin simple-realm-test -- \
  --model /path/to/model.gguf \
  --prompt "Hello, world!"

# Metal
RUST_LOG=info cargo run --features metal --release --bin simple-realm-test -- \
  --model /path/to/model.gguf \
  --prompt "Hello, world!"
```

### Verify GPU Usage

Look for log messages indicating GPU backend usage:

```
INFO realm_compute_gpu::candle_backend: Using CUDA backend
INFO realm_runtime::memory64_host: GPU matmul called (CUDA)
```

Or CPU fallback:

```
WARN realm_compute_gpu::candle_backend: CUDA not available, using CPU backend
INFO realm_runtime::memory64_host: CPU matmul called
```

---

## Benchmarking GPU Performance

### Compare CPU vs GPU

```bash
# CPU only
cargo bench -p realm-compute-cpu --bench matmul

# GPU (CUDA)
cargo bench -p realm-compute-gpu --features cuda --bench matmul

# GPU (Metal)
cargo bench -p realm-compute-gpu --features metal --bench matmul
```

### Expected Speedups

- **CUDA**: 6-7x faster than CPU
- **Metal**: 4-5x faster than CPU
- **WebGPU**: Varies by implementation

---

## Troubleshooting

### CUDA Issues

**Problem**: `CUDA not available`

**Solutions**:
1. Check CUDA installation: `nvcc --version`
2. Check GPU: `nvidia-smi`
3. Verify CUDA libraries are in LD_LIBRARY_PATH
4. Check Rust CUDA feature compilation

**Problem**: `CUDA out of memory`

**Solutions**:
1. Reduce batch size
2. Use smaller model
3. Check GPU memory: `nvidia-smi`

### Metal Issues

**Problem**: `Metal not available`

**Solutions**:
1. Verify Apple Silicon Mac
2. Check macOS version (11.0+)
3. Verify Metal framework available

### WebGPU Issues

**Problem**: `WebGPU not available`

**Solutions**:
1. Check browser WebGPU support
2. Verify WebGPU runtime installed
3. Check feature flags enabled

---

## CI Integration

### Current CI Behavior

The CI currently handles GPU tests gracefully:

```yaml
- name: Run GPU backend tests
  if: matrix.os == 'ubuntu-latest'
  run: |
    echo "Running GPU backend tests..."
    cargo test -p realm-compute-gpu --lib || echo "⚠️  GPU tests skipped (GPU not available in CI)"
  continue-on-error: true
```

This allows CI to pass even when GPU is not available.

### Adding GPU-Enabled CI

When you have GPU-enabled CI runners, update the workflow:

```yaml
gpu-test:
  name: 🎮 GPU Tests (CUDA)
  runs-on: [self-hosted, gpu, cuda]
  steps:
    - uses: actions/checkout@v4
    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable
    - name: Install CUDA
      run: |
        # Install CUDA toolkit
        sudo apt-get update
        sudo apt-get install -y cuda-toolkit
    - name: Run GPU tests
      run: cargo test -p realm-compute-gpu --features cuda --lib
    - name: Run GPU benchmarks
      run: cargo bench -p realm-compute-gpu --features cuda --bench matmul
```

---

## Performance Profiling

### CUDA Profiling

```bash
# Use nvprof
nvprof cargo run --features cuda --release --bin simple-realm-test

# Use nsys
nsys profile --output profile.nsys-rep cargo run --features cuda --release
```

### Metal Profiling

Use Xcode Instruments or `metal-system-monitor` for Metal profiling.

---

## Summary

- **Building**: Use `--features cuda`, `--features metal`, or `--features webgpu`
- **Testing**: Run `cargo test -p realm-compute-gpu --features <backend> --lib`
- **CI**: Currently handles GPU tests gracefully (skips if unavailable)
- **Future**: Add GPU-enabled CI runners when hardware is available

**Note**: GPU tests are optional and CI passes without GPU hardware.

