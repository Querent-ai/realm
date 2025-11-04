# Feature Completion Plan

**Goal**: Make Realm feature-complete for production deployment

## Priority Assessment

### High Priority (Core Production Features)

1. **Metal Flash Attention** ⚡ (Quick win - 2-3 hours)
   - Similar to CUDA implementation
   - Uses Candle Metal operations
   - Impact: 3-5x speedup for attention on Apple Silicon

2. **Continuous Batching** 🚀 (High impact - 4-6 hours)
   - Dynamic batching of requests
   - Better GPU utilization
   - Impact: 2-5x throughput improvement

3. **LoRA Adapters** 🎯 (Critical for multi-tenant - 6-8 hours)
   - Per-tenant fine-tuned models
   - Load/unload adapters dynamically
   - Impact: Enables per-tenant customization

### Medium Priority (Performance Optimizations)

4. **Speculative Decoding** ⚡ (Nice speedup - 6-8 hours)
   - Draft model + verify model pattern
   - Impact: 2-3x speedup for generation

5. **WebGPU Flash Attention** (Optional - 3-4 hours)
   - Similar to CUDA/Metal
   - Impact: Cross-platform GPU support

## Implementation Order

1. **Metal Flash Attention** - Quick win, similar to CUDA
2. **Continuous Batching** - High production value
3. **LoRA Adapters** - Critical for multi-tenant use case
4. **Speculative Decoding** - Performance optimization

Let's start with Metal Flash Attention and Continuous Batching!

