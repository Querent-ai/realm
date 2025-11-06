# WebGPU Shader Optimization Plan

**Date**: 2025-01-31  
**Status**: CPU confirmed ready, README updated ✅

---

## ✅ Current Status

### CPU Crate
- ✅ **82 tests passing**
- ✅ **Only 2 minor SIMD TODOs** (low priority, Q5_0/Q5_1 optimization)
- ✅ **Production-ready**

### README Updates
- ✅ Streaming status updated (Node.js & Python SDKs)
- ✅ WebGPU section added
- ✅ Infrastructure status updated (Terraform/Helm)
- ✅ SDK examples updated with streaming

### WebGPU Shaders (Current)

**Existing Shaders**:
- ✅ `matmul.wgsl` - Basic matrix multiplication
- ✅ `matmul_tiled.wgsl` - Optimized tiled matmul with shared memory
- ✅ `q4k.wgsl` - Q4_K dequantization shader
- ✅ `rmsnorm.wgsl` - RMS normalization
- ✅ `rope.wgsl` - Rotary position embedding
- ✅ `softmax.wgsl` - Softmax operation

**Missing Shaders**:
- ❌ `q5k.wgsl` - Q5_K dequantization shader
- ❌ `q6k.wgsl` - Q6_K dequantization shader
- ❌ `q8k.wgsl` - Q8_K dequantization shader

**Current Implementation**:
- WebGPU backend uses **CPU dequantization + GPU matmul** for Q5_K, Q6_K, Q8_K
- Q4_K shader exists but may not be wired up yet
- Matmul shader is loaded but may not use the optimized tiled version

---

## 🎯 Optimization Plan

### Phase 1: Complete Dequant Shaders (Priority: High)

**Goal**: Create GPU-native dequantization shaders for all quantization formats

**Tasks**:
1. **Create `q5k.wgsl`** - Q5_K dequantization shader
   - Similar structure to `q4k.wgsl`
   - Handle Q5_K block format (d, dmin, scales, qh, ql)
   - Process 256-element blocks

2. **Create `q6k.wgsl`** - Q6_K dequantization shader
   - Handle Q6_K block format
   - Process 256-element blocks

3. **Create `q8k.wgsl`** - Q8_K dequantization shader
   - Handle Q8_K block format
   - Process 256-element blocks

**Reference**: Use `q4k.wgsl` as template, adapt for each format's block structure

**Files to Create**:
- `crates/realm-compute-gpu/src/q5k.wgsl`
- `crates/realm-compute-gpu/src/q6k.wgsl`
- `crates/realm-compute-gpu/src/q8k.wgsl`

---

### Phase 2: Wire Up Shaders (Priority: High)

**Goal**: Connect shaders to the WebGPU backend implementation

**Tasks**:
1. **Load dequant shaders in `GpuBackend::new()`**
   - Load Q4_K, Q5_K, Q6_K, Q8_K shaders
   - Create compute pipelines for each

2. **Update `fused_dequant_matmul_*` methods**
   - Use GPU shaders instead of CPU dequantization
   - Upload quantized blocks directly to GPU
   - Execute dequant + matmul in single kernel (or pipeline)

3. **Use optimized tiled matmul**
   - Switch from `matmul.wgsl` to `matmul_tiled.wgsl`
   - Or create separate pipeline for tiled version

**Files to Modify**:
- `crates/realm-compute-gpu/src/lib.rs` - Load shaders, create pipelines
- `crates/realm-compute-gpu/src/lib.rs` - Update `fused_dequant_matmul_*` methods

---

### Phase 3: Optimize Existing Shaders (Priority: Medium)

**Goal**: Improve performance of existing shaders

**Tasks**:
1. **Optimize `matmul_tiled.wgsl`**
   - Increase tile size if beneficial (16x16 → 32x32?)
   - Optimize memory access patterns
   - Add prefetching hints

2. **Optimize `q4k.wgsl`**
   - Review scale extraction logic
   - Optimize loop unrolling
   - Reduce register pressure

3. **Create fused dequant+matmul shaders**
   - Combine dequant and matmul in single kernel
   - Avoid intermediate buffer writes
   - Better memory locality

**Files to Modify**:
- `crates/realm-compute-gpu/src/matmul_tiled.wgsl`
- `crates/realm-compute-gpu/src/q4k.wgsl`
- Create new fused shaders: `fused_q4k_matmul.wgsl`, etc.

---

## 📋 Implementation Order

### Week 1: Complete Missing Shaders
1. Day 1-2: Create `q5k.wgsl`
2. Day 3-4: Create `q6k.wgsl`
3. Day 5: Create `q8k.wgsl`

### Week 2: Wire Up Shaders
1. Day 1-2: Load all shaders in `GpuBackend::new()`
2. Day 3-4: Update `fused_dequant_matmul_*` methods
3. Day 5: Testing and validation

### Week 3: Optimize (Optional)
1. Optimize existing shaders
2. Create fused kernels
3. Benchmark improvements

---

## 🎯 Expected Benefits

**Current**: CPU dequant + GPU matmul
- CPU-GPU transfer overhead
- Limited parallelism

**After Phase 1-2**: GPU-native dequant + matmul
- ✅ No CPU-GPU transfers for dequantized weights
- ✅ Better GPU utilization
- ✅ **20-30% speedup** (estimated)

**After Phase 3**: Fused kernels
- ✅ Single kernel execution
- ✅ Better memory locality
- ✅ **Additional 10-15% speedup** (estimated)

---

## 📝 Notes

- **Reference Implementation**: Use CPU dequantization code in `realm-core` as reference
- **Block Formats**: Each quantization format has different block structure
- **Testing**: Test each shader independently before integration
- **Benchmarking**: Compare CPU dequant vs GPU dequant performance

---

## ✅ Success Criteria

- [ ] All dequant shaders created (Q4_K, Q5_K, Q6_K, Q8_K)
- [ ] Shaders wired up to WebGPU backend
- [ ] Tests passing with GPU-native dequantization
- [ ] Performance improvement measured (20-30% speedup)
- [ ] No regressions in existing functionality

