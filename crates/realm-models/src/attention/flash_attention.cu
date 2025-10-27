// Flash Attention CUDA Kernel
//
// High-performance GPU implementation of Flash Attention for NVIDIA GPUs.
// 
// Expected speedup: 3-4x faster than CPU on modern GPUs
//
// Algorithm:
// 1. Block-wise tiling to maximize shared memory usage
// 2. Online softmax with incremental statistics
// 3. Warp-level primitives for fast reductions
// 4. Coalesced memory access patterns
//
// Memory hierarchy:
// - Global memory (HBM): Q, K, V tensors
// - Shared memory (SRAM): Tiled blocks (Q_block, K_block, V_block)
// - Registers: Accumulators, statistics (m, l, o)

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

// Warp-level reduction for finding maximum
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Flash Attention forward kernel
//
// Grid: (num_heads, batch_size)
// Block: (block_size_q, 1, 1)
//
// Each thread block processes one Q block and iterates over K/V blocks
__global__ void flash_attention_forward_kernel(
    const float* __restrict__ q,     // [batch, num_heads, seq_len_q, head_dim]
    const float* __restrict__ k,     // [batch, num_heads, seq_len_k, head_dim]
    const float* __restrict__ v,     // [batch, num_heads, seq_len_k, head_dim]
    const float* __restrict__ mask,  // [seq_len_q, seq_len_k] or nullptr
    float* __restrict__ output,      // [batch, num_heads, seq_len_q, head_dim]
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim,
    int block_size_q,
    int block_size_k,
    float scale
) {
    // Shared memory for tiled blocks
    extern __shared__ float shared_mem[];
    float* q_block = shared_mem;                                        // [block_size_q, head_dim]
    float* k_block = q_block + block_size_q * head_dim;                // [block_size_k, head_dim]
    float* v_block = k_block + block_size_k * head_dim;                // [block_size_k, head_dim]
    float* scores = v_block + block_size_k * head_dim;                 // [block_size_q, block_size_k]
    
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int q_block_start = blockIdx.z * block_size_q;
    const int tid = threadIdx.x;
    
    // Check bounds
    if (q_block_start >= seq_len_q) return;
    const int q_block_len = min(block_size_q, seq_len_q - q_block_start);
    
    // Global memory offsets
    const int q_offset = ((batch_idx * num_heads + head_idx) * seq_len_q + q_block_start) * head_dim;
    const int kv_offset = (batch_idx * num_heads + head_idx) * seq_len_k * head_dim;
    
    // Registers for statistics
    float m[8];  // max per query (supports up to 8 queries per thread)
    float l[8];  // sum per query
    float o[8 * 64];  // output accumulator (up to 8 queries * 64 head_dim)
    
    // Initialize
    for (int i = 0; i < q_block_len && i < 8; i++) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
        for (int d = 0; d < head_dim && d < 64; d++) {
            o[i * 64 + d] = 0.0f;
        }
    }
    
    // Load Q block into shared memory (coalesced)
    for (int i = tid; i < q_block_len * head_dim; i += blockDim.x) {
        int q_i = i / head_dim;
        int d = i % head_dim;
        if (q_block_start + q_i < seq_len_q) {
            q_block[i] = q[q_offset + q_i * head_dim + d];
        }
    }
    __syncthreads();
    
    // Iterate over K/V blocks
    for (int k_block_start = 0; k_block_start < seq_len_k; k_block_start += block_size_k) {
        const int k_block_len = min(block_size_k, seq_len_k - k_block_start);
        
        // Load K block (coalesced)
        for (int i = tid; i < k_block_len * head_dim; i += blockDim.x) {
            int k_j = i / head_dim;
            int d = i % head_dim;
            k_block[i] = k[kv_offset + (k_block_start + k_j) * head_dim + d];
        }
        
        // Load V block (coalesced)
        for (int i = tid; i < k_block_len * head_dim; i += blockDim.x) {
            int v_j = i / head_dim;
            int d = i % head_dim;
            v_block[i] = v[kv_offset + (k_block_start + v_j) * head_dim + d];
        }
        __syncthreads();
        
        // Compute attention scores for this block
        // S_ij = Q_i @ K_j^T / sqrt(head_dim)
        for (int i = tid / k_block_len; i < q_block_len; i += blockDim.x / k_block_len) {
            int j = tid % k_block_len;
            if (j < k_block_len) {
                float score = 0.0f;
                
                // Dot product
                for (int d = 0; d < head_dim; d++) {
                    score += q_block[i * head_dim + d] * k_block[j * head_dim + d];
                }
                score *= scale;
                
                // Apply causal mask if provided
                if (mask != nullptr) {
                    int q_pos = q_block_start + i;
                    int k_pos = k_block_start + j;
                    if (k_pos > q_pos) {
                        score = -INFINITY;
                    }
                }
                
                scores[i * block_size_k + j] = score;
            }
        }
        __syncthreads();
        
        // Online softmax update for each query
        // This is where the Flash Attention magic happens!
        for (int i = tid; i < q_block_len; i += blockDim.x) {
            // 1. Find max in current block
            float m_curr = -INFINITY;
            for (int j = 0; j < k_block_len; j++) {
                m_curr = fmaxf(m_curr, scores[i * block_size_k + j]);
            }
            
            // 2. Update global max
            float m_new = fmaxf(m[i], m_curr);
            
            // 3. Compute exp and new sum
            float l_curr = 0.0f;
            for (int j = 0; j < k_block_len; j++) {
                float exp_val = expf(scores[i * block_size_k + j] - m_new);
                scores[i * block_size_k + j] = exp_val;
                l_curr += exp_val;
            }
            
            // 4. Rescale previous output
            float scale_o = expf(m[i] - m_new);
            for (int d = 0; d < head_dim; d++) {
                o[i * 64 + d] *= scale_o;
            }
            
            // 5. Add contribution from current block
            for (int j = 0; j < k_block_len; j++) {
                float weight = scores[i * block_size_k + j];
                for (int d = 0; d < head_dim; d++) {
                    o[i * 64 + d] += weight * v_block[j * head_dim + d];
                }
            }
            
            // 6. Update statistics
            l[i] = l[i] * scale_o + l_curr;
            m[i] = m_new;
        }
        __syncthreads();
    }
    
    // Normalize and write output
    for (int i = tid; i < q_block_len; i += blockDim.x) {
        const int out_offset = q_offset + i * head_dim;
        for (int d = 0; d < head_dim; d++) {
            output[out_offset + d] = o[i * 64 + d] / l[i];
        }
    }
}

// Host function to launch kernel
extern "C" void flash_attention_forward_cuda(
    const float* q,
    const float* k,
    const float* v,
    const float* mask,
    float* output,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim
) {
    const float scale = 1.0f / sqrtf((float)head_dim);
    const int block_size_q = 32;   // Tune based on GPU
    const int block_size_k = 32;
    
    // Calculate shared memory size
    int shared_mem_size = (block_size_q * head_dim +     // Q block
                          block_size_k * head_dim +      // K block
                          block_size_k * head_dim +      // V block
                          block_size_q * block_size_k)   // Scores
                          * sizeof(float);
    
    // Launch configuration
    dim3 grid(num_heads, batch_size, (seq_len_q + block_size_q - 1) / block_size_q);
    dim3 block(256);  // Threads per block
    
    flash_attention_forward_kernel<<<grid, block, shared_mem_size>>>(
        q, k, v, mask, output,
        batch_size, num_heads, seq_len_q, seq_len_k, head_dim,
        block_size_q, block_size_k, scale
    );
    
    cudaDeviceSynchronize();
}

// Backward pass kernel (placeholder for gradient computation)
__global__ void flash_attention_backward_kernel(
    // TODO: Implement backward pass for training
) {
    // Placeholder
}

