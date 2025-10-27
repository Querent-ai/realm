// Rotary Position Embedding (RoPE) compute shader
// Applies rotary embeddings to query and key tensors for positional encoding

@group(0) @binding(0) var<storage, read_write> tensor: array<f32>;  // Shape: [batch, seq_len, n_heads, head_dim]
@group(0) @binding(1) var<storage, read> freqs_cos: array<f32>;     // Precomputed cos(m*theta)
@group(0) @binding(2) var<storage, read> freqs_sin: array<f32>;     // Precomputed sin(m*theta)

struct RoPEParams {
    batch_size: u32,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    position_offset: u32,  // For KV cache
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(3) var<uniform> params: RoPEParams;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // Calculate indices
    let total_elements = params.batch_size * params.seq_len * params.n_heads * params.head_dim;
    if (idx >= total_elements) {
        return;
    }

    // Decompose flat index
    let head_dim = params.head_dim;
    let n_heads = params.n_heads;
    let seq_len = params.seq_len;

    let dim_idx = idx % head_dim;
    let head_idx = (idx / head_dim) % n_heads;
    let pos = (idx / (head_dim * n_heads)) % seq_len;
    let batch_idx = idx / (seq_len * n_heads * head_dim);

    // Only apply to first half of head_dim (complex rotation)
    if (dim_idx >= head_dim / 2u) {
        return;
    }

    // Get position with offset for KV cache
    let abs_pos = pos + params.position_offset;

    // Get frequency index
    let freq_idx = abs_pos * (head_dim / 2u) + dim_idx;

    // Load cos and sin values
    let cos_val = freqs_cos[freq_idx];
    let sin_val = freqs_sin[freq_idx];

    // Get the pair of values to rotate
    let idx1 = idx;
    let idx2 = idx + head_dim / 2u;

    let x1 = tensor[idx1];
    let x2 = tensor[idx2];

    // Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    tensor[idx1] = x1 * cos_val - x2 * sin_val;
    tensor[idx2] = x1 * sin_val + x2 * cos_val;
}
