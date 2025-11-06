// WebGPU compute shader for fused Q4_K dequantization + matrix multiplication
// Q4_K format: 256-element blocks (QK_K) with hierarchical scaling
// Structure: d (f16), dmin (f16), scales[12], qs[128]
// Each block processes 256 values in 4 groups of 64

// Block structure (148 bytes total: 4+4+12+128)
// Note: We convert f16 to f32, so d and dmin are 4 bytes each
struct BlockQ4K {
    d: f32,              // Super-block scale (converted from f16, 4 bytes)
    dmin: f32,           // Super-block min scale (converted from f16, 4 bytes)
    scales: array<u32, 3>, // Packed scales (12 bytes = 3 u32s)
    qs: array<u32, 32>,  // Quantized values (128 bytes = 32 u32s)
}

@group(0) @binding(0) var<storage, read> blocks: array<BlockQ4K>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    batch_size: u32,
    n: u32,              // Number of output features
    k: u32,              // Input dimension (must be multiple of 256)
    num_k_blocks: u32,   // k / 256
    _pad: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

// Extract scale and min from Q4_K scales array (matches get_scale_min_k4)
// scales is 3 u32s = 12 bytes, we need to extract individual bytes
fn get_scale_min_k4(j: u32, scales: array<u32, 3>) -> vec2<u32> {
    // Scales are packed: 6 bits per scale, 12 bytes total
    // We have 3 u32s, each containing 4 bytes
    let scale_val: u32;
    let min_val: u32;
    
    // Calculate which byte index we need (0-11)
    let byte_idx_scale = j; // Scale byte index
    let byte_idx_min = j + 4u; // Min byte index (offset by 4)
    
    // Extract bytes from u32 array
    let u32_idx_scale = byte_idx_scale / 4u;
    let byte_offset_scale = (byte_idx_scale % 4u) * 8u;
    let scale_byte = (scales[u32_idx_scale] >> byte_offset_scale) & 0xFFu;
    
    let u32_idx_min = byte_idx_min / 4u;
    let byte_offset_min = (byte_idx_min % 4u) * 8u;
    let min_byte = (scales[u32_idx_min] >> byte_offset_min) & 0xFFu;
    
    if (j < 4u) {
        // First 4 groups: simple extraction
        scale_val = scale_byte & 63u;
        min_val = min_byte & 63u;
    } else {
        // Remaining groups: more complex packing
        let idx = j - 4u;
        let byte_idx_scale2 = idx + 4u;
        let byte_idx_min2 = idx;
        
        let u32_idx_scale2 = byte_idx_scale2 / 4u;
        let byte_offset_scale2 = (byte_idx_scale2 % 4u) * 8u;
        let scale_byte2 = (scales[u32_idx_scale2] >> byte_offset_scale2) & 0xFFu;
        
        let u32_idx_min2 = byte_idx_min2 / 4u;
        let byte_offset_min2 = (byte_idx_min2 % 4u) * 8u;
        let min_byte2 = (scales[u32_idx_min2] >> byte_offset_min2) & 0xFFu;
        
        scale_val = (scale_byte2 & 0xFu) | ((min_byte2 >> 6u) << 4u);
        min_val = (scale_byte2 >> 4u) | ((min_byte2 >> 6u) << 4u);
    }
    
    return vec2<u32>(scale_val, min_val);
}

// Workgroup size: 16x16 threads per output element
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    let batch_idx = global_id.y;
    
    // Bounds check
    if (out_idx >= params.n || batch_idx >= params.batch_size) {
        return;
    }
    
    var sum = 0.0;
    
    // Process each K-block (256 elements each)
    for (var k_block = 0u; k_block < params.num_k_blocks; k_block++) {
        let k_start = k_block * 256u;
        let block_idx = out_idx * params.num_k_blocks + k_block;
        
        if (block_idx >= arrayLength(&blocks)) {
            break;
        }
        
        let block = blocks[block_idx];
        let d = block.d;
        let min = block.dmin;
        
        // Process 4 groups of 64 values each
        for (var i = 0u; i < 4u; i++) {
            let is = i * 2u; // Scale index
            
            // Get scales for this group
            let scale_min1 = get_scale_min_k4(is, block.scales);
            let d1 = d * f32(scale_min1.x);
            let m1 = min * f32(scale_min1.y);
            
            let scale_min2 = get_scale_min_k4(is + 1u, block.scales);
            let d2 = d * f32(scale_min2.x);
            let m2 = min * f32(scale_min2.y);
            
            // Process 64 values: 32 from lower nibbles, 32 from upper nibbles
            let q_offset = i * 32u;
            let y_offset = i * 64u;
            
            // Lower nibbles (32 values)
            for (var j = 0u; j < 32u; j++) {
                let q_byte_idx = (q_offset + j) / 4u; // Which u32 in qs array
                let q_bit_offset = ((q_offset + j) % 4u) * 8u;
                let q_byte = (block.qs[q_byte_idx] >> q_bit_offset) & 0xFFu;
                let x = f32(q_byte & 0xFu);
                let dequant = d1 * x - m1;
                
                let k_idx = k_start + y_offset + j;
                if (k_idx < params.k) {
                    sum += dequant * input[batch_idx * params.k + k_idx];
                }
            }
            
            // Upper nibbles (32 values)
            for (var j = 0u; j < 32u; j++) {
                let q_byte_idx = (q_offset + j) / 4u;
                let q_bit_offset = ((q_offset + j) % 4u) * 8u;
                let q_byte = (block.qs[q_byte_idx] >> q_bit_offset) & 0xFFu;
                let x = f32(q_byte >> 4u);
                let dequant = d2 * x - m2;
                
                let k_idx = k_start + y_offset + 32u + j;
                if (k_idx < params.k) {
                    sum += dequant * input[batch_idx * params.k + k_idx];
                }
            }
        }
    }
    
    // Write result
    let output_idx = batch_idx * params.n + out_idx;
    output[output_idx] = sum;
}




