// WebGPU compute shader for fused Q4_K dequantization + matrix multiplication
// Q4_K format: 256-element blocks (QK_K) with hierarchical scaling
// Structure: d (f16), dmin (f16), scales[12], qs[128]
// Each block processes 256 values in 4 groups of 64

// Block structure (148 bytes total: 4+4+12+128)
// Note: We convert f16 to f32, so d and dmin are 4 bytes each
// scales: 12 bytes stored as 3 u32s (we'll extract bytes)
// qs: 128 bytes stored as 32 u32s (we'll extract bytes)
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

// Extract a byte from scales array (12 bytes = 3 u32s)
// Must use constant indices, so we unroll for 0-11
fn get_scale_byte(scales: array<u32, 3>, byte_idx: u32) -> u32 {
    if (byte_idx == 0u) {
        return scales[0u] & 0xFFu;
    } else if (byte_idx == 1u) {
        return (scales[0u] >> 8u) & 0xFFu;
    } else if (byte_idx == 2u) {
        return (scales[0u] >> 16u) & 0xFFu;
    } else if (byte_idx == 3u) {
        return (scales[0u] >> 24u) & 0xFFu;
    } else if (byte_idx == 4u) {
        return scales[1u] & 0xFFu;
    } else if (byte_idx == 5u) {
        return (scales[1u] >> 8u) & 0xFFu;
    } else if (byte_idx == 6u) {
        return (scales[1u] >> 16u) & 0xFFu;
    } else if (byte_idx == 7u) {
        return (scales[1u] >> 24u) & 0xFFu;
    } else if (byte_idx == 8u) {
        return scales[2u] & 0xFFu;
    } else if (byte_idx == 9u) {
        return (scales[2u] >> 8u) & 0xFFu;
    } else if (byte_idx == 10u) {
        return (scales[2u] >> 16u) & 0xFFu;
    } else {
        return (scales[2u] >> 24u) & 0xFFu;
    }
}

// Extract scale and min from Q4_K scales array (matches get_scale_min_k4)
fn get_scale_min_k4(j: u32, scales: array<u32, 3>) -> vec2<u32> {
    var scale_val: u32 = 0u;
    var min_val: u32 = 0u;
    
    if (j < 4u) {
        // First 4 groups: scales[j] & 63, scales[j+4] & 63
        let scale_byte = get_scale_byte(scales, j);
        let min_byte = get_scale_byte(scales, j + 4u);
        scale_val = scale_byte & 63u;
        min_val = min_byte & 63u;
    } else {
        // Remaining groups: more complex packing
        let idx = j - 4u;
        let scale_byte = get_scale_byte(scales, idx + 4u);
        let min_byte = get_scale_byte(scales, idx);
        scale_val = (scale_byte & 0xFu) | ((min_byte >> 6u) << 4u);
        min_val = (scale_byte >> 4u) | ((min_byte >> 6u) << 4u);
    }
    
    return vec2<u32>(scale_val, min_val);
}

// Extract a byte from qs array (128 bytes = 32 u32s)
// For qs, we can use dynamic indexing since it's a larger array
// But WGSL still requires constant indices, so we need to unroll or use a different approach
// Actually, for storage buffers with dynamic arrays, indexing should work
// But for fixed-size arrays in structs, we need constants
// Let's use a helper that extracts bytes from a u32
fn extract_byte_from_u32(val: u32, byte_pos: u32) -> u32 {
    if (byte_pos == 0u) {
        return val & 0xFFu;
    } else if (byte_pos == 1u) {
        return (val >> 8u) & 0xFFu;
    } else if (byte_pos == 2u) {
        return (val >> 16u) & 0xFFu;
    } else {
        return (val >> 24u) & 0xFFu;
    }
}

// Get qs byte - unroll for u32 index, then extract byte
fn get_qs_byte(qs: array<u32, 32>, byte_idx: u32) -> u32 {
    let u32_idx = byte_idx / 4u;
    let byte_pos = byte_idx % 4u;
    
    // Unroll for all 32 possible u32 indices (this is verbose but necessary)
    // For now, let's try a different approach: use a loop with constant bounds
    // Actually, WGSL might allow this if we structure it correctly
    var result: u32 = 0u;
    
    // We need to unroll all 32 cases - this is very verbose
    // Alternative: restructure to avoid this
    // For now, let's use a pattern matching approach
    if (u32_idx == 0u) {
        result = extract_byte_from_u32(qs[0u], byte_pos);
    } else if (u32_idx == 1u) {
        result = extract_byte_from_u32(qs[1u], byte_pos);
    } else if (u32_idx == 2u) {
        result = extract_byte_from_u32(qs[2u], byte_pos);
    } else if (u32_idx == 3u) {
        result = extract_byte_from_u32(qs[3u], byte_pos);
    } else if (u32_idx == 4u) {
        result = extract_byte_from_u32(qs[4u], byte_pos);
    } else if (u32_idx == 5u) {
        result = extract_byte_from_u32(qs[5u], byte_pos);
    } else if (u32_idx == 6u) {
        result = extract_byte_from_u32(qs[6u], byte_pos);
    } else if (u32_idx == 7u) {
        result = extract_byte_from_u32(qs[7u], byte_pos);
    } else if (u32_idx == 8u) {
        result = extract_byte_from_u32(qs[8u], byte_pos);
    } else if (u32_idx == 9u) {
        result = extract_byte_from_u32(qs[9u], byte_pos);
    } else if (u32_idx == 10u) {
        result = extract_byte_from_u32(qs[10u], byte_pos);
    } else if (u32_idx == 11u) {
        result = extract_byte_from_u32(qs[11u], byte_pos);
    } else if (u32_idx == 12u) {
        result = extract_byte_from_u32(qs[12u], byte_pos);
    } else if (u32_idx == 13u) {
        result = extract_byte_from_u32(qs[13u], byte_pos);
    } else if (u32_idx == 14u) {
        result = extract_byte_from_u32(qs[14u], byte_pos);
    } else if (u32_idx == 15u) {
        result = extract_byte_from_u32(qs[15u], byte_pos);
    } else if (u32_idx == 16u) {
        result = extract_byte_from_u32(qs[16u], byte_pos);
    } else if (u32_idx == 17u) {
        result = extract_byte_from_u32(qs[17u], byte_pos);
    } else if (u32_idx == 18u) {
        result = extract_byte_from_u32(qs[18u], byte_pos);
    } else if (u32_idx == 19u) {
        result = extract_byte_from_u32(qs[19u], byte_pos);
    } else if (u32_idx == 20u) {
        result = extract_byte_from_u32(qs[20u], byte_pos);
    } else if (u32_idx == 21u) {
        result = extract_byte_from_u32(qs[21u], byte_pos);
    } else if (u32_idx == 22u) {
        result = extract_byte_from_u32(qs[22u], byte_pos);
    } else if (u32_idx == 23u) {
        result = extract_byte_from_u32(qs[23u], byte_pos);
    } else if (u32_idx == 24u) {
        result = extract_byte_from_u32(qs[24u], byte_pos);
    } else if (u32_idx == 25u) {
        result = extract_byte_from_u32(qs[25u], byte_pos);
    } else if (u32_idx == 26u) {
        result = extract_byte_from_u32(qs[26u], byte_pos);
    } else if (u32_idx == 27u) {
        result = extract_byte_from_u32(qs[27u], byte_pos);
    } else if (u32_idx == 28u) {
        result = extract_byte_from_u32(qs[28u], byte_pos);
    } else if (u32_idx == 29u) {
        result = extract_byte_from_u32(qs[29u], byte_pos);
    } else if (u32_idx == 30u) {
        result = extract_byte_from_u32(qs[30u], byte_pos);
    } else {
        result = extract_byte_from_u32(qs[31u], byte_pos);
    }
    
    return result;
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
                let q_byte = get_qs_byte(block.qs, q_offset + j);
                let x = f32(q_byte & 0xFu);
                let dequant = d1 * x - m1;
                
                let k_idx = k_start + y_offset + j;
                if (k_idx < params.k) {
                    sum += dequant * input[batch_idx * params.k + k_idx];
                }
            }
            
            // Upper nibbles (32 values)
            for (var j = 0u; j < 32u; j++) {
                let q_byte = get_qs_byte(block.qs, q_offset + j);
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
