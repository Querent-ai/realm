// WebGPU compute shader for fused Q6_K dequantization + matrix multiplication
// Q6_K format: 256-element blocks (QK_K) with 6-bit quantization
// Structure: d (f16), ql[128] (lower 4 bits), qh[64] (upper 2 bits), scales[16]
// Each block processes 256 values in 32 groups of 8

// Block structure (210 bytes total)
struct BlockQ6K {
    ql: array<u32, 32>,      // Lower 4 bits (128 bytes, read as u32 array)
    qh: array<u32, 16>,      // Upper 2 bits (64 bytes, read as u32 array)
    scales: array<i32, 16>,  // 16 scales per block (16 bytes, read as i32 array)
    d: f32,                  // Super-block scale (converted from f16)
}

@group(0) @binding(0) var<storage, read> blocks: array<BlockQ6K>;
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

// Helper: Get scale with constant index (unroll for 0-15)
fn get_scale(scales: array<i32, 16>, idx: u32) -> f32 {
    if (idx == 0u) { return f32(scales[0u]); }
    else if (idx == 1u) { return f32(scales[1u]); }
    else if (idx == 2u) { return f32(scales[2u]); }
    else if (idx == 3u) { return f32(scales[3u]); }
    else if (idx == 4u) { return f32(scales[4u]); }
    else if (idx == 5u) { return f32(scales[5u]); }
    else if (idx == 6u) { return f32(scales[6u]); }
    else if (idx == 7u) { return f32(scales[7u]); }
    else if (idx == 8u) { return f32(scales[8u]); }
    else if (idx == 9u) { return f32(scales[9u]); }
    else if (idx == 10u) { return f32(scales[10u]); }
    else if (idx == 11u) { return f32(scales[11u]); }
    else if (idx == 12u) { return f32(scales[12u]); }
    else if (idx == 13u) { return f32(scales[13u]); }
    else if (idx == 14u) { return f32(scales[14u]); }
    else { return f32(scales[15u]); }
}

// Helper: Extract byte from u32 array (for ql/qh)
fn get_byte_from_u32_array(arr: array<u32, 32>, byte_idx: u32) -> u32 {
    let u32_idx = byte_idx / 4u;
    let byte_pos = byte_idx % 4u;
    
    // Unroll for u32 index (0-31) - this is very verbose but necessary
    // For now, use a pattern: extract byte from the u32
    var val: u32 = 0u;
    if (u32_idx == 0u) { val = arr[0u]; }
    else if (u32_idx == 1u) { val = arr[1u]; }
    else if (u32_idx == 2u) { val = arr[2u]; }
    else if (u32_idx == 3u) { val = arr[3u]; }
    else if (u32_idx == 4u) { val = arr[4u]; }
    else if (u32_idx == 5u) { val = arr[5u]; }
    else if (u32_idx == 6u) { val = arr[6u]; }
    else if (u32_idx == 7u) { val = arr[7u]; }
    else if (u32_idx == 8u) { val = arr[8u]; }
    else if (u32_idx == 9u) { val = arr[9u]; }
    else if (u32_idx == 10u) { val = arr[10u]; }
    else if (u32_idx == 11u) { val = arr[11u]; }
    else if (u32_idx == 12u) { val = arr[12u]; }
    else if (u32_idx == 13u) { val = arr[13u]; }
    else if (u32_idx == 14u) { val = arr[14u]; }
    else if (u32_idx == 15u) { val = arr[15u]; }
    else if (u32_idx == 16u) { val = arr[16u]; }
    else if (u32_idx == 17u) { val = arr[17u]; }
    else if (u32_idx == 18u) { val = arr[18u]; }
    else if (u32_idx == 19u) { val = arr[19u]; }
    else if (u32_idx == 20u) { val = arr[20u]; }
    else if (u32_idx == 21u) { val = arr[21u]; }
    else if (u32_idx == 22u) { val = arr[22u]; }
    else if (u32_idx == 23u) { val = arr[23u]; }
    else if (u32_idx == 24u) { val = arr[24u]; }
    else if (u32_idx == 25u) { val = arr[25u]; }
    else if (u32_idx == 26u) { val = arr[26u]; }
    else if (u32_idx == 27u) { val = arr[27u]; }
    else if (u32_idx == 28u) { val = arr[28u]; }
    else if (u32_idx == 29u) { val = arr[29u]; }
    else if (u32_idx == 30u) { val = arr[30u]; }
    else { val = arr[31u]; }
    
    // Extract byte position
    if (byte_pos == 0u) { return val & 0xFFu; }
    else if (byte_pos == 1u) { return (val >> 8u) & 0xFFu; }
    else if (byte_pos == 2u) { return (val >> 16u) & 0xFFu; }
    else { return (val >> 24u) & 0xFFu; }
}

// Helper for qh array (16 u32s)
fn get_byte_from_qh_array(arr: array<u32, 16>, byte_idx: u32) -> u32 {
    let u32_idx = byte_idx / 4u;
    let byte_pos = byte_idx % 4u;
    
    var val: u32 = 0u;
    if (u32_idx == 0u) { val = arr[0u]; }
    else if (u32_idx == 1u) { val = arr[1u]; }
    else if (u32_idx == 2u) { val = arr[2u]; }
    else if (u32_idx == 3u) { val = arr[3u]; }
    else if (u32_idx == 4u) { val = arr[4u]; }
    else if (u32_idx == 5u) { val = arr[5u]; }
    else if (u32_idx == 6u) { val = arr[6u]; }
    else if (u32_idx == 7u) { val = arr[7u]; }
    else if (u32_idx == 8u) { val = arr[8u]; }
    else if (u32_idx == 9u) { val = arr[9u]; }
    else if (u32_idx == 10u) { val = arr[10u]; }
    else if (u32_idx == 11u) { val = arr[11u]; }
    else if (u32_idx == 12u) { val = arr[12u]; }
    else if (u32_idx == 13u) { val = arr[13u]; }
    else if (u32_idx == 14u) { val = arr[14u]; }
    else { val = arr[15u]; }
    
    if (byte_pos == 0u) { return val & 0xFFu; }
    else if (byte_pos == 1u) { return (val >> 8u) & 0xFFu; }
    else if (byte_pos == 2u) { return (val >> 16u) & 0xFFu; }
    else { return (val >> 24u) & 0xFFu; }
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
        
        // Process first half (0-127): use scales[is], scales[is+2], scales[is+4], scales[is+6]
        for (var l = 0u; l < 32u; l++) {
            let is = l / 16u; // 0 or 1 (scale base index)
            
            // Get ql bytes using helper
            let ql_l_byte = get_byte_from_u32_array(block.ql, l);
            let ql_l32_byte = get_byte_from_u32_array(block.ql, l + 32u);
            
            // Get qh byte
            let qh_l_byte = get_byte_from_qh_array(block.qh, l);
            
            // Extract 4 quantized values (matches CPU: as i8 - 32)
            let q1 = i32((ql_l_byte & 0xFu) | (((qh_l_byte >> 0u) & 3u) << 4u)) - 32i;
            let q2 = i32((ql_l32_byte & 0xFu) | (((qh_l_byte >> 2u) & 3u) << 4u)) - 32i;
            let q3 = i32((ql_l_byte >> 4u) | (((qh_l_byte >> 4u) & 3u) << 4u)) - 32i;
            let q4 = i32((ql_l32_byte >> 4u) | (((qh_l_byte >> 6u) & 3u) << 4u)) - 32i;
            
            // Dequantize with correct scales using helper
            let k_idx1 = k_start + l;
            let k_idx2 = k_start + l + 32u;
            let k_idx3 = k_start + l + 64u;
            let k_idx4 = k_start + l + 96u;
            
            if (k_idx1 < params.k) {
                sum += d * get_scale(block.scales, is) * f32(q1) * input[batch_idx * params.k + k_idx1];
            }
            if (k_idx2 < params.k) {
                sum += d * get_scale(block.scales, is + 2u) * f32(q2) * input[batch_idx * params.k + k_idx2];
            }
            if (k_idx3 < params.k) {
                sum += d * get_scale(block.scales, is + 4u) * f32(q3) * input[batch_idx * params.k + k_idx3];
            }
            if (k_idx4 < params.k) {
                sum += d * get_scale(block.scales, is + 6u) * f32(q4) * input[batch_idx * params.k + k_idx4];
            }
        }
        
        // Process second half (128-255): use scales[8+is], scales[8+is+2], scales[8+is+4], scales[8+is+6]
        for (var l = 0u; l < 32u; l++) {
            let is = l / 16u; // 0 or 1 (scale base index)
            let sc_offset = 8u; // Offset into scales array for second half
            
            // Get ql bytes
            let ql_l64_byte = get_byte_from_u32_array(block.ql, l + 64u);
            let ql_l96_byte = get_byte_from_u32_array(block.ql, l + 96u);
            
            // Get qh byte
            let qh_l32_byte = get_byte_from_qh_array(block.qh, l + 32u);
            
            // Extract 4 quantized values (as i8 - 32)
            let q1 = i32((ql_l64_byte & 0xFu) | (((qh_l32_byte >> 0u) & 3u) << 4u)) - 32i;
            let q2 = i32((ql_l96_byte & 0xFu) | (((qh_l32_byte >> 2u) & 3u) << 4u)) - 32i;
            let q3 = i32((ql_l64_byte >> 4u) | (((qh_l32_byte >> 4u) & 3u) << 4u)) - 32i;
            let q4 = i32((ql_l96_byte >> 4u) | (((qh_l32_byte >> 6u) & 3u) << 4u)) - 32i;
            
            // Dequantize with correct scales
            let k_idx1 = k_start + l + 128u;
            let k_idx2 = k_start + l + 160u;
            let k_idx3 = k_start + l + 192u;
            let k_idx4 = k_start + l + 224u;
            
            if (k_idx1 < params.k) {
                sum += d * get_scale(block.scales, sc_offset + is) * f32(q1) * input[batch_idx * params.k + k_idx1];
            }
            if (k_idx2 < params.k) {
                sum += d * get_scale(block.scales, sc_offset + is + 2u) * f32(q2) * input[batch_idx * params.k + k_idx2];
            }
            if (k_idx3 < params.k) {
                sum += d * get_scale(block.scales, sc_offset + is + 4u) * f32(q3) * input[batch_idx * params.k + k_idx3];
            }
            if (k_idx4 < params.k) {
                sum += d * get_scale(block.scales, sc_offset + is + 6u) * f32(q4) * input[batch_idx * params.k + k_idx4];
            }
        }
    }
    
    // Write result
    let output_idx = batch_idx * params.n + out_idx;
    output[output_idx] = sum;
}
