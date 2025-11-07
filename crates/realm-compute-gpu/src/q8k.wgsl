// WebGPU compute shader for fused Q8_K dequantization + matrix multiplication
// Q8_K format: 256-element blocks (QK_K) with 8-bit quantization
// Structure: d (f16), dmin (f16), quants[256] (8-bit), scales[32] (4 bits each)
// Each block processes 256 values in 32 groups of 8

// Block structure (322 bytes total)
struct BlockQ8K {
    quants: array<i32, 64>,  // 8-bit quantized values (256 bytes, read as i32 array)
    scales: array<u32, 8>,    // 32 scales per block, 4 bits each (32 bytes, read as u32 array)
    d: f32,                   // Super-block scale (converted from f16)
    dmin: f32,                // Super-block min scale (converted from f16)
}

@group(0) @binding(0) var<storage, read> blocks: array<BlockQ8K>;
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

// Helper: Get scale byte from scales array (8 u32s = 32 bytes)
fn get_scale_byte(scales: array<u32, 8>, byte_idx: u32) -> u32 {
    let u32_idx = byte_idx / 4u;
    let byte_pos = byte_idx % 4u;
    
    var val: u32 = 0u;
    if (u32_idx == 0u) { val = scales[0u]; }
    else if (u32_idx == 1u) { val = scales[1u]; }
    else if (u32_idx == 2u) { val = scales[2u]; }
    else if (u32_idx == 3u) { val = scales[3u]; }
    else if (u32_idx == 4u) { val = scales[4u]; }
    else if (u32_idx == 5u) { val = scales[5u]; }
    else if (u32_idx == 6u) { val = scales[6u]; }
    else { val = scales[7u]; }
    
    if (byte_pos == 0u) { return val & 0xFFu; }
    else if (byte_pos == 1u) { return (val >> 8u) & 0xFFu; }
    else if (byte_pos == 2u) { return (val >> 16u) & 0xFFu; }
    else { return (val >> 24u) & 0xFFu; }
}

// Helper: Get quant byte from quants array (64 i32s = 256 bytes)
fn get_quant_byte(quants: array<i32, 64>, byte_idx: u32) -> i32 {
    let u32_idx = byte_idx / 4u;
    let byte_pos = byte_idx % 4u;
    
    // Unroll for 64 possible indices - this is very verbose
    // For now, use a pattern with if-else chain
    var val: i32 = 0i;
    if (u32_idx < 64u) {
        // We need to unroll all 64 cases - this is too verbose
        // Alternative: use a loop with constant bounds, but WGSL doesn't allow that either
        // Let's use a switch-like structure
        if (u32_idx == 0u) { val = quants[0u]; }
        else if (u32_idx == 1u) { val = quants[1u]; }
        else if (u32_idx == 2u) { val = quants[2u]; }
        else if (u32_idx == 3u) { val = quants[3u]; }
        else if (u32_idx == 4u) { val = quants[4u]; }
        else if (u32_idx == 5u) { val = quants[5u]; }
        else if (u32_idx == 6u) { val = quants[6u]; }
        else if (u32_idx == 7u) { val = quants[7u]; }
        else if (u32_idx == 8u) { val = quants[8u]; }
        else if (u32_idx == 9u) { val = quants[9u]; }
        else if (u32_idx == 10u) { val = quants[10u]; }
        else if (u32_idx == 11u) { val = quants[11u]; }
        else if (u32_idx == 12u) { val = quants[12u]; }
        else if (u32_idx == 13u) { val = quants[13u]; }
        else if (u32_idx == 14u) { val = quants[14u]; }
        else if (u32_idx == 15u) { val = quants[15u]; }
        else if (u32_idx == 16u) { val = quants[16u]; }
        else if (u32_idx == 17u) { val = quants[17u]; }
        else if (u32_idx == 18u) { val = quants[18u]; }
        else if (u32_idx == 19u) { val = quants[19u]; }
        else if (u32_idx == 20u) { val = quants[20u]; }
        else if (u32_idx == 21u) { val = quants[21u]; }
        else if (u32_idx == 22u) { val = quants[22u]; }
        else if (u32_idx == 23u) { val = quants[23u]; }
        else if (u32_idx == 24u) { val = quants[24u]; }
        else if (u32_idx == 25u) { val = quants[25u]; }
        else if (u32_idx == 26u) { val = quants[26u]; }
        else if (u32_idx == 27u) { val = quants[27u]; }
        else if (u32_idx == 28u) { val = quants[28u]; }
        else if (u32_idx == 29u) { val = quants[29u]; }
        else if (u32_idx == 30u) { val = quants[30u]; }
        else if (u32_idx == 31u) { val = quants[31u]; }
        else if (u32_idx == 32u) { val = quants[32u]; }
        else if (u32_idx == 33u) { val = quants[33u]; }
        else if (u32_idx == 34u) { val = quants[34u]; }
        else if (u32_idx == 35u) { val = quants[35u]; }
        else if (u32_idx == 36u) { val = quants[36u]; }
        else if (u32_idx == 37u) { val = quants[37u]; }
        else if (u32_idx == 38u) { val = quants[38u]; }
        else if (u32_idx == 39u) { val = quants[39u]; }
        else if (u32_idx == 40u) { val = quants[40u]; }
        else if (u32_idx == 41u) { val = quants[41u]; }
        else if (u32_idx == 42u) { val = quants[42u]; }
        else if (u32_idx == 43u) { val = quants[43u]; }
        else if (u32_idx == 44u) { val = quants[44u]; }
        else if (u32_idx == 45u) { val = quants[45u]; }
        else if (u32_idx == 46u) { val = quants[46u]; }
        else if (u32_idx == 47u) { val = quants[47u]; }
        else if (u32_idx == 48u) { val = quants[48u]; }
        else if (u32_idx == 49u) { val = quants[49u]; }
        else if (u32_idx == 50u) { val = quants[50u]; }
        else if (u32_idx == 51u) { val = quants[51u]; }
        else if (u32_idx == 52u) { val = quants[52u]; }
        else if (u32_idx == 53u) { val = quants[53u]; }
        else if (u32_idx == 54u) { val = quants[54u]; }
        else if (u32_idx == 55u) { val = quants[55u]; }
        else if (u32_idx == 56u) { val = quants[56u]; }
        else if (u32_idx == 57u) { val = quants[57u]; }
        else if (u32_idx == 58u) { val = quants[58u]; }
        else if (u32_idx == 59u) { val = quants[59u]; }
        else if (u32_idx == 60u) { val = quants[60u]; }
        else if (u32_idx == 61u) { val = quants[61u]; }
        else if (u32_idx == 62u) { val = quants[62u]; }
        else { val = quants[63u]; }
    }
    
    // Extract byte and sign extend
    var byte_val: u32 = 0u;
    if (byte_pos == 0u) { byte_val = u32(val) & 0xFFu; }
    else if (byte_pos == 1u) { byte_val = (u32(val) >> 8u) & 0xFFu; }
    else if (byte_pos == 2u) { byte_val = (u32(val) >> 16u) & 0xFFu; }
    else { byte_val = (u32(val) >> 24u) & 0xFFu; }
    
    // Sign extend from 8-bit to i32
    if ((byte_val & 0x80u) != 0u) {
        return i32(byte_val) - 256i;
    } else {
        return i32(byte_val);
    }
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
        
        // Process 256 elements in 32 groups of 8
        for (var l = 0u; l < 32u; l++) {
            let is = l; // Scale index (0..31)
            
            // Extract scale for this group (4 bits per scale, 2 per byte)
            let scale_byte_idx = is / 2u;
            let scale_byte = get_scale_byte(block.scales, scale_byte_idx);
            
            var scale_val: u32 = 0u;
            if (is % 2u == 0u) {
                scale_val = scale_byte & 0xFu; // Lower 4 bits
            } else {
                scale_val = scale_byte >> 4u; // Upper 4 bits
            }
            
            // Process 8 values in this group
            for (var k = 0u; k < 8u; k++) {
                let idx = l * 8u + k;
                
                if (idx >= 256u) {
                    break;
                }
                
                // Get quantized value (8-bit signed)
                let quant_val = get_quant_byte(block.quants, idx);
                
                // Dequantize: d * scale * quant_val + min
                let dequant = d * f32(scale_val) * f32(quant_val) + min;
                
                let k_idx = k_start + idx;
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
