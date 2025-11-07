// WebGPU compute shader for fused Q5_K dequantization + matrix multiplication
// Q5_K format: 256-element blocks (QK_K) with 5-bit quantization
// Structure: d (f16), ql[128] (lower 4 bits), qh[32] (upper 1 bit), scales[16]
// Each block processes 256 values in 16 groups of 16

// Block structure (176 bytes total)
struct BlockQ5K {
    ql: array<u32, 32>,      // Lower 4 bits (128 bytes, read as u32 array)
    qh: array<u32, 8>,       // Upper 1 bit (32 bytes, read as u32 array)
    scales: array<i32, 16>,  // 16 scales per block (16 bytes, read as i32 array)
    d: f32,                  // Super-block scale (converted from f16)
}

@group(0) @binding(0) var<storage, read> blocks: array<BlockQ5K>;
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

// Helper: Get scale with constant index
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

// Helper: Get byte from ql array (32 u32s = 128 bytes)
fn get_ql_byte(ql: array<u32, 32>, byte_idx: u32) -> u32 {
    let u32_idx = byte_idx / 4u;
    let byte_pos = byte_idx % 4u;
    
    var val: u32 = 0u;
    if (u32_idx == 0u) { val = ql[0u]; }
    else if (u32_idx == 1u) { val = ql[1u]; }
    else if (u32_idx == 2u) { val = ql[2u]; }
    else if (u32_idx == 3u) { val = ql[3u]; }
    else if (u32_idx == 4u) { val = ql[4u]; }
    else if (u32_idx == 5u) { val = ql[5u]; }
    else if (u32_idx == 6u) { val = ql[6u]; }
    else if (u32_idx == 7u) { val = ql[7u]; }
    else if (u32_idx == 8u) { val = ql[8u]; }
    else if (u32_idx == 9u) { val = ql[9u]; }
    else if (u32_idx == 10u) { val = ql[10u]; }
    else if (u32_idx == 11u) { val = ql[11u]; }
    else if (u32_idx == 12u) { val = ql[12u]; }
    else if (u32_idx == 13u) { val = ql[13u]; }
    else if (u32_idx == 14u) { val = ql[14u]; }
    else if (u32_idx == 15u) { val = ql[15u]; }
    else if (u32_idx == 16u) { val = ql[16u]; }
    else if (u32_idx == 17u) { val = ql[17u]; }
    else if (u32_idx == 18u) { val = ql[18u]; }
    else if (u32_idx == 19u) { val = ql[19u]; }
    else if (u32_idx == 20u) { val = ql[20u]; }
    else if (u32_idx == 21u) { val = ql[21u]; }
    else if (u32_idx == 22u) { val = ql[22u]; }
    else if (u32_idx == 23u) { val = ql[23u]; }
    else if (u32_idx == 24u) { val = ql[24u]; }
    else if (u32_idx == 25u) { val = ql[25u]; }
    else if (u32_idx == 26u) { val = ql[26u]; }
    else if (u32_idx == 27u) { val = ql[27u]; }
    else if (u32_idx == 28u) { val = ql[28u]; }
    else if (u32_idx == 29u) { val = ql[29u]; }
    else if (u32_idx == 30u) { val = ql[30u]; }
    else { val = ql[31u]; }
    
    if (byte_pos == 0u) { return val & 0xFFu; }
    else if (byte_pos == 1u) { return (val >> 8u) & 0xFFu; }
    else if (byte_pos == 2u) { return (val >> 16u) & 0xFFu; }
    else { return (val >> 24u) & 0xFFu; }
}

// Helper: Get byte from qh array (8 u32s = 32 bytes)
fn get_qh_byte(qh: array<u32, 8>, byte_idx: u32) -> u32 {
    let u32_idx = byte_idx / 4u;
    let byte_pos = byte_idx % 4u;
    
    var val: u32 = 0u;
    if (u32_idx == 0u) { val = qh[0u]; }
    else if (u32_idx == 1u) { val = qh[1u]; }
    else if (u32_idx == 2u) { val = qh[2u]; }
    else if (u32_idx == 3u) { val = qh[3u]; }
    else if (u32_idx == 4u) { val = qh[4u]; }
    else if (u32_idx == 5u) { val = qh[5u]; }
    else if (u32_idx == 6u) { val = qh[6u]; }
    else { val = qh[7u]; }
    
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
        
        // Process 256 elements in 16 groups of 16
        for (var l = 0u; l < 16u; l++) {
            let scale = get_scale(block.scales, l);
            
            // Process 16 values in this group
            for (var k = 0u; k < 16u; k++) {
                let ql_idx = l * 16u + k;
                
                if (ql_idx >= 256u) {
                    break;
                }
                
                // Extract ql value (ql is stored as bytes, each byte has 2 values)
                let ql_byte_idx = ql_idx / 2u;
                let ql_byte = get_ql_byte(block.ql, ql_byte_idx);
                
                // Extract lower or upper nibble
                var ql_val: u32 = 0u;
                if (ql_idx % 2u == 0u) {
                    ql_val = ql_byte & 0xFu; // Lower nibble
                } else {
                    ql_val = ql_byte >> 4u; // Upper nibble
                }
                
                // Extract qh value (qh is stored as 8 values per byte)
                // qh_idx = l * 2 + (k / 8)
                let qh_idx = l * 2u + (k / 8u);
                let qh_byte = get_qh_byte(block.qh, qh_idx);
                
                // qh_bit = (k % 8) / 4
                let qh_bit = (k % 8u) / 4u;
                // Extract 4 bits from qh: shift by (qh_bit * 4) and mask with 0xF
                let qh_val = (qh_byte >> (qh_bit * 4u)) & 0xFu;
                
                // Combine: ql_val | (qh_val << 4)
                let quant_val_u32 = ql_val | (qh_val << 4u);
                let quant_val = i32(quant_val_u32);
                
                // Dequantize: d * scale * quant_val
                let dequant = d * scale * f32(quant_val);
                
                let k_idx = k_start + ql_idx;
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
