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
            let scale_u32_idx = scale_byte_idx / 4u;
            let scale_bit_offset = ((scale_byte_idx % 4u) * 8u);
            let scale_byte = (block.scales[scale_u32_idx] >> scale_bit_offset) & 0xFFu;
            
            let scale_val: u32;
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
                let quant_u32_idx = idx / 4u;
                let quant_bit_offset = ((idx % 4u) * 8u);
                let quant_byte = (block.quants[quant_u32_idx] >> quant_bit_offset) & 0xFFu;
                
                // Sign extend from 8-bit to i32
                let quant_val: i32;
                if ((quant_byte & 0x80u) != 0u) {
                    quant_val = i32(quant_byte) | 0xFFFFFF00i32; // Sign extend
                } else {
                    quant_val = i32(quant_byte);
                }
                
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

