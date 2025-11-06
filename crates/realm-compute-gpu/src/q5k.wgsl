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
            let is = l; // Scale index (0..15)
            let scale = f32(block.scales[is]);
            
            // Process 16 values in this group
            for (var k = 0u; k < 16u; k++) {
                let ql_idx = l * 16u + k;
                
                if (ql_idx >= 256u) {
                    break;
                }
                
                // Extract ql value (ql is stored as bytes, each byte has 2 values)
                let ql_byte_idx = ql_idx / 2u;
                let ql_u32_idx = ql_byte_idx / 4u;
                let ql_bit_offset = ((ql_byte_idx % 4u) * 8u);
                let ql_byte = (block.ql[ql_u32_idx] >> ql_bit_offset) & 0xFFu;
                
                // ql_val is the full byte value (will be combined with qh)
                let ql_val = i32(ql_byte);
                
                // Extract qh value (qh is stored as 8 values per byte)
                // qh_idx = l * 2 + (k / 8)
                let qh_idx = l * 2u + (k / 8u);
                let qh_byte_idx = qh_idx / 4u;
                let qh_bit_offset = ((qh_idx % 4u) * 8u);
                let qh_byte = (block.qh[qh_byte_idx] >> qh_bit_offset) & 0xFFu;
                
                // qh_bit = (k % 8) / 4
                let qh_bit = (k % 8u) / 4u;
                // Extract 4 bits from qh: shift by (qh_bit * 4) and mask with 0xF
                let qh_val = (qh_byte >> (qh_bit * 4u)) & 0xFu;
                
                // Combine: ql_val | (qh_val << 4)
                let quant_val = ql_val | (i32(qh_val) << 4i);
                
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

