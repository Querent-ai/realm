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
            
            // Get ql bytes (ql is stored as bytes, 2 values per byte)
            let ql_l_byte_idx = l;
            let ql_l_u32_idx = ql_l_byte_idx / 4u;
            let ql_l_bit_offset = ((ql_l_byte_idx % 4u) * 8u);
            let ql_l_byte = (block.ql[ql_l_u32_idx] >> ql_l_bit_offset) & 0xFFu;
            
            let ql_l32_byte_idx = l + 32u;
            let ql_l32_u32_idx = ql_l32_byte_idx / 4u;
            let ql_l32_bit_offset = ((ql_l32_byte_idx % 4u) * 8u);
            let ql_l32_byte = (block.ql[ql_l32_u32_idx] >> ql_l32_bit_offset) & 0xFFu;
            
            // Get qh byte (qh is stored as bytes, 4 values per byte)
            let qh_l_byte_idx = l;
            let qh_l_u32_idx = qh_l_byte_idx / 4u;
            let qh_l_bit_offset = ((qh_l_byte_idx % 4u) * 8u);
            let qh_l_byte = (block.qh[qh_l_u32_idx] >> qh_l_bit_offset) & 0xFFu;
            
            // Extract 4 quantized values (matches CPU: as i8 - 32)
            let q1 = i32((ql_l_byte & 0xFu) | (((qh_l_byte >> 0u) & 3u) << 4u)) - 32i;
            let q2 = i32((ql_l32_byte & 0xFu) | (((qh_l_byte >> 2u) & 3u) << 4u)) - 32i;
            let q3 = i32((ql_l_byte >> 4u) | (((qh_l_byte >> 4u) & 3u) << 4u)) - 32i;
            let q4 = i32((ql_l32_byte >> 4u) | (((qh_l_byte >> 6u) & 3u) << 4u)) - 32i;
            
            // Dequantize with correct scales (scales[is], scales[is+2], scales[is+4], scales[is+6])
            let k_idx1 = k_start + l;
            let k_idx2 = k_start + l + 32u;
            let k_idx3 = k_start + l + 64u;
            let k_idx4 = k_start + l + 96u;
            
            if (k_idx1 < params.k) {
                sum += d * f32(block.scales[is]) * f32(q1) * input[batch_idx * params.k + k_idx1];
            }
            if (k_idx2 < params.k) {
                sum += d * f32(block.scales[is + 2u]) * f32(q2) * input[batch_idx * params.k + k_idx2];
            }
            if (k_idx3 < params.k) {
                sum += d * f32(block.scales[is + 4u]) * f32(q3) * input[batch_idx * params.k + k_idx3];
            }
            if (k_idx4 < params.k) {
                sum += d * f32(block.scales[is + 6u]) * f32(q4) * input[batch_idx * params.k + k_idx4];
            }
        }
        
        // Process second half (128-255): use scales[8+is], scales[8+is+2], scales[8+is+4], scales[8+is+6]
        for (var l = 0u; l < 32u; l++) {
            let is = l / 16u; // 0 or 1 (scale base index)
            let sc_offset = 8u; // Offset into scales array for second half
            
            // Get ql bytes
            let ql_l64_byte_idx = l + 64u;
            let ql_l64_u32_idx = ql_l64_byte_idx / 4u;
            let ql_l64_bit_offset = ((ql_l64_byte_idx % 4u) * 8u);
            let ql_l64_byte = (block.ql[ql_l64_u32_idx] >> ql_l64_bit_offset) & 0xFFu;
            
            let ql_l96_byte_idx = l + 96u;
            let ql_l96_u32_idx = ql_l96_byte_idx / 4u;
            let ql_l96_bit_offset = ((ql_l96_byte_idx % 4u) * 8u);
            let ql_l96_byte = (block.ql[ql_l96_u32_idx] >> ql_l96_bit_offset) & 0xFFu;
            
            // Get qh byte
            let qh_l32_byte_idx = l + 32u;
            let qh_l32_u32_idx = qh_l32_byte_idx / 4u;
            let qh_l32_bit_offset = ((qh_l32_byte_idx % 4u) * 8u);
            let qh_l32_byte = (block.qh[qh_l32_u32_idx] >> qh_l32_bit_offset) & 0xFFu;
            
            // Extract 4 quantized values (as i8 - 32)
            let q1 = i32((ql_l64_byte & 0xFu) | (((qh_l32_byte >> 0u) & 3u) << 4u)) - 32i;
            let q2 = i32((ql_l96_byte & 0xFu) | (((qh_l32_byte >> 2u) & 3u) << 4u)) - 32i;
            let q3 = i32((ql_l64_byte >> 4u) | (((qh_l32_byte >> 4u) & 3u) << 4u)) - 32i;
            let q4 = i32((ql_l96_byte >> 4u) | (((qh_l32_byte >> 6u) & 3u) << 4u)) - 32i;
            
            // Dequantize with correct scales (sc_offset + is, sc_offset + is + 2, etc.)
            let k_idx1 = k_start + l + 128u;
            let k_idx2 = k_start + l + 160u;
            let k_idx3 = k_start + l + 192u;
            let k_idx4 = k_start + l + 224u;
            
            if (k_idx1 < params.k) {
                sum += d * f32(block.scales[sc_offset + is]) * f32(q1) * input[batch_idx * params.k + k_idx1];
            }
            if (k_idx2 < params.k) {
                sum += d * f32(block.scales[sc_offset + is + 2u]) * f32(q2) * input[batch_idx * params.k + k_idx2];
            }
            if (k_idx3 < params.k) {
                sum += d * f32(block.scales[sc_offset + is + 4u]) * f32(q3) * input[batch_idx * params.k + k_idx3];
            }
            if (k_idx4 < params.k) {
                sum += d * f32(block.scales[sc_offset + is + 6u]) * f32(q4) * input[batch_idx * params.k + k_idx4];
            }
        }
    }
    
    // Write result
    let output_idx = batch_idx * params.n + out_idx;
    output[output_idx] = sum;
}

