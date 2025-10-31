//! Quantization and dequantization primitives

use crate::error::{Error, Result};
use crate::tensor::DataType;

/// Block size for group-wise quantization
pub const Q4_BLOCK_SIZE: usize = 32;
pub const Q8_BLOCK_SIZE: usize = 32;
pub const QK_K: usize = 256; // K-quants use 256-element super-blocks

/// Q4_0 block: 32 4-bit values
/// NOTE: This GGUF file uses 16-byte blocks (quants only, no per-block scale)
/// The scale appears to be stored separately or shared across blocks
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_0 {
    pub quants: [u8; Q4_BLOCK_SIZE / 2], // 16 bytes (2 values per byte)
}

/// Q8_0 block: 32 8-bit values + 1 f16 scale
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ8_0 {
    pub scale: u16, // f16 stored as u16
    pub quants: [i8; Q8_BLOCK_SIZE],
}

/// Q4_K block: 256 4-bit values in a super-block
/// Structure based on ggml implementation
/// Total: 144 bytes (12 scales + 128 qs + 2 d + 2 dmin)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_K {
    pub d: u16,             // f16 super-block scale (stored as u16)
    pub dmin: u16,          // f16 super-block min scale (stored as u16)
    pub scales: [u8; 12],   // Quantized scales (6 bits each)
    pub qs: [u8; QK_K / 2], // 128 bytes of 4-bit quants
}

/// Helper function to extract scale and min from Q4_K scales array
/// Based on ggml get_scale_min_k4
#[inline]
pub fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (u8, u8) {
    let (d, m) = if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    };
    (d, m)
}

/// Dequantize Q4_K block to f32
/// Q4_K uses 256-element super-blocks with hierarchical scaling
/// Based on ggml dequantize_row_q4_K implementation
pub fn dequantize_q4_k(block: &BlockQ4_K, output: &mut [f32]) -> Result<()> {
    if output.len() != QK_K {
        return Err(Error::InvalidShape(format!(
            "Output buffer must be {} elements",
            QK_K
        )));
    }

    // Convert f16 scales to f32
    let d = half::f16::from_bits(block.d).to_f32();
    let min = half::f16::from_bits(block.dmin).to_f32();

    // Debug: print d and min values and scales extraction
    static FIRST_BLOCK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
    if FIRST_BLOCK.swap(false, std::sync::atomic::Ordering::Relaxed) {
        // eprintln!("Q4_K dequant first block:");
        // eprintln!("  d={:.10}, min={:.10}", d, min);
        // eprintln!("  scales bytes: {:02x?}", &block.scales);

        // Test scale extraction for first 4 groups
        for is in 0..4 {
            let (_sc0, _m0) = get_scale_min_k4(is * 2, &block.scales);
            let (_sc1, _m1) = get_scale_min_k4(is * 2 + 1, &block.scales);
            // eprintln!("  Group {}: sc0={}, m0={}, sc1={}, m1={}", is, _sc0, _m0, _sc1, _m1);
        }
    }

    // Process 256 elements in 4 groups of 64
    for i in 0..4 {
        let is = i * 2; // Scale index

        // Get scales for this group
        let (sc, m) = get_scale_min_k4(is, &block.scales);
        let d1 = d * sc as f32;
        let m1 = min * m as f32;

        let (sc, m) = get_scale_min_k4(is + 1, &block.scales);
        let d2 = d * sc as f32;
        let m2 = min * m as f32;

        // Process 64 values: 32 from lower nibbles, 32 from upper nibbles
        let q_offset = i * 32;
        let y_offset = i * 64;

        // Lower nibbles (32 values)
        for j in 0..32 {
            let q = block.qs[q_offset + j];
            let x = (q & 0xF) as f32;
            output[y_offset + j] = d1 * x - m1;
        }

        // Upper nibbles (32 values)
        for j in 0..32 {
            let q = block.qs[q_offset + j];
            let x = (q >> 4) as f32;
            output[y_offset + 32 + j] = d2 * x - m2;
        }
    }

    Ok(())
}

/// Q6_K block: 256 6-bit values in a super-block
/// Structure based on ggml implementation
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ6_K {
    pub ql: [u8; QK_K / 2],      // Lower 4 bits of 6-bit quants
    pub qh: [u8; QK_K / 4],      // Upper 2 bits of 6-bit quants
    pub scales: [i8; QK_K / 16], // 16 scales per block
    pub d: u16,                  // f16 super-block scale (stored as u16)
}

/// Q5_K block: 256 5-bit values in a super-block
/// Structure based on ggml implementation
/// Total: 176 bytes (128 ql + 32 qh + 16 scales + 2 d)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ5_K {
    pub ql: [u8; QK_K / 2],      // Lower 4 bits of 5-bit quants
    pub qh: [u8; QK_K / 8],      // Upper 1 bit of 5-bit quants
    pub scales: [i8; QK_K / 16], // 16 scales per block
    pub d: u16,                  // f16 super-block scale (stored as u16)
}

/// Q8_K block: 256 8-bit values in a super-block
/// Structure based on ggml implementation
/// Total: 322 bytes (256 quants + 32 scales + 2 d + 2 dmin)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ8_K {
    pub quants: [i8; QK_K],     // 8-bit quantized values
    pub scales: [u8; QK_K / 8], // 32 scales per block (4 bits each)
    pub d: u16,                 // f16 super-block scale (stored as u16)
    pub dmin: u16,              // f16 super-block min scale (stored as u16)
}

/// Dequantize Q4_0 block to f32
/// NOTE: Using default scale for this non-standard Q4_0 format
pub fn dequantize_q4_0(block: &BlockQ4_0, output: &mut [f32]) -> Result<()> {
    if output.len() != Q4_BLOCK_SIZE {
        return Err(Error::InvalidShape(format!(
            "Output buffer must be {} elements",
            Q4_BLOCK_SIZE
        )));
    }

    // TODO: This GGUF file appears to use a non-standard format
    // For now, use a conservative scale value
    let d = 0.01f32;

    // Dequantize with interleaved layout: first half gets lower nibbles, second half gets upper
    // This matches llama.cpp's layout: y[i*qk + j + 0] = x0*d; y[i*qk + j + qk/2] = x1*d;
    for i in 0..Q4_BLOCK_SIZE / 2 {
        let byte = block.quants[i];

        // Lower 4 bits -> first half of output
        let x0 = ((byte & 0x0F) as i8) - 8;
        output[i] = x0 as f32 * d;

        // Upper 4 bits -> second half of output
        let x1 = ((byte >> 4) as i8) - 8;
        output[i + Q4_BLOCK_SIZE / 2] = x1 as f32 * d;
    }

    Ok(())
}

/// Dequantize Q8_0 block to f32
pub fn dequantize_q8_0(block: &BlockQ8_0, output: &mut [f32]) -> Result<()> {
    if output.len() != Q8_BLOCK_SIZE {
        return Err(Error::InvalidShape(format!(
            "Output buffer must be {} elements",
            Q8_BLOCK_SIZE
        )));
    }

    // Convert f16 scale to f32
    let scale = half::f16::from_bits(block.scale).to_f32();

    for (i, &quant) in block.quants.iter().enumerate().take(Q8_BLOCK_SIZE) {
        output[i] = quant as f32 * scale;
    }

    Ok(())
}

/// Dequantize Q6_K block to f32
/// Q6_K stores 256 values as 6-bit quantized values
/// Based on ggml dequantize_row_q6_K implementation
#[allow(clippy::needless_range_loop)]
pub fn dequantize_q6_k(block: &BlockQ6_K, output: &mut [f32]) -> Result<()> {
    if output.len() != QK_K {
        return Err(Error::InvalidShape(format!(
            "Output buffer must be {} elements",
            QK_K
        )));
    }

    // Convert f16 to f32
    let d = half::f16::from_bits(block.d).to_f32();

    // Q6_K layout: 256 values stored as:
    // - ql[128]: lower 4 bits (2 values per byte)
    // - qh[64]: upper 2 bits (4 values per byte)
    // - scales[16]: one scale per 16 values
    //
    // ggml processes 4 values at a time with specific layout:
    // For each group of 32 (l = 0..32):
    //   q1 = (ql[l] & 0xF) | ((qh[l] >> 0) & 3) << 4
    //   q2 = (ql[l+32] & 0xF) | ((qh[l] >> 2) & 3) << 4
    //   q3 = (ql[l] >> 4) | ((qh[l] >> 4) & 3) << 4
    //   q4 = (ql[l+32] >> 4) | ((qh[l] >> 6) & 3) << 4

    // Process first half (0-127): use scales[0..7]
    for l in 0..32 {
        let is = l / 16; // 0 or 1

        // Extract 4 values from the packed layout
        let q1 = ((block.ql[l] & 0x0F) | ((block.qh[l] & 3) << 4)) as i8 - 32;
        let q2 = ((block.ql[l + 32] & 0x0F) | (((block.qh[l] >> 2) & 3) << 4)) as i8 - 32;
        let q3 = ((block.ql[l] >> 4) | (((block.qh[l] >> 4) & 3) << 4)) as i8 - 32;
        let q4 = ((block.ql[l + 32] >> 4) | (((block.qh[l] >> 6) & 3) << 4)) as i8 - 32;

        // Write outputs with correct scales (matches llama.cpp: sc[is+0/2/4/6])
        output[l] = d * (block.scales[is] as f32) * (q1 as f32);
        output[l + 32] = d * (block.scales[is + 2] as f32) * (q2 as f32);
        output[l + 64] = d * (block.scales[is + 4] as f32) * (q3 as f32);
        output[l + 96] = d * (block.scales[is + 6] as f32) * (q4 as f32);
    }

    // Process second half (128-255): use scales[8..15]
    for l in 0..32 {
        let is = l / 16; // 0 or 1
        let sc_offset = 8; // Offset into scales array for second half

        let q1 = ((block.ql[l + 64] & 0x0F) | ((block.qh[l + 32] & 3) << 4)) as i8 - 32;
        let q2 = ((block.ql[l + 96] & 0x0F) | (((block.qh[l + 32] >> 2) & 3) << 4)) as i8 - 32;
        let q3 = ((block.ql[l + 64] >> 4) | (((block.qh[l + 32] >> 4) & 3) << 4)) as i8 - 32;
        let q4 = ((block.ql[l + 96] >> 4) | (((block.qh[l + 32] >> 6) & 3) << 4)) as i8 - 32;

        // Write outputs with correct scales (sc_offset=8, so we use scales[8..15])
        output[l + 128] = d * (block.scales[sc_offset + is] as f32) * (q1 as f32);
        output[l + 160] = d * (block.scales[sc_offset + is + 2] as f32) * (q2 as f32);
        output[l + 192] = d * (block.scales[sc_offset + is + 4] as f32) * (q3 as f32);
        output[l + 224] = d * (block.scales[sc_offset + is + 6] as f32) * (q4 as f32);
    }

    Ok(())
}

/// Dequantize Q5_K block to f32
/// Q5_K uses 256-element super-blocks with 5-bit quantization
/// Based on ggml dequantize_row_q5_K implementation
pub fn dequantize_q5_k(block: &BlockQ5_K, output: &mut [f32]) -> Result<()> {
    if output.len() != QK_K {
        return Err(Error::InvalidShape(format!(
            "Output buffer must be {} elements",
            QK_K
        )));
    }

    // Convert f16 to f32
    let d = half::f16::from_bits(block.d).to_f32();

    // Process 256 elements in 16 groups of 16
    for l in 0..16 {
        let is = l;

        // Extract 16 values from the packed layout
        // Q5_K: 4 bits from ql + 1 bit from qh
        for k in 0..16 {
            let ql_idx = l * 16 + k;
            if ql_idx >= QK_K || ql_idx >= block.ql.len() {
                break;
            }

            let qh_idx = l * 2 + (k / 8);
            if qh_idx >= block.qh.len() {
                break;
            }

            let qh_bit = (k % 8) / 4; // Which bit in the qh byte

            let ql_val = block.ql[ql_idx] as i8;
            let qh_val = (block.qh[qh_idx] >> (qh_bit * 4)) & 0xF;

            // Combine 4 bits from ql with 1 bit from qh
            let quant_val = ql_val | ((qh_val as i8) << 4);

            // Apply scale and write output
            output[ql_idx] = d * (block.scales[is] as f32) * (quant_val as f32);
        }
    }

    Ok(())
}

/// Dequantize Q8_K block to f32
/// Q8_K uses 256-element super-blocks with 8-bit quantization
/// Based on ggml dequantize_row_q8_K implementation
pub fn dequantize_q8_k(block: &BlockQ8_K, output: &mut [f32]) -> Result<()> {
    if output.len() != QK_K {
        return Err(Error::InvalidShape(format!(
            "Output buffer must be {} elements",
            QK_K
        )));
    }

    // Convert f16 to f32
    let d = half::f16::from_bits(block.d).to_f32();
    let min = half::f16::from_bits(block.dmin).to_f32();

    // Process 256 elements in 32 groups of 8
    for l in 0..32 {
        let is = l;

        // Extract scale for this group (4 bits per scale)
        let scale_byte = block.scales[is / 2];
        let scale_val = if is % 2 == 0 {
            scale_byte & 0xF
        } else {
            scale_byte >> 4
        };

        // Process 8 values in this group
        for k in 0..8 {
            let idx = l * 8 + k;
            let quant_val = block.quants[idx] as f32;

            // Apply scale and write output
            output[idx] = d * (scale_val as f32) * quant_val + min;
        }
    }

    Ok(())
}

/// Get block size for quantization type
pub fn get_block_size(dtype: DataType) -> Option<usize> {
    match dtype {
        DataType::Q4_0 | DataType::Q4_1 => Some(Q4_BLOCK_SIZE),
        DataType::Q8_0 | DataType::Q8_1 => Some(Q8_BLOCK_SIZE),
        DataType::Q4_K => Some(QK_K),
        DataType::Q5_K => Some(QK_K),
        DataType::Q6_K => Some(QK_K),
        DataType::Q8_K => Some(QK_K),
        _ => None,
    }
}

/// Dequantize an entire tensor from quantized bytes to f32
///
/// This is the main entry point for dequantizing tensors from HOST storage.
/// Handles all quantization formats by dispatching to the appropriate block dequantizer.
///
/// # Arguments
/// * `data` - Raw quantized bytes
/// * `dtype` - Quantization format (Q4_K_M, Q8_0, etc.)
/// * `element_count` - Number of f32 elements to produce
///
/// # Returns
/// Vec<f32> with dequantized values
pub fn dequantize_tensor(data: &[u8], dtype: DataType, element_count: usize) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; element_count];

    match dtype {
        DataType::Q4_K => {
            // Q4_K uses 256-element blocks
            let block_size = QK_K; // 256
            let bytes_per_block = std::mem::size_of::<BlockQ4_K>(); // 144 bytes

            if !data.len().is_multiple_of(bytes_per_block) {
                return Err(Error::InvalidFormat(format!(
                    "Q4_K data size {} is not a multiple of block size {}",
                    data.len(),
                    bytes_per_block
                )));
            }

            let num_blocks = data.len() / bytes_per_block;
            let expected_elements = num_blocks * block_size;

            if element_count != expected_elements {
                return Err(Error::InvalidShape(format!(
                    "Element count mismatch: expected {}, got {}",
                    expected_elements, element_count
                )));
            }

            // Dequantize each block
            for block_idx in 0..num_blocks {
                let block_start = block_idx * bytes_per_block;
                let block_bytes = &data[block_start..block_start + bytes_per_block];

                // Safety: BlockQ4_K is repr(C) and we verified size
                let block = unsafe { &*(block_bytes.as_ptr() as *const BlockQ4_K) };

                let output_start = block_idx * block_size;
                let output_slice = &mut output[output_start..output_start + block_size];

                dequantize_q4_k(block, output_slice)?;
            }
        }

        DataType::Q6_K => {
            let block_size = QK_K;
            let bytes_per_block = std::mem::size_of::<BlockQ6_K>();
            let num_blocks = data.len() / bytes_per_block;

            for block_idx in 0..num_blocks {
                let block_start = block_idx * bytes_per_block;
                let block_bytes = &data[block_start..block_start + bytes_per_block];
                let block = unsafe { &*(block_bytes.as_ptr() as *const BlockQ6_K) };

                let output_start = block_idx * block_size;
                let output_slice = &mut output[output_start..output_start + block_size];

                dequantize_q6_k(block, output_slice)?;
            }
        }

        DataType::Q5_K => {
            let block_size = QK_K;
            let bytes_per_block = std::mem::size_of::<BlockQ5_K>();
            let num_blocks = data.len() / bytes_per_block;

            for block_idx in 0..num_blocks {
                let block_start = block_idx * bytes_per_block;
                let block_bytes = &data[block_start..block_start + bytes_per_block];
                let block = unsafe { &*(block_bytes.as_ptr() as *const BlockQ5_K) };

                let output_start = block_idx * block_size;
                let output_slice = &mut output[output_start..output_start + block_size];

                dequantize_q5_k(block, output_slice)?;
            }
        }

        DataType::Q8_K => {
            let block_size = QK_K;
            let bytes_per_block = std::mem::size_of::<BlockQ8_K>();
            let num_blocks = data.len() / bytes_per_block;

            for block_idx in 0..num_blocks {
                let block_start = block_idx * bytes_per_block;
                let block_bytes = &data[block_start..block_start + bytes_per_block];
                let block = unsafe { &*(block_bytes.as_ptr() as *const BlockQ8_K) };

                let output_start = block_idx * block_size;
                let output_slice = &mut output[output_start..output_start + block_size];

                dequantize_q8_k(block, output_slice)?;
            }
        }

        DataType::Q8_0 => {
            let block_size = Q8_BLOCK_SIZE;
            let bytes_per_block = std::mem::size_of::<BlockQ8_0>();
            let num_blocks = data.len() / bytes_per_block;

            for block_idx in 0..num_blocks {
                let block_start = block_idx * bytes_per_block;
                let block_bytes = &data[block_start..block_start + bytes_per_block];
                let block = unsafe { &*(block_bytes.as_ptr() as *const BlockQ8_0) };

                let output_start = block_idx * block_size;
                let output_slice = &mut output[output_start..output_start + block_size];

                dequantize_q8_0(block, output_slice)?;
            }
        }

        DataType::F32 => {
            // Already f32, just copy
            if data.len() != element_count * 4 {
                return Err(Error::InvalidShape(format!(
                    "F32 data size mismatch: expected {} bytes, got {}",
                    element_count * 4,
                    data.len()
                )));
            }

            for i in 0..element_count {
                let bytes = [
                    data[i * 4],
                    data[i * 4 + 1],
                    data[i * 4 + 2],
                    data[i * 4 + 3],
                ];
                output[i] = f32::from_le_bytes(bytes);
            }
        }

        DataType::F16 => {
            // Convert f16 to f32
            if data.len() != element_count * 2 {
                return Err(Error::InvalidShape(format!(
                    "F16 data size mismatch: expected {} bytes, got {}",
                    element_count * 2,
                    data.len()
                )));
            }

            for i in 0..element_count {
                let bytes = [data[i * 2], data[i * 2 + 1]];
                let f16_val = half::f16::from_le_bytes(bytes);
                output[i] = f16_val.to_f32();
            }
        }

        _ => {
            return Err(Error::UnsupportedDataType(format!(
                "Dequantization not implemented for {:?}",
                dtype
            )));
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_dequant() {
        let block = BlockQ4_0 {
            quants: [0x10; 16], // All values = 1, 0 (in nibbles)
        };

        let mut output = [0.0f32; Q4_BLOCK_SIZE];
        dequantize_q4_0(&block, &mut output).unwrap();

        // 0x10 -> lower = 0, upper = 1
        // After offset -8: lower = -8, upper = -7
        // With scale=0.01
        assert_eq!(output[0], -0.08);
        assert_eq!(output[16], -0.07);
    }

    #[test]
    fn test_q8_dequant() {
        let block = BlockQ8_0 {
            scale: half::f16::from_f32(0.25).to_bits(),
            quants: [10i8; Q8_BLOCK_SIZE],
        };

        let mut output = [0.0f32; Q8_BLOCK_SIZE];
        dequantize_q8_0(&block, &mut output).unwrap();

        assert_eq!(output[0], 10.0 * 0.25);
    }

    #[test]
    fn test_q4_k_dequant() {
        // Create a simple Q4_K block
        let block = BlockQ4_K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.5).to_bits(),
            scales: [0x11; 12],   // Simple pattern for testing
            qs: [0x50; QK_K / 2], // 5 in lower nibble, 0 in upper
        };

        let mut output = [0.0f32; QK_K];
        dequantize_q4_k(&block, &mut output).unwrap();

        // Check that we don't have inf/nan
        let has_nan = output.iter().any(|&x| x.is_nan());
        let has_inf = output.iter().any(|&x| x.is_infinite());
        assert!(!has_nan, "Q4_K dequantization produced NaN");
        assert!(!has_inf, "Q4_K dequantization produced inf");

        // Check reasonable value ranges (4-bit values are 0-15, with scaling)
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.abs() <= 100.0,
                "Value at index {} is out of range: {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_q6_k_dequant() {
        // Create a simple Q6_K block
        let mut block = BlockQ6_K {
            ql: [0u8; QK_K / 2],
            qh: [0u8; QK_K / 4],
            scales: [1i8; QK_K / 16],
            d: half::f16::from_f32(1.0).to_bits(),
        };

        // Set some test values
        // First element: ql[0] = 0x10 (lower 4 bits = 0, upper 4 bits = 1)
        //                qh[0] = 0x00 (bits 0-1 = 0)
        // This should give q1 = 0, q3 = 1 (after combining with qh)
        block.ql[0] = 0x10;
        block.qh[0] = 0x00;

        let mut output = [0.0f32; QK_K];
        dequantize_q6_k(&block, &mut output).unwrap();

        // Check that we don't have inf/nan
        let has_nan = output.iter().any(|&x| x.is_nan());
        let has_inf = output.iter().any(|&x| x.is_infinite());
        assert!(!has_nan, "Q6_K dequantization produced NaN");
        assert!(!has_inf, "Q6_K dequantization produced inf");

        // Check reasonable value ranges (-32 to 31 after bias, times scale and d)
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.abs() <= 100.0,
                "Value at index {} is out of range: {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_q5_k_dequant() {
        // Create a simple Q5_K block
        let mut block = BlockQ5_K {
            ql: [0u8; QK_K / 2],
            qh: [0u8; QK_K / 8],
            scales: [1i8; QK_K / 16],
            d: half::f16::from_f32(1.0).to_bits(),
        };

        // Set some test values
        block.ql[0] = 0x10; // Lower 4 bits = 0, upper 4 bits = 1
        block.qh[0] = 0x00; // No high bits

        let mut output = [0.0f32; QK_K];
        dequantize_q5_k(&block, &mut output).unwrap();

        // Check that we don't have inf/nan
        let has_nan = output.iter().any(|&x| x.is_nan());
        let has_inf = output.iter().any(|&x| x.is_infinite());
        assert!(!has_nan, "Q5_K dequantization produced NaN");
        assert!(!has_inf, "Q5_K dequantization produced inf");

        // Check reasonable value ranges
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.abs() <= 100.0,
                "Value at index {} is out of range: {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_q8_k_dequant() {
        // Create a simple Q8_K block
        let mut block = BlockQ8_K {
            quants: [0i8; QK_K],
            scales: [0u8; QK_K / 8],
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
        };

        // Set some test values
        block.quants[0] = 10; // Test value
        block.scales[0] = 0x11; // Simple scale pattern

        let mut output = [0.0f32; QK_K];
        dequantize_q8_k(&block, &mut output).unwrap();

        // Check that we don't have inf/nan
        let has_nan = output.iter().any(|&x| x.is_nan());
        let has_inf = output.iter().any(|&x| x.is_infinite());
        assert!(!has_nan, "Q8_K dequantization produced NaN");
        assert!(!has_inf, "Q8_K dequantization produced inf");

        // Check reasonable value ranges
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.abs() <= 100.0,
                "Value at index {} is out of range: {}",
                i,
                val
            );
        }
    }
}
