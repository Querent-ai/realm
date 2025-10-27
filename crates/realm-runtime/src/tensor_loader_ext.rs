//! Extended tensor loading with optimal weight format selection

use crate::weight_format::WeightFormat;
use realm_core::{
    error::Result,
    quant::{BlockQ4_K, BlockQ5_K, BlockQ6_K, BlockQ8_K},
    tensor::DataType,
    GGUFParser, TensorMetadata,
};

/// Load weights in optimal format (quantized or F32) based on tensor metadata
///
/// # Arguments
/// * `tensor_name` - Name of the tensor for debugging
/// * `metadata` - Tensor metadata from GGUF parser
/// * `parser` - GGUF parser instance
/// * `data_offset` - Offset to tensor data section
///
/// # Returns
/// * `Result<WeightFormat>` - Weight tensor in optimal format
pub fn load_weight_optimal<R: std::io::Read + std::io::Seek>(
    tensor_name: &str,
    metadata: &TensorMetadata,
    parser: &mut GGUFParser<R>,
    _data_offset: u64,
) -> Result<WeightFormat> {
    // Calculate absolute offset
    // metadata.offset is absolute from file start, not relative to tensor data section
    let absolute_offset = metadata.offset;

    // Read raw tensor data
    let raw_data = parser.read_tensor_data(absolute_offset, metadata.size_bytes)?;

    // Return appropriate format based on dtype
    match metadata.desc.dtype {
        DataType::F32 => {
            // F32: Parse as float array
            let mut result = Vec::with_capacity(metadata.desc.element_count());
            for chunk in raw_data.chunks_exact(4) {
                let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                result.push(f32::from_le_bytes(bytes));
            }
            Ok(WeightFormat::F32(result))
        }
        DataType::Q4_K => {
            // Q4_K: Keep as quantized blocks
            const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ4_K>();
            let num_blocks = raw_data.len() / BLOCK_SIZE;
            let mut blocks = Vec::with_capacity(num_blocks);

            for block_data in raw_data.chunks_exact(BLOCK_SIZE) {
                let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ4_K) };
                blocks.push(block);
            }
            Ok(WeightFormat::Q4K(blocks))
        }
        DataType::Q5_K => {
            // Q5_K: Keep as quantized blocks
            const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ5_K>();
            let num_blocks = raw_data.len() / BLOCK_SIZE;
            let mut blocks = Vec::with_capacity(num_blocks);

            for block_data in raw_data.chunks_exact(BLOCK_SIZE) {
                let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ5_K) };
                blocks.push(block);
            }
            Ok(WeightFormat::Q5K(blocks))
        }
        DataType::Q6_K => {
            // Q6_K: Keep as quantized blocks
            const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ6_K>();
            let num_blocks = raw_data.len() / BLOCK_SIZE;
            let mut blocks = Vec::with_capacity(num_blocks);

            for block_data in raw_data.chunks_exact(BLOCK_SIZE) {
                let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ6_K) };
                blocks.push(block);
            }
            Ok(WeightFormat::Q6K(blocks))
        }
        DataType::Q8_K => {
            // Q8_K: Keep as quantized blocks
            const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ8_K>();
            let num_blocks = raw_data.len() / BLOCK_SIZE;
            let mut blocks = Vec::with_capacity(num_blocks);

            for block_data in raw_data.chunks_exact(BLOCK_SIZE) {
                let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ8_K) };
                blocks.push(block);
            }
            Ok(WeightFormat::Q8K(blocks))
        }
        _ => {
            // Unsupported format, fall back to F32
            eprintln!(
                "WARN: Unsupported weight format {:?} for {}, falling back to F32",
                metadata.desc.dtype, tensor_name
            );

            // Try to parse as F32 anyway
            let mut result = Vec::with_capacity(metadata.desc.element_count());
            for chunk in raw_data.chunks_exact(4) {
                let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                result.push(f32::from_le_bytes(bytes));
            }
            Ok(WeightFormat::F32(result))
        }
    }
}
