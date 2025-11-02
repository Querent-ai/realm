//! Extended tensor loading with optimal weight format selection

use crate::weight_format::WeightFormat;
use realm_core::{
    error::Result,
    quant::{
        BlockQ2_K, BlockQ3_K, BlockQ4_0, BlockQ4_1, BlockQ4_K, BlockQ5_0, BlockQ5_1, BlockQ5_K,
        BlockQ6_K, BlockQ8_0, BlockQ8_1, BlockQ8_K,
    },
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
    _tensor_name: &str,
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
        DataType::Q2_K => {
            const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ2_K>();
            let num_blocks = raw_data.len() / BLOCK_SIZE;
            let mut blocks = Vec::with_capacity(num_blocks);
            for block_data in raw_data.chunks_exact(BLOCK_SIZE) {
                let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ2_K) };
                blocks.push(block);
            }
            Ok(WeightFormat::Q2K(blocks))
        }
        DataType::Q3_K => {
            const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ3_K>();
            let num_blocks = raw_data.len() / BLOCK_SIZE;
            let mut blocks = Vec::with_capacity(num_blocks);
            for block_data in raw_data.chunks_exact(BLOCK_SIZE) {
                let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ3_K) };
                blocks.push(block);
            }
            Ok(WeightFormat::Q3K(blocks))
        }
        DataType::Q4_0 => {
            const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ4_0>();
            let num_blocks = raw_data.len() / BLOCK_SIZE;
            let mut blocks = Vec::with_capacity(num_blocks);
            for block_data in raw_data.chunks_exact(BLOCK_SIZE) {
                let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ4_0) };
                blocks.push(block);
            }
            Ok(WeightFormat::Q40(blocks))
        }
        DataType::Q4_1 => {
            const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ4_1>();
            let num_blocks = raw_data.len() / BLOCK_SIZE;
            let mut blocks = Vec::with_capacity(num_blocks);
            for block_data in raw_data.chunks_exact(BLOCK_SIZE) {
                let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ4_1) };
                blocks.push(block);
            }
            Ok(WeightFormat::Q41(blocks))
        }
        DataType::Q5_0 => {
            const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ5_0>();
            let num_blocks = raw_data.len() / BLOCK_SIZE;
            let mut blocks = Vec::with_capacity(num_blocks);
            for block_data in raw_data.chunks_exact(BLOCK_SIZE) {
                let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ5_0) };
                blocks.push(block);
            }
            Ok(WeightFormat::Q50(blocks))
        }
        DataType::Q5_1 => {
            const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ5_1>();
            let num_blocks = raw_data.len() / BLOCK_SIZE;
            let mut blocks = Vec::with_capacity(num_blocks);
            for block_data in raw_data.chunks_exact(BLOCK_SIZE) {
                let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ5_1) };
                blocks.push(block);
            }
            Ok(WeightFormat::Q51(blocks))
        }
        DataType::Q8_0 => {
            const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ8_0>();
            let num_blocks = raw_data.len() / BLOCK_SIZE;
            let mut blocks = Vec::with_capacity(num_blocks);
            for block_data in raw_data.chunks_exact(BLOCK_SIZE) {
                let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ8_0) };
                blocks.push(block);
            }
            Ok(WeightFormat::Q80(blocks))
        }
        DataType::Q8_1 => {
            const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ8_1>();
            let num_blocks = raw_data.len() / BLOCK_SIZE;
            let mut blocks = Vec::with_capacity(num_blocks);
            for block_data in raw_data.chunks_exact(BLOCK_SIZE) {
                let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ8_1) };
                blocks.push(block);
            }
            Ok(WeightFormat::Q81(blocks))
        }
        _ => {
            // Unsupported format, fall back to F32
            // eprintln!("WARN: Unsupported weight format {:?} for {}, falling back to F32", metadata.desc.dtype, tensor_name);

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
