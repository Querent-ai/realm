//! Weight format definitions for optimized matrix multiplication

use realm_core::quant::{
    BlockQ2_K, BlockQ3_K, BlockQ4_0, BlockQ4_1, BlockQ4_K, BlockQ5_0, BlockQ5_1, BlockQ5_K,
    BlockQ6_K, BlockQ8_0, BlockQ8_1, BlockQ8_K, Q4_BLOCK_SIZE, Q8_BLOCK_SIZE, QK_K,
};

/// Weight format enum for different quantization types
/// This allows us to store weights in their optimal format and dispatch
/// to the appropriate matrix multiplication kernel
#[derive(Debug, Clone)]
pub enum WeightFormat {
    /// F32 weights (standard floating point)
    F32(Vec<f32>),
    /// Q4_K quantized weights (256 elements per block)
    Q4K(Vec<BlockQ4_K>),
    /// Q5_K quantized weights (256 elements per block)
    Q5K(Vec<BlockQ5_K>),
    /// Q6_K quantized weights (256 elements per block)
    Q6K(Vec<BlockQ6_K>),
    /// Q8_K quantized weights (256 elements per block)
    Q8K(Vec<BlockQ8_K>),
    /// Q2_K quantized weights (256 elements per block)
    Q2K(Vec<BlockQ2_K>),
    /// Q3_K quantized weights (256 elements per block)
    Q3K(Vec<BlockQ3_K>),
    /// Q4_0 quantized weights (32 elements per block)
    Q40(Vec<BlockQ4_0>),
    /// Q4_1 quantized weights (32 elements per block)
    Q41(Vec<BlockQ4_1>),
    /// Q5_0 quantized weights (32 elements per block)
    Q50(Vec<BlockQ5_0>),
    /// Q5_1 quantized weights (32 elements per block)
    Q51(Vec<BlockQ5_1>),
    /// Q8_0 quantized weights (32 elements per block)
    Q80(Vec<BlockQ8_0>),
    /// Q8_1 quantized weights (32 elements per block)
    Q81(Vec<BlockQ8_1>),
}

impl WeightFormat {
    /// Get the format name for debugging
    pub fn format_name(&self) -> &'static str {
        match self {
            WeightFormat::F32(_) => "F32",
            WeightFormat::Q4K(_) => "Q4_K",
            WeightFormat::Q5K(_) => "Q5_K",
            WeightFormat::Q6K(_) => "Q6_K",
            WeightFormat::Q8K(_) => "Q8_K",
            WeightFormat::Q2K(_) => "Q2_K",
            WeightFormat::Q3K(_) => "Q3_K",
            WeightFormat::Q40(_) => "Q4_0",
            WeightFormat::Q41(_) => "Q4_1",
            WeightFormat::Q50(_) => "Q5_0",
            WeightFormat::Q51(_) => "Q5_1",
            WeightFormat::Q80(_) => "Q8_0",
            WeightFormat::Q81(_) => "Q8_1",
        }
    }

    /// Get the number of elements this weight format represents
    pub fn element_count(&self) -> usize {
        match self {
            WeightFormat::F32(data) => data.len(),
            WeightFormat::Q4K(blocks) => blocks.len() * QK_K,
            WeightFormat::Q5K(blocks) => blocks.len() * QK_K,
            WeightFormat::Q6K(blocks) => blocks.len() * QK_K,
            WeightFormat::Q8K(blocks) => blocks.len() * QK_K,
            WeightFormat::Q2K(blocks) => blocks.len() * QK_K,
            WeightFormat::Q3K(blocks) => blocks.len() * QK_K,
            WeightFormat::Q40(blocks) => blocks.len() * Q4_BLOCK_SIZE,
            WeightFormat::Q41(blocks) => blocks.len() * Q4_BLOCK_SIZE,
            WeightFormat::Q50(blocks) => blocks.len() * Q4_BLOCK_SIZE,
            WeightFormat::Q51(blocks) => blocks.len() * Q4_BLOCK_SIZE,
            WeightFormat::Q80(blocks) => blocks.len() * Q8_BLOCK_SIZE,
            WeightFormat::Q81(blocks) => blocks.len() * Q8_BLOCK_SIZE,
        }
    }

    /// Get the number of elements (alias for element_count for compatibility)
    pub fn len(&self) -> usize {
        self.element_count()
    }

    /// Check if the weight tensor is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
