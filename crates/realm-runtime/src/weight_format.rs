//! Weight format definitions for optimized matrix multiplication

use realm_core::quant::{BlockQ4_K, BlockQ5_K, BlockQ6_K, BlockQ8_K};

/// Weight format enum for different quantization types
/// This allows us to store weights in their optimal format and dispatch
/// to the appropriate matrix multiplication kernel
#[derive(Debug, Clone)]
pub enum WeightFormat {
    /// F32 weights (standard floating point)
    F32(Vec<f32>),
    /// Q4_K quantized weights
    Q4K(Vec<BlockQ4_K>),
    /// Q5_K quantized weights
    Q5K(Vec<BlockQ5_K>),
    /// Q6_K quantized weights
    Q6K(Vec<BlockQ6_K>),
    /// Q8_K quantized weights
    Q8K(Vec<BlockQ8_K>),
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
        }
    }

    /// Get the number of elements this weight format represents
    pub fn element_count(&self) -> usize {
        match self {
            WeightFormat::F32(data) => data.len(),
            WeightFormat::Q4K(blocks) => blocks.len() * 256, // QK_K = 256
            WeightFormat::Q5K(blocks) => blocks.len() * 256,
            WeightFormat::Q6K(blocks) => blocks.len() * 256,
            WeightFormat::Q8K(blocks) => blocks.len() * 256,
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
