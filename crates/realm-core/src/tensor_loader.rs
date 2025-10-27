//! Tensor loading and management from GGUF files
//!
//! Supports lazy loading, caching, and efficient memory management.

use crate::error::{Error, Result};
use crate::formats::gguf::GGUFParser;
use crate::quant::{
    dequantize_q4_0, dequantize_q4_k, dequantize_q6_k, dequantize_q8_0, BlockQ4_0, BlockQ4_K,
    BlockQ6_K, BlockQ8_0,
};
use crate::tensor::{DataType, TensorDesc};
use std::collections::HashMap;
use std::io::{Read, Seek};

/// Tensor loader with lazy loading and caching
pub struct TensorLoader {
    /// Loaded tensors cache (name -> tensor data)
    cache: HashMap<String, Vec<f32>>,
    /// Tensor metadata (name -> descriptor)
    metadata: HashMap<String, TensorMetadata>,
    /// Base offset for tensor data in file
    data_offset: u64,
}

/// Metadata about a tensor's location and format
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    /// Tensor descriptor (shape, dtype)
    pub desc: TensorDesc,
    /// Byte offset from data_offset
    pub offset: u64,
    /// Size in bytes
    pub size_bytes: usize,
}

impl TensorLoader {
    /// Create a new tensor loader
    pub fn new(data_offset: u64) -> Self {
        Self {
            cache: HashMap::new(),
            metadata: HashMap::new(),
            data_offset,
        }
    }

    /// Register a tensor from GGUF metadata
    pub fn register_tensor(&mut self, name: String, desc: TensorDesc, offset: u64) {
        let metadata = TensorMetadata {
            desc: desc.clone(),
            offset,
            size_bytes: desc.size_bytes,
        };

        self.metadata.insert(name, metadata);
    }

    /// Load a tensor by name (with caching)
    pub fn load_tensor<R: Read + Seek>(
        &mut self,
        name: &str,
        parser: &mut GGUFParser<R>,
    ) -> Result<&[f32]> {
        // Check cache first
        if self.cache.contains_key(name) {
            return Ok(&self.cache[name]);
        }

        // Get metadata
        let metadata = self
            .metadata
            .get(name)
            .ok_or_else(|| Error::Unknown(format!("Tensor not found: {}", name)))?
            .clone();

        // Load raw data
        let absolute_offset = self.data_offset + metadata.offset;
        let raw_data = parser.read_tensor_data(absolute_offset, metadata.size_bytes)?;

        // Dequantize based on dtype
        let f32_data = match metadata.desc.dtype {
            DataType::F32 => {
                // Already float32, just reinterpret bytes
                let mut result = Vec::with_capacity(metadata.desc.element_count());
                for chunk in raw_data.chunks_exact(4) {
                    let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    result.push(f32::from_le_bytes(bytes));
                }
                result
            }
            DataType::F16 => {
                // F16 -> F32 conversion
                let mut result = Vec::with_capacity(metadata.desc.element_count());
                for chunk in raw_data.chunks_exact(2) {
                    let bytes = [chunk[0], chunk[1]];
                    let f16_bits = u16::from_le_bytes(bytes);
                    result.push(half::f16::from_bits(f16_bits).to_f32());
                }
                result
            }
            DataType::Q4_0 => {
                // Dequantize Q4_0
                self.dequantize_q4_0(name, &raw_data, metadata.desc.element_count())?
            }
            DataType::Q8_0 => {
                // Dequantize Q8_0
                self.dequantize_q8_0(&raw_data, metadata.desc.element_count())?
            }
            DataType::Q4_K => {
                // Dequantize Q4_K
                self.dequantize_q4_k(&raw_data, metadata.desc.element_count())?
            }
            DataType::Q6_K => {
                // Dequantize Q6_K
                self.dequantize_q6_k(&raw_data, metadata.desc.element_count())?
            }
            DataType::Q5_K => {
                let block_size = std::mem::size_of::<crate::quant::BlockQ5_K>();
                let num_blocks = raw_data.len() / block_size;
                let mut output = vec![0.0f32; num_blocks * 256];

                for (i, block_data) in raw_data.chunks_exact(block_size).enumerate() {
                    let block = unsafe {
                        std::ptr::read(block_data.as_ptr() as *const crate::quant::BlockQ5_K)
                    };
                    let output_slice = &mut output[i * 256..(i + 1) * 256];
                    crate::quant::dequantize_q5_k(&block, output_slice)?;
                }

                output
            }
            DataType::Q8_K => {
                let block_size = std::mem::size_of::<crate::quant::BlockQ8_K>();
                let num_blocks = raw_data.len() / block_size;
                let mut output = vec![0.0f32; num_blocks * 256];

                for (i, block_data) in raw_data.chunks_exact(block_size).enumerate() {
                    let block = unsafe {
                        std::ptr::read(block_data.as_ptr() as *const crate::quant::BlockQ8_K)
                    };
                    let output_slice = &mut output[i * 256..(i + 1) * 256];
                    crate::quant::dequantize_q8_k(&block, output_slice)?;
                }

                output
            }
            _ => {
                return Err(Error::UnsupportedDataType(format!(
                    "Unsupported dtype for loading: {:?}",
                    metadata.desc.dtype
                )))
            }
        };

        // Cache and return
        self.cache.insert(name.to_string(), f32_data);
        Ok(&self.cache[name])
    }

    /// Load multiple tensors at once
    pub fn load_tensors<R: Read + Seek>(
        &mut self,
        names: &[&str],
        parser: &mut GGUFParser<R>,
    ) -> Result<HashMap<String, Vec<f32>>> {
        let mut result = HashMap::new();
        for &name in names {
            let data = self.load_tensor(name, parser)?.to_vec();
            result.insert(name.to_string(), data);
        }
        Ok(result)
    }

    /// Check if a tensor is loaded in cache
    pub fn is_cached(&self, name: &str) -> bool {
        self.cache.contains_key(name)
    }

    /// Get tensor metadata
    pub fn get_metadata(&self, name: &str) -> Option<&TensorMetadata> {
        self.metadata.get(name)
    }

    /// List all available tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.metadata.keys().map(|s| s.as_str()).collect()
    }

    /// Clear cache to free memory
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache size in bytes (approximate)
    pub fn cache_size_bytes(&self) -> usize {
        self.cache
            .values()
            .map(|v| v.len() * std::mem::size_of::<f32>())
            .sum()
    }

    // Helper methods for dequantization

    fn dequantize_q4_0(
        &self,
        tensor_name: &str,
        data: &[u8],
        element_count: usize,
    ) -> Result<Vec<f32>> {
        const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ4_0>();
        const ELEMENTS_PER_BLOCK: usize = 32;

        // Calculate actual number of blocks from data size
        let num_blocks = data.len() / BLOCK_SIZE;
        let actual_elements = num_blocks * ELEMENTS_PER_BLOCK;

        // Use the smaller of the two to avoid buffer overruns
        let result_size = actual_elements.min(element_count);
        let mut result = vec![0.0f32; result_size];

        // Debug output - use a counter to track calls
        static CALL_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let call_num = CALL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if call_num < 5 {
            eprintln!(
                "[Q4_0 dequant call #{}] tensor='{}', BLOCK_SIZE={}, data.len()={}, element_count={}",
                call_num,
                tensor_name,
                BLOCK_SIZE,
                data.len(),
                element_count
            );

            // Check if size matches expected
            let expected_blocks = element_count / 32;
            let expected_bytes = expected_blocks * BLOCK_SIZE;
            if data.len() != expected_bytes {
                eprintln!(
                    "  WARNING: Size mismatch! Expected {} bytes but got {}",
                    expected_bytes,
                    data.len()
                );
            }

            // Dump first 18 bytes to see actual layout
            if !data.is_empty() {
                eprintln!("  First block raw bytes:");
                eprint!("    ");
                for (i, &byte) in data.iter().take(18).enumerate() {
                    eprint!("{:02x} ", byte);
                    if i == 1 {
                        eprint!("| "); // After scale (2 bytes)
                    }
                }
                eprintln!();
            }
        }

        for (block_idx, block_bytes) in data.chunks_exact(BLOCK_SIZE).enumerate() {
            let block: BlockQ4_0 = unsafe { std::ptr::read(block_bytes.as_ptr() as *const _) };

            let offset = block_idx * 32;
            let result_len = result.len();
            let end = (offset + 32).min(result_len);
            if offset < result_len {
                dequantize_q4_0(&block, &mut result[offset..end])?;

                // Debug first few blocks
                if block_idx < 3 {
                    let has_nan = result[offset..end].iter().any(|&x| x.is_nan());
                    let has_inf = result[offset..end].iter().any(|&x| x.is_infinite());
                    if has_nan || has_inf {
                        eprintln!(
                            "WARNING: Q4_0 block {} produced nan={}, inf={}",
                            block_idx, has_nan, has_inf
                        );
                        eprintln!("  Block quants[0]={:#x}", block.quants[0]);
                    }
                }
            }
        }

        Ok(result)
    }

    fn dequantize_q8_0(&self, data: &[u8], element_count: usize) -> Result<Vec<f32>> {
        const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ8_0>();
        let mut result = vec![0.0f32; element_count];

        for (block_idx, block_bytes) in data.chunks_exact(BLOCK_SIZE).enumerate() {
            let block: BlockQ8_0 = unsafe { std::ptr::read(block_bytes.as_ptr() as *const _) };

            let offset = block_idx * 32;
            let result_len = result.len();
            let end = (offset + 32).min(result_len);
            if offset < result_len {
                dequantize_q8_0(&block, &mut result[offset..end])?;
            }
        }

        Ok(result)
    }

    fn dequantize_q6_k(&self, data: &[u8], element_count: usize) -> Result<Vec<f32>> {
        const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ6_K>();
        const QK_K: usize = 256;
        let mut result = vec![0.0f32; element_count];

        static FIRST_CALL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
        if FIRST_CALL.swap(false, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "Q6_K dequant: BLOCK_SIZE={}, data.len()={}, element_count={}, expected_blocks={}",
                BLOCK_SIZE,
                data.len(),
                element_count,
                element_count / QK_K
            );
        }

        for (block_idx, block_bytes) in data.chunks_exact(BLOCK_SIZE).enumerate() {
            let block: BlockQ6_K = unsafe { std::ptr::read(block_bytes.as_ptr() as *const _) };

            let offset = block_idx * QK_K;
            let result_len = result.len();
            let end = (offset + QK_K).min(result_len);
            if offset < result_len {
                dequantize_q6_k(&block, &mut result[offset..end])?;

                // Debug first block
                if block_idx == 0 {
                    let has_nan = result[offset..end].iter().any(|&x| x.is_nan());
                    let has_inf = result[offset..end].iter().any(|&x| x.is_infinite());
                    if has_nan || has_inf {
                        eprintln!(
                            "WARNING: First Q6_K block produced nan={}, inf={}",
                            has_nan, has_inf
                        );
                        eprintln!(
                            "  Block: d={}, scale[0]={}, ql[0]={:#x}, qh[0]={:#x}",
                            half::f16::from_bits(block.d).to_f32(),
                            block.scales[0],
                            block.ql[0],
                            block.qh[0]
                        );
                    }
                }
            }
        }

        Ok(result)
    }

    fn dequantize_q4_k(&self, data: &[u8], element_count: usize) -> Result<Vec<f32>> {
        const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ4_K>();
        const QK_K: usize = 256;
        let mut result = vec![0.0f32; element_count];

        static FIRST_CALL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
        if FIRST_CALL.swap(false, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "Q4_K dequant: BLOCK_SIZE={}, data.len()={}, element_count={}, expected_blocks={}",
                BLOCK_SIZE,
                data.len(),
                element_count,
                element_count / QK_K
            );
        }

        for (block_idx, block_bytes) in data.chunks_exact(BLOCK_SIZE).enumerate() {
            // Debug first block raw bytes
            if block_idx == 0 {
                eprintln!("  First Q4_K block raw bytes (first 20):");
                eprint!("    ");
                for (i, &byte) in block_bytes.iter().take(20).enumerate() {
                    eprint!("{:02x} ", byte);
                    if i == 1 {
                        eprint!("| "); // After d (2 bytes)
                    } else if i == 3 {
                        eprint!("| "); // After dmin (2 bytes)
                    } else if i == 15 {
                        eprint!("| "); // After scales (12 bytes)
                    }
                }
                eprintln!();

                // Show d and dmin as raw u16 values
                let d_bytes = [block_bytes[0], block_bytes[1]];
                let dmin_bytes = [block_bytes[2], block_bytes[3]];
                let d_raw = u16::from_le_bytes(d_bytes);
                let dmin_raw = u16::from_le_bytes(dmin_bytes);
                eprintln!("    d_raw={:#06x}, dmin_raw={:#06x}", d_raw, dmin_raw);
            }

            let block: BlockQ4_K = unsafe { std::ptr::read(block_bytes.as_ptr() as *const _) };

            let offset = block_idx * QK_K;
            let result_len = result.len();
            let end = (offset + QK_K).min(result_len);
            if offset < result_len {
                dequantize_q4_k(&block, &mut result[offset..end])?;

                // Debug first block
                if block_idx == 0 {
                    let has_nan = result[offset..end].iter().any(|&x| x.is_nan());
                    let has_inf = result[offset..end].iter().any(|&x| x.is_infinite());
                    eprintln!(
                        "  First Q4_K block: d={}, dmin={}, nan={}, inf={}",
                        half::f16::from_bits(block.d).to_f32(),
                        half::f16::from_bits(block.dmin).to_f32(),
                        has_nan,
                        has_inf
                    );
                    eprintln!(
                        "  scales[0]={:#x}, qs[0]={:#x}",
                        block.scales[0], block.qs[0]
                    );
                    let end_idx = (offset + 10).min(result_len);
                    eprintln!(
                        "  First 10 dequantized values: {:?}",
                        &result[offset..end_idx]
                    );
                }
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_loader_creation() {
        let loader = TensorLoader::new(1024);
        assert_eq!(loader.cache_size_bytes(), 0);
        assert_eq!(loader.tensor_names().len(), 0);
    }

    #[test]
    fn test_register_tensor() {
        let mut loader = TensorLoader::new(0);
        let desc = TensorDesc::new(
            "test".to_string(),
            DataType::F32,
            crate::tensor::Shape::new(vec![2, 2]),
            0,
        )
        .unwrap();

        loader.register_tensor("test".to_string(), desc, 0);

        assert_eq!(loader.tensor_names().len(), 1);
        assert!(loader.get_metadata("test").is_some());
    }

    #[test]
    fn test_cache_management() {
        let mut loader = TensorLoader::new(0);

        // Insert some dummy data
        loader
            .cache
            .insert("tensor1".to_string(), vec![1.0, 2.0, 3.0]);
        loader.cache.insert("tensor2".to_string(), vec![4.0, 5.0]);

        assert!(loader.is_cached("tensor1"));
        assert!(loader.is_cached("tensor2"));
        assert!(!loader.is_cached("tensor3"));

        let cache_size = loader.cache_size_bytes();
        assert_eq!(cache_size, 5 * std::mem::size_of::<f32>());

        loader.clear_cache();
        assert_eq!(loader.cache_size_bytes(), 0);
    }
}
