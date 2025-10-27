/// GGUF (GPT-Generated Unified Format) streaming parser
///
/// Specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
use crate::error::{Error, Result};
use crate::tensor::{DataType, Shape, TensorDesc};
use std::io::{Read, Seek, SeekFrom};

/// GGUF magic number
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"

/// GGUF metadata value types
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum GGUFValueType {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

/// Metadata value
#[derive(Debug, Clone)]
pub enum MetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    UInt64(u64),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
}

impl MetadataValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            MetadataValue::UInt8(v) => Some(*v as u32),
            MetadataValue::UInt16(v) => Some(*v as u32),
            MetadataValue::UInt32(v) => Some(*v),
            MetadataValue::UInt64(v) => Some(*v as u32),
            MetadataValue::Int8(v) => Some(*v as u32),
            MetadataValue::Int16(v) => Some(*v as u32),
            MetadataValue::Int32(v) => Some(*v as u32),
            MetadataValue::Int64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetadataValue::Float32(v) => Some(*v),
            MetadataValue::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[MetadataValue]> {
        match self {
            MetadataValue::Array(arr) => Some(arr),
            _ => None,
        }
    }
}

/// Model metadata extracted from GGUF header
#[derive(Debug, Clone)]
pub struct ModelMeta {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub architecture: String,
    pub vocab_size: Option<u32>,
    pub tensors: Vec<TensorDesc>,
    pub metadata: std::collections::HashMap<String, MetadataValue>,
}

/// GGUF streaming parser
pub struct GGUFParser<R: Read + Seek> {
    reader: R,
    meta: Option<ModelMeta>,
}

impl<R: Read + Seek> GGUFParser<R> {
    pub fn new(reader: R) -> Self {
        Self { reader, meta: None }
    }

    /// Parse GGUF header and extract metadata
    pub fn parse_header(&mut self) -> Result<ModelMeta> {
        // Read magic
        let magic = self.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(Error::InvalidFormat(format!(
                "Invalid GGUF magic: 0x{:08X}",
                magic
            )));
        }

        // Read version
        let version = self.read_u32()?;
        if !(2..=3).contains(&version) {
            return Err(Error::InvalidFormat(format!(
                "Unsupported GGUF version: {}",
                version
            )));
        }

        // Read counts
        let tensor_count = self.read_u64()?;
        let metadata_kv_count = self.read_u64()?;

        // Parse metadata key-values (simplified - skip for now)
        // In a real implementation, we'd parse these to extract architecture, vocab, etc.
        let metadata = self.parse_metadata(metadata_kv_count)?;

        // Parse tensor info
        let tensors = self.parse_tensor_info(tensor_count)?;

        // Extract commonly used fields
        let architecture = metadata
            .get("general.architecture")
            .and_then(|v| v.as_string())
            .unwrap_or("unknown")
            .to_string();

        let vocab_size = metadata
            .get(&format!("{}.vocab_size", architecture))
            .or_else(|| metadata.get("vocab_size"))
            .and_then(|v| v.as_u32());

        let meta = ModelMeta {
            version,
            tensor_count,
            metadata_kv_count,
            architecture,
            vocab_size,
            tensors,
            metadata,
        };

        self.meta = Some(meta.clone());
        Ok(meta)
    }

    /// Get current position for tensor data (aligned to 32 bytes)
    pub fn tensor_data_offset(&mut self) -> Result<u64> {
        const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

        let current_pos = self.reader.stream_position().map_err(Error::from)?;

        // Align to 32-byte boundary
        let aligned_pos =
            (current_pos + GGUF_DEFAULT_ALIGNMENT - 1) & !(GGUF_DEFAULT_ALIGNMENT - 1);

        Ok(aligned_pos)
    }

    /// Read tensor data chunk
    pub fn read_tensor_data(&mut self, offset: u64, size: usize) -> Result<Vec<u8>> {
        self.reader.seek(SeekFrom::Start(offset))?;
        let mut buffer = vec![0u8; size];
        self.reader.read_exact(&mut buffer)?;
        Ok(buffer)
    }

    /// Get data offset (for TensorLoader)
    pub fn data_offset(&self) -> u64 {
        // This would be calculated after parsing header
        // For now, return a placeholder
        0
    }

    /// Get tensor information
    pub fn tensor_info(&self) -> Vec<(String, TensorDesc, u64)> {
        if let Some(meta) = &self.meta {
            meta.tensors
                .iter()
                .enumerate()
                .map(|(i, desc)| (format!("tensor_{}", i), desc.clone(), 0))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get metadata (if parsed)
    pub fn metadata(&self) -> Option<&ModelMeta> {
        self.meta.as_ref()
    }

    /// Extract transformer config from metadata
    pub fn extract_config(&self) -> Option<crate::TransformerConfigData> {
        let meta = self.meta.as_ref()?;
        let arch = &meta.architecture;

        Some(crate::TransformerConfigData {
            vocab_size: meta
                .metadata
                .get(&format!("{}.vocab_size", arch))
                .or_else(|| meta.metadata.get("vocab_size"))
                .and_then(|v| v.as_u32())
                .unwrap_or(32000) as usize,

            hidden_size: meta
                .metadata
                .get(&format!("{}.embedding_length", arch))
                .or_else(|| meta.metadata.get(&format!("{}.hidden_size", arch)))
                .and_then(|v| v.as_u32())
                .unwrap_or(2048) as usize,

            num_layers: meta
                .metadata
                .get(&format!("{}.block_count", arch))
                .or_else(|| meta.metadata.get(&format!("{}.num_layers", arch)))
                .and_then(|v| v.as_u32())
                .unwrap_or(22) as usize,

            num_heads: meta
                .metadata
                .get(&format!("{}.attention.head_count", arch))
                .or_else(|| meta.metadata.get(&format!("{}.num_heads", arch)))
                .and_then(|v| v.as_u32())
                .unwrap_or(32) as usize,

            num_kv_heads: meta
                .metadata
                .get(&format!("{}.attention.head_count_kv", arch))
                .or_else(|| meta.metadata.get(&format!("{}.num_kv_heads", arch)))
                .and_then(|v| v.as_u32())
                .unwrap_or(4) as usize,

            intermediate_size: meta
                .metadata
                .get(&format!("{}.feed_forward_length", arch))
                .or_else(|| meta.metadata.get(&format!("{}.intermediate_size", arch)))
                .and_then(|v| v.as_u32())
                .unwrap_or(5632) as usize,

            max_seq_len: meta
                .metadata
                .get(&format!("{}.context_length", arch))
                .or_else(|| meta.metadata.get(&format!("{}.max_seq_len", arch)))
                .and_then(|v| v.as_u32())
                .unwrap_or(2048) as usize,

            rms_norm_eps: meta
                .metadata
                .get(&format!("{}.attention.layer_norm_rms_epsilon", arch))
                .or_else(|| meta.metadata.get("rms_norm_eps"))
                .and_then(|v| v.as_f32())
                .unwrap_or(1e-5),

            rope_theta: meta
                .metadata
                .get(&format!("{}.rope.freq_base", arch))
                .or_else(|| meta.metadata.get("rope_theta"))
                .and_then(|v| v.as_f32())
                .unwrap_or(10000.0),
        })
    }

    // Helper methods

    fn read_u32(&mut self) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let mut buf = vec![0u8; len];
        self.reader.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(|e| Error::ParseError(format!("Invalid UTF-8: {}", e)))
    }

    fn read_i8(&mut self) -> Result<i8> {
        let mut buf = [0u8; 1];
        self.reader.read_exact(&mut buf)?;
        Ok(i8::from_le_bytes(buf))
    }

    fn read_u8(&mut self) -> Result<u8> {
        let mut buf = [0u8; 1];
        self.reader.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    fn read_u16(&mut self) -> Result<u16> {
        let mut buf = [0u8; 2];
        self.reader.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_i16(&mut self) -> Result<i16> {
        let mut buf = [0u8; 2];
        self.reader.read_exact(&mut buf)?;
        Ok(i16::from_le_bytes(buf))
    }

    fn read_i32(&mut self) -> Result<i32> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_i64(&mut self) -> Result<i64> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    }

    fn read_f32(&mut self) -> Result<f32> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_f64(&mut self) -> Result<f64> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }

    fn parse_metadata(
        &mut self,
        count: u64,
    ) -> Result<std::collections::HashMap<String, MetadataValue>> {
        let mut map = std::collections::HashMap::new();

        for _ in 0..count {
            let key = self.read_string()?;
            let value = self.read_metadata_value()?;
            map.insert(key, value);
        }

        Ok(map)
    }

    fn read_metadata_value(&mut self) -> Result<MetadataValue> {
        let value_type = self.read_u32()?;

        match value_type {
            0 => Ok(MetadataValue::UInt8(self.read_u8()?)),
            1 => Ok(MetadataValue::Int8(self.read_i8()?)),
            2 => Ok(MetadataValue::UInt16(self.read_u16()?)),
            3 => Ok(MetadataValue::Int16(self.read_i16()?)),
            4 => Ok(MetadataValue::UInt32(self.read_u32()?)),
            5 => Ok(MetadataValue::Int32(self.read_i32()?)),
            6 => Ok(MetadataValue::Float32(self.read_f32()?)),
            7 => Ok(MetadataValue::Bool(self.read_u8()? != 0)),
            8 => Ok(MetadataValue::String(self.read_string()?)),
            9 => {
                // Array
                let element_type = self.read_u32()?;
                let array_len = self.read_u64()? as usize;
                let mut array = Vec::with_capacity(array_len);

                for _ in 0..array_len {
                    // Temporarily set the reader state to read the element
                    let value = self.read_typed_value(element_type)?;
                    array.push(value);
                }

                Ok(MetadataValue::Array(array))
            }
            10 => Ok(MetadataValue::UInt64(self.read_u64()?)),
            11 => Ok(MetadataValue::Int64(self.read_i64()?)),
            12 => Ok(MetadataValue::Float64(self.read_f64()?)),
            _ => Err(Error::ParseError(format!(
                "Unknown metadata value type: {}",
                value_type
            ))),
        }
    }

    fn read_typed_value(&mut self, value_type: u32) -> Result<MetadataValue> {
        match value_type {
            0 => Ok(MetadataValue::UInt8(self.read_u8()?)),
            1 => Ok(MetadataValue::Int8(self.read_i8()?)),
            2 => Ok(MetadataValue::UInt16(self.read_u16()?)),
            3 => Ok(MetadataValue::Int16(self.read_i16()?)),
            4 => Ok(MetadataValue::UInt32(self.read_u32()?)),
            5 => Ok(MetadataValue::Int32(self.read_i32()?)),
            6 => Ok(MetadataValue::Float32(self.read_f32()?)),
            7 => Ok(MetadataValue::Bool(self.read_u8()? != 0)),
            8 => Ok(MetadataValue::String(self.read_string()?)),
            10 => Ok(MetadataValue::UInt64(self.read_u64()?)),
            11 => Ok(MetadataValue::Int64(self.read_i64()?)),
            12 => Ok(MetadataValue::Float64(self.read_f64()?)),
            _ => Err(Error::ParseError(format!(
                "Unknown typed value: {}",
                value_type
            ))),
        }
    }

    fn parse_tensor_info(&mut self, count: u64) -> Result<Vec<TensorDesc>> {
        let mut tensors = Vec::new();

        for _ in 0..count {
            let name = self.read_string()?;

            // Read n_dims
            let n_dims = self.read_u32()? as usize;

            // Read dimensions
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(self.read_u64()? as usize);
            }

            // Read dtype
            let dtype_val = self.read_u32()?;
            let dtype = self.parse_dtype(dtype_val)?;

            // Read offset (relative to tensor data section)
            let offset = self.read_u64()?;

            let shape = Shape::new(dims);
            // Use offset directly as read from file - it's already relative to tensor_data
            let desc = TensorDesc::new(name, dtype, shape, offset)?;

            tensors.push(desc);
        }

        Ok(tensors)
    }

    fn parse_dtype(&self, val: u32) -> Result<DataType> {
        match val {
            0 => Ok(DataType::F32),
            1 => Ok(DataType::F16),
            2 => Ok(DataType::Q4_0),
            3 => Ok(DataType::Q4_1),
            6 => Ok(DataType::Q5_0),
            7 => Ok(DataType::Q5_1),
            8 => Ok(DataType::Q8_0),
            9 => Ok(DataType::Q8_1),
            10 => Ok(DataType::Q2_K),
            11 => Ok(DataType::Q3_K),
            12 => Ok(DataType::Q4_K),
            13 => Ok(DataType::Q5_K),
            14 => Ok(DataType::Q6_K),
            // Unknown types are treated as unsupported but don't fail parsing
            _ => {
                eprintln!(
                    "Warning: Unsupported dtype: {}, treating as Unsupported",
                    val
                );
                Ok(DataType::Unsupported)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_gguf_magic() {
        // Create minimal GGUF header
        const GGUF_VERSION: u32 = 3;
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count

        let cursor = Cursor::new(data);
        let mut parser = GGUFParser::new(cursor);

        let meta = parser.parse_header().unwrap();
        assert_eq!(meta.version, GGUF_VERSION);
        assert_eq!(meta.tensor_count, 0);
    }
}
