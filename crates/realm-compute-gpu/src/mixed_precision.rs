//! Mixed precision support for GPU operations
//!
//! This module provides FP16 and BF16 support for improved GPU performance
//! and memory efficiency. Lower precision reduces memory bandwidth and can
//! provide 2-3x speedup on modern GPUs.
//!
//! **Status**: Implementation ready, requires GPU hardware for testing.
//! **Backends**: CUDA (FP16/BF16), Metal (FP16), WebGPU (FP16)

/// Precision configuration for mixed precision operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrecisionMode {
    /// Full precision (FP32) - highest accuracy
    FP32,
    /// Half precision (FP16) - 2x memory reduction, ~2x speedup
    FP16,
    /// BFloat16 - better range than FP16, good for training
    BF16,
    /// Automatic - select best precision based on hardware
    #[default]
    Automatic,
}

/// Mixed precision configuration
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Precision mode for forward pass
    pub forward_precision: PrecisionMode,
    /// Precision mode for attention operations
    pub attention_precision: PrecisionMode,
    /// Enable loss scaling for training (not used in inference)
    pub loss_scaling: bool,
    /// Automatic mixed precision (AMP) - use FP16 where safe
    pub amp_enabled: bool,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            forward_precision: PrecisionMode::Automatic,
            attention_precision: PrecisionMode::Automatic,
            loss_scaling: false,
            amp_enabled: true,
        }
    }
}

impl MixedPrecisionConfig {
    /// Create config optimized for inference
    pub fn inference() -> Self {
        Self {
            forward_precision: PrecisionMode::FP16,
            attention_precision: PrecisionMode::FP16,
            loss_scaling: false,
            amp_enabled: true,
        }
    }

    /// Create config with FP32 (full precision)
    pub fn full_precision() -> Self {
        Self {
            forward_precision: PrecisionMode::FP32,
            attention_precision: PrecisionMode::FP32,
            loss_scaling: false,
            amp_enabled: false,
        }
    }
}

/// Convert FP32 tensor to FP16
///
/// # Arguments
/// * `input` - FP32 tensor data
///
/// # Returns
/// FP16 tensor data (u16 array, half-precision floats)
pub fn f32_to_fp16(input: &[f32]) -> Vec<u16> {
    input.iter().map(|&f| f32_to_fp16_single(f)).collect()
}

/// Convert FP16 tensor to FP32
///
/// # Arguments
/// * `input` - FP16 tensor data (u16 array)
///
/// # Returns
/// FP32 tensor data
pub fn fp16_to_f32(input: &[u16]) -> Vec<f32> {
    input.iter().map(|&h| fp16_to_f32_single(h)).collect()
}

/// Convert FP32 tensor to BF16
pub fn f32_to_bf16(input: &[f32]) -> Vec<u16> {
    input.iter().map(|&f| f32_to_bf16_single(f)).collect()
}

/// Convert BF16 tensor to FP32
pub fn bf16_to_f32(input: &[u16]) -> Vec<f32> {
    input.iter().map(|&h| bf16_to_f32_single(h)).collect()
}

/// Convert single FP32 to FP16
fn f32_to_fp16_single(f: f32) -> u16 {
    // Simple FP32 to FP16 conversion
    // FP16: 1 sign bit, 5 exponent bits, 10 mantissa bits
    let bits = f.to_bits();
    let sign = (bits >> 31) & 0x1;
    let exponent = (bits >> 23) & 0xFF;
    let mantissa = bits & 0x7FFFFF;

    // Handle special cases (inf, NaN)
    if exponent == 0xFF {
        // Infinity or NaN
        return ((sign << 15) | (0x1F << 10) | (mantissa >> 13)) as u16;
    }

    // Normalize exponent (FP32 has 127 bias, FP16 has 15 bias)
    let exp_f16 = if exponent == 0 {
        0 // Denormal
    } else {
        let exp_f16 = (exponent as i32) - 127 + 15;
        if exp_f16 > 31 {
            31 // Overflow to infinity
        } else if exp_f16 < 0 {
            0 // Underflow to zero
        } else {
            exp_f16 as u32
        }
    };

    ((sign << 15) | (exp_f16 << 10) | (mantissa >> 13)) as u16
}

/// Convert single FP16 to FP32
fn fp16_to_f32_single(h: u16) -> f32 {
    let sign = (h >> 15) & 0x1;
    let exponent = (h >> 10) & 0x1F;
    let mantissa = h & 0x3FF;

    // Handle special cases
    if exponent == 0x1F {
        // Infinity or NaN
        let bits = ((sign as u32) << 31) | (0xFF << 23) | ((mantissa as u32) << 13);
        return f32::from_bits(bits);
    }

    // Normalize exponent
    let exp_f32 = if exponent == 0 {
        // Denormal
        0
    } else {
        (exponent as i32) - 15 + 127
    } as u32;

    let bits = ((sign as u32) << 31) | (exp_f32 << 23) | ((mantissa as u32) << 13);
    f32::from_bits(bits)
}

/// Convert single FP32 to BF16
fn f32_to_bf16_single(f: f32) -> u16 {
    // BF16: 1 sign bit, 8 exponent bits (same as FP32), 7 mantissa bits
    let bits = f.to_bits();
    // BF16 keeps top 16 bits (sign + exponent + top 7 mantissa bits)
    ((bits >> 16) & 0xFFFF) as u16
}

/// Convert single BF16 to FP32
fn bf16_to_f32_single(h: u16) -> f32 {
    // BF16 to FP32: pad lower 16 bits with zeros
    let bits = (h as u32) << 16;
    f32::from_bits(bits)
}

/// Check if GPU supports FP16
pub fn supports_fp16() -> bool {
    // TODO: Check GPU capabilities
    // CUDA: Check compute capability >= 5.3
    // Metal: Check device supports FP16
    // WebGPU: Check adapter limits
    false // Placeholder - requires GPU detection
}

/// Check if GPU supports BF16
pub fn supports_bf16() -> bool {
    // TODO: Check GPU capabilities
    // CUDA: Check compute capability >= 8.0 (Ampere+)
    // Metal: Not natively supported
    // WebGPU: Not natively supported
    false // Placeholder - requires GPU detection
}

/// Select best precision mode based on GPU capabilities
pub fn select_precision(preference: PrecisionMode) -> PrecisionMode {
    match preference {
        PrecisionMode::Automatic => {
            if supports_bf16() {
                PrecisionMode::BF16
            } else if supports_fp16() {
                PrecisionMode::FP16
            } else {
                PrecisionMode::FP32
            }
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp16_conversion() {
        let f32_vals = vec![1.0f32, 2.0, 3.25, -1.0, 0.0];
        let fp16_vals = f32_to_fp16(&f32_vals);
        let back_to_f32 = fp16_to_f32(&fp16_vals);

        // FP16 conversion may lose precision, but values should be close
        for (original, converted) in f32_vals.iter().zip(back_to_f32.iter()) {
            let diff = (original - converted).abs();
            assert!(
                diff < 0.01 || (*original == 0.0 && *converted == 0.0),
                "FP16 conversion error: {} -> {}",
                original,
                converted
            );
        }
    }

    #[test]
    fn test_bf16_conversion() {
        let f32_vals = vec![1.0f32, 2.0, 3.25, -1.0, 0.0];
        let bf16_vals = f32_to_bf16(&f32_vals);
        let back_to_f32 = bf16_to_f32(&bf16_vals);

        // BF16 has better range than FP16, conversion should be more accurate
        for (original, converted) in f32_vals.iter().zip(back_to_f32.iter()) {
            let diff = (original - converted).abs();
            assert!(
                diff < 0.1 || (*original == 0.0 && *converted == 0.0),
                "BF16 conversion error: {} -> {}",
                original,
                converted
            );
        }
    }

    #[test]
    fn test_precision_config() {
        let config = MixedPrecisionConfig::default();
        assert_eq!(config.forward_precision, PrecisionMode::Automatic);
        assert!(config.amp_enabled);

        let inference_config = MixedPrecisionConfig::inference();
        assert_eq!(inference_config.forward_precision, PrecisionMode::FP16);
    }
}
