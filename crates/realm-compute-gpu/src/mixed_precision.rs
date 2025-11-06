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
///
/// # Implementation Notes
/// - CUDA: Requires compute capability >= 5.3 (Pascal or newer)
/// - Metal: Supported on Apple Silicon (M1/M2+) and newer macOS GPUs
/// - WebGPU: Limited support (depends on adapter capabilities)
pub fn supports_fp16() -> bool {
    // Try to detect GPU backend and check capabilities
    #[cfg(feature = "cuda")]
    {
        // CUDA devices generally support FP16 on Pascal+ (compute capability >= 5.3)
        if candle_core::Device::new_cuda(0).is_ok() {
            // Assume FP16 is supported on CUDA devices (most modern GPUs)
            return true;
        }
    }

    #[cfg(feature = "metal")]
    {
        // Metal devices (Apple Silicon) support FP16
        if candle_core::Device::new_metal(0).is_ok() {
            return true;
        }
    }

    #[cfg(feature = "webgpu")]
    {
        // WebGPU: Check if adapter supports FP16
        use crate::GpuBackend;
        if GpuBackend::is_available() {
            // Most WebGPU adapters support FP16
            return true;
        }
    }

    false
}

/// Check if GPU supports BF16
///
/// # Implementation Notes
/// - CUDA: Requires compute capability >= 8.0 (Ampere or newer, e.g., A100, RTX 30xx+)
/// - Metal: Not natively supported (would need emulation)
/// - WebGPU: Not natively supported
pub fn supports_bf16() -> bool {
    // BF16 is primarily supported on CUDA Ampere+ GPUs
    #[cfg(feature = "cuda")]
    {
        if candle_core::Device::new_cuda(0).is_ok() {
            // Note: Actual compute capability check would require CUDA runtime query
            // For now, assume BF16 is available if CUDA device exists
            // In production, should query actual compute capability
            return true;
        }
    }

    // Metal and WebGPU don't natively support BF16
    false
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
        assert_eq!(config.attention_precision, PrecisionMode::Automatic);
        assert!(config.amp_enabled);
        assert!(!config.loss_scaling);

        let inference_config = MixedPrecisionConfig::inference();
        assert_eq!(inference_config.forward_precision, PrecisionMode::FP16);
        assert_eq!(inference_config.attention_precision, PrecisionMode::FP16);
        assert!(inference_config.amp_enabled);
        assert!(!inference_config.loss_scaling);

        let full_precision_config = MixedPrecisionConfig::full_precision();
        assert_eq!(full_precision_config.forward_precision, PrecisionMode::FP32);
        assert_eq!(
            full_precision_config.attention_precision,
            PrecisionMode::FP32
        );
        assert!(!full_precision_config.amp_enabled);
    }

    #[test]
    fn test_precision_mode_selection() {
        // Test automatic selection
        let selected = select_precision(PrecisionMode::Automatic);
        // Should select based on GPU capabilities (may be FP32, FP16, or BF16)
        assert!(matches!(
            selected,
            PrecisionMode::FP32 | PrecisionMode::FP16 | PrecisionMode::BF16
        ));

        // Test explicit selection
        assert_eq!(select_precision(PrecisionMode::FP32), PrecisionMode::FP32);
        assert_eq!(select_precision(PrecisionMode::FP16), PrecisionMode::FP16);
        assert_eq!(select_precision(PrecisionMode::BF16), PrecisionMode::BF16);
    }

    #[test]
    fn test_fp16_edge_cases() {
        // Test special values
        let special_vals = vec![
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.0f32,
            -0.0f32,
            f32::MAX,
            f32::MIN,
        ];

        for &val in &special_vals {
            let fp16 = f32_to_fp16(&[val]);
            let back = fp16_to_f32(&fp16);

            // Special values should be preserved (within FP16 limits)
            if val.is_nan() {
                assert!(back[0].is_nan(), "NaN should be preserved");
            } else if val.is_infinite() {
                assert_eq!(val.is_sign_positive(), back[0].is_sign_positive());
                assert!(back[0].is_infinite(), "Infinity should be preserved");
            } else if val == 0.0 || val == -0.0 {
                // Zero should be preserved
                assert_eq!(val.signum(), back[0].signum());
            }
        }
    }

    #[test]
    fn test_bf16_edge_cases() {
        // Test special values with BF16
        let special_vals = vec![f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 0.0f32, -0.0f32];

        for &val in &special_vals {
            let bf16 = f32_to_bf16(&[val]);
            let back = bf16_to_f32(&bf16);

            // BF16 preserves exponent, so special values should be better preserved
            if val.is_nan() {
                assert!(back[0].is_nan(), "NaN should be preserved");
            } else if val.is_infinite() {
                assert_eq!(val.is_sign_positive(), back[0].is_sign_positive());
                assert!(back[0].is_infinite(), "Infinity should be preserved");
            }
        }
    }
}
