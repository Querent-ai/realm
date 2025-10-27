/// Additional tensor kernels (softmax, layernorm, etc.)
use realm_core::error::Result;

/// Softmax activation: `output[i] = exp(input[i]) / sum(exp(input))`
pub fn softmax(input: &[f32], output: &mut [f32]) -> Result<()> {
    assert_eq!(input.len(), output.len());

    // Find max for numerical stability
    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let mut sum = 0.0;
    for (i, &x) in input.iter().enumerate() {
        let exp_val = (x - max).exp();
        output[i] = exp_val;
        sum += exp_val;
    }

    // Normalize
    for val in output.iter_mut() {
        *val /= sum;
    }

    Ok(())
}

/// ReLU activation: `output[i] = max(0, input[i])`
pub fn relu(input: &[f32], output: &mut [f32]) -> Result<()> {
    assert_eq!(input.len(), output.len());

    for (i, &x) in input.iter().enumerate() {
        output[i] = x.max(0.0);
    }

    Ok(())
}

/// GELU activation (approximate): x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
pub fn gelu(input: &[f32], output: &mut [f32]) -> Result<()> {
    assert_eq!(input.len(), output.len());

    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;

    for (i, &x) in input.iter().enumerate() {
        let x3 = x * x * x;
        let inner = SQRT_2_OVER_PI * (x + COEFF * x3);
        output[i] = 0.5 * x * (1.0 + inner.tanh());
    }

    Ok(())
}

/// Layer normalization
pub fn layer_norm(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Result<()> {
    assert_eq!(input.len(), output.len());
    assert_eq!(input.len(), gamma.len());
    assert_eq!(input.len(), beta.len());

    // Compute mean
    let mean: f32 = input.iter().sum::<f32>() / input.len() as f32;

    // Compute variance
    let variance: f32 = input.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / input.len() as f32;

    let std = (variance + eps).sqrt();

    // Normalize and scale
    for (i, &x) in input.iter().enumerate() {
        output[i] = gamma[i] * ((x - mean) / std) + beta[i];
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];

        softmax(&input, &mut output).unwrap();

        // Check sum is 1.0
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check monotonicity
        assert!(output[0] < output[1]);
        assert!(output[1] < output[2]);
    }

    #[test]
    fn test_relu() {
        let input = vec![-1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 4];

        relu(&input, &mut output).unwrap();

        assert_eq!(output, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_gelu() {
        let input = vec![0.0, 1.0, -1.0];
        let mut output = vec![0.0; 3];

        gelu(&input, &mut output).unwrap();

        // GELU(0) ≈ 0
        assert!(output[0].abs() < 1e-6);

        // GELU(1) should be positive and close to 1
        assert!(output[1] > 0.8 && output[1] < 0.9);

        // GELU(-1) should be negative
        assert!(output[2] < 0.0);
    }
}
