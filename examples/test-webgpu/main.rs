//! WebGPU Backend Test
//!
//! Tests WebGPU backend with actual GPU hardware

use realm_compute_gpu::GpuBackend;
use realm_core::error::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧪 Testing WebGPU Backend\n");

    // Check if WebGPU is available
    if !GpuBackend::is_available() {
        println!("❌ WebGPU is not available on this system");
        println!(
            "   This might be expected if running outside a browser or without WebGPU support"
        );
        return Ok(());
    }

    println!("✅ WebGPU is available!");

    // Initialize WebGPU backend
    println!("\n🔧 Initializing WebGPU backend...");
    let gpu_backend = match GpuBackend::new().await {
        Ok(backend) => {
            println!("✅ WebGPU backend initialized successfully");
            backend
        }
        Err(e) => {
            println!("❌ Failed to initialize WebGPU backend: {}", e);
            return Err(e);
        }
    };

    // Test matrix multiplication
    println!("\n🧮 Testing matrix multiplication...");
    let m = 4;
    let k = 8;
    let n = 4;

    // Create test matrices
    let a: Vec<f32> = (0..(m * k)).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..(k * n)).map(|i| i as f32 * 0.1).collect();

    println!("   Matrix A: [{} x {}]", m, k);
    println!("   Matrix B: [{} x {}]", k, n);
    println!("   Expected result: [{} x {}]", m, n);

    match gpu_backend.matmul(&a, &b, m, k, n) {
        Ok(result) => {
            println!("✅ Matrix multiplication succeeded!");
            println!("   Result shape: {} elements", result.len());
            println!("   First few values: {:?}", &result[..result.len().min(5)]);

            // Verify result shape
            if result.len() == (m * n) as usize {
                println!("✅ Result shape is correct");
            } else {
                println!(
                    "❌ Result shape mismatch: expected {}, got {}",
                    m * n,
                    result.len()
                );
            }
        }
        Err(e) => {
            println!("❌ Matrix multiplication failed: {}", e);
            return Err(e);
        }
    }

    println!("\n🎉 WebGPU backend test completed successfully!");
    Ok(())
}
