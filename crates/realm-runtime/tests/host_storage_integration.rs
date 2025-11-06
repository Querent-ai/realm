//! Integration test for host-side storage system
//!
//! This test verifies the complete pipeline:
//! 1. Store model in HOST storage (quantized)
//! 2. Retrieve and dequantize tensors
//! 3. Verify memory efficiency

use realm_runtime::model_storage::get_global_model_storage;
use std::fs;

#[test]
#[ignore] // Run with: cargo test --test host_storage_integration -- --ignored
fn test_store_and_retrieve_model() {
    // Skip if model file doesn't exist
    let model_path = "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf";
    if !std::path::Path::new(model_path).exists() {
        println!("Skipping test: model file not found at {}", model_path);
        return;
    }

    println!("📁 Reading model file: {}", model_path);
    let gguf_bytes = fs::read(model_path).expect("Failed to read model file");
    println!("✅ Read {} bytes", gguf_bytes.len());

    // Store model
    println!("💾 Storing model in HOST storage...");
    let model_id = get_global_model_storage()
        .lock()
        .store_model(&gguf_bytes, None)
        .expect("Failed to store model");

    println!("✅ Model stored with ID: {}", model_id);

    // Verify model info
    let model = {
        let storage = get_global_model_storage().lock();
        storage
            .get_model(model_id)
            .expect("Failed to get model")
            .clone()
    };

    println!("📊 Model info:");
    println!("  - Tensors: {}", model.tensor_count());
    println!(
        "  - Total size: {:.2} MB",
        model.total_size as f64 / 1024.0 / 1024.0
    );
    println!("  - Model ID: {}", model.id);

    assert!(model.tensor_count() > 0, "Model should have tensors");
    assert!(model.total_size > 0, "Model should have non-zero size");

    // Test retrieving a specific tensor
    println!("🔍 Testing tensor retrieval...");
    let tensor_name = "token_embd.weight";

    let tensor = {
        let storage = get_global_model_storage().lock();
        let model = storage
            .get_model(model_id)
            .expect("Failed to get model")
            .clone();
        drop(storage);
        model
            .get_tensor(tensor_name)
            .expect("Tensor not found")
            .clone()
    };

    println!("✅ Retrieved tensor '{}'", tensor_name);
    println!("  - Size: {} bytes", tensor.size_bytes());
    println!("  - Shape: {:?}", tensor.shape);
    println!("  - Dtype: {:?}", tensor.dtype);
    println!("  - Elements: {}", tensor.element_count());

    assert!(tensor.size_bytes() > 0, "Tensor should have data");
    assert!(tensor.element_count() > 0, "Tensor should have elements");

    // Test dequantization
    println!("🔧 Testing dequantization...");
    use realm_core::quant::dequantize_tensor;

    let element_count = tensor.element_count() as usize;
    let dequantized = dequantize_tensor(&tensor.data, tensor.dtype, element_count)
        .expect("Failed to dequantize tensor");

    println!("✅ Dequantized {} elements", dequantized.len());
    println!(
        "  - Memory before: {} bytes (quantized)",
        tensor.size_bytes()
    );
    println!("  - Memory after: {} bytes (f32)", dequantized.len() * 4);
    println!(
        "  - Compression: {:.1}×",
        (dequantized.len() * 4) as f64 / tensor.size_bytes() as f64
    );

    assert_eq!(dequantized.len(), element_count);
    assert!(
        dequantized.iter().any(|&x| x != 0.0),
        "Dequantized data should have non-zero values"
    );

    // Cleanup
    println!("🧹 Cleaning up...");
    get_global_model_storage()
        .lock()
        .remove_model(model_id)
        .expect("Failed to remove model");

    println!("✅ Model removed");
    println!("\n🎉 Integration test PASSED!");
}

#[test]
fn test_storage_thread_safety() {
    use std::thread;

    // Create dummy model data
    let dummy_gguf = create_dummy_gguf();

    // Store model
    let model_id = get_global_model_storage()
        .lock()
        .store_model(&dummy_gguf, None)
        .expect("Failed to store dummy model");

    // Spawn multiple threads accessing the same model
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let mid = model_id;
            thread::spawn(move || {
                for _ in 0..100 {
                    let _model = get_global_model_storage().lock().get_model(mid);
                    // Simulate work
                    std::thread::sleep(std::time::Duration::from_micros(10));
                }
                println!("Thread {} completed", i);
            })
        })
        .collect();

    // Wait for all threads
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Cleanup
    get_global_model_storage()
        .lock()
        .remove_model(model_id)
        .expect("Failed to remove dummy model");

    println!("✅ Thread safety test PASSED!");
}

/// Create a minimal valid GGUF file for testing
fn create_dummy_gguf() -> Vec<u8> {
    // Minimal GGUF header:
    // - Magic: "GGUF" (4 bytes)
    // - Version: 3 (u32)
    // - Tensor count: 1 (u64)
    // - Metadata KV count: 0 (u64)

    let mut data = Vec::new();

    // Magic
    data.extend_from_slice(b"GGUF");

    // Version (3)
    data.extend_from_slice(&3u32.to_le_bytes());

    // Tensor count (1)
    data.extend_from_slice(&1u64.to_le_bytes());

    // Metadata KV count (0)
    data.extend_from_slice(&0u64.to_le_bytes());

    // Tensor info:
    // - Name length + name
    // - Dimensions
    // - Type
    // - Offset

    // Name: "test.weight" (11 bytes)
    data.extend_from_slice(&11u64.to_le_bytes());
    data.extend_from_slice(b"test.weight");

    // Dimensions: 2D [32, 32]
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&32u64.to_le_bytes());
    data.extend_from_slice(&32u64.to_le_bytes());

    // Type: Q4_K (12)
    data.extend_from_slice(&12u32.to_le_bytes());

    // Offset: 0
    data.extend_from_slice(&0u64.to_le_bytes());

    // Padding to alignment
    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Tensor data: dummy quantized data (144 bytes for one Q4_K block of 256 elements)
    data.extend_from_slice(&vec![0u8; 144 * 4]); // 4 blocks = 1024 elements

    data
}
