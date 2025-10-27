//! Integration tests for Realm
//!
//! These tests verify that all components work together correctly.

use wasmtime::*;

#[test]
fn test_wasm_module_loads() {
    let wasm_path = "../crates/realm-wasm/pkg/realm_wasm_bg.wasm";

    // Check if WASM module exists
    if !std::path::Path::new(wasm_path).exists() {
        eprintln!("WASM module not built yet, run: cd crates/realm-wasm && wasm-pack build --target web");
        return;
    }

    let mut config = Config::new();
    config.wasm_bulk_memory(true);
    config.wasm_multi_memory(true);

    let engine = Engine::new(&config).expect("Failed to create engine");
    let module = Module::from_file(&engine, wasm_path).expect("Failed to load WASM module");

    // Verify module has expected exports
    let exports: Vec<_> = module.exports().map(|e| e.name().to_string()).collect();

    // Should export Realm constructor and methods
    assert!(exports.iter().any(|e| e.contains("Realm")), "Should export Realm struct");
}

#[test]
fn test_host_functions_linkable() {
    let mut config = Config::new();
    config.wasm_bulk_memory(true);

    let engine = Engine::new(&config).expect("Failed to create engine");
    let mut linker = Linker::new(&engine);

    // Define host functions - these should be linkable
    linker
        .func_wrap(
            "realm_host",
            "candle_matmul",
            |_: Caller<'_, ()>,
             _a_ptr: i32,
             _a_len: i32,
             _b_ptr: i32,
             _b_len: i32,
             _m: i32,
             _k: i32,
             _n: i32,
             _result_ptr: i32| -> i32 { 0 },
        )
        .expect("Failed to define candle_matmul");

    linker
        .func_wrap(
            "realm_host",
            "memory64_load_layer",
            |_: Caller<'_, ()>, _model_id: i32, _layer_id: i32, _buffer_ptr: i32, _buffer_len: i32| -> i32 { 0 },
        )
        .expect("Failed to define memory64_load_layer");

    linker
        .func_wrap(
            "realm_host",
            "memory64_store_layer",
            |_: Caller<'_, ()>, _model_id: i32, _layer_id: i32, _buffer_ptr: i32, _buffer_len: i32| -> i32 { 0 },
        )
        .expect("Failed to define memory64_store_layer");
}

#[test]
fn test_crate_dependencies() {
    // This test just verifies that all crates can be imported together
    use realm_core::error::Error;
    use realm_compute_cpu::cpu_backend_trait::CpuBackendTrait;
    use realm_compute_gpu::gpu_backend_trait::GpuBackendTrait;
    use realm_models::config::TransformerConfig;
    use realm_runtime::Memory64Runtime;

    // If we can construct these types, dependencies are correct
    let _config = TransformerConfig {
        vocab_size: 32000,
        hidden_size: 4096,
        intermediate_size: 11008,
        num_hidden_layers: 32,
        num_attention_heads: 32,
        num_key_value_heads: 32,
        max_position_embeddings: 2048,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
    };

    // Verify error types work
    let _err: Error = Error::InvalidShape("test".to_string());
}
