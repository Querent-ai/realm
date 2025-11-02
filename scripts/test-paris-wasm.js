#!/usr/bin/env node
/**
 * Test script for Realm WASM + HOST storage
 *
 * This script tests the complete flow:
 * 1. Load WASM module
 * 2. Provide HOST functions for model storage
 * 3. Load TinyLlama model
 * 4. Generate "Paris" from prompt
 */

const fs = require('fs');
const path = require('path');

// Load the Rust runtime library that provides HOST functions
const realmRuntime = require('./target/release/librealm_runtime.so');

// Load WASM module
const wasmPath = '/home/puneet/realm/wasm-pkg/realm_wasm_bg.wasm';
const wasmBytes = fs.readFileSync(wasmPath);

// Model path
const modelPath = '/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf';

console.log('🚀 Realm WASM + HOST Test');
console.log('========================\n');

// Check if model exists
if (!fs.existsSync(modelPath)) {
    console.error(`❌ Model not found: ${modelPath}`);
    process.exit(1);
}

const modelBytes = fs.readFileSync(modelPath);
console.log(`✅ Loaded model: ${modelBytes.length} bytes (${(modelBytes.length / 1024 / 1024).toFixed(2)} MB)\n`);

// Initialize WASM with HOST functions
async function main() {
    const { WASI } = await import('@wasmer/wasi');
    const { init, lowerI64Imports } = await import('@wasmer/wasm-transformer');

    await init();

    // Transform WASM to handle i64 imports
    const transformedBytes = await lowerI64Imports(wasmBytes);

    // Create WASI instance
    const wasi = new WASI({
        env: {},
        args: [],
    });

    // Compile WASM module
    const module = await WebAssembly.compile(transformedBytes);

    // HOST function implementations
    // These would normally be provided by the Rust runtime via Neon/FFI
    let modelStorage = new Map();
    let nextModelId = 1;

    const hostFunctions = {
        env: {
            realm_store_model: (ggufPtr, ggufLen, modelId) => {
                console.log(`🔧 HOST: realm_store_model(ptr=${ggufPtr}, len=${ggufLen}, id=${modelId})`);

                // In real implementation, this would call into Rust runtime
                // For now, we'll simulate success
                const id = modelId > 0 ? modelId : nextModelId++;

                // Store model metadata (in real impl, this is in Rust)
                modelStorage.set(id, {
                    size: ggufLen,
                    tensorCount: 201, // TinyLlama has 201 tensors
                });

                console.log(`✅ HOST: Stored model with ID ${id}`);
                return id;
            },

            realm_get_tensor: (modelId, tensorNamePtr, tensorNameLen, outPtr, outMaxLen) => {
                // This would retrieve and dequantize tensor from Rust storage
                console.log(`🔧 HOST: realm_get_tensor(id=${modelId}, name_len=${tensorNameLen}, out_max=${outMaxLen})`);

                // For now, return error (real impl needed)
                return -1;
            },

            realm_get_model_info: (modelId, outTensorCountPtr, outTotalSizePtr) => {
                console.log(`🔧 HOST: realm_get_model_info(id=${modelId})`);

                const model = modelStorage.get(modelId);
                if (!model) {
                    return -1;
                }

                // Write metadata to WASM memory (simplified)
                return 0;
            },

            realm_remove_model: (modelId) => {
                console.log(`🔧 HOST: realm_remove_model(id=${modelId})`);
                modelStorage.delete(modelId);
                return 0;
            },
        },
    };

    // Instantiate WASM with HOST functions
    const instance = await WebAssembly.instantiate(module, {
        ...wasi.getImports(module),
        ...hostFunctions,
    });

    wasi.start(instance);

    // Access WASM exports
    const { Realm, memory } = instance.exports;

    console.log('✅ WASM module loaded and instantiated\n');

    // Create Realm instance
    console.log('📦 Creating Realm instance...');
    const realm = Realm.new();
    console.log('✅ Realm instance created\n');

    // Load model
    console.log('🔄 Loading model into HOST storage...');
    try {
        // Copy model bytes to WASM memory
        const modelPtr = instance.exports.__wbindgen_malloc(modelBytes.length);
        const wasmMem = new Uint8Array(memory.buffer);
        wasmMem.set(modelBytes, modelPtr);

        // Call loadModel
        realm.loadModel(modelBytes);
        console.log('✅ Model loaded successfully\n');
    } catch (error) {
        console.error('❌ Failed to load model:', error.message);
        return;
    }

    // Generate text
    console.log('🎯 Generating text from prompt...');
    const prompt = 'What is the capital of France?';
    console.log(`Prompt: "${prompt}"\n`);

    try {
        const response = realm.generate(prompt);
        console.log('✅ Generation successful!\n');
        console.log(`Response: "${response}"\n`);

        // Check if response contains "Paris"
        if (response.toLowerCase().includes('paris')) {
            console.log('🎉 SUCCESS: Model correctly identified Paris as the capital of France!');
        } else {
            console.log('⚠️  Response did not mention Paris');
        }
    } catch (error) {
        console.error('❌ Generation failed:', error.message);
    }
}

main().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
});
