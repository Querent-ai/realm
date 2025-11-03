#!/usr/bin/env node
/**
 * Realm JS Example: Load large model and generate "Paris"
 *
 * This example demonstrates:
 * 1. Loading WASM module with HOST-side storage
 * 2. Providing HOST functions for model management
 * 3. Loading TinyLlama (637MB quantized)
 * 4. Generating "Paris" from prompt
 */

import { init, Wasmer } from '@wasmer/sdk';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { createRequire } from 'module';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const require = createRequire(import.meta.url);

// Paths
const WASM_PATH = join(__dirname, '../../wasm-pkg/realm_wasm_bg.wasm');
const MODEL_PATH = process.argv[2] || '/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf';

console.log('🚀 Realm JS + WASM + HOST Example');
console.log('==================================\n');

// Load native library for HOST functions
let realmHost;
try {
    // Try to load the compiled Rust library
    realmHost = require('../../target/release/librealm_runtime.so');
    console.log('✅ Loaded native HOST library\n');
} catch (err) {
    console.log('⚠️  Native library not found, using mock HOST functions\n');

    // Mock HOST functions for testing
    realmHost = {
        realm_store_model: (ptr, len, id) => {
            console.log(`  [MOCK] realm_store_model(len=${len}, id=${id})`);
            return id > 0 ? id : 1;
        },
        realm_get_tensor: (modelId, namePtr, nameLen, outPtr, outMax) => {
            console.log(`  [MOCK] realm_get_tensor(id=${modelId}, name_len=${nameLen})`);
            return -1; // Not implemented in mock
        },
        realm_get_model_info: (modelId, tensorCountPtr, totalSizePtr) => {
            console.log(`  [MOCK] realm_get_model_info(id=${modelId})`);
            return 0;
        },
        realm_remove_model: (modelId) => {
            console.log(`  [MOCK] realm_remove_model(id=${modelId})`);
            return 0;
        },
    };
}

async function main() {
    // Initialize Wasmer
    await init();
    console.log('✅ Wasmer SDK initialized\n');

    // Load WASM module
    const wasmBytes = readFileSync(WASM_PATH);
    console.log(`✅ Loaded WASM module: ${wasmBytes.length} bytes\n`);

    // Load model file
    const modelBytes = readFileSync(MODEL_PATH);
    console.log(`✅ Loaded model: ${(modelBytes.length / 1024 / 1024).toFixed(2)} MB\n`);

    // Create Wasmer instance with HOST functions
    const wasmer = await Wasmer.fromFile(WASM_PATH);

    // Get the module
    const module = await wasmer.module();

    console.log('📦 WASM module loaded\n');

    // Create imports object with HOST functions
    const imports = {
        env: {
            realm_store_model: realmHost.realm_store_model,
            realm_get_tensor: realmHost.realm_get_tensor,
            realm_get_model_info: realmHost.realm_get_model_info,
            realm_remove_model: realmHost.realm_remove_model,
        }
    };

    // Instantiate WASM module with HOST functions
    const instance = await WebAssembly.instantiate(wasmBytes, imports);
    console.log('✅ WASM instance created with HOST functions\n');

    // Access exports
    const { memory, __wbindgen_malloc, __wbindgen_free } = instance.exports;

    console.log('🔧 Creating Realm instance...');

    // For wasm-bindgen modules, we need to use the generated JS bindings
    // Let's load those as well
    const realmWasm = await import('../../wasm-pkg/realm_wasm.js');

    // Initialize with our custom instance
    await realmWasm.default(wasmBytes);

    const realm = new realmWasm.Realm();
    console.log('✅ Realm instance created\n');

    // Load model
    console.log('📥 Loading model into HOST storage...');
    try {
        realm.loadModel(modelBytes);
        console.log('✅ Model loaded successfully!\n');
    } catch (error) {
        console.error('❌ Failed to load model:', error.message);
        console.error('\nNote: The WASM module needs to be connected to real HOST functions.');
        console.error('Current status: WASM compiles, HOST functions exist, but not yet wired together.');
        process.exit(1);
    }

    // Check if loaded
    if (!realm.isLoaded()) {
        console.error('❌ Model not loaded');
        process.exit(1);
    }
    console.log('✅ Model is loaded\n');

    // Set generation config
    const config = new realmWasm.WasmGenerationConfig();
    config.max_tokens = 20;
    config.temperature = 0.7;
    config.top_p = 0.9;
    config.top_k = 40;
    realm.setConfig(config);

    // Generate text
    console.log('🎯 Generating response...');
    const prompt = 'What is the capital of France?';
    console.log(`Prompt: "${prompt}"\n`);

    try {
        const response = realm.generate(prompt);
        console.log('✅ Generation successful!\n');
        console.log(`Response: "${response}"\n`);

        // Check for Paris
        if (response.toLowerCase().includes('paris')) {
            console.log('🎉 SUCCESS: Model correctly identified Paris!');
        } else {
            console.log('⚠️  Response did not mention Paris');
        }
    } catch (error) {
        console.error('❌ Generation failed:', error.message);
        console.error(error.stack);
    }
}

main().catch(error => {
    console.error('Fatal error:', error);
    console.error(error.stack);
    process.exit(1);
});
