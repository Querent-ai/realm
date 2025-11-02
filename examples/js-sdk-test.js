#!/usr/bin/env node
/**
 * Simple working JS test using WASM module
 */

const fs = require('fs');
const path = require('path');

// Load WASM module using wasm-bindgen generated bindings
const wasm = require('../wasm-pkg/realm_wasm.js');

async function main() {
    console.log('🚀 Realm JS SDK Test\n');

    // Initialize WASM
    await wasm.default();
    console.log('✅ WASM initialized\n');

    // Create Realm instance
    const realm = new wasm.Realm();
    console.log('✅ Realm instance created\n');

    // Load model
    const modelPath = process.argv[2] || '/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf';
    console.log(`📥 Loading model: ${modelPath}`);

    const modelBytes = fs.readFileSync(modelPath);
    console.log(`✅ Read ${(modelBytes.length / 1024 / 1024).toFixed(2)} MB\n`);

    try {
        realm.loadModel(modelBytes);
        console.log('✅ Model loaded!\n');
    } catch (error) {
        console.error('❌ Load failed:', error.message);
        console.log('\nℹ️  This is expected - WASM needs HOST functions wired up');
        console.log('   Next step: Wire up realm_store_model, realm_get_tensor, etc.\n');
        process.exit(1);
    }

    // Generate
    console.log('🎯 Generating...\n');
    const response = realm.generate('What is the capital of France?');
    console.log(`✅ Response: "${response}"\n`);

    if (response.toLowerCase().includes('paris')) {
        console.log('🎉 SUCCESS!');
    }
}

main().catch(console.error);
