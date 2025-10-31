#!/usr/bin/env node
/**
 * Paris Generation Test - WASM Version
 *
 * This should produce the same "Paris" response as the native Rust version.
 */

const fs = require('fs');
const path = require('path');
const { Realm, WasmGenerationConfig } = require('./pkg/realm_wasm.js');

async function main() {
    console.log('=== Realm WASM - Paris Generation Test ===\n');

    // Create Realm instance
    console.log('Creating Realm instance...');
    const realm = new Realm();
    console.log('✓ Realm created\n');

    // Configure generation (matching native example)
    const config = new WasmGenerationConfig();
    config.max_tokens = 50;
    config.temperature = 0.0;  // Greedy/deterministic for consistent results
    config.top_p = 0.9;
    config.top_k = 40;
    config.repetition_penalty = 1.1;
    realm.setConfig(config);
    console.log('✓ Config set (greedy decoding for deterministic output)\n');

    // Load model
    const modelPath = '/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf';
    console.log(`Loading model: ${modelPath}`);
    console.log('This may take a few seconds...');

    const modelBytes = fs.readFileSync(modelPath);
    console.log(`Model size: ${(modelBytes.length / 1024 / 1024).toFixed(1)} MB`);

    const loadStart = Date.now();
    realm.loadModel(new Uint8Array(modelBytes));
    const loadTime = Date.now() - loadStart;

    console.log(`✓ Model loaded in ${loadTime}ms`);
    console.log(`✓ Vocab size: ${realm.vocabSize()}\n`);

    // Generate text
    const prompt = 'What is the capital of France?';
    console.log(`Prompt: "${prompt}"`);
    console.log('Generating...\n');

    const genStart = Date.now();
    const response = realm.generate(prompt);
    const genTime = Date.now() - genStart;

    console.log('=== GENERATED TEXT ===');
    console.log(response);
    console.log('======================\n');

    console.log(`Generation time: ${genTime}ms`);
    console.log(`Tokens generated: ~${config.max_tokens}`);
    console.log(`Speed: ~${(config.max_tokens / (genTime / 1000)).toFixed(1)} tokens/sec`);

    // Check if response contains "Paris"
    if (response.toLowerCase().includes('paris')) {
        console.log('\n✅ SUCCESS: Response contains "Paris"!');
    } else {
        console.log('\n⚠️  Note: Response does not contain "Paris"');
    }
}

main().catch(err => {
    console.error('Error:', err);
    process.exit(1);
});
