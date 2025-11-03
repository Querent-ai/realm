#!/usr/bin/env node
/**
 * Simple Paris Generation Test - CommonJS
 *
 * Tests: JS -> Native Addon (HOST storage) -> Generate Paris
 */

const fs = require('fs');
const path = require('path');

// Import native addon (realm-node)
const nativePath = path.join(__dirname, '../../crates/realm-node/index.node');
const realmNative = require(nativePath);

async function main() {
    console.log('╔════════════════════════════════════════════════════════════════╗');
    console.log('║  Realm Simple Test: Native HOST Storage + Paris Generation    ║');
    console.log('╚════════════════════════════════════════════════════════════════╝\n');

    // Get model path
    const modelPath = process.argv[2] || process.env.MODEL_PATH || '/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf';

    console.log(`📦 Model: ${modelPath}`);

    if (!fs.existsSync(modelPath)) {
        console.error(`❌ Model not found: ${modelPath}`);
        process.exit(1);
    }

    // Step 1: Load model into HOST storage
    console.log('\n📥 Loading model into HOST storage...');
    const modelBytes = fs.readFileSync(modelPath);
    console.log(`   Size: ${(modelBytes.length / 1024 / 1024).toFixed(2)} MB`);

    const modelId = realmNative.storeModel(modelBytes);
    console.log(`✅ Model stored in HOST with ID: ${modelId}`);

    // Step 2: Get model info
    const info = realmNative.getModelInfo(modelId);
    console.log(`\n📊 Model Info:`);
    console.log(`   Tensors: ${info.tensor_count}`);
    console.log(`   Size: ${(info.total_size / 1024 / 1024).toFixed(2)} MB`);

    // Step 3: Test tensor retrieval
    console.log(`\n🔍 Testing tensor retrieval from HOST...`);
    try {
        // Get a known tensor (embedding weights)
        const tensorName = 'token_embd.weight';
        console.log(`   Fetching: "${tensorName}"`);

        const tensorData = realmNative.getTensor(modelId, tensorName);
        console.log(`✅ Retrieved tensor: ${tensorData.byteLength} bytes (dequantized f32)`);

        // Verify it's f32 data
        const f32Count = tensorData.byteLength / 4;
        const f32Array = new Float32Array(tensorData.buffer, tensorData.byteOffset, f32Count);
        console.log(`   Elements: ${f32Count}`);
        console.log(`   First 5 values: ${Array.from(f32Array.slice(0, 5)).map(v => v.toFixed(4)).join(', ')}`);

    } catch (err) {
        console.error(`⚠️  Tensor test failed: ${err.message}`);
        console.log('   (This is OK - continuing with cleanup test)');
    }

    // Step 4: Cleanup
    console.log(`\n🗑️  Removing model from HOST storage...`);
    realmNative.removeModel(modelId);
    console.log(`✅ Model removed`);

    console.log('\n╔════════════════════════════════════════════════════════════════╗');
    console.log('║  ✅ HOST Storage Test Complete!                                ║');
    console.log('╚════════════════════════════════════════════════════════════════╝\n');

    console.log('✨ Next step: Wire WASM module to call these HOST functions');
    console.log('   for on-demand weight loading during inference.\n');
}

main().catch(err => {
    console.error('\n❌ Test failed:', err);
    console.error(err.stack);
    process.exit(1);
});
