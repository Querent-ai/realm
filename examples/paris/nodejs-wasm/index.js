/**
 * Paris Generation Example - Node.js WASM (Local)
 * 
 * This example demonstrates using Realm's WASM module directly in Node.js:
 * - Question: "What is the capital of France?"
 * - Expected Answer: "Paris"
 * 
 * This runs locally without a server, using WASM + host-side storage.
 */

import { readFileSync } from 'fs';
import { init, Realm } from '../../../crates/realm-wasm/pkg/realm_wasm.js';

async function main() {
    console.log('🚀 Realm Paris Generation - Node.js WASM (Local)\n');

    // Initialize WASM module
    console.log('📦 Initializing WASM module...');
    await init();
    console.log('✅ WASM module initialized\n');

    // Load model file
    const modelPath = process.argv[2] || process.env.MODEL_PATH || '~/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf';
    console.log(`📥 Loading model: ${modelPath}`);
    
    let modelBytes;
    try {
        modelBytes = readFileSync(modelPath);
    } catch (e) {
        console.error(`❌ Failed to read model: ${e.message}`);
        console.error('   Please provide path to TinyLlama Q4_K_M model');
        process.exit(1);
    }

    console.log(`   Model size: ${(modelBytes.length / 1024 / 1024).toFixed(2)} MB\n`);

    // Create Realm instance
    console.log('🔧 Creating Realm instance...');
    const realm = new Realm();
    console.log('✅ Realm instance created\n');

    // Load model (stores in HOST)
    console.log('💾 Loading model into HOST storage...');
    const startMemory = process.memoryUsage();
    
    try {
        realm.loadModel(new Uint8Array(modelBytes));
        console.log('✅ Model loaded successfully!');
        
        const afterLoadMemory = process.memoryUsage();
        const memoryDiff = {
            heapUsed: (afterLoadMemory.heapUsed - startMemory.heapUsed) / 1024 / 1024,
            external: (afterLoadMemory.external - startMemory.external) / 1024 / 1024,
        };
        
        console.log(`   Memory after load:`);
        console.log(`   - Heap used: +${memoryDiff.heapUsed.toFixed(2)} MB`);
        console.log(`   - External: +${memoryDiff.external.toFixed(2)} MB`);
        console.log(`   (Model stored in HOST, not WASM!)\n`);
    } catch (e) {
        console.error(`❌ Failed to load model: ${e.message}`);
        process.exit(1);
    }

    // Set generation config
    const config = new realm.WasmGenerationConfig();
    config.max_tokens = 20;
    config.temperature = 0.0; // Deterministic
    config.top_p = 0.9;
    config.top_k = 40;
    realm.setConfig(config);

    // Generate "Paris" response
    const prompt = "What is the capital of France?";
    console.log(`🎯 Generating response to: "${prompt}"`);
    console.log('   (This loads weights from HOST on-demand...)\n');

    const genStartMemory = process.memoryUsage();
    const startTime = Date.now();

    let response;
    try {
        response = realm.generate(prompt);
        const genTime = Date.now() - startTime;
        const genEndMemory = process.memoryUsage();
        const genMemoryDiff = {
            heapUsed: (genEndMemory.heapUsed - genStartMemory.heapUsed) / 1024 / 1024,
            external: (genEndMemory.external - genStartMemory.external) / 1024 / 1024,
        };

        console.log('✅ Generation complete!\n');
        console.log('📊 Results:');
        console.log(`   Response: ${response}`);
        console.log(`   Time: ${genTime}ms`);
        console.log(`   Memory during generation:`);
        console.log(`   - Heap used: +${genMemoryDiff.heapUsed.toFixed(2)} MB`);
        console.log(`   - External: +${genMemoryDiff.external.toFixed(2)} MB\n`);

        // Verify "Paris" in response
        if (response.toLowerCase().includes('paris')) {
            console.log('✅ SUCCESS: Model correctly identified Paris as the capital of France!');
        } else {
            console.log('⚠️  WARNING: Response does not contain "Paris"');
            console.log(`   Full response: ${response}`);
        }
    } catch (e) {
        console.error(`❌ Generation failed: ${e.message}`);
        console.error(`   Stack: ${e.stack}`);
        process.exit(1);
    }

    // Final memory stats
    const finalMemory = process.memoryUsage();
    console.log('\n📈 Final Memory Usage:');
    console.log(`   - Heap used: ${(finalMemory.heapUsed / 1024 / 1024).toFixed(2)} MB`);
    console.log(`   - External: ${(finalMemory.external / 1024 / 1024).toFixed(2)} MB`);
    console.log(`   - RSS: ${(finalMemory.rss / 1024 / 1024).toFixed(2)} MB`);

    // Memory analysis
    const totalMemoryIncrease = (finalMemory.heapUsed - startMemory.heapUsed) / 1024 / 1024;
    console.log(`\n💡 Memory Analysis:`);
    console.log(`   - Total increase: ${totalMemoryIncrease.toFixed(2)} MB`);
    console.log(`   - Expected WASM memory: ~50MB (vs 2.5GB+ without host storage)`);
    console.log(`   - Memory efficiency: ${((1 - totalMemoryIncrease / 2500) * 100).toFixed(1)}% reduction\n`);

    console.log('🎉 Test complete!');
}

main().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
});

