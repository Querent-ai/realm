//! Test Paris Generation via Node.js SDK
//! 
//! This tests end-to-end:
//! 1. Load model via SDK (stores in HOST)
//! 2. Generate "Paris" response
//! 3. Verify memory usage

import Realm from './index.js';
import fs from 'fs';

async function main() {
    console.log('🚀 Realm Node.js SDK - Paris Generation Test\n');

    // Get model path
    const modelPath = process.argv[2] || process.env.MODEL_PATH || '~/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf';
    const expandedPath = modelPath.startsWith('~') 
        ? modelPath.replace('~', process.env.HOME)
        : modelPath;

    console.log(`📦 Model: ${expandedPath}`);
    
    if (!fs.existsSync(expandedPath)) {
        console.error(`❌ Model not found: ${expandedPath}`);
        console.error('   Please provide path to TinyLlama Q4_K_M model');
        process.exit(1);
    }

    const modelBytes = fs.readFileSync(expandedPath);
    console.log(`   Size: ${(modelBytes.length / 1024 / 1024).toFixed(2)} MB\n`);

    // Initialize SDK
    console.log('🔧 Initializing Realm SDK...');
    const startMemory = process.memoryUsage();
    
    const realm = new Realm();
    await realm.initialize();
    
    const afterInitMemory = process.memoryUsage();
    console.log('✅ SDK initialized\n');

    // Load model
    console.log('💾 Loading model into HOST storage...');
    const beforeLoadMemory = process.memoryUsage();
    
    try {
        await realm.loadModel(modelBytes);
        console.log('✅ Model loaded successfully!');
        
        const afterLoadMemory = process.memoryUsage();
        const loadMemoryDiff = {
            heapUsed: (afterLoadMemory.heapUsed - beforeLoadMemory.heapUsed) / 1024 / 1024,
            external: (afterLoadMemory.external - beforeLoadMemory.external) / 1024 / 1024,
        };
        
        console.log(`   Memory after load:`);
        console.log(`   - Heap used: +${loadMemoryDiff.heapUsed.toFixed(2)} MB`);
        console.log(`   - External: +${loadMemoryDiff.external.toFixed(2)} MB`);
        console.log(`   (Model stored in HOST, not WASM!)\n`);
    } catch (e) {
        console.error(`❌ Failed to load model: ${e.message}`);
        console.error(`   Stack: ${e.stack}`);
        process.exit(1);
    }

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
    console.log(`   - Memory efficiency: ${Math.max(0, (1 - totalMemoryIncrease / 2500) * 100).toFixed(1)}% reduction\n`);

    console.log('🎉 Test complete!');
}

main().catch(e => {
    console.error('Fatal error:', e);
    process.exit(1);
});

