#!/usr/bin/env node
/**
 * Test HOST-side computation: Direct native calls (bypassing WASM)
 *
 * This tests the native addon functions directly without WASM,
 * to verify the HOST-side computation logic works correctly.
 */

const fs = require('fs');

async function main() {
    console.log('╔════════════════════════════════════════════════════════════════╗');
    console.log('║  Realm Native Direct Test: HOST Computation Functions         ║');
    console.log('╚════════════════════════════════════════════════════════════════╝\n');

    const modelPath = process.argv[2] || '/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf';
    console.log(`📦 Model: ${modelPath}\n`);

    if (!fs.existsSync(modelPath)) {
        console.error(`❌ Model not found: ${modelPath}`);
        process.exit(1);
    }

    // Load native addon
    console.log('📥 Loading native addon...');
    const native = require('../../crates/realm-node/index.node');
    console.log('✅ Loaded\n');

    // Load model into HOST storage
    console.log('📥 Loading model into HOST storage...');
    const modelBytes = fs.readFileSync(modelPath);
    const modelId = native.storeModel(modelBytes);
    const info = native.getModelInfo(modelId);

    console.log(`✅ Model in HOST:`);
    console.log(`   ID: ${modelId}`);
    console.log(`   Tensors: ${info.tensor_count}`);
    console.log(`   Size: ${(info.total_size / 1024 / 1024).toFixed(2)} MB\n`);

    // Test 1: Embed tokens
    console.log('🧪 Test 1: Embed tokens (8 tokens)...');
    const tokenIds = new Uint32Array([1, 1724, 338, 278, 7483, 310, 3444, 29973]); // "What is the capital of France?"
    const tokenIdsBuffer = Buffer.from(tokenIds.buffer);

    try {
        const hiddenStates = native.embedTokens(modelId, tokenIdsBuffer);
        console.log(`   ✅ Embedded ${tokenIds.length} tokens → ${hiddenStates.byteLength / 4} f32 values`);
        console.log(`   ✅ Hidden states size: ${(hiddenStates.byteLength / 1024).toFixed(2)} KB\n`);

        // Test 2: Forward layer (simplified - just norms, no attention)
        console.log('🧪 Test 2: Forward layer 0...');
        try {
            // Convert ArrayBuffer to Buffer
            const hiddenStatesBuffer = Buffer.from(hiddenStates);
            const layerOutput = native.forwardLayer(modelId, 0, hiddenStatesBuffer, 0);
            console.log(`   ✅ Layer 0 forward complete: ${layerOutput.byteLength / 4} f32 values`);
            console.log(`   ✅ Output size: ${(layerOutput.byteLength / 1024).toFixed(2)} KB\n`);

            // Test 3: Compute logits (final token only)
            console.log('🧪 Test 3: Compute logits (final token)...');
            const hiddenSize = 2048; // TinyLlama hidden size
            const finalTokenHidden = Buffer.from(layerOutput).slice(layerOutput.byteLength - hiddenSize * 4);

            try {
                const logits = native.computeLogits(modelId, finalTokenHidden);
                console.log(`   ✅ Computed logits: ${logits.byteLength / 4} values (vocab_size)`);
                console.log(`   ✅ Logits size: ${(logits.byteLength / 1024).toFixed(2)} KB\n`);

                // Find top token
                const logitsBuffer = Buffer.from(logits);
                const logitsArray = new Float32Array(logitsBuffer.buffer, logitsBuffer.byteOffset, logitsBuffer.byteLength / 4);
                let maxIdx = 0;
                let maxVal = logitsArray[0] || 0;
                for (let i = 1; i < logitsArray.length; i++) {
                    if (logitsArray[i] > maxVal) {
                        maxVal = logitsArray[i];
                        maxIdx = i;
                    }
                }

                console.log('╔════════════════════════════════════════════════════════════════╗');
                console.log('║  RESULT                                                        ║');
                console.log('╚════════════════════════════════════════════════════════════════╝');
                console.log(`\n✅ Top token ID: ${maxIdx} (logit: ${maxVal.toFixed ? maxVal.toFixed(4) : maxVal})\n`);
                console.log('✨ HOST-side computation working:');
                console.log('   ✅ embed_tokens: Native implementation');
                console.log('   ✅ forward_layer: Native implementation (simplified)');
                console.log('   ✅ compute_logits: Native implementation');
                console.log('   ✅ All operations on HOST, no WASM memory issues\n');
            } catch (err) {
                console.error('❌ compute_logits failed:', err.message);
            }
        } catch (err) {
            console.error('❌ forward_layer failed:', err.message);
        }
    } catch (err) {
        console.error('❌ embed_tokens failed:', err.message);
    }

    // Cleanup
    console.log('🗑️  Cleanup...');
    native.removeModel(modelId);
    console.log('✅ Done\n');
}

main().catch(err => {
    console.error('\n❌ Fatal:', err.message);
    console.error(err.stack);
    process.exit(1);
});
