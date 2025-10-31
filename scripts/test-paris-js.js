#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

async function main() {
  console.log('Realm WASM - Paris Generation Test\n');
  console.log('====================================\n');

  // Load WASM module
  console.log('1. Loading WASM module...');
  const wasmPath = path.join(__dirname, 'crates/realm-wasm/pkg/realm_wasm_bg.wasm');
  const wasmBytes = fs.readFileSync(wasmPath);
  
  const realmWasm = require('../crates/realm-wasm/pkg/realm_wasm.js');
  await realmWasm.default(wasmBytes);
  console.log('   ✓ WASM initialized\n');

  // Find model
  console.log('2. Loading model...');
  const modelPath = process.env.GGUF || path.join(process.env.HOME || process.env.USERPROFILE, '.ollama/models/tinyllama-1.1b.Q4_K_M.gguf');
  
  if (!fs.existsSync(modelPath)) {
    console.error(`   ✗ Model not found: ${modelPath}`);
    console.error('   Set GGUF environment variable to point to your model file');
    process.exit(1);
  }

  console.log(`   Loading: ${modelPath}`);
  const modelBytes = fs.readFileSync(modelPath);
  console.log(`   Model size: ${(modelBytes.length / 1024 / 1024).toFixed(2)} MB\n`);

  // Create Realm instance
  console.log('3. Creating Realm instance...');
  const realm = new realmWasm.Realm();
  console.log('   ✓ Instance created\n');

  // Load model
  console.log('4. Loading model into Realm...');
  const startLoad = Date.now();
  // Convert Buffer to Uint8Array for WASM
  const modelUint8 = new Uint8Array(modelBytes);
  try {
    realm.load_model(modelUint8);
  } catch (e) {
    console.error(`   ✗ Load failed: ${e.message}`);
    console.error('   Note: The WASM package may need to be rebuilt with wasm-pack');
    console.error('   The current bindings might not match the Rust implementation');
    process.exit(1);
  }
  const loadTime = ((Date.now() - startLoad) / 1000).toFixed(2);
  console.log(`   ✓ Model loaded in ${loadTime}s\n`);

  // Display model info (skip vocabSize check since it might not be in current API)
  console.log(`   Vocabulary size: ${vocabSize}\n`);

  // Generate
  const prompt = `<|system|>
You are a helpful AI assistant.</s>
<|user|>
What is the capital of France?</s>
<|assistant|>
`;

  console.log('5. Generating response...');
  console.log(`   Prompt: "${prompt.trim()}"\n`);
  
  const startGen = Date.now();
  const response = realm.generate(prompt);
  const genTime = ((Date.now() - startGen) / 1000).toFixed(2);

  console.log('6. Result:');
  console.log('   ====================================');
  console.log(`   ${response}`);
  console.log('   ====================================\n');

  // Check if Paris is in the response
  const lowerResponse = response.toLowerCase();
  if (lowerResponse.includes('paris')) {
    console.log('   ✓ SUCCESS: Model correctly identified Paris as the capital of France!');
  } else {
    console.log('   ⚠ Response does not contain "Paris"');
  }

  console.log(`\n   Generation time: ${genTime}s`);
}

main().catch((err) => {
  console.error('Error:', err);
  process.exit(1);
});

