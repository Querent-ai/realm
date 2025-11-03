/**
 * Basic example - Single model usage
 */

import { Realm } from '@realm-ai/sdk';
import * as fs from 'fs';

async function main() {
  // Initialize Realm (single model instance)
  const realm = new Realm();

  // Initialize WASM module
  // In production, WASM should be bundled or provided via wasmPath
  await realm.init(undefined, '../wasm/realm_wasm.js');

  // Load model
  console.log('Loading model...');
  const modelBytes = fs.readFileSync('./models/llama-2-7b.gguf');
  await realm.loadModel(modelBytes, 'llama-7b');
  console.log('✅ Model loaded');

  // Set generation config
  realm.setConfig({
    maxTokens: 50,
    temperature: 0.7,
  });

  // Generate
  const response = await realm.generate('What is the capital of France?');
  console.log('Response:', response.text);
  console.log('Model:', response.model);

  // Cleanup
  realm.dispose();
}

main().catch(console.error);

