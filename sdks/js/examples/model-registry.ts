/**
 * Example: Using RealmRegistry for Multiple Models
 * 
 * Shows how to:
 * - Load multiple models using RealmRegistry
 * - Switch between models
 * - Use defaultModel
 */

import { RealmRegistry } from '@realm-ai/sdk';
import * as fs from 'fs';

async function main() {
  // Initialize registry with default model
  const registry = new RealmRegistry('llama-7b');

  // Initialize WASM (needed for all Realm instances)
  // In production, WASM should be bundled
  const wasmPath = '../wasm/realm_wasm.js';

  // Load multiple models into registry
  console.log('Loading models...');
  
  const model7b = fs.readFileSync('./models/llama-2-7b.gguf');
  await registry.loadModel('llama-7b', model7b, wasmPath);
  console.log('✅ llama-7b loaded');

  const model13b = fs.readFileSync('./models/llama-2-13b.gguf');
  await registry.loadModel('llama-13b', model13b, wasmPath);
  console.log('✅ llama-13b loaded');

  // List models in registry
  const models = registry.getModels();
  console.log('\nModels in registry:');
  models.forEach(model => {
    console.log(`  - ${model.id}: ${model.name} (loaded: ${model.loaded})`);
  });

  // Generate with default model (llama-7b)
  console.log('\n=== Using Default Model (llama-7b) ===');
  const response1 = await registry.generate('What is AI?', {
    maxTokens: 50,
  });
  console.log('Response:', response1.text);
  console.log('Model used:', response1.model);

  // Switch to different model
  console.log('\n=== Using llama-13b ===');
  const response2 = await registry.generate('What is AI?', {
    model: 'llama-13b',  // Specify model
    maxTokens: 50,
  });
  console.log('Response:', response2.text);
  console.log('Model used:', response2.model);

  // Or use default
  registry.setDefaultModel('llama-13b');
  const response3 = await registry.generate('What is AI?', {
    maxTokens: 50,
  });
  console.log('Response:', response3.text);
  console.log('Model used:', response3.model);

  // Check model status
  console.log('\n=== Model Status ===');
  console.log('Default model:', registry.getDefaultModel());
  console.log('llama-7b loaded?', registry.hasModel('llama-7b'));
  console.log('llama-13b loaded?', registry.hasModel('llama-13b'));

  // Cleanup
  registry.dispose();
}

main().catch(console.error);

