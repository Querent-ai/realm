#!/usr/bin/env node
/**
 * Quick test to verify WASM bindings are working correctly
 */

const { Realm, WasmGenerationConfig } = require('./pkg/realm_wasm.js');

console.log('✓ WASM module loaded');

// Test 1: Create Realm instance
const realm = new Realm();
console.log('✓ Realm instance created');

// Test 2: Check isLoaded before loading model
const loadedBefore = realm.isLoaded();
console.log(`✓ isLoaded() before loading: ${loadedBefore}`);
if (loadedBefore !== false) {
    throw new Error('Expected isLoaded() to be false before loading model');
}

// Test 3: Create and set config
const config = new WasmGenerationConfig();
config.max_tokens = 50;
config.temperature = 0.8;
realm.setConfig(config);
console.log('✓ Config created and set');

// Test 4: Try to generate without model (should fail gracefully)
try {
    realm.generate('test');
    throw new Error('Expected generate() to fail without model');
} catch (e) {
    if (e.message.includes('Model not loaded')) {
        console.log('✓ generate() correctly fails without model');
    } else {
        throw e;
    }
}

console.log('\n✅ All binding tests passed!');
console.log('\nAPI verified:');
console.log('  - Realm constructor');
console.log('  - isLoaded()');
console.log('  - setConfig()');
console.log('  - generate() (error handling)');
console.log('  - WasmGenerationConfig');
