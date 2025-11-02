//! Realm Node.js SDK
//! 
//! Complete SDK for running LLM inference via WASM with host-side storage

import * as wasm from './pkg/realm_wasm.js';
import * as native from '@realm/realm-node';
import { readFileSync } from 'fs';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);

// WASM memory reference (set after init)
let wasmMemory = null;

/**
 * Initialize WASM module with host functions
 */
async function init() {
    // CRITICAL: Patch Module.require to inject host functions BEFORE wasm.default() runs
    // wasm-bindgen generated code does: imports['env'] = require('env');
    // We intercept this require() call to provide our host functions
    const Module = require('module');
    const originalRequire = Module.prototype.require;
    
    // Track if we've already patched
    let hostFunctionsProvided = false;
    
    Module.prototype.require = function(id) {
        if (id === 'env' && !hostFunctionsProvided) {
            hostFunctionsProvided = true;
            // Return our host functions object
            // Note: memory will be set after wasm.default() runs
            // We'll create a lazy getter that uses wasm.memory
            return {
                realm_store_model: (ggufPtr, ggufLen, modelId) => {
                    const memory = wasmMemory || wasm.memory || (typeof wasm.__wbindgen_memory === 'function' ? wasm.__wbindgen_memory() : null);
                    if (!memory) {
                        console.error('WASM memory not available for realm_store_model');
                        return -1;
                    }
                    try {
                        const ggufBytes = new Uint8Array(memory.buffer, ggufPtr, ggufLen);
                        const buffer = Buffer.from(ggufBytes);
                        const requestedId = modelId === 0 ? null : modelId;
                        return native.storeModel(buffer, requestedId);
                    } catch (e) {
                        console.error('realm_store_model error:', e);
                        return -1;
                    }
                },
                realm_get_tensor: (modelId, tensorNamePtr, tensorNameLen, outPtr, outMaxLen) => {
                    const memory = wasmMemory || wasm.memory || (typeof wasm.__wbindgen_memory === 'function' ? wasm.__wbindgen_memory() : null);
                    if (!memory) {
                        console.error('WASM memory not available for realm_get_tensor');
                        return -1;
                    }
                    try {
                        const nameBytes = new Uint8Array(memory.buffer, tensorNamePtr, tensorNameLen);
                        const tensorName = new TextDecoder().decode(nameBytes);
                        const tensorBytes = native.getTensor(modelId, tensorName);
                        if (tensorBytes.byteLength > outMaxLen) {
                            console.error(`Buffer too small: need ${tensorBytes.byteLength}, got ${outMaxLen}`);
                            return -1;
                        }
                        const outBuffer = new Uint8Array(memory.buffer, outPtr, tensorBytes.byteLength);
                        outBuffer.set(new Uint8Array(tensorBytes));
                        return tensorBytes.byteLength;
                    } catch (e) {
                        console.error('realm_get_tensor error:', e);
                        return -1;
                    }
                },
                realm_get_model_info: (modelId, outTensorCountPtr, outTotalSizePtr) => {
                    const memory = wasmMemory || wasm.memory || (typeof wasm.__wbindgen_memory === 'function' ? wasm.__wbindgen_memory() : null);
                    if (!memory) {
                        console.error('WASM memory not available for realm_get_model_info');
                        return -1;
                    }
                    try {
                        const info = native.getModelInfo(modelId);
                        const countView = new DataView(memory.buffer, outTensorCountPtr, 4);
                        countView.setUint32(0, info.tensor_count, true);
                        const sizeView = new DataView(memory.buffer, outTotalSizePtr, 8);
                        sizeView.setBigUint64(0, BigInt(info.total_size), true);
                        return 0;
                    } catch (e) {
                        console.error('realm_get_model_info error:', e);
                        return -1;
                    }
                },
                realm_remove_model: (modelId) => {
                    try {
                        native.removeModel(modelId);
                        return 0;
                    } catch (e) {
                        console.error('realm_remove_model error:', e);
                        return -1;
                    }
                },
                realm_forward_layer: (modelId, layerIdx, hiddenStatesPtr, hiddenStatesLen, position, outPtr) => {
                    const memory = wasmMemory || wasm.memory || (typeof wasm.__wbindgen_memory === 'function' ? wasm.__wbindgen_memory() : null);
                    if (!memory) {
                        console.error('WASM memory not available for realm_forward_layer');
                        return -1;
                    }
                    try {
                        // Read hidden states from WASM memory
                        const hiddenStatesBytes = new Uint8Array(memory.buffer, hiddenStatesPtr, hiddenStatesLen * 4);
                        const hiddenStatesBuffer = Buffer.from(hiddenStatesBytes);

                        // Call native addon
                        const output = native.forwardLayer(modelId, layerIdx, hiddenStatesBuffer, position);

                        // Write to WASM memory
                        if (output.byteLength > memory.buffer.byteLength - outPtr) {
                            console.error('Output buffer too small for forward_layer');
                            return -1;
                        }

                        const outBuffer = new Uint8Array(memory.buffer, outPtr, output.byteLength);
                        outBuffer.set(new Uint8Array(output));

                        return output.byteLength;
                    } catch (e) {
                        console.error('realm_forward_layer error:', e);
                        return -1;
                    }
                },
                realm_embed_tokens: (modelId, tokenIdsPtr, tokenCount, outPtr) => {
                    const memory = wasmMemory || wasm.memory || (typeof wasm.__wbindgen_memory === 'function' ? wasm.__wbindgen_memory() : null);
                    if (!memory) {
                        console.error('WASM memory not available for realm_embed_tokens');
                        return -1;
                    }
                    try {
                        // Read token IDs from WASM memory
                        const tokenIdsBytes = new Uint8Array(memory.buffer, tokenIdsPtr, tokenCount * 4);
                        const tokenIdsBuffer = Buffer.from(tokenIdsBytes);

                        // Call native addon
                        const hiddenStates = native.embedTokens(modelId, tokenIdsBuffer);

                        // Write to WASM memory
                        if (hiddenStates.byteLength > memory.buffer.byteLength - outPtr) {
                            console.error('Output buffer too small for embed_tokens');
                            return -1;
                        }

                        const outBuffer = new Uint8Array(memory.buffer, outPtr, hiddenStates.byteLength);
                        outBuffer.set(new Uint8Array(hiddenStates));

                        return hiddenStates.byteLength;
                    } catch (e) {
                        console.error('realm_embed_tokens error:', e);
                        return -1;
                    }
                },
                realm_compute_logits: (modelId, hiddenStatePtr, hiddenSize, outPtr) => {
                    const memory = wasmMemory || wasm.memory || (typeof wasm.__wbindgen_memory === 'function' ? wasm.__wbindgen_memory() : null);
                    if (!memory) {
                        console.error('WASM memory not available for realm_compute_logits');
                        return -1;
                    }
                    try {
                        // Read hidden state from WASM memory
                        const hiddenStateBytes = new Uint8Array(memory.buffer, hiddenStatePtr, hiddenSize * 4);
                        const hiddenStateBuffer = Buffer.from(hiddenStateBytes);

                        // Call native addon
                        const logits = native.computeLogits(modelId, hiddenStateBuffer);

                        // Write to WASM memory
                        if (logits.byteLength > memory.buffer.byteLength - outPtr) {
                            console.error('Output buffer too small for compute_logits');
                            return -1;
                        }

                        const outBuffer = new Uint8Array(memory.buffer, outPtr, logits.byteLength);
                        outBuffer.set(new Uint8Array(logits));

                        return logits.byteLength;
                    } catch (e) {
                        console.error('realm_compute_logits error:', e);
                        return -1;
                    }
                },
            };
        }
        return originalRequire.apply(this, arguments);
    };
    
    try {
        // Initialize wasm-bindgen (this will call require('env') which we intercept above)
        await wasm.default();
        
        // Restore original require
        Module.prototype.require = originalRequire;
        
        // Get WASM memory from exports
        wasmMemory = wasm.memory 
            || (typeof wasm.__wbindgen_memory === 'function' ? wasm.__wbindgen_memory() : null)
            || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory : null);
        
        if (!wasmMemory) {
            console.warn('Could not access WASM memory directly. Host functions may not work correctly.');
        }
    } catch (e) {
        // Restore original require on error
        Module.prototype.require = originalRequire;
        throw e;
    }
    
    return true;
}

/**
 * Create host function imports for WASM
 */
function createHostFunctions() {
    // Get memory - try multiple approaches
    const memory = wasmMemory 
        || wasm.memory
        || (typeof wasm.__wbindgen_memory === 'function' ? wasm.__wbindgen_memory() : null)
        || null;
    
    if (!memory) {
        throw new Error('WASM not initialized or memory not accessible. Call init() first.');
    }
    
    return {
        env: {
            // realm_store_model(gguf_ptr, gguf_len, model_id) -> model_id
            realm_store_model: (ggufPtr, ggufLen, modelId) => {
                try {
                    // Read GGUF bytes from WASM memory
                    const ggufBytes = new Uint8Array(memory.buffer, ggufPtr, ggufLen);
                    
                    // Convert to Buffer
                    const buffer = Buffer.from(ggufBytes);
                    
                    // Call native storage
                    const requestedId = modelId === 0 ? null : modelId;
                    return native.storeModel(buffer, requestedId);
                } catch (e) {
                    console.error('realm_store_model error:', e);
                    return -1;
                }
            },

            // realm_get_tensor(model_id, tensor_name_ptr, tensor_name_len, out_ptr, out_max_len) -> bytes_written
            realm_get_tensor: (modelId, tensorNamePtr, tensorNameLen, outPtr, outMaxLen) => {
                try {
                    // Read tensor name from WASM memory
                    const nameBytes = new Uint8Array(memory.buffer, tensorNamePtr, tensorNameLen);
                    const tensorName = new TextDecoder().decode(nameBytes);
                    
                    // Get tensor from native storage (already dequantized)
                    const tensorBytes = native.getTensor(modelId, tensorName);
                    
                    if (tensorBytes.byteLength > outMaxLen) {
                        console.error(`Buffer too small: need ${tensorBytes.byteLength}, got ${outMaxLen}`);
                        return -1;
                    }
                    
                    // Write to WASM memory
                    const outBuffer = new Uint8Array(memory.buffer, outPtr, tensorBytes.byteLength);
                    outBuffer.set(new Uint8Array(tensorBytes));
                    
                    return tensorBytes.byteLength;
                } catch (e) {
                    console.error('realm_get_tensor error:', e);
                    return -1;
                }
            },

            // realm_get_model_info(model_id, out_tensor_count_ptr, out_total_size_ptr) -> status
            realm_get_model_info: (modelId, outTensorCountPtr, outTotalSizePtr) => {
                try {
                    const info = native.getModelInfo(modelId);
                    
                    // Write u32 tensor_count (4 bytes, little-endian)
                    const countView = new DataView(memory.buffer, outTensorCountPtr, 4);
                    countView.setUint32(0, info.tensor_count, true);
                    
                    // Write u64 total_size (8 bytes, little-endian)
                    const sizeView = new DataView(memory.buffer, outTotalSizePtr, 8);
                    sizeView.setBigUint64(0, BigInt(info.total_size), true);
                    
                    return 0;
                } catch (e) {
                    console.error('realm_get_model_info error:', e);
                    return -1;
                }
            },

            // realm_remove_model(model_id) -> status
            realm_remove_model: (modelId) => {
                try {
                    native.removeModel(modelId);
                    return 0;
                } catch (e) {
                    console.error('realm_remove_model error:', e);
                    return -1;
                }
            },
        },
    };
}

/**
 * Realm SDK Class
 */
export class Realm {
    constructor() {
        this.realm = null;
        this.initialized = false;
    }

    /**
     * Initialize SDK
     */
    async initialize() {
        if (this.initialized) {
            return;
        }

        await init();
        this.realm = new wasm.Realm();
        this.initialized = true;
    }

    /**
     * Load model from GGUF file
     * @param {string|Buffer|Uint8Array} modelPathOrBytes - Path to GGUF file or bytes
     * @param {number|null} modelId - Optional model ID
     */
    async loadModel(modelPathOrBytes, modelId = null) {
        if (!this.initialized) {
            await this.initialize();
        }

        let modelBytes;
        if (typeof modelPathOrBytes === 'string') {
            modelBytes = readFileSync(modelPathOrBytes);
        } else {
            modelBytes = modelPathOrBytes;
        }

        this.realm.loadModel(new Uint8Array(modelBytes), modelId);
    }

    /**
     * Generate text from prompt
     * @param {string} prompt - Input prompt
     * @returns {string} Generated text
     */
    generate(prompt) {
        if (!this.initialized || !this.realm) {
            throw new Error('SDK not initialized. Call loadModel() first.');
        }

        return this.realm.generate(prompt);
    }

    /**
     * Check if model is loaded
     */
    isLoaded() {
        return this.realm?.isLoaded() || false;
    }

    /**
     * Set generation config
     */
    setConfig(config) {
        if (!this.realm) {
            throw new Error('SDK not initialized.');
        }
        this.realm.setConfig(config);
    }

    /**
     * Get vocabulary size
     */
    vocabSize() {
        if (!this.realm) {
            throw new Error('SDK not initialized.');
        }
        return this.realm.vocabSize();
    }
}

export default Realm;

