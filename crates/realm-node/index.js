//! Realm Node.js SDK
//! 
//! Provides JavaScript interface to native Rust host-side storage

const native = require('./index.node');

/**
 * Store model in HOST storage
 * @param {Buffer|Uint8Array} ggufBytes - GGUF file bytes
 * @param {number|null|undefined} modelId - Optional model ID (null/undefined = auto-generate from hash)
 * @returns {number} Model ID
 */
function storeModel(ggufBytes, modelId = null) {
    // Convert Uint8Array to Buffer if needed
    const buffer = Buffer.isBuffer(ggufBytes) 
        ? ggufBytes 
        : Buffer.from(ggufBytes);
    
    if (modelId === null || modelId === undefined || modelId === 0) {
        return native.storeModel(buffer);
    }
    return native.storeModel(buffer, modelId);
}

/**
 * Get tensor from HOST storage (dequantized to f32)
 * @param {number} modelId - Model ID
 * @param {string} tensorName - Tensor name (e.g., "blk.0.attn_q.weight")
 * @returns {ArrayBuffer} Dequantized f32 data as bytes
 */
function getTensor(modelId, tensorName) {
    return native.getTensor(modelId, tensorName);
}

/**
 * Get model metadata
 * @param {number} modelId - Model ID
 * @returns {{tensor_count: number, total_size: number}}
 */
function getModelInfo(modelId) {
    return native.getModelInfo(modelId);
}

/**
 * Remove model from storage
 * @param {number} modelId - Model ID
 */
function removeModel(modelId) {
    native.removeModel(modelId);
}

/**
 * Embed tokens using HOST-side computation (bypasses WASM)
 * @param {number} modelId - Model ID
 * @param {Uint32Array|Buffer} tokenIds - Token IDs (as Uint32Array or Buffer of u32 values)
 * @returns {ArrayBuffer} Hidden states (f32 array: token_count * hidden_size)
 */
function embedTokens(modelId, tokenIds) {
    // Convert to Buffer if needed
    let buffer;
    if (Buffer.isBuffer(tokenIds)) {
        buffer = tokenIds;
    } else if (tokenIds instanceof Uint32Array) {
        // Convert Uint32Array to Buffer (little-endian u32)
        buffer = Buffer.from(tokenIds.buffer, tokenIds.byteOffset, tokenIds.length * 4);
    } else {
        throw new Error('tokenIds must be Uint32Array or Buffer');
    }
    
    return native.embedTokens(modelId, buffer);
}

/**
 * Forward through one transformer layer using HOST-side computation (bypasses WASM)
 * @param {number} modelId - Model ID
 * @param {number} layerIdx - Layer index (0..num_layers-1)
 * @param {Float32Array|Buffer} hiddenStates - Hidden states (f32 array)
 * @param {number} position - Position in sequence (for KV cache)
 * @returns {ArrayBuffer} Output hidden states (same size as input)
 */
function forwardLayer(modelId, layerIdx, hiddenStates, position) {
    // Convert to Buffer if needed
    let buffer;
    if (Buffer.isBuffer(hiddenStates)) {
        buffer = hiddenStates;
    } else if (hiddenStates instanceof Float32Array) {
        // Convert Float32Array to Buffer (little-endian f32)
        buffer = Buffer.from(hiddenStates.buffer, hiddenStates.byteOffset, hiddenStates.length * 4);
    } else {
        throw new Error('hiddenStates must be Float32Array or Buffer');
    }
    
    return native.forwardLayer(modelId, layerIdx, buffer, position);
}

/**
 * Compute logits from hidden states using HOST-side computation (bypasses WASM)
 * @param {number} modelId - Model ID
 * @param {Float32Array|Buffer} hiddenState - Hidden state (f32 array, size = hidden_size)
 * @returns {ArrayBuffer} Logits (f32 array: vocab_size)
 */
function computeLogits(modelId, hiddenState) {
    // Convert to Buffer if needed
    let buffer;
    if (Buffer.isBuffer(hiddenState)) {
        buffer = hiddenState;
    } else if (hiddenState instanceof Float32Array) {
        // Convert Float32Array to Buffer (little-endian f32)
        buffer = Buffer.from(hiddenState.buffer, hiddenState.byteOffset, hiddenState.length * 4);
    } else {
        throw new Error('hiddenState must be Float32Array or Buffer');
    }
    
    return native.computeLogits(modelId, buffer);
}

module.exports = {
    storeModel,
    getTensor,
    getModelInfo,
    removeModel,
    embedTokens,
    forwardLayer,
    computeLogits,
};

