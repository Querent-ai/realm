//! Host Bridge for WASM Module
//! 
//! Provides host functions that WASM calls via extern "C" imports.
//! This bridges between wasm-bindgen WASM and native Node.js storage.

import nativeBridge from '../../bridge/index.js';
import { readFileSync } from 'fs';

// WASM memory access helpers
function readStringFromWasm(wasmMemory, ptr, len) {
    const bytes = new Uint8Array(wasmMemory.buffer, ptr, len);
    return new TextDecoder().decode(bytes);
}

function readBytesFromWasm(wasmMemory, ptr, len) {
    return new Uint8Array(wasmMemory.buffer, ptr, len);
}

function writeBytesToWasm(wasmMemory, ptr, data) {
    const target = new Uint8Array(wasmMemory.buffer, ptr, data.length);
    target.set(data);
}

/**
 * Create host function imports for WASM module
 * 
 * These functions match the extern "C" declarations in realm-wasm/src/lib.rs
 */
export function createHostFunctions(wasmMemory) {
    return {
        env: {
            // realm_store_model(gguf_ptr, gguf_len, model_id) -> model_id
            realm_store_model: (ggufPtr, ggufLen, modelId) => {
                try {
                    // Read GGUF bytes from WASM memory
                    const ggufBytes = readBytesFromWasm(wasmMemory, ggufPtr, ggufLen);
                    
                    // Convert to Buffer for native bridge
                    const buffer = Buffer.from(ggufBytes);
                    
                    // Call native bridge (converts to Option<u32>)
                    const requestedId = modelId === 0 ? null : modelId;
                    const finalId = nativeBridge.storeModel(buffer, requestedId);
                    
                    return finalId;
                } catch (e) {
                    console.error('realm_store_model error:', e);
                    return -1;
                }
            },

            // realm_get_tensor(model_id, tensor_name_ptr, tensor_name_len, out_ptr, out_max_len) -> bytes_written
            realm_get_tensor: (modelId, tensorNamePtr, tensorNameLen, outPtr, outMaxLen) => {
                try {
                    // Read tensor name from WASM memory
                    const tensorName = readStringFromWasm(wasmMemory, tensorNamePtr, tensorNameLen);
                    
                    // Get tensor from native bridge (already dequantized to f32 bytes)
                    const tensorBytes = nativeBridge.getTensor(modelId, tensorName);
                    
                    if (tensorBytes.length > outMaxLen) {
                        console.error(`Buffer too small: need ${tensorBytes.length}, got ${outMaxLen}`);
                        return -1;
                    }
                    
                    // Write dequantized f32 bytes to WASM memory
                    writeBytesToWasm(wasmMemory, outPtr, tensorBytes);
                    
                    return tensorBytes.length;
                } catch (e) {
                    console.error('realm_get_tensor error:', e);
                    return -1;
                }
            },

            // realm_get_model_info(model_id, out_tensor_count_ptr, out_total_size_ptr) -> status
            realm_get_model_info: (modelId, outTensorCountPtr, outTotalSizePtr) => {
                try {
                    const info = nativeBridge.getModelInfo(modelId);
                    
                    // Write u32 tensor_count (4 bytes)
                    const countBytes = new Uint8Array(wasmMemory.buffer, outTensorCountPtr, 4);
                    const countView = new DataView(countBytes.buffer, countBytes.byteOffset, 4);
                    countView.setUint32(0, info.tensor_count, true); // little-endian
                    
                    // Write u64 total_size (8 bytes)
                    const sizeBytes = new Uint8Array(wasmMemory.buffer, outTotalSizePtr, 8);
                    const sizeView = new DataView(sizeBytes.buffer, sizeBytes.byteOffset, 8);
                    sizeView.setBigUint64(0, BigInt(info.total_size), true); // little-endian
                    
                    return 0;
                } catch (e) {
                    console.error('realm_get_model_info error:', e);
                    return -1;
                }
            },

            // realm_remove_model(model_id) -> status
            realm_remove_model: (modelId) => {
                try {
                    nativeBridge.removeModel(modelId);
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
 * Initialize WASM module with host functions
 */
export async function initWasmWithHost(wasmModule, wasmMemory) {
    const imports = createHostFunctions(wasmMemory);
    
    // Instantiate WASM with host function imports
    // Note: This assumes wasm-bindgen's instantiate function accepts imports
    // May need to use WebAssembly.instantiate directly
    
    return imports;
}

