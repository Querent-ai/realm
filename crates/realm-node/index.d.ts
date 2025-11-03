/**
 * Realm Node.js SDK Type Definitions
 */

/**
 * Store model in HOST storage
 */
export function storeModel(ggufBytes: Buffer | Uint8Array, modelId?: number | null): number;

/**
 * Get tensor from HOST storage (dequantized to f32)
 */
export function getTensor(modelId: number, tensorName: string): ArrayBuffer;

/**
 * Get model metadata
 */
export function getModelInfo(modelId: number): { tensor_count: number; total_size: number };

/**
 * Remove model from storage
 */
export function removeModel(modelId: number): void;

/**
 * Embed tokens using HOST-side computation (bypasses WASM)
 * @param modelId - Model ID
 * @param tokenIds - Token IDs (Uint32Array or Buffer of u32 values)
 * @returns Hidden states as ArrayBuffer (f32 array: token_count * hidden_size)
 */
export function embedTokens(modelId: number, tokenIds: Uint32Array | Buffer): ArrayBuffer;

/**
 * Forward through one transformer layer using HOST-side computation (bypasses WASM)
 * @param modelId - Model ID
 * @param layerIdx - Layer index (0..num_layers-1)
 * @param hiddenStates - Hidden states (Float32Array or Buffer of f32 values)
 * @param position - Position in sequence (for KV cache)
 * @returns Output hidden states as ArrayBuffer (same size as input)
 */
export function forwardLayer(modelId: number, layerIdx: number, hiddenStates: Float32Array | Buffer, position: number): ArrayBuffer;

/**
 * Compute logits from hidden states using HOST-side computation (bypasses WASM)
 * @param modelId - Model ID
 * @param hiddenState - Hidden state (Float32Array or Buffer of f32 values, size = hidden_size)
 * @returns Logits as ArrayBuffer (f32 array: vocab_size)
 */
export function computeLogits(modelId: number, hiddenState: Float32Array | Buffer): ArrayBuffer;

