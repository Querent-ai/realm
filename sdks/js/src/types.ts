/**
 * TypeScript type definitions for Realm.ai SDK
 */

/**
 * SDK mode: how to connect to Realm
 */
export type RealmMode = 'local' | 'server';

/**
 * Realm client configuration
 */
export interface RealmOptions {
  /** SDK mode: 'local' uses WASM directly, 'server' uses HTTP API */
  mode?: RealmMode;
  /** Default model to use for all requests (model ID or name from registry) */
  defaultModel?: string;
  /** Server endpoint (required for 'server' mode) */
  endpoint?: string;
  /** API key for server authentication (optional) */
  apiKey?: string;
  /** Path to WASM module (for 'local' mode, defaults to bundled WASM) */
  wasmPath?: string;
}

/**
 * Generation configuration
 */
export interface GenerationConfig {
  /** Maximum tokens to generate */
  maxTokens?: number;
  /** Sampling temperature (0.0 to 2.0) */
  temperature?: number;
  /** Top-k sampling */
  topK?: number;
  /** Top-p (nucleus) sampling */
  topP?: number;
  /** Repetition penalty */
  repetitionPenalty?: number;
}

/**
 * Generation response
 */
export interface GenerationResponse {
  /** Generated text */
  text: string;
  /** Model used */
  model: string;
  /** Number of tokens generated */
  tokens?: number;
  /** Finish reason */
  finishReason?: 'stop' | 'length' | 'error';
}

/**
 * Model information from registry
 */
export interface ModelInfo {
  /** Model identifier (ID or name) */
  id: string;
  /** Model display name */
  name: string;
  /** Whether model is currently loaded */
  loaded: boolean;
  /** Model description (optional) */
  description?: string;
  /** Model parameters (e.g., "7B", "13B") */
  parameters?: string;
}

/**
 * Error classes
 */
export class RealmError extends Error {
  constructor(
    message: string,
    public code?: string
  ) {
    super(message);
    this.name = 'RealmError';
  }
}

export class ModelNotLoadedError extends RealmError {
  constructor(modelId?: string) {
    super(
      modelId
        ? `Model "${modelId}" is not loaded. Call loadModel() first.`
        : 'No model loaded. Call loadModel() first.',
      'MODEL_NOT_LOADED'
    );
    this.name = 'ModelNotLoadedError';
  }
}
