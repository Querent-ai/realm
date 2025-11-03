/**
 * Realm.ai SDK - WASM-based inference
 * 
 * Wraps the WASM bindings with a cleaner API and model registry support
 */

import type {
  RealmOptions,
  GenerationConfig,
  GenerationResponse,
  ModelInfo,
} from './types';
import { ModelNotLoadedError, RealmError } from './types';

// WASM bindings will be imported at runtime
let wasmModule: any = null;

/**
 * Initialize WASM module
 */
export async function initWasm(
  wasmBytes?: Uint8Array | Buffer,
  wasmPath?: string
): Promise<void> {
  if (wasmModule) {
    return; // Already initialized
  }

  try {
    // Try to load WASM bindings
    // In production, these come from realm-wasm/pkg
    if (wasmBytes) {
      // Load from bytes - WASM bindings expect the .wasm binary
      // For now, we'll require wasmPath for proper initialization
      throw new RealmError(
        'WASM initialization from bytes requires wasmPath. Use wasmPath option.',
        'WASM_BYTES_NOT_SUPPORTED'
      );
    } else if (wasmPath) {
      // Load from path
      const wasmBindings: any = await import(wasmPath);
      wasmModule = wasmBindings;
      
      // If default() is a function, call it to initialize
      if (wasmBindings.default && typeof wasmBindings.default === 'function') {
        const initialized = await (wasmBindings.default as () => Promise<any>)();
        if (initialized && initialized.Realm) {
          wasmModule = initialized;
        }
      }
    } else {
      // Try default path (wasm-pack output)
      try {
        // Default to realm-wasm/pkg location
        const wasmBindings: any = await import('../wasm/realm_wasm.js');
        wasmModule = wasmBindings;
        
        // If default() is a function, call it to initialize
        if (wasmBindings.default && typeof wasmBindings.default === 'function') {
          const initialized = await (wasmBindings.default as () => Promise<any>)();
          if (initialized && initialized.Realm) {
            wasmModule = initialized;
          }
        }
      } catch (e) {
        throw new RealmError(
          'WASM module not found. Install realm-wasm package or provide wasmPath option. ' +
          'Expected: ../wasm/realm_wasm.js from realm-wasm/pkg/',
          'WASM_NOT_FOUND'
        );
      }
    }
  } catch (error) {
    throw new RealmError(
      `Failed to initialize WASM: ${error instanceof Error ? error.message : String(error)}`,
      'WASM_INIT_FAILED'
    );
  }
}

/**
 * Main Realm class - WASM-based inference
 * 
 * Note: Each Realm instance can hold ONE model at a time.
 * For multiple models, use RealmRegistry or create multiple Realm instances.
 */
export class Realm {
  private wasmRealm: any = null;
  private currentModelId: string | null = null;
  private defaultModel: string | undefined;

  constructor(options: RealmOptions = {}) {
    const { mode = 'local', defaultModel, wasmPath } = options;

    if (mode !== 'local') {
      throw new RealmError(
        'Server mode not yet implemented. Use mode: "local" for WASM inference.',
        'SERVER_MODE_NOT_IMPLEMENTED'
      );
    }

    this.defaultModel = defaultModel;

    // Auto-initialize WASM on first use
    // Can be called explicitly: await realm.init()
  }

  /**
   * Initialize WASM module (called automatically, but can be called explicitly)
   */
  async init(wasmBytes?: Uint8Array | Buffer, wasmPath?: string): Promise<void> {
    await initWasm(wasmBytes, wasmPath || '../wasm/realm_wasm.js');
    
    if (!wasmModule || !wasmModule.Realm) {
      throw new RealmError('WASM module not loaded. Ensure realm-wasm bindings are available.', 'WASM_NOT_LOADED');
    }

    if (!this.wasmRealm) {
      this.wasmRealm = new wasmModule.Realm();
    }
  }

  /**
   * Load a model from GGUF bytes
   * 
   * Note: Each Realm instance can hold ONE model.
   * Calling loadModel() again will replace the current model.
   * For multiple models, create multiple Realm instances or use RealmRegistry.
   * 
   * @param modelBytes GGUF model file bytes
   * @param modelId Optional model identifier (for tracking)
   * @returns Model ID (provided or auto-generated)
   */
  async loadModel(
    modelBytes: Uint8Array | Buffer,
    modelId?: string
  ): Promise<string> {
    await this.ensureInitialized();

    try {
      // Convert Buffer to Uint8Array if needed
      const bytes = modelBytes instanceof Buffer
        ? new Uint8Array(modelBytes)
        : modelBytes;

      this.wasmRealm.loadModel(bytes);

      // Generate model ID if not provided
      const id = modelId || `model_${Date.now()}`;
      this.currentModelId = id;

      return id;
    } catch (error) {
      throw new RealmError(
        `Failed to load model: ${error instanceof Error ? error.message : String(error)}`,
        'MODEL_LOAD_FAILED'
      );
    }
  }

  /**
   * Get current model ID
   */
  getCurrentModel(): string | null {
    return this.currentModelId;
  }

  /**
   * Check if a model is loaded in this Realm instance
   */
  isModelLoaded(): boolean {
    return this.wasmRealm?.isLoaded() || false;
  }

  /**
   * Set generation configuration
   */
  setConfig(config: GenerationConfig): void {
    this.ensureInitialized();
    this.ensureModelLoaded();

    if (!wasmModule) {
      throw new RealmError('WASM module not initialized', 'WASM_NOT_INITIALIZED');
    }

    const wasmConfig = new wasmModule.WasmGenerationConfig();
    
    if (config.maxTokens !== undefined) {
      wasmConfig.max_tokens = config.maxTokens;
    }
    if (config.temperature !== undefined) {
      wasmConfig.temperature = config.temperature;
    }
    if (config.topK !== undefined) {
      wasmConfig.top_k = config.topK;
    }
    if (config.topP !== undefined) {
      wasmConfig.top_p = config.topP;
    }
    if (config.repetitionPenalty !== undefined) {
      wasmConfig.repetition_penalty = config.repetitionPenalty;
    }

    this.wasmRealm.setConfig(wasmConfig);
    
    // Clean up config object
    wasmConfig.free();
  }

  /**
   * Generate text from a prompt
   */
  async generate(
    prompt: string,
    options?: GenerationConfig & { model?: string }
  ): Promise<GenerationResponse> {
    this.ensureInitialized();
    this.ensureModelLoaded();

    // Note: Model switching per-request not supported in single Realm instance
    // For multiple models, use RealmRegistry
    if (options?.model && options.model !== this.currentModelId) {
      throw new RealmError(
        `Cannot switch to model "${options.model}" in single Realm instance. ` +
        `Use RealmRegistry for multiple models, or create separate Realm instances.`,
        'MODEL_SWITCHING_NOT_SUPPORTED'
      );
    }

    // Apply generation config if provided
    if (options) {
      const { model, ...config } = options;
      if (Object.keys(config).length > 0) {
        this.setConfig(config);
      }
    }

    try {
      const text = this.wasmRealm.generate(prompt);
      const modelId = this.currentModelId || this.defaultModel || 'unknown';

      return {
        text,
        model: modelId,
      };
    } catch (error) {
      throw new RealmError(
        `Generation failed: ${error instanceof Error ? error.message : String(error)}`,
        'GENERATION_FAILED'
      );
    }
  }

  /**
   * Get model vocabulary size
   */
  vocabSize(): number {
    this.ensureInitialized();
    this.ensureModelLoaded();
    return this.wasmRealm.vocabSize();
  }

  /**
   * Get model configuration
   */
  getModelConfig(): any {
    this.ensureInitialized();
    this.ensureModelLoaded();
    
    try {
      const configJson = this.wasmRealm.getModelConfig();
      return JSON.parse(configJson);
    } catch (error) {
      throw new RealmError(
        `Failed to parse model config: ${error instanceof Error ? error.message : String(error)}`,
        'CONFIG_PARSE_FAILED'
      );
    }
  }

  /**
   * Free resources
   */
  dispose(): void {
    if (this.wasmRealm) {
      this.wasmRealm.free();
      this.wasmRealm = null;
    }
    this.currentModelId = null;
  }

  // Private helpers

  private async ensureInitialized(): Promise<void> {
    if (!wasmModule || !this.wasmRealm) {
      await this.init();
    }
  }

  private ensureModelLoaded(): void {
    if (!this.wasmRealm?.isLoaded()) {
      throw new ModelNotLoadedError(this.currentModelId || undefined);
    }
  }
}

