/**
 * Realm Registry - Manages multiple Realm instances (one per model)
 * 
 * Since each Realm instance can only hold one model, the registry
 * creates and manages multiple Realm instances for multiple models.
 */

import { Realm } from './realm';
import type {
  GenerationConfig,
  GenerationResponse,
  ModelInfo,
} from './types';
import { RealmError } from './types';

/**
 * Registry for managing multiple Realm instances (models)
 */
export class RealmRegistry {
  private realms: Map<string, Realm> = new Map();
  private defaultModel?: string;

  constructor(defaultModel?: string) {
    this.defaultModel = defaultModel;
  }

  /**
   * Load a model into the registry
   * Creates a new Realm instance for this model
   */
  async loadModel(
    modelId: string,
    modelBytes: Uint8Array | Buffer,
    wasmPath?: string
  ): Promise<void> {
    if (this.realms.has(modelId)) {
      throw new RealmError(
        `Model "${modelId}" already loaded. Use unloadModel() first.`,
        'MODEL_ALREADY_LOADED'
      );
    }

    const realm = new Realm({ defaultModel: modelId });
    await realm.init(undefined, wasmPath);
    await realm.loadModel(modelBytes, modelId);

    this.realms.set(modelId, realm);
  }

  /**
   * Unload a model from the registry
   */
  unloadModel(modelId: string): void {
    const realm = this.realms.get(modelId);
    if (realm) {
      realm.dispose();
      this.realms.delete(modelId);
    }
  }

  /**
   * Generate text using a specific model
   */
  async generate(
    prompt: string,
    options?: GenerationConfig & { model?: string }
  ): Promise<GenerationResponse> {
    const modelId = options?.model || this.defaultModel;

    if (!modelId) {
      throw new RealmError(
        'No model specified. Provide model in options or set defaultModel in constructor.',
        'NO_MODEL_SPECIFIED'
      );
    }

    const realm = this.realms.get(modelId);
    if (!realm) {
      throw new RealmError(
        `Model "${modelId}" not found in registry. Load it first with loadModel().`,
        'MODEL_NOT_FOUND'
      );
    }

    return await realm.generate(prompt, options);
  }

  /**
   * Get list of models in registry
   */
  getModels(): ModelInfo[] {
    return Array.from(this.realms.keys()).map(id => ({
      id,
      name: id,
      loaded: true,
    }));
  }

  /**
   * Check if a model is loaded
   */
  hasModel(modelId: string): boolean {
    return this.realms.has(modelId);
  }

  /**
   * Get a Realm instance for a model (advanced usage)
   */
  getRealm(modelId: string): Realm | undefined {
    return this.realms.get(modelId);
  }

  /**
   * Set default model
   */
  setDefaultModel(modelId: string): void {
    if (!this.realms.has(modelId)) {
      throw new RealmError(
        `Model "${modelId}" not found in registry. Load it first.`,
        'MODEL_NOT_FOUND'
      );
    }
    this.defaultModel = modelId;
  }

  /**
   * Get default model
   */
  getDefaultModel(): string | undefined {
    return this.defaultModel;
  }

  /**
   * Cleanup all models
   */
  dispose(): void {
    for (const realm of this.realms.values()) {
      realm.dispose();
    }
    this.realms.clear();
  }
}

