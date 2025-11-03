/**
 * Realm.ai JavaScript/TypeScript SDK
 * 
 * Official SDK for Realm multi-tenant LLM inference runtime.
 * Uses WASM bindings for local inference or HTTP client for server mode.
 */

export { Realm, initWasm } from './realm';
export { RealmRegistry } from './registry';
export * from './types';
export type {
  RealmOptions,
  GenerationConfig,
  GenerationResponse,
  ModelInfo,
  RealmMode,
} from './types';

// Default export
import { Realm } from './realm';
export default Realm;
