// Type declaration for dynamically imported WASM module
// This allows TypeScript to compile even when the WASM file doesn't exist yet

declare module '../wasm/realm_wasm.js' {
  export default function init(): Promise<any>;
  export class Realm {
    // Add WASM-specific methods here as needed
    new(): Realm;
  }
}
