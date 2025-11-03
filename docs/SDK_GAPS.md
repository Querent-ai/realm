# SDK Gaps Analysis - What's Missing

## Current Status

### ✅ JavaScript SDK - Partial Implementation
- ✅ Realm class wrapper exists
- ✅ TypeScript types
- ✅ Model registry concept (but needs fixes - see below)
- ✅ Compiles successfully
- ⚠️ **Not fully tested**
- ⚠️ **Model registry won't work as designed** (see issue below)

### ⚠️ Python SDK - Placeholder Only
- ⚠️ Just a placeholder file
- ⚠️ No implementation
- ⚠️ No tests

---

## Critical Issues Found

### Issue 1: Model Registry Doesn't Work as Designed ❌

**Problem:**
The WASM `Realm` class can only hold **ONE model** per instance:
- `self.model: Option<Model>` - single model
- `self.tokenizer: Option<Tokenizer>` - single tokenizer  
- `loadModel()` replaces the current model if called again

**Current SDK Implementation:**
```typescript
// This won't work correctly:
await realm.loadModel(bytes1, 'llama-7b');
await realm.loadModel(bytes2, 'llama-13b'); // ❌ Replaces llama-7b!
realm.useModel('llama-7b'); // ❌ Model already replaced
```

**Solution:**
For multiple models, we need **multiple Realm instances**:
```typescript
// Correct approach:
const realm7b = new Realm();
await realm7b.loadModel(bytes7b);

const realm13b = new Realm();
await realm13b.loadModel(bytes13b);

// Or: Realm registry that manages multiple instances
const registry = new RealmRegistry();
await registry.loadModel('llama-7b', bytes7b);
await registry.loadModel('llama-13b', bytes13b);
await registry.generate('llama-7b', 'Hello!');
```

---

## Missing Features

### JavaScript SDK

1. **✅ Model Registry Manager** (Needs redesign)
   - Current: Won't work with single-instance limitation
   - Needed: `RealmRegistry` class that manages multiple `Realm` instances

2. **❌ Streaming Support**
   - WASM `generate()` returns `string`, not a stream
   - Need to add streaming wrapper or wait for WASM to support it

3. **❌ Chat Completions**
   - No chat format support
   - Need message formatting (system/user/assistant)

4. **❌ Error Handling**
   - Need better WASM error propagation
   - JsError from WASM needs proper conversion

5. **❌ Initialization Examples**
   - No working example showing WASM loading
   - No browser vs Node.js handling

6. **❌ Tests**
   - No unit tests
   - No integration tests
   - No test with real WASM module

7. **❌ Proper WASM Loading**
   - Current: Assumes WASM is already built
   - Need: Proper initialization workflow
   - Need: Handle browser vs Node.js environments

8. **❌ Context Manager / Cleanup**
   - `dispose()` exists but not tested
   - Need proper resource cleanup

### Python SDK

1. **❌ Complete Implementation**
   - Only placeholder exists
   - Need full implementation

2. **❌ Architecture Decision**
   - HTTP client vs PyO3 vs wasmer?
   - Recommend: HTTP client (simplest)

3. **❌ All Features**
   - Everything missing

---

## What Needs to Be Fixed/Added

### Priority 1: Critical Fixes

1. **Fix Model Registry** (JavaScript)
   - Create `RealmRegistry` class that manages multiple Realm instances
   - Each model = separate Realm instance
   - Implement proper model switching

2. **Add Working Examples** (JavaScript)
   - Example that actually works with real WASM
   - Show proper initialization
   - Show multiple models (via registry)

3. **Implement Python SDK** (Python)
   - At minimum: HTTP client wrapper
   - Full API parity with JavaScript

### Priority 2: Important Features

4. **Streaming Support** (Both)
   - If WASM doesn't support streaming, add wrapper
   - Or document limitation

5. **Chat Completions** (Both)
   - Format messages correctly
   - Handle system/user/assistant roles

6. **Tests** (Both)
   - Unit tests
   - Integration tests
   - Test with real models

7. **Error Handling** (JavaScript)
   - Better WASM error conversion
   - Proper error types

### Priority 3: Nice to Have

8. **Documentation**
   - API reference
   - Migration guide
   - Performance tips

9. **Advanced Features**
   - Batch generation
   - Token-level callbacks
   - Model info/details

---

## Recommended Fixes

### Fix 1: RealmRegistry for Multiple Models

```typescript
export class RealmRegistry {
  private realms: Map<string, Realm> = new Map();
  private defaultModel?: string;

  async loadModel(modelId: string, modelBytes: Uint8Array): Promise<void> {
    const realm = new Realm();
    await realm.init();
    await realm.loadModel(modelBytes);
    this.realms.set(modelId, realm);
  }

  async generate(modelId: string, prompt: string, options?: GenerationConfig): Promise<GenerationResponse> {
    const realm = this.realms.get(modelId);
    if (!realm) {
      throw new RealmError(`Model "${modelId}" not loaded`);
    }
    return await realm.generate(prompt, options);
  }
}
```

### Fix 2: Working Example

```typescript
import { Realm } from '@realm-ai/sdk';
import * as fs from 'fs';

const realm = new Realm();
await realm.init('../wasm/realm_wasm.js');

const modelBytes = fs.readFileSync('./model.gguf');
await realm.loadModel(modelBytes, 'my-model');

const response = await realm.generate('Hello!');
console.log(response.text);
```

---

## Summary

**JavaScript SDK:**
- ✅ Core structure exists
- ✅ Types defined
- ⚠️ Model registry needs redesign (multiple Realm instances)
- ❌ Missing: tests, examples, streaming, chat

**Python SDK:**
- ❌ Just placeholder
- ❌ Need full implementation

**Recommendation:**
1. Fix model registry design (RealmRegistry pattern)
2. Add working example with real WASM
3. Implement Python SDK (HTTP client first)
4. Add tests for both

