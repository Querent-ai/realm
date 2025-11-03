# SDK Implementation Plan

## Architecture Understanding

Realm uses **WASM-based multi-tenancy**:

```
┌─────────────────────────────────────────┐
│  realm-runtime (Native Process)         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │ Tenant A│  │ Tenant B│  │ Tenant C│  │
│  │ WASM    │  │ WASM    │  │ WASM    │  │
│  └────┬────┘  └────┬────┘  └────┬────┘  │
│       └────────────┴────────────┘       │
│              Host Functions              │
│  (candle_matmul, memory64_load_layer)    │
│       └────────────┬────────────┘       │
│              Shared GPU/Memory64         │
└─────────────────────────────────────────┘
```

**Key Points:**
- Models stored in HOST-side Memory64 (shared across tenants)
- Each tenant = isolated WASM instance
- Model registry: Multiple models can be loaded, identified by model ID/name
- One endpoint can serve multiple models

## SDK Design

### Two Modes:

1. **Embedded/Local Mode** (WASM directly)
   - Load `realm.wasm` in browser/Node.js
   - Call WASM functions directly
   - Models loaded via HOST storage (native addon)

2. **Server Mode** (HTTP client) - Future
   - Connect to `realm-runtime server`
   - Models pre-loaded on server
   - HTTP API for inference

### JavaScript/TypeScript SDK

**Structure:**
```
sdks/js/
├── src/
│   ├── index.ts              # Main exports
│   ├── realm.ts              # Realm class (WASM wrapper)
│   ├── client.ts             # HTTP client (future server mode)
│   ├── types.ts              # TypeScript types
│   └── wasm-loader.ts        # WASM initialization
├── wasm/                     # WASM bindings (from realm-wasm/pkg)
│   ├── realm_wasm.js
│   ├── realm_wasm_bg.wasm
│   └── realm_wasm.d.ts
└── package.json
```

**API Design:**
```typescript
// Local/Embedded Mode
const realm = new Realm({
  mode: 'local',              // Use WASM directly
  defaultModel: 'llama-7b',  // Default model from registry
});

// Load model (stores in HOST, returns model ID)
const modelId = await realm.loadModel('./model.gguf');
// OR load from registry if already loaded
await realm.useModel('llama-7b');

// Generate
const response = await realm.generate({
  prompt: 'Hello!',
  model: 'llama-7b',  // Optional, uses defaultModel if not specified
});

// Server Mode (future)
const realm = new Realm({
  mode: 'server',
  endpoint: 'http://localhost:8080',
  defaultModel: 'llama-7b',
});

// Same API, but HTTP instead of WASM
```

### Python SDK

**Structure:**
```
sdks/python/
├── realm/
│   ├── __init__.py           # Main exports
│   ├── realm.py              # Realm class
│   ├── client.py             # HTTP client (future)
│   ├── types.py              # Type definitions
│   └── wasm.py               # WASM bindings (PyO3 or wasmer)
└── setup.py
```

**API Design:**
```python
# Local/Embedded Mode
from realm import Realm

realm = Realm(
    mode='local',
    default_model='llama-7b'
)

# Load model
model_id = realm.load_model('./model.gguf')
# OR use from registry
realm.use_model('llama-7b')

# Generate
response = realm.generate(
    prompt='Hello!',
    model='llama-7b'  # Optional
)

# Server Mode (future)
realm = Realm(
    mode='server',
    endpoint='http://localhost:8080',
    default_model='llama-7b'
)
```

## Implementation Plan

### Phase 1: JavaScript SDK (Local Mode)
1. ✅ Copy WASM bindings from `crates/realm-wasm/pkg/`
2. ✅ Create Realm class wrapper
3. ✅ Add model registry support (list, load, use)
4. ✅ TypeScript types
5. ✅ Examples

### Phase 2: Python SDK (Local Mode)
1. Use PyO3 to wrap Rust runtime OR
2. Use wasmer-python to load WASM
3. Same API as JavaScript
4. Type hints

### Phase 3: Server Mode (Both SDKs)
1. HTTP client implementation
2. Same API surface
3. Auto-detect mode (local vs server)

## Model Registry Concept

**On Server:**
- Models loaded in Memory64 (HOST-side)
- Identified by model ID (hash) or name
- `/v1/models` endpoint lists available models
- Models can be loaded via `/v1/models/load` (future)

**In SDK:**
- `realm.models()` - List available models
- `realm.useModel(id)` - Switch to model (if loaded)
- `realm.loadModel(path)` - Load model (local mode only)
- `defaultModel` in constructor - Set default

## Next Steps

1. Build JavaScript SDK with WASM bindings
2. Add model registry support
3. Test with existing WASM module
4. Build Python SDK (simpler: start with HTTP client wrapper)

