# Examples Organization Plan

**Goal**: Organize examples showing different ways to use Realm, all producing "Paris" as output.

---

## 🎯 Proposed Structure

```
examples/
├── paris/
│   ├── native/              # Pure Rust (no WASM)
│   │   ├── Cargo.toml
│   │   ├── src/main.rs
│   │   └── README.md
│   │
│   ├── wasm/                 # WASM with host functions
│   │   ├── Cargo.toml
│   │   ├── src/main.rs       # Host runner
│   │   ├── wasm/             # WASM module code
│   │   └── README.md
│   │
│   ├── nodejs-wasm/          # Node.js + WASM (local)
│   │   ├── package.json
│   │   ├── index.js
│   │   └── README.md
│   │
│   ├── nodejs-sdk/           # Node.js WebSocket SDK
│   │   ├── package.json
│   │   ├── index.js
│   │   └── README.md
│   │
│   ├── python-sdk/           # Python WebSocket SDK
│   │   ├── requirements.txt
│   │   ├── main.py
│   │   └── README.md
│   │
│   └── server/               # Via WebSocket server
│       ├── start-server.sh
│       ├── client-nodejs.js
│       ├── client-python.py
│       └── README.md
│
└── README.md                 # Overview of all examples
```

---

## 📋 Each Example Should:

1. **Load a model** (tinyllama-1.1b.Q4_K_M.gguf or similar)
2. **Prompt**: "What is the capital of France?"
3. **Expected**: Response containing "Paris"
4. **Show**: Clear success/failure output
5. **Document**: How to run it

---

## 🎯 Implementation Plan

### 1. Native Rust Example ✅
- **Location**: `examples/paris/native/`
- **Status**: Already exists as `examples/paris-generation/`
- **Action**: Move/rename to `examples/paris/native/`

### 2. WASM Example ✅
- **Location**: `examples/paris/wasm/`
- **Status**: Partially exists (wasm-host-runner, wasm-paris-generation)
- **Action**: Consolidate into clean example

### 3. Node.js WASM Example ✅
- **Location**: `examples/paris/nodejs-wasm/`
- **Status**: Exists in `examples/js-paris-generation/`
- **Action**: Move/consolidate

### 4. Node.js SDK Example ⚠️
- **Location**: `examples/paris/nodejs-sdk/`
- **Status**: SDK exists, need example using it
- **Action**: Create example using WebSocket SDK

### 5. Python SDK Example ⚠️
- **Location**: `examples/paris/python-sdk/`
- **Status**: SDK exists, need example using it
- **Action**: Create example using WebSocket SDK

### 6. Server Example ⚠️
- **Location**: `examples/paris/server/`
- **Status**: Server exists, need end-to-end example
- **Action**: Create example showing server + client

---

## 💡 Benefits

1. **Clear Organization**: Each folder shows one way to use Realm
2. **Easy Comparison**: Same test ("Paris") across all methods
3. **Complete Coverage**: Shows all usage patterns
4. **Easy to Find**: Developers can quickly find their use case

---

## 📝 Next Steps

1. Create `examples/paris/` directory structure
2. Move/consolidate existing examples
3. Create missing examples (Node.js SDK, Python SDK, Server)
4. Add comprehensive README
5. Test all examples produce "Paris"

