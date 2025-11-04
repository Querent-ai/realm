# Paris Generation Examples

This directory contains **complete examples** showing all ways to use Realm, all producing "Paris" as output:

- **Question**: "What is the capital of France?"
- **Expected Answer**: "Paris"

## 🎯 Why Paris?

"Paris" is our **test case** - a simple question with a clear answer that validates:
- ✅ Model loading works
- ✅ Inference pipeline works
- ✅ Tokenization works
- ✅ Generation works

## 📁 Examples

### 1. Native Rust (`native/`)
- **Direct Rust API** - no WASM, no server
- **Simplest** way to use Realm
- Perfect for embedding in Rust applications

```bash
cd native
cargo run --release -- /path/to/model.gguf
```

### 2. WASM (`wasm/`)
- **WASM module** with host functions
- Demonstrates WASM orchestration + native compute
- Shows multi-tenancy architecture

```bash
cd wasm
cargo run --release -- /path/to/model.gguf
```

### 3. Node.js WASM (`nodejs-wasm/`)
- **WASM module in Node.js** (local, no server)
- Shows 98% memory reduction with host-side storage
- Perfect for edge deployment

```bash
cd nodejs-wasm
node index.js /path/to/model.gguf
```

### 4. Node.js SDK (`nodejs-sdk/`)
- **WebSocket client** connecting to Realm server
- Production-ready SDK
- Multi-tenant ready

```bash
# Start server first
cd ../server && cargo run --release

# Then run client
cd ../nodejs-sdk
node index.js
```

### 5. Python SDK (`python-sdk/`)
- **WebSocket client** in Python
- Async/await support
- Production-ready SDK

```bash
# Start server first
cd ../server && cargo run --release

# Then run client
cd ../python-sdk
python main.py
```

### 6. Server (`server/`)
- **WebSocket server** setup
- Model loading and serving
- Client connection examples

```bash
cd server
./start-server.sh
```

## 🏗️ Architecture Comparison

| Example | Architecture | Server? | WASM? | Best For |
|---------|-------------|---------|-------|----------|
| **Native** | Direct Rust API | ❌ | ❌ | Rust apps |
| **WASM** | WASM + Host functions | ❌ | ✅ | Multi-tenant |
| **Node.js WASM** | WASM in Node.js | ❌ | ✅ | Edge deployment |
| **Node.js SDK** | WebSocket client | ✅ | ✅ | Production apps |
| **Python SDK** | WebSocket client | ✅ | ✅ | Production apps |
| **Server** | WebSocket server | ✅ | ✅ | Serving models |

## 🚀 Quick Start

### Test All Examples

```bash
# 1. Native Rust
cd native && cargo run --release -- /path/to/model.gguf

# 2. WASM
cd ../wasm && cargo run --release -- /path/to/model.gguf

# 3. Node.js WASM
cd ../nodejs-wasm && node index.js /path/to/model.gguf

# 4. Node.js SDK (requires server)
cd ../server && cargo run --release &
cd ../nodejs-sdk && node index.js

# 5. Python SDK (requires server)
cd ../server && cargo run --release &
cd ../python-sdk && python main.py
```

## 📝 Expected Output

All examples should produce:

```
✅ SUCCESS: Model correctly identified Paris as the capital of France!
```

## 🔍 What Each Example Validates

| Example | Validates |
|---------|-----------|
| Native | Core inference works |
| WASM | WASM orchestration works |
| Node.js WASM | Host-side storage works |
| Node.js SDK | WebSocket protocol works |
| Python SDK | Cross-language SDK works |
| Server | Multi-tenant server works |

## 📚 Documentation

Each example has its own README with:
- Detailed setup instructions
- Architecture diagrams
- Expected output
- Troubleshooting

## 🎯 Next Steps

1. **Try each example** to understand different usage patterns
2. **Modify prompts** to test different questions
3. **Compare performance** across different approaches
4. **Integrate** into your own applications

---

**All examples produce "Paris" - proving Realm works end-to-end! 🎉**

