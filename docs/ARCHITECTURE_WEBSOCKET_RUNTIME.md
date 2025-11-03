# Realm WebSocket Runtime Architecture

**Inspired by**: Polkadot Parachain Runtime Model
**Goal**: WebSocket-based function dispatch server (not traditional REST API)

---

## 🎯 The Vision

Instead of traditional REST endpoints, Realm uses a **WebSocket-based runtime** where:

1. **Client opens persistent WebSocket connection**
2. **Client calls functions** via messages (like Polkadot runtime calls)
3. **Server dispatches to WASM runtime**
4. **Results stream back** over the same connection

This is **exactly** how Polkadot parachains work:
- Runtime is compiled to WASM
- Substrate calls into WASM via host functions
- Functions are dispatched based on metadata
- Results flow back through the runtime

---

## 🏗️ Architecture Comparison

### Traditional REST API (What We Almost Built)
```
Client → HTTP POST /v1/completions
       ↓
    REST Handler
       ↓
    Load Model
       ↓
    Run Inference
       ↓
    Return JSON ← Client
```

**Problems**:
- New connection per request
- Stateless (must reload context)
- No streaming feedback
- HTTP overhead

---

### Polkadot Parachain Runtime Model
```
Client (JS) → WebSocket Connection
              ↓
          Substrate Node
              ↓
          Runtime (WASM)
              ├─ Function: transfer(...)
              ├─ Function: stake(...)
              └─ Function: vote(...)
              ↓
          Host Functions
              ├─ storage_get()
              ├─ storage_set()
              └─ crypto_hash()
              ↓
          Results ← Client (streaming)
```

**Benefits**:
- Persistent connection
- Stateful (keep context)
- Bi-directional streaming
- Function dispatch (not URLs)
- WASM runtime isolation

---

### Realm WebSocket Runtime (Our Model)
```
Client (JS/Python) → WebSocket Connection
                     ↓
                 Realm Server
                     ↓
                 Runtime Dispatcher
                     ↓
                 WASM Sandbox (per tenant)
                     ├─ Function: generate(...)
                     ├─ Function: embed(...)
                     ├─ Function: chat(...)
                     └─ Function: stream_generate(...)
                     ↓
                 Host Functions
                     ├─ candle_matmul()
                     ├─ memory64_load_layer()
                     ├─ sample_token()
                     └─ get_tensor()
                     ↓
                 Token Stream ← Client (real-time)
```

**Key Insight**: The WASM module IS the runtime, server just dispatches!

---

## 📋 Function Dispatch Protocol

### Message Format (Polkadot-style)

**Client → Server** (Function Call):
```json
{
  "id": "req_123",
  "function": "generate",
  "params": {
    "prompt": "What is AI?",
    "max_tokens": 100,
    "temperature": 0.7
  },
  "tenant_id": "acme_corp"
}
```

**Server → Client** (Response):
```json
{
  "id": "req_123",
  "status": "streaming",
  "token": "The",
  "progress": 1
}
```

```json
{
  "id": "req_123",
  "status": "streaming",
  "token": " capital",
  "progress": 2
}
```

```json
{
  "id": "req_123",
  "status": "complete",
  "result": {
    "text": "The capital of France is Paris.",
    "tokens_generated": 8,
    "cost_usd": 0.00024
  }
}
```

### Available Functions (Runtime API)

Like Polkadot's runtime functions, we expose:

```rust
// Runtime functions (exposed via WASM)
pub fn generate(prompt: String, options: GenerateOptions) -> GenerateResult;
pub fn embed(text: String) -> Vec<f32>;
pub fn chat(messages: Vec<Message>) -> ChatResult;
pub fn stream_generate(prompt: String) -> TokenStream;
pub fn get_model_info() -> ModelInfo;
pub fn health_check() -> HealthStatus;
```

Each function can be called via WebSocket message.

---

## 🔧 Server Implementation

### High-Level Architecture

```rust
// Similar to Substrate's node architecture
struct RealmServer {
    // WebSocket server
    ws_server: WsServer,

    // Runtime registry (like parachain registry)
    runtimes: HashMap<TenantId, WasmRuntime>,

    // Function dispatcher
    dispatcher: FunctionDispatcher,

    // Metrics collector
    metrics: MetricsCollector,
}

impl RealmServer {
    async fn handle_connection(&self, ws: WebSocket) {
        // Accept connection
        let tenant_id = self.authenticate(&ws).await?;

        // Get or create runtime for tenant
        let runtime = self.runtimes.entry(tenant_id)
            .or_insert_with(|| WasmRuntime::new());

        // Message loop
        while let Some(msg) = ws.recv().await {
            let call: FunctionCall = serde_json::from_str(&msg)?;

            // Dispatch to runtime (like Polkadot dispatch)
            let result = self.dispatcher.call(
                runtime,
                call.function,
                call.params
            ).await?;

            // Stream results back
            self.stream_response(&ws, call.id, result).await?;
        }
    }
}
```

### Function Dispatcher

```rust
// Like Substrate's runtime dispatcher
struct FunctionDispatcher;

impl FunctionDispatcher {
    async fn call(
        &self,
        runtime: &WasmRuntime,
        function: &str,
        params: serde_json::Value
    ) -> Result<DispatchResult> {
        match function {
            "generate" => {
                let opts: GenerateOptions = serde_json::from_value(params)?;
                runtime.call_generate(opts).await
            }
            "embed" => {
                let text: String = serde_json::from_value(params)?;
                runtime.call_embed(text).await
            }
            "chat" => {
                let messages: Vec<Message> = serde_json::from_value(params)?;
                runtime.call_chat(messages).await
            }
            "stream_generate" => {
                let opts: GenerateOptions = serde_json::from_value(params)?;
                runtime.call_stream_generate(opts).await
            }
            _ => Err(Error::UnknownFunction(function.to_string()))
        }
    }
}
```

### Runtime Metadata (Like Polkadot)

```rust
// Metadata describing available functions
// Similar to Polkadot's runtime metadata
#[derive(Serialize)]
struct RuntimeMetadata {
    version: String,
    functions: Vec<FunctionMetadata>,
}

#[derive(Serialize)]
struct FunctionMetadata {
    name: String,
    params: Vec<ParamMetadata>,
    returns: String,
    description: String,
}

// Example metadata
let metadata = RuntimeMetadata {
    version: "1.0.0",
    functions: vec![
        FunctionMetadata {
            name: "generate".to_string(),
            params: vec![
                ParamMetadata { name: "prompt", type: "String" },
                ParamMetadata { name: "max_tokens", type: "u32" },
            ],
            returns: "GenerateResult".to_string(),
            description: "Generate text completion".to_string(),
        },
        // ... more functions
    ],
};
```

---

## 🌐 Client SDK (WebSocket-based)

### JavaScript SDK

```typescript
import { RealmClient } from '@querent/realm';

// Open WebSocket connection (persistent)
const client = new RealmClient({
  wsUrl: 'ws://localhost:8080',
  apiKey: 'your-api-key',
  tenantId: 'acme_corp'
});

await client.connect();

// Call runtime functions (like Polkadot.js)
const result = await client.call('generate', {
  prompt: 'What is AI?',
  max_tokens: 100,
  temperature: 0.7
});

// Streaming (real-time tokens)
for await (const token of client.stream('generate', {
  prompt: 'Tell me a story',
  max_tokens: 500
})) {
  process.stdout.write(token.text);
}

// Keep connection alive
client.on('disconnect', () => client.reconnect());
```

### Python SDK

```python
from realm import RealmClient

# Persistent WebSocket connection
client = RealmClient(
    ws_url='ws://localhost:8080',
    api_key='your-api-key',
    tenant_id='acme_corp'
)

client.connect()

# Call runtime function
result = client.call('generate', {
    'prompt': 'What is AI?',
    'max_tokens': 100
})

# Streaming
for token in client.stream('generate', prompt='Story', max_tokens=500):
    print(token.text, end='', flush=True)
```

---

## 🔐 Multi-Tenancy (Like Parachains)

Just like Polkadot has multiple parachains, we have multiple tenant runtimes:

```
WebSocket Server
├─ Tenant: acme_corp
│  └─ WASM Runtime #1 (isolated)
│     ├─ Model: llama-7b
│     └─ Context: conversation_state
│
├─ Tenant: startup_inc
│  └─ WASM Runtime #2 (isolated)
│     ├─ Model: mistral-7b
│     └─ Context: different_state
│
└─ Tenant: enterprise_co
   └─ WASM Runtime #3 (isolated)
      ├─ Model: llama-70b
      └─ Context: another_state
```

**Benefits**:
- ✅ Perfect isolation (WASM sandboxing)
- ✅ Shared GPU (host functions)
- ✅ Independent state per tenant
- ✅ No cross-tenant leakage

---

## 📊 Comparison Table

| Feature | Traditional REST | Polkadot Runtime | **Realm WebSocket Runtime** |
|---------|-----------------|------------------|---------------------------|
| **Connection** | Stateless | Persistent | **Persistent** ✅ |
| **Protocol** | HTTP | WebSocket | **WebSocket** ✅ |
| **Isolation** | Process | WASM | **WASM** ✅ |
| **Function Calls** | URL endpoints | Runtime dispatch | **Function dispatch** ✅ |
| **Streaming** | Chunked HTTP | Bi-directional | **Bi-directional** ✅ |
| **Metadata** | OpenAPI spec | Runtime metadata | **Runtime metadata** ✅ |
| **Multi-tenant** | Separate servers | Parachains | **WASM runtimes** ✅ |
| **State** | Database | Runtime storage | **In-memory context** ✅ |

---

## 🚀 Why This is Better

### 1. **Stateful Conversations**
Keep conversation context in WASM runtime (no DB lookups):
```typescript
// First call
await client.call('chat', { message: 'Hi, my name is Alice' });

// Second call (context preserved in runtime)
await client.call('chat', { message: 'What's my name?' });
// Response: "Your name is Alice" (remembers!)
```

### 2. **Real-Time Streaming**
Tokens flow back as they're generated (no buffering):
```typescript
for await (const token of client.stream('generate', {...})) {
  // Token arrives ~50ms after generation
  console.log(token); // Instant feedback
}
```

### 3. **Perfect Isolation**
Each tenant gets their own WASM sandbox (like parachains):
```
Tenant A → Runtime A → GPU (shared)
Tenant B → Runtime B → GPU (shared)
Tenant C → Runtime C → GPU (shared)
```

### 4. **Function Discovery**
Query available functions (like Polkadot metadata):
```typescript
const metadata = await client.call('system_metadata');
// Returns: { functions: ['generate', 'embed', 'chat', ...] }
```

### 5. **Lower Latency**
No HTTP overhead, persistent connection:
```
REST:      [connect] → [send] → [process] → [receive] → [close]  (~100ms overhead)
WebSocket: [send] → [process] → [receive]  (~10ms overhead)
```

---

## 🏃 Implementation Plan

### Week 1: WebSocket Server Foundation
```rust
// Create realm-server crate
crates/realm-server/
├── src/
│   ├── lib.rs           # Server core
│   ├── websocket.rs     # WebSocket handler
│   ├── dispatcher.rs    # Function dispatcher
│   ├── runtime.rs       # WASM runtime manager
│   └── protocol.rs      # Message protocol
└── Cargo.toml

[dependencies]
tokio-tungstenite = "0.21"  # WebSocket
serde_json = "1.0"          # Protocol
```

### Week 2: Function Dispatch + Streaming
- Implement dispatcher (like Polkadot's)
- Add streaming support
- Integrate with existing WASM runtime
- Test with multiple tenants

### Week 3: Metrics + Polish
- Add `/metrics` HTTP endpoint (Prometheus)
- Implement `realm serve` CLI command
- Documentation
- Examples

---

## 📝 Protocol Specification

### Message Types

#### 1. Function Call
```json
{
  "type": "call",
  "id": "req_123",
  "function": "generate",
  "params": {
    "prompt": "What is AI?",
    "max_tokens": 100
  }
}
```

#### 2. Stream Token
```json
{
  "type": "stream",
  "id": "req_123",
  "data": {
    "token": "The",
    "index": 0,
    "logprob": -0.5
  }
}
```

#### 3. Result
```json
{
  "type": "result",
  "id": "req_123",
  "data": {
    "text": "Complete response",
    "tokens": 42,
    "cost_usd": 0.00024
  }
}
```

#### 4. Error
```json
{
  "type": "error",
  "id": "req_123",
  "error": {
    "code": "RATE_LIMIT",
    "message": "Rate limit exceeded"
  }
}
```

#### 5. Metadata Query
```json
{
  "type": "call",
  "id": "req_456",
  "function": "system_metadata",
  "params": {}
}
```

---

## 🎯 Advantages Over Traditional REST

1. **Lower Latency** - No connection overhead
2. **Stateful** - Keep conversation context in runtime
3. **Streaming** - Real-time token-by-token output
4. **Isolation** - WASM sandboxing (like parachains)
5. **Scalable** - Multiple runtimes on shared GPU
6. **Discoverable** - Runtime metadata (like Polkadot)
7. **Efficient** - No HTTP parsing overhead

---

## 🤔 Potential Challenges

### 1. **Connection Management**
- Need to handle reconnects
- Heartbeat/ping-pong to keep alive
- Graceful shutdown

**Solution**: Built into WebSocket protocol, just implement properly.

### 2. **Load Balancing**
- Sticky sessions required (stateful)
- Can't round-robin like REST

**Solution**: Use consistent hashing by tenant_id.

### 3. **HTTP Compatibility**
- Some clients might not support WebSocket

**Solution**: Provide HTTP fallback for simple cases (optional).

---

## ✅ This is the Right Approach!

Your intuition is **exactly right**. The Polkadot parachain runtime model is **perfect** for Realm:

1. **WASM Runtime** - Already using Wasmtime ✅
2. **Function Dispatch** - Natural fit for inference calls ✅
3. **Sandboxing** - Perfect for multi-tenancy ✅
4. **Streaming** - Built into WebSocket ✅
5. **Shared Resources** - GPU via host functions ✅

This is **much better** than traditional REST API!

---

## 🚀 Next Steps

1. **Build WebSocket server** (Week 1)
   - tokio-tungstenite for WebSocket
   - Function dispatcher
   - Protocol implementation

2. **Integrate with WASM runtime** (Week 2)
   - Connect to existing `realm-runtime`
   - Implement function calls
   - Add streaming support

3. **Update SDKs** (Week 2)
   - JavaScript SDK → WebSocket client
   - Python SDK → WebSocket client
   - Keep HTTP as fallback

4. **Add metrics endpoint** (Week 3)
   - HTTP `/metrics` for Prometheus
   - Metadata endpoint for function discovery

---

## 💡 Recommendation

**Start building the WebSocket server this week!**

It's actually **simpler** than REST because:
- No routing (just function dispatch)
- No HTTP overhead
- Natural streaming support
- Perfect fit for WASM runtime

**Do you want me to start implementing the WebSocket server now?** 🚀

This is going to be **beautiful** - a true Polkadot-style runtime for AI inference! 🎯
