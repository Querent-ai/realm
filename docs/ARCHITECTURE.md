# Realm Production Architecture

## The Core Insight

**Release exactly 2 files:**
1. `realm-runtime` - Native binary (one per platform)
2. `realm.wasm` - WASM module (universal)

**That's the entire product.**

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  realm-runtime (One Process Per Node)                       │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │ Customer A │  │ Customer B │  │ Customer C │  ← WASM    │
│  │ realm.wasm │  │ realm.wasm │  │ realm.wasm │    instances│
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘           │
│        │                │                │                   │
│        └────────────────┴────────────────┘                   │
│                         │                                    │
│         Host Function Calls (candle_matmul, memory64_*)     │
│                         │                                    │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │  Shared Native Resources                            │   │
│  │  ├─ Memory64 Manager (LRU cache, models)           │   │
│  │  ├─ Candle Backends (GPU/CPU)                      │   │
│  │  └─ Wasmtime Engine                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                         │                                    │
│                    GPU (shared)                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Architecture Decisions:**

1. **One process = N customers** (not one container per customer)
2. **WASM provides isolation** (not Docker/k8s)
3. **GPU is shared** (not allocated per customer)
4. **Models in Memory64** (not duplicated per customer)

This gives you **10-16x higher GPU density** than vLLM/container-based approaches.

---

## Deployment Scenarios

### 1. Your SaaS (realm.ai)

**Infrastructure:**
```
Load Balancer (nginx/Cloudflare)
        ↓
┌───────────────────────────────────────┐
│  GPU Node 1 (A100 80GB)               │
│  └─ realm-runtime server              │
│     ├─ Customer 1 (realm.wasm)        │
│     ├─ Customer 2 (realm.wasm)        │
│     ├─ ... (up to 16 customers)       │
│     └─ Customer 16 (realm.wasm)       │
└───────────────────────────────────────┘
┌───────────────────────────────────────┐
│  GPU Node 2 (A100 80GB)               │
│  └─ realm-runtime server              │
│     └─ ... (16 more customers)        │
└───────────────────────────────────────┘
┌───────────────────────────────────────┐
│  GPU Node 3 (A100 80GB)               │
│  └─ ...                               │
└───────────────────────────────────────┘
```

**Commands:**
```bash
# On each GPU node
docker run -d \
  --gpus all \
  -p 8080:8080 \
  -v /models:/models \
  realm-ai/runtime:latest \
  realm-runtime server \
    --port 8080 \
    --max-realms 16 \
    --backend cuda
```

**Customer Usage:**
```javascript
const realm = new Realm({
  apiKey: 'cust_abc123',
  endpoint: 'https://api.realm.ai'
});

await realm.loadModel('llama-2-7b-Q4_K_M');
const result = await realm.generate('What is AI?');
```

**Economics:**
- **Cost:** $3/hour per A100
- **Customers per GPU:** 16
- **Cost per customer:** $3 ÷ 16 = $0.19/hour
- **Charge customer:** $1/hour
- **Margin:** $0.81/hour per customer (81%)

With 100 customers on 7 GPUs:
- **Revenue:** 100 × $1/hour = $100/hour
- **Cost:** 7 × $3/hour = $21/hour
- **Profit:** $79/hour = **79% margin**

---

### 2. Customer Self-Hosting (Small)

**Setup:**
```
Customer's laptop/server
└─ realm-runtime (local mode)
   └─ realm.wasm (single instance)
```

**Commands:**
```bash
# Download binary + WASM
curl -L https://realm.ai/install.sh | sh

# Run locally
realm run --model ./models/llama-2-7b-Q4_K_M.gguf

# Or in code
const realm = new Realm({ mode: 'local' });
await realm.loadModel('./models/llama-2-7b-Q4_K_M.gguf');
```

**Customer benefits:**
- Free (just downloads binary + WASM)
- Complete data privacy
- Works offline
- No ongoing costs

**Your business:**
- Freemium acquisition funnel
- Eventually upgrade to your SaaS
- Good for open source adoption

---

### 3. Customer Self-Hosting (Enterprise)

**Setup:**
```
Customer's cloud (AWS/Azure/GCP)
├─ GPU Instance 1 (g5.xlarge)
│  └─ realm-runtime
│     ├─ Production realm.wasm
│     └─ Staging realm.wasm
├─ GPU Instance 2 (for HA)
│  └─ ...
```

**Commands:**
```bash
# Customer deploys on their cloud
terraform apply  # Uses your Terraform module

# Runs your binary + WASM
realm-runtime server \
  --port 8080 \
  --backend cuda \
  --max-realms 4
```

**Customer benefits:**
- Runs on their infrastructure
- Their compliance/security requirements
- Still uses your binary + WASM (gets updates)

**Your business:**
- Enterprise license revenue
- Support contracts
- Professional services

---

### 4. Edge Deployment

**Setup:**
```
Edge Device (Raspberry Pi, Jetson)
└─ realm-runtime (CPU/ARM)
   └─ realm.wasm (tiny model)
```

**Commands:**
```bash
# On ARM device
realm run \
  --model tinyllama-1.1b-Q4_K_M.gguf \
  --backend cpu
```

**Use cases:**
- IoT devices
- On-premise AI
- Offline inference
- Low-latency edge

---

## Multi-Tenancy Implementation

### How Isolation Works

```rust
// In realm-runtime

pub struct MultiRealmManager {
    realms: HashMap<CustomerId, RealmInstance>,
    gpu_backend: Arc<CandleGpuBackend>,
    memory64: Arc<Memory64Manager>,
}

pub struct RealmInstance {
    wasm_instance: wasmtime::Instance,
    store: wasmtime::Store<RealmState>,
    limits: ResourceLimits,
    metrics: RealmMetrics,
}

impl MultiRealmManager {
    pub fn create_realm(&mut self, customer_id: CustomerId) -> Result<RealmHandle> {
        // Create new WASM instance (isolated sandbox)
        let engine = self.wasmtime_engine.clone();
        let mut store = wasmtime::Store::new(&engine, RealmState::new(customer_id));

        // Set resource limits
        store.limiter(|state| &mut state.limits);

        // Link host functions (shared GPU/Memory64)
        let mut linker = wasmtime::Linker::new(&engine);

        // Customer A's WASM can only call these with its own context
        let gpu = self.gpu_backend.clone();
        linker.func_wrap("env", "candle_matmul",
            move |caller: Caller<RealmState>, a_ptr, b_ptr, ...| {
                let customer_id = caller.data().customer_id;
                // GPU is shared, but results go to this customer's memory
                gpu.matmul(...)
            }
        )?;

        // Load realm.wasm (same binary for all customers)
        let wasm_bytes = include_bytes!("../realm.wasm");
        let module = wasmtime::Module::new(&engine, wasm_bytes)?;
        let instance = linker.instantiate(&mut store, &module)?;

        let realm = RealmInstance { wasm_instance: instance, store, ... };
        self.realms.insert(customer_id, realm);

        Ok(RealmHandle(customer_id))
    }

    pub fn generate(&mut self, handle: RealmHandle, prompt: &str) -> Result<String> {
        let realm = self.realms.get_mut(&handle.0)
            .ok_or(Error::InvalidRealm)?;

        // Call WASM function in isolated instance
        let generate_fn = realm.wasm_instance
            .get_typed_func::<(i32, i32), i32>(&mut realm.store, "generate")?;

        // Execution happens in customer's isolated WASM sandbox
        // But GPU calls go through shared backend
        let result_ptr = generate_fn.call(&mut realm.store, (prompt_ptr, prompt_len))?;

        // Read result from customer's WASM memory
        let memory = realm.wasm_instance.get_memory(&mut realm.store, "memory")?;
        let result = read_string_from_wasm_memory(memory, result_ptr)?;

        Ok(result)
    }
}
```

**Security guarantees:**
1. **Memory isolation**: Customer A cannot read Customer B's WASM memory
2. **Resource limits**: Each realm has CPU/memory/time limits
3. **Shared compute**: GPU is shared but results isolated
4. **Model isolation**: Memory64 manager tracks per-customer model access

---

## Resource Efficiency

### Traditional Multi-Tenancy (Containers)

```
Customer A: Container + Process + Model Load = 8GB RAM + GPU allocation
Customer B: Container + Process + Model Load = 8GB RAM + GPU allocation
...
Total for 16 customers: 128GB RAM + can't share GPU efficiently
```

**Problems:**
- High memory overhead (containers)
- Can't safely share GPU
- Slow startup (container + model load)
- Need orchestration (k8s)

### Realm Multi-Tenancy (WASM)

```
realm-runtime process:
├─ Model loaded once in Memory64: 4GB
├─ GPU backend (shared): Allocated once
├─ Customer A realm.wasm: 50MB (just code + small state)
├─ Customer B realm.wasm: 50MB
├─ ...
├─ Customer 16 realm.wasm: 50MB
Total: 4GB + (16 × 50MB) = ~4.8GB
```

**Benefits:**
- **95% less memory** (4.8GB vs 128GB)
- **GPU safely shared** (WASM isolation)
- **Fast startup** (<100ms for new realm)
- **No orchestration needed** (one process)

---

## What You Ship

### Release Artifacts

```
GitHub Releases: realm v0.1.0

Binaries (realm-runtime):
├─ realm-runtime-linux-x64-cuda.tar.gz (52MB)
├─ realm-runtime-linux-x64-cpu.tar.gz (48MB)
├─ realm-runtime-linux-aarch64.tar.gz (47MB)
├─ realm-runtime-darwin-arm64.tar.gz (49MB)
├─ realm-runtime-darwin-x64.tar.gz (50MB)
└─ realm-runtime-windows-x64-cuda.zip (54MB)

WASM Module (universal):
└─ realm.wasm (2MB)

SDKs:
├─ @realm-ai/sdk (npm) - Bundles binary + WASM
├─ realm-ai (PyPI) - Bundles binary + WASM
└─ realm (crates.io) - Pure Rust, no bundle needed
```

### Docker Images

```bash
# Your official images
docker pull realm-ai/runtime:latest            # Auto-detect GPU
docker pull realm-ai/runtime:cuda              # NVIDIA
docker pull realm-ai/runtime:cpu               # CPU only
docker pull realm-ai/runtime:rocm              # AMD (future)

# One Dockerfile, multi-stage build
FROM nvidia/cuda:12.2-runtime AS cuda
COPY realm-runtime-linux-x64-cuda /usr/local/bin/realm-runtime
COPY realm.wasm /usr/local/share/realm/realm.wasm

FROM ubuntu:22.04 AS cpu
COPY realm-runtime-linux-x64-cpu /usr/local/bin/realm-runtime
COPY realm.wasm /usr/local/share/realm/realm.wasm

# Runtime
EXPOSE 8080
CMD ["realm-runtime", "server", "--port", "8080"]
```

---

## API Modes

### Mode 1: Embedded (Local)

**Customer runs binary directly in their app:**

```javascript
const { Realm } = require('@realm-ai/sdk');

// SDK loads realm-runtime binary + realm.wasm locally
const realm = new Realm({ mode: 'local' });
await realm.loadModel('./models/llama-2-7b.gguf');

const result = await realm.generate('prompt');
// Runs entirely in customer's process
```

**Data flow:**
```
Customer's Node.js → realm-runtime.node (N-API) → realm.wasm → GPU
(all in same process, zero network)
```

### Mode 2: Local Server

**Customer runs binary as standalone server:**

```bash
# Terminal 1: Start server
realm-runtime server --port 8080

# Terminal 2: Use SDK in client mode
```

```javascript
const realm = new Realm({
  mode: 'server',
  endpoint: 'http://localhost:8080'
});

await realm.loadModel('llama-2-7b');
const result = await realm.generate('prompt');
```

**Data flow:**
```
Customer's Node.js → HTTP → realm-runtime server → realm.wasm → GPU
(localhost network, ~1ms latency)
```

### Mode 3: Your SaaS

**Customer uses your hosted infrastructure:**

```javascript
const realm = new Realm({
  apiKey: 'cust_abc123',
  endpoint: 'https://api.realm.ai'  // Your servers
});

await realm.loadModel('llama-2-7b');
const result = await realm.generate('prompt');
```

**Data flow:**
```
Customer's Node.js → HTTPS → Your Load Balancer → realm-runtime → realm.wasm → GPU
(internet latency + your processing)
```

---

## Competitive Economics

### vLLM / Traditional Approach

**Setup:**
- One container per customer
- One model load per container
- Can't share GPU safely

**Costs (A100 80GB):**
- 1 customer per GPU: $3/hour GPU cost
- Charge customer: $4/hour (33% margin)
- To serve 100 customers: 100 GPUs = $300/hour cost

**Challenges:**
- Low margins
- High infrastructure costs
- Complex orchestration (Kubernetes)
- Slow scaling (minutes to spin up new containers)

### Realm Approach

**Setup:**
- One process per GPU
- 16 customers per GPU (WASM instances)
- Model shared via Memory64

**Costs (A100 80GB):**
- 16 customers per GPU: $3/hour ÷ 16 = $0.19/hour per customer
- Charge customer: $1/hour (81% margin)
- To serve 100 customers: 7 GPUs = $21/hour cost

**Advantages:**
- **14x lower infrastructure cost** ($21 vs $300)
- **High margins** (81% vs 33%)
- **Simple ops** (one binary)
- **Instant scaling** (new realm = 100ms)

**Revenue comparison (100 customers):**
- vLLM: $400/hour revenue - $300/hour cost = $100/hour profit (25% margin)
- Realm: $100/hour revenue - $21/hour cost = $79/hour profit (79% margin)

---

## Security Model

### Isolation Guarantees

1. **Memory Isolation**
   - Each realm.wasm has separate linear memory
   - Customer A cannot read Customer B's memory
   - Enforced by WASM spec + Wasmtime

2. **Compute Isolation**
   - Host functions check caller's customer_id
   - Results written only to caller's memory
   - GPU is shared but outputs isolated

3. **Model Access Control**
   - Memory64 manager tracks per-customer permissions
   - Customer A can only load models they own/paid for
   - Shared model cache for efficiency

4. **Resource Limits**
   - Each realm has CPU time limit
   - Memory limit (WASM heap size)
   - Request rate limiting
   - Enforced via Wasmtime resource limiter

### Attack Scenarios

**Scenario 1: Customer tries to access another's data**
- **Attack**: WASM tries to read outside its linear memory
- **Defense**: Impossible - WASM memory is isolated by design
- **Result**: Instant trap, realm terminated

**Scenario 2: Customer tries to DOS via infinite loop**
- **Attack**: WASM runs infinite loop to hog CPU
- **Defense**: Wasmtime fuel metering - execution times out
- **Result**: Realm terminated, others unaffected

**Scenario 3: Customer tries to use too much memory**
- **Attack**: WASM tries to allocate huge arrays
- **Defense**: Memory limit set per realm (e.g., 2GB)
- **Result**: Allocation fails, realm OOM, others unaffected

**Scenario 4: Customer tries to access GPU directly**
- **Attack**: WASM tries to call CUDA directly
- **Defense**: Impossible - WASM has no GPU access
- **Result**: Only host functions can access GPU

---

## Summary

**You release:**
- 1 binary (`realm-runtime`) × N platforms
- 1 WASM (`realm.wasm`) - universal

**Customers can:**
- Use your SaaS (pay you per hour/request)
- Self-host (download binary, run locally)
- Deploy on their cloud (enterprise license)
- Run on edge devices (same binary)

**Your SaaS wins because:**
- 10-16x higher GPU density than competition
- 79% margins vs 25% for traditional approaches
- Simple ops (no Kubernetes needed)
- Instant scaling (new realm = 100ms)

**This is the entire business model.**
