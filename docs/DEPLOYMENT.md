# Realm Deployment Guide

This guide shows exactly how to deploy Realm in production for your SaaS offering.

---

## Production Setup (Your SaaS)

### Infrastructure Requirements

**Minimum for MVP:**
- 1-2 GPU nodes (A100/H100 or even T4 for testing)
- 1 load balancer (nginx/Cloudflare)
- Object storage for models (S3/GCS)

**Recommended for Production:**
- 5-10 GPU nodes (for redundancy + scale)
- Load balancer with health checks
- Object storage + CDN
- Monitoring (Prometheus/Grafana)
- Logging (Loki/ELK)

---

## Deployment Steps

### 1. Build Binaries

```bash
# Clone repo
git clone https://github.com/yourusername/realm.git
cd realm

# Build realm-runtime for Linux + CUDA
CUDA_COMPUTE_CAP=80 cargo build --release \
  --bin realm-runtime \
  --features cuda

# Binary at: target/release/realm-runtime
```

```bash
# Build realm.wasm
cd crates/realm-wasm
wasm-pack build --target web --features memory64

# WASM at: pkg/realm_wasm_bg.wasm
```

**Upload to releases:**
```bash
# Tag release
git tag v0.1.0
git push origin v0.1.0

# GitHub Actions automatically builds all platforms
# Uploads to: github.com/yourusername/realm/releases/tag/v0.1.0
```

### 2. Create Docker Image

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.2-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy binary and WASM
COPY target/release/realm-runtime /usr/local/bin/
COPY crates/realm-wasm/pkg/realm_wasm_bg.wasm /usr/local/share/realm/realm.wasm

# Create directory for models
RUN mkdir -p /models

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run server
CMD ["realm-runtime", "server", \
     "--port", "8080", \
     "--max-realms", "16", \
     "--backend", "cuda", \
     "--model-dir", "/models"]
```

```bash
# Build image
docker build -t realm-ai/runtime:v0.1.0 .

# Push to registry
docker push realm-ai/runtime:v0.1.0
docker tag realm-ai/runtime:v0.1.0 realm-ai/runtime:latest
docker push realm-ai/runtime:latest
```

### 3. Deploy to Cloud

#### Option A: Docker Compose (Simple)

```yaml
# docker-compose.yml
version: '3.8'

services:
  realm-node-1:
    image: realm-ai/runtime:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - REALM_MAX_CUSTOMERS=16
      - REALM_API_KEY=${API_KEY}
    ports:
      - "8080:8080"
    volumes:
      - /data/models:/models
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  realm-node-2:
    image: realm-ai/runtime:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=1
      - REALM_MAX_CUSTOMERS=16
    ports:
      - "8081:8080"
    volumes:
      - /data/models:/models
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - realm-node-1
      - realm-node-2
```

```bash
# Deploy
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f realm-node-1
```

#### Option B: Kubernetes (Production)

```yaml
# realm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: realm-runtime
  namespace: realm-prod
spec:
  replicas: 5  # 5 GPU nodes
  selector:
    matchLabels:
      app: realm-runtime
  template:
    metadata:
      labels:
        app: realm-runtime
    spec:
      containers:
      - name: realm
        image: realm-ai/runtime:latest
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: REALM_MAX_CUSTOMERS
          value: "16"
        - name: REALM_BACKEND
          value: "cuda"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
        volumeMounts:
        - name: models
          mountPath: /models
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: realm-models-pvc
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-A100-SXM4-80GB

---
apiVersion: v1
kind: Service
metadata:
  name: realm-service
  namespace: realm-prod
spec:
  selector:
    app: realm-runtime
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

```bash
# Deploy
kubectl apply -f realm-deployment.yaml

# Check status
kubectl get pods -n realm-prod
kubectl logs -f deployment/realm-runtime -n realm-prod

# Scale up
kubectl scale deployment realm-runtime --replicas=10 -n realm-prod
```

### 4. Load Balancer Configuration

```nginx
# nginx.conf
upstream realm_backend {
    least_conn;  # Route to least busy node
    server realm-node-1:8080 max_fails=3 fail_timeout=30s;
    server realm-node-2:8080 max_fails=3 fail_timeout=30s;
    server realm-node-3:8080 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.realm.ai;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.realm.ai;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req zone=api_limit burst=20 nodelay;

    location / {
        proxy_pass http://realm_backend;
        proxy_http_version 1.1;

        # WebSocket support (for streaming)
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Customer-ID $http_x_customer_id;

        # Timeouts (long for streaming)
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    location /health {
        proxy_pass http://realm_backend/health;
        access_log off;
    }
}
```

---

## Model Management

### 1. Download Models

```bash
# On each node, download models
mkdir -p /data/models

# Example: Download from HuggingFace
wget https://huggingface.co/.../llama-2-7b-chat.Q4_K_M.gguf \
  -O /data/models/llama-2-7b-Q4_K_M.gguf

# Or use your own model conversion pipeline
./scripts/convert-model.sh \
  --input hf://meta-llama/Llama-2-7b-chat-hf \
  --output /data/models/llama-2-7b-Q4_K_M.gguf \
  --quantize Q4_K_M
```

### 2. Sync Models Across Nodes

```bash
# Option 1: Shared filesystem (NFS/EFS)
# All nodes mount same /models directory

# Option 2: S3 with caching
# Each node downloads from S3 on first access
# realm-runtime handles caching automatically

# Option 3: Rsync
rsync -avz /data/models/ node-2:/data/models/
rsync -avz /data/models/ node-3:/data/models/
```

---

## Monitoring

### 1. Prometheus Metrics

realm-runtime exposes metrics at `/metrics`:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'realm'
    static_configs:
    - targets:
      - 'realm-node-1:8080'
      - 'realm-node-2:8080'
      - 'realm-node-3:8080'
```

**Key metrics:**
- `realm_active_customers` - Number of active realms
- `realm_requests_total` - Total requests per customer
- `realm_request_duration_seconds` - Latency histogram
- `realm_gpu_utilization` - GPU usage %
- `realm_memory_usage_bytes` - Memory per realm
- `realm_tokens_generated_total` - Total tokens

### 2. Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Realm Production",
    "panels": [
      {
        "title": "Active Customers",
        "targets": [{"expr": "realm_active_customers"}]
      },
      {
        "title": "GPU Utilization",
        "targets": [{"expr": "realm_gpu_utilization"}]
      },
      {
        "title": "Request Latency (p95)",
        "targets": [{
          "expr": "histogram_quantile(0.95, realm_request_duration_seconds_bucket)"
        }]
      },
      {
        "title": "Tokens/Second",
        "targets": [{
          "expr": "rate(realm_tokens_generated_total[5m])"
        }]
      }
    ]
  }
}
```

### 3. Alerting

```yaml
# alerts.yml
groups:
- name: realm
  rules:
  - alert: HighGPUUtilization
    expr: realm_gpu_utilization > 90
    for: 5m
    annotations:
      summary: "GPU utilization above 90% for 5 minutes"

  - alert: RealmDown
    expr: up{job="realm"} == 0
    for: 1m
    annotations:
      summary: "Realm node is down"

  - alert: HighLatency
    expr: histogram_quantile(0.95, realm_request_duration_seconds_bucket) > 5
    for: 5m
    annotations:
      summary: "P95 latency above 5 seconds"
```

---

## Customer Onboarding

### 1. Create Customer Account

```bash
# Generate API key
realm-runtime create-customer \
  --name "Acme Corp" \
  --email "admin@acme.com" \
  --tier "pro"

# Output:
# Customer ID: cust_abc123
# API Key: sk_live_xyz789
```

### 2. Customer Usage

```javascript
// Customer's code
const { Realm } = require('@realm-ai/sdk');

const realm = new Realm({
  apiKey: 'sk_live_xyz789',  // Their API key
  endpoint: 'https://api.realm.ai'
});

await realm.loadModel('llama-2-7b-Q4_K_M');
const result = await realm.generate('Hello world');
```

### 3. Billing

```bash
# Export usage metrics
realm-runtime export-usage \
  --customer cust_abc123 \
  --start 2024-01-01 \
  --end 2024-01-31 \
  --format json

# Output:
# {
#   "customer_id": "cust_abc123",
#   "total_requests": 10000,
#   "total_tokens": 500000,
#   "total_compute_hours": 12.5,
#   "estimated_cost": "$12.50"
# }
```

---

## Self-Hosting Documentation (For Customers)

### Quick Start

```bash
# Download binary
curl -L https://realm.ai/install.sh | sh

# Or manually
wget https://github.com/realm-ai/realm/releases/download/v0.1.0/realm-runtime-linux-x64-cuda
chmod +x realm-runtime-linux-x64-cuda
sudo mv realm-runtime-linux-x64-cuda /usr/local/bin/realm-runtime

# Download WASM
wget https://github.com/realm-ai/realm/releases/download/v0.1.0/realm.wasm -O /usr/local/share/realm/realm.wasm

# Run
realm-runtime run --model ./models/llama-2-7b-Q4_K_M.gguf
```

### Docker

```bash
docker run -d \
  --gpus all \
  -p 8080:8080 \
  -v ./models:/models \
  realm-ai/runtime:latest \
  realm-runtime run --model /models/llama-2-7b-Q4_K_M.gguf
```

### Systemd Service

```ini
# /etc/systemd/system/realm.service
[Unit]
Description=Realm Runtime
After=network.target

[Service]
Type=simple
User=realm
ExecStart=/usr/local/bin/realm-runtime server \
  --port 8080 \
  --model-dir /var/lib/realm/models
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable realm
sudo systemctl start realm
sudo systemctl status realm
```

---

## Scaling Strategy

### Vertical Scaling (Per Node)

**Current:** 1 GPU, 16 customers, 125-200 tok/s per customer

**Upgrade options:**
- More powerful GPU (H100 vs A100): 2x throughput
- More customers per GPU (32 vs 16): Test memory limits
- Bigger models (70B vs 7B): Reduce customer count

### Horizontal Scaling (More Nodes)

```bash
# Start with 1 node
Capacity: 16 customers

# Add node 2
docker run ... realm-runtime (on GPU node 2)
# Update load balancer to include node 2
Capacity: 32 customers

# Add node 3, 4, 5...
Capacity: 80 customers (5 nodes × 16)
```

**Auto-scaling (Kubernetes):**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: realm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: realm-runtime
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 80
```

---

## Cost Optimization

### 1. Spot Instances

```bash
# Use spot instances for 70% cost savings
# Add node taints/tolerations in k8s
# realm-runtime handles graceful shutdown on eviction
```

### 2. Model Caching

```bash
# Pre-load hot models to avoid cold starts
realm-runtime preload \
  --models llama-2-7b-Q4_K_M,mistral-7b-Q4_K_M \
  --cache-size 20GB
```

### 3. Request Batching

```bash
# Enable automatic batching
realm-runtime server \
  --batch-size 8 \
  --batch-timeout 100ms
```

---

## Security Checklist

- [ ] Enable HTTPS only (TLS 1.3)
- [ ] API key authentication
- [ ] Rate limiting per customer
- [ ] Request size limits
- [ ] Timeout limits
- [ ] Memory limits per realm
- [ ] Network policies (k8s)
- [ ] Regular security updates
- [ ] Audit logging
- [ ] Secrets management (Vault/k8s secrets)

---

## Summary

**To deploy Realm SaaS:**

1. Build binary + WASM
2. Create Docker image
3. Deploy to 5-10 GPU nodes
4. Configure load balancer
5. Set up monitoring
6. Onboard customers with API keys

**Operating cost:** $3/hour per A100 × 10 nodes = $30/hour = $720/day

**Revenue at 150 customers:** 150 × $1/hour × 24 hours = $3,600/day

**Profit:** $3,600 - $720 = $2,880/day = **80% margin**

**You're ready to scale.**
