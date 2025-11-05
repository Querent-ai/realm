# Realm Helm Charts

**Status**: 🚧 Planned

---

## Vision

Production-ready Kubernetes deployment templates for Realm with:

- **Auto-scaling** configurations
- **Multi-tenant** resource management
- **GPU node** scheduling
- **High availability** setup

---

## Planned Charts

### `realm-server`
- Realm WebSocket server deployment
- Multi-replica support
- Horizontal pod autoscaling
- GPU node affinity

### `realm-gpu`
- GPU node pool configuration
- GPU resource allocation
- Node selector for GPU instances

### `realm-metrics`
- Prometheus metrics exporter
- Grafana dashboard integration
- Alerting rules

### `realm-full-stack`
- Complete Realm deployment
- Includes server, metrics, ingress
- Production-ready configuration

---

## Example Usage

### Basic Deployment
```bash
helm install realm ./helm/realm-server \
  --set modelStorage.type=s3 \
  --set modelStorage.bucket=realm-models \
  --set replicas=3 \
  --set gpu.enabled=true
```

### Production Deployment
```bash
helm install realm ./helm/realm-full-stack \
  --set replicaCount=5 \
  --set autoscaling.enabled=true \
  --set autoscaling.minReplicas=3 \
  --set autoscaling.maxReplicas=10 \
  --set gpu.nodeSelector.accelerator=nvidia-tesla-t4 \
  --set ingress.enabled=true \
  --set metrics.enabled=true
```

---

## Chart Structure

```
helm/realm/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── hpa.yaml
│   └── configmap.yaml
└── README.md
```

---

## Key Features

### Auto-scaling
- Horizontal Pod Autoscaler (HPA)
- Custom metrics support
- GPU utilization-based scaling

### GPU Support
- Node selector for GPU instances
- GPU resource requests/limits
- Multi-GPU support

### High Availability
- Multi-replica deployment
- Pod disruption budgets
- Health checks and readiness probes

### Security
- RBAC configuration
- Network policies
- Secret management

---

## Implementation Plan

### Phase 1: Basic Chart
- [ ] Chart.yaml and values.yaml
- [ ] Deployment template
- [ ] Service template
- [ ] Basic configuration

### Phase 2: Production Features
- [ ] HPA configuration
- [ ] Ingress setup
- [ ] GPU node affinity
- [ ] Resource limits

### Phase 3: Advanced Features
- [ ] Multi-tenant configuration
- [ ] Metrics integration
- [ ] Monitoring setup
- [ ] Alerting rules

---

**Status**: Ready to start implementation when needed.

