# Realm Helm Chart

Production-ready Kubernetes deployment for Realm multi-tenant LLM inference server.

## Features

- ✅ GPU support (NVIDIA, with node selectors and tolerations)
- ✅ Auto-scaling (HPA with CPU/memory metrics)
- ✅ Model storage (S3, PVC, or local)
- ✅ Health checks (liveness and readiness probes)
- ✅ Service monitor (Prometheus integration)
- ✅ Pod disruption budget (high availability)
- ✅ Ingress support (with TLS)
- ✅ Configurable resources and limits

## Prerequisites

- Kubernetes 1.24+
- Helm 3.0+
- (Optional) NVIDIA GPU operator for GPU support
- (Optional) Prometheus Operator for ServiceMonitor

## Installation

### Basic Installation

```bash
helm install realm ./infrastructure/helm/realm \
  --set modelStorage.s3.bucket=realm-models \
  --set modelStorage.s3.region=us-east-1
```

### With GPU Support

```bash
helm install realm ./infrastructure/helm/realm \
  --set gpu.enabled=true \
  --set gpu.count=1 \
  --set gpu.nodeSelector.accelerator=nvidia-tesla-t4 \
  --set modelStorage.s3.bucket=realm-models
```

### Production Deployment

```bash
helm install realm ./infrastructure/helm/realm \
  --set replicaCount=3 \
  --set autoscaling.enabled=true \
  --set autoscaling.minReplicas=3 \
  --set autoscaling.maxReplicas=10 \
  --set gpu.enabled=true \
  --set gpu.count=1 \
  --set resources.limits.cpu=8 \
  --set resources.limits.memory=16Gi \
  --set resources.requests.cpu=4 \
  --set resources.requests.memory=8Gi \
  --set modelStorage.s3.bucket=realm-models \
  --set modelStorage.s3.region=us-east-1 \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=realm.example.com \
  --set serviceMonitor.enabled=true \
  --set podDisruptionBudget.enabled=true
```

## Configuration

### Model Storage

#### S3 Storage
```yaml
modelStorage:
  type: s3
  s3:
    bucket: "realm-models"
    region: "us-east-1"
    prefix: "models"
```

For AWS EKS with IRSA (IAM Roles for Service Accounts):
```yaml
serviceAccount:
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::ACCOUNT:role/realm-s3-role
```

#### PVC Storage
```yaml
modelStorage:
  type: pvc
  pvc:
    claimName: "realm-models-pvc"
    mountPath: "/models"
```

#### Local Storage
```yaml
modelStorage:
  type: local
  local:
    path: "/models"
```

### GPU Configuration

```yaml
gpu:
  enabled: true
  count: 1
  nodeSelector:
    accelerator: nvidia-tesla-t4
  # or:
  # nodeSelector:
  #   nvidia.com/gpu: "true"
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
```

### Auto-scaling

```yaml
autoscaling:
  enabled: true
  minReplicas: 1
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
  targetMemoryUtilizationPercentage: 80
```

### Ingress

```yaml
ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: realm.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: realm-tls
      hosts:
        - realm.example.com
```

## Values Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `1` |
| `image.repository` | Container image repository | `realm-ai/runtime` |
| `image.tag` | Container image tag | `0.1.0` |
| `gpu.enabled` | Enable GPU support | `false` |
| `gpu.count` | Number of GPUs per pod | `1` |
| `modelStorage.type` | Storage type: s3, pvc, or local | `s3` |
| `server.port` | Server port | `8080` |
| `server.maxTenants` | Maximum tenants | `16` |
| `autoscaling.enabled` | Enable HPA | `false` |
| `ingress.enabled` | Enable ingress | `false` |
| `serviceMonitor.enabled` | Enable ServiceMonitor | `false` |

See `values.yaml` for all available options.

## Examples

### Development
```bash
helm install realm-dev ./infrastructure/helm/realm \
  --set replicaCount=1 \
  --set resources.requests.cpu=1 \
  --set resources.requests.memory=2Gi \
  --set modelStorage.type=local
```

### Production with GPU
```bash
helm install realm-prod ./infrastructure/helm/realm \
  --set replicaCount=3 \
  --set gpu.enabled=true \
  --set gpu.count=1 \
  --set autoscaling.enabled=true \
  --set modelStorage.s3.bucket=realm-models-prod \
  --set ingress.enabled=true \
  --set serviceMonitor.enabled=true
```

## Uninstallation

```bash
helm uninstall realm
```

## Troubleshooting

### Pods not starting
- Check resource requests/limits
- Verify model storage access (S3 permissions, PVC exists)
- Check GPU node availability (if GPU enabled)

### GPU not available
- Verify NVIDIA GPU operator is installed
- Check node labels: `kubectl get nodes --show-labels`
- Verify tolerations match node taints

### Models not loading
- Check S3 bucket permissions (if using S3)
- Verify PVC is mounted correctly (if using PVC)
- Check logs: `kubectl logs <pod-name>`

## Support

For issues and questions, please open an issue on GitHub.
