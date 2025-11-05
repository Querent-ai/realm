# Realm Docker Compose Setup

**Status**: 🚧 Planned

---

## Vision

Multi-service local development and production deployment setup with:

- **Realm Server** - WebSocket inference server
- **Prometheus** - Metrics collection
- **Grafana** - Visualization and dashboards
- **Optional Services** - Redis, PostgreSQL, etc.

---

## Quick Start

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f realm-server

# Stop services
docker-compose down
```

---

## Services

### Realm Server
- WebSocket server on port 8080
- Metrics endpoint on port 9090
- Model storage via volume mount

### Prometheus
- Metrics scraping from Realm server
- Persistent storage for metrics
- Port 9091

### Grafana
- Pre-configured dashboards
- Prometheus data source
- Port 3000 (admin/admin)

---

## Configuration

### Environment Variables
- `RUST_LOG` - Logging level
- `REALM_MODEL_DIR` - Model directory path
- `CUDA_VISIBLE_DEVICES` - GPU selection (if applicable)

### Volumes
- `./models` - Model files directory
- `./wasm` - WASM modules directory
- `prometheus-data` - Prometheus metrics storage
- `grafana-data` - Grafana configuration storage

---

## Production Deployment

For production, use:
- Kubernetes (via Helm charts)
- Docker Swarm
- ECS/EKS (via Terraform)

---

**Status**: Ready to start implementation when needed.

