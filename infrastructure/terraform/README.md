# Realm Terraform Modules

**Status**: 🚧 Planned

---

## Vision

Infrastructure as Code (IaC) modules for deploying Realm on:

- **AWS** (EC2, ECS, EKS)
- **GCP** (GCE, GKE)
- **Azure** (VM, AKS)
- **Multi-cloud** configurations

---

## Planned Modules

### AWS Modules

#### `modules/aws/ec2`
- Single EC2 instance deployment
- GPU instance support (g4dn, p3, etc.)
- Auto-scaling configuration
- Security groups and networking

#### `modules/aws/ecs`
- ECS Fargate deployment
- ECS EC2 deployment with GPU support
- Service discovery
- Load balancing

#### `modules/aws/eks`
- EKS cluster setup
- GPU node groups
- Auto-scaling
- Ingress configuration

### GCP Modules

#### `modules/gcp/gce`
- Compute Engine instance
- GPU instance support
- Auto-scaling

#### `modules/gcp/gke`
- GKE cluster setup
- GPU node pools
- Workload identity

### Azure Modules

#### `modules/azure/vm`
- Virtual Machine deployment
- GPU VM support
- Availability sets

#### `modules/azure/aks`
- AKS cluster setup
- GPU node pools
- Service mesh integration

---

## Example Usage

### AWS EKS
```hcl
module "realm_eks" {
  source = "./modules/aws/eks"
  
  cluster_name = "realm-cluster"
  node_groups = {
    gpu = {
      instance_type = "g4dn.xlarge"
      min_size      = 1
      max_size      = 10
    }
  }
  
  model_storage = "s3://realm-models"
}
```

### GCP GKE
```hcl
module "realm_gke" {
  source = "./modules/gcp/gke"
  
  cluster_name = "realm-cluster"
  gpu_node_pools = {
    nvidia = {
      machine_type = "n1-standard-4"
      gpu_type     = "nvidia-tesla-t4"
      min_nodes    = 1
      max_nodes    = 10
    }
  }
}
```

---

## Implementation Plan

### Phase 1: AWS EC2
- [ ] Basic EC2 module
- [ ] GPU instance support
- [ ] Security groups
- [ ] User data scripts

### Phase 2: AWS EKS
- [ ] EKS cluster module
- [ ] GPU node groups
- [ ] Ingress configuration

### Phase 3: GCP & Azure
- [ ] GCP modules
- [ ] Azure modules

### Phase 4: Multi-cloud
- [ ] Cross-cloud configurations
- [ ] Federation support

---

**Status**: Ready to start implementation when needed.

