# Realm EKS Module

Terraform module for deploying Realm server on AWS EKS with GPU support.

## Features

- EKS cluster with managed node groups
- GPU node group support (g4dn, p3, p4d, etc.)
- Automatic IAM roles and policies
- S3 model storage integration
- CloudWatch logging
- Kubernetes and Helm provider setup

## Usage

```hcl
module "realm_eks" {
  source = "./modules/aws/eks"
  
  cluster_name = "realm-cluster"
  region       = "us-east-1"
  vpc_id       = "vpc-12345678"
  subnet_ids   = ["subnet-12345678", "subnet-87654321"]
  
  # Node groups
  node_groups = {
    cpu = {
      instance_type = "t3.medium"
      min_size      = 1
      max_size      = 5
      desired_size  = 2
      gpu_enabled   = false
    }
    gpu = {
      instance_type = "g4dn.xlarge"
      min_size      = 1
      max_size      = 10
      desired_size  = 2
      gpu_enabled   = true
    }
  }
  
  model_storage_bucket = "realm-models"
  
  tags = {
    Environment = "production"
    Project     = "realm"
  }
}
```

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|----------|
| cluster_name | EKS cluster name | string | - | yes |
| region | AWS region | string | "us-east-1" | no |
| vpc_id | VPC ID | string | - | yes |
| subnet_ids | Subnet IDs (at least 2) | list(string) | - | yes |
| node_groups | Node group configurations | map(object) | {} | no |
| model_storage_bucket | S3 bucket for models | string | null | no |

## Outputs

| Name | Description |
|------|-------------|
| cluster_id | EKS cluster ID |
| cluster_endpoint | EKS cluster endpoint |
| cluster_name | EKS cluster name |
| kubeconfig | Kubeconfig command |

## GPU Node Groups

GPU node groups automatically get:
- Taint: `nvidia.com/gpu=true:NO_SCHEDULE`
- Label: `gpu=true`

Use node selectors in Helm charts to schedule Realm pods on GPU nodes.

