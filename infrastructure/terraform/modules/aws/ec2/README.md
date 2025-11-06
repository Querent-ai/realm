# Realm EC2 Module

Terraform module for deploying Realm server on AWS EC2.

## Features

- Single EC2 instance deployment
- GPU instance support (g4dn, p3, p4d, etc.)
- Automatic security group creation
- S3 model storage integration
- Systemd service setup
- Optional CUDA installation for GPU instances

## Usage

```hcl
module "realm_ec2" {
  source = "./modules/aws/ec2"
  
  name         = "realm-production"
  instance_type = "t3.medium"
  vpc_id       = "vpc-12345678"
  subnet_id    = "subnet-12345678"
  
  # Optional: Use GPU instance
  use_gpu           = true
  gpu_instance_type = "g4dn.xlarge"
  
  # Model storage
  model_storage_type   = "s3"
  model_storage_bucket = "realm-models"
  model_storage_path   = "models"
  
  # Server configuration
  server_port = 8080
  max_tenants = 16
  
  tags = {
    Environment = "production"
    Project     = "realm"
  }
}
```

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|----------|
| name | Name prefix for resources | string | - | yes |
| instance_type | EC2 instance type | string | "t3.medium" | no |
| gpu_instance_type | GPU instance type | string | null | no |
| use_gpu | Whether to use GPU instance | bool | false | no |
| vpc_id | VPC ID | string | - | yes |
| subnet_id | Subnet ID | string | - | yes |
| model_storage_type | Model storage type: s3, ebs, or local | string | "s3" | no |
| model_storage_bucket | S3 bucket for models | string | null | no |
| server_port | Realm server port | number | 8080 | no |
| max_tenants | Maximum number of tenants | number | 16 | no |

## Outputs

| Name | Description |
|------|-------------|
| instance_id | EC2 instance ID |
| instance_private_ip | EC2 instance private IP |
| instance_public_ip | EC2 instance public IP |
| server_url | Realm server WebSocket URL |

## GPU Instance Types

Recommended GPU instance types:
- `g4dn.xlarge` - 1x T4 GPU, 4 vCPU, 16 GB RAM
- `g4dn.2xlarge` - 1x T4 GPU, 8 vCPU, 32 GB RAM
- `p3.2xlarge` - 1x V100 GPU, 8 vCPU, 61 GB RAM
- `p4d.24xlarge` - 8x A100 GPU, 96 vCPU, 1152 GB RAM

