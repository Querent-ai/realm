# Realm EC2 Instance Module
#
# Deploys a single EC2 instance running Realm server with optional GPU support

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Variables
variable "name" {
  description = "Name prefix for resources"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.medium"
}

variable "gpu_instance_type" {
  description = "GPU instance type (e.g., g4dn.xlarge, p3.2xlarge)"
  type        = string
  default     = null
}

variable "use_gpu" {
  description = "Whether to use GPU instance"
  type        = bool
  default     = false
}

variable "ami_id" {
  description = "AMI ID (if not provided, will use latest Ubuntu 22.04)"
  type        = string
  default     = null
}

variable "key_name" {
  description = "AWS key pair name for SSH access"
  type        = string
  default     = null
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID"
  type        = string
}

variable "security_group_ids" {
  description = "Additional security group IDs"
  type        = list(string)
  default     = []
}

variable "model_storage_type" {
  description = "Model storage type: s3, ebs, or local"
  type        = string
  default     = "s3"
}

variable "model_storage_bucket" {
  description = "S3 bucket for models (if model_storage_type is s3)"
  type        = string
  default     = null
}

variable "model_storage_path" {
  description = "Path to models (S3 prefix or local path)"
  type        = string
  default     = "models"
}

variable "wasm_path" {
  description = "Path to realm_wasm.wasm file (S3 or local)"
  type        = string
  default     = null
}

variable "server_port" {
  description = "Realm server port"
  type        = number
  default     = 8080
}

variable "max_tenants" {
  description = "Maximum number of tenants"
  type        = number
  default     = 16
}

variable "tags" {
  description = "Additional tags"
  type        = map(string)
  default     = {}
}

# Data sources
data "aws_ami" "ubuntu" {
  count       = var.ami_id == null ? 1 : 0
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hubuntu-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Security group
resource "aws_security_group" "realm" {
  name        = "${var.name}-realm-sg"
  description = "Security group for Realm server"
  vpc_id      = var.vpc_id

  ingress {
    description = "Realm WebSocket server"
    from_port   = var.server_port
    to_port     = var.server_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.name}-realm-sg"
    }
  )
}

# IAM role for EC2 instance
resource "aws_iam_role" "realm" {
  name = "${var.name}-realm-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

# IAM policy for S3 model access
resource "aws_iam_role_policy" "realm_s3" {
  count = var.model_storage_type == "s3" && var.model_storage_bucket != null ? 1 : 0
  name  = "${var.name}-realm-s3-policy"
  role  = aws_iam_role.realm.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.model_storage_bucket}",
          "arn:aws:s3:::${var.model_storage_bucket}/*"
        ]
      }
    ]
  })
}

# IAM instance profile
resource "aws_iam_instance_profile" "realm" {
  name = "${var.name}-realm-profile"
  role = aws_iam_role.realm.name
}

# User data script
locals {
  user_data = <<-EOF
    #!/bin/bash
    set -e
    
    # Update system
    apt-get update
    apt-get install -y curl wget unzip
    
    # Install Rust
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
    
    # Install CUDA (if GPU instance)
    %{ if var.use_gpu ~}
    # Install NVIDIA drivers and CUDA
    apt-get install -y nvidia-driver-535 nvidia-cuda-toolkit
    %{ endif ~}
    
    # Create directories
    mkdir -p /opt/realm/{bin,wasm,models}
    
    # Download Realm binary (from releases or build)
    # TODO: Replace with actual download URL
    # wget -O /opt/realm/bin/realm https://github.com/yourusername/realm/releases/download/v0.1.0/realm-linux-amd64
    # chmod +x /opt/realm/bin/realm
    
    # Download WASM module
    %{ if var.wasm_path != null ~}
    if [[ "${var.wasm_path}" == s3://* ]]; then
      aws s3 cp "${var.wasm_path}" /opt/realm/wasm/realm_wasm.wasm
    else
      wget -O /opt/realm/wasm/realm_wasm.wasm "${var.wasm_path}"
    fi
    %{ endif ~}
    
    # Download models (if S3)
    %{ if var.model_storage_type == "s3" && var.model_storage_bucket != null ~}
    aws s3 sync "s3://${var.model_storage_bucket}/${var.model_storage_path}" /opt/realm/models/
    %{ endif ~}
    
    # Create systemd service
    cat > /etc/systemd/system/realm.service <<'SERVICE'
    [Unit]
    Description=Realm LLM Inference Server
    After=network.target
    
    [Service]
    Type=simple
    User=root
    WorkingDirectory=/opt/realm
    ExecStart=/opt/realm/bin/realm serve \\
      --host 0.0.0.0 \\
      --port ${var.server_port} \\
      --wasm /opt/realm/wasm/realm_wasm.wasm \\
      --model-dir /opt/realm/models \\
      --max-tenants ${var.max_tenants}
    Restart=always
    RestartSec=10
    
    [Install]
    WantedBy=multi-user.target
    SERVICE
    
    systemctl daemon-reload
    systemctl enable realm
    systemctl start realm
  EOF
}

# EC2 instance
resource "aws_instance" "realm" {
  ami                    = var.ami_id != null ? var.ami_id : data.aws_ami.ubuntu[0].id
  instance_type          = var.use_gpu && var.gpu_instance_type != null ? var.gpu_instance_type : var.instance_type
  key_name               = var.key_name
  subnet_id              = var.subnet_id
  vpc_security_group_ids = concat([aws_security_group.realm.id], var.security_group_ids)
  iam_instance_profile   = aws_iam_instance_profile.realm.name

  user_data = local.user_data

  root_block_device {
    volume_type = "gp3"
    volume_size = 100
    encrypted   = true
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.name}-realm"
    }
  )
}

# Outputs
output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.realm.id
}

output "instance_private_ip" {
  description = "EC2 instance private IP"
  value       = aws_instance.realm.private_ip
}

output "instance_public_ip" {
  description = "EC2 instance public IP"
  value       = aws_instance.realm.public_ip
}

output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.realm.id
}

output "server_url" {
  description = "Realm server WebSocket URL"
  value       = "ws://${aws_instance.realm.public_ip}:${var.server_port}"
}

