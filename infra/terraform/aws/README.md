# Terraform for AWS (Skeleton)

This directory provides a minimal skeleton to provision:
- ECR repositories for backend and frontend images
- An EC2 instance with Docker runtime
- A security group allowing SSH(22) and HTTP(80)

## Usage
1. Ensure you have AWS credentials configured (via environment or profile).
2. Prepare variables (VPC ID, Subnet ID, AMI ID).
3. Initialize and apply:
```bash
terraform init
terraform plan -var "vpc_id=vpc-xxxx" -var "subnet_id=subnet-xxxx" -var "ami_id=ami-xxxx" -var "region=us-east-1"
terraform apply -auto-approve -var "vpc_id=vpc-xxxx" -var "subnet_id=subnet-xxxx" -var "ami_id=ami-xxxx" -var "region=us-east-1"
```

## Notes
- Replace the AMI with your preferred Linux distribution.
- EC2 user_data installs Docker; consider Ansible for further configuration.
- For production, consider using ECS/EKS with Terraform modules and a load balancer.