# Ansible Deployment

## Prerequisites
- Remote hosts with Docker installed (or allow Ansible to install)
- SSH access with private key in GitHub Actions secrets
- Replace inventory hostnames and users

## Blue-Green Strategy
- Active color stored at `/etc/codeinsight/active_color`
- Next color is started with `COMPOSE_PROJECT_NAME=codeinsight_<color>`
- Health check calls `http://localhost:8000/health`
- If healthy, switch active color and stop old stack; otherwise rollback

## Variables passed from CI
- `registry`: container registry (e.g., `ghcr.io`)
- `backend_image`: repository path for backend image
- `frontend_image`: repository path for frontend image
- `tag`: image tag (commit SHA or version)

## Running locally
```bash
ansible-playbook -i ansible/inventory/dev ansible/site.yml \
  --extra-vars "registry=ghcr.io backend_image=OWNER/REPO/backend frontend_image=OWNER/REPO/frontend tag=latest"
```