#!/bin/bash

# Documentation Integration Script
# Provides comprehensive documentation integration and automation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_ROOT}/config/docs-integration.yml"
GIT_REPO="${GIT_REPO:-$(pwd)}"
CONFLUENCE_SPACE="${CONFLUENCE_SPACE:-}"
DOCS_ROOT="${PROJECT_ROOT}/docs"
WEBSITE_URL="${WEBSITE_URL:-https://docs.example.com}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] DOCS:${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] DOCS WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] DOCS ERROR:${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] DOCS SUCCESS:${NC} $1"
}

# Initialize documentation workspace
init_docs_workspace() {
    local workspace="$1"
    
    log "Initializing documentation workspace: $workspace"
    
    mkdir -p "$workspace"
    mkdir -p "$workspace/api"
    mkdir -p "$workspace/guides"
    mkdir -p "$workspace/tutorials"
    mkdir -p "$workspace/architecture"
    mkdir -p "$workspace/changelogs"
    
    # Create workspace structure
    cat > "$workspace/README.md" << EOF
# Documentation Workspace

This workspace contains generated documentation for the project.

## Structure
- \`api/\` - API documentation
- \`guides/\` - User guides and tutorials
- \`architecture/\` - System architecture documentation
- \`changelogs/\` - Release changelogs

## Generated Files
- API specifications in OpenAPI/Swagger format
- User guides in Markdown format
- Architecture diagrams in Mermaid format
- Changelogs in Keep a Changelog format
EOF
    
    success "Documentation workspace initialized: $workspace"
}

# Generate API documentation
generate_api_docs() {
    local workspace="$1"
    local output_dir="$workspace/api"
    
    log "Generating API documentation"
    
    # This would integrate with your API documentation system
    # For demo, generate sample API docs
    
    cat > "$output_dir/feature-flags.md" << 'EOF
# Feature Flags API Documentation

## Overview
The Feature Flags API provides comprehensive functionality for managing feature flags, including evaluation, management, and monitoring.

## Base URL
\`\${WEBSITE_URL}/api/v1\`

## Authentication
All API requests must include an \`Authorization\` header with a valid API key.

\`\`\`bash
curl -H "Authorization: Bearer YOUR_API_KEY" \\
     -H "Content-Type: application/json" \\
     "\${WEBSITE_URL}/api/v1/feature-flags"
\`\`\`

## Endpoints

### Feature Flags
- \`GET /feature-flags\` - List all feature flags
- \`GET /feature-flags/{key}\` - Get specific feature flag
- \`POST /feature-flags\` - Create new feature flag
- \`PUT /feature-flags/{key}\` - Update feature flag
- \`DELETE /feature-flags/{key}\` - Delete feature flag

### Evaluation
- \`POST /feature-flags/evaluate\` - Evaluate feature flag
- \`POST /feature-flags/batch-evaluate\` - Evaluate multiple flags

### Management
- \`POST /feature-flags/{key}/kill-switch\` - Activate/deactivate kill switch
- \`GET /feature-flags/{key}/evaluations\` - Get evaluation history

### Monitoring
- \`GET /feature-flags/{key}/stats\` - Get flag statistics
- \`GET /feature-flags/metrics\` - Get system metrics

## Data Models

### FeatureFlag
\`\`\`json
{
  "id": "string",
  "key": "string",
  "name": "string",
  "description": "string",
  "type": "boolean|string|number|json",
  "enabled": "boolean",
  "value": "any",
  "rules": "array",
  "targeting": "object",
  "rollout": "object",
  "metadata": "object"
}
\`\`\`

### EvaluationContext
\`\`\`json
{
  "user_id": "string",
  "email": "string",
  "attributes": "object",
  "environment": "string"
  "version": "string"
}
\`\`\`

### EvaluationResult
\`\`\`json
{
  "enabled": "boolean",
  "value": "any",
  "reason": "string",
  "variation": "string"
}
\`\`\`

## Examples

### Get All Feature Flags
\`\`\`bash
curl -H "Authorization: Bearer YOUR_API_KEY" \\
     -H "Content-Type: application/json" \\
     "\${WEBSITE_URL}/api/v1/feature-flags"
\`\`\`

### Evaluate Feature Flag
\`\`\`bash
curl -H "Authorization: Bearer YOUR_API_KEY" \\
     -H "Content-Type: application/json" \\
     -d '{
       "flag_key": "new_dashboard",
       "context": {
         "user_id": "user123",
         "tier": "premium"
       }
     }' \\
     "\${WEBSITE_URL}/api/v1/feature-flags/evaluate"
\`\`\`

### Activate Kill Switch
\`\`\`bash
curl -H "Authorization: Bearer YOUR_API_KEY" \\
     -H "Content-Type: application/json" \\
     -d '{
       "action": "activate",
       "reason": "Performance issues detected"
     }' \\
     "\${WEBSITE_URL}/api/v1/feature-flags/new_dashboard/kill-switch"
\`\`\`
EOF
    
    success "API documentation generated: $output_dir/feature-flags.md"
}

# Generate user guides
generate_user_guides() {
    local workspace="$1"
    local output_dir="$workspace/guides"
    
    log "Generating user guides"
    
    cat > "$output_dir/feature-flag-management.md" << 'EOF
# Feature Flag Management Guide

## Overview
This guide covers how to use the Feature Flags system for managing feature flags in your applications.

## Getting Started

### 1. API Key Setup
Contact your administrator to get an API key for accessing the Feature Flags API.

### 2. Basic Usage
The simplest way to use feature flags is through direct API calls.

## Creating Your First Feature Flag

### Step 1: Create a Flag
\`\`\`bash
curl -X POST \\
     -H "Authorization: Bearer YOUR_API_KEY" \\
     -H "Content-Type: application/json" \\
     -d '{
       "key": "my_first_flag",
       "name": "My First Flag",
       "description": "A simple boolean flag for testing",
       "type": "boolean",
       "enabled": false
     }' \\
     "\${WEBSITE_URL}/api/v1/feature-flags"
\`\`\`

### Step 2: Evaluate the Flag
\`\`\`bash
curl -X POST \\
     -H "Authorization: Bearer YOUR_API_KEY" \\
     -H "Content-Type: application/json" \\
     -d '{
       "flag_key": "my_first_flag",
       "context": {
         "user_id": "test_user"
       }
     }' \\
     "\${WEBSITE_URL}/api/v1/feature-flags/evaluate"
\`\`\`

### Step 3: Enable the Flag
\`\`\`bash
curl -X PUT \\
     -H "Authorization: Bearer YOUR_API_KEY" \\
     -H "Content-Type: application/json" \\
     -d '{
       "enabled": true
     }' \\
     "\${WEBSITE_URL}/api/v1/feature-flags/my_first_flag"
\`\`\`

## Advanced Usage

### User Targeting
You can target specific users or user segments with your feature flags.

### Rollout Strategies
Implement gradual rollouts to safely test new features.

### Kill Switch
Use the kill switch functionality for emergency situations.

## Best Practices

1. **Use Descriptive Names**: Make your flag names and descriptions clear and meaningful.
2. **Default to False**: Always start new flags in the disabled state.
3. **Clean Up**: Remove unused flags to keep your system organized.
4. **Monitor Usage**: Track how your flags are being used in production.

## Troubleshooting

### Common Issues
- API key authentication failures
- Flag evaluation timeouts
- Kill switch not working

### Support
For support and questions, contact the development team or check the documentation.
EOF
    
    success "User guide generated: $output_dir/feature-flag-management.md"
}

# Generate architecture documentation
generate_architecture_docs() {
    local workspace="$1"
    local output_dir="$workspace/architecture"
    
    log "Generating architecture documentation"
    
    cat > "$output_dir/system-overview.md" << 'EOF
# System Architecture Overview

## Components

### Feature Flag Service
- **Purpose**: Centralized feature flag management
- **Technology**: Node.js with Express.js
- **Database**: PostgreSQL for persistence
- **Cache**: Redis for fast lookups
- **API**: RESTful API with JSON responses

### Backport Bot
- **Purpose**: Automated backport with conflict detection
- **Intelligence**: Smart conflict analysis and resolution
- **Integration**: GitHub API for pull requests

### Release Pipeline
- **Purpose**: Automated release with approval workflow
- **Stages**: Validation → Approval → Build → Deploy → Verify
- **Integration**: CI/CD platforms (GitHub Actions, GitLab CI)

### Rollback System
- **Purpose**: Emergency release rollback
- **Triggers**: Manual, automatic, and emergency
- **Safety**: Pre-rollback validation and verification

## Data Flow

\`\`\`mermaid
graph LR
    A[Developer] --> B[Git Push]
    B --> C{Pre-commit Hook}
    C --> D[Guardrails Service]
    D --> E[Release Pipeline]
    E --> F[Production]
    F --> G[Monitoring]
    
    G[Backport Bot] --> H[GitHub API]
    H --> I[Pull Requests]
    I --> J[Developer]
    
    K[Rollback System] --> F
    K --> L[Emergency Release]
\`\`\`

## Security Considerations

- **Authentication**: API key-based access control
- **Authorization**: Role-based permissions for different operations
- **Audit Trail**: Complete logging of all actions
- **Encryption**: Data encryption in transit and at rest

## Scalability

- **Horizontal Scaling**: Multiple service instances behind load balancer
- **Database Sharding**: Partitioning for large datasets
- **Caching Strategy**: Multi-level caching for performance
- **Rate Limiting**: API throttling to prevent abuse

## Monitoring

- **Health Checks**: Service health and dependency monitoring
- **Performance Metrics**: Response times, throughput, error rates
- **Alerting**: Multi-channel notifications for critical issues
- **Logging**: Structured logging for debugging and analysis
EOF
    
    success "Architecture documentation generated: $output_dir/system-overview.md"
}

# Generate changelogs
generate_changelogs() {
    local workspace="$1"
    local output_dir="$workspace/changelogs"
    
    log "Generating changelogs"
    
    # Get recent releases
    local releases
    releases=$(git tag --sort=-v:refname | head -10)
    
    cat > "$output_dir/CHANGELOG.md" << 'EOF'
# Changelog

All notable changes to this project will be documented in this file following the [Keep a Changelog](https://keepachangelog.com/) format.

$(for tag in $releases; do
    local date
    date=$(git log -1 --format="%ai" "$tag" --date=short)
    local message
    message=$(git tag -l --format="%(contents)" "$tag" 2>/dev/null || echo "Release $tag")
    
    echo "## [$tag] - $date"
    echo "$message"
    echo ""
done)
EOF
    
    success "Changelog generated: $output_dir/CHANGELOG.md"
}

# Sync documentation to website
sync_docs_to_website() {
    local workspace="$1"
    
    log "Syncing documentation to website"
    
    # This would integrate with your documentation deployment system
    # For demo, simulate the sync process
    
    local docs_to_sync=(
        "api/feature-flags.md"
        "guides/feature-flag-management.md"
        "architecture/system-overview.md"
        "changelogs/CHANGELOG.md"
    )
    
    for doc in "${docs_to_sync[@]}"; do
        if [ -f "$workspace/$doc" ]; then
            log "Syncing: $doc"
            # Simulate upload to website
            echo "Uploaded: $doc to ${WEBSITE_URL}/docs/$doc"
        else
            warn "Document not found: $doc"
        fi
    done
    
    success "Documentation sync completed"
}

# Generate documentation index
generate_docs_index() {
    local workspace="$1"
    
    log "Generating documentation index"
    
    cat > "$workspace/index.md" << 'EOF
# Documentation Index

## API Documentation
- [Feature Flags API](api/feature-flags.md)

## User Guides
- [Feature Flag Management](guides/feature-flag-management.md)

## Architecture
- [System Overview](architecture/system-overview.md)

## Changelogs
- [CHANGELOG](changelogs/CHANGELOG.md)

## Quick Links
- [API Reference](api/) - Complete API documentation
- [Getting Started](guides/) - User guides and tutorials
- [System Architecture](architecture/) - Technical documentation
EOF
    
    success "Documentation index generated: $workspace/index.md"
}

# Show usage
show_usage() {
    cat << EOF
Documentation Integration Script

USAGE:
    $0 <command> [options]

DOCUMENTATION COMMANDS:
    init <workspace>                           Initialize documentation workspace
    generate-api <workspace>                     Generate API documentation
    generate-guides <workspace>                   Generate user guides
    generate-architecture <workspace>                Generate architecture docs
    generate-changelogs <workspace>                Generate changelogs
    generate-index <workspace>                     Generate documentation index
    sync <workspace>                              Sync documentation to website

MONITORING COMMANDS:
    validate <workspace>                        Validate generated documentation
    build <workspace>                           Build documentation site
    deploy <workspace>                            Deploy documentation to staging
    promote <workspace>                          Deploy documentation to production

EXAMPLES:
    $0 init ./docs-workspace
    $0 generate-api ./docs-workspace
    $0 generate-guides ./docs-workspace
    $0 sync ./docs-workspace
    $0 validate ./docs-workspace
    $0 build ./docs-workspace
    $0 deploy ./docs-workspace

CONFIGURATION:
    Workspace: ${DOCS_ROOT:-./docs}
    Website URL: ${WEBSITE_URL:-https://docs.example.com}
    Confluence Space: ${CONFLUENCE_SPACE:-}
    Git Repository: $GIT_REPO}

EOF
}

# Validate documentation
validate_docs() {
    local workspace="$1"
    
    log "Validating generated documentation"
    
    local validation_errors=0
    
    # Check if required files exist
    local required_files=(
        "$workspace/index.md"
        "$workspace/api/feature-flags.md"
        "$workspace/guides/feature-flag-management.md"
        "$workspace/architecture/system-overview.md"
        "$workspace/changelogs/CHANGELOG.md"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            error "Missing required file: $file"
            ((validation_errors++))
        fi
    done
    
    if [ $validation_errors -eq 0 ]; then
        success "Documentation validation passed"
    else
        error "Documentation validation failed with $validation_errors errors"
    fi
}

# Build documentation site
build_docs_site() {
    local workspace="$1"
    
    log "Building documentation site"
    
    # This would integrate with your static site generator
    # For demo, simulate the build process
    
    cat > "$workspace/build.log" << EOF
Building documentation site...
- Processing API documentation
- Processing user guides
- Processing architecture documentation
- Generating index
- Site built successfully
EOF
    
    success "Documentation site built: $workspace"
}

# Deploy documentation
deploy_docs() {
    local workspace="$1"
    local environment="${2:-staging}"
    
    log "Deploying documentation to $environment"
    
    # This would integrate with your deployment system
    # For demo, simulate the deployment process
    
    local deploy_log="$workspace/deploy-$environment.log"
    
    cat > "$deploy_log" << EOF
Deploying documentation to $environment...
- Validating documentation
- Uploading files to $environment
- Deployment completed successfully
- Site available at: ${WEBSITE_URL}/docs/
EOF
    
    success "Documentation deployed to $environment: $workspace"
}

# Main execution
main() {
    case "${1:-}" in
        "init")
            if [ $# -lt 2 ]; then
                error "Usage: $0 init <workspace>"
                exit 1
            fi
            init_docs_workspace "$2"
            ;;
        "generate-api")
            if [ $# -lt 2 ]; then
                error "Usage: $0 generate-api <workspace>"
                exit 1
            fi
            generate_api_docs "$2"
            ;;
        "generate-guides")
            if [ $# -lt 2 ]; then
                error "Usage: $0 generate-guides <workspace>"
                exit 1
            fi
            generate_user_guides "$2"
            ;;
        "generate-architecture")
            if [ $# -lt 2 ]; then
                error "Usage: $0 generate-architecture <workspace>"
                exit 1
            fi
            generate_architecture_docs "$2"
            ;;
        "generate-changelogs")
            if [ $# -lt 2 ]; then
                error "Usage: $0 generate-changelogs <workspace>"
                exit 1
            fi
            generate_changelogs "$2"
            ;;
        "generate-index")
            if [ $# -lt 2 ]; then
                error "Usage: $0 generate-index <workspace>"
                exit 1
            fi
            generate_docs_index "$2"
            ;;
        "sync")
            if [ $# -lt 2 ]; then
                error "Usage: $0 sync <workspace>"
                exit 1
            fi
            sync_docs_to_website "$2"
            ;;
        "validate")
            if [ $# -lt 2 ]; then
                error "Usage: $0 validate <workspace>"
                exit 1
            fi
            validate_docs "$2"
            ;;
        "build")
            if [ $# -lt 2 ]; then
                error "Usage: $0 build <workspace>"
                exit 1
            fi
            build_docs_site "$2"
            ;;
        "deploy")
            if [ $# -lt 2 ]; then
                error "Usage: $0 deploy <workspace> [environment]"
                exit 1
            fi
            deploy_docs "$2" "${3:-staging}"
            ;;
        "promote")
            if [ $# -lt 2 ]; then
                error "Usage: $0 promote <workspace>"
                exit 1
            fi
            deploy_docs "$2" "production"
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"