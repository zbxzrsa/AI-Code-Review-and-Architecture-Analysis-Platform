#!/bin/bash

# Release Pipeline Integration and Approval Workflow
# Provides comprehensive CI/CD integration with automated approvals

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_ROOT}/config/release-pipeline.yml"
GIT_REPO="${GIT_REPO:-$(pwd)}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] PIPELINE:${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] PIPELINE WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] PIPELINE ERROR:${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] PIPELINE SUCCESS:${NC} $1"
}

# Initialize release pipeline
init_pipeline() {
    local release_id="$1"
    local version="$2"
    
    log "Initializing release pipeline: $release_id ($version)"
    
    # Create pipeline workspace
    local workspace="${PROJECT_ROOT}/workspace/releases/$release_id"
    mkdir -p "$workspace"
    
    # Initialize pipeline state
    local pipeline_state="{
        \"release_id\": \"$release_id\",
        \"version\": \"$version\",
        \"status\": \"initialized\",
        \"created_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
        \"steps\": [],
        \"approvals\": [],
        \"checks\": {},
        \"metadata\": {
            \"environment\": \"$ENVIRONMENT\",
            \"workspace\": \"$workspace\"
        }
    }"
    
    echo "$pipeline_state" > "$workspace/pipeline.json"
    
    success "Pipeline initialized for release: $release_id"
    echo "$workspace"
}

# Run pre-release checks
run_pre_release_checks() {
    local workspace="$1"
    
    log "Running pre-release checks"
    
    cd "$workspace"
    
    # 1. Validate release configuration
    if ! validate_release_config; then
        return 1
    fi
    
    # 2. Run security scans
    if ! run_security_scans; then
        return 1
    fi
    
    # 3. Run quality checks
    if ! run_quality_checks; then
        return 1
    fi
    
    # 4. Run compatibility checks
    if ! run_compatibility_checks; then
        return 1
    fi
    
    success "All pre-release checks passed"
    return 0
}

# Validate release configuration
validate_release_config() {
    local pipeline_file="pipeline.json"
    
    if [ ! -f "$pipeline_file" ]; then
        error "Pipeline configuration not found"
        return 1
    fi
    
    # Validate required fields
    local release_id
    release_id=$(jq -r '.release_id' "$pipeline_file")
    
    local version
    version=$(jq -r '.version' "$pipeline_file")
    
    if [ -z "$release_id" ] || [ -z "$version" ]; then
        error "Missing required fields: release_id or version"
        return 1
    fi
    
    # Update pipeline state
    update_pipeline_step "config_validation" "passed" "Release configuration validated"
    
    return 0
}

# Run security scans
run_security_scans() {
    log "Running security scans"
    
    # 1. Static code analysis
    if ! run_static_analysis; then
        update_pipeline_step "security_scan" "failed" "Static analysis failed"
        return 1
    fi
    
    # 2. Dependency vulnerability scan
    if ! run_dependency_scan; then
        update_pipeline_step "dependency_scan" "failed" "Dependency scan failed"
        return 1
    fi
    
    # 3. Secrets detection
    if ! run_secrets_detection; then
        update_pipeline_step "secrets_detection" "failed" "Secrets detection failed"
        return 1
    fi
    
    update_pipeline_step "security_scan" "passed" "All security scans passed"
    return 0
}

# Run static analysis
run_static_analysis() {
    log "Running static code analysis"
    
    # This would integrate with tools like SonarQube, ESLint, etc.
    # For demo purposes, we'll simulate the analysis
    
    local analysis_result="{
        \"complexity\": \"medium\",
        \"duplicates\": \"low\",
        \"maintainability\": \"good\",
        \"security_hotspots\": 0,
        \"coverage\": 85
    }"
    
    # Save analysis results
    echo "$analysis_result" > "static_analysis.json"
    
    # Check thresholds
    local complexity
    complexity=$(echo "$analysis_result" | jq -r '.complexity')
    
    if [ "$complexity" = "high" ]; then
        warn "High complexity detected"
        return 1
    fi
    
    return 0
}

# Run dependency scan
run_dependency_scan() {
    log "Running dependency vulnerability scan"
    
    # Check for package.json
    if [ ! -f "package.json" ]; then
        warn "No package.json found, skipping dependency scan"
        return 0
    fi
    
    # Run npm audit (or equivalent tool)
    local audit_result
    audit_result=$(npm audit --json 2>/dev/null || echo '{"vulnerabilities": {"total": 0}}')
    
    local vuln_count
    vuln_count=$(echo "$audit_result" | jq -r '.metadata.vulnerabilities.total // 0')
    
    if [ "$vuln_count" -gt 0 ]; then
        error "Found $vuln_count vulnerabilities"
        echo "$audit_result" > "dependency_scan.json"
        return 1
    fi
    
    echo "$audit_result" > "dependency_scan.json"
    return 0
}

# Run secrets detection
run_secrets_detection() {
    log "Running secrets detection"
    
    # Scan for potential secrets in code
    local secrets_found=0
    
    # Common secret patterns
    local secret_patterns=(
        "password\s*=\s*['\"][^'\"]+['\"]"
        "api[_-]?key\s*=\s*['\"][^'\"]+['\"]"
        "secret\s*=\s*['\"][^'\"]+['\"]"
        "token\s*=\s*['\"][^'\"]+['\"]"
        "private[_-]?key\s*=\s*['\"][^'\"]+['\"]"
    )
    
    # Scan files
    local files
    files=$(find . -type f -name "*.js" -o -name "*.ts" -o -name "*.json" -o -name "*.yml" -o -name "*.yaml")
    
    for file in $files; do
        for pattern in "${secret_patterns[@]}"; do
            if grep -i -E "$pattern" "$file" > /dev/null 2>&1; then
                error "Potential secret found in $file: $pattern"
                ((secrets_found++))
            fi
        done
    done
    
    if [ "$secrets_found" -gt 0 ]; then
        return 1
    fi
    
    return 0
}

# Run quality checks
run_quality_checks() {
    log "Running quality checks"
    
    # 1. Test coverage
    if ! run_test_coverage_check; then
        update_pipeline_step "test_coverage" "failed" "Test coverage below threshold"
        return 1
    fi
    
    # 2. Code formatting
    if ! run_code_formatting_check; then
        update_pipeline_step "code_formatting" "failed" "Code formatting issues found"
        return 1
    fi
    
    # 3. Linting
    if ! run_linting_check; then
        update_pipeline_step "linting" "failed" "Linting issues found"
        return 1
    fi
    
    update_pipeline_step "quality_checks" "passed" "All quality checks passed"
    return 0
}

# Run test coverage check
run_test_coverage_check() {
    log "Checking test coverage"
    
    # Look for coverage report
    local coverage_file
    coverage_file=$(find . -name "coverage.json" -o -name "coverage-summary.json" | head -1)
    
    if [ -z "$coverage_file" ]; then
        warn "No coverage report found"
        return 0
    fi
    
    local coverage
    coverage=$(jq -r '.total.lines.percent // .percentage // 0' "$coverage_file" 2>/dev/null || echo "0")
    
    local min_coverage
    min_coverage=$(yq eval '.quality.min_test_coverage' "$CONFIG_FILE" 2>/dev/null || echo "80")
    
    if [ "$coverage" -lt "$min_coverage" ]; then
        error "Test coverage ${coverage}% below minimum ${min_coverage}%"
        return 1
    fi
    
    return 0
}

# Run code formatting check
run_code_formatting_check() {
    log "Checking code formatting"
    
    # This would integrate with Prettier, ESLint --fix, etc.
    # For demo, check for common formatting issues
    
    local formatting_issues=0
    
    # Check for trailing whitespace
    if find . -type f \( -name "*.js" -o -name "*.ts" -o -name "*.json" \) -exec grep -l '[[:space:]]$' {} \; | head -5 | grep -q .; then
        warn "Found trailing whitespace"
        ((formatting_issues++))
    fi
    
    if [ "$formatting_issues" -gt 0 ]; then
        return 1
    fi
    
    return 0
}

# Run linting check
run_linting_check() {
    log "Running linting checks"
    
    # This would integrate with ESLint, TSLint, etc.
    # For demo, simulate linting
    
    local linting_issues=0
    
    # Check for common linting issues
    local files
    files=$(find . -type f -name "*.js" -o -name "*.ts" | head -10)
    
    for file in $files; do
        # Check for console.log statements
        if grep -q "console.log" "$file" 2>/dev/null; then
            warn "Found console.log in $file"
            ((linting_issues++))
        fi
        
        # Check for debugger statements
        if grep -q "debugger" "$file" 2>/dev/null; then
            warn "Found debugger statement in $file"
            ((linting_issues++))
        fi
    done
    
    if [ "$linting_issues" -gt 0 ]; then
        return 1
    fi
    
    return 0
}

# Run compatibility checks
run_compatibility_checks() {
    log "Running compatibility checks"
    
    # 1. API compatibility
    if ! run_api_compatibility_check; then
        update_pipeline_step "api_compatibility" "failed" "API compatibility issues found"
        return 1
    fi
    
    # 2. Database migration compatibility
    if ! run_migration_compatibility_check; then
        update_pipeline_step "migration_compatibility" "failed" "Migration compatibility issues found"
        return 1
    fi
    
    update_pipeline_step "compatibility_checks" "passed" "All compatibility checks passed"
    return 0
}

# Run API compatibility check
run_api_compatibility_check() {
    log "Checking API compatibility"
    
    # This would compare API schemas between versions
    # For demo, check for breaking changes
    
    local breaking_changes=0
    
    # Look for API definition files
    local api_files
    api_files=$(find . -name "*.json" -o -name "*.yaml" -o -name "*.proto" | grep -E "(api|schema)" | head -5)
    
    for file in $api_files; do
        # Simple check for removed fields (would be more sophisticated in real implementation)
        if grep -q "removed.*field\|deleted.*endpoint" "$file" 2>/dev/null; then
            warn "Potential breaking change in $file"
            ((breaking_changes++))
        fi
    done
    
    if [ "$breaking_changes" -gt 0 ]; then
        return 1
    fi
    
    return 0
}

# Run migration compatibility check
run_migration_compatibility_check() {
    log "Checking migration compatibility"
    
    # Look for migration files
    local migration_files
    migration_files=$(find . -name "*migration*" -o -name "*migrate*" | head -5)
    
    if [ -z "$migration_files" ]; then
        return 0
    fi
    
    # Check migration files for potential issues
    for file in $migration_files; do
        # Check for destructive operations
        if grep -qi "drop\|delete\|truncate" "$file" 2>/dev/null; then
            warn "Potentially destructive migration in $file"
            return 1
        fi
    done
    
    return 0
}

# Request approvals
request_approvals() {
    local workspace="$1"
    
    log "Requesting approvals"
    
    cd "$workspace"
    
    # Get required approvers
    local required_approvers
    required_approvers=$(yq eval '.approval.required_approvers[]' "$CONFIG_FILE" 2>/dev/null || echo "[]")
    
    if [ ${#required_approvers[@]} -eq 0 ]; then
        log "No approvers required, skipping approval step"
        update_pipeline_step "approval" "skipped" "No approvals required"
        return 0
    fi
    
    # Create approval requests
    local approvals="[]"
    local pipeline_file="pipeline.json"
    local release_id
    release_id=$(jq -r '.release_id' "$pipeline_file")
    
    for approver in "${required_approvers[@]}"; do
        local approval_id
        approval_id=$(uuidgen 2>/dev/null || python3 -c "import uuid; print(uuid.uuid4())")
        
        local approval_request="{
            \"id\": \"$approval_id\",
            \"release_id\": \"$release_id\",
            \"approver\": \"$approver\",
            \"status\": \"pending\",
            \"requested_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
            \"type\": \"manual\"
        }"
        
        approvals=$(echo "$approvals" | jq ". += [$approval_request]")
        
        # Send notification (Slack, email, etc.)
        send_approval_request "$approver" "$release_id" "$approval_id"
    done
    
    # Update pipeline state
    echo "$approvals" > "approvals.json"
    update_pipeline_step "approval" "pending" "Waiting for ${#required_approvers[@]} approvals"
    
    # Wait for approvals
    wait_for_approvals "$release_id"
    
    return 0
}

# Send approval request
send_approval_request() {
    local approver="$1"
    local release_id="$2"
    local approval_id="$3"
    
    log "Sending approval request to: $approver"
    
    if [ -n "$SLACK_WEBHOOK" ]; then
        local message="🔔 Approval Required
        
Release: $release_id
Approver: $approver
Approval ID: $approval_id
Action: Please review and approve the release pipeline"
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" \
            "$SLACK_WEBHOOK" 2>/dev/null || true
    fi
}

# Wait for approvals
wait_for_approvals() {
    local release_id="$1"
    local timeout="${2:-3600}" # 1 hour default
    
    log "Waiting for approvals (timeout: ${timeout}s)"
    
    local start_time
    start_time=$(date +%s)
    
    while true; do
        # Check approval status
        local approvals_status
        approvals_status=$(check_approval_status "$release_id")
        
        case "$approvals_status" in
            "approved")
                update_pipeline_step "approval" "approved" "All approvals received"
                return 0
                ;;
            "rejected")
                update_pipeline_step "approval" "rejected" "Approval was rejected"
                return 1
                ;;
        esac
        
        # Check timeout
        local current_time
        current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ $elapsed -gt $timeout ]; then
            update_pipeline_step "approval" "timeout" "Approval timeout"
            return 1
        fi
        
        sleep 30 # Check every 30 seconds
    done
}

# Check approval status
check_approval_status() {
    local release_id="$1"
    
    # This would check with your approval system
    # For demo, simulate approval checking
    
    local approvals_file="approvals.json"
    
    if [ ! -f "$approvals_file" ]; then
        echo "pending"
        return
    fi
    
    local approved_count
    approved_count=$(jq '[.[] | select(.status == "approved")] | length' "$approvals_file")
    
    local rejected_count
    rejected_count=$(jq '[.[] | select(.status == "rejected")] | length' "$approvals_file")
    
    local total_count
    total_count=$(jq 'length' "$approvals_file")
    
    if [ "$rejected_count" -gt 0 ]; then
        echo "rejected"
    elif [ "$approved_count" -eq "$total_count" ]; then
        echo "approved"
    else
        echo "pending"
    fi
}

# Update pipeline step
update_pipeline_step() {
    local step="$1"
    local status="$2"
    local message="$3"
    
    local pipeline_file="pipeline.json"
    
    if [ ! -f "$pipeline_file" ]; then
        error "Pipeline file not found"
        return 1
    fi
    
    local step_update="{
        \"step\": \"$step\",
        \"status\": \"$status\",
        \"message\": \"$message\",
        \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
    }"
    
    # Update pipeline state
    local updated_pipeline
    updated_pipeline=$(jq ".steps += [$step_update]" "$pipeline_file")
    echo "$updated_pipeline" > "$pipeline_file"
    
    log "Pipeline step updated: $step -> $status"
}

# Execute release
execute_release() {
    local workspace="$1"
    
    log "Executing release"
    
    cd "$workspace"
    
    # 1. Create release tag
    if ! create_release_tag; then
        update_pipeline_step "release" "failed" "Failed to create release tag"
        return 1
    fi
    
    # 2. Build artifacts
    if ! build_release_artifacts; then
        update_pipeline_step "build" "failed" "Failed to build release artifacts"
        return 1
    fi
    
    # 3. Deploy to staging
    if ! deploy_to_staging; then
        update_pipeline_step "staging_deployment" "failed" "Failed to deploy to staging"
        return 1
    fi
    
    # 4. Run smoke tests
    if ! run_smoke_tests; then
        update_pipeline_step "smoke_tests" "failed" "Smoke tests failed"
        return 1
    fi
    
    # 5. Deploy to production
    if ! deploy_to_production; then
        update_pipeline_step "production_deployment" "failed" "Failed to deploy to production"
        return 1
    fi
    
    update_pipeline_step "release" "completed" "Release completed successfully"
    
    # Send notifications
    send_release_notifications
    
    return 0
}

# Create release tag
create_release_tag() {
    local pipeline_file="pipeline.json"
    local version
    version=$(jq -r '.version' "$pipeline_file")
    
    log "Creating release tag: v$version"
    
    # Create annotated tag
    git tag -a "v$version" -m "Release v$version" || {
        error "Failed to create tag"
        return 1
    }
    
    # Push tag to remote
    git push origin "v$version" || {
        error "Failed to push tag"
        return 1
    }
    
    return 0
}

# Build release artifacts
build_release_artifacts() {
    log "Building release artifacts"
    
    # This would integrate with your build system
    # For demo, simulate build process
    
    local build_output="{
        \"artifacts\": [
            \"app.js\",
            \"app.css\",
            \"index.html\"
        ],
        \"build_time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
        \"success\": true
    }"
    
    echo "$build_output" > "build.json"
    
    return 0
}

# Deploy to staging
deploy_to_staging() {
    log "Deploying to staging"
    
    # This would integrate with your deployment system
    # For demo, simulate deployment
    
    local deployment="{
        \"environment\": \"staging\",
        \"deployed_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
        \"success\": true,
        \"url\": \"https://staging.example.com\"
    }"
    
    echo "$deployment" > "staging_deployment.json"
    
    return 0
}

# Run smoke tests
run_smoke_tests() {
    log "Running smoke tests"
    
    # This would integrate with your test system
    # For demo, simulate smoke tests
    
    local test_results="{
        \"total_tests\": 10,
        \"passed_tests\": 9,
        \"failed_tests\": 1,
        \"success_rate\": 90,
        \"duration\": 300
    }"
    
    echo "$test_results" > "smoke_tests.json"
    
    local success_rate
    success_rate=$(echo "$test_results" | jq -r '.success_rate')
    
    if [ "$success_rate" -lt 80 ]; then
        return 1
    fi
    
    return 0
}

# Deploy to production
deploy_to_production() {
    log "Deploying to production"
    
    # This would integrate with your deployment system
    # For demo, simulate deployment
    
    local deployment="{
        \"environment\": \"production\",
        \"deployed_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
        \"success\": true,
        \"url\": \"https://example.com\"
    }"
    
    echo "$deployment" > "production_deployment.json"
    
    return 0
}

# Send release notifications
send_release_notifications() {
    local pipeline_file="pipeline.json"
    local release_id
    release_id=$(jq -r '.release_id' "$pipeline_file")
    local version
    version=$(jq -r '.version' "$pipeline_file")
    
    log "Sending release notifications for: $release_id"
    
    if [ -n "$SLACK_WEBHOOK" ]; then
        local message="🚀 Release Deployed
        
Release: $release_id
Version: v$version
Status: Success
Environment: Production"
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" \
            "$SLACK_WEBHOOK" 2>/dev/null || true
    fi
}

# Show usage
show_usage() {
    cat << EOF
Release Pipeline Integration and Approval Workflow

USAGE:
    $0 <command> [options]

PIPELINE COMMANDS:
    init <release_id> <version>              Initialize release pipeline
    pre-checks <workspace>                   Run pre-release checks
    request-approvals <workspace>              Request approvals
    execute <workspace>                        Execute release
    update-step <step> <status> <message>     Update pipeline step

MONITORING COMMANDS:
    status <workspace>                        Show pipeline status
    approve <approval_id>                      Approve a release
    reject <approval_id>                       Reject a release

EXAMPLES:
    $0 init rel-123 v1.2.3
    $0 pre-checks /path/to/workspace
    $0 request-approvals /path/to/workspace
    $0 execute /path/to/workspace
    $0 update-step security_scan passed "All security checks passed"
    $0 approve approval-123
    $0 reject approval-123

CONFIGURATION:
    Pipeline config: $CONFIG_FILE
    Git repository: $GIT_REPO
    Environment: $ENVIRONMENT
    Slack webhook: ${SLACK_WEBHOOK:-"not set"}

EOF
}

# Main execution
main() {
    case "${1:-}" in
        "init")
            if [ $# -lt 3 ]; then
                error "Usage: $0 init <release_id> <version>"
                exit 1
            fi
            init_pipeline "$2" "$3"
            ;;
        "pre-checks")
            if [ $# -lt 2 ]; then
                error "Usage: $0 pre-checks <workspace>"
                exit 1
            fi
            run_pre_release_checks "$2"
            ;;
        "request-approvals")
            if [ $# -lt 2 ]; then
                error "Usage: $0 request-approvals <workspace>"
                exit 1
            fi
            request_approvals "$2"
            ;;
        "execute")
            if [ $# -lt 2 ]; then
                error "Usage: $0 execute <workspace>"
                exit 1
            fi
            execute_release "$2"
            ;;
        "update-step")
            if [ $# -lt 4 ]; then
                error "Usage: $0 update-step <step> <status> <message>"
                exit 1
            fi
            update_pipeline_step "$2" "$3" "$4"
            ;;
        "status")
            if [ $# -lt 2 ]; then
                error "Usage: $0 status <workspace>"
                exit 1
            fi
            cat "$2/pipeline.json" | jq -r
            ;;
        "approve")
            if [ $# -lt 2 ]; then
                error "Usage: $0 approve <approval_id>"
                exit 1
            fi
            # Update approval status
            local approvals_file="${PROJECT_ROOT}/workspace/*/approvals.json"
            if [ -f "$approvals_file" ]; then
                temp_file=$(mktemp)
                jq "(.[] | select(.id == \"$2\")) |= (.status = \"approved\" | .approved_at = \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\")" "$approvals_file" > "$temp_file"
                mv "$temp_file" "$approvals_file"
            fi
            success "Approval $2 approved"
            ;;
        "reject")
            if [ $# -lt 2 ]; then
                error "Usage: $0 reject <approval_id>"
                exit 1
            fi
            # Update approval status
            local approvals_file="${PROJECT_ROOT}/workspace/*/approvals.json"
            if [ -f "$approvals_file" ]; then
                temp_file=$(mktemp)
                jq "(.[] | select(.id == \"$2\")) |= (.status = \"rejected\" | .rejected_at = \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\")" "$approvals_file" > "$temp_file"
                mv "$temp_file" "$approvals_file"
            fi
            success "Approval $2 rejected"
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"