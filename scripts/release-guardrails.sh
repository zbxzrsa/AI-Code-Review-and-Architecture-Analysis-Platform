#!/bin/bash

# Release Guardrails and Validation Script
# Provides comprehensive automated validation and compliance checks for releases

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_ROOT}/config/release-guardrails.yml"
GIT_REPO="${GIT_REPO:-$(pwd)}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] GUARDRAILS:${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] GUARDRAILS WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] GUARDRAILS ERROR:${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] GUARDRAILS SUCCESS:${NC} $1"
}

# Validate semantic version
validate_semantic_version() {
    local version="$1"
    
    # Semantic versioning regex
    local semantic_regex="^v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*)?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
    
    if [[ $version =~ $semantic_regex ]]; then
        local major="${BASH_REMATCH[1]}"
        local minor="${BASH_REMATCH[2]}"
        local patch="${BASH_REMATCH[3]}"
        local prerelease="${BASH_REMATCH[4]}"
        local build="${BASH_REMATCH[5]}"
        
        success "✓ Valid semantic version: v$major.$minor.$patch"
        if [ -n "$prerelease" ]; then
            log "  Prerelease: $prerelease"
        fi
        if [ -n "$build" ]; then
            log "  Build: $build"
        fi
        return 0
    else
        error "✗ Invalid semantic version: $version"
        log "  Expected format: v1.2.3 or v1.2.3-alpha.1"
        return 1
    fi
}

# Validate branch protection
validate_branch_protection() {
    local branch="$1"
    
    log "Validating branch protection for: $branch"
    
    # Get protected branches from config
    local protected_branches
    protected_branches=$(yq eval '.protected_branches[]' "$CONFIG_FILE" 2>/dev/null || echo "['main', 'master', 'develop']")
    
    # Check if branch is protected
    local is_protected=false
    for protected_branch in $protected_branches; do
        # Remove quotes and spaces
        protected_branch=$(echo "$protected_branch" | tr -d "'" | tr -d ' ')
        if [ "$branch" = "$protected_branch" ]; then
            is_protected=true
            break
        fi
    done
    
    if [ "$is_protected" = true ]; then
        success "✓ Branch '$branch' is protected"
        return 0
    else
        warn "⚠ Branch '$branch' is not in protected branches list"
        log "  Protected branches: $protected_branches"
        return 1
    fi
}

# Validate commit messages
validate_commit_messages() {
    local commit_hash="$1"
    local commit_range="${2:-$commit_hash^..$commit_hash}"
    
    log "Validating commit messages for: $commit_hash"
    
    # Get commit message
    local commit_message
    commit_message=$(git log --format="%s" -n 1 "$commit_hash")
    
    # Check for required patterns
    local required_patterns
    required_patterns=$(yq eval '.commit_message.required_patterns[]' "$CONFIG_FILE" 2>/dev/null || echo "['^(feat|fix|docs|style|refactor|test|chore)(\(.+\))?: .+']")
    
    local pattern_valid=false
    for pattern in $required_patterns; do
        pattern=$(echo "$pattern" | tr -d "'" | tr -d ' ')
        if [[ $commit_message =~ $pattern ]]; then
            pattern_valid=true
            break
        fi
    done
    
    if [ "$pattern_valid" = true ]; then
        success "✓ Commit message follows required pattern"
        return 0
    else
        error "✗ Commit message doesn't follow required pattern"
        log "  Message: $commit_message"
        log "  Expected patterns: $required_patterns"
        return 1
    fi
}

# Run security scan
run_security_scan() {
    local commit_hash="$1"
    
    log "Running security scan for commit: $commit_hash"
    
    # Check for sensitive data
    local sensitive_patterns
    sensitive_patterns=$(yq eval '.security.sensitive_patterns[]' "$CONFIG_FILE" 2>/dev/null || echo "['password', 'secret', 'key', 'token', 'api_key']")
    
    local security_issues=0
    
    # Scan files in commit
    local files
    files=$(git diff-tree --no-commit-id --name-only -r "$commit_hash")
    
    for file in $files; do
        local content
        content=$(git show "$commit_hash:$file" 2>/dev/null || continue)
        
        for pattern in $sensitive_patterns; do
            pattern=$(echo "$pattern" | tr -d "'" | tr -d ' ')
            if echo "$content" | grep -i -q "$pattern"; then
                error "✗ Sensitive data pattern found in $file: $pattern"
                ((security_issues++))
            fi
        done
    done
    
    if [ $security_issues -eq 0 ]; then
        success "✓ No security issues found"
        return 0
    else
        error "✗ Found $security_issues security issues"
        return 1
    fi
}

# Check test coverage
check_test_coverage() {
    local commit_hash="$1"
    local min_coverage="${2:-80}"
    
    log "Checking test coverage for commit: $commit_hash"
    
    # Run coverage report (placeholder - integrate with your coverage tool)
    local coverage_result
    coverage_result=$(git show "$commit_hash:coverage.json" 2>/dev/null || echo '{"percentage":0}')
    
    local coverage
    coverage=$(echo "$coverage_result" | jq -r '.percentage // 0')
    
    if [ "$coverage" -ge "$min_coverage" ]; then
        success "✓ Test coverage ${coverage}% meets minimum requirement ${min_coverage}%"
        return 0
    else
        error "✗ Test coverage ${coverage}% below minimum requirement ${min_coverage}%"
        return 1
    fi
}

# Validate dependencies
validate_dependencies() {
    local commit_hash="$1"
    
    log "Validating dependencies for commit: $commit_hash"
    
    # Get package.json changes
    local package_changes
    package_changes=$(git diff-tree --no-commit-id --name-only -r "$commit_hash" | grep -E "(package\.json|yarn\.lock|package-lock\.json)" || true)
    
    if [ -z "$package_changes" ]; then
        success "✓ No dependency changes detected"
        return 0
    fi
    
    # Check for vulnerable dependencies
    local vulnerabilities=0
    
    for file in $package_changes; do
        local content
        content=$(git show "$commit_hash:$file" 2>/dev/null || continue)
        
        # Run vulnerability check (placeholder - integrate with security scanner)
        if echo "$content" | jq -e '.dependencies' > /dev/null 2>&1; then
            local vuln_check
            vuln_check=$(echo "$content" | npm audit --json 2>/dev/null | jq -r '.metadata.vulnerabilities.total // 0' || echo "0")
            
            if [ "$vuln_check" -gt 0 ]; then
                error "✗ Found $vuln_check vulnerabilities in $file"
                ((vulnerabilities++))
            fi
        fi
    done
    
    if [ $vulnerabilities -eq 0 ]; then
        success "✓ No vulnerable dependencies found"
        return 0
    else
        error "✗ Found $vulnerabilities vulnerable dependencies"
        return 1
    fi
}

# Check API breaking changes
check_api_breaking_changes() {
    local commit_hash="$1"
    local base_branch="${2:-main}"
    
    log "Checking for API breaking changes against: $base_branch"
    
    # Get API definition files
    local api_files
    api_files=$(git diff-tree --no-commit-id --name-only -r "$base_branch..$commit_hash" | grep -E "\.(json|yaml|yml|proto)$" || true)
    
    if [ -z "$api_files" ]; then
        success "✓ No API definition changes detected"
        return 0
    fi
    
    # Analyze changes for breaking patterns
    local breaking_changes=0
    
    for file in $api_files; do
        local changes
        changes=$(git diff "$base_branch..$commit_hash" -- "$file")
        
        # Check for breaking change patterns
        local breaking_patterns
        breaking_patterns=$(yq eval '.api_breaking_patterns[]' "$CONFIG_FILE" 2>/dev/null || echo "['removed.*field', 'deleted.*endpoint', 'changed.*type', 'required.*field']")
        
        for pattern in $breaking_patterns; do
            pattern=$(echo "$pattern" | tr -d "'" | tr -d ' ')
            if echo "$changes" | grep -i -q "$pattern"; then
                error "✗ Potential breaking change in $file: $pattern"
                ((breaking_changes++))
            fi
        done
    done
    
    if [ $breaking_changes -eq 0 ]; then
        success "✓ No API breaking changes detected"
        return 0
    else
        error "✗ Found $breaking_changes potential breaking changes"
        return 1
    fi
}

# Validate release readiness
validate_release_readiness() {
    local tag="$1"
    local commit_hash="${2:-$(git rev-list -n 1 "$tag")}"
    
    log "Validating release readiness for: $tag"
    
    local validation_errors=0
    local validation_warnings=0
    
    # 1. Semantic version check
    if ! validate_semantic_version "$tag"; then
        ((validation_errors++))
    fi
    
    # 2. Branch protection check
    local current_branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    if ! validate_branch_protection "$current_branch"; then
        ((validation_warnings++))
    fi
    
    # 3. Commit message validation
    if ! validate_commit_messages "$commit_hash"; then
        ((validation_errors++))
    fi
    
    # 4. Security scan
    if ! run_security_scan "$commit_hash"; then
        ((validation_errors++))
    fi
    
    # 5. Test coverage check
    local min_coverage
    min_coverage=$(yq eval '.test_coverage.minimum' "$CONFIG_FILE" 2>/dev/null || echo "80")
    if ! check_test_coverage "$commit_hash" "$min_coverage"; then
        ((validation_warnings++))
    fi
    
    # 6. Dependency validation
    if ! validate_dependencies "$commit_hash"; then
        ((validation_errors++))
    fi
    
    # 7. API breaking changes check
    if ! check_api_breaking_changes "$commit_hash"; then
        ((validation_warnings++))
    fi
    
    # Generate report
    echo ""
    log "=== RELEASE READINESS REPORT ==="
    log "Tag: $tag"
    log "Commit: $commit_hash"
    log "Branch: $current_branch"
    log "Errors: $validation_errors"
    log "Warnings: $validation_warnings"
    echo ""
    
    if [ $validation_errors -eq 0 ]; then
        if [ $validation_warnings -eq 0 ]; then
            success "✅ Release is READY - No issues found"
            return 0
        else
            warn "⚠️ Release is READY with warnings - Review before proceeding"
            return 0
        fi
    else
        error "❌ Release is NOT READY - Fix errors before proceeding"
        return 1
    fi
}

# Create release guardrail
create_guardrail() {
    local name="$1"
    local type="$2"
    local conditions="$3"
    local actions="$4"
    local enabled="${5:-true}"
    
    log "Creating release guardrail: $name"
    
    local guardrail="{
        \"id\": \"$(uuidgen 2>/dev/null || python3 -c "import uuid; print(uuid.uuid4())")\",
        \"name\": \"$name\",
        \"type\": \"$type\",
        \"enabled\": $enabled,
        \"conditions\": $conditions,
        \"actions\": $actions,
        \"metadata\": {
            \"created_by\": \"$(git config user.name)\",
            \"created_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
            \"environment\": \"$ENVIRONMENT\"
        }
    }"
    
    # Save to config
    local temp_file
    temp_file=$(mktemp)
    echo "$guardrail" > "$temp_file"
    
    # Merge with existing guardrails
    if [ -f "$CONFIG_FILE" ]; then
        yq eval '.guardrails += [input]' "$temp_file" -i "$CONFIG_FILE"
    else
        echo "guardrails: [$guardrail]" > "$CONFIG_FILE"
    fi
    
    rm -f "$temp_file"
    
    success "Guardrail '$name' created successfully"
}

# List guardrails
list_guardrails() {
    log "Listing release guardrails..."
    
    if [ ! -f "$CONFIG_FILE" ]; then
        warn "No guardrails configuration found"
        return 1
    fi
    
    yq eval '.guardrails[] | .name + " (" + .type + ") - " + (.enabled | if . then "ENABLED" else "DISABLED" end)' "$CONFIG_FILE"
}

# Test guardrail
test_guardrail() {
    local guardrail_name="$1"
    local test_data="$2"
    
    log "Testing guardrail: $guardrail_name"
    
    # Get guardrail configuration
    local guardrail
    guardrail=$(yq eval ".guardrails[] | select(.name == \"$guardrail_name\")" "$CONFIG_FILE")
    
    if [ -z "$guardrail" ]; then
        error "Guardrail '$guardrail_name' not found"
        return 1
    fi
    
    # Evaluate conditions
    local conditions_met=true
    local conditions
    conditions=$(echo "$guardrail" | jq -r '.conditions')
    
    # Simple condition evaluation (extend as needed)
    if echo "$test_data" | jq -e "$conditions" > /dev/null 2>&1; then
        conditions_met=true
    else
        conditions_met=false
    fi
    
    if [ "$conditions_met" = true ]; then
        success "✓ Guardrail conditions met"
        log "  Actions to execute: $(echo "$guardrail" | jq -r '.actions')"
    else
        warn "✗ Guardrail conditions not met"
        return 1
    fi
}

# Show usage
show_usage() {
    cat << EOF
Release Guardrails and Validation Script

USAGE:
    $0 <command> [options]

VALIDATION COMMANDS:
    validate-version <version>                    Validate semantic versioning
    validate-branch <branch>                    Validate branch protection
    validate-commit <commit_hash>                 Validate commit message
    security-scan <commit_hash>                   Run security scan
    check-coverage <commit_hash> [min_coverage]     Check test coverage
    validate-deps <commit_hash>                   Validate dependencies
    check-api-breaking <commit_hash> [base_branch]   Check for API breaking changes
    release-readiness <tag> [commit_hash]         Complete release validation

GUARDRAIL COMMANDS:
    create <name> <type> <conditions> <actions> [enabled]
    list                                         List all guardrails
    test <name> <test_data>                     Test specific guardrail

EXAMPLES:
    $0 validate-version v1.2.3
    $0 validate-branch main
    $0 validate-commit abc123
    $0 security-scan abc123
    $0 check-coverage abc123 85
    $0 validate-deps abc123
    $0 check-api-breaking abc123 main
    $0 release-readiness v1.2.3 abc123
    $0 create "No Breaking Changes" "pre_release" '[{"field":"api_breaking","operator":"equals","value":false}]' '[{"type":"block"}]'
    $0 list
    $0 test "No Breaking Changes" '{"api_breaking":false}'

CONFIGURATION:
    Config file: $CONFIG_FILE
    Git repository: $GIT_REPO
    Environment: $ENVIRONMENT

EOF
}

# Main execution
main() {
    case "${1:-}" in
        "validate-version")
            if [ $# -lt 2 ]; then
                error "Usage: $0 validate-version <version>"
                exit 1
            fi
            validate_semantic_version "$2"
            ;;
        "validate-branch")
            if [ $# -lt 2 ]; then
                error "Usage: $0 validate-branch <branch>"
                exit 1
            fi
            validate_branch_protection "$2"
            ;;
        "validate-commit")
            if [ $# -lt 2 ]; then
                error "Usage: $0 validate-commit <commit_hash>"
                exit 1
            fi
            validate_commit_messages "$2"
            ;;
        "security-scan")
            if [ $# -lt 2 ]; then
                error "Usage: $0 security-scan <commit_hash>"
                exit 1
            fi
            run_security_scan "$2"
            ;;
        "check-coverage")
            if [ $# -lt 2 ]; then
                error "Usage: $0 check-coverage <commit_hash> [min_coverage]"
                exit 1
            fi
            check_test_coverage "$2" "${3:-80}"
            ;;
        "validate-deps")
            if [ $# -lt 2 ]; then
                error "Usage: $0 validate-deps <commit_hash>"
                exit 1
            fi
            validate_dependencies "$2"
            ;;
        "check-api-breaking")
            if [ $# -lt 2 ]; then
                error "Usage: $0 check-api-breaking <commit_hash> [base_branch]"
                exit 1
            fi
            check_api_breaking_changes "$2" "${3:-main}"
            ;;
        "release-readiness")
            if [ $# -lt 2 ]; then
                error "Usage: $0 release-readiness <tag> [commit_hash]"
                exit 1
            fi
            validate_release_readiness "$2" "${3:-}"
            ;;
        "create")
            if [ $# -lt 5 ]; then
                error "Usage: $0 create <name> <type> <conditions> <actions> [enabled]"
                exit 1
            fi
            create_guardrail "$2" "$3" "$4" "$5" "${6:-true}"
            ;;
        "list")
            list_guardrails
            ;;
        "test")
            if [ $# -lt 3 ]; then
                error "Usage: $0 test <name> <test_data>"
                exit 1
            fi
            test_guardrail "$2" "$3"
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"