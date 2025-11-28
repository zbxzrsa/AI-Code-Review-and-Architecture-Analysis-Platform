#!/bin/bash

# Rollback and Emergency Release Mechanisms
# Provides comprehensive rollback and emergency release capabilities

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_ROOT}/config/rollback-config.yml"
GIT_REPO="${GIT_REPO:-$(pwd)}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
PAGERDUTY_KEY="${PAGERDUTY_KEY:-}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] ROLLBACK:${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ROLLBACK WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ROLLBACK ERROR:${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ROLLBACK SUCCESS:${NC} $1"
}

# Create rollback plan
create_rollback_plan() {
    local release_id="$1"
    local reason="$2"
    local trigger="${3:-manual}"
    
    log "Creating rollback plan for release: $release_id"
    
    # Get release information
    local release_info
    release_info=$(get_release_info "$release_id")
    
    if [ -z "$release_info" ]; then
        error "Release not found: $release_id"
        return 1
    fi
    
    # Create rollback plan
    local rollback_id
    rollback_id=$(uuidgen 2>/dev/null || python3 -c "import uuid; print(uuid.uuid4())")
    
    local rollback_plan="{
        \"id\": \"$rollback_id\",
        \"release_id\": \"$release_id\",
        \"reason\": \"$reason\",
        \"trigger\": \"$trigger\",
        \"status\": \"planned\",
        \"created_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
        \"release_info\": $release_info,
        \"steps\": [],
        \"rollback_type\": \"full\",
        \"estimated_duration\": 300,
        \"risk_level\": \"medium\"
    }"
    
    # Save rollback plan
    local plans_file="${PROJECT_ROOT}/data/rollback_plans.json"
    mkdir -p "$(dirname "$plans_file")"
    
    if [ -f "$plans_file" ]; then
        temp_file=$(mktemp)
        jq ". += [$rollback_plan]" "$plans_file" > "$temp_file"
        mv "$temp_file" "$plans_file"
    else
        echo "[$rollback_plan]" > "$plans_file"
    fi
    
    success "Rollback plan created: $rollback_id"
    echo "$rollback_id"
}

# Get release information
get_release_info() {
    local release_id="$1"
    
    # This would integrate with your release management system
    # For demo, simulate getting release info
    local release_info="{
        \"id\": \"$release_id\",
        \"version\": \"v1.2.3\",
        \"tag\": \"v1.2.3\",
        \"commit\": \"abc123def456\",
        \"deployed_at\": \"2024-01-15T10:30:00Z\",
        \"environment\": \"$ENVIRONMENT\",
        \"services\": [
            \"web-app\",
            \"api-service\",
            \"database\"
        ],
        \"artifacts\": [
            \"app.js\",
            \"styles.css\",
            \"index.html\"
        ],
        \"rollback_info\": {
            \"previous_tag\": \"v1.2.2\",
            \"previous_commit\": \"def456abc123\",
            \"rollback_commands\": [
                \"git checkout v1.2.2\",
                \"docker-compose down\",
                \"kubectl apply -f k8s-v1.2.2.yaml\"
            ]
        }
    }"
    
    echo "$release_info"
}

# Execute rollback
execute_rollback() {
    local rollback_id="$1"
    local force="${2:-false}"
    
    log "Executing rollback: $rollback_id (force: $force)"
    
    # Get rollback plan
    local plans_file="${PROJECT_ROOT}/data/rollback_plans.json"
    local rollback_plan
    rollback_plan=$(jq ".[] | select(.id == \"$rollback_id\")" "$plans_file")
    
    if [ -z "$rollback_plan" ]; then
        error "Rollback plan not found: $rollback_id"
        return 1
    fi
    
    # Update status
    update_rollback_status "$rollback_id" "executing" "Rollback execution started"
    
    # Pre-execution safety checks
    if [ "$force" != "true" ]; then
        if ! run_safety_checks "$rollback_plan"; then
            update_rollback_status "$rollback_id" "failed" "Safety checks failed"
            return 1
        fi
    fi
    
    # Execute rollback steps
    local steps
    steps=$(echo "$rollback_plan" | jq -r '.release_info.rollback_info.rollback_commands[]')
    
    local step_number=1
    for step in $steps; do
        log "Executing step $step_number: $step"
        
        if execute_rollback_step "$step" "$rollback_id"; then
            update_rollback_step "$rollback_id" "$step_number" "completed" "Step completed successfully"
        else
            update_rollback_step "$rollback_id" "$step_number" "failed" "Step failed"
            
            if [ "$force" != "true" ]; then
                update_rollback_status "$rollback_id" "failed" "Step $step_number failed, aborting rollback"
                return 1
            fi
        fi
        
        ((step_number++))
    done
    
    # Post-rollback verification
    if ! verify_rollback "$rollback_plan"; then
        update_rollback_status "$rollback_id" "failed" "Rollback verification failed"
        return 1
    fi
    
    # Update final status
    update_rollback_status "$rollback_id" "completed" "Rollback completed successfully"
    
    # Send notifications
    send_rollback_notifications "$rollback_plan" "completed"
    
    success "Rollback completed: $rollback_id"
}

# Execute rollback step
execute_rollback_step() {
    local step="$1"
    local rollback_id="$2"
    
    case "$step" in
        "git checkout "*")
            local tag
            tag=$(echo "$step" | cut -d' ' -f2)
            
            log "Checking out tag: $tag"
            git checkout "$tag" 2>/dev/null || {
                error "Failed to checkout tag: $tag"
                return 1
            }
            ;;
        "docker-compose down"*)
            log "Stopping Docker services"
            docker-compose down 2>/dev/null || {
                warn "Docker Compose not available, skipping"
                return 0
            }
            ;;
        "kubectl apply "*")
            local manifest
            manifest=$(echo "$step" | cut -d' ' -f2)
            
            log "Applying Kubernetes manifest: $manifest"
            kubectl apply -f "$manifest" 2>/dev/null || {
                error "Failed to apply manifest: $manifest"
                return 1
            }
            ;;
        *)
            warn "Unknown rollback step: $step"
            return 0
            ;;
    esac
    
    return 0
}

# Run safety checks
run_safety_checks() {
    local rollback_plan="$1"
    
    log "Running safety checks"
    
    local checks_passed=0
    local total_checks=4
    
    # 1. Check if release is recent
    local release_date
    release_date=$(echo "$rollback_plan" | jq -r '.release_info.deployed_at')
    
    local days_since_release
    days_since_release=$(( ($(date +%s) - $(date -d "$release_date" +%s)) / 86400))
    
    if [ "$days_since_release" -gt 7 ]; then
        warn "Release is more than 7 days old - rollback may be risky"
    else
        ((checks_passed++))
    fi
    
    # 2. Check for newer releases
    local newer_releases
    newer_releases=$(check_for_newer_releases "$rollback_plan")
    
    if [ -n "$newer_releases" ]; then
        warn "Newer releases exist - rollback may override recent changes"
    else
        ((checks_passed++))
    fi
    
    # 3. Check service health
    if ! check_service_health; then
        warn "Service health issues detected - rollback may be unsafe"
    else
        ((checks_passed++))
    fi
    
    # 4. Check database state
    if ! check_database_state; then
        warn "Database issues detected - rollback may cause data loss"
    else
        ((checks_passed++))
    fi
    
    log "Safety checks passed: $checks_passed/$total_checks"
    
    if [ "$checks_passed" -eq "$total_checks" ]; then
        return 0
    else
        return 1
    fi
}

# Check for newer releases
check_for_newer_releases() {
    local rollback_plan="$1"
    local current_version
    current_version=$(echo "$rollback_plan" | jq -r '.release_info.version')
    
    # This would check with your release management system
    # For demo, simulate checking for newer releases
    echo "" # Return empty (no newer releases found)
}

# Check service health
check_service_health() {
    log "Checking service health"
    
    # This would integrate with your monitoring system
    # For demo, simulate health checks
    
    local services_healthy=0
    local total_services=3
    
    # Check web service
    if curl -f http://localhost:3000/health 2>/dev/null; then
        ((services_healthy++))
    fi
    
    # Check API service
    if curl -f http://localhost:3001/health 2>/dev/null; then
        ((services_healthy++))
    fi
    
    # Check database
    if pg_isready -h localhost 2>/dev/null; then
        ((services_healthy++))
    fi
    
    log "Services healthy: $services_healthy/$total_services"
    
    if [ "$services_healthy" -eq "$total_services" ]; then
        return 0
    else
        return 1
    fi
}

# Check database state
check_database_state() {
    log "Checking database state"
    
    # This would integrate with your database monitoring
    # For demo, simulate database state check
    
    # Check for active connections
    local active_connections
    active_connections=$(pg_activity -h localhost -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';" -t 2>/dev/null || echo "0")
    
    if [ "$active_connections" -gt 10 ]; then
        warn "High number of active database connections: $active_connections"
        return 1
    fi
    
    # Check for long-running transactions
    local long_transactions
    long_transactions=$(pg_activity -h localhost -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active in backend' AND query_start < now() - interval '5 minutes';" -t 2>/dev/null || echo "0")
    
    if [ "$long_transactions" -gt 0 ]; then
        warn "Long-running transactions detected: $long_transactions"
        return 1
    fi
    
    return 0
}

# Verify rollback
verify_rollback() {
    local rollback_plan="$1"
    
    log "Verifying rollback"
    
    # Get expected state
    local expected_tag
    expected_tag=$(echo "$rollback_plan" | jq -r '.release_info.rollback_info.previous_tag')
    
    # Check current state
    local current_tag
    current_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "unknown")
    
    local current_commit
    current_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    
    # Verify rollback
    if [ "$current_tag" = "$expected_tag" ]; then
        success "Rollback verification passed: on expected tag $expected_tag"
        return 0
    else
        warn "Rollback verification warning: on tag $current_tag (expected $expected_tag)"
        return 1
    fi
}

# Update rollback status
update_rollback_status() {
    local rollback_id="$1"
    local status="$2"
    local message="$3"
    
    local plans_file="${PROJECT_ROOT}/data/rollback_plans.json"
    
    if [ -f "$plans_file" ]; then
        temp_file=$(mktemp)
        jq "(.[] | select(.id == \"$rollback_id\")) |= (.status = \"$status\" | .message = \"$message\" | .updated_at = \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\")" "$plans_file" > "$temp_file"
        mv "$temp_file" "$plans_file"
    fi
    
    log "Rollback status updated: $rollback_id -> $status"
}

# Update rollback step
update_rollback_step() {
    local rollback_id="$1"
    local step_number="$2"
    local status="$3"
    local message="$4"
    
    local plans_file="${PROJECT_ROOT}/data/rollback_plans.json"
    
    if [ -f "$plans_file" ]; then
        temp_file=$(mktemp)
        jq "(.[] | select(.id == \"$rollback_id\")) |= (.steps[$((step_number-1))] = {\"status\": \"$status\", \"message\": \"$message\", \"updated_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"})" "$plans_file" > "$temp_file"
        mv "$temp_file" "$plans_file"
    fi
}

# Send rollback notifications
send_rollback_notifications() {
    local rollback_plan="$1"
    local status="$2"
    
    local rollback_id
    rollback_id=$(echo "$rollback_plan" | jq -r '.id')
    local release_id
    release_id=$(echo "$rollback_plan" | jq -r '.release_id')
    local reason
    reason=$(echo "$rollback_plan" | jq -r '.reason')
    
    log "Sending rollback notifications"
    
    # Send Slack notification
    if [ -n "$SLACK_WEBHOOK" ]; then
        local status_emoji="✅"
        if [ "$status" = "failed" ]; then
            status_emoji="❌"
        elif [ "$status" = "executing" ]; then
            status_emoji="🔄"
        fi
        
        local message="$status_emoji Rollback $status
        
Release ID: $rollback_id
Release: $release_id
Reason: $reason"
Status: $status"
Time: $(date)"
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" \
            "$SLACK_WEBHOOK" 2>/dev/null || true
    fi
    
    # Send PagerDuty alert
    if [ -n "$PAGERDUTY_KEY" ] && [ "$status" = "failed" ]; then
        local severity="critical"
        if [ "$status" = "completed" ]; then
            severity="warning"
        fi
        
        local pd_message="Rollback $status for release $release_id: $reason"
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{
                \"routing_key\": \"$PAGERDUTY_KEY\",
                \"event_action\": \"trigger\",
                \"payload\": {
                    \"summary\": \"$pd_message\",
                    \"severity\": \"$severity\",
                    \"source\": \"rollback_system\",
                    \"custom_details\": {
                        \"rollback_id\": \"$rollback_id\",
                        \"release_id\": \"$release_id\",
                        \"reason\": \"$reason\"
                    }
                }
            }" \
            "https://events.pagerduty.com/v2/enqueue" 2>/dev/null || true
    fi
}

# Create emergency release
create_emergency_release() {
    local issue="$1"
    local description="$2"
    local severity="${3:-critical}"
    local fix_commit="${4:-}"
    
    log "Creating emergency release for issue: $issue"
    
    # Create emergency release plan
    local emergency_id
    emergency_id=$(uuidgen 2>/dev/null || python3 -c "import uuid; print(uuid.uuid4())")
    
    local emergency_plan="{
        \"id\": \"$emergency_id\",
        \"issue\": \"$issue\",
        \"description\": \"$description\",
        \"severity\": \"$severity\",
        \"fix_commit\": \"$fix_commit\",
        \"status\": \"planned\",
        \"created_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
        \"type\": \"emergency\",
        \"bypass_checks\": true,
        \"auto_deploy\": true
    }"
    
    # Save emergency plan
    local plans_file="${PROJECT_ROOT}/data/emergency_releases.json"
    mkdir -p "$(dirname "$plans_file")"
    
    if [ -f "$plans_file" ]; then
        temp_file=$(mktemp)
        jq ". += [$emergency_plan]" "$plans_file" > "$temp_file"
        mv "$temp_file" "$plans_file"
    else
        echo "[$emergency_plan]" > "$plans_file"
    fi
    
    # Execute emergency release immediately
    execute_emergency_release "$emergency_id"
    
    success "Emergency release created: $emergency_id"
}

# Execute emergency release
execute_emergency_release() {
    local emergency_id="$1"
    
    log "Executing emergency release: $emergency_id"
    
    # Get emergency plan
    local plans_file="${PROJECT_ROOT}/data/emergency_releases.json"
    local emergency_plan
    emergency_plan=$(jq ".[] | select(.id == \"$emergency_id\")" "$plans_file")
    
    if [ -z "$emergency_plan" ]; then
        error "Emergency release plan not found: $emergency_id"
        return 1
    fi
    
    # Update status
    update_emergency_status "$emergency_id" "executing" "Emergency release execution started"
    
    # Create hotfix branch
    local hotfix_branch="emergency/$emergency_id"
    git checkout -b "$hotfix_branch" 2>/dev/null || {
        update_emergency_status "$emergency_id" "failed" "Failed to create hotfix branch"
        return 1
    }
    
    # Apply fix if provided
    if [ -n "$(echo "$emergency_plan" | jq -r '.fix_commit')" ]; then
        local fix_commit
        fix_commit=$(echo "$emergency_plan" | jq -r '.fix_commit')
        
        log "Applying fix commit: $fix_commit"
        git cherry-pick "$fix_commit" 2>/dev/null || {
            update_emergency_status "$emergency_id" "failed" "Failed to apply fix commit"
            return 1
        fi
    fi
    
    # Create emergency tag
    local emergency_tag="emergency-v$(date +%Y%m%d-%H%M%S)"
    git tag -a "$emergency_tag" -m "Emergency release for issue $(echo "$emergency_plan" | jq -r '.issue')" 2>/dev/null || {
        update_emergency_status "$emergency_id" "failed" "Failed to create emergency tag"
        return 1
    }
    
    # Push to remote
    git push origin "$hotfix_branch" 2>/dev/null || {
        update_emergency_status "$emergency_id" "failed" "Failed to push hotfix branch"
        return 1
    }
    
    git push origin "$emergency_tag" 2>/dev/null || {
        update_emergency_status "$emergency_id" "failed" "Failed to push emergency tag"
        return 1
    }
    
    # Update final status
    update_emergency_status "$emergency_id" "completed" "Emergency release completed successfully"
    
    # Send notifications
    send_emergency_notifications "$emergency_plan" "completed"
    
    success "Emergency release completed: $emergency_id"
}

# Update emergency status
update_emergency_status() {
    local emergency_id="$1"
    local status="$2"
    local message="$3"
    
    local plans_file="${PROJECT_ROOT}/data/emergency_releases.json"
    
    if [ -f "$plans_file" ]; then
        temp_file=$(mktemp)
        jq "(.[] | select(.id == \"$emergency_id\")) |= (.status = \"$status\" | .message = \"$message\" | .updated_at = \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\")" "$plans_file" > "$temp_file"
        mv "$temp_file" "$plans_file"
    fi
    
    log "Emergency release status updated: $emergency_id -> $status"
}

# Send emergency notifications
send_emergency_notifications() {
    local emergency_plan="$1"
    local status="$2"
    
    local emergency_id
    emergency_id=$(echo "$emergency_plan" | jq -r '.id')
    local issue
    issue=$(echo "$emergency_plan" | jq -r '.issue')
    local severity
    severity=$(echo "$emergency_plan" | jq -r '.severity')
    
    log "Sending emergency notifications"
    
    # Send critical alert to all channels
    if [ -n "$SLACK_WEBHOOK" ]; then
        local status_emoji="🚨"
        if [ "$status" = "completed" ]; then
            status_emoji="✅"
        elif [ "$status" = "failed" ]; then
            status_emoji="❌"
        fi
        
        local message="$status_emoji EMERGENCY RELEASE $status
        
Issue: $issue
Severity: $severity
Emergency ID: $emergency_id
Time: $(date)"
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" \
            "$SLACK_WEBHOOK" 2>/dev/null || true
    fi
    
    # Send PagerDuty critical alert
    if [ -n "$PAGERDUTY_KEY" ]; then
        local pd_message="Emergency release $status for issue $issue (severity: $severity)"
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{
                \"routing_key\": \"$PAGERDUTY_KEY\",
                \"event_action\": \"trigger\",
                \"payload\": {
                    \"summary\": \"$pd_message\",
                    \"severity\": \"critical\",
                    \"source\": \"emergency_release\",
                    \"custom_details\": {
                        \"emergency_id\": \"$emergency_id\",
                        \"issue\": \"$issue\",
                        \"severity\": \"$severity\"
                    }
                }
            }" \
            "https://events.pagerduty.com/v2/enqueue" 2>/dev/null || true
    fi
}

# List rollback plans
list_rollback_plans() {
    local status_filter="${1:-}"
    local limit="${2:-20}"
    
    local plans_file="${PROJECT_ROOT}/data/rollback_plans.json"
    
    if [ ! -f "$plans_file" ]; then
        warn "No rollback plans found"
        return 1
    fi
    
    local filter=""
    if [ -n "$status_filter" ]; then
        filter=" | select(.status == \"$status_filter\")"
    fi
    
    jq ".[] $filter | limit($limit)" "$plans_file"
}

# Show rollback plan details
show_rollback_plan() {
    local rollback_id="$1"
    
    local plans_file="${PROJECT_ROOT}/data/rollback_plans.json"
    
    if [ ! -f "$plans_file" ]; then
        error "Rollback plan not found: $rollback_id"
        return 1
    fi
    
    local plan
    plan=$(jq ".[] | select(.id == \"$rollback_id\")" "$plans_file")
    
    if [ -z "$plan" ]; then
        error "Rollback plan not found: $rollback_id"
        return 1
    fi
    
    echo "$plan" | jq -r '
    "=== ROLLBACK PLAN ===",
    "ID: " + .id,
    "Release ID: " + .release_id,
    "Reason: " + .reason,
    "Trigger: " + .trigger,
    "Status: " + .status,
    "Created: " + .created_at,
    "",
    "=== RELEASE INFO ===",
    "Version: " + (.release_info.version // "unknown"),
    "Tag: " + (.release_info.tag // "unknown"),
    "Deployed: " + (.release_info.deployed_at // "unknown"),
    "",
    "=== ROLLBACK STEPS ===",
    (.steps[] | "- Step " + (.index + 1) + ": " + .status + " - " + .message)
    '
}

# Show usage
show_usage() {
    cat << EOF
Rollback and Emergency Release Mechanisms

USAGE:
    $0 <command> [options]

ROLLBACK COMMANDS:
    plan <release_id> <reason> [trigger]           Create rollback plan
    execute <rollback_id> [force]                 Execute rollback
    list [status] [limit]                       List rollback plans
    show <rollback_id>                             Show rollback plan details

EMERGENCY COMMANDS:
    emergency <issue> <description> [severity] [fix_commit]
                                            Create emergency release

MONITORING COMMANDS:
    health-check                                 Run system health checks
    verify <rollback_id>                          Verify rollback completion

EXAMPLES:
    $0 plan rel-123 "Critical bug in production" manual
    $0 execute plan-456
    $0 execute plan-456 force
    $0 list completed 10
    $0 show plan-456
    $0 emergency PROD-123 "Database connection failed" critical fix-abc123
    $0 health-check

CONFIGURATION:
    Rollback plans: ${PROJECT_ROOT}/data/rollback_plans.json
    Emergency releases: ${PROJECT_ROOT}/data/emergency_releases.json
    Slack webhook: ${SLACK_WEBHOOK:-"not set"}
    PagerDuty key: ${PAGERDUTY_KEY:-"not set"}

EOF
}

# Main execution
main() {
    case "${1:-}" in
        "plan")
            if [ $# -lt 3 ]; then
                error "Usage: $0 plan <release_id> <reason> [trigger]"
                exit 1
            fi
            create_rollback_plan "$2" "$3" "${4:-manual}"
            ;;
        "execute")
            if [ $# -lt 2 ]; then
                error "Usage: $0 execute <rollback_id> [force]"
                exit 1
            fi
            execute_rollback "$2" "${3:-false}"
            ;;
        "list")
            list_rollback_plans "${2:-}" "${3:-20}"
            ;;
        "show")
            if [ $# -lt 2 ]; then
                error "Usage: $0 show <rollback_id>"
                exit 1
            fi
            show_rollback_plan "$2"
            ;;
        "emergency")
            if [ $# -lt 2 ]; then
                error "Usage: $0 emergency <issue> <description> [severity] [fix_commit]"
                exit 1
            fi
            create_emergency_release "$2" "$3" "${4:-critical}" "${5:-}"
            ;;
        "health-check")
            run_safety_checks
            ;;
        "verify")
            if [ $# -lt 2 ]; then
                error "Usage: $0 verify <rollback_id>"
                exit 1
            fi
            
            local plans_file="${PROJECT_ROOT}/data/rollback_plans.json"
            local plan
            plan=$(jq ".[] | select(.id == \"$2\")" "$plans_file")
            
            if [ -n "$plan" ]; then
                verify_rollback "$plan"
            else
                error "Rollback plan not found: $2"
            fi
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"