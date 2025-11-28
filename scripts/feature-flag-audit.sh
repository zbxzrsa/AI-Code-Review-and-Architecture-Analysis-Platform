#!/bin/bash

# Feature Flag Audit and Change Tracking Script
# Provides comprehensive audit logging and change tracking for feature flags

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-feature_flags}"
ENVIRONMENT="${ENVIRONMENT:-development}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] AUDIT:${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] AUDIT WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] AUDIT ERROR:${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] AUDIT SUCCESS:${NC} $1"
}

# Log audit event
log_audit_event() {
    local user_id="$1"
    local action="$2"
    local resource_type="$3"
    local resource_id="$4"
    local old_value="$5"
    local new_value="$6"
    local metadata="$7"
    local ip_address="${8:-127.0.0.1}"
    local user_agent="${9:-Feature Flag CLI}"
    
    local event_id
    event_id=$(uuidgen 2>/dev/null || python3 -c "import uuid; print(uuid.uuid4())")
    
    local timestamp
    timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    
    # Create audit event
    local audit_event="{
        \"id\": \"$event_id\",
        \"timestamp\": \"$timestamp\",
        \"user_id\": \"$user_id\",
        \"action\": \"$action\",
        \"resource_type\": \"$resource_type\",
        \"resource_id\": \"$resource_id\",
        \"old_value\": $old_value,
        \"new_value\": $new_value,
        \"ip_address\": \"$ip_address\",
        \"user_agent\": \"$user_agent\",
        \"metadata\": $metadata
    }"
    
    # Store in database
    local sql="
    INSERT INTO audit_events (id, timestamp, user_id, action, resource_type, resource_id, old_value, new_value, ip_address, user_agent, metadata)
    VALUES ('$event_id', '$timestamp', '$user_id', '$action', '$resource_type', '$resource_id', '$old_value'::jsonb, '$new_value'::jsonb, '$ip_address', '$user_agent', '$metadata'::jsonb);
    "
    
    echo "$sql" | PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" > /dev/null
    
    # Store in Redis stream for real-time monitoring
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" xadd audit_log "*" \
        "event_id" "$event_id" \
        "timestamp" "$timestamp" \
        "user_id" "$user_id" \
        "action" "$action" \
        "resource_type" "$resource_type" \
        "resource_id" "$resource_id" \
        "audit_event" "$audit_event" > /dev/null
    
    # Log to file for backup
    echo "$audit_event" >> "${PROJECT_ROOT}/logs/audit_events.log" 2>/dev/null || true
    
    # Publish to Redis pub/sub for real-time notifications
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" publish "audit:event" "$audit_event" > /dev/null
    
    success "Audit event logged: $action on $resource_type:$resource_id by $user_id"
}

# Get audit events
get_audit_events() {
    local resource_type="${1:-}"
    local resource_id="${2:-}"
    local user_id="${3:-}"
    local limit="${4:-50}"
    local hours="${5:-24}"
    
    log "Retrieving audit events..."
    
    local where_clauses=()
    
    if [ -n "$resource_type" ]; then
        where_clauses+=("resource_type = '$resource_type'")
    fi
    
    if [ -n "$resource_id" ]; then
        where_clauses+=("resource_id = '$resource_id'")
    fi
    
    if [ -n "$user_id" ]; then
        where_clauses+=("user_id = '$user_id'")
    fi
    
    # Add time filter
    where_clauses+=("timestamp >= NOW() - INTERVAL '$hours hours'")
    
    local where_clause=""
    if [ ${#where_clauses[@]} -gt 0 ]; then
        where_clause="WHERE ${where_clauses[*]}"
    fi
    
    local sql="
    SELECT 
        id,
        timestamp,
        user_id,
        action,
        resource_type,
        resource_id,
        old_value,
        new_value,
        ip_address,
        user_agent,
        metadata
    FROM audit_events 
    $where_clause
    ORDER BY timestamp DESC 
    LIMIT $limit;
    "
    
    PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" -c "$sql"
}

# Get audit summary
get_audit_summary() {
    local hours="${1:-24}"
    
    log "Generating audit summary (last $hours hours)..."
    
    local sql="
    SELECT 
        action,
        resource_type,
        COUNT(*) as count,
        COUNT(DISTINCT user_id) as unique_users,
        MIN(timestamp) as first_occurrence,
        MAX(timestamp) as last_occurrence
    FROM audit_events 
    WHERE timestamp >= NOW() - INTERVAL '$hours hours'
    GROUP BY action, resource_type
    ORDER BY count DESC;
    "
    
    PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" -c "$sql"
}

# Track flag change
track_flag_change() {
    local flag_key="$1"
    local action="$2"
    local user_id="$3"
    local old_value="$4"
    local new_value="$5"
    local reason="${6:-}"
    
    # Get current flag state for old_value if not provided
    if [ -z "$old_value" ]; then
        old_value=$(PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" -t -c "SELECT row_to_json(feature_flags) FROM feature_flags WHERE key = '$flag_key';" | tr -d ' ')
    fi
    
    local metadata="{\"environment\": \"$ENVIRONMENT\", \"reason\": \"$reason\"}"
    
    log_audit_event "$user_id" "$action" "feature_flag" "$flag_key" "$old_value" "$new_value" "$metadata"
}

# Track kill switch activation
track_kill_switch() {
    local flag_key="$1"
    local action="$2"  # activate or deactivate
    local user_id="$3"
    local reason="$4"
    
    local metadata="{\"environment\": \"$ENVIRONMENT\", \"reason\": \"$reason\", \"emergency_action\": true}"
    local event_value="{\"action\": \"$action\", \"reason\": \"$reason\"}"
    
    log_audit_event "$user_id" "kill_switch_$action" "feature_flag" "$flag_key" "" "$event_value" "$metadata"
}

# Track user access
track_user_access() {
    local user_id="$1"
    local resource_type="$2"
    local resource_id="$3"
    local action="${4:-access}"
    local ip_address="${5:-127.0.0.1}"
    local user_agent="${6:-}"
    
    local metadata="{\"environment\": \"$ENVIRONMENT\"}"
    local event_value="{\"access_granted\": true}"
    
    log_audit_event "$user_id" "$action" "$resource_type" "$resource_id" "" "$event_value" "$metadata" "$ip_address" "$user_agent"
}

# Track security event
track_security_event() {
    local event_type="$1"
    local user_id="$2"
    local description="$3"
    local severity="${4:-medium}"  # low, medium, high, critical
    local ip_address="${5:-127.0.0.1}"
    local user_agent="${6:-}"
    
    local metadata="{\"environment\": \"$ENVIRONMENT\", \"severity\": \"$severity\", \"security_event\": true}"
    local event_value="{\"event_type\": \"$event_type\", \"description\": \"$description\", \"severity\": \"$severity\"}"
    
    log_audit_event "$user_id" "security_event" "system" "$event_type" "" "$event_value" "$metadata" "$ip_address" "$user_agent"
}

# Generate audit report
generate_audit_report() {
    local start_date="${1:-$(date -d '30 days ago' -I)}"
    local end_date="${2:-$(date -I)}"
    local format="${3:-text}"  # text, json, csv
    
    log "Generating audit report from $start_date to $end_date..."
    
    local sql="
    SELECT 
        timestamp,
        user_id,
        action,
        resource_type,
        resource_id,
        CASE 
            WHEN old_value IS NOT NULL THEN 'changed'
            ELSE 'created'
        END as change_type,
        ip_address
    FROM audit_events 
    WHERE timestamp >= '$start_date' 
    AND timestamp <= '$end_date'
    ORDER BY timestamp DESC;
    "
    
    case "$format" in
        "json")
            PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" -c "SELECT json_agg(row_to_json(t)) FROM ($sql) t;" | jq -r '.'
            ;;
        "csv")
            PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" -c "COPY ($sql) TO STDOUT WITH CSV HEADER"
            ;;
        *)
            PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" -c "$sql"
            ;;
    esac
}

# Analyze audit patterns
analyze_audit_patterns() {
    local days="${1:-7}"
    
    log "Analyzing audit patterns (last $days days)..."
    
    echo "=== USER ACTIVITY ANALYSIS ==="
    local user_sql="
    SELECT 
        user_id,
        COUNT(*) as total_actions,
        COUNT(DISTINCT action) as unique_actions,
        COUNT(DISTINCT resource_type) as resource_types,
        MIN(timestamp) as first_activity,
        MAX(timestamp) as last_activity
    FROM audit_events 
    WHERE timestamp >= NOW() - INTERVAL '$days days'
    GROUP BY user_id
    ORDER BY total_actions DESC
    LIMIT 10;
    "
    
    PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" -c "$user_sql"
    
    echo ""
    echo "=== RESOURCE ACCESS PATTERNS ==="
    local resource_sql="
    SELECT 
        resource_type,
        resource_id,
        COUNT(*) as access_count,
        COUNT(DISTINCT user_id) as unique_users,
        COUNT(DISTINCT action) as action_types
    FROM audit_events 
    WHERE timestamp >= NOW() - INTERVAL '$days days'
    GROUP BY resource_type, resource_id
    ORDER BY access_count DESC
    LIMIT 10;
    "
    
    PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" -c "$resource_sql"
    
    echo ""
    echo "=== ACTION FREQUENCY ==="
    local action_sql="
    SELECT 
        action,
        COUNT(*) as count,
        COUNT(DISTINCT user_id) as unique_users,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
    FROM audit_events 
    WHERE timestamp >= NOW() - INTERVAL '$days days'
    GROUP BY action
    ORDER BY count DESC;
    "
    
    PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" -c "$action_sql"
}

# Export audit data
export_audit_data() {
    local output_file="${1:-audit_export_$(date +%Y%m%d_%H%M%S).json}"
    local format="${2:-json}"
    local start_date="${3:-$(date -d '30 days ago' -I)}"
    local end_date="${4:-$(date -I)}"
    
    log "Exporting audit data to $output_file..."
    
    case "$format" in
        "json")
            generate_audit_report "$start_date" "$end_date" "json" > "$output_file"
            ;;
        "csv")
            generate_audit_report "$start_date" "$end_date" "csv" > "$output_file"
            ;;
        *)
            error "Unsupported format: $format"
            return 1
            ;;
    esac
    
    success "Audit data exported to $output_file"
}

# Cleanup old audit events
cleanup_old_events() {
    local days="${1:-90}"  # Keep events for 90 days by default
    
    log "Cleaning up audit events older than $days days..."
    
    local sql="
    DELETE FROM audit_events 
    WHERE timestamp < NOW() - INTERVAL '$days days';
    
    SELECT COUNT(*) as deleted_count FROM audit_events 
    WHERE timestamp < NOW() - INTERVAL '$days days';
    "
    
    PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" -c "$sql"
    
    success "Old audit events cleaned up"
}

# Monitor audit events in real-time
monitor_audit_events() {
    log "Starting real-time audit event monitoring..."
    
    # Create Redis consumer group if it doesn't exist
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" xgroup create audit_log audit_monitor "$" MKSTREAM > /dev/null 2>&1 || true
    
    while true; do
        # Read new events from Redis stream
        local events
        events=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" xreadgroup GROUP audit_monitor consumer COUNT 1 BLOCK 1000 STREAMS audit_log ">")
        
        if [ "$events" != "" ]; then
            echo "$events" | jq -r '.[] | .[] | .[] | select(.event_id) | "\(.timestamp): \(.user_id) performed \(.action) on \(.resource_type):\(.resource_id)"'
            
            # Acknowledge the event
            local event_id
            event_id=$(echo "$events" | jq -r '.[] | .[] | .[] | select(.event_id) | .event_id')
            redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" xack audit_log audit_monitor "$event_id" > /dev/null
        fi
    done
}

# Show usage
show_usage() {
    cat << EOF
Feature Flag Audit and Change Tracking Script

USAGE:
    $0 <command> [options]

AUDIT COMMANDS:
    log <user_id> <action> <resource_type> <resource_id> [old_value] [new_value] [metadata] [ip] [user_agent]
    get [resource_type] [resource_id] [user_id] [limit] [hours]
    summary [hours]
    track-flag <flag_key> <action> <user_id> [old_value] [new_value] [reason]
    track-kill-switch <flag_key> <action> <user_id> <reason>
    track-access <user_id> <resource_type> <resource_id> [action] [ip] [user_agent]
    track-security <event_type> <user_id> <description> [severity] [ip] [user_agent]
    
REPORTING COMMANDS:
    report [start_date] [end_date] [format]
    analyze [days]
    export <output_file> [format] [start_date] [end_date]
    cleanup [days]
    monitor

EXAMPLES:
    $0 log admin@company.com flag_updated feature_flag new_dashboard '{"enabled":false}' '{"enabled":true}' '{"reason":"A/B test"}'
    $0 get feature_flag new_dashboard 100 24
    $0 track-flag new_dashboard updated admin@company.com '{"enabled":false}' '{"enabled":true}' "Enable for testing"
    $0 track-kill-switch new_dashboard activate ops@company.com "Performance issues"
    $0 report 2024-01-01 2024-01-31 json
    $0 analyze 7
    $0 export audit_export.json json
    $0 cleanup 90
    $0 monitor

ENVIRONMENT VARIABLES:
    REDIS_HOST              Redis host (default: localhost)
    REDIS_PORT              Redis port (default: 6379)
    DB_HOST                 Database host (default: localhost)
    DB_PORT                 Database port (default: 5432)
    DB_NAME                 Database name (default: feature_flags)
    DB_USER                 Database user (default: postgres)
    DB_PASSWORD             Database password
    ENVIRONMENT             Environment (default: development)

EOF
}

# Main execution
main() {
    case "${1:-}" in
        "log")
            if [ $# -lt 5 ]; then
                error "Usage: $0 log <user_id> <action> <resource_type> <resource_id> [old_value] [new_value] [metadata] [ip] [user_agent]"
                exit 1
            fi
            log_audit_event "$2" "$3" "$4" "$5" "${6:-null}" "${7:-null}" "${8:-null}" "${9:-127.0.0.1}" "${10:-Feature Flag CLI}"
            ;;
        "get")
            get_audit_events "${2:-}" "${3:-}" "${4:-}" "${5:-50}" "${6:-24}"
            ;;
        "summary")
            get_audit_summary "${2:-24}"
            ;;
        "track-flag")
            if [ $# -lt 4 ]; then
                error "Usage: $0 track-flag <flag_key> <action> <user_id> [old_value] [new_value] [reason]"
                exit 1
            fi
            track_flag_change "$2" "$3" "$4" "${5:-}" "${6:-}" "${7:-}"
            ;;
        "track-kill-switch")
            if [ $# -lt 4 ]; then
                error "Usage: $0 track-kill-switch <flag_key> <action> <user_id> <reason>"
                exit 1
            fi
            track_kill_switch "$2" "$3" "$4" "$5"
            ;;
        "track-access")
            if [ $# -lt 4 ]; then
                error "Usage: $0 track-access <user_id> <resource_type> <resource_id> [action] [ip] [user_agent]"
                exit 1
            fi
            track_user_access "$2" "$3" "$4" "${5:-access}" "${6:-127.0.0.1}" "${7:-}"
            ;;
        "track-security")
            if [ $# -lt 4 ]; then
                error "Usage: $0 track-security <event_type> <user_id> <description> [severity] [ip] [user_agent]"
                exit 1
            fi
            track_security_event "$2" "$3" "$4" "${5:-medium}" "${6:-127.0.0.1}" "${7:-}"
            ;;
        "report")
            generate_audit_report "${2:-$(date -d '30 days ago' -I)}" "${3:-$(date -I)}" "${4:-text}"
            ;;
        "analyze")
            analyze_audit_patterns "${2:-7}"
            ;;
        "export")
            if [ $# -lt 2 ]; then
                error "Usage: $0 export <output_file> [format] [start_date] [end_date]"
                exit 1
            fi
            export_audit_data "$2" "${3:-json}" "${4:-$(date -d '30 days ago' -I)}" "${5:-$(date -I)}"
            ;;
        "cleanup")
            cleanup_old_events "${2:-90}"
            ;;
        "monitor")
            monitor_audit_events
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"