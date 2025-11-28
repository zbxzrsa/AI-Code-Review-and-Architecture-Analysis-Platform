#!/bin/bash

# Feature Flags & Kill Switch Management Script
# Provides comprehensive feature flag management with runtime toggle capabilities

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_ROOT}/config/feature-flags.yml"
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
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] FEATURE-FLAGS:${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] FEATURE-FLAGS WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] FEATURE-FLAGS ERROR:${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] FEATURE-FLAGS SUCCESS:${NC} $1"
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    local missing_deps=()
    
    if ! command -v redis-cli &> /dev/null; then
        missing_deps+=("redis-cli")
    fi
    
    if ! command -v psql &> /dev/null; then
        missing_deps+=("psql")
    fi
    
    if ! command -v jq &> /dev/null; then
        missing_deps+=("jq")
    fi
    
    if ! command -v curl &> /dev/null; then
        missing_deps+=("curl")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        error "Missing dependencies: ${missing_deps[*]}"
        error "Please install the missing dependencies and try again."
        exit 1
    fi
    
    success "All dependencies are available"
}

# Check Redis connection
check_redis() {
    log "Checking Redis connection..."
    
    if ! redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping > /dev/null 2>&1; then
        error "Cannot connect to Redis at $REDIS_HOST:$REDIS_PORT"
        return 1
    fi
    
    success "Redis connection successful"
}

# Check database connection
check_database() {
    log "Checking database connection..."
    
    if ! PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" -c "SELECT 1;" > /dev/null 2>&1; then
        error "Cannot connect to database at $DB_HOST:$DB_PORT"
        return 1
    fi
    
    success "Database connection successful"
}

# Initialize database schema
init_database() {
    log "Initializing database schema..."
    
    # Create tables
    local sql="
    CREATE TABLE IF NOT EXISTS feature_flags (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        key VARCHAR(255) UNIQUE NOT NULL,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        type VARCHAR(50) NOT NULL DEFAULT 'boolean',
        enabled BOOLEAN NOT NULL DEFAULT false,
        value JSONB,
        rules JSONB DEFAULT '[]',
        targeting JSONB DEFAULT '[]',
        metadata JSONB DEFAULT '{}',
        rollout JSONB DEFAULT '{}',
        kill_switch JSONB DEFAULT '{}',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        created_by VARCHAR(255),
        updated_by VARCHAR(255)
    );
    
    CREATE TABLE IF NOT EXISTS flag_evaluations (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        flag_key VARCHAR(255) NOT NULL,
        context JSONB NOT NULL,
        result JSONB NOT NULL,
        evaluation_time_ms INTEGER,
        cache_hit BOOLEAN DEFAULT false,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        user_id VARCHAR(255),
        ip_address INET
    );
    
    CREATE TABLE IF NOT EXISTS audit_events (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        user_id VARCHAR(255) NOT NULL,
        action VARCHAR(100) NOT NULL,
        resource_type VARCHAR(100) NOT NULL,
        resource_id VARCHAR(255) NOT NULL,
        old_value JSONB,
        new_value JSONB,
        ip_address INET,
        user_agent TEXT,
        metadata JSONB DEFAULT '{}'
    );
    
    CREATE INDEX IF NOT EXISTS idx_feature_flags_key ON feature_flags(key);
    CREATE INDEX IF NOT EXISTS idx_feature_flags_enabled ON feature_flags(enabled);
    CREATE INDEX IF NOT EXISTS idx_flag_evaluations_flag_key ON flag_evaluations(flag_key);
    CREATE INDEX IF NOT EXISTS idx_flag_evaluations_created_at ON flag_evaluations(created_at);
    CREATE INDEX IF NOT EXISTS idx_audit_events_resource ON audit_events(resource_type, resource_id);
    CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp);
    "
    
    echo "$sql" | PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME"
    
    success "Database schema initialized"
}

# Create feature flag
create_flag() {
    local flag_key="$1"
    local flag_name="$2"
    local flag_description="${3:-}"
    local flag_type="${4:-boolean}"
    local flag_value="${5:-false}"
    local created_by="${6:-system}"
    
    log "Creating feature flag: $flag_key"
    
    # Check if flag already exists
    local existing_flag
    existing_flag=$(PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" -t -c "SELECT id FROM feature_flags WHERE key = '$flag_key';" | tr -d ' ')
    
    if [ -n "$existing_flag" ]; then
        error "Feature flag '$flag_key' already exists"
        return 1
    fi
    
    # Insert new flag
    local sql="
    INSERT INTO feature_flags (key, name, description, type, enabled, value, created_by, metadata)
    VALUES ('$flag_key', '$flag_name', '$flag_description', '$flag_type', false, '$flag_value', '$created_by', '{\"environment\": \"$ENVIRONMENT\"}')
    RETURNING id;
    "
    
    local flag_id
    flag_id=$(echo "$sql" | PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" -t -c - | tr -d ' ')
    
    # Cache in Redis
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" setex "flag:$flag_key" 300 "$flag_value"
    
    # Log audit event
    log_audit_event "$created_by" "flag_created" "feature_flag" "$flag_key" "" "$flag_value"
    
    success "Feature flag '$flag_key' created with ID: $flag_id"
}

# Update feature flag
update_flag() {
    local flag_key="$1"
    local updates="$2"
    local updated_by="${3:-system}"
    
    log "Updating feature flag: $flag_key"
    
    # Get current flag
    local current_flag
    current_flag=$(PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" -t -c "SELECT row_to_json(feature_flags) FROM feature_flags WHERE key = '$flag_key';" | tr -d ' ')
    
    if [ -z "$current_flag" ]; then
        error "Feature flag '$flag_key' not found"
        return 1
    fi
    
    # Parse updates (simple key=value format)
    local update_clauses=""
    IFS=',' read -ra update_array <<< "$updates"
    for update in "${update_array[@]}"; do
        local key="${update%%=*}"
        local value="${update#*=}"
        
        if [ -n "$update_clauses" ]; then
            update_clauses+=", "
        fi
        
        case "$key" in
            "enabled")
                update_clauses+="enabled = $value"
                ;;
            "value")
                update_clauses+="value = '$value'"
                ;;
            "name")
                update_clauses+="name = '$value'"
                ;;
            "description")
                update_clauses+="description = '$value'"
                ;;
            *)
                warn "Unknown update field: $key"
                ;;
        esac
    done
    
    if [ -z "$update_clauses" ]; then
        error "No valid update fields provided"
        return 1
    fi
    
    # Update flag
    local sql="
    UPDATE feature_flags 
    SET $update_clauses, updated_by = '$updated_by', updated_at = NOW()
    WHERE key = '$flag_key'
    RETURNING row_to_json(feature_flags);
    "
    
    local updated_flag
    updated_flag=$(echo "$sql" | PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" -t -c - | tr -d ' ')
    
    # Invalidate cache
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" del "flag:$flag_key"
    
    # Log audit event
    log_audit_event "$updated_by" "flag_updated" "feature_flag" "$flag_key" "$current_flag" "$updated_flag"
    
    success "Feature flag '$flag_key' updated"
}

# Delete feature flag
delete_flag() {
    local flag_key="$1"
    local deleted_by="${2:-system}"
    
    log "Deleting feature flag: $flag_key"
    
    # Get current flag
    local current_flag
    current_flag=$(PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" -t -c "SELECT row_to_json(feature_flags) FROM feature_flags WHERE key = '$flag_key';" | tr -d ' ')
    
    if [ -z "$current_flag" ]; then
        error "Feature flag '$flag_key' not found"
        return 1
    fi
    
    # Delete flag
    local sql="DELETE FROM feature_flags WHERE key = '$flag_key';"
    echo "$sql" | PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME"
    
    # Remove from cache
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" del "flag:$flag_key"
    
    # Log audit event
    log_audit_event "$deleted_by" "flag_deleted" "feature_flag" "$flag_key" "$current_flag" ""
    
    success "Feature flag '$flag_key' deleted"
}

# Get feature flag
get_flag() {
    local flag_key="$1"
    
    log "Getting feature flag: $flag_key"
    
    # Try cache first
    local cached_value
    cached_value=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" get "flag:$flag_key" 2>/dev/null || echo "")
    
    if [ -n "$cached_value" ]; then
        echo "$cached_value"
        return 0
    fi
    
    # Get from database
    local flag
    flag=$(PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" -t -c "SELECT row_to_json(feature_flags) FROM feature_flags WHERE key = '$flag_key';" | tr -d ' ')
    
    if [ -z "$flag" ]; then
        error "Feature flag '$flag_key' not found"
        return 1
    fi
    
    # Extract value and cache it
    local value
    value=$(echo "$flag" | jq -r '.value // false')
    
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" setex "flag:$flag_key" 300 "$value"
    
    echo "$value"
}

# List all feature flags
list_flags() {
    log "Listing all feature flags"
    
    local sql="
    SELECT 
        key,
        name,
        type,
        enabled,
        created_at,
        updated_by
    FROM feature_flags 
    ORDER BY created_at DESC;
    "
    
    PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" -c "$sql"
}

# Activate kill switch
activate_kill_switch() {
    local flag_key="$1"
    local reason="$2"
    local triggered_by="${3:-system}"
    
    log "Activating kill switch for flag: $flag_key"
    
    # Check if flag exists
    local flag_exists
    flag_exists=$(PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" -t -c "SELECT id FROM feature_flags WHERE key = '$flag_key';" | tr -d ' ')
    
    if [ -z "$flag_exists" ]; then
        error "Feature flag '$flag_key' not found"
        return 1
    fi
    
    # Update kill switch
    local sql="
    UPDATE feature_flags 
    SET 
        kill_switch = jsonb_set(
            jsonb_set(
                jsonb_set(
                    kill_switch,
                    '{enabled}',
                    'true'
                ),
                '{reason}',
                    '\"$reason\"'
                ),
            '{triggered_by}',
            '\"$triggered_by\"'
        ),
        kill_switch = jsonb_set(kill_switch, '{triggered_at}', '\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"'),
        updated_by = '$triggered_by',
        updated_at = NOW()
    WHERE key = '$flag_key';
    "
    
    echo "$sql" | PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME"
    
    # Invalidate all caches
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" del "flag:$flag_key"
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" del "kill_switch:$flag_key"
    
    # Set kill switch in Redis
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" setex "kill_switch:$flag_key" 86400 "true"
    
    # Broadcast kill switch activation
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" publish "kill_switch:activated" "{\"flag_key\":\"$flag_key\",\"reason\":\"$reason\",\"triggered_by\":\"$triggered_by\"}"
    
    # Log audit event
    log_audit_event "$triggered_by" "kill_switch_activated" "feature_flag" "$flag_key" "" "{\"reason\":\"$reason\"}"
    
    # Send emergency alert
    send_emergency_alert "$flag_key" "$reason" "$triggered_by"
    
    success "Kill switch activated for '$flag_key'"
}

# Deactivate kill switch
deactivate_kill_switch() {
    local flag_key="$1"
    local triggered_by="${2:-system}"
    
    log "Deactivating kill switch for flag: $flag_key"
    
    # Update kill switch
    local sql="
    UPDATE feature_flags 
    SET 
        kill_switch = '{}',
        updated_by = '$triggered_by',
        updated_at = NOW()
    WHERE key = '$flag_key';
    "
    
    echo "$sql" | PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME"
    
    # Remove kill switch from Redis
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" del "kill_switch:$flag_key"
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" del "flag:$flag_key"
    
    # Broadcast kill switch deactivation
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" publish "kill_switch:deactivated" "{\"flag_key\":\"$flag_key\",\"triggered_by\":\"$triggered_by\"}"
    
    # Log audit event
    log_audit_event "$triggered_by" "kill_switch_deactivated" "feature_flag" "$flag_key" "" ""
    
    success "Kill switch deactivated for '$flag_key'"
}

# Check kill switch status
check_kill_switch() {
    local flag_key="$1"
    
    # Check Redis first
    local kill_switch_active
    kill_switch_active=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" get "kill_switch:$flag_key" 2>/dev/null || echo "")
    
    if [ "$kill_switch_active" = "true" ]; then
        echo "ACTIVE"
        return 0
    fi
    
    # Check database
    local sql="
    SELECT kill_switch->>'enabled' as enabled
    FROM feature_flags 
    WHERE key = '$flag_key';
    "
    
    local db_result
    db_result=$(PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" -t -c "$sql" | tr -d ' ')
    
    if [ "$db_result" = "true" ]; then
        echo "ACTIVE"
        # Update cache
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" setex "kill_switch:$flag_key" 86400 "true"
    else
        echo "INACTIVE"
    fi
}

# Evaluate feature flag
evaluate_flag() {
    local flag_key="$1"
    local user_id="${2:-anonymous}"
    local context="${3:-{}}"
    
    log "Evaluating flag '$flag_key' for user '$user_id'"
    
    # Check kill switch first
    local kill_switch_status
    kill_switch_status=$(check_kill_switch "$flag_key")
    
    if [ "$kill_switch_status" = "ACTIVE" ]; then
        echo "{\"enabled\":false,\"reason\":\"kill_switch_active\"}"
        return 0
    fi
    
    # Get flag value
    local flag_value
    flag_value=$(get_flag "$flag_key")
    
    if [ $? -ne 0 ]; then
        echo "{\"enabled\":false,\"reason\":\"flag_not_found\"}"
        return 1
    fi
    
    # Simple evaluation (can be extended with complex rules)
    local result="{\"enabled\":$flag_value,\"reason\":\"evaluation\"}"
    
    # Log evaluation
    log_evaluation "$flag_key" "$user_id" "$result" "$context"
    
    echo "$result"
}

# Log audit event
log_audit_event() {
    local user_id="$1"
    local action="$2"
    local resource_type="$3"
    local resource_id="$4"
    local old_value="$5"
    local new_value="$6"
    
    local sql="
    INSERT INTO audit_events (user_id, action, resource_type, resource_id, old_value, new_value, metadata)
    VALUES ('$user_id', '$action', '$resource_type', '$resource_id', '$old_value'::jsonb, '$new_value'::jsonb, '{\"environment\": \"$ENVIRONMENT\"}');
    "
    
    echo "$sql" | PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" > /dev/null
}

# Log evaluation
log_evaluation() {
    local flag_key="$1"
    local user_id="$2"
    local result="$3"
    local context="$4"
    
    local sql="
    INSERT INTO flag_evaluations (flag_key, context, result, user_id, ip_address)
    VALUES ('$flag_key', '$context'::jsonb, '$result'::jsonb, '$user_id', '127.0.0.1');
    "
    
    echo "$sql" | PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" > /dev/null
}

# Send emergency alert
send_emergency_alert() {
    local flag_key="$1"
    local reason="$2"
    local triggered_by="$3"
    
    local message="🚨 KILL SWITCH ACTIVATED 🚨
    
Flag: $flag_key
Reason: $reason
Triggered by: $triggered_by
Time: $(date -u)
Environment: $ENVIRONMENT"

    # Send to Slack (if webhook is configured)
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" \
            "$SLACK_WEBHOOK_URL" > /dev/null 2>&1 || true
    fi
    
    # Send email (if configured)
    if [ -n "${ALERT_EMAIL:-}" ]; then
        echo "$message" | mail -s "Emergency: Kill Switch Activated for $flag_key" "$ALERT_EMAIL" || true
    fi
    
    log "Emergency alert sent for kill switch activation"
}

# Get flag statistics
get_flag_stats() {
    local flag_key="$1"
    local hours="${2:-24}"
    
    log "Getting statistics for flag: $flag_key (last $hours hours)"
    
    local sql="
    SELECT 
        COUNT(*) as total_evaluations,
        COUNT(*) FILTER (WHERE result->>'enabled' = 'true') as enabled_count,
        COUNT(*) FILTER (WHERE result->>'enabled' = 'false') as disabled_count,
        AVG(evaluation_time_ms) as avg_evaluation_time,
        COUNT(*) FILTER (WHERE cache_hit = true) as cache_hits,
        COUNT(*) FILTER (WHERE cache_hit = false) as cache_misses
    FROM flag_evaluations 
    WHERE flag_key = '$flag_key' 
    AND created_at >= NOW() - INTERVAL '$hours hours';
    "
    
    PGPASSWORD="${DB_PASSWORD:-}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:-postgres}" -d "$DB_NAME" -c "$sql"
}

# Health check
health_check() {
    log "Performing health check..."
    
    local status="healthy"
    local issues=()
    
    # Check Redis
    if ! check_redis; then
        status="unhealthy"
        issues+=("Redis connection failed")
    fi
    
    # Check database
    if ! check_database; then
        status="unhealthy"
        issues+=("Database connection failed")
    fi
    
    # Check cache performance
    local cache_test
    cache_test=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" setex "health_check" 10 "test" 2>/dev/null && echo "OK" || echo "FAIL")
    if [ "$cache_test" != "OK" ]; then
        status="degraded"
        issues+=("Cache performance degraded")
    fi
    
    echo "Status: $status"
    if [ ${#issues[@]} -gt 0 ]; then
        echo "Issues:"
        printf ' - %s\n' "${issues[@]}"
    fi
    
    return 0
}

# Show usage
show_usage() {
    cat << EOF
Feature Flags & Kill Switch Management Script

USAGE:
    $0 <command> [options]

COMMANDS:
    init                    Initialize database and dependencies
    create <key> <name> [description] [type] [value] [created_by]
    update <key> <updates>  Update flag (key=value,key=value format)
    delete <key> [deleted_by]
    get <key>               Get flag value
    list                    List all flags
    evaluate <key> [user_id] [context]
    kill-switch-activate <key> <reason> [triggered_by]
    kill-switch-deactivate <key> [triggered_by]
    kill-switch-status <key>
    stats <key> [hours]     Get flag statistics
    health-check            Perform health check

EXAMPLES:
    $0 init
    $0 create new_dashboard "New Dashboard UI" "Enable new dashboard interface" boolean true admin@company.com
    $0 update new_dashboard "enabled=true,value=true"
    $0 evaluate new_dashboard user123 '{"tier":"premium"}'
    $0 kill-switch-activate new_dashboard "Performance issues detected" ops@company.com
    $0 stats new_dashboard 24

ENVIRONMENT VARIABLES:
    REDIS_HOST              Redis host (default: localhost)
    REDIS_PORT              Redis port (default: 6379)
    DB_HOST                 Database host (default: localhost)
    DB_PORT                 Database port (default: 5432)
    DB_NAME                 Database name (default: feature_flags)
    DB_USER                 Database user (default: postgres)
    DB_PASSWORD             Database password
    ENVIRONMENT             Environment (default: development)
    SLACK_WEBHOOK_URL       Slack webhook for alerts
    ALERT_EMAIL             Email address for alerts

EOF
}

# Main execution
main() {
    case "${1:-}" in
        "init")
            check_dependencies
            check_redis
            check_database
            init_database
            ;;
        "create")
            if [ $# -lt 3 ]; then
                error "Usage: $0 create <key> <name> [description] [type] [value] [created_by]"
                exit 1
            fi
            check_dependencies
            check_redis
            check_database
            create_flag "$2" "$3" "${4:-}" "${5:-boolean}" "${6:-false}" "${7:-system}"
            ;;
        "update")
            if [ $# -lt 3 ]; then
                error "Usage: $0 update <key> <updates>"
                exit 1
            fi
            check_dependencies
            check_redis
            check_database
            update_flag "$2" "$3" "${4:-system}"
            ;;
        "delete")
            if [ $# -lt 2 ]; then
                error "Usage: $0 delete <key> [deleted_by]"
                exit 1
            fi
            check_dependencies
            check_redis
            check_database
            delete_flag "$2" "${3:-system}"
            ;;
        "get")
            if [ $# -lt 2 ]; then
                error "Usage: $0 get <key>"
                exit 1
            fi
            check_dependencies
            check_redis
            check_database
            get_flag "$2"
            ;;
        "list")
            check_dependencies
            check_database
            list_flags
            ;;
        "evaluate")
            if [ $# -lt 2 ]; then
                error "Usage: $0 evaluate <key> [user_id] [context]"
                exit 1
            fi
            check_dependencies
            check_redis
            check_database
            evaluate_flag "$2" "${3:-anonymous}" "${4:-{}}"
            ;;
        "kill-switch-activate")
            if [ $# -lt 3 ]; then
                error "Usage: $0 kill-switch-activate <key> <reason> [triggered_by]"
                exit 1
            fi
            check_dependencies
            check_redis
            check_database
            activate_kill_switch "$2" "$3" "${4:-system}"
            ;;
        "kill-switch-deactivate")
            if [ $# -lt 2 ]; then
                error "Usage: $0 kill-switch-deactivate <key> [triggered_by]"
                exit 1
            fi
            check_dependencies
            check_redis
            check_database
            deactivate_kill_switch "$2" "${3:-system}"
            ;;
        "kill-switch-status")
            if [ $# -lt 2 ]; then
                error "Usage: $0 kill-switch-status <key>"
                exit 1
            fi
            check_dependencies
            check_redis
            check_database
            check_kill_switch "$2"
            ;;
        "stats")
            if [ $# -lt 2 ]; then
                error "Usage: $0 stats <key> [hours]"
                exit 1
            fi
            check_dependencies
            check_database
            get_flag_stats "$2" "${3:-24}"
            ;;
        "health-check")
            check_dependencies
            health_check
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"