#!/bin/bash

# Kill Switch Emergency Response Script
# Provides immediate emergency response capabilities for critical feature failures

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_ROOT}/config/kill-switch.yml"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
ENVIRONMENT="${ENVIRONMENT:-development}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] KILL-SWITCH:${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] KILL-SWITCH WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] KILL-SWITCH ERROR:${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] KILL-SWITCH SUCCESS:${NC} $1"
}

# Emergency kill switch activation
emergency_activate() {
    local flag_key="$1"
    local reason="$2"
    local triggered_by="${3:-emergency_system}"
    local severity="${4:-critical}"
    
    log "🚨 EMERGENCY KILL SWITCH ACTIVATION 🚨"
    log "Flag: $flag_key"
    log "Reason: $reason"
    log "Severity: $severity"
    log "Triggered by: $triggered_by"
    
    # 1. Immediate Redis kill switch (fastest path)
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" setex "emergency_kill:$flag_key" 86400 "true"
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" setex "kill_switch:$flag_key" 86400 "true"
    
    # 2. Broadcast emergency signal
    local emergency_signal="{\"flag_key\":\"$flag_key\",\"reason\":\"$reason\",\"severity\":\"$severity\",\"triggered_by\":\"$triggered_by\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"type\":\"emergency\"}"
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" publish "emergency:kill_switch" "$emergency_signal"
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" publish "kill_switch:activated" "$emergency_signal"
    
    # 3. Update database (async, don't wait)
    (
        sleep 1
        "$SCRIPT_DIR/feature-flags.sh" kill-switch-activate "$flag_key" "$reason" "$triggered_by" > /dev/null 2>&1 || true
    ) &
    
    # 4. Send immediate alerts
    send_emergency_alert "$flag_key" "$reason" "$triggered_by" "$severity"
    
    # 5. Log emergency action
    log_emergency_action "$flag_key" "$reason" "$triggered_by" "$severity"
    
    success "Emergency kill switch activated for '$flag_key'"
}

# Emergency kill switch deactivation
emergency_deactivate() {
    local flag_key="$1"
    local triggered_by="${2:-emergency_system}"
    local recovery_reason="${3:-Issue resolved}"
    
    log "🔄 EMERGENCY KILL SWITCH DEACTIVATION 🔄"
    log "Flag: $flag_key"
    log "Recovery reason: $recovery_reason"
    log "Triggered by: $triggered_by"
    
    # 1. Remove emergency kill switch from Redis
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" del "emergency_kill:$flag_key"
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" del "kill_switch:$flag_key"
    
    # 2. Broadcast recovery signal
    local recovery_signal="{\"flag_key\":\"$flag_key\",\"reason\":\"$recovery_reason\",\"triggered_by\":\"$triggered_by\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"type\":\"recovery\"}"
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" publish "emergency:kill_switch_deactivated" "$recovery_signal"
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" publish "kill_switch:deactivated" "$recovery_signal"
    
    # 3. Update database (async)
    (
        sleep 1
        "$SCRIPT_DIR/feature-flags.sh" kill-switch-deactivate "$flag_key" "$triggered_by" > /dev/null 2>&1 || true
    ) &
    
    # 4. Send recovery notifications
    send_recovery_notification "$flag_key" "$recovery_reason" "$triggered_by"
    
    # 5. Log recovery action
    log_recovery_action "$flag_key" "$recovery_reason" "$triggered_by"
    
    success "Emergency kill switch deactivated for '$flag_key'"
}

# Global emergency stop (all flags)
global_emergency_stop() {
    local reason="$1"
    local triggered_by="${2:-emergency_system}"
    local severity="${3:-critical}"
    
    log "🚨 GLOBAL EMERGENCY STOP 🚨"
    log "Reason: $reason"
    log "Severity: $severity"
    log "Triggered by: $triggered_by"
    
    # 1. Set global emergency flag
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" setex "global_emergency_stop" 86400 "true"
    
    # 2. Get all active flags
    local active_flags
    active_flags=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" keys "flag:*" 2>/dev/null || echo "")
    
    # 3. Activate kill switch for all flags
    local activated_count=0
    for flag_key_full in $active_flags; do
        local flag_key="${flag_key_full#flag:}"
        emergency_activate "$flag_key" "Global emergency: $reason" "$triggered_by" "$severity"
        ((activated_count++))
    done
    
    # 4. Broadcast global emergency
    local global_signal="{\"reason\":\"$reason\",\"severity\":\"$severity\",\"triggered_by\":\"$triggered_by\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"flags_affected\":$activated_count,\"type\":\"global_emergency\"}"
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" publish "global:emergency_stop" "$global_signal"
    
    # 5. Send global emergency alerts
    send_global_emergency_alert "$reason" "$triggered_by" "$severity" "$activated_count"
    
    # 6. Log global emergency
    log_global_emergency "$reason" "$triggered_by" "$severity" "$activated_count"
    
    success "Global emergency stop activated. $activated_count flags affected."
}

# Check emergency status
check_emergency_status() {
    local flag_key="${1:-}"
    
    if [ -n "$flag_key" ]; then
        # Check specific flag
        local emergency_status
        emergency_status=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" get "emergency_kill:$flag_key" 2>/dev/null || echo "")
        
        if [ "$emergency_status" = "true" ]; then
            echo "EMERGENCY_ACTIVE"
            return 0
        fi
        
        # Check regular kill switch
        local kill_switch_status
        kill_switch_status=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" get "kill_switch:$flag_key" 2>/dev/null || echo "")
        
        if [ "$kill_switch_status" = "true" ]; then
            echo "KILL_SWITCH_ACTIVE"
            return 0
        fi
        
        echo "NORMAL"
        return 0
    else
        # Check global emergency status
        local global_status
        global_status=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" get "global_emergency_stop" 2>/dev/null || echo "")
        
        if [ "$global_status" = "true" ]; then
            echo "GLOBAL_EMERGENCY_ACTIVE"
            return 0
        fi
        
        # Count active emergency kill switches
        local emergency_count
        emergency_count=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" keys "emergency_kill:*" 2>/dev/null | wc -l || echo "0")
        
        if [ "$emergency_count" -gt 0 ]; then
            echo "EMERGENCY_FLAGS_ACTIVE:$emergency_count"
            return 0
        fi
        
        echo "ALL_NORMAL"
        return 0
    fi
}

# Send emergency alert
send_emergency_alert() {
    local flag_key="$1"
    local reason="$2"
    local triggered_by="$3"
    local severity="$4"
    
    local emoji="🚨"
    if [ "$severity" = "critical" ]; then
        emoji="🔴"
    elif [ "$severity" = "high" ]; then
        emoji="🟠"
    fi
    
    local message="$emoji EMERGENCY KILL SWITCH ACTIVATED $emoji

Flag: $flag_key
Reason: $reason
Severity: $severity
Triggered by: $triggered_by
Time: $(date -u)
Environment: $ENVIRONMENT

IMMEDIATE ACTION REQUIRED!"

    # Send to Slack
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\", \"username\":\"Emergency Bot\", \"icon_emoji\":\":warning:\"}" \
            "$SLACK_WEBHOOK_URL" > /dev/null 2>&1 || true
    fi
    
    # Send to PagerDuty (if configured)
    if [ -n "${PAGERDUTY_INTEGRATION_KEY:-}" ]; then
        local pd_payload="{\"routing_key\":\"$PAGERDUTY_INTEGRATION_KEY\",\"event_action\":\"trigger\",\"payload\":{\"summary\":\"Emergency Kill Switch: $flag_key\",\"source\":\"feature-flags\",\"severity\":\"$severity\",\"custom_details\":{\"flag_key\":\"$flag_key\",\"reason\":\"$reason\",\"triggered_by\":\"$triggered_by\",\"environment\":\"$ENVIRONMENT\"}}}"
        curl -X POST -H 'Content-type: application/json' \
            --data "$pd_payload" \
            "https://events.pagerduty.com/v2/enqueue" > /dev/null 2>&1 || true
    fi
    
    # Send email
    if [ -n "${EMERGENCY_EMAIL:-}" ]; then
        echo "$message" | mail -s "🚨 EMERGENCY: Kill Switch Activated for $flag_key" "$EMERGENCY_EMAIL" || true
    fi
    
    # Send SMS (if configured)
    if [ -n "${TWILIO_PHONE_NUMBER:-}" ] && [ -n "${EMERGENCY_SMS:-}" ]; then
        curl -X POST "https://api.twilio.com/2010-04-01/Accounts/${TWILIO_ACCOUNT_SID}/Messages.json" \
            --data-urlencode "To=$EMERGENCY_SMS" \
            --data-urlencode "From=$TWILIO_PHONE_NUMBER" \
            --data-urlencode "Body=EMERGENCY: Kill Switch activated for $flag_key. Reason: $reason" \
            -u "${TWILIO_ACCOUNT_SID}:${TWILIO_AUTH_TOKEN}" > /dev/null 2>&1 || true
    fi
    
    log "Emergency alerts sent for flag: $flag_key"
}

# Send recovery notification
send_recovery_notification() {
    local flag_key="$1"
    local recovery_reason="$2"
    local triggered_by="$3"
    
    local message="✅ KILL SWITCH DEACTIVATED ✅

Flag: $flag_key
Recovery reason: $recovery_reason
Triggered by: $triggered_by
Time: $(date -u)
Environment: $ENVIRONMENT

Service恢复正常"

    # Send to Slack
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\", \"username\":\"Recovery Bot\", \"icon_emoji\":\":white_check_mark:\"}" \
            "$SLACK_WEBHOOK_URL" > /dev/null 2>&1 || true
    fi
    
    # Send email
    if [ -n "${RECOVERY_EMAIL:-}" ]; then
        echo "$message" | mail -s "✅ Recovery: Kill Switch Deactivated for $flag_key" "$RECOVERY_EMAIL" || true
    fi
    
    log "Recovery notifications sent for flag: $flag_key"
}

# Send global emergency alert
send_global_emergency_alert() {
    local reason="$1"
    local triggered_by="$2"
    local severity="$3"
    local flags_affected="$4"
    
    local message="🔴 GLOBAL EMERGENCY STOP 🔴

Reason: $reason
Severity: $severity
Triggered by: $triggered_by
Flags affected: $flags_affected
Time: $(date -u)
Environment: $ENVIRONMENT

ALL FEATURE FLAGS DISABLED!"

    # Send to all alert channels
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\", \"username\":\"Global Emergency Bot\", \"icon_emoji\":\":rotating_light:\"}" \
            "$SLACK_WEBHOOK_URL" > /dev/null 2>&1 || true
    fi
    
    if [ -n "${PAGERDUTY_INTEGRATION_KEY:-}" ]; then
        local pd_payload="{\"routing_key\":\"$PAGERDUTY_INTEGRATION_KEY\",\"event_action\":\"trigger\",\"payload\":{\"summary\":\"Global Emergency Stop Activated\",\"source\":\"feature-flags\",\"severity\":\"critical\",\"custom_details\":{\"reason\":\"$reason\",\"triggered_by\":\"$triggered_by\",\"flags_affected\":$flags_affected,\"environment\":\"$ENVIRONMENT\"}}}"
        curl -X POST -H 'Content-type: application/json' \
            --data "$pd_payload" \
            "https://events.pagerduty.com/v2/enqueue" > /dev/null 2>&1 || true
    fi
    
    if [ -n "${EMERGENCY_EMAIL:-}" ]; then
        echo "$message" | mail -s "🔴 GLOBAL EMERGENCY: All Flags Disabled" "$EMERGENCY_EMAIL" || true
    fi
    
    log "Global emergency alerts sent. $flags_affected flags affected."
}

# Log emergency action
log_emergency_action() {
    local flag_key="$1"
    local reason="$2"
    local triggered_by="$3"
    local severity="$4"
    
    local log_entry="{\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"type\":\"emergency_activation\",\"flag_key\":\"$flag_key\",\"reason\":\"$reason\",\"triggered_by\":\"$triggered_by\",\"severity\":\"$severity\",\"environment\":\"$ENVIRONMENT\"}"
    
    # Log to Redis stream
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" xadd emergency_log "*" "log_entry" "$log_entry" > /dev/null 2>&1 || true
    
    # Log to file
    echo "$log_entry" >> "${PROJECT_ROOT}/logs/emergency_actions.log" 2>/dev/null || true
    
    log "Emergency action logged for flag: $flag_key"
}

# Log recovery action
log_recovery_action() {
    local flag_key="$1"
    local recovery_reason="$2"
    local triggered_by="$3"
    
    local log_entry="{\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"type\":\"emergency_deactivation\",\"flag_key\":\"$flag_key\",\"recovery_reason\":\"$recovery_reason\",\"triggered_by\":\"$triggered_by\",\"environment\":\"$ENVIRONMENT\"}"
    
    # Log to Redis stream
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" xadd emergency_log "*" "log_entry" "$log_entry" > /dev/null 2>&1 || true
    
    # Log to file
    echo "$log_entry" >> "${PROJECT_ROOT}/logs/emergency_actions.log" 2>/dev/null || true
    
    log "Recovery action logged for flag: $flag_key"
}

# Log global emergency
log_global_emergency() {
    local reason="$1"
    local triggered_by="$2"
    local severity="$3"
    local flags_affected="$4"
    
    local log_entry="{\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"type\":\"global_emergency\",\"reason\":\"$reason\",\"triggered_by\":\"$triggered_by\",\"severity\":\"$severity\",\"flags_affected\":$flags_affected,\"environment\":\"$ENVIRONMENT\"}"
    
    # Log to Redis stream
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" xadd emergency_log "*" "log_entry" "$log_entry" > /dev/null 2>&1 || true
    
    # Log to file
    echo "$log_entry" >> "${PROJECT_ROOT}/logs/emergency_actions.log" 2>/dev/null || true
    
    log "Global emergency logged. $flags_affected flags affected."
}

# Monitor system health and auto-trigger kill switches
auto_monitor() {
    local check_interval="${1:-30}" # seconds
    local error_threshold="${2:-10}" # errors per minute
    local latency_threshold="${3:-1000}" # milliseconds
    
    log "Starting automatic emergency monitoring (interval: ${check_interval}s)"
    
    while true; do
        # Check error rates
        local error_rate
        error_rate=$(get_current_error_rate)
        
        if [ "$error_rate" -gt "$error_threshold" ]; then
            warn "High error rate detected: $error_rate/min (threshold: $error_threshold)"
            # Auto-trigger emergency for problematic flags
            auto_trigger_emergency "High error rate: $error_rate/min"
        fi
        
        # Check latency
        local avg_latency
        avg_latency=$(get_current_latency)
        
        if [ "$avg_latency" -gt "$latency_threshold" ]; then
            warn "High latency detected: ${avg_latency}ms (threshold: ${latency_threshold}ms)"
            # Auto-trigger emergency for latency-sensitive flags
            auto_trigger_emergency "High latency: ${avg_latency}ms"
        fi
        
        # Check system health
        local health_status
        health_status=$(check_system_health)
        
        if [ "$health_status" != "healthy" ]; then
            warn "System health degraded: $health_status"
            auto_trigger_emergency "System health degraded: $health_status"
        fi
        
        sleep "$check_interval"
    done
}

# Get current error rate (placeholder - integrate with monitoring)
get_current_error_rate() {
    # This would integrate with your monitoring system
    # For now, return a simulated value
    echo "5"
}

# Get current latency (placeholder - integrate with monitoring)
get_current_latency() {
    # This would integrate with your monitoring system
    # For now, return a simulated value
    echo "200"
}

# Check system health (placeholder - integrate with health checks)
check_system_health() {
    # This would integrate with your health check system
    # For now, return healthy
    echo "healthy"
}

# Auto-trigger emergency for problematic flags
auto_trigger_emergency() {
    local reason="$1"
    local triggered_by="auto_monitor"
    
    # Get flags that are likely causing issues
    local problematic_flags
    problematic_flags=$(get_problematic_flags)
    
    for flag_key in $problematic_flags; do
        emergency_activate "$flag_key" "$reason" "$triggered_by" "high"
    done
}

# Get problematic flags (placeholder - integrate with analytics)
get_problematic_flags() {
    # This would analyze which flags are likely causing issues
    # For now, return empty
    echo ""
}

# Show emergency dashboard
show_dashboard() {
    clear
    echo "=========================================="
    echo "    EMERGENCY KILL SWITCH DASHBOARD"
    echo "=========================================="
    echo ""
    
    # Global status
    local global_status
    global_status=$(check_emergency_status)
    echo "Global Status: $global_status"
    echo ""
    
    # Active emergency kill switches
    local emergency_count
    emergency_count=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" keys "emergency_kill:*" 2>/dev/null | wc -l || echo "0")
    echo "Active Emergency Kill Switches: $emergency_count"
    
    if [ "$emergency_count" -gt 0 ]; then
        echo "Emergency Flags:"
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" keys "emergency_kill:*" 2>/dev/null | while read -r key; do
            local flag_key="${key#emergency_kill:}"
            echo "  - $flag_key: EMERGENCY ACTIVE"
        done
    fi
    echo ""
    
    # Active regular kill switches
    local kill_switch_count
    kill_switch_count=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" keys "kill_switch:*" 2>/dev/null | wc -l || echo "0")
    echo "Active Regular Kill Switches: $kill_switch_count"
    
    if [ "$kill_switch_count" -gt 0 ]; then
        echo "Kill Switch Flags:"
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" keys "kill_switch:*" 2>/dev/null | while read -r key; do
            local flag_key="${key#kill_switch:}"
            # Skip emergency flags (already shown)
            if ! redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" get "emergency_kill:$flag_key" > /dev/null 2>&1; then
                echo "  - $flag_key: KILL SWITCH ACTIVE"
            fi
        done
    fi
    echo ""
    
    # Recent emergency actions
    echo "Recent Emergency Actions:"
    if [ -f "${PROJECT_ROOT}/logs/emergency_actions.log" ]; then
        tail -5 "${PROJECT_ROOT}/logs/emergency_actions.log" | while read -r line; do
            local timestamp
            timestamp=$(echo "$line" | jq -r '.timestamp' 2>/dev/null || echo "Unknown")
            local type
            type=$(echo "$line" | jq -r '.type' 2>/dev/null || echo "Unknown")
            local flag_key
            flag_key=$(echo "$line" | jq -r '.flag_key // "N/A"' 2>/dev/null || echo "N/A")
            echo "  - $timestamp: $type ($flag_key)"
        done
    else
        echo "  No recent actions"
    fi
    echo ""
    
    echo "=========================================="
}

# Show usage
show_usage() {
    cat << EOF
Emergency Kill Switch Management Script

USAGE:
    $0 <command> [options]

EMERGENCY COMMANDS:
    emergency-activate <flag_key> <reason> [triggered_by] [severity]
    emergency-deactivate <flag_key> [triggered_by] [recovery_reason]
    global-emergency-stop <reason> [triggered_by] [severity]
    
MONITORING COMMANDS:
    check-status [flag_key]     Check emergency status
    auto-monitor [interval] [error_threshold] [latency_threshold]
    dashboard                    Show emergency dashboard

EXAMPLES:
    $0 emergency-activate new_dashboard "Performance degradation" ops@company.com critical
    $0 emergency-deactivate new_dashboard admin@company.com "Issue resolved"
    $0 global-emergency-stop "System overload" emergency_system critical
    $0 check-status new_dashboard
    $0 auto-monitor 30 10 1000
    $0 dashboard

ENVIRONMENT VARIABLES:
    REDIS_HOST              Redis host (default: localhost)
    REDIS_PORT              Redis port (default: 6379)
    ENVIRONMENT             Environment (default: development)
    SLACK_WEBHOOK_URL       Slack webhook for alerts
    PAGERDUTY_INTEGRATION_KEY PagerDuty integration key
    EMERGENCY_EMAIL         Email for emergency alerts
    RECOVERY_EMAIL         Email for recovery notifications
    TWILIO_ACCOUNT_SID     Twilio account SID for SMS
    TWILIO_AUTH_TOKEN      Twilio auth token for SMS
    TWILIO_PHONE_NUMBER    Twilio phone number for SMS
    EMERGENCY_SMS          Emergency SMS number

EOF
}

# Main execution
main() {
    case "${1:-}" in
        "emergency-activate")
            if [ $# -lt 3 ]; then
                error "Usage: $0 emergency-activate <flag_key> <reason> [triggered_by] [severity]"
                exit 1
            fi
            emergency_activate "$2" "$3" "${4:-emergency_system}" "${5:-critical}"
            ;;
        "emergency-deactivate")
            if [ $# -lt 2 ]; then
                error "Usage: $0 emergency-deactivate <flag_key> [triggered_by] [recovery_reason]"
                exit 1
            fi
            emergency_deactivate "$2" "${3:-emergency_system}" "${4:-Issue resolved}"
            ;;
        "global-emergency-stop")
            if [ $# -lt 2 ]; then
                error "Usage: $0 global-emergency-stop <reason> [triggered_by] [severity]"
                exit 1
            fi
            global_emergency_stop "$2" "${3:-emergency_system}" "${4:-critical}"
            ;;
        "check-status")
            check_emergency_status "${2:-}"
            ;;
        "auto-monitor")
            auto_monitor "${2:-30}" "${3:-10}" "${4:-1000}"
            ;;
        "dashboard")
            show_dashboard
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"