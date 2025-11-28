#!/bin/bash

# Feature Flag Testing and Validation Framework
# Provides comprehensive testing capabilities for feature flags before deployment

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-feature_flags}"
ENVIRONMENT="${ENVIRONMENT:-test}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] TEST:${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] TEST WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] TEST ERROR:${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] TEST SUCCESS:${NC} $1"
}

# Test flag evaluation
test_flag_evaluation() {
    local flag_key="$1"
    local test_context="$2"
    local expected_result="$3"
    local test_name="${4:-Flag Evaluation Test}"
    
    log "Running test: $test_name"
    
    # Evaluate flag
    local result
    result=$(curl -s -X POST "http://localhost:3000/api/feature-flags/evaluate" \
        -H "Content-Type: application/json" \
        -d "{\"flag_key\":\"$flag_key\",\"context\":$test_context}" | jq -r '.enabled')
    
    if [ "$result" = "$expected_result" ]; then
        success "✓ $test_name - Expected: $expected_result, Got: $result"
        return 0
    else
        error "✗ $test_name - Expected: $expected_result, Got: $result"
        return 1
    fi
}

# Test flag value
test_flag_value() {
    local flag_key="$1"
    local test_context="$2"
    local expected_value="$3"
    local test_name="${4:-Flag Value Test}"
    
    log "Running test: $test_name"
    
    # Get flag value
    local result
    result=$(curl -s -X POST "http://localhost:3000/api/feature-flags/evaluate" \
        -H "Content-Type: application/json" \
        -d "{\"flag_key\":\"$flag_key\",\"context\":$test_context}" | jq -r '.value')
    
    if [ "$result" = "$expected_value" ]; then
        success "✓ $test_name - Expected: $expected_value, Got: $result"
        return 0
    else
        error "✗ $test_name - Expected: $expected_value, Got: $result"
        return 1
    fi
}

# Test kill switch functionality
test_kill_switch() {
    local flag_key="$1"
    local test_name="Kill Switch Test"
    
    log "Running test: $test_name"
    
    # Activate kill switch
    curl -s -X POST "http://localhost:3000/api/feature-flags/$flag_key/kill-switch" \
        -H "Content-Type: application/json" \
        -d '{"action":"activate","reason":"Test activation"}' > /dev/null
    
    # Test evaluation (should be false)
    local result
    result=$(curl -s -X POST "http://localhost:3000/api/feature-flags/evaluate" \
        -H "Content-Type: application/json" \
        -d "{\"flag_key\":\"$flag_key\",\"context\":{}}" | jq -r '.enabled')
    
    if [ "$result" = "false" ]; then
        success "✓ $test_name - Kill switch properly disabled flag"
        
        # Deactivate kill switch
        curl -s -X POST "http://localhost:3000/api/feature-flags/$flag_key/kill-switch" \
            -H "Content-Type: application/json" \
            -d '{"action":"deactivate"}' > /dev/null
        
        return 0
    else
        error "✗ $test_name - Kill switch failed to disable flag"
        return 1
    fi
}

# Test user targeting
test_user_targeting() {
    local flag_key="$1"
    local test_name="User Targeting Test"
    
    log "Running test: $test_name"
    
    # Test with different user contexts
    local contexts=(
        '{"user_id":"test_user_1","email":"test1@example.com","tier":"premium"}'
        '{"user_id":"test_user_2","email":"test2@example.com","tier":"free"}'
        '{"user_id":"test_user_3","email":"test3@example.com","region":"us-east"}'
    )
    
    local passed=0
    local total=${#contexts[@]}
    
    for context in "${contexts[@]}"; do
        local result
        result=$(curl -s -X POST "http://localhost:3000/api/feature-flags/evaluate" \
            -H "Content-Type: application/json" \
            -d "{\"flag_key\":\"$flag_key\",\"context\":$context}" | jq -r '.enabled')
        
        # Just check that evaluation works (no specific expected result)
        if [ "$result" = "true" ] || [ "$result" = "false" ]; then
            ((passed++))
        fi
    done
    
    if [ $passed -eq $total ]; then
        success "✓ $test_name - All $total user targeting tests passed"
        return 0
    else
        error "✗ $test_name - $passed/$total user targeting tests passed"
        return 1
    fi
}

# Test performance
test_performance() {
    local flag_key="$1"
    local iterations="${2:-100}"
    local max_response_time="${3:-100}" # milliseconds
    local test_name="Performance Test"
    
    log "Running test: $test_name ($iterations iterations)"
    
    local total_time=0
    local failed=0
    
    for ((i=1; i<=iterations; i++)); do
        local start_time
        start_time=$(date +%s%N)
        
        local result
        result=$(curl -s -X POST "http://localhost:3000/api/feature-flags/evaluate" \
            -H "Content-Type: application/json" \
            -d "{\"flag_key\":\"$flag_key\",\"context\":{}}" | jq -r '.enabled')
        
        local end_time
        end_time=$(date +%s%N)
        
        local response_time=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
        
        total_time=$((total_time + response_time))
        
        if [ "$result" != "true" ] && [ "$result" != "false" ]; then
            ((failed++))
        fi
    done
    
    local avg_time=$((total_time / iterations))
    
    if [ $failed -eq 0 ] && [ $avg_time -le $max_response_time ]; then
        success "✓ $test_name - Avg: ${avg_time}ms, Failed: $failed/$iterations"
        return 0
    else
        error "✗ $test_name - Avg: ${avg_time}ms, Failed: $failed/$iterations"
        return 1
    fi
}

# Test cache invalidation
test_cache_invalidation() {
    local flag_key="$1"
    local test_name="Cache Invalidation Test"
    
    log "Running test: $test_name"
    
    # First evaluation (should cache)
    local result1
    result1=$(curl -s -X POST "http://localhost:3000/api/feature-flags/evaluate" \
        -H "Content-Type: application/json" \
        -d "{\"flag_key\":\"$flag_key\",\"context\":{}}" | jq -r '.enabled')
    
    # Update flag
    curl -s -X PATCH "http://localhost:3000/api/feature-flags/$flag_key" \
        -H "Content-Type: application/json" \
        -d '{"enabled":false}' > /dev/null
    
    sleep 1 # Allow cache invalidation to propagate
    
    # Second evaluation (should reflect change)
    local result2
    result2=$(curl -s -X POST "http://localhost:3000/api/feature-flags/evaluate" \
        -H "Content-Type: application/json" \
        -d "{\"flag_key\":\"$flag_key\",\"context\":{}}" | jq -r '.enabled')
    
    # Restore original state
    curl -s -X PATCH "http://localhost:3000/api/feature-flags/$flag_key" \
        -H "Content-Type: application/json" \
        -d "{\"enabled\":$result1}" > /dev/null
    
    if [ "$result1" != "$result2" ]; then
        success "✓ $test_name - Cache properly invalidated"
        return 0
    else
        error "✗ $test_name - Cache not properly invalidated"
        return 1
    fi
}

# Test concurrent access
test_concurrent_access() {
    local flag_key="$1"
    local concurrent_requests="${2:-50}"
    local test_name="Concurrent Access Test"
    
    log "Running test: $test_name ($concurrent_requests concurrent requests)"
    
    # Create temporary file for results
    local temp_file
    temp_file=$(mktemp)
    
    # Launch concurrent requests
    for ((i=1; i<=concurrent_requests; i++)); do
        {
            local result
            result=$(curl -s -X POST "http://localhost:3000/api/feature-flags/evaluate" \
                -H "Content-Type: application/json" \
                -d "{\"flag_key\":\"$flag_key\",\"context\":{}}" | jq -r '.enabled')
            echo "$result" >> "$temp_file"
        } &
    done
    
    # Wait for all background jobs
    wait
    
    # Analyze results
    local true_count
    true_count=$(grep -c "true" "$temp_file" || echo "0")
    local false_count
    false_count=$(grep -c "false" "$temp_file" || echo "0")
    local total_count
    total_count=$((true_count + false_count))
    
    rm -f "$temp_file"
    
    if [ $total_count -eq $concurrent_requests ]; then
        success "✓ $test_name - All $concurrent_requests requests completed (True: $true_count, False: $false_count)"
        return 0
    else
        error "✗ $test_name - Only $total_count/$concurrent_requests requests completed"
        return 1
    fi
}

# Test rollback functionality
test_rollback() {
    local flag_key="$1"
    local test_name="Rollback Test"
    
    log "Running test: $test_name"
    
    # Get original state
    local original_state
    original_state=$(curl -s "http://localhost:3000/api/feature-flags/$flag_key" | jq -r '.enabled')
    
    # Change state
    local new_state
    if [ "$original_state" = "true" ]; then
        new_state="false"
    else
        new_state="true"
    fi
    
    curl -s -X PATCH "http://localhost:3000/api/feature-flags/$flag_key" \
        -H "Content-Type: application/json" \
        -d "{\"enabled\":$new_state}" > /dev/null
    
    # Verify change
    local changed_state
    changed_state=$(curl -s "http://localhost:3000/api/feature-flags/$flag_key" | jq -r '.enabled')
    
    if [ "$changed_state" = "$new_state" ]; then
        # Rollback
        curl -s -X PATCH "http://localhost:3000/api/feature-flags/$flag_key" \
            -H "Content-Type: application/json" \
            -d "{\"enabled\":$original_state}" > /dev/null
        
        # Verify rollback
        local rollback_state
        rollback_state=$(curl -s "http://localhost:3000/api/feature-flags/$flag_key" | jq -r '.enabled')
        
        if [ "$rollback_state" = "$original_state" ]; then
            success "✓ $test_name - Rollback successful"
            return 0
        else
            error "✗ $test_name - Rollback failed"
            return 1
        fi
    else
        error "✗ $test_name - State change failed"
        return 1
    fi
}

# Run comprehensive test suite
run_test_suite() {
    local flag_key="$1"
    local test_type="${2:-all}"
    
    log "Running test suite for flag: $flag_key"
    
    local total_tests=0
    local passed_tests=0
    
    # Check if flag exists
    local flag_exists
    flag_exists=$(curl -s "http://localhost:3000/api/feature-flags/$flag_key" | jq -r '.key' 2>/dev/null || echo "")
    
    if [ -z "$flag_exists" ]; then
        error "Flag '$flag_key' does not exist"
        return 1
    fi
    
    case "$test_type" in
        "basic")
            ((total_tests++))
            test_flag_evaluation "$flag_key" '{}' "true" "Basic Evaluation Test" && ((passed_tests++))
            
            ((total_tests++))
            test_flag_value "$flag_key" '{}' "true" "Basic Value Test" && ((passed_tests++))
            ;;
        "performance")
            ((total_tests++))
            test_performance "$flag_key" 100 100 && ((passed_tests++))
            ;;
        "concurrent")
            ((total_tests++))
            test_concurrent_access "$flag_key" 50 && ((passed_tests++))
            ;;
        "kill-switch")
            ((total_tests++))
            test_kill_switch "$flag_key" && ((passed_tests++))
            ;;
        "cache")
            ((total_tests++))
            test_cache_invalidation "$flag_key" && ((passed_tests++))
            ;;
        "targeting")
            ((total_tests++))
            test_user_targeting "$flag_key" && ((passed_tests++))
            ;;
        "rollback")
            ((total_tests++))
            test_rollback "$flag_key" && ((passed_tests++))
            ;;
        "all"|*)
            ((total_tests++))
            test_flag_evaluation "$flag_key" '{}' "true" "Basic Evaluation Test" && ((passed_tests++))
            
            ((total_tests++))
            test_flag_value "$flag_key" '{}' "true" "Basic Value Test" && ((passed_tests++))
            
            ((total_tests++))
            test_performance "$flag_key" 100 100 && ((passed_tests++))
            
            ((total_tests++))
            test_concurrent_access "$flag_key" 50 && ((passed_tests++))
            
            ((total_tests++))
            test_kill_switch "$flag_key" && ((passed_tests++))
            
            ((total_tests++))
            test_cache_invalidation "$flag_key" && ((passed_tests++))
            
            ((total_tests++))
            test_user_targeting "$flag_key" && ((passed_tests++))
            
            ((total_tests++))
            test_rollback "$flag_key" && ((passed_tests++))
            ;;
    esac
    
    echo ""
    log "Test Results: $passed_tests/$total_tests tests passed"
    
    if [ $passed_tests -eq $total_tests ]; then
        success "All tests passed for flag: $flag_key"
        return 0
    else
        error "Some tests failed for flag: $flag_key"
        return 1
    fi
}

# Validate flag configuration
validate_flag_config() {
    local flag_key="$1"
    
    log "Validating configuration for flag: $flag_key"
    
    # Get flag configuration
    local flag_config
    flag_config=$(curl -s "http://localhost:3000/api/feature-flags/$flag_key")
    
    if [ -z "$flag_config" ]; then
        error "Flag '$flag_key' not found"
        return 1
    fi
    
    local validation_errors=0
    
    # Check required fields
    local key
    key=$(echo "$flag_config" | jq -r '.key // empty')
    if [ -z "$key" ]; then
        error "Missing required field: key"
        ((validation_errors++))
    fi
    
    local name
    name=$(echo "$flag_config" | jq -r '.name // empty')
    if [ -z "$name" ]; then
        error "Missing required field: name"
        ((validation_errors++))
    fi
    
    local type
    type=$(echo "$flag_config" | jq -r '.type // empty')
    if [ -z "$type" ]; then
        error "Missing required field: type"
        ((validation_errors++))
    fi
    
    # Check valid types
    local valid_types=("boolean" "string" "number" "json")
    local type_valid=false
    for valid_type in "${valid_types[@]}"; do
        if [ "$type" = "$valid_type" ]; then
            type_valid=true
            break
        fi
    done
    
    if [ "$type_valid" = false ]; then
        error "Invalid type: $type (must be one of: ${valid_types[*]})"
        ((validation_errors++))
    fi
    
    # Check rollout percentage
    local rollout_percentage
    rollout_percentage=$(echo "$flag_config" | jq -r '.rollout.percentage // 100')
    if ! [[ "$rollout_percentage" =~ ^[0-9]+$ ]] || [ "$rollout_percentage" -lt 0 ] || [ "$rollout_percentage" -gt 100 ]; then
        error "Invalid rollout percentage: $rollout_percentage (must be 0-100)"
        ((validation_errors++))
    fi
    
    if [ $validation_errors -eq 0 ]; then
        success "Flag configuration is valid"
        return 0
    else
        error "Flag configuration has $validation_errors validation errors"
        return 1
    fi
}

# Generate test report
generate_test_report() {
    local flag_key="$1"
    local output_file="${2:-test_report_$(date +%Y%m%d_%H%M%S).json}"
    
    log "Generating test report for flag: $flag_key"
    
    # Run all tests and collect results
    local report="{
        \"flag_key\": \"$flag_key\",
        \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
        \"environment\": \"$ENVIRONMENT\",
        \"tests\": [
    "
    
    # Add test results
    local tests=(
        "basic_evaluation"
        "basic_value"
        "performance"
        "concurrent_access"
        "kill_switch"
        "cache_invalidation"
        "user_targeting"
        "rollback"
    )
    
    local first=true
    for test in "${tests[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            report+=","
        fi
        
        # Run test and capture result
        local test_result="pass"
        case "$test" in
            "basic_evaluation")
                test_flag_evaluation "$flag_key" '{}' "true" "Basic Evaluation Test" > /dev/null 2>&1 || test_result="fail"
                ;;
            "basic_value")
                test_flag_value "$flag_key" '{}' "true" "Basic Value Test" > /dev/null 2>&1 || test_result="fail"
                ;;
            "performance")
                test_performance "$flag_key" 100 100 > /dev/null 2>&1 || test_result="fail"
                ;;
            "concurrent_access")
                test_concurrent_access "$flag_key" 50 > /dev/null 2>&1 || test_result="fail"
                ;;
            "kill_switch")
                test_kill_switch "$flag_key" > /dev/null 2>&1 || test_result="fail"
                ;;
            "cache_invalidation")
                test_cache_invalidation "$flag_key" > /dev/null 2>&1 || test_result="fail"
                ;;
            "user_targeting")
                test_user_targeting "$flag_key" > /dev/null 2>&1 || test_result="fail"
                ;;
            "rollback")
                test_rollback "$flag_key" > /dev/null 2>&1 || test_result="fail"
                ;;
        esac
        
        report+="{
            \"name\": \"$test\",
            \"status\": \"$test_result\",
            \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
        }"
    done
    
    report+="
        ]
    }"
    
    echo "$report" | jq '.' > "$output_file"
    success "Test report generated: $output_file"
}

# Show usage
show_usage() {
    cat << EOF
Feature Flag Testing and Validation Framework

USAGE:
    $0 <command> [options]

TEST COMMANDS:
    test <flag_key> [test_type]           Run test suite (all, basic, performance, concurrent, kill-switch, cache, targeting, rollback)
    validate <flag_key>                   Validate flag configuration
    report <flag_key> [output_file]        Generate comprehensive test report

INDIVIDUAL TESTS:
    evaluation <flag_key> <context> <expected> [test_name]
    value <flag_key> <context> <expected> [test_name]
    kill-switch <flag_key>
    targeting <flag_key>
    performance <flag_key> [iterations] [max_time]
    concurrent <flag_key> [requests]
    cache <flag_key>
    rollback <flag_key>

EXAMPLES:
    $0 test new_dashboard all
    $0 test new_dashboard basic
    $0 validate new_dashboard
    $0 report new_dashboard test_report.json
    $0 evaluation new_dashboard '{"user_id":"test"}' true "User Test"
    $0 performance new_dashboard 100 50
    $0 concurrent new_dashboard 100

ENVIRONMENT VARIABLES:
    REDIS_HOST              Redis host (default: localhost)
    REDIS_PORT              Redis port (default: 6379)
    DB_HOST                 Database host (default: localhost)
    DB_PORT                 Database port (default: 5432)
    DB_NAME                 Database name (default: feature_flags)
    ENVIRONMENT             Environment (default: test)

EOF
}

# Main execution
main() {
    case "${1:-}" in
        "test")
            if [ $# -lt 2 ]; then
                error "Usage: $0 test <flag_key> [test_type]"
                exit 1
            fi
            run_test_suite "$2" "${3:-all}"
            ;;
        "validate")
            if [ $# -lt 2 ]; then
                error "Usage: $0 validate <flag_key>"
                exit 1
            fi
            validate_flag_config "$2"
            ;;
        "report")
            if [ $# -lt 2 ]; then
                error "Usage: $0 report <flag_key> [output_file]"
                exit 1
            fi
            generate_test_report "$2" "${3:-}"
            ;;
        "evaluation")
            if [ $# -lt 4 ]; then
                error "Usage: $0 evaluation <flag_key> <context> <expected> [test_name]"
                exit 1
            fi
            test_flag_evaluation "$2" "$3" "$4" "${5:-Flag Evaluation Test}"
            ;;
        "value")
            if [ $# -lt 4 ]; then
                error "Usage: $0 value <flag_key> <context> <expected> [test_name]"
                exit 1
            fi
            test_flag_value "$2" "$3" "$4" "${5:-Flag Value Test}"
            ;;
        "kill-switch")
            if [ $# -lt 2 ]; then
                error "Usage: $0 kill-switch <flag_key>"
                exit 1
            fi
            test_kill_switch "$2"
            ;;
        "targeting")
            if [ $# -lt 2 ]; then
                error "Usage: $0 targeting <flag_key>"
                exit 1
            fi
            test_user_targeting "$2"
            ;;
        "performance")
            if [ $# -lt 2 ]; then
                error "Usage: $0 performance <flag_key> [iterations] [max_time]"
                exit 1
            fi
            test_performance "$2" "${3:-100}" "${4:-100}"
            ;;
        "concurrent")
            if [ $# -lt 2 ]; then
                error "Usage: $0 concurrent <flag_key> [requests]"
                exit 1
            fi
            test_concurrent_access "$2" "${3:-50}"
            ;;
        "cache")
            if [ $# -lt 2 ]; then
                error "Usage: $0 cache <flag_key>"
                exit 1
            fi
            test_cache_invalidation "$2"
            ;;
        "rollback")
            if [ $# -lt 2 ]; then
                error "Usage: $0 rollback <flag_key>"
                exit 1
            fi
            test_rollback "$2"
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"