#!/bin/bash

# Intelligent Backport Bot with Conflict Detection
# Provides automated backport capabilities with intelligent conflict resolution

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_ROOT}/config/backport-bot.yml"
GIT_REPO="${GIT_REPO:-$(pwd)}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] BACKPORT:${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] BACKPORT WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] BACKPORT ERROR:${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] BACKPORT SUCCESS:${NC} $1"
}

# Analyze commit for backport feasibility
analyze_commit_feasibility() {
    local commit_hash="$1"
    local target_branch="$2"
    
    log "Analyzing commit feasibility: $commit_hash -> $target_branch"
    
    # Get commit details
    local commit_details
    commit_details=$(git show --format="%H|%s|%b|%an|%ae" "$commit_hash")
    local commit_hash_full=$(echo "$commit_details" | cut -d'|' -f1)
    local commit_subject=$(echo "$commit_details" | cut -d'|' -f2)
    local commit_body=$(echo "$commit_details" | cut -d'|' -f3)
    local commit_author=$(echo "$commit_details" | cut -d'|' -f4)
    local commit_email=$(echo "$commit_details" | cut -d'|' -f5)
    
    # Get changed files
    local changed_files
    changed_files=$(git diff-tree --no-commit-id --name-only -r "$commit_hash")
    
    # Analyze changes
    local analysis="{
        \"commit_hash\": \"$commit_hash_full\",
        \"subject\": \"$commit_subject\",
        \"body\": \"$commit_body\",
        \"author\": \"$commit_author\",
        \"email\": \"$commit_email\",
        \"changed_files\": [$(echo "$changed_files" | sed 's/^/"/; s/$/"/; $!s/^/"/; $!s/$/"/; $!s/, /",/g')],
        \"file_count\": $(echo "$changed_files" | wc -l),
        \"has_merge_conflicts\": false,
        \"has_api_changes\": false,
        \"has_db_changes\": false,
        \"has_config_changes\": false,
        \"complexity_score\": 0
    }"
    
    # Detect file types
    if echo "$changed_files" | grep -q -E "\.(sql|migration)"; then
        analysis=$(echo "$analysis" | jq '.has_db_changes = true')
    fi
    
    if echo "$changed_files" | grep -q -E "\.(json|yaml|yml|proto)"; then
        analysis=$(echo "$analysis" | jq '.has_config_changes = true')
    fi
    
    if echo "$changed_files" | grep -q -E "(api|endpoint|route|controller)"; then
        analysis=$(echo "$analysis" | jq '.has_api_changes = true')
    fi
    
    # Calculate complexity score
    local file_count=$(echo "$analysis" | jq -r '.file_count')
    local complexity_score=$((file_count * 2))
    
    if echo "$analysis" | jq -r '.has_merge_conflicts' | grep -q true; then
        complexity_score=$((complexity_score + 20))
    fi
    
    if echo "$analysis" | jq -r '.has_api_changes' | grep -q true; then
        complexity_score=$((complexity_score + 15))
    fi
    
    if echo "$analysis" | jq -r '.has_db_changes' | grep -q true; then
        complexity_score=$((complexity_score + 25))
    fi
    
    analysis=$(echo "$analysis" | jq ".complexity_score = $complexity_score")
    
    echo "$analysis"
}

# Detect potential conflicts
detect_conflicts() {
    local commit_hash="$1"
    local target_branch="$2"
    
    log "Detecting conflicts for: $commit_hash -> $target_branch"
    
    local conflicts="[]"
    
    # 1. Check for existing changes in target branch
    local target_files
    target_files=$(git diff-tree --no-commit-id --name-only -r "origin/$target_branch" 2>/dev/null || echo "")
    
    local source_files
    source_files=$(git diff-tree --no-commit-id --name-only -r "$commit_hash")
    
    # Find overlapping files
    local overlapping_files
    overlapping_files=$(comm -12 <(echo "$target_files" | sort) <(echo "$source_files" | sort))
    
    if [ -n "$overlapping_files" ]; then
        for file in $overlapping_files; do
            local conflict_type
            conflict_type=$(analyze_conflict_type "$file" "$commit_hash" "$target_branch")
            
            if [ -n "$conflict_type" ]; then
                conflicts=$(echo "$conflicts" | jq ". += [{
                    \"file_path\": \"$file\",
                    \"conflict_type\": \"$conflict_type\",
                    \"severity\": \"medium\",
                    \"description\": \"Potential conflict in $file\",
                    \"suggested_resolution\": \"Manual review required\"
                }]")
            fi
        done
    fi
    
    # 2. Check for semantic conflicts
    local semantic_conflicts
    semantic_conflicts=$(detect_semantic_conflicts "$commit_hash" "$target_branch")
    conflicts=$(echo "$conflicts" | jq ". += $semantic_conflicts")
    
    # 3. Check for dependency conflicts
    local dep_conflicts
    dep_conflicts=$(detect_dependency_conflicts "$commit_hash" "$target_branch")
    conflicts=$(echo "$conflicts" | jq ". += $dep_conflicts")
    
    echo "$conflicts"
}

# Analyze conflict type
analyze_conflict_type() {
    local file="$1"
    local commit_hash="$2"
    local target_branch="$3"
    
    # Get file content from both sides
    local source_content
    source_content=$(git show "$commit_hash:$file" 2>/dev/null || echo "")
    
    local target_content
    target_content=$(git show "origin/$target_branch:$file" 2>/dev/null || echo "")
    
    # Simple conflict detection (can be enhanced)
    if [ -n "$source_content" ] && [ -n "$target_content" ]; then
        if [ "$source_content" != "$target_content" ]; then
            echo "merge_conflict"
            return
        fi
    fi
    
    echo ""
}

# Detect semantic conflicts
detect_semantic_conflicts() {
    local commit_hash="$1"
    local target_branch="$2"
    
    local conflicts="[]"
    
    # Get commit message
    local commit_msg
    commit_msg=$(git log --format="%s" -n 1 "$commit_hash")
    
    # Check for conflicting patterns
    if echo "$commit_msg" | grep -qi "remove.*deprecated"; then
        # Check if target branch has recent additions to same feature
        local recent_commits
        recent_commits=$(git log --oneline --since="2 weeks ago" "origin/$target_branch" | grep -i "add.*deprecated" || true)
        
        if [ -n "$recent_commits" ]; then
            conflicts=$(echo "$conflicts" | jq ". += [{
                \"file_path\": \"semantic\",
                \"conflict_type\": \"semantic\",
                \"severity\": \"high\",
                \"description\": \"Deprecated feature removal conflicts with recent additions\",
                \"suggested_resolution\": \"Coordinate feature deprecation timeline\"
            }]")
        fi
    fi
    
    echo "$conflicts"
}

# Detect dependency conflicts
detect_dependency_conflicts() {
    local commit_hash="$1"
    local target_branch="$2"
    
    local conflicts="[]"
    
    # Check package.json changes
    local source_package
    source_package=$(git show "$commit_hash:package.json" 2>/dev/null || echo "{}")
    
    local target_package
    target_package=$(git show "origin/$target_branch:package.json" 2>/dev/null || echo "{}")
    
    # Compare dependency versions
    local source_deps
    source_deps=$(echo "$source_package" | jq -r '.dependencies // {}')
    
    local target_deps
    target_deps=$(echo "$target_package" | jq -r '.dependencies // {}')
    
    # Find version conflicts
    for dep in $(echo "$source_deps" | jq -r 'keys[]'); do
        local source_ver
        source_ver=$(echo "$source_deps" | jq -r ".\"$dep\" // empty")
        
        local target_ver
        target_ver=$(echo "$target_deps" | jq -r ".\"$dep\" // empty")
        
        if [ -n "$source_ver" ] && [ -n "$target_ver" ] && [ "$source_ver" != "$target_ver" ]; then
            conflicts=$(echo "$conflicts" | jq ". += [{
                \"file_path\": \"package.json\",
                \"conflict_type\": \"dependency\",
                \"severity\": \"medium\",
                \"description\": \"Dependency version conflict for $dep: $source_ver vs $target_ver\",
                \"suggested_resolution\": \"Resolve dependency version conflict\"
            }]")
        fi
    done
    
    echo "$conflicts"
}

# Create backport request
create_backport_request() {
    local commit_hash="$1"
    local target_branches="$2"
    local title="${3:-Backport $(git log --format="%h" -n 1 "$commit_hash") to $target_branches}"
    local description="${4:-Automated backport of commit $(git log --format="%h" -n 1 "$commit_hash")}"
    
    log "Creating backport request: $commit_hash -> $target_branches"
    
    # Analyze feasibility
    local feasibility
    feasibility=$(analyze_commit_feasibility "$commit_hash" "$target_branches")
    
    # Detect conflicts
    local conflicts
    conflicts=$(detect_conflicts "$commit_hash" "$target_branches")
    
    # Calculate priority
    local priority
    priority=$(calculate_backport_priority "$feasibility" "$conflicts")
    
    # Create request object
    local request_id
    request_id=$(uuidgen 2>/dev/null || python3 -c "import uuid; print(uuid.uuid4())")
    
    local backport_request="{
        \"id\": \"$request_id\",
        \"source_commit\": \"$commit_hash\",
        \"target_branches\": [$(echo "$target_branches" | sed 's/^/"/; s/$/"/; $!s/^/"/; $!s/$/"/; $!s/, /",/g')],
        \"title\": \"$title\",
        \"description\": \"$description\",
        \"author\": \"$(git config user.name)\",
        \"status\": \"pending\",
        \"created_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
        \"feasibility\": $feasibility,
        \"conflicts\": $conflicts,
        \"priority\": \"$priority\",
        \"metadata\": {
            \"labels\": [\"backport\", \"automated\"],
            \"environment\": \"$(git branch --show-current 2>/dev/null || echo 'unknown')\"
        }
    }"
    
    # Save request
    local requests_file="${PROJECT_ROOT}/data/backport_requests.json"
    mkdir -p "$(dirname "$requests_file")"
    
    # Add to existing requests or create new file
    if [ -f "$requests_file" ]; then
        temp_file=$(mktemp)
        jq ". += [$backport_request]" "$requests_file" > "$temp_file"
        mv "$temp_file" "$requests_file"
    else
        echo "[$backport_request]" > "$requests_file"
    fi
    
    success "Backport request created: $request_id"
    
    # Notify if conflicts detected
    local conflict_count
    conflict_count=$(echo "$conflicts" | jq 'length')
    
    if [ "$conflict_count" -gt 0 ]; then
        warn "⚠️  $conflict_count conflicts detected - Manual review required"
        send_conflict_notification "$request_id" "$conflicts"
    fi
    
    # Attempt automatic resolution if no conflicts
    if [ "$conflict_count" -eq 0 ]; then
        log "No conflicts detected - Attempting automatic backport"
        attempt_automatic_backport "$request_id" "$commit_hash" "$target_branches"
    fi
    
    echo "$request_id"
}

# Calculate backport priority
calculate_backport_priority() {
    local feasibility="$1"
    local conflicts="$2"
    
    local base_priority=50
    
    # Adjust based on complexity
    local complexity
    complexity=$(echo "$feasibility" | jq -r '.complexity_score')
    base_priority=$((base_priority + complexity / 5))
    
    # Adjust based on conflicts
    local conflict_count
    conflict_count=$(echo "$conflicts" | jq 'length')
    base_priority=$((base_priority + conflict_count * 10))
    
    # Determine priority level
    if [ $base_priority -ge 80 ]; then
        echo "critical"
    elif [ $base_priority -ge 60 ]; then
        echo "high"
    elif [ $base_priority -ge 40 ]; then
        echo "medium"
    else
        echo "low"
    fi
}

# Attempt automatic backport
attempt_automatic_backport() {
    local request_id="$1"
    local commit_hash="$2"
    local target_branches="$3"
    
    log "Attempting automatic backport for request: $request_id"
    
    for target_branch in $target_branches; do
        # Create temporary branch
        local temp_branch="backport/$request_id/$target_branch"
        git checkout -b "$temp_branch" "origin/$target_branch" 2>/dev/null || {
            error "Failed to create branch: $temp_branch"
            continue
        }
        
        # Attempt cherry-pick
        if git cherry-pick "$commit_hash" 2>/dev/null; then
            success "✓ Automatic backport successful for $target_branch"
            
            # Push to remote
            git push origin "$temp_branch" 2>/dev/null || {
                warn "Failed to push branch: $temp_branch"
            }
            
            # Create PR if configured
            if [ -n "$GITHUB_TOKEN" ]; then
                create_pull_request "$temp_branch" "$target_branch" "Automatic backport $request_id"
            fi
        else
            warn "✗ Automatic backport failed for $target_branch - Manual intervention required"
            update_request_status "$request_id" "conflict" "Automatic cherry-pick failed"
        fi
        
        # Return to original branch
        git checkout - 2>/dev/null
    done
}

# Create pull request
create_pull_request() {
    local head_branch="$1"
    local base_branch="$2"
    local title="$3"
    
    # This would integrate with GitHub API
    log "Creating pull request: $head_branch -> $base_branch"
    
    # Placeholder for GitHub API integration
    local pr_data="{
        \"title\": \"$title\",
        \"head\": \"$head_branch\",
        \"base\": \"$base_branch\",
        \"body\": \"Automated backport pull request\",
        \"labels\": [\"backport\", \"automated\"]
    }"
    
    # Save PR data for manual creation
    local pr_file="${PROJECT_ROOT}/data/pull_requests.json"
    mkdir -p "$(dirname "$pr_file")"
    
    if [ -f "$pr_file" ]; then
        temp_file=$(mktemp)
        jq ". += [$pr_data]" "$pr_file" > "$temp_file"
        mv "$temp_file" "$pr_file"
    else
        echo "[$pr_data]" > "$pr_file"
    fi
}

# Update request status
update_request_status() {
    local request_id="$1"
    local status="$2"
    local reason="${3:-}"
    
    local requests_file="${PROJECT_ROOT}/data/backport_requests.json"
    
    if [ -f "$requests_file" ]; then
        temp_file=$(mktemp)
        jq "(.[] | select(.id == \"$request_id\")) |= (.status = \"$status\" | .reason = \"$reason\")" "$requests_file" > "$temp_file"
        mv "$temp_file" "$requests_file"
    fi
}

# Send conflict notification
send_conflict_notification() {
    local request_id="$1"
    local conflicts="$2"
    
    if [ -n "$SLACK_WEBHOOK" ]; then
        local conflict_count
        conflict_count=$(echo "$conflicts" | jq 'length')
        
        local message="🚨 Backport Conflict Detected
        
Request ID: $request_id
Conflicts: $conflict_count
Review required: Yes
        
Conflicts:
$(echo "$conflicts" | jq -r '.[] | "- \(.file_path): \(.conflict_type) - \(.description)"')"
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" \
            "$SLACK_WEBHOOK" 2>/dev/null || true
    fi
}

# List backport requests
list_backport_requests() {
    local status_filter="${1:-}"
    local limit="${2:-20}"
    
    local requests_file="${PROJECT_ROOT}/data/backport_requests.json"
    
    if [ ! -f "$requests_file" ]; then
        warn "No backport requests found"
        return 1
    fi
    
    local filter=""
    if [ -n "$status_filter" ]; then
        filter=" | select(.status == \"$status_filter\")"
    fi
    
    jq ".[] $filter | limit($limit)" "$requests_file"
}

# Show backport request details
show_backport_request() {
    local request_id="$1"
    
    local requests_file="${PROJECT_ROOT}/data/backport_requests.json"
    
    if [ ! -f "$requests_file" ]; then
        error "No backport requests found"
        return 1
    fi
    
    local request
    request=$(jq ".[] | select(.id == \"$request_id\")" "$requests_file")
    
    if [ -z "$request" ]; then
        error "Backport request not found: $request_id"
        return 1
    fi
    
    echo "$request" | jq -r '
    "=== BACKPORT REQUEST ===",
    "ID: " + .id,
    "Title: " + .title,
    "Author: " + .author,
    "Status: " + .status,
    "Priority: " + .priority,
    "Created: " + .created_at,
    "",
    "Source Commit: " + .source_commit,
    "Target Branches: " + (.target_branches | join(", ")),
    "",
    "=== FEASIBILITY ANALYSIS ===",
    "Complexity Score: " + (.feasibility.complexity_score | tostring),
    "File Count: " + (.feasibility.file_count | tostring),
    "Has API Changes: " + (.feasibility.has_api_changes | tostring),
    "Has DB Changes: " + (.feasibility.has_db_changes | tostring),
    "",
    "=== CONFLICTS ===",
    (.conflicts | length | tostring) + " conflicts found",
    (.conflicts[] | "- \(.file_path): \(.conflict_type) (\(.severity))")
    '
}

# Show usage
show_usage() {
    cat << EOF
Intelligent Backport Bot with Conflict Detection

USAGE:
    $0 <command> [options]

BACKPORT COMMANDS:
    create <commit_hash> <target_branches> [title] [description]
                                    Create backport request
    list [status] [limit]                    List backport requests
    show <request_id>                           Show request details
    auto <request_id>                           Attempt automatic backport

ANALYSIS COMMANDS:
    analyze <commit_hash> <target_branch>           Analyze commit feasibility
    conflicts <commit_hash> <target_branch>          Detect conflicts
    feasibility <commit_hash> <target_branches>      Full feasibility analysis

EXAMPLES:
    $0 create abc123 "release/1.2" "Fix critical bug" "Backport of critical security fix"
    $0 list pending 10
    $0 show req-123
    $0 analyze abc123 release/1.2
    $0 conflicts abc123 release/1.2

CONFIGURATION:
    Git repository: $GIT_REPO
    Config file: $CONFIG_FILE
    GitHub token: ${GITHUB_TOKEN:-"not set"}
    Slack webhook: ${SLACK_WEBHOOK:-"not set"}

EOF
}

# Main execution
main() {
    case "${1:-}" in
        "create")
            if [ $# -lt 3 ]; then
                error "Usage: $0 create <commit_hash> <target_branches> [title] [description]"
                exit 1
            fi
            create_backport_request "$2" "$3" "${4:-}" "${5:-}"
            ;;
        "list")
            list_backport_requests "${2:-}" "${3:-20}"
            ;;
        "show")
            if [ $# -lt 2 ]; then
                error "Usage: $0 show <request_id>"
                exit 1
            fi
            show_backport_request "$2"
            ;;
        "auto")
            if [ $# -lt 2 ]; then
                error "Usage: $0 auto <request_id>"
                exit 1
            fi
            # Get request details and attempt automatic backport
            local request
            request=$(jq ".[] | select(.id == \"$2\")" "${PROJECT_ROOT}/data/backport_requests.json" 2>/dev/null || echo "{}")
            
            local commit_hash
            commit_hash=$(echo "$request" | jq -r '.source_commit')
            
            local target_branches
            target_branches=$(echo "$request" | jq -r '.target_branches | join(" ")')
            
            attempt_automatic_backport "$2" "$commit_hash" "$target_branches"
            ;;
        "analyze")
            if [ $# -lt 3 ]; then
                error "Usage: $0 analyze <commit_hash> <target_branch>"
                exit 1
            fi
            analyze_commit_feasibility "$2" "$3"
            ;;
        "conflicts")
            if [ $# -lt 3 ]; then
                error "Usage: $0 conflicts <commit_hash> <target_branch>"
                exit 1
            fi
            detect_conflicts "$2" "$3"
            ;;
        "feasibility")
            if [ $# -lt 3 ]; then
                error "Usage: $0 feasibility <commit_hash> <target_branches>"
                exit 1
            fi
            local feasibility
            feasibility=$(analyze_commit_feasibility "$2" "$3")
            local conflicts
            conflicts=$(detect_conflicts "$2" "$3")
            local priority
            priority=$(calculate_backport_priority "$feasibility" "$conflicts")
            
            echo "=== FEASIBILITY ANALYSIS ==="
            echo "$feasibility" | jq -r
            echo ""
            echo "=== CONFLICTS ==="
            echo "$conflicts" | jq -r
            echo ""
            echo "=== PRIORITY ==="
            echo "Priority: $priority"
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"