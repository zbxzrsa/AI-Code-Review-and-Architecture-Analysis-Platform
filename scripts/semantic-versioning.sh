#!/bin/bash

# Semantic Versioning and Changelog Generation
# Provides automated semantic versioning and changelog generation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_ROOT}/config/semantic-versioning.yml"
GIT_REPO="${GIT_REPO:-$(pwd)}"
CHANGELOG_FILE="${PROJECT_ROOT}/CHANGELOG.md"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] SEMVER:${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] SEMVER WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] SEMVER ERROR:${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SEMVER SUCCESS:${NC} $1"
}

# Get current version
get_current_version() {
    local current_tag
    current_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
    
    # Remove 'v' prefix if present
    if [[ $current_tag =~ ^v(.+)$ ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "$current_tag"
    fi
}

# Determine next version
determine_next_version() {
    local current_version="$1"
    local version_type="$2" # major, minor, patch, prerelease
    
    log "Determining next version for: $current_version ($version_type)"
    
    # Parse current version
    if [[ $current_version =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(?:-([0-9A-Za-z-]+))?$ ]]; then
        local major="${BASH_REMATCH[1]}"
        local minor="${BASH_REMATCH[2]}"
        local patch="${BASH_REMATCH[3]}"
        local prerelease="${BASH_REMATCH[4]:-}"
        
        case "$version_type" in
            "major")
                major=$((major + 1))
                minor=0
                patch=0
                prerelease=""
                ;;
            "minor")
                minor=$((minor + 1))
                patch=0
                prerelease=""
                ;;
            "patch")
                patch=$((patch + 1))
                prerelease=""
                ;;
            "prerelease")
                if [ -n "$prerelease" ]; then
                    # Increment prerelease version
                    if [[ $prerelease =~ ^([a-zA-Z]+)([0-9]+)$ ]]; then
                        local prefix="${BASH_REMATCH[1]}"
                        local number="${BASH_REMATCH[2]}"
                        number=$((number + 1))
                        prerelease="${prefix}${number}"
                    else
                        prerelease="alpha.1"
                    fi
                else
                    patch=$((patch + 1))
                    prerelease="alpha.1"
                fi
                ;;
            *)
                error "Invalid version type: $version_type"
                echo "Valid types: major, minor, patch, prerelease"
                return 1
                ;;
        esac
        
        local next_version="$major.$minor.$patch"
        if [ -n "$prerelease" ]; then
            next_version="$next_version-$prerelease"
        fi
        
        echo "$next_version"
    else
        error "Invalid current version format: $current_version"
        return 1
    fi
}

# Analyze commits since last tag
analyze_commits_since_last_tag() {
    log "Analyzing commits since last tag"
    
    local last_tag
    last_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
    
    local commits
    commits=$(git log --pretty=format:"%H|%s|%b|%an" "$last_tag"..HEAD)
    
    local analysis="{
        \"total_commits\": 0,
        \"breaking_changes\": 0,
        \"features\": 0,
        \"fixes\": 0,
        \"chores\": 0,
        \"performance_improvements\": 0,
        \"docs_changes\": 0,
        \"security_fixes\": 0,
        \"commits\": []
    }"
    
    while IFS='|' read -r commit_hash commit_subject commit_body commit_author; do
        if [ -z "$commit_hash" ]; then
            continue
        fi
        
        local commit_type
        commit_type=$(determine_commit_type "$commit_subject")
        
        # Update counters
        analysis=$(echo "$analysis" | jq ".total_commits += 1")
        
        case "$commit_type" in
            "breaking")
                analysis=$(echo "$analysis" | jq ".breaking_changes += 1")
                ;;
            "feature")
                analysis=$(echo "$analysis" | jq ".features += 1")
                ;;
            "fix")
                analysis=$(echo "$analysis" | jq ".fixes += 1")
                ;;
            "chore")
                analysis=$(echo "$analysis" | jq ".chores += 1")
                ;;
            "perf")
                analysis=$(echo "$analysis" | jq ".performance_improvements += 1")
                ;;
            "docs")
                analysis=$(echo "$analysis" | jq ".docs_changes += 1")
                ;;
            "security")
                analysis=$(echo "$analysis" | jq ".security_fixes += 1")
                ;;
        esac
        
        # Add commit details
        local commit_details="{
            \"hash\": \"$commit_hash\",
            \"subject\": \"$commit_subject\",
            \"body\": \"$commit_body\",
            \"author\": \"$commit_author\",
            \"type\": \"$commit_type\"
        }"
        
        analysis=$(echo "$analysis" | jq ".commits += [$commit_details]")
    done
    
    echo "$analysis"
}

# Determine commit type
determine_commit_type() {
    local commit_subject="$1"
    
    # Check for breaking changes first
    if echo "$commit_subject" | grep -qiE "^(feat|fix)!\!.*BREAKING CHANGE|feat.*BREAKING CHANGE"; then
        echo "breaking"
        return
    fi
    
    # Check conventional commit types
    if echo "$commit_subject" | grep -qE "^(feat|feature|new)"; then
        echo "feature"
    elif echo "$commit_subject" | grep -qE "^(fix|bugfix)"; then
        echo "fix"
    elif echo "$commit_subject" | grep -qE "^(chore|maint|update|refactor|style)"; then
        echo "chore"
    elif echo "$commit_subject" | grep -qE "^(perf|performance|optimize)"; then
        echo "perf"
    elif echo "$commit_subject" | grep -qE "^(docs|doc|readme)"; then
        echo "docs"
    elif echo "$commit_subject" | grep -qE "^(security|sec)"; then
        echo "security"
    else
        echo "other"
    fi
}

# Generate changelog entry
generate_changelog_entry() {
    local version="$1"
    local date="$2"
    local analysis="$3"
    
    log "Generating changelog entry for: $version"
    
    local entry="## [$version] - $date"
    
    # Add breaking changes section
    local breaking_count
    breaking_count=$(echo "$analysis" | jq -r '.breaking_changes')
    
    if [ "$breaking_count" -gt 0 ]; then
        entry+="
### 💥 BREAKING CHANGES"
        
        echo "$analysis" | jq -r '.commits[] | select(.type == "breaking") | "- \(.subject) (\(.hash[:8]))"' | while read -r line; do
            entry+="
$line"
        done
    fi
    
    # Add features section
    local feature_count
    feature_count=$(echo "$analysis" | jq -r '.features')
    
    if [ "$feature_count" -gt 0 ]; then
        entry+="
### ✨ Features"
        
        echo "$analysis" | jq -r '.commits[] | select(.type == "feature") | "- \(.subject) (\(.hash[:8]))"' | while read -r line; do
            entry+="
$line"
        done
    fi
    
    # Add fixes section
    local fix_count
    fix_count=$(echo "$analysis" | jq -r '.fixes')
    
    if [ "$fix_count" -gt 0 ]; then
        entry+="
### 🐛 Bug Fixes"
        
        echo "$analysis" | jq -r '.commits[] | select(.type == "fix") | "- \(.subject) (\(.hash[:8]))"' | while read -r line; do
            entry+="
$line"
        done
    fi
    
    # Add performance improvements section
    local perf_count
    perf_count=$(echo "$analysis" | jq -r '.performance_improvements')
    
    if [ "$perf_count" -gt 0 ]; then
        entry+="
### ⚡ Performance Improvements"
        
        echo "$analysis" | jq -r '.commits[] | select(.type == "perf") | "- \(.subject) (\(.hash[:8]))"' | while read -r line; do
            entry+="
$line"
        done
    fi
    
    # Add security fixes section
    local security_count
    security_count=$(echo "$analysis" | jq -r '.security_fixes')
    
    if [ "$security_count" -gt 0 ]; then
        entry+="
### 🔒 Security Fixes"
        
        echo "$analysis" | jq -r '.commits[] | select(.type == "security") | "- \(.subject) (\(.hash[:8]))"' | while read -r line; do
            entry+="
$line"
        done
    fi
    
    # Add other changes section
    local other_count
    other_count=$(echo "$analysis" | jq -r '.chores + .docs_changes')
    
    if [ "$other_count" -gt 0 ]; then
        entry+="
### 📝 Other Changes"
        
        echo "$analysis" | jq -r '.commits[] | select(.type == "chore" or .type == "docs") | "- \(.subject) (\(.hash[:8]))"' | while read -r line; do
            entry+="
$line"
        done
    fi
    
    echo "$entry"
}

# Update changelog file
update_changelog() {
    local new_entry="$1"
    local version="$2"
    
    log "Updating changelog with version: $version"
    
    # Create backup of existing changelog
    if [ -f "$CHANGELOG_FILE" ]; then
        cp "$CHANGELOG_FILE" "${CHANGELOG_FILE}.backup"
    fi
    
    # Create new changelog
    local header="# Changelog
    
All notable changes to this project will be documented in this file.
"
    
    # Add new entry at the top
    echo -e "$header\n\n$new_entry\n$(cat "$CHANGELOG_FILE" 2>/dev/null || echo "")" > "$CHANGELOG_FILE"
    
    # Remove backup if successful
    if [ -f "${CHANGELOG_FILE}.backup" ]; then
        rm "${CHANGELOG_FILE}.backup"
    fi
    
    success "Changelog updated with version: $version"
}

# Create version tag
create_version_tag() {
    local version="$1"
    local message="$2"
    
    log "Creating version tag: v$version"
    
    # Create annotated tag
    git tag -a "v$version" -m "$message" || {
        error "Failed to create tag: v$version"
        return 1
    }
    
    # Push tag to remote
    git push origin "v$version" || {
        error "Failed to push tag: v$version"
        return 1
    }
    
    success "Tag created and pushed: v$version"
}

# Generate version bump commit
generate_version_bump_commit() {
    local version="$1"
    local version_type="$2"
    
    log "Generating version bump commit for: $version ($version_type)"
    
    local commit_message
    case "$version_type" in
        "major")
            commit_message="chore: bump major version to $version"
            ;;
        "minor")
            commit_message="chore: bump minor version to $version"
            ;;
        "patch")
            commit_message="chore: bump patch version to $version"
            ;;
        "prerelease")
            commit_message="chore: bump prerelease version to $version"
            ;;
        *)
            commit_message="chore: bump version to $version"
            ;;
    esac
    
    # Update version files
    update_version_files "$version"
    
    # Create commit
    git add package.json version.txt 2>/dev/null || true
    git commit -m "$commit_message" || {
        error "Failed to create version bump commit"
        return 1
    }
    
    success "Version bump commit created: $commit_message"
}

# Update version files
update_version_files() {
    local version="$1"
    
    # Update package.json
    if [ -f "package.json" ]; then
        temp_file=$(mktemp)
        jq ".version = \"$version\"" package.json > "$temp_file"
        mv "$temp_file" package.json
    fi
    
    # Update version.txt
    echo "$version" > version.txt 2>/dev/null || true
    
    # Update other version files
    local version_files
    version_files=$(yq eval '.version_files[]' "$CONFIG_FILE" 2>/dev/null || echo "[]")
    
    for file in $version_files; do
        file=$(echo "$file" | tr -d "'" | tr -d ' ')
        echo "$version" > "$file" 2>/dev/null || true
    done
}

# Validate version format
validate_version() {
    local version="$1"
    
    # Semantic versioning regex
    local semver_regex="^([0-9]+)\.([0-9]+)\.([0-9]+)(?:-([0-9A-Za-z-]+))?$"
    
    if [[ $version =~ $semver_regex ]]; then
        success "✓ Valid semantic version: $version"
        return 0
    else
        error "✗ Invalid semantic version: $version"
        echo "Expected format: X.Y.Z or X.Y.Z-prerelease"
        return 1
    fi
}

# Get version history
get_version_history() {
    local limit="${1:-10}"
    
    log "Getting version history (last $limit versions)"
    
    git tag --sort=-v:refname | head -"$limit" | while read -r tag; do
        local version=${tag#v}
        local date
        date=$(git log -1 --format="%ai" "$tag" --date=short)
        local message
        message=$(git tag -l --format="%(contents)" "$tag" 2>/dev/null || echo "No message")
        
        echo "v$version ($date): $message"
    done
}

# Compare versions
compare_versions() {
    local version1="$1"
    local version2="$2"
    
    log "Comparing versions: $version1 vs $version2"
    
    # Parse versions
    if [[ $version1 =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(?:-([0-9A-Za-z-]+))?$ ]]; then
        local major1="${BASH_REMATCH[1]}" minor1="${BASH_REMATCH[2]}" patch1="${BASH_REMATCH[3]}" prerelease1="${BASH_REMATCH[4]:-}"
    else
        error "Invalid version format: $version1"
        return 1
    fi
    
    if [[ $version2 =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(?:-([0-9A-Za-z-]+))?$ ]]; then
        local major2="${BASH_REMATCH[1]}" minor2="${BASH_REMATCH[2]}" patch2="${BASH_REMATCH[3]}" prerelease2="${BASH_REMATCH[4]:-}"
    else
        error "Invalid version format: $version2"
        return 1
    fi
    
    # Compare versions
    if [ "$major1" -gt "$major2" ]; then
        echo "major"
    elif [ "$major1" -eq "$major2" ] && [ "$minor1" -gt "$minor2" ]; then
        echo "minor"
    elif [ "$major1" -eq "$major2" ] && [ "$minor1" -eq "$minor2" ] && [ "$patch1" -gt "$patch2" ]; then
        echo "patch"
    elif [ -n "$prerelease1" ] && [ -z "$prerelease2" ]; then
        echo "prerelease"
    elif [ "$major1" -eq "$major2" ] && [ "$minor1" -eq "$minor2" ] && [ "$patch1" -eq "$patch2" ] && [ "$prerelease1" = "$prerelease2" ]; then
        echo "equal"
    else
        echo "lower"
    fi
}

# Show usage
show_usage() {
    cat << EOF
Semantic Versioning and Changelog Generation

USAGE:
    $0 <command> [options]

VERSION COMMANDS:
    current                                   Get current version
    bump <type>                             Bump version (major, minor, patch, prerelease)
    release <type>                           Create release with changelog
    changelog <version> <date> <analysis>     Generate changelog entry
    tag <version> <message>                   Create version tag
    validate <version>                        Validate version format
    history [limit]                           Show version history
    compare <version1> <version2>              Compare two versions

EXAMPLES:
    $0 current
    $0 bump minor
    $0 release minor
    $0 changelog v1.2.3 "2024-01-15" '{"features": 3, "fixes": 2}'
    $0 tag v1.2.3 "Release version 1.2.3"
    $0 validate v1.2.3
    $0 history 20
    $0 compare v1.2.3 v1.3.0

CONFIGURATION:
    Config file: $CONFIG_FILE
    Git repository: $GIT_REPO
    Changelog file: $CHANGELOG_FILE

EOF
}

# Main execution
main() {
    case "${1:-}" in
        "current")
            get_current_version
            ;;
        "bump")
            if [ $# -lt 2 ]; then
                error "Usage: $0 bump <type>"
                exit 1
            fi
            
            local current_version
            current_version=$(get_current_version)
            
            local next_version
            next_version=$(determine_next_version "$current_version" "$2")
            
            generate_version_bump_commit "$next_version" "$2"
            ;;
        "release")
            if [ $# -lt 2 ]; then
                error "Usage: $0 release <type>"
                exit 1
            fi
            
            local current_version
            current_version=$(get_current_version)
            
            local next_version
            next_version=$(determine_next_version "$current_version" "$2")
            
            # Analyze commits
            local analysis
            analysis=$(analyze_commits_since_last_tag)
            
            # Generate changelog entry
            local changelog_entry
            changelog_entry=$(generate_changelog_entry "$next_version" "$(date +%Y-%m-%d)" "$analysis")
            
            # Update changelog
            update_changelog "$changelog_entry" "$next_version"
            
            # Create version bump commit
            generate_version_bump_commit "$next_version" "$2"
            
            # Create tag
            create_version_tag "$next_version" "Release $next_version"
            
            success "Release $next_version completed successfully"
            ;;
        "changelog")
            if [ $# -lt 3 ]; then
                error "Usage: $0 changelog <version> <date> <analysis>"
                exit 1
            fi
            
            local changelog_entry
            changelog_entry=$(generate_changelog_entry "$2" "$3" "$4")
            echo "$changelog_entry"
            ;;
        "tag")
            if [ $# -lt 3 ]; then
                error "Usage: $0 tag <version> <message>"
                exit 1
            fi
            
            create_version_tag "$2" "$3"
            ;;
        "validate")
            if [ $# -lt 2 ]; then
                error "Usage: $0 validate <version>"
                exit 1
            fi
            
            validate_version "$2"
            ;;
        "history")
            get_version_history "${2:-10}"
            ;;
        "compare")
            if [ $# -lt 3 ]; then
                error "Usage: $0 compare <version1> <version2>"
                exit 1
            fi
            
            compare_versions "$2" "$3"
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"