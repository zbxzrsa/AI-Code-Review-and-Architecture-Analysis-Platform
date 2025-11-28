#!/bin/bash

# Secret Hygiene Management Script
# This script helps manage secret detection and hygiene practices

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install tools if missing
install_tool() {
    local tool=$1
    local install_cmd=$2
    
    if ! command_exists "$tool"; then
        print_status "Installing $tool..."
        eval "$install_cmd"
        print_success "$tool installed successfully"
    else
        print_status "$tool is already installed"
    fi
}

# Function to run secret scans
run_secret_scans() {
    print_status "Running comprehensive secret scans..."
    
    # Gitleaks scan
    if command_exists gitleaks; then
        print_status "Running Gitleaks scan..."
        gitleaks detect --config .gitleaks.toml --verbose --report-path gitleaks-report.json || true
        print_success "Gitleaks scan completed"
    else
        print_warning "Gitleaks not found. Install with: go install github.com/gitleaks/gitleaks/v8/cmd/gitleaks@latest"
    fi
    
    # TruffleHog scan
    if command_exists trufflehog; then
        print_status "Running TruffleHog scan..."
        trufflehog git . --only-verified --json --output trufflehog-report.json || true
        print_success "TruffleHog scan completed"
    else
        print_warning "TruffleHog not found. Install with: go install github.com/trufflesecurity/trufflehog/v3/cmd/trufflehog@latest"
    fi
    
    # Detect-secrets scan
    if command_exists detect-secrets; then
        print_status "Running detect-secrets scan..."
        if [ -f .secrets.baseline ]; then
            detect-secrets scan --baseline .secrets.baseline --all-files --report-file detect-secrets-report.json || true
        else
            detect-secrets scan --all-files --report-file detect-secrets-report.json || true
        fi
        print_success "Detect-secrets scan completed"
    else
        print_warning "detect-secrets not found. Install with: pip install detect-secrets"
    fi
}

# Function to create/update baseline
update_baseline() {
    print_status "Updating secret detection baseline..."
    
    if command_exists detect-secrets; then
        detect-secrets scan --all-files --baseline .secrets.baseline
        print_success "Baseline updated successfully"
        print_warning "Review .secrets.baseline before committing!"
    else
        print_error "detect-secrets not found. Install with: pip install detect-secrets"
        exit 1
    fi
}

# Function to check for common secret patterns
quick_scan() {
    print_status "Running quick secret pattern scan..."
    
    local secrets_found=false
    
    # Check for common secret patterns in environment files
    if find . -name "*.env*" -not -path "./node_modules/*" -not -path "./.git/*" -exec grep -lE "(password|secret|key|token|api_key|private_key|access_key).*=.*" {} \; 2>/dev/null; then
        print_error "Potential secrets found in environment files!"
        secrets_found=true
    fi
    
    # Check for hardcoded secrets in code
    if grep -rE "(sk-[a-zA-Z0-9]{48}|ghp_[a-zA-Z0-9]{36}|AKIA[0-9A-Z]{16})" . --include="*.py" --include="*.js" --include="*.ts" --include="*.jsx" --include="*.tsx" --exclude-dir=node_modules --exclude-dir=.git 2>/dev/null; then
        print_error "Potential hardcoded secrets found in source code!"
        secrets_found=true
    fi
    
    # Check for private keys
    if find . -name "*.pem" -o -name "*.key" -not -path "./node_modules/*" -not -path "./.git/*" 2>/dev/null | grep -q .; then
        print_error "Private key files found!"
        secrets_found=true
    fi
    
    if [ "$secrets_found" = false ]; then
        print_success "No obvious secrets found in quick scan"
    fi
}

# Function to install pre-commit hooks
install_hooks() {
    print_status "Installing pre-commit hooks..."
    
    if command_exists pre-commit; then
        pre-commit install
        pre-commit install --hook-type commit-msg
        print_success "Pre-commit hooks installed"
    else
        print_error "pre-commit not found. Install with: pip install pre-commit"
        exit 1
    fi
}

# Function to validate secret hygiene
validate_hygiene() {
    print_status "Validating secret hygiene practices..."
    
    local issues=0
    
    # Check if .gitignore exists and has proper entries
    if [ ! -f .gitignore ]; then
        print_warning ".gitignore not found"
        ((issues++))
    else
        if ! grep -q "\.env" .gitignore; then
            print_warning ".env files not in .gitignore"
            ((issues++))
        fi
        
        if ! grep -q "\.secrets.baseline" .gitignore; then
            print_warning ".secrets.baseline not in .gitignore"
            ((issues++))
        fi
        
        if ! grep -q "*.pem" .gitignore; then
            print_warning "*.pem files not in .gitignore"
            ((issues++))
        fi
        
        if ! grep -q "*.key" .gitignore; then
            print_warning "*.key files not in .gitignore"
            ((issues++))
        fi
    fi
    
    # Check if secret detection configs exist
    if [ ! -f .gitleaks.toml ]; then
        print_warning ".gitleaks.toml not found"
        ((issues++))
    fi
    
    if [ ! -f .pre-commit-config.yaml ]; then
        print_warning ".pre-commit-config.yaml not found"
        ((issues++))
    fi
    
    # Check GitHub token permissions
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        print_status "GitHub token detected - ensure it has minimal permissions"
    fi
    
    if [ $issues -eq 0 ]; then
        print_success "Secret hygiene validation passed!"
    else
        print_warning "Found $issues secret hygiene issues to address"
    fi
}

# Function to show help
show_help() {
    echo "Secret Hygiene Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  scan        Run comprehensive secret scans"
    echo "  baseline    Update secret detection baseline"
    echo "  quick       Run quick secret pattern scan"
    echo "  install     Install pre-commit hooks"
    echo "  validate    Validate secret hygiene practices"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 scan       # Run all secret scans"
    echo "  $0 baseline    # Update baseline"
    echo "  $0 validate    # Check hygiene practices"
}

# Main script logic
case "${1:-help}" in
    scan)
        run_secret_scans
        ;;
    baseline)
        update_baseline
        ;;
    quick)
        quick_scan
        ;;
    install)
        install_hooks
        ;;
    validate)
        validate_hygiene
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac