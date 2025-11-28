#!/bin/bash

# Container Security Management Script
# This script manages container security scanning, compliance checking, and runtime monitoring

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

# Function to install security tools
install_tools() {
    print_status "Installing container security tools..."
    
    # Docker tools
    if command_exists docker; then
        print_status "Docker found"
    else
        print_error "Docker not found. Please install Docker."
        exit 1
    fi
    
    # Install Trivy
    if ! command_exists trivy; then
        print_status "Installing Trivy..."
        sudo apt-get update
        sudo apt-get install wget apt-transport-https gnupg lsb-release
        wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key
        sudo apt-key add public.key
        echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
        sudo apt-get update
        sudo apt-get install trivy
        print_success "Trivy installed"
    else
        print_status "Trivy is already installed"
    fi
    
    # Install Docker Bench
    if ! command_exists docker-bench-security; then
        print_status "Installing Docker Bench..."
        curl -sSfL https://raw.githubusercontent.com/docker/docker-bench-security/master/docker-bench-security.sh | sudo sh -s -- -b /usr/local/bin
        print_success "Docker Bench installed"
    else
        print_status "Docker Bench is already installed"
    fi
    
    # Install additional tools
    if command_exists apt-get; then
        print_status "Installing additional security tools..."
        sudo apt-get install -y \
            clamav \
            chkrootkit \
            rkhunter \
            lynis \
            lsof \
            net-tools \
            curl \
            jq
        print_success "Additional security tools installed"
    fi
}

# Function to build container images
build_images() {
    local backend_image=${1:-"ai-review-backend:latest"}
    local frontend_image=${2:-"ai-review-frontend:latest"}
    
    print_status "Building container images..."
    
    # Build backend image
    if [ -d "backend" ]; then
        print_status "Building backend image..."
        docker build \
            -t "$backend_image" \
            -f backend/Dockerfile.security \
            backend/
        print_success "Backend image built: $backend_image"
    fi
    
    # Build frontend image
    if [ -d "frontend" ]; then
        print_status "Building frontend image..."
        docker build \
            -t "$frontend_image" \
            -f frontend/Dockerfile \
            frontend/
        print_success "Frontend image built: $frontend_image"
    fi
}

# Function to scan for vulnerabilities
scan_vulnerabilities() {
    local image_name=${1:-"ai-review-backend:latest"}
    local output_dir=${2:-"security-scans"}
    mkdir -p "$output_dir"
    
    print_status "Scanning container image for vulnerabilities..."
    
    if command_exists trivy; then
        trivy image \
            --format json \
            --output "$output_dir/trivy-results.json" \
            --exit-code 0 \
            "$image_name"
        
        print_success "Vulnerability scan completed: $output_dir/trivy-results.json"
        
        # Display summary
        if command_exists jq; then
            local critical=$(jq '.Results[0].Vulnerabilities | map(select(.Severity == "CRITICAL")) | length' "$output_dir/trivy-results.json" 2>/dev/null || echo "0")
            local high=$(jq '.Results[0].Vulnerabilities | map(select(.Severity == "HIGH")) | length' "$output_dir/trivy-results.json" 2>/dev/null || echo "0")
            local medium=$(jq '.Results[0].Vulnerabilities | map(select(.Severity == "MEDIUM")) | length' "$output_dir/trivy-results.json" 2>/dev/null || echo "0")
            local low=$(jq '.Results[0].Vulnerabilities | map(select(.Severity == "LOW")) | length' "$output_dir/trivy-results.json" 2>/dev/null || echo "0")
            
            echo ""
            print_status "Vulnerability Summary:"
            echo "  🔴 Critical: $critical"
            echo "  🟠 High: $high"
            echo "  🟡 Medium: $medium"
            echo "  🟢 Low: $low"
            echo "  📊 Total: $((critical + high + medium + low))"
            
            if [ "$critical" -gt 0 ] || [ "$high" -gt 0 ]; then
                print_warning "Critical or High vulnerabilities found!"
                return 1
            fi
        fi
    else
        print_error "Trivy not found. Install with: ./container-security.sh install"
        exit 1
    fi
}

# Function to run CIS benchmark
run_cis_benchmark() {
    local container_name=${1:-"ai-review-backend-test"}
    local output_dir=${2:-"security-scans"}
    mkdir -p "$output_dir"
    
    print_status "Running CIS Docker Benchmark..."
    
    if command_exists docker-bench-security; then
        # Run container for testing
        docker run -d --name "$container_name" ai-review-backend:latest
        
        # Run CIS benchmark
        docker-bench-security \
            --format json \
            --no-color \
            --container "$container_name" \
            > "$output_dir/cis-benchmark.json"
        
        # Stop test container
        docker stop "$container_name"
        docker rm "$container_name"
        
        print_success "CIS benchmark completed: $output_dir/cis-benchmark.json"
        
        # Calculate compliance score
        if command_exists python3; then
            python3 << EOF
import json
import sys

try:
    with open('$output_dir/cis-benchmark.json', 'r') as f:
        results = json.load(f)
except FileNotFoundError:
    print("No CIS benchmark results found")
    sys.exit(0)

# Calculate compliance score
total_tests = 0
passed_tests = 0

for test in results.get('tests', []):
    total_tests += 1
    if test.get('result', False):
        passed_tests += 1

compliance_score = (passed_tests / total_tests * 100) if total_tests > 0 else 0

# Generate summary
summary = {
    'compliance_score': round(compliance_score, 1),
    'total_tests': total_tests,
    'passed_tests': passed_tests,
    'failed_tests': total_tests - passed_tests,
    'level': 'Level 1' if compliance_score >= 80 else 'Level 2' if compliance_score >= 60 else 'Not Compliant'
}

with open('$output_dir/cis-compliance-summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"CIS Compliance Score: {summary['compliance_score']}%")
print(f"Compliance Level: {summary['level']}")
EOF
        fi
    else
        print_error "Docker Bench not found. Install with: ./container-security.sh install"
        exit 1
    fi
}

# Function to check runtime security
check_runtime_security() {
    local output_dir=${1:-"security-scans"}
    mkdir -p "$output_dir"
    
    print_status "Checking runtime security..."
    
    # Check running containers
    print_status "Checking running containers..."
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" > "$output_dir/running-containers.txt"
    
    # Check for suspicious processes
    print_status "Checking for suspicious processes..."
    ps aux | grep -E "(nc|ncat|telnet)" > "$output_dir/suspicious-processes.txt" || true
    
    # Check open ports
    print_status "Checking open ports..."
    netstat -tuln | grep LISTEN > "$output_dir/open-ports.txt" || true
    
    # Check file permissions
    print_status "Checking file permissions..."
    find /var/lib/docker -type f -perm /o=w -ls > "$output_dir/world-writable-files.txt" 2>/dev/null || true
    
    # Check for setuid files
    print_status "Checking for setuid files..."
    find /var/lib/docker -type f -perm -4000 -ls > "$output_dir/setuid-files.txt" 2>/dev/null || true
    
    # Generate security report
    cat > "$output_dir/runtime-security-report.json" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "containers": "$(wc -l < "$output_dir/running-containers.txt")",
  "suspicious_processes": "$(wc -l < "$output_dir/suspicious-processes.txt")",
  "open_ports": "$(wc -l < "$output_dir/open-ports.txt")",
  "world_writable_files": "$(wc -l < "$output_dir/world-writable-files.txt")",
  "setuid_files": "$(wc -l < "$output_dir/setuid-files.txt")"
}
EOF
    
    print_success "Runtime security check completed: $output_dir/runtime-security-report.json"
}

# Function to generate security report
generate_report() {
    local output_dir=${1:-"security-scans"}
    
    print_status "Generating comprehensive security report..."
    
    cat > "$output_dir/security-summary.md" << EOF
# Container Security Report

Generated: $(date)

## Executive Summary

This report provides a comprehensive overview of container security for the AI Code Review Platform.

## Vulnerability Scan Results

EOF
    
    # Add vulnerability summary if available
    if [ -f "$output_dir/trivy-results.json" ]; then
        echo "### Vulnerability Findings" >> "$output_dir/security-summary.md"
        echo '```json' >> "$output_dir/security-summary.md"
        cat "$output_dir/trivy-results.json" >> "$output_dir/security-summary.md"
        echo '```' >> "$output_dir/security-summary.md"
        echo "" >> "$output_dir/security-summary.md"
    fi
    
    # Add CIS compliance if available
    if [ -f "$output_dir/cis-compliance-summary.json" ]; then
        echo "### CIS Compliance" >> "$output_dir/security-summary.md"
        echo '```json' >> "$output_dir/security-summary.md"
        cat "$output_dir/cis-compliance-summary.json" >> "$output_dir/security-summary.md"
        echo '```' >> "$output_dir/security-summary.md"
        echo "" >> "$output_dir/security-summary.md"
    fi
    
    # Add runtime security if available
    if [ -f "$output_dir/runtime-security-report.json" ]; then
        echo "### Runtime Security" >> "$output_dir/security-summary.md"
        echo '```json' >> "$output_dir/security-summary.md"
        cat "$output_dir/runtime-security-report.json" >> "$output_dir/security-summary.md"
        echo '```' >> "$output_dir/security-summary.md"
        echo "" >> "$output_dir/security-summary.md"
    fi
    
    cat >> "$output_dir/security-summary.md" << EOF
## Recommendations

### Immediate Actions
1. **Address Critical Vulnerabilities**: Immediately fix any CRITICAL or HIGH severity vulnerabilities
2. **Improve CIS Compliance**: Work on failing CIS benchmark tests
3. **Review Runtime Security**: Address any runtime security concerns

### Long-term Improvements
1. **Regular Scanning**: Implement regular vulnerability scanning in CI/CD
2. **Security Monitoring**: Deploy runtime security monitoring
3. **Compliance Automation**: Automate compliance checking and reporting

## Next Steps

1. Review detailed findings in attached JSON files
2. Prioritize remediation based on risk assessment
3. Implement security improvements
4. Schedule regular security assessments

---
*Report generated by container security management script*
EOF
    
    print_success "Security report generated: $output_dir/security-summary.md"
}

# Function to show help
show_help() {
    echo "Container Security Management Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  install     Install security tools"
    echo "  build       Build container images"
    echo "  scan        Scan images for vulnerabilities"
    echo "  benchmark   Run CIS Docker benchmark"
    echo "  runtime     Check runtime security"
    echo "  report      Generate comprehensive security report"
    echo "  full        Run complete security analysis"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 install                    # Install all security tools"
    echo "  $0 build                      # Build all container images"
    echo "  $0 scan ai-review-backend     # Scan specific image"
    echo "  $0 benchmark                  # Run CIS benchmark"
    echo "  $0 full                       # Run complete analysis"
}

# Main script logic
COMMAND=${1:-help}
OUTPUT_DIR="security-scans"

case $COMMAND in
    install)
        install_tools
        ;;
    build)
        build_images
        ;;
    scan)
        scan_vulnerabilities "${2:-ai-review-backend:latest}" "$OUTPUT_DIR"
        ;;
    benchmark)
        run_cis_benchmark "${2:-ai-review-backend-test}" "$OUTPUT_DIR"
        ;;
    runtime)
        check_runtime_security "$OUTPUT_DIR"
        ;;
    report)
        generate_report "$OUTPUT_DIR"
        ;;
    full)
        print_status "Running complete container security analysis..."
        install_tools
        build_images
        scan_vulnerabilities "ai-review-backend:latest" "$OUTPUT_DIR"
        scan_vulnerabilities "ai-review-frontend:latest" "$OUTPUT_DIR"
        run_cis_benchmark "ai-review-backend-test" "$OUTPUT_DIR"
        check_runtime_security "$OUTPUT_DIR"
        generate_report "$OUTPUT_DIR"
        print_success "Complete container security analysis finished"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac