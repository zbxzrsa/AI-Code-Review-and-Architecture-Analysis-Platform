#!/bin/bash

# Supply Chain Security Management Script
# This script manages SBOM generation, vulnerability scanning, and license compliance

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
install_tools() {
    print_status "Installing supply chain security tools..."
    
    # Python tools
    if ! command_exists pip; then
        print_error "pip not found. Please install Python and pip."
        exit 1
    fi
    
    print_status "Installing Python security tools..."
    pip install cyclonedx-bom grype fossa-cli pip-licenses
    
    # Node.js tools
    if command_exists npm; then
        print_status "Installing Node.js security tools..."
        npm install -g @cyclonedx/cyclonedx-cli
        npm audit --audit-level moderate || true
    fi
    
    # Go tools
    if command_exists go; then
        print_status "Installing Go security tools..."
        go install github.com/anchore/grype/cmd/grype@latest
        go install github.com/anchore/syft/cmd/syft@latest
    fi
    
    print_success "Security tools installation completed"
}

# Function to generate SBOM
generate_sbom() {
    local output_dir=${1:-"sbom-output"}
    mkdir -p "$output_dir"
    
    print_status "Generating Software Bill of Materials (SBOM)..."
    
    # Generate Python SBOM
    if [ -d "backend" ]; then
        print_status "Generating Python SBOM..."
        cyclonedx-py -o "$output_dir/backend-sbom.json" -i backend/
        print_success "Python SBOM generated: $output_dir/backend-sbom.json"
    fi
    
    # Generate Frontend SBOM
    if [ -d "frontend" ]; then
        print_status "Generating Frontend SBOM..."
        cd frontend
        if command_exists npx && [ -f "package.json" ]; then
            npx @cyclonedx/cyclonedx-cli -o "../$output_dir/frontend-sbom.json" -p .
            cd ..
            print_success "Frontend SBOM generated: $output_dir/frontend-sbom.json"
        else
            print_warning "Frontend package.json not found or npx not available"
            cd ..
        fi
    fi
    
    # Generate combined SBOM
    if [ -f "$output_dir/backend-sbom.json" ] && [ -f "$output_dir/frontend-sbom.json" ]; then
        print_status "Generating combined SBOM..."
        jq -s 'add(.[0] | .components) + add(.[1] | .components)' \
            "$output_dir/backend-sbom.json" "$output_dir/frontend-sbom.json" > "$output_dir/combined-sbom.json"
        print_success "Combined SBOM generated: $output_dir/combined-sbom.json"
    fi
}

# Function to scan for vulnerabilities
scan_vulnerabilities() {
    local sbom_file=${1:-"sbom-output/combined-sbom.json"}
    
    if [ ! -f "$sbom_file" ]; then
        print_error "SBOM file not found: $sbom_file"
        print_status "Run 'generate' first to create SBOM"
        exit 1
    fi
    
    print_status "Scanning for vulnerabilities..."
    
    if command_exists grype; then
        grype "sbom:$sbom_file" \
            --output json \
            --file vulnerability-report.json \
            --add-cpes-if-none
        
        print_success "Vulnerability scan completed: vulnerability-report.json"
        
        # Display summary
        if command_exists jq; then
            local critical=$(jq '[.matches[] | select(.vulnerability.severity == "Critical")] | length' vulnerability-report.json 2>/dev/null || echo "0")
            local high=$(jq '[.matches[] | select(.vulnerability.severity == "High")] | length' vulnerability-report.json 2>/dev/null || echo "0")
            local medium=$(jq '[.matches[] | select(.vulnerability.severity == "Medium")] | length' vulnerability-report.json 2>/dev/null || echo "0")
            local low=$(jq '[.matches[] | select(.vulnerability.severity == "Low")] | length' vulnerability-report.json 2>/dev/null || echo "0")
            
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
        print_error "Grype not found. Install with: go install github.com/anchore/grype/cmd/grype@latest"
        exit 1
    fi
}

# Function to check license compliance
check_licenses() {
    print_status "Checking license compliance..."
    
    if [ -d "backend" ]; then
        cd backend
        
        if command_exists pip-licenses; then
            print_status "Analyzing Python package licenses..."
            pip-licenses --from=mixed --format=json > ../license-report.json 2>/dev/null || true
            
            # Check for problematic licenses
            if command_exists node && [ -f "../license-report.json" ]; then
                node -e "
                    const fs = require('fs');
                    const licenses = JSON.parse(fs.readFileSync('../license-report.json', 'utf8'));
                    const problematic = ['GPL-3.0', 'AGPL-3.0', 'LGPL-3.0', 'SSPL', 'Proprietary'];
                    const copyleft = ['GPL-2.0', 'GPL-3.0', 'LGPL-2.0', 'LGPL-3.0', 'AGPL-3.0'];
                    
                    const problematicPkgs = licenses.filter(pkg => 
                        pkg.license && problematic.some(license => 
                            pkg.license.includes(license)
                        )
                    );
                    
                    const copyleftPkgs = licenses.filter(pkg => 
                        pkg.license && copyleft.some(license => 
                            pkg.license.includes(license)
                        )
                    );
                    
                    console.log('\\n📄 License Analysis Results:');
                    console.log('  Total packages:', licenses.length);
                    console.log('  Problematic licenses:', problematicPkgs.length);
                    console.log('  Copyleft licenses:', copyleftPkgs.length);
                    
                    if (problematicPkgs.length > 0) {
                        console.log('\\n❌ Problematic licenses found:');
                        problematicPkgs.forEach(pkg => {
                            console.log('  -', pkg.name + ':', pkg.license);
                        });
                    }
                    
                    if (copyleftPkgs.length > 0) {
                        console.log('\\n⚠️  Copyleft licenses found:');
                        copyleftPkgs.forEach(pkg => {
                            console.log('  -', pkg.name + ':', pkg.license);
                        });
                    }
                    
                    if (problematicPkgs.length === 0) {
                        console.log('\\n✅ No problematic licenses found');
                    }
                "
            fi
        else
            print_warning "pip-licenses not found. Install with: pip install pip-licenses"
        fi
        
        cd ..
    fi
    
    # Check frontend licenses
    if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
        cd frontend
        
        if command_exists npx; then
            print_status "Analyzing frontend package licenses..."
            npx license-checker --onlyAllow 'MIT;Apache-2.0;BSD;ISC;0BSD' --excludePrivatePackages || true
        fi
        
        cd ..
    fi
}

# Function to detect dependency drift
detect_drift() {
    local base_branch=${1:-"main"}
    local compare_branch=${2:-"HEAD"}
    
    print_status "Detecting dependency drift between $base_branch and $compare_branch..."
    
    # Create temporary directory for comparison
    local temp_dir=$(mktemp -d)
    mkdir -p "$temp_dir/base" "$temp_dir/compare"
    
    # Checkout base branch
    git checkout "$base_branch" 2>/dev/null || {
        print_error "Failed to checkout base branch: $base_branch"
        rm -rf "$temp_dir"
        exit 1
    }
    
    # Generate base SBOM
    if [ -d "backend" ]; then
        cyclonedx-py -o "$temp_dir/base/backend-sbom.json" -i backend/ 2>/dev/null || true
    fi
    
    if [ -d "frontend" ]; then
        cd frontend
        npx @cyclonedx/cyclonedx-cli -o "../$temp_dir/base/frontend-sbom.json" -p . 2>/dev/null || true
        cd ..
    fi
    
    # Checkout compare branch
    git checkout "$compare_branch" 2>/dev/null || {
        print_error "Failed to checkout compare branch: $compare_branch"
        rm -rf "$temp_dir"
        exit 1
    }
    
    # Generate compare SBOM
    if [ -d "backend" ]; then
        cyclonedx-py -o "$temp_dir/compare/backend-sbom.json" -i backend/ 2>/dev/null || true
    fi
    
    if [ -d "frontend" ]; then
        cd frontend
        npx @cyclonedx/cyclonedx-cli -o "../$temp_dir/compare/frontend-sbom.json" -p . 2>/dev/null || true
        cd ..
    fi
    
    # Compare dependencies
    print_status "Analyzing dependency changes..."
    
    if [ -f "$temp_dir/base/backend-sbom.json" ] && [ -f "$temp_dir/compare/backend-sbom.json" ]; then
        python3 << EOF
import json

with open('$temp_dir/base/backend-sbom.json') as f:
    base_backend = json.load(f)
with open('$temp_dir/compare/backend-sbom.json') as f:
    compare_backend = json.load(f)

base_deps = {comp['name'] for comp in base_backend.get('components', [])}
compare_deps = {comp['name'] for comp in compare_backend.get('components', [])}

added = compare_deps - base_deps
removed = base_deps - compare_deps

print(f"Backend dependencies:")
print(f"  Added: {len(added)}")
print(f"  Removed: {len(removed)}")

if added:
    print("  New dependencies:")
    for dep in sorted(list(added))[:10]:
        print(f"    + {dep}")
    if len(added) > 10:
        print(f"    ... and {len(added) - 10} more")

if removed:
    print("  Removed dependencies:")
    for dep in sorted(list(removed))[:10]:
        print(f"    - {dep}")
    if len(removed) > 10:
        print(f"    ... and {len(removed) - 10} more")

with open('dependency-drift.json', 'w') as f:
    json.dump({
        'backend': {
            'added': list(added),
            'removed': list(removed)
        }
    }, f, indent=2)
EOF
    fi
    
    # Cleanup
    rm -rf "$temp_dir"
    
    # Return to original branch
    git checkout - 2>/dev/null
    
    print_success "Dependency drift analysis completed: dependency-drift.json"
}

# Function to submit SBOM to external systems
submit_sbom() {
    local sbom_file=${1:-"sbom-output/combined-sbom.json"}
    local dt_url=${2:-"${DEPENDENCY_TRACK_URL}"}
    local api_key=${3:-"${DEPENDENCY_TRACK_API_KEY}"}}
    
    if [ ! -f "$sbom_file" ]; then
        print_error "SBOM file not found: $sbom_file"
        exit 1
    fi
    
    if [ -z "$dt_url" ] || [ -z "$api_key" ]; then
        print_warning "Dependency Track URL or API key not configured"
        print_status "Set DEPENDENCY_TRACK_URL and DEPENDENCY_TRACK_API_KEY environment variables"
        return 1
    fi
    
    print_status "Submitting SBOM to Dependency Track..."
    
    curl_response=$(curl -s -w "%{http_code}" \
        -X POST \
        -H "X-Api-Key: $api_key" \
        -H "Content-Type: application/json" \
        --data-binary @"$sbom_file" \
        "$dt_url/api/v1/bom")
    
    http_code=$(echo "$curl_response" | tail -c 3)
    
    if [ "$http_code" = "201" ] || [ "$http_code" = "200" ]; then
        print_success "SBOM submitted successfully to Dependency Track"
    else
        print_error "Failed to submit SBOM. HTTP code: $http_code"
        return 1
    fi
}

# Function to show help
show_help() {
    echo "Supply Chain Security Management Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  install     Install security tools"
    echo "  generate    Generate SBOM for all components"
    echo "  scan        Scan SBOM for vulnerabilities"
    echo "  licenses    Check license compliance"
    echo "  drift       Detect dependency drift"
    echo "  submit      Submit SBOM to Dependency Track"
    echo "  full        Run complete supply chain analysis"
    echo "  help        Show this help message"
    echo ""
    echo "Options:"
    echo "  --output-dir DIR    Output directory for generated files (default: sbom-output)"
    echo "  --sbom FILE        SBOM file to scan (default: sbom-output/combined-sbom.json)"
    echo "  --base BRANCH      Base branch for drift detection (default: main)"
    echo "  --compare BRANCH   Compare branch for drift detection (default: HEAD)"
    echo ""
    echo "Examples:"
    echo "  $0 install                    # Install all security tools"
    echo "  $0 generate --output-dir ./sbom  # Generate SBOM to custom directory"
    echo "  $0 scan --sbom ./sbom.json    # Scan specific SBOM file"
    echo "  $0 drift --base main --compare develop  # Compare main vs develop"
    echo "  $0 full                        # Run complete analysis"
}

# Main script logic
COMMAND=${1:-help}
OUTPUT_DIR="sbom-output"
SBOM_FILE="sbom-output/combined-sbom.json"
BASE_BRANCH="main"
COMPARE_BRANCH="HEAD"

# Parse arguments
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --sbom)
            SBOM_FILE="$2"
            shift 2
            ;;
        --base)
            BASE_BRANCH="$2"
            shift 2
            ;;
        --compare)
            COMPARE_BRANCH="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

case $COMMAND in
    install)
        install_tools
        ;;
    generate)
        generate_sbom "$OUTPUT_DIR"
        ;;
    scan)
        scan_vulnerabilities "$SBOM_FILE"
        ;;
    licenses)
        check_licenses
        ;;
    drift)
        detect_drift "$BASE_BRANCH" "$COMPARE_BRANCH"
        ;;
    submit)
        submit_sbom "$SBOM_FILE"
        ;;
    full)
        print_status "Running complete supply chain security analysis..."
        install_tools
        generate_sbom "$OUTPUT_DIR"
        scan_vulnerabilities "$SBOM_FILE"
        check_licenses
        print_success "Complete supply chain analysis finished"
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