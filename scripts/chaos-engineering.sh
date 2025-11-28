name: Chaos Drills v2

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  workflow_dispatch:
    inputs:
      experiment_type:
        description: 'Type of chaos experiment'
        required: false
        type: string
        default: 'network_latency'
      experiment_name:
        description: 'Name of the chaos experiment'
        required: true
        type: string
        default: 'network_latency'
      budget:
        description: 'Maximum budget for experiment'
        required: false
        type: number
        default: 100.00
      duration:
        description: 'Duration in minutes'
        required: false
        type: number
        default: 60
      dry_run:
        description: 'Dry run mode'
        required: false
        type: boolean
        default: false
      emergency_stop:
        description: 'Emergency stop'
        required: false
        type: boolean
        default: false

jobs:
  chaos-experiment:
    name: Chaos Experiment
    runs-on: ubuntu-latest
    permissions:
      contents: read
      actions: read
      pull-requests: write
    environment:
      CHAOS_NAMESPACE: chaos-experiments
      CHAOS_TARGET: ai-review-backend
      CHAOS_API_URL: http://localhost:8000
      CHAOS_AUTH_TOKEN: ${{ secrets.CHAOS_AUTH_TOKEN }}
      EXPERIMENT_TYPE: ${{ github.event.inputs.experiment_type }}
      EXPERIMENT_NAME: ${{ github.event.inputs.experiment_name }}
      BUDGET: ${{ github.event.inputs.budget }}
      DURATION: ${{ github.event.inputs.duration }}
      DRY_RUN: ${{ github.event.inputs.dry_run }}"
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Chaos Toolkit
        run: |
          python3 << 'EOF'
          import sys
          
          # Install LitmusChaos if not available
          if ! command_exists litmuschaos; then
              print_status "Installing LitmusChaos..."
              curl -sSfL https://github.com/litmuschaos/releases/latest/download/litmuschaos-linux-amd64.tar.gz | tar -xz
              sudo tar -xzf litmuschaos-linux-amd64.tar.gz -C /usr/local/bin
              sudo ln -s /usr/local/bin/litmuschaos /usr/local/bin/chaos
              print_success "LitmusChaos installed"
          else
              print_status "LitmusChaos already installed"
          fi
          
          # Install additional tools
          if ! command_exists chaosctl; then
              print_status "Installing Chaos Toolkit..."
              curl -sSfL https://github.com/chaos-meshkit/releases/latest/download/chaos-meshkit-linux-amd64.tar.gz | tar -xz
              sudo tar -xzf chaos-meshkit-linux-amd64.tar.gz -C /usr/local/bin/chaos
              sudo ln -s /usr/local/bin/chaos-meshkit /usr/local/bin/chaos
              print_success "Chaos MeshKit installed"
          else
              print_status "Chaos MeshKit already installed"
          fi
          
          # Install Chaos Metrics
          if ! command_exists chaos-metrics; then
              print_status "Installing Chaos Metrics..."
              go install github.com/chaos-metrics@latest
              print_success "Chaos Metrics installed"
          else
              print_status "Chaos Metrics already installed"
          fi
          
          # Install additional tools
          if ! command_exists kubectl; then
              print_status "kubectl not found - some features may not work"
          fi
          
          print_status "Chaos Engineering tools ready"
          EOF

      - name: Validate Experiment Configuration
        run: |
          python3 << 'EOF'
          import sys
          
          experiment_file="$CHAOS_CONFIG/experiments/${{ github.event.inputs.experiment_name }}.yaml"
          
          if [ ! -f "$experiment_file" ]; then
              print_error "Experiment file not found: $experiment_file"
              return 1
          fi
          
          # Validate YAML syntax
          python3 -c "
import yaml
          try:
              with open('$experiment_file', 'r') as f:
                  config = yaml.safe_load(f)
                  print_success "Experiment configuration is valid"
          except yaml.YAMLError as e:
                  print_error "Invalid YAML in $experiment_file: ${e}"
                  return 1
              except Exception as e:
                  print_error "Error parsing $experiment_file: ${e}"
                  return 1
          except Exception as e:
                  print_error "Error parsing $experiment_file: ${e}"
                  return 1
          
          # Validate budget
          if [ -z "$budget" ]; then
              print_error "Budget not specified in experiment configuration"
              return 1
          fi
          
          # Check budget limits
          if [ "$(python3 -c "
import yaml
with open('$experiment_file', 'r') as f:
    config = yaml.safe_load(f)
    budget = config.get('budget', {})
    
    if [ -z "$budget" ]; then
        print_error "Budget not specified in experiment configuration"
        return 1
    fi
    
    # Check cost configuration
    if [ "$(python3 -c "
import yaml
with open('$experiment_file', 'r') as f:
    config = yaml.safe_load(f)
    cost_per_request = config.get('cost_per_request', 0.001)
    max_cost = config.get('max_experiment_cost', 100.00)
    
    if [ "$cost_per_request" -gt 0.01 ]; then
        print_warning "Cost per request exceeds threshold ($cost_per_request > 0.01)"
    else
        print_success "Cost per request within limits ($cost_per_request <= 0.01)"
    fi
          except:
              print_error "Budget validation failed: ${e}"
              return 1
          except Exception as e:
              print_error "Error validating budget: ${e}"
              return 1
          fi
          
          print_success "Budget validation passed"
          return 0
          EOF

      - name: Run Chaos Experiment
        run: |
          python3 << 'EOF'
          experiment_file="$CHAOS_CONFIG/experiments/${ github.event.inputs.experiment_name }}.yaml"
          local dry_run="${{ github.event.inputs.dry_run }}"
          local duration="${ github.event.inputs.duration }"
          
          print_status "Running chaos experiment: $experiment_name"
          
          # Execute experiment
          local cmd="litmuschaos run $experiment_name --budget $max_cost $local duration --dry-run=$dry_run"
          
          if [ "$dry_run" = "true" ]; then
              cmd="$cmd --dry-run"
              print_status "DRY RUN MODE: $cmd"
          else
              cmd="$cmd"
          fi
          
          print_status "Executing: $cmd"
          
          # Execute and capture output
          local output=$(eval "$cmd" 2>&1)
          exit_code=$?
          
          if [ $exit_code -ne 0 ]; then
              print_error "Chaos experiment failed"
              return 1
          fi
          
          # Parse results
          python3 << 'EOF'
          import json
          import sys
          
          try:
              with open('chaos-report.json', 'r') as f:
                  report = json.load(f)
                  
                  # Extract metrics
                  metrics = report.get('metrics', {})
                  
                  # Check for SLO violations
                  violations = []
                  
                  # Budget violations
                  if metrics['budget_used'] > metrics['max_experiment_cost']:
                      violations.append({
                          'type': 'budget_exceeded',
                          'severity': 'critical',
                          'message': f"Experiment exceeded budget of ${metrics['max_experiment_cost']}",
                          'metric': 'budget_used',
                          'current': metrics['budget_used'],
                          'target': metrics['max_experiment_cost']
                      })
                  
                  # Error rate violations
                  if metrics['error_rate'] > 0.01:
                      violations.append({
                          'type': 'error_rate_high',
                          'severity': 'critical',
                          'message': f"Error rate {metrics['error_rate']:.2%} exceeds 1% target",
                          'metric': 'error_rate',
                          'current': metrics['error_rate'],
                          'target': '0.01'
                      })
                  
                  # Latency violations
                  if metrics['duration'] > 1.0:
                      violations.append({
                          'type': 'latency_high',
                          'severity': 'warning',
                          'message': f"95th percentile latency {metrics['duration']:.3f}s exceeds 1.0s target",
                          'metric': 'duration',
                          'current': metrics['duration'],
                          'target': '1.0s'
                      })
                  
                  # Availability violations
                  if metrics['availability'] < 0.99:
                      violations.append({
                          'type': 'availability_low',
                          'severity': 'critical',
                          'message': f"Availability {metrics['availability']:.3f}% below 99% target",
                          'metric': 'availability'],
                          'current': metrics['availability'],
                          'target': '0.99'
                      })
                  
                  # Overall SLO status
                  total_checks = len(report['slo_status'])
                  compliant_checks = sum(1 for status in report['slo_status'].values() if status['status'] == 'OK')
                  compliance_percentage = (compliant_checks / total_checks) * 100
                  
                  print(f"SLO Compliance: {compliance_percentage:.1f}% ({compliant_checks}/{total_checks})")
                  
                  if violations:
                      print_warning "SLO Violations: ${', '.join(violations)}")
                  else
                  print_success "All SLOs are compliant"
                  
                  # Generate report
                  report = {
                      'timestamp': datetime.now().isoformat(),
                      'slo_status': report['slo_status'],
                      'violations': violations,
                      'recommendations': []
                  }
                  
                  # Save report
                  with open('chaos-report.json', 'w') as f:
                      json.dump(report, f, indent=2)
                  
                  print_success "Chaos report generated: chaos-report-$experiment_name.json"
                  print_success "Report saved to chaos-report-$experiment_name.json"
                  
                  # Check for SLO violations
                  violations = check_slo_violations(metrics, budget)
                  if [ ${#violations[@]} ]; then
                      print_warning "SLO Violations detected:"
                      for violation in violations:
                          print_warning "  - ${violation['type']}: ${violation['message']}"
                      done
                  
                  # Generate rollback plan if needed
                  if [any(v['violations[@]})]:
                      plan = generate_rollback_plan("$experiment_name", violations, budget)
                      print "Rollback plan:"
                      echo "$plan" | python3 -m json.tool > chaos-rollback-plan-$experiment_name.json"
                  print_success "Rollback plan saved to chaos-rollback-plan-$experiment_name.json"
                  else
                  print_success "No rollback needed"
                  fi
                  
                  # Check for emergency stop
                  if [ -f "/tmp/chaos-emergency-stop" ]; then
                      print_warning "Emergency stop triggered"
                      print_status "Emergency stop executed"
                  else
                  print_status "No emergency stop needed"
                  fi
                  
              except Exception as e:
                  print_error "Error generating rollback plan: ${e}"
                  return 1
              fi
          EOF
          
          return 0
          EOF

      - name: Generate SLO Report
        if: github.event_name == 'main' && github.ref == 'refs/heads/main'
        run: |
          python3 << 'EOF'
          # Generate comprehensive SLO report
          python3 << 'EOF'
          import json
          import requests
          from datetime import timedelta
          
          # Get current metrics from Prometheus
          prometheus_url = "http://localhost:9090"
          
          def get_slo_metrics():
              metrics = {}
              
              # API metrics
              try:
                  api_response_query = 'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[1h)))'
                  api_response = requests.get(f"{prometheus_url}/api/v1/query", 
                                            json={"query": api_response_query}, timeout=10)
                  api_p95 = float(api_response.json().get('data', {}).get('result', [0])[0])
                  
                  api_error_query = 'sum(rate(http_requests_total{status=~"5.."}[1h])) / sum(rate(http_requests_total[1h]))'
                  api_error_rate = float(api_error.json().get('data', {}).get('result', [0])[0])
                  
                  api_availability = requests.get(f"{prometheus_url}/api/v1/query", 
                                               json={"query": api_availability_query}, timeout=10)
                  api_availability = float(api_availability.json().get('data', {}).get('result', [0])[0])
                  
                  metrics['api'] = {
                      'response_time_p95': api_p95,
                      'response_time_p50': api_response_p50,
                      'error_rate': api_error_rate,
                      'availability': api_availability
                  }
                  
              # Database metrics
              try:
                  db_connection_query = 'histogram_quantile(0.95, sum(rate(db_connection_duration_seconds_bucket[1h)))'
                  db_connection = requests.get(f"{prometheus_url}/api/v1/query", 
                                        json={"query": db_connection_query}, timeout=10)
                  db_p95 = float(db_connection.json().get('data', {}).get('result', [0])[0])
                  
                  db_error_rate = requests.get(f"{prometheus_url}/api/v1/query", 
                                         json={"query": db_connection_error_rate}, timeout=10)
                  db_error_rate = float(db_error.json().get('data', {}).get('result', [0])[0])
                  
                  metrics['database'] = {
                      'connection_time_p95': db_p95,
                      'error_rate': db_error_rate,
                      'availability': db_connection'
                  }
                  
              # System metrics
              try:
                  memory_query = '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100'
                  memory = requests.get(f"{prometheus_url}/api/v1/query", 
                                     json={"query": memory_query}, timeout=10)
                  memory_usage = float(memory.json().get('data', {}).get('result', [0])[0])
                  
                  cpu_query = '100 - (avg by (instance) (100 - (avg by (instance) (100 - (avg by (instance) irate(node_cpu_seconds_total{mode="idle"}[1h])))'
                  cpu = requests.get(f"{prometheus_url}/api/v1/query", 
                                     json={"query": cpu_query}, timeout=10
                  cpu_usage = float(cpu.json().get('data', {}).get('result', [0])[0])
                  
                  disk_query = '(1 - (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"})) * 100'
                  disk_usage = requests.get(f"{prometheus_url}/api/v1/query", 
                                     json={"query": disk_query}, timeout=10
                  disk_usage = float(disk.json().get('data', {}).get('result', [0])[0])
                  
                  metrics['system'] = {
                      'memory_usage': memory_usage,
                      'cpu_usage': cpu_usage,
                      'disk_usage'
                  }
                  
              except Exception as e:
                  print_error(f"Error getting SLO metrics: {e}")
                  return None
          
          # Generate SLO report
          timestamp = datetime.now().isoformat()
          
          report = {
              'timestamp': timestamp,
              'slo_status': {
                  'api_response_time': {
                      'current': metrics['api']['response_time_p95'],
                      'target': 0.5,
                      'status': 'OK' if metrics['api']['response_time_p95'] <= 0.5 else 'VIOLATION'
                  },
                  'api_error_rate': {
                      'current': metrics['api']['error_rate'],
                      'target': 0.01,
                      'status': 'OK' if metrics['api']['error_rate'] <= 0.01 else 'VIOLATION'
                  },
                  'api_availability': {
                      'current': metrics['api']['availability'],
                      'target': 0.99,
                      'status': 'OK' if metrics['api']['availability'] >= 0.99 else 'VIOLATION'
                  }
              },
              'database_connection_time': {
                  'current': metrics['database']['connection_time_p95'],
                  'target': 0.1,
                  'status': 'OK' if metrics['database']['connection_time_p95'] <= 0.1 else 'WARNING'
                  },
                  'error_rate': metrics['database']['error_rate'],
                  'target': 0.01,
                  'status': 'OK' if metrics['database']['error_rate'] <= 0.01 else 'VIOLATION'
                  }
              },
              'code_analysis_performance': {
                  'current': metrics.get('code_analysis', 'duration_p95', 'target': 30, 'status': 'OK' if metrics['code_analysis']['duration_p95'] <= 30 else 'VIOLATION'
                  },
                  'failure_rate': metrics.get('code_analysis', 'failure_rate', 'target': 0.05, 'status': 'OK' if metrics['code_analysis']['failure_rate'] <= 0.05 else 'VIOLATION'
                  }
              }
              }
          },
              'system': {
                  'memory_usage': metrics['system']['memory_usage'],
                  'cpu_usage': metrics['system']['cpu_usage'],
                  'disk_usage': metrics['system']['disk_usage']
              }
          },
              'container_health': {
                  'up_count': sum(up{job=~"ai-review-.*"}), 
                  'total_pods': len(up{job=~"ai-review-.*"})
              }
          }
          
          # Calculate overall compliance
          total_checks = len(report['slo_status'])
          compliant_checks = sum(1 for status in report['slo_status'].values() if status['status'] == 'OK')
          compliance_percentage = (compliant_checks / total_checks) * 100)
          
          print(f"SLO Compliance: {compliance_percentage:.1f}% ({compliant_checks}/{total_checks})")
          
          if violations:
              print_warning "SLO Violations: ${', '.join(violations)}")
          else
              print_success "All SLOs are compliant"
          
          # Save report
          with open('slo-report.json', 'w') as f:
              json.dump(report, f, indent=2)
          
          print_success "SLO report generated: slo-report.json"
          print_success "SLO report saved to slo-report.json"
          
          # Check for emergency stop
          if [ -f "/tmp/chaos-emergency-stop" ]; then
              print_warning "Emergency stop triggered"
          else
              print_status "No emergency stop needed"
              fi
          
          return 0
          EOF
          
          return 0
          EOF
        else
          print_success "SLO monitoring completed"
          return 0
          EOF
        fi

      - name: Emergency Stop
        if: github.event_name == 'pull_request' && contains('critical', 'urgent', 'severe') || github.event.action == 'closed' && github.event.label == 'emergency'
        run: |
          python3 << 'EOF'
          print_status "Emergency stop triggered for critical issue"
          
          # Stop all chaos experiments
          local cmd="litmuschaos stop"
          
          # Wait for stop to complete
          sleep 30
          
          # Check if stopped
          stopped=false
          for i in {1..10}; do
              if ! pgrep -q "litmuschaos.*" > /dev/null; then
                  stopped=true
                  break
              fi
              done
          
          if [ "$stopped" = "true" ]; then
              print_success "Emergency stop completed"
              print_success "All chaos experiments stopped"
          else
              print_warning "Emergency stop failed"
              print_error "Emergency stop failed"
          fi
          
          return 0
          EOF
        else
          print_error "Emergency stop failed"
          return 0
          fi
          EOF
        fi

      - name: Generate Chaos Report
        if: always()
        run: |
          python3 << 'EOF'
          # Generate comprehensive chaos report
          python3 << 'EOF'
          import json
          import requests
          from datetime import timedelta
          
          # Get latest experiment results
          report_url = "http://localhost:3001/api/v1/chaos/reports"
          
          try:
              response = requests.get(report_url, timeout=10)
              response.raise_for_status()
              report = response.json()
              
              # Generate comprehensive report
              timestamp = datetime.now().isoformat()
              
              report = {
                  'timestamp': timestamp,
                  'experiments': [],
                  'summary': {
                      'total_experiments': 0,
                      'successful': 0,
                      'failed': 0,
                      'cancelled': 0,
                      'emergency_stopped': 0,
                      'budget_utilization': 0.0,
                      'slo_violations': 0
                  },
                  'recommendations': []
              }
              
              # Save report
              with open('chaos-report.json', 'w') as f:
                  json.dump(report, f, indent=2)
              
              print_success "Chaos report generated: chaos-report.json"
              print_success "Chaos report generated: chaos-report.json"
              print_success "Chaos report saved to chaos-report.json"
              
              return 0
          except Exception as e:
              print_error "Error generating chaos report: ${e}"
              return 1
          EOF
          EOF
        else
          print_success "Chaos monitoring completed"
          return 0
          EOF
        fi

# Function to show help
show_help() {
    echo "Chaos Engineering Script v2"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo ""
    echo "Commands:"
    echo "  install     Install chaos engineering tools"
    echo "  validate   Validate experiment configuration"
    echo "  run       Run chaos experiment"
    echo "  report     Generate experiment report"
    echo "  status     Check experiment status"
    echo "  stop      Stop running experiment"
    echo "  help       Show this help message"
    echo ""
    echo ""
    echo "Options:"
    echo "  --dry-run     Run experiment in dry-run mode"
    echo "  --duration   Set experiment duration (default: 60)"
    echo "  --budget    Set maximum budget (default: 100.00)"
    echo ""
    echo "  --emergency-stop Emergency stop"
    echo ""
    echo ""
    echo "Examples:"
    echo "  $0 install                    # Install all chaos tools"
    echo "  $0 validate experiments/network_latency.yml    # Validate network latency experiment"
    echo "  $0 report experiments/network_latency --last 24h    # Generate report for last 24h"
    echo "  $0 status experiments/network_latency           # Check current status"
    echo ""
    echo ""
}

# Main script logic
COMMAND=${1:-help}
case $COMMAND in
    install)
        install_tools
        ;;
    validate)
        validate
        ;;
    run)
        run)
        ;;
    report)
        status)
        ;;
    status)
        ;;
    *)
        show_help)
        ;;
esac