#!/bin/bash

# 翻译系统自动化部署脚本
# 支持多环境部署和回滚

set -e

# 配置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
ENVIRONMENT="${ENVIRONMENT:-development}"
NAMESPACE="${NAMESPACE:-translation}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
翻译系统部署脚本

用法: $0 [选项] <命令>

命令:
    build           构建Docker镜像
    deploy          部署到指定环境
    update          更新现有部署
    rollback        回滚到上一个版本
    scale           扩缩容服务
    status          查看部署状态
    logs            查看服务日志
    cleanup         清理资源

选项:
    -e, --env ENV           部署环境 (development|staging|production|edge)
    -t, --tag TAG           镜像标签 (默认: latest)
    -n, --namespace NS      Kubernetes命名空间 (默认: translation)
    -r, --registry REG      Docker镜像仓库 (默认: localhost:5000)
    -s, --service SERVICE   指定服务名称
    -c, --replicas COUNT    副本数量
    -h, --help              显示帮助信息

示例:
    $0 -e production deploy
    $0 -e staging -t v1.2.3 update
    $0 -s translation-service -c 5 scale
    $0 rollback
EOF
}

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."
    
    local deps=("docker" "docker-compose")
    if [[ "$ENVIRONMENT" == "production" || "$ENVIRONMENT" == "staging" ]]; then
        deps+=("kubectl")
    fi
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "缺少依赖: $dep"
            exit 1
        fi
    done
    
    log_success "依赖检查完成"
}

# 构建Docker镜像
build_images() {
    log_info "构建Docker镜像..."
    
    cd "$PROJECT_ROOT"
    
    # 构建主应用镜像
    docker build -t "${DOCKER_REGISTRY}/translation-service:${IMAGE_TAG}" \
        --target production \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
        .
    
    # 构建边缘版本镜像
    if [[ "$ENVIRONMENT" == "edge" ]]; then
        docker build -t "${DOCKER_REGISTRY}/translation-service:${IMAGE_TAG}-edge" \
            --target edge \
            .
    fi
    
    log_success "镜像构建完成"
}

# 推送镜像到仓库
push_images() {
    log_info "推送镜像到仓库..."
    
    docker push "${DOCKER_REGISTRY}/translation-service:${IMAGE_TAG}"
    
    if [[ "$ENVIRONMENT" == "edge" ]]; then
        docker push "${DOCKER_REGISTRY}/translation-service:${IMAGE_TAG}-edge"
    fi
    
    log_success "镜像推送完成"
}

# Docker Compose部署
deploy_docker_compose() {
    log_info "使用Docker Compose部署..."
    
    cd "$PROJECT_ROOT"
    
    local compose_file="docker-compose.yml"
    if [[ "$ENVIRONMENT" == "edge" ]]; then
        compose_file="docker-compose.edge.yml"
    fi
    
    # 设置环境变量
    export IMAGE_TAG
    export ENVIRONMENT
    
    # 部署服务
    docker-compose -f "$compose_file" down --remove-orphans
    docker-compose -f "$compose_file" up -d
    
    log_success "Docker Compose部署完成"
}

# Kubernetes部署
deploy_kubernetes() {
    log_info "使用Kubernetes部署..."
    
    # 创建命名空间
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # 应用配置
    kubectl apply -f "$PROJECT_ROOT/k8s/" -n "$NAMESPACE"
    
    # 等待部署完成
    kubectl rollout status deployment/translation-service -n "$NAMESPACE" --timeout=300s
    
    log_success "Kubernetes部署完成"
}

# 部署应用
deploy() {
    log_info "开始部署到 $ENVIRONMENT 环境..."
    
    build_images
    
    if [[ "$ENVIRONMENT" == "development" || "$ENVIRONMENT" == "edge" ]]; then
        deploy_docker_compose
    else
        push_images
        deploy_kubernetes
    fi
    
    # 等待服务就绪
    wait_for_services
    
    log_success "部署完成!"
}

# 更新部署
update() {
    log_info "更新部署..."
    
    if [[ "$ENVIRONMENT" == "development" || "$ENVIRONMENT" == "edge" ]]; then
        build_images
        deploy_docker_compose
    else
        build_images
        push_images
        kubectl set image deployment/translation-service \
            translation-service="${DOCKER_REGISTRY}/translation-service:${IMAGE_TAG}" \
            -n "$NAMESPACE"
        kubectl rollout status deployment/translation-service -n "$NAMESPACE" --timeout=300s
    fi
    
    log_success "更新完成!"
}

# 回滚部署
rollback() {
    log_info "回滚部署..."
    
    if [[ "$ENVIRONMENT" == "development" || "$ENVIRONMENT" == "edge" ]]; then
        log_warning "Docker Compose环境不支持自动回滚"
        return 1
    fi
    
    kubectl rollout undo deployment/translation-service -n "$NAMESPACE"
    kubectl rollout status deployment/translation-service -n "$NAMESPACE" --timeout=300s
    
    log_success "回滚完成!"
}

# 扩缩容服务
scale_service() {
    local service_name="${SERVICE_NAME:-translation-service}"
    local replicas="${REPLICAS:-3}"
    
    log_info "扩缩容服务 $service_name 到 $replicas 个副本..."
    
    if [[ "$ENVIRONMENT" == "development" || "$ENVIRONMENT" == "edge" ]]; then
        docker-compose up -d --scale "$service_name=$replicas"
    else
        kubectl scale deployment/"$service_name" --replicas="$replicas" -n "$NAMESPACE"
    fi
    
    log_success "扩缩容完成!"
}

# 查看部署状态
show_status() {
    log_info "查看部署状态..."
    
    if [[ "$ENVIRONMENT" == "development" || "$ENVIRONMENT" == "edge" ]]; then
        docker-compose ps
    else
        kubectl get pods,services,deployments -n "$NAMESPACE"
        kubectl top pods -n "$NAMESPACE" 2>/dev/null || true
    fi
}

# 查看服务日志
show_logs() {
    local service_name="${SERVICE_NAME:-translation-service}"
    
    log_info "查看服务日志: $service_name"
    
    if [[ "$ENVIRONMENT" == "development" || "$ENVIRONMENT" == "edge" ]]; then
        docker-compose logs -f "$service_name"
    else
        kubectl logs -f deployment/"$service_name" -n "$NAMESPACE"
    fi
}

# 等待服务就绪
wait_for_services() {
    log_info "等待服务就绪..."
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if check_service_health; then
            log_success "服务已就绪"
            return 0
        fi
        
        log_info "等待服务启动... ($attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    log_error "服务启动超时"
    return 1
}

# 检查服务健康状态
check_service_health() {
    local health_url="http://localhost:8000/health"
    
    if [[ "$ENVIRONMENT" != "development" && "$ENVIRONMENT" != "edge" ]]; then
        # 在Kubernetes环境中，通过端口转发检查
        kubectl port-forward service/translation-service 8000:80 -n "$NAMESPACE" &
        local port_forward_pid=$!
        sleep 2
    fi
    
    if curl -f -s "$health_url" > /dev/null 2>&1; then
        [[ -n "${port_forward_pid:-}" ]] && kill "$port_forward_pid" 2>/dev/null || true
        return 0
    else
        [[ -n "${port_forward_pid:-}" ]] && kill "$port_forward_pid" 2>/dev/null || true
        return 1
    fi
}

# 清理资源
cleanup() {
    log_info "清理资源..."
    
    if [[ "$ENVIRONMENT" == "development" || "$ENVIRONMENT" == "edge" ]]; then
        docker-compose down --volumes --remove-orphans
        docker system prune -f
    else
        kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
    fi
    
    log_success "清理完成"
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -t|--tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -r|--registry)
                DOCKER_REGISTRY="$2"
                shift 2
                ;;
            -s|--service)
                SERVICE_NAME="$2"
                shift 2
                ;;
            -c|--replicas)
                REPLICAS="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            build|deploy|update|rollback|scale|status|logs|cleanup)
                COMMAND="$1"
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 主函数
main() {
    parse_args "$@"
    
    if [[ -z "${COMMAND:-}" ]]; then
        log_error "请指定命令"
        show_help
        exit 1
    fi
    
    log_info "部署配置:"
    log_info "  环境: $ENVIRONMENT"
    log_info "  镜像标签: $IMAGE_TAG"
    log_info "  命名空间: $NAMESPACE"
    log_info "  镜像仓库: $DOCKER_REGISTRY"
    
    check_dependencies
    
    case "$COMMAND" in
        build)
            build_images
            ;;
        deploy)
            deploy
            ;;
        update)
            update
            ;;
        rollback)
            rollback
            ;;
        scale)
            scale_service
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        cleanup)
            cleanup
            ;;
        *)
            log_error "未知命令: $COMMAND"
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"