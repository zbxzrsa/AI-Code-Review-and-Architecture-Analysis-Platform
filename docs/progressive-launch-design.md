# 渐进式启动流程设计文档

## 概述

本文档详细描述了 CodeInsight 平台的渐进式启动流程设计，包含四个主要阶段和完整的错误处理系统。该设计旨在提供友好的用户体验、清晰的状态反馈和强大的故障恢复能力。

## 四阶段启动流程

### 阶段一：初始检查

#### 用户界面设计
- **启动界面**: 显示平台Logo和"正在启动 CodeInsight..."的欢迎信息
- **进度指示器**: 圆形进度条显示初始检查进度
- **实时状态**: 显示当前检查项目的文字说明
- **一键修复**: 当检测到问题时显示"一键修复"按钮

#### 检查项目
1. **操作系统兼容性检查**
   - 检测 Windows/Linux/macOS 版本
   - 验证系统架构 (x64/arm64)
   - 检查必要的系统库

2. **Docker 环境预检**
   - Docker 安装状态检查
   - Docker 服务运行状态
   - Docker 版本兼容性

3. **系统资源评估**
   - 内存可用量检查（最低 4GB 推荐 8GB）
   - 磁盘空间检查（最低 10GB 可用空间）
   - CPU 核心数检查

4. **端口可用性检查**
   - 3000 (前端服务)
   - 8000 (后端服务)
   - 5432 (数据库服务)
   - 6379 (Redis 服务)

#### 状态反馈机制
```typescript
interface InitialCheckStatus {
  phase: 'initial_check';
  currentStep: string;
  progress: number;
  items: {
    osCheck: { status: 'pending' | 'success' | 'error'; message: string };
    dockerCheck: { status: 'pending' | 'success' | 'error'; message: string };
    resourceCheck: { status: 'pending' | 'success' | 'error'; message: string };
    portCheck: { status: 'pending' | 'success' | 'error'; message: string };
  };
  autoFixAvailable: boolean;
}
```

#### 错误处理策略
- **自动修复**: Docker 安装缺失、端口占用等可通过脚本自动解决的问题
- **用户引导**: 需要用户干预的问题（如管理员权限）
- **跳过选项**: 对于非关键问题提供跳过继续启动的选项
- **详细说明**: 每个错误提供具体的问题描述和解决方案

### 阶段二：依赖验证

#### 可视化设计
- **Docker 状态面板**: 
  - Docker 服务状态指示灯（绿色/红色）
  - 版本信息显示
  - 资源使用情况图表

- **服务依赖关系图**:
  - 可视化显示服务间的依赖关系
  - 实时更新依赖状态
  - 鼠标悬停显示详细信息

- **进度时间轴**:
  - 水平时间轴显示启动顺序
  - 当前执行步骤高亮显示
  - 预估完成时间倒计时

#### 实时监控机制
```typescript
interface DependencyValidationStatus {
  phase: 'dependency_validation';
  dockerStatus: {
    installed: boolean;
    running: boolean;
    version: string;
    resources: {
      memory: number;
      cpu: number;
    };
  };
  dependencies: {
    [serviceName: string]: {
      status: 'pending' | 'pulling' | 'building' | 'ready' | 'error';
      progress: number;
      estimatedTime: number;
      lastUpdate: Date;
    };
  };
  timeline: {
    currentStep: string;
    nextSteps: string[];
    estimatedCompletion: Date;
  };
}
```

#### 详细状态说明
- **pulling**: 显示镜像下载进度和剩余时间
- **building**: 显示构建步骤和编译进度
- **ready**: 服务就绪，显示绿色确认标记
- **error**: 错误详情和修复建议

### 阶段三：服务部署

#### 分步骤容器启动
1. **基础服务启动**
   - PostgreSQL 数据库
   - Redis 缓存服务
   - Neo4j 图数据库

2. **应用服务启动**
   - 后端 API 服务
   - 代码解析服务
   - AI 分析服务

3. **前端服务启动**
   - React 前端应用
   - 静态资源加载
   - 开发服务器启动

#### 预估等待时间
- 基于历史数据计算各步骤所需时间
- 实时调整预估（根据当前系统负载）
- 显示剩余时间范围（最短/最长时间）

#### 实时日志输出
- **结构化日志**: 按服务分类显示日志
- **过滤功能**: 可按级别（info/warn/error）过滤
- **搜索功能**: 实时搜索日志内容
- **自动滚动**: 自动滚动到最新日志

```typescript
interface ServiceDeploymentStatus {
  phase: 'service_deployment';
  currentStep: string;
  services: {
    [serviceName: string]: {
      status: 'pending' | 'starting' | 'running' | 'error';
      startTime: Date;
      logs: LogEntry[];
      healthChecks: HealthCheckResult[];
    };
  };
  timeline: {
    completed: string[];
    current: string;
    remaining: string[];
    estimatedCompletion: Date;
  };
}
```

### 阶段四：就绪确认

#### 服务健康检查结果
- **状态面板**: 显示所有服务的健康状态
- **响应时间**: 各服务的响应时间统计
- **错误统计**: 错误数量和类型分析
- **资源使用**: CPU、内存、磁盘使用情况

#### 访问链接自动生成
- **前端地址**: http://localhost:3000
- **API 文档**: http://localhost:8000/docs
- **监控面板**: http://localhost:9090 (Prometheus)


#### 浏览器自动打开引导
- **延迟打开**: 等待服务完全就绪后打开
- **多标签页**: 同时打开所有相关页面
- **错误处理**: 如果打开失败提供手动链接

```typescript
interface ReadinessConfirmationStatus {
  phase: 'readiness_confirmation';
  healthResults: {
    [serviceName: string]: {
      status: 'healthy' | 'warning' | 'unhealthy';
      checks: HealthCheck[];
      responseTime: number;
    };
  };
  accessUrls: {
    frontend: string;
    api: string;
    docs: string;
    monitoring: string;
  };
  systemStatus: {
    overall: 'ready' | 'partial' | 'error';
    resources: SystemResource;
  };
}
```

## 完整错误处理和恢复系统

### 错误分类

#### 1. 环境配置错误

**错误检测方法**:
```typescript
class EnvironmentChecker {
  async checkDockerInstallation(): Promise<EnvironmentCheckResult> {
    try {
      const { stdout } = await exec('docker --version');
      const versionMatch = stdout.match(/Docker version (\d+\.\d+\.\d+)/);
      return {
        success: true,
        version: versionMatch?.[1] || 'unknown',
        details: 'Docker 安装正常'
      };
    } catch (error) {
      return {
        success: false,
        error: 'DOCKER_NOT_INSTALLED',
        message: 'Docker 未安装或不在 PATH 中',
        severity: 'critical',
        autoFix: true,
        fixCommand: 'Install-Docker'
      };
    }
  }

  async checkSystemResources(): Promise<EnvironmentCheckResult> {
    const resources = await systemMonitor.getResources();
    const memoryGB = resources.memory / 1024;
    
    if (memoryGB < 2) {
      return {
        success: false,
        error: 'INSUFFICIENT_MEMORY',
        message: `可用内存不足 (当前: ${memoryGB.toFixed(1)}GB, 最低要求: 4GB)`,
        severity: 'critical',
        autoFix: false,
        suggestions: [
          '关闭其他应用程序释放内存',
          '增加系统内存',
          '调整 Docker 内存限制'
        ]
      };
    }
    
    return {
      success: true,
      details: `系统资源充足 (内存: ${memoryGB.toFixed(1)}GB)`
    };
  }
}
```

**自动修复尝试**:
- **Docker 安装缺失**: 下载并安装 Docker Desktop
- **端口占用**: 自动停止占用端口的进程
- **磁盘空间不足**: 清理临时文件和旧容器
- **内存不足**: 调整 Docker 内存限制

**用户干预指导**:
```typescript
interface UserGuidance {
  steps: string[];
  commands: string[];
  precautions: string[];
  estimatedTime: string;
}
```

**预防措施建议**:
- 定期检查系统资源使用情况
- 监控端口使用状态
- 保持 Docker 版本更新
- 定期清理无用容器和镜像

#### 2. 权限和安全错误

**错误检测方法**:
```typescript
class SecurityChecker {
  async checkPermissions(): Promise<SecurityCheckResult> {
    const checks = {
      adminPrivileges: await this.checkAdminPrivileges(),
      fileAccess: await this.checkFileAccess(),
      firewall: await this.checkFirewall(),
      dockerAccess: await this.checkDockerAccess()
    };
    
    const issues = Object.entries(checks)
      .filter(([_, result]) => !result.success)
      .map(([key, result]) => ({
        type: key,
        ...result
      }));
    
    return {
      overall: issues.length === 0 ? 'passed' : 'issues_found',
      issues,
      recommendations: this.generateRecommendations(issues)
    };
  }

  private async checkAdminPrivileges(): Promise<CheckResult> {
    if (process.platform === 'win32') {
      try {
        const { stdout } = await exec('whoami /groups');
        const isAdmin = stdout.includes('S-1-16-12288');
        return {
          success: isAdmin,
          message: isAdmin ? '管理员权限正常' : '需要管理员权限'
        };
      } catch {
        return { success: false, message: '无法检查权限状态' };
      }
    }
    return { success: true, message: 'Unix 系统权限检查通过' };
  }
}
```

**自动修复尝试**:
- **PowerShell 执行策略**: 自动设置合适的执行策略
- **文件权限**: 修复必要的文件和目录权限
- **Docker 权限**: 配置 Docker 用户组

**用户干预指导**:
- 提供明确的命令行操作指导
- 生成权限修复脚本
- 提供图形界面的权限设置指南

#### 3. 网络和下载错误

**错误检测方法**:
```typescript
class NetworkChecker {
  async checkConnectivity(): Promise<NetworkCheckResult> {
    const checks = {
      internet: await this.checkInternetConnectivity(),
      dockerHub: await this.checkDockerHubAccess(),
      ports: await this.checkPortAvailability(),
      proxy: await this.checkProxySettings()
    };
    
    return {
      overall: this.evaluateOverallStatus(checks),
      details: checks,
      recommendations: this.generateNetworkRecommendations(checks)
    };
  }

  private async checkInternetConnectivity(): Promise<CheckResult> {
    try {
      const response = await axios.get('https://registry-1.docker.io', {
        timeout: 5000
      });
      return {
        success: response.status === 200,
        message: '网络连接正常'
      };
    } catch (error) {
      return {
        success: false,
        message: `网络连接失败: ${error.message}`,
        autoFix: true,
        fixCommand: 'Test-Network'
      };
    }
  }
}
```

**自动修复尝试**:
- **网络重连**: 尝试重置网络连接
- **代理配置**: 自动检测和配置代理设置
- **DNS 刷新**: 刷新 DNS 缓存
- **镜像加速**: 配置 Docker 镜像加速器

**用户干预指导**:
- 网络故障排查步骤
- 代理配置指导
- 防火墙规则设置
- DNS 服务器配置建议

#### 4. 服务运行时错误

**错误检测方法**:
```typescript
class RuntimeChecker {
  async checkServiceHealth(): Promise<ServiceHealthResult> {
    const services = ['backend', 'frontend', 'database'];
    const healthResults = {};
    
    for (const service of services) {
      try {
        const health = await this.checkSingleService(service);
        healthResults[service] = health;
      } catch (error) {
        healthResults[service] = {
          status: 'error',
          error: error.message,
          timestamp: new Date()
        };
      }
    }
    
    return {
      overall: this.evaluateOverallHealth(healthResults),
      services: healthResults,
      recommendations: this.generateHealthRecommendations(healthResults)
    };
  }

  private async checkSingleService(serviceName: string): Promise<ServiceHealth> {
    const config = this.getServiceConfig(serviceName);
    const url = `http://localhost:${config.port}/health`;
    
    try {
      const response = await axios.get(url, { timeout: 5000 });
      return {
        status: 'healthy',
        responseTime: response.headers['x-response-time'] || 'unknown',
        checks: response.data.healthChecks || []
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        error: error.message,
        lastCheck: new Date()
      };
    }
  }
}
```

**自动修复尝试**:
- **容器重启**: 自动重启失败的容器
- **服务重启**: 重启不健康的服务
- **资源调整**: 根据负载调整资源分配
- **配置重载**: 重新加载配置文件

**用户干预指导**:
- 服务故障排查清单
- 日志分析方法
- 配置文件检查指南
- 性能优化建议

### 恢复策略

#### 自动重试机制（带指数退避）
```typescript
class RetryManager {
  async executeWithRetry<T>(
    operation: () => Promise<T>,
    maxRetries: number = 3,
    baseDelay: number = 1000
  ): Promise<T> {
    let lastError: Error;
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error as Error;
        
        if (attempt === maxRetries) {
          throw lastError;
        }
        
        const delay = baseDelay * Math.pow(2, attempt - 1);
        await this.sleep(delay);
      }
    }
    
    throw lastError;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
```

#### 替代方案降级处理
```typescript
class DegradationManager {
  async startWithDegradation(): Promise<StartupResult> {
    const essentialServices = ['database', 'backend'];
    const optionalServices = ['ai_service', 'monitoring'];
    
    // 先启动核心服务
    const essentialResults = await this.startServices(essentialServices);
    
    if (essentialResults.success) {
      // 尝试启动可选服务
      const optionalResults = await this.startServicesWithFallback(optionalServices);
      
      return {
        success: true,
        essential: essentialResults,
        optional: optionalResults,
        mode: optionalResults.allSuccess ? 'full' : 'degraded'
      };
    }
    
    return {
      success: false,
      error: 'Essential services failed to start',
      essential: essentialResults
    };
  }
}
```

#### 用户引导式修复
```typescript
class UserGuidanceManager {
  generateGuidance(error: Error): Guidance {
    const errorType = this.classifyError(error);
    
    return {
      title: this.getErrorTitle(errorType),
      description: this.getErrorDescription(errorType),
      steps: this.getResolutionSteps(errorType),
      estimatedTime: this.getEstimatedTime(errorType),
      difficulty: this.getDifficultyLevel(errorType),
      requiredActions: this.getRequiredActions(errorType)
    };
  }

  private classifyError(error: Error): ErrorType {
    if (error.message.includes('Docker')) {
      return 'docker_error';
    } else if (error.message.includes('port')) {
      return 'port_conflict';
    } else if (error.message.includes('memory')) {
      return 'resource_exhausted';
    } else if (error.message.includes('permission')) {
      return 'permission_denied';
    }
    return 'unknown_error';
  }
}
```

#### 安全回滚到已知状态
```typescript
class RollbackManager {
  async safeRollback(): Promise<RollbackResult> {
    const currentState = await this.captureCurrentState();
    const rollbackPoints = await this.findRollbackPoints();
    
    try {
      // 停止所有服务
      await this.stopAllServices();
      
      // 回滚到最近的稳定状态
      const targetState = rollbackPoints.find(p => p.status === 'stable');
      if (targetState) {
        await this.restoreState(targetState);
        return {
          success: true,
          restoredTo: targetState,
          currentState: currentState
        };
      }
      
      throw new Error('No stable rollback point found');
    } catch (error) {
      await this.restoreState(currentState);
      return {
        success: false,
        error: error as Error,
        restoredTo: currentState
      };
    }
  }
}
```

## 集成建议

### 前端界面设计
- 使用 React + TypeScript 构建响应式界面
- 集成 Chart.js 展示系统资源使用情况
- 使用 WebSocket 实现实时状态更新
- 集成 Monaco Editor 提供日志查看功能

### 后端架构
```typescript
class ProgressiveStartupManager {
  private checker: EnvironmentChecker;
  private networkChecker: NetworkChecker;
  private securityChecker: SecurityChecker;
  private runtimeChecker: RuntimeChecker;
  private retryManager: RetryManager;
  private rollbackManager: RollbackManager;

  async startup(): Promise<StartupResult> {
    const startTime = Date.now();
    
    try {
      // 阶段一：初始检查
      const initialCheck = await this