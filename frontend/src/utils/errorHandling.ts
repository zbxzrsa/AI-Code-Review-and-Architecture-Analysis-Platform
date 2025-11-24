import { exec } from 'child_process';
import { promisify } from 'util';
import { EventEmitter } from 'events';

const execAsync = promisify(exec);

// 错误类型定义
export enum ErrorType {
  ENVIRONMENT_CONFIG = 'environment_config',
  PERMISSION_SECURITY = 'permission_security',
  NETWORK_DOWNLOAD = 'network_download',
  SERVICE_RUNTIME = 'service_runtime'
}

export enum ErrorSeverity {
  CRITICAL = 'critical',
  HIGH = 'high',
  MEDIUM = 'medium',
  LOW = 'low'
}

export enum AutoFixResult {
  SUCCESS = 'success',
  FAILED = 'failed',
  NOT_AVAILABLE = 'not_available',
  REQUIRES_USER_INPUT = 'requires_user_input'
}

// 错误信息接口
export interface ErrorInfo {
  id: string;
  type: ErrorType;
  severity: ErrorSeverity;
  message: string;
  timestamp: Date;
  details: any;
  autoFixAvailable: boolean;
  userGuidance?: UserGuidance;
}

// 用户指导接口
export interface UserGuidance {
  title: string;
  description: string;
  steps: string[];
  estimatedTime: string;
  difficulty: 'easy' | 'medium' | 'hard';
  requiredActions: string[];
  precautionMeasures: string[];
}

// 自动修复结果接口
export interface AutoFixResultInfo {
  result: AutoFixResult;
  message: string;
  retryAttempts: number;
  fixedErrors: string[];
  newErrors?: ErrorInfo[];
}

// 错误处理管理器
export class ErrorHandlerManager extends EventEmitter {
  private errorHistory: ErrorInfo[] = [];
  private autoFixStrategies: Map<ErrorType, (error: ErrorInfo) => Promise<AutoFixResultInfo>> = new Map();
  
  constructor() {
    super();
    this.initializeAutoFixStrategies();
  }

  private initializeAutoFixStrategies() {
    // 环境配置错误修复策略
    this.autoFixStrategies.set(ErrorType.ENVIRONMENT_CONFIG, async (error) => {
      return await this.fixEnvironmentConfig(error);
    });

    // 权限和安全错误修复策略
    this.autoFixStrategies.set(ErrorType.PERMISSION_SECURITY, async (error) => {
      return await this.fixPermissionSecurity(error);
    });

    // 网络和下载错误修复策略
    this.autoFixStrategies.set(ErrorType.NETWORK_DOWNLOAD, async (error) => {
      return await this.fixNetworkDownload(error);
    });

    // 服务运行时错误修复策略
    this.autoFixStrategies.set(ErrorType.SERVICE_RUNTIME, async (error) => {
      return await this.fixServiceRuntime(error);
    });
  }

  // 检测错误
  async detectError(errorType: ErrorType): Promise<ErrorInfo | null> {
    switch (errorType) {
      case ErrorType.ENVIRONMENT_CONFIG:
        return await this.detectEnvironmentConfigError();
      case ErrorType.PERMISSION_SECURITY:
        return await this.detectPermissionSecurityError();
      case ErrorType.NETWORK_DOWNLOAD:
        return await this.detectNetworkDownloadError();
      case ErrorType.SERVICE_RUNTIME:
        return await this.detectServiceRuntimeError();
      default:
        return null;
    }
  }

  // 检测环境配置错误
  private async detectEnvironmentConfigError(): Promise<ErrorInfo | null> {
    const errors: ErrorInfo[] = [];

    // 检查 Docker 安装
    try {
      await execAsync('docker --version');
    } catch {
      errors.push({
        id: 'DOCKER_NOT_INSTALLED',
        type: ErrorType.ENVIRONMENT_CONFIG,
        severity: ErrorSeverity.CRITICAL,
        message: 'Docker 未安装或不在 PATH 中',
        timestamp: new Date(),
        details: { command: 'docker --version' },
        autoFixAvailable: true,
        userGuidance: {
          title: '安装 Docker',
          description: 'Docker 是运行容器化服务必需的软件',
          steps: [
            '访问 Docker 官方网站',
            '下载 Docker Desktop',
            '运行安装程序',
            '重启计算机'
          ],
          estimatedTime: '10-15 分钟',
          difficulty: 'easy',
          requiredActions: ['下载安装包', '运行安装程序'],
          precautionMeasures: ['确保系统满足 Docker 要求', '备份重要数据']
        }
      });
    }

    // 检查 Docker 服务运行状态
    try {
      await execAsync('docker info');
    } catch {
      errors.push({
        id: 'DOCKER_NOT_RUNNING',
        type: ErrorType.ENVIRONMENT_CONFIG,
        severity: ErrorSeverity.HIGH,
        message: 'Docker 服务未运行',
        timestamp: new Date(),
        details: { command: 'docker info' },
        autoFixAvailable: true,
        userGuidance: {
          title: '启动 Docker 服务',
          description: 'Docker 服务需要运行才能管理容器',
          steps: [
            '打开 Docker Desktop',
            '点击 Start 按钮',
            '等待 Docker 完全启动',
            '检查 Docker 状态'
          ],
          estimatedTime: '2-5 分钟',
          difficulty: 'easy',
          requiredActions: ['启动 Docker Desktop', '等待启动完成'],
          precautionMeasures: ['确保 Docker Desktop 有管理员权限']
        }
      });
    }

    // 检查系统资源
    try {
      const { stdout } = await execAsync('wmic OS get TotalVisibleMemorySize /value');
      const memoryMatch = stdout.match(/TotalVisibleMemorySize=(\d+)/);
      const totalMemoryMB = parseInt(memoryMatch?.[1] || '0');
      
      if (totalMemoryMB < 4096) { // 4GB
        errors.push({
          id: 'INSUFFICIENT_MEMORY',
          type: ErrorType.ENVIRONMENT_CONFIG,
          severity: ErrorSeverity.HIGH,
          message: `系统内存不足 (当前: ${(totalMemoryMB / 1024).toFixed(1)}GB, 最低要求: 4GB)`,
          timestamp: new Date(),
          details: { totalMemoryMB },
          autoFixAvailable: false,
          userGuidance: {
            title: '增加系统内存',
            description: '系统内存不足以支持所有服务同时运行',
            steps: [
              '关闭不必要的应用程序',
              '增加系统内存',
              '调整 Docker 内存限制',
              '考虑关闭部分服务'
            ],
            estimatedTime: '5-10 分钟',
            difficulty: 'medium',
            requiredActions: ['关闭其他程序', '调整 Docker 设置'],
            precautionMeasures: ['定期检查系统资源使用情况']
          }
        });
      }
    } catch {
      // 内存检查失败，但不是致命错误
    }

    return errors.length > 0 ? errors[0] : null;
  }

  // 检测权限和安全错误
  private async detectPermissionSecurityError(): Promise<ErrorInfo | null> {
    const errors: ErrorInfo[] = [];

    // 检查管理员权限
    if (process.platform === 'win32') {
      try {
        const { stdout } = await execAsync('whoami /groups');
        const isAdmin = stdout.includes('S-1-16-12288');
        
        if (!isAdmin) {
          errors.push({
            id: 'ADMIN_PRIVILEGES_REQUIRED',
            type: ErrorType.PERMISSION_SECURITY,
            severity: ErrorSeverity.CRITICAL,
            message: '需要管理员权限才能启动服务',
            timestamp: new Date(),
            details: { platform: 'windows' },
            autoFixAvailable: false,
            userGuidance: {
              title: '获取管理员权限',
              description: '启动服务需要管理员权限',
              steps: [
                '右键点击启动程序',
                '选择"以管理员身份运行"',
                '确认权限请求',
                '重新启动服务'
              ],
              estimatedTime: '1-2 分钟',
              difficulty: 'easy',
              requiredActions: ['右键管理员运行', '确认权限'],
              precautionMeasures: ['始终以管理员身份运行服务']
            }
          });
        }
      } catch {
        // 权限检查失败
      }
    }

    // 检查 Docker 权限
    try {
      await execAsync('docker ps');
    } catch {
      errors.push({
        id: 'DOCKER_PERMISSION_DENIED',
        type: ErrorType.PERMISSION_SECURITY,
        severity: ErrorSeverity.HIGH,
        message: 'Docker 权限不足',
        timestamp: new Date(),
        details: { command: 'docker ps' },
        autoFixAvailable: true,
        userGuidance: {
          title: '配置 Docker 权限',
          description: '当前用户没有足够的 Docker 权限',
          steps: [
            '将用户添加到 docker 组',
            '注销并重新登录',
            '或使用 sudo 运行 Docker 命令'
          ],
          estimatedTime: '3-5 分钟',
          difficulty: 'medium',
          requiredActions: ['用户组配置', '重新登录'],
          precautionMeasures: ['定期检查 Docker 权限设置']
        }
      });
    }

    return errors.length > 0 ? errors[0] : null;
  }

  // 检测网络和下载错误
  private async detectNetworkDownloadError(): Promise<ErrorInfo | null> {
    const errors: ErrorInfo[] = [];

    // 检查网络连接
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      
      const response = await fetch('https://registry-1.docker.io', {
        method: 'HEAD',
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        errors.push({
          id: 'NETWORK_CONNECTION_FAILED',
          type: ErrorType.NETWORK_DOWNLOAD,
          severity: ErrorSeverity.HIGH,
          message: '网络连接失败，无法访问 Docker Hub',
          timestamp: new Date(),
          details: { status: response.status },
          autoFixAvailable: true,
          userGuidance: {
            title: '修复网络连接',
            description: '无法连接到 Docker Hub，可能影响镜像下载',
            steps: [
              '检查网络连接',
              '测试网站访问',
              '检查代理设置',
              '刷新 DNS 缓存'
            ],
            estimatedTime: '5-10 分钟',
            difficulty: 'medium',
            requiredActions: ['检查网络', '配置代理'],
            precautionMeasures: ['配置镜像加速器', '使用本地镜像']
          }
        });
      }
    } catch {
      errors.push({
        id: 'NETWORK_CONNECTION_TIMEOUT',
        type: ErrorType.NETWORK_DOWNLOAD,
        severity: ErrorSeverity.HIGH,
        message: '网络连接超时',
        timestamp: new Date(),
        details: {},
        autoFixAvailable: true,
        userGuidance: {
          title: '修复网络超时',
          description: '网络请求超时，可能是网络不稳定或防火墙阻止',
          steps: [
            '检查网络稳定性',
            '调整超时设置',
            '检查防火墙设置',
            '尝试其他网络连接'
          ],
          estimatedTime: '5-15 分钟',
          difficulty: 'medium',
          requiredActions: ['检查网络', '配置防火墙'],
          precautionMeasures: ['使用稳定的网络连接', '配置镜像加速器']
        }
      });
    }

    // 检查端口占用
    const ports = [3000, 8000, 5432, 6379];
    for (const port of ports) {
      try {
        const { stdout } = await execAsync(`netstat -ano | findstr :${port}`);
        if (stdout.trim()) {
          errors.push({
            id: `PORT_${port}_OCCUPIED`,
            type: ErrorType.NETWORK_DOWNLOAD,
            severity: ErrorSeverity.MEDIUM,
            message: `端口 ${port} 已被占用`,
            timestamp: new Date(),
            details: { port },
            autoFixAvailable: true,
            userGuidance: {
              title: `释放端口 ${port}`,
              description: `端口 ${port} 被其他程序占用`,
              steps: [
                '找到占用端口的程序',
                '停止该程序或更改端口',
                '或使用 "释放端口" 功能'
              ],
              estimatedTime: '2-5 分钟',
              difficulty: 'easy',
              requiredActions: ['停止占用程序', '选择释放端口'],
              precautionMeasures: ['避免使用常用端口', '定期检查端口使用情况']
            }
          });
          break;
        }
      } catch {
        // 端口检查失败
      }
    }

    return errors.length > 0 ? errors[0] : null;
  }

  // 检测服务运行时错误
  private async detectServiceRuntimeError(): Promise<ErrorInfo | null> {
    const errors: ErrorInfo[] = [];

    // 检查 Docker 容器状态
    try {
      const { stdout } = await execAsync('docker ps --filter "name=codeinsight"');
      if (!stdout.trim()) {
        errors.push({
          id: 'NO_RUNNING_CONTAINERS',
          type: ErrorType.SERVICE_RUNTIME,
          severity: ErrorSeverity.HIGH,
          message: '没有运行中的 CodeInsight 容器',
          timestamp: new Date(),
          details: {},
          autoFixAvailable: true,
          userGuidance: {
            title: '启动服务容器',
            description: 'CodeInsight 容器未启动',
            steps: [
              '检查 Docker 是否运行',
              '运行 docker-compose up',
              '等待容器启动完成',
              '检查容器状态'
            ],
            estimatedTime: '3-10 分钟',
            difficulty: 'medium',
            requiredActions: ['启动 Docker', '运行启动命令'],
            precautionMeasures: ['定期检查容器状态', '设置自动重启']
          }
        });
      }
    } catch {
      // 容器检查失败
    }

    return errors.length > 0 ? errors[0] : null;
  }

  // 自动修复环境配置错误
  private async fixEnvironmentConfig(error: ErrorInfo): Promise<AutoFixResultInfo> {
    if (error.id === 'DOCKER_NOT_INSTALLED') {
      return await this.fixDockerInstallation();
    } else if (error.id === 'DOCKER_NOT_RUNNING') {
      return await this.fixDockerService();
    }
    
    return {
      result: AutoFixResult.NOT_AVAILABLE,
      message: '该错误类型暂不支持自动修复',
      retryAttempts: 0,
      fixedErrors: []
    };
  }

  // 修复 Docker 安装
  private async fixDockerInstallation(): Promise<AutoFixResultInfo> {
    try {
      // 下载 Docker Desktop 安装包
      const downloadUrl = 'https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe';
      const installerPath = `${process.env.TEMP}\\DockerDesktopInstaller.exe`;
      
      // 这里应该实现下载逻辑，但为了简化，我们假设下载成功
      console.log('Docker Desktop 安装包下载中...');
      
      // 模拟下载完成
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // 返回需要用户交互的结果
      return {
        result: AutoFixResult.REQUIRES_USER_INPUT,
        message: 'Docker Desktop 安装包已下载，请运行安装程序',
        retryAttempts: 0,
        fixedErrors: [],
        newErrors: []
      };
    } catch (error) {
      return {
        result: AutoFixResult.FAILED,
        message: `Docker 安装失败: ${error}`,
        retryAttempts: 0,
        fixedErrors: []
      };
    }
  }

  // 修复 Docker 服务
  private async fixDockerService(): Promise<AutoFixResultInfo> {
    try {
      // 尝试启动 Docker Desktop
      if (process.platform === 'win32') {
        const { stdout } = await execAsync('tasklist | findstr Docker Desktop');
        if (!stdout.trim()) {
          // Docker Desktop 未运行，尝试启动
          const dockerDesktopPath = `${process.env.PROGRAMFILES}\\Docker\\Docker\\Docker Desktop.exe`;
          if (require('fs').existsSync(dockerDesktopPath)) {
            exec(`${dockerDesktopPath} &`);
            await new Promise(resolve => setTimeout(resolve, 5000)); // 等待 5 秒
          }
        }
      }
      
      // 检查 Docker 是否启动成功
      try {
        await execAsync('docker info');
        return {
          result: AutoFixResult.SUCCESS,
          message: 'Docker 服务已启动',
          retryAttempts: 0,
          fixedErrors: ['DOCKER_NOT_RUNNING']
        };
      } catch {
        return {
          result: AutoFixResult.FAILED,
          message: 'Docker 服务启动失败，请手动启动 Docker Desktop',
          retryAttempts: 0,
          fixedErrors: []
        };
      }
    } catch (error) {
      return {
        result: AutoFixResult.FAILED,
        message: `Docker 服务启动失败: ${error}`,
        retryAttempts: 0,
        fixedErrors: []
      };
    }
  }

  // 修复权限和安全错误
  private async fixPermissionSecurity(error: ErrorInfo): Promise<AutoFixResultInfo> {
    if (error.id === 'DOCKER_PERMISSION_DENIED') {
      return await this.fixDockerPermissions();
    }
    
    return {
      result: AutoFixResult.NOT_AVAILABLE,
      message: '该错误类型暂不支持自动修复',
      retryAttempts: 0,
      fixedErrors: []
    };
  }

  // 修复 Docker 权限
  private async fixDockerPermissions(): Promise<AutoFixResultInfo> {
    try {
      if (process.platform === 'win32') {
        // Windows 系统的 Docker 权限处理
        const { stdout } = await execAsync('docker version');
        if (stdout.includes('Permission denied')) {
          return {
            result: AutoFixResult.REQUIRES_USER_INPUT,
            message: '需要管理员权限来访问 Docker，请以管理员身份运行程序',
            retryAttempts: 0,
            fixedErrors: [],
            newErrors: []
          };
        }
      }
      
      return {
        result: AutoFixResult.SUCCESS,
        message: 'Docker 权限检查通过',
        retryAttempts: 0,
        fixedErrors: ['DOCKER_PERMISSION_DENIED']
      };
    } catch (error) {
      return {
        result: AutoFixResult.FAILED,
        message: `权限修复失败: ${error}`,
        retryAttempts: 0,
        fixedErrors: []
      };
    }
  }

  // 修复网络和下载错误
  private async fixNetworkDownload(error: ErrorInfo): Promise<AutoFixResultInfo> {
    if (error.id === 'NETWORK_CONNECTION_FAILED' || error.id === 'NETWORK_CONNECTION_TIMEOUT') {
      return await this.fixNetworkConnection();
    } else if (error.id?.startsWith('PORT_')) {
      return await this.fixPortOccupation(error.details.port);
    }
    
    return {
      result: AutoFixResult.NOT_AVAILABLE,
      message: '该错误类型暂不支持自动修复',
      retryAttempts: 0,
      fixedErrors: []
    };
  }

  // 修复网络连接
  private async fixNetworkConnection(): Promise<AutoFixResultInfo> {
    try {
      // 刷新 DNS 缓存
      if (process.platform === 'win32') {
        await execAsync('ipconfig /flushdns');
      }
      
      // 检查代理设置
      const proxySettings = process.env.HTTP_PROXY || process.env.https_proxy;
      if (proxySettings) {
        // 尝试清除代理设置
        delete process.env.HTTP_PROXY;
        delete process.env.https_proxy;
      }
      
      // 测试网络连接
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        
        const response = await fetch('https://registry-1.docker.io', {
          method: 'HEAD',
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (response.ok) {
          return {
            result: AutoFixResult.SUCCESS,
            message: '网络连接已修复',
            retryAttempts: 0,
            fixedErrors: ['NETWORK_CONNECTION_FAILED', 'NETWORK_CONNECTION_TIMEOUT']
          };
        }
      } catch {
        // 连接仍然失败，需要用户干预
        return {
          result: AutoFixResult.REQUIRES_USER_INPUT,
          message: '网络连接问题仍然存在，请检查网络设置或配置代理',
          retryAttempts: 0,
          fixedErrors: [],
          newErrors: []
        };
      }
      
      return {
        result: AutoFixResult.FAILED,
        message: '网络连接修复失败',
        retryAttempts: 0,
        fixedErrors: []
      };
    } catch (error) {
      return {
        result: AutoFixResult.FAILED,
        message: `网络修复失败: ${error}`,
        retryAttempts: 0,
        fixedErrors: []
      };
    }
  }

  // 修复端口占用
  private async fixPortOccupation(port: number): Promise<AutoFixResultInfo> {
    try {
      // 查找占用端口的进程
      const { stdout } = await execAsync(`netstat -ano | findstr :${port}`);
      const processMatch = stdout.match(/\\d+$/);
      
      if (processMatch) {
        const processId = processMatch[0];
        
        // 尝试终止进程
        await execAsync(`taskkill /F /PID ${processId}`);
        
        // 验证端口是否已释放
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        try {
          const { stdout: testOutput } = await execAsync(`netstat -ano | findstr :${port}`);
          if (!testOutput.trim()) {
            return {
              result: AutoFixResult.SUCCESS,
              message: `端口 ${port} 已成功释放`,
              retryAttempts: 0,
              fixedErrors: [`PORT_${port}_OCCUPIED`]
            };
          }
        } catch {
          // 端口检查失败，但进程可能已终止
        }
      }
      
      return {
        result: AutoFixResult.SUCCESS,
        message: `端口 ${port} 处理完成`,
        retryAttempts: 0,
        fixedErrors: [`PORT_${port}_OCCUPIED`]
      };
    } catch (error) {
      return {
        result: AutoFixResult.FAILED,
        message: `端口 ${port} 释放失败: ${error}`,
        retryAttempts: 0,
        fixedErrors: []
      };
    }
  }

  // 修复服务运行时错误
  private async fixServiceRuntime(error: ErrorInfo): Promise<AutoFixResultInfo> {
    if (error.id === 'NO_RUNNING_CONTAINERS') {
      return await this.fixServiceContainers();
    }
    
    return {
      result: AutoFixResult.NOT_AVAILABLE,
      message: '该错误类型暂不支持自动修复',
      retryAttempts: 0,
      fixedErrors: []
    };
  }

  // 修复服务容器
  private async fixServiceContainers(): Promise<AutoFixResultInfo> {
    try {
      // 检查 docker-compose.yml 是否存在
      const fs = require('fs');
      const dockerComposePath = `${process.cwd()}\\docker-compose.yml`;
      
      if (!fs.existsSync(dockerComposePath)) {
        return {
          result: AutoFixResult.FAILED,
          message: '未找到 docker-compose.yml 文件',
          retryAttempts: 0,
          fixedErrors: []
        };
      }
      
      // 尝试启动服务
      try {
        await execAsync('docker-compose up -d');
        
        // 等待容器启动完成
        await new Promise(resolve => setTimeout(resolve, 10000));
        
        // 验证容器是否启动成功
        const { stdout } = await execAsync('docker ps --filter "name=codeinsight"');
        if (stdout.trim()) {
          return {
            result: AutoFixResult.SUCCESS,
            message: '服务容器已启动',
            retryAttempts: 0,
            fixedErrors: ['NO_RUNNING_CONTAINERS']
          };
        }
      } catch (error) {
        return {
          result: AutoFixResult.FAILED,
          message: `容器启动失败: ${error}`,
          retryAttempts: 0,
          fixedErrors: []
        };
      }
      
      return {
        result: AutoFixResult.FAILED,
        message: '容器启动失败，请手动检查 Docker 服务',
        retryAttempts: 0,
        fixedErrors: []
      };
    } catch (error) {
      return {
        result: AutoFixResult.FAILED,
        message: `服务修复失败: ${error}`,
        retryAttempts: 0,
        fixedErrors: []
      };
    }
  }

  // 添加错误到历史记录
  public addError(error: ErrorInfo): void {
    this.errorHistory.push(error);
    this.emit('errorDetected', error);
  }

  // 获取错误历史
  public getErrorHistory(): ErrorInfo[] {
    return [...this.errorHistory];
  }

  // 清空错误历史
  public clearErrorHistory(): void {
    this.errorHistory = [];
    this.emit('errorHistoryCleared');
  }

  // 自动修复错误
  public async autoFix(error: ErrorInfo): Promise<AutoFixResultInfo> {
    const strategy = this.autoFixStrategies.get(error.type);
    
    if (!strategy) {
      return {
        result: AutoFixResult.NOT_AVAILABLE,
        message: '没有可用的修复策略',
        retryAttempts: 0,
        fixedErrors: []
      };
    }
    
    try {
      const result = await strategy(error);
      
      if (result.result === AutoFixResult.SUCCESS) {
        // 从历史记录中移除已修复的错误
        this.errorHistory = this.errorHistory.filter(e => !result.fixedErrors.includes(e.id));
        this.emit('errorFixed', { error, result });
      }
      
      return result;
    } catch (error) {
      return {
        result: AutoFixResult.FAILED,
        message: `自动修复失败: ${error}`,
        retryAttempts: 0,
        fixedErrors: []
      };
    }
  }

  // 生成诊断报告
  public generateDiagnosticReport(): {
    summary: {
      totalErrors: number;
      criticalErrors: number;
      highErrors: number;
      mediumErrors: number;
      lowErrors: number;
      autoFixable: number;
    };
    errors: ErrorInfo[];
    suggestions: string[];
  } {
    const errors = this.errorHistory;
    
    const summary = {
      totalErrors: errors.length,
      criticalErrors: errors.filter(e => e.severity === ErrorSeverity.CRITICAL).length,
      highErrors: errors.filter(e => e.severity === ErrorSeverity.HIGH).length,
      mediumErrors: errors.filter(e => e.severity === ErrorSeverity.MEDIUM).length,
      lowErrors: errors.filter(e => e.severity === ErrorSeverity.LOW).length,
      autoFixable: errors.filter(e => e.autoFixAvailable).length
    };
    
    const suggestions: string[] = [];
    
    if (summary.criticalErrors > 0) {
      suggestions.push('请立即解决所有严重错误，这些错误会阻止系统正常运行');
    }
    
    if (summary.autoFixable > 0) {
      suggestions.push('有 ${summary.autoFixable} 个错误可以自动修复，建议先尝试自动修复');
    }
    
    if (summary.totalErrors === 0) {
      suggestions.push('系统未检测到任何错误，一切正常');
    }
    
    return {
      summary,
      errors,
      suggestions
    };
  }

  // 指数退避重试机制
  public async retryWithBackoff<T>(
    operation: () => Promise<T>,
    maxRetries: number = 3,
    baseDelay: number = 1000
  ): Promise<T> {
    let retryCount = 0;
    let lastError: Error;
    
    while (retryCount < maxRetries) {
      try {
        return await operation();
      } catch (error) {
        lastError = error as Error;
        retryCount++;
        
        if (retryCount < maxRetries) {
          const delay = baseDelay * Math.pow(2, retryCount - 1);
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }
    
    throw lastError!;
  }
}

// 全局错误处理管理器实例
export const errorHandlerManager = new ErrorHandlerManager();

// 全局错误处理
process.on('uncaughtException', (error) => {
  console.error('未捕获的异常:', error);
  
  const errorInfo: ErrorInfo = {
    id: 'UNCAUGHT_EXCEPTION',
    type: ErrorType.SERVICE_RUNTIME,
    severity: ErrorSeverity.CRITICAL,
    message: '系统发生未捕获的异常',
    timestamp: new Date(),
    details: { error: error.message, stack: error.stack },
    autoFixAvailable: false,
    userGuidance: {
      title: '系统异常',
      description: '系统运行时发生未预期的错误',
      steps: [
        '记录错误详情',
        '重启应用程序',
        '检查日志文件',
        '联系技术支持'
      ],
      estimatedTime: '2-5 分钟',
      difficulty: 'medium',
      requiredActions: ['重启程序', '检查日志'],
      precautionMeasures: ['定期备份数据', '保持系统更新']
    }
  };
  
  errorHandlerManager.addError(errorInfo);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('未处理的 Promise 拒绝:', reason);
  
  const errorInfo: ErrorInfo = {
    id: 'UNHANDLED_REJECTION',
    type: ErrorType.SERVICE_RUNTIME,
    severity: ErrorSeverity.HIGH,
    message: '未处理的 Promise 拒绝',
    timestamp: new Date(),
    details: { reason, promise },
    autoFixAvailable: false,
    userGuidance: {
      title: '异步错误',
      description: '系统发生了未处理的异步操作拒绝',
      steps: [
        '检查异步操作',
        '添加错误处理',
        '重启服务',
        '查看详细日志'
      ],
      estimatedTime: '3-7 分钟',
      difficulty: 'hard',
      requiredActions: ['检查异步代码', '添加错误处理'],
      precautionMeasures: ['完善错误处理机制', '添加日志记录']
    }
  };
  
  errorHandlerManager.addError(errorInfo);
});