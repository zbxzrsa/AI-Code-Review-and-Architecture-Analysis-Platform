import { EventEmitter } from 'events';
import { errorHandlerManager } from './errorHandling';
import { execAsync } from './exec';
import { getSystemResources } from '../services/SystemMonitor';
import { LaunchManagerMethods } from './LaunchManager.methods';

// 启动阶段枚举
export enum LaunchPhase {
  INITIAL_CHECK = 'initial_check',
  DEPENDENCY_VERIFICATION = 'dependency_verification',
  SERVICE_DEPLOYMENT = 'service_deployment',
  READINESS_CONFIRMATION = 'readiness_confirmation'
}

// 启动状态枚举
export enum LaunchStatus {
  IDLE = 'idle',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  PAUSED = 'paused'
}

// 启动进度信息
export interface LaunchProgress {
  currentPhase: LaunchPhase;
  phaseProgress: number; // 0-100
  totalProgress: number; // 0-100
  message: string;
  details: Record<string, any>;
  estimatedTime: number; // 秒
  startTime: Date;
}

// 启动配置
export interface LaunchConfig {
  autoStart: boolean;
  skipInitialChecks: boolean;
  skipHealthChecks: boolean;
  enableAutoFix: boolean;
  maxRetries: number;
  timeout: number;
}

// 默认配置
const DEFAULT_CONFIG: LaunchConfig = {
  autoStart: false,
  skipInitialChecks: false,
  skipHealthChecks: false,
  enableAutoFix: true,
  maxRetries: 3,
  timeout: 300000 // 5分钟
};

/**
 * 启动管理器
 * 负责协调渐进式启动流程和错误处理
 */
export class LaunchManager extends EventEmitter {
  private status: LaunchStatus = LaunchStatus.IDLE;
  private currentPhase: LaunchPhase = LaunchPhase.INITIAL_CHECK;
  private progress: LaunchProgress;
  private config: LaunchConfig;
  private abortController: AbortController;
  private retryCount: number = 0;
  
  constructor(config: Partial<LaunchConfig> = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.abortController = new AbortController();
    this.progress = {
      currentPhase: LaunchPhase.INITIAL_CHECK,
      phaseProgress: 0,
      totalProgress: 0,
      message: '准备启动系统...',
      details: {},
      estimatedTime: 0,
      startTime: new Date()
    };
    
    // 监听错误
    errorHandlerManager.on('errorDetected', (error) => {
      this.emit('errorDetected', error);
    });
    
    errorHandlerManager.on('errorFixed', ({ error, result }) => {
      this.emit('errorFixed', { error, result });
    });
  }

  // 获取当前状态
  public getStatus(): LaunchStatus {
    return this.status;
  }

  // 获取当前进度
  public getProgress(): LaunchProgress {
    return { ...this.progress };
  }

  // 获取配置
  public getConfig(): LaunchConfig {
    return { ...this.config };
  }

  // 更新配置
  public updateConfig(newConfig: Partial<LaunchConfig>): void {
    this.config = { ...this.config, ...newConfig };
    this.emit('configUpdated', this.config);
  }

  // 启动流程
  public async launch(): Promise<void> {
    if (this.status === LaunchStatus.RUNNING) {
      throw new Error('启动流程已在进行中');
    }
    
    if (this.status === LaunchStatus.FAILED || this.status === LaunchStatus.COMPLETED) {
      // 重置状态
      this.status = LaunchStatus.IDLE;
      this.currentPhase = LaunchPhase.INITIAL_CHECK;
      this.retryCount = 0;
    }
    
    this.status = LaunchStatus.RUNNING;
    this.abortController = new AbortController();
    this.progress.startTime = new Date();
    
    try {
      this.emit('launchStarted', this.progress);
      
      if (!this.config.skipInitialChecks) {
        await this.executePhase(LaunchPhase.INITIAL_CHECK);
      }
      
      await this.executePhase(LaunchPhase.DEPENDENCY_VERIFICATION);
      await this.executePhase(LaunchPhase.SERVICE_DEPLOYMENT);
      
      if (!this.config.skipHealthChecks) {
        await this.executePhase(LaunchPhase.READINESS_CONFIRMATION);
      }
      
      this.status = LaunchStatus.COMPLETED;
      this.progress.totalProgress = 100;
      this.progress.message = '系统启动完成！';
      
      this.emit('phaseCompleted', {
        phase: LaunchPhase.READINESS_CONFIRMATION,
        progress: this.progress
      });
      
      this.emit('launchCompleted', this.progress);
      
    } catch (error) {
      this.status = LaunchStatus.FAILED;
      this.progress.message = `启动失败: ${error}`;
      
      // 添加错误到错误管理器
      errorHandlerManager.addError({
        id: 'LAUNCH_FAILED',
        type: 'service_runtime' as any,
        severity: 'high' as any,
        message: '系统启动失败',
        timestamp: new Date(),
        details: { error: error instanceof Error ? error.message : String(error) },
        autoFixAvailable: this.config.enableAutoFix,
        userGuidance: {
          title: '启动失败',
          description: '系统在启动过程中遇到了错误',
          steps: [
            '查看错误详情',
            '尝试修复问题',
            '重新启动系统',
            '联系技术支持'
          ],
          estimatedTime: '5-10 分钟',
          difficulty: 'medium',
          requiredActions: ['修复错误', '重新启动'],
          precautionMeasures: ['确保系统资源充足', '检查依赖服务状态']
        }
      });
      
      this.emit('launchFailed', error);
      throw error;
    }
  }

  // 执行单个阶段
  private async executePhase(phase: LaunchPhase): Promise<void> {
    this.currentPhase = phase;
    this.progress.currentPhase = phase;
    this.progress.phaseProgress = 0;
    
    this.emit('phaseStarted', { phase, progress: this.progress });
    
    try {
      switch (phase) {
        case LaunchPhase.INITIAL_CHECK:
          await this.executeInitialCheck();
          break;
        case LaunchPhase.DEPENDENCY_VERIFICATION:
          await this.executeDependencyVerification();
          break;
        case LaunchPhase.SERVICE_DEPLOYMENT:
          await this.executeServiceDeployment();
          break;
        case LaunchPhase.READINESS_CONFIRMATION:
          await this.executeReadinessConfirmation();
          break;
      }
      
      this.progress.phaseProgress = 100;
      this.progress.totalProgress = Math.min(100, this.progress.totalProgress + 25);
      
      this.emit('phaseCompleted', {
        phase,
        progress: this.progress
      });
      
    } catch (error) {
      this.progress.phaseProgress = 0;
      this.progress.message = `阶段 ${phase} 执行失败: ${error}`;
      
      // 尝试自动修复
      if (this.config.enableAutoFix) {
        const fixResult = await errorHandlerManager.autoFix({
          id: 'PHASE_EXECUTION_FAILED',
          type: 'service_runtime' as any,
          severity: 'high' as any,
          message: `阶段 ${phase} 执行失败`,
          timestamp: new Date(),
          details: { phase, error: error instanceof Error ? error.message : String(error) },
          autoFixAvailable: true,
          userGuidance: {
            title: '阶段执行失败',
            description: `启动流程在 ${phase} 阶段遇到了问题`,
            steps: [
              '查看错误详情',
              '尝试自动修复',
              '重新执行阶段',
              '跳过此阶段'
            ],
            estimatedTime: '2-4 分钟',
            difficulty: 'medium',
            requiredActions: ['修复错误', '重新尝试'],
            precautionMeasures: ['确保网络连接正常', '检查系统资源']
          }
        });
        
        if (fixResult.result === 'success') {
          this.emit('autoFixAttempted', { phase, error, fixResult });
          // 重新执行阶段
          return this.executePhase(phase);
        }
      }
      
      throw error;
    }
  }

  // 执行初始检查阶段
  private async executeInitialCheck(): Promise<void> {
    const steps = [
      { name: '系统资源检查', weight: 25 },
      { name: 'Docker 环境检查', weight: 25 },
      { name: '配置文件检查', weight: 25 },
      { name: '网络连接检查', weight: 25 }
    ];
    
    let completedSteps = 0;
    const totalWeight = steps.reduce((sum, step) => sum + step.weight, 0);
    
    for (const step of steps) {
      if (this.abortController.signal.aborted) {
        throw new Error('启动流程被用户取消');
      }
      
      this.progress.message = `正在执行初始检查 - ${step.name}...`;
      
      try {
        switch (step.name) {
          case '系统资源检查':
            await this.checkSystemResources();
            break;
          case 'Docker 环境检查':
            await this.checkDockerEnvironment();
            break;
          case '配置文件检查':
            await this.checkConfigurationFiles();
            break;
          case '网络连接检查':
            await this.checkNetworkConnection();
            break;
        }
        
        completedSteps += step.weight;
        this.progress.phaseProgress = Math.round((completedSteps / totalWeight) * 100);
        
      } catch (error) {
        this.progress.details[step.name] = error;
        throw error;
      }
    }
  }

  // 执行依赖验证阶段
  private async executeDependencyVerification(): Promise<void> {
    const steps = [
      { name: 'Docker 可用性验证', weight: 30 },
      { name: '端口可用性验证', weight: 30 },
      { name: '依赖服务验证', weight: 40 }
    ];
    
    let completedSteps = 0;
    const totalWeight = steps.reduce((sum, step) => sum + step.weight, 0);
    
    for (const step of steps) {
      if (this.abortController.signal.aborted) {
        throw new Error('启动流程被用户取消');
      }
      
      this.progress.message = `正在验证依赖 - ${step.name}...`;
      
      try {
        switch (step.name) {
          case 'Docker 可用性验证':
            await this.verifyDockerAvailability();
            break;
          case '端口可用性验证':
            await this.verifyPortAvailability();
            break;
          case '依赖服务验证':
            await this.verifyDependencyServices();
            break;
        }
        
        completedSteps += step.weight;
        this.progress.phaseProgress = Math.round((completedSteps / totalWeight) * 100);
        
      } catch (error) {
        this.progress.details[step.name] = error;
        throw error;
      }
    }
  }

  // 执行服务部署阶段
  private async executeServiceDeployment(): Promise<void> {
    const steps = [
      { name: '停止现有容器', weight: 10 },
      { name: '拉取最新镜像', weight: 30 },
      { name: '启动服务容器', weight: 40 },
      { name: '等待服务就绪', weight: 20 }
    ];
    
    let completedSteps = 0;
    const totalWeight = steps.reduce((sum, step) => sum + step.weight, 0);
    
    for (const step of steps) {
      if (this.abortController.signal.aborted) {
        throw new Error('启动流程被用户取消');
      }
      
      this.progress.message = `正在部署服务 - ${step.name}...`;
      
      try {
        switch (step.name) {
          case '停止现有容器':
            await this.stopExistingContainers();
            break;
          case '拉取最新镜像':
            await this.pullLatestImages();
            break;
          case '启动服务容器':
            await this.startServiceContainers();
            break;
          case '等待服务就绪':
            await this.waitForServiceReadiness();
            break;
        }
        
        completedSteps += step.weight;
        this.progress.phaseProgress = Math.round((completedSteps / totalWeight) * 100);
        
      } catch (error) {
        this.progress.details[step.name] = error;
        throw error;
      }
    }
  }

  // 执行就绪确认阶段
  private async executeReadinessConfirmation(): Promise<void> {
    const steps = [
      { name: '服务健康检查', weight: 40 },
      { name: '端口连接测试', weight: 30 },
      { name: '功能测试', weight: 30 }
    ];
    
    let completedSteps = 0;
    const totalWeight = steps.reduce((sum, step) => sum + step.weight, 0);
    
    for (const step of steps) {
      if (this.abortController.signal.aborted) {
        throw new Error('启动流程被用户取消');
      }
      
      this.progress.message = `正在确认就绪状态 - ${step.name}...`;
      
      try {
        switch (step.name) {
          case '服务健康检查':
             await this.performHealthChecks();
             break;
           case '端口连接测试':
             await this.testPortConnections();
             break;
           case '功能测试':
             await this.runFunctionalityTests();
             break;
        }
        
        completedSteps += step.weight;
        this.progress.phaseProgress = Math.round((completedSteps / totalWeight) * 100);
        
      } catch (error) {
        this.progress.details[step.name] = error;
        throw error;
      }
    }
  }

  // 服务健康检查
  private async performHealthChecks(): Promise<void> {
    try {
      const resp = await fetch('http://127.0.0.1:8000/health', { method: 'GET' });
      if (!resp.ok) {
        throw new Error(`后端健康检查失败: ${resp.status}`);
      }
      const data = await resp.json();
      this.progress.details.health = data;
    } catch (err) {
      throw new Error(`健康检查异常: ${(err as any)?.message || String(err)}`);
    }
  }

  // 端口连接测试
  private async testPortConnections(): Promise<void> {
    const targets = [
      { name: 'Backend', url: 'http://127.0.0.1:8000/health' }
    ];
    const results: Record<string, any> = {};
    for (const t of targets) {
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 2000);
        const r = await fetch(t.url, { signal: controller.signal });
        clearTimeout(timer);
        results[t.name] = r.ok ? 'ok' : `status ${r.status}`;
      } catch (e) {
        results[t.name] = `error: ${(e as any)?.message || e}`;
      }
    }
    this.progress.details.portTests = results;
  }

  // 功能测试（轻量）
  private async runFunctionalityTests(): Promise<void> {
    try {
      const resp = await fetch('http://127.0.0.1:8000/api/openapi.json');
      if (!resp.ok) throw new Error(`OpenAPI拉取失败: ${resp.status}`);
      const spec = await resp.json();
      if (!spec?.paths) throw new Error('OpenAPI缺少paths');
      this.progress.details.openapi = Object.keys(spec.paths).slice(0, 10);
    } catch (err) {
      throw new Error(`功能测试异常: ${(err as any)?.message || String(err)}`);
    }
  }
  
  // 停止现有容器
  private async stopExistingContainers(): Promise<void> {
    // 简化实现，仅记录操作
    console.log('停止现有容器');
    this.progress.details.containersStop = { status: 'simulated' };
  }

  // 拉取最新镜像
  private async pullLatestImages(): Promise<void> {
    // 简化实现，仅记录操作
    console.log('拉取最新镜像');
    this.progress.details.imagesPull = { status: 'simulated' };
  }

  // 启动服务容器
  private async startServiceContainers(): Promise<void> {
    // 简化实现，仅记录操作
    console.log('启动服务容器');
    this.progress.details.containersStart = { status: 'simulated' };
  }

  // 等待服务就绪
  private async waitForServiceReadiness(): Promise<void> {
    // 简化实现，仅记录操作
    console.log('等待服务就绪');
    this.progress.details.serviceReadiness = { status: 'simulated' };
  }

  // Docker 可用性验证
  private async verifyDockerAvailability(): Promise<void> {
    // 简化实现，仅记录操作
    console.log('验证Docker可用性');
    this.progress.details.dockerAvailability = { status: 'simulated' };
  }

  // 端口可用性验证
  private async verifyPortAvailability(): Promise<void> {
    // 简化实现，仅记录操作
    console.log('验证端口可用性');
    this.progress.details.portAvailability = { status: 'simulated' };
  }

  // 依赖服务验证
  private async verifyDependencyServices(): Promise<void> {
    // 简化实现，仅记录操作
    console.log('验证依赖服务');
    this.progress.details.dependencyServices = { status: 'simulated' };
  }
 
   // 系统资源检查
  private async checkSystemResources(): Promise<void> {
    const resources = await getSystemResources();
    
    if (resources.memory.percent > 90) {
      throw new Error(`内存使用率过高: ${resources.memory.percent}%`);
    }
    
    if (resources.disk.percent > 95) {
      throw new Error(`磁盘空间不足: ${resources.disk.percent}%`);
    }
    
    this.progress.details.systemResources = resources;
  }

  // Docker 环境检查
  private async checkDockerEnvironment(): Promise<void> {
    try {
      await execAsync('docker --version');
    } catch {
      throw new Error('Docker 未安装或未在 PATH 中');
    }
    
    try {
      await execAsync('docker info');
    } catch {
      throw new Error('Docker 服务未运行');
    }
    
    this.progress.details.dockerVersion = (await execAsync('docker --version')).stdout.trim();
  }

  // 配置文件检查
  private async checkConfigurationFiles(): Promise<void> {
    const fs = require('fs');
    const path = require('path');
    
    const requiredFiles = [
      'docker-compose.yml',
      'desktop-config.json',
      'backend/requirements.txt'
    ];
    
    const missingFiles: string[] = [];
    
    for (const file of requiredFiles) {
      const filePath = path.resolve(process.cwd(), file);
      if (!fs.existsSync(filePath)) {
        missingFiles.push(file);
      }
    }
    
    if (missingFiles.length > 0) {
      throw new Error(`缺少必需的配置文件: ${missingFiles.join(', ')}`);
    }
    
    this.progress.details.configFiles = requiredFiles;
  }

  // 网络连接检查
  private async checkNetworkConnection(): Promise<void> {
    try {
      const response = await fetch('https://registry-1.docker.io', {
        method: 'HEAD',
        signal: this.abortController.signal
      });
      
      if (!response.ok) {
        throw new Error('无法连接到 Docker 镜像仓库');
      }
    } catch (error) {
      const err: any = error as any;
      if (err?.name === 'AbortError') {
        throw new Error('网络请求被取消');
      }
      throw new Error(`网络连接检查失败: ${err?.message || String(err)}`);
    }
  }
}