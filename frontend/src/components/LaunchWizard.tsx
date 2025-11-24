import React, { useState, useEffect, useCallback } from 'react';
import { SystemMonitor } from '../services/SystemMonitor';
import { Button, Card, Progress, Alert, List, Spin, Space, Typography, message, Steps, Tag, Divider, Modal, Tabs, Result, Statistic } from 'antd';
import { PlayCircleOutlined, PauseCircleOutlined, StopOutlined, ReloadOutlined, InfoCircleOutlined, ExclamationCircleOutlined, CheckCircleOutlined, ClockCircleOutlined } from '@ant-design/icons';


interface LogEntry {
  id: number;
  timestamp: Date;
  level: 'info' | 'warn' | 'error';
  message: string;
}

interface SystemResource {
  cpu: number;
  memory: number;
  disk: number;
}

const LaunchWizard: React.FC = () => {
  const [status, setStatus] = useState<'idle' | 'checking' | 'starting' | 'running' | 'error'>('idle');
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [systemResources, setSystemResources] = useState<SystemResource>({ cpu: 0, memory: 0, disk: 0 });
  const [error, setError] = useState<string | null>(null);
  const [suggestedSolutions, setSuggestedSolutions] = useState<string[]>([]);
  const [currentPhase, setCurrentPhase] = useState(0);
  const [phaseProgress, setPhaseProgress] = useState([0, 0, 0, 0]);
  const [systemDiagnostics, setSystemDiagnostics] = useState<any>(null);
  const [errorDetails, setErrorDetails] = useState<any>(null);
  const [isAutoFixEnabled, setIsAutoFixEnabled] = useState(true);
  const [estimatedTime, setEstimatedTime] = useState(0);
  const [launchManager, setLaunchManager] = useState<any | null>(null);
  const [errorHandler, setErrorHandler] = useState<any | null>(null);
  const [systemMonitor] = useState(() => new SystemMonitor());
  const [activeTab, setActiveTab] = useState('progress');
  const [showDiagnosticReport, setShowDiagnosticReport] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  // 移除 Node 专用依赖，避免浏览器打包失败
  // import { ServiceManager } from '../services/ServiceManager';
  // import { LaunchManager } from '../services/LaunchManager';
  // import { ErrorHandlerManager } from '../services/errorHandling';
  
  // 添加日志条目
  const addLog = useCallback((level: 'info' | 'warn' | 'error', message: string) => {
    const newLog: LogEntry = {
      id: Date.now(),
      timestamp: new Date(),
      level,
      message
    };
    setLogs(prev => [...prev.slice(-20), newLog]); // 只保留最近20条日志
  }, []);

  // 更新系统资源监控
  const updateSystemResources = useCallback(async () => {
    try {
      if (status === 'running') {
        const resources = await systemMonitor.getResources();
        setSystemResources(resources);
      }
    } catch (_) {}
  }, [status, systemMonitor]);

  // 生成错误解决方案
  const generateSolutions = useCallback((errorMessage: string) => {
    const solutions: string[] = [];
    
    if (errorMessage.includes('端口占用') || errorMessage.includes('port')) {
      solutions.push('1. 检查是否有其他程序占用了相关端口');
      solutions.push('2. 尝试停止占用端口的进程');
      solutions.push('3. 修改配置文件中的端口设置');
    }
    
    if (errorMessage.includes('Docker') || errorMessage.includes('容器')) {
      solutions.push('1. 确保 Docker 服务已启动');
      solutions.push('2. 检查 Docker 配置是否正确');
      solutions.push('3. 尝试重启 Docker 服务');
    }
    
    if (errorMessage.includes('内存') || errorMessage.includes('memory')) {
      solutions.push('1. 关闭不必要的应用程序');
      solutions.push('2. 增加系统内存');
      solutions.push('3. 调整 Docker 内存限制');
    }
    
    if (errorMessage.includes('权限') || errorMessage.includes('permission')) {
      solutions.push('1. 以管理员身份运行应用程序');
      solutions.push('2. 检查文件权限设置');
      solutions.push('3. 重新安装应用程序');
    }
    
    // 默认解决方案
    if (solutions.length === 0) {
      solutions.push('1. 重启应用程序');
      solutions.push('2. 检查系统日志获取更多信息');
      solutions.push('3. 联系技术支持');
    }
    
    return solutions;
  }, []);

  // 错误处理函数（提前声明，避免依赖数组引用报错）
  const handleError = useCallback((title: string, description: string, errorType?: string) => {
    setError(description);
    setStatus('error');
    
    if (errorHandler) {
      const errorInfo = errorHandler.addError({
        type: errorType || 'runtime',
        message: description,
        timestamp: new Date(),
        severity: 'high',
        context: { title }
      });
      
      const solutions = errorHandler.generateDiagnosticReport().solutions;
      setSuggestedSolutions(solutions);
      setErrorDetails(errorInfo);
    }
    
    addLog('error', `${title}: ${description}`);
  }, [errorHandler]);

  // 重试函数（提前声明，供启动失败时调用）
  const handleRetry = useCallback(async () => {
    setRetryCount(prev => prev + 1);
    addLog('info', `Retry starting (attempt ${retryCount + 1})...`);
    
    // 指数退避
    const delay = Math.min(1000 * Math.pow(2, retryCount), 30000);
    await new Promise(resolve => setTimeout(resolve, delay));
    
    startServices();
  }, [retryCount]);

  // 启动服务
  const startServices = useCallback(async () => {
    try {
      setStatus('starting');
      setProgress(0);
      setCurrentPhase(0);
      setPhaseProgress([0, 0, 0, 0]);
      setRetryCount(0);
      setIsPaused(false);
      setError(null);
      setSuggestedSolutions([]);
      addLog('info', 'Starting launch process...');
    
      if (launchManager?.launch) {
        await launchManager.launch();
        return;
      }
    
      // Electron 环境：通过预加载暴露的 API 启动
      if ((window as any).codeinsight?.startServices) {
        await (window as any).codeinsight.startServices();
        setStatus('running');
        addLog('info', 'Launch request sent');
      } else {
        // 浏览器预览模式：不实际启动，进入运行态以便查看 UI
        addLog('warn', 'Browser preview mode: not actually starting services');
        setStatus('running');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to start services';
      handleError('Launch Failed', errorMessage);
    
      if (retryCount < 3) {
        setTimeout(() => {
          Modal.confirm({
            title: 'Launch Failed',
            content: `Retry launch? (attempts ${retryCount + 1})`,
            onOk: handleRetry,
            onCancel: () => {},
            okText: 'Retry',
            cancelText: 'Cancel'
          });
        }, 1000);
      }
    }
  }, [launchManager, handleError, retryCount]);

  // 新增：停止服务
  const stopServices = useCallback(async () => {
    try {
      await (window as any).codeinsight?.stopServices?.();
      addLog('info', 'Stop services requested');
      setStatus('idle');
    } catch (_) {
      addLog('warn', 'Stopping services not available in this environment');
    }
  }, []);
  
  // 监听启动管理器的事件
  useEffect(() => {
    if (!launchManager) return;
    
    const handleProgressUpdate = (phase: number, phaseProgress: number, overallProgress: number, estimatedTime: number) => {
      setCurrentPhase(phase);
      setPhaseProgress(prev => {
        const newProgress = [...prev];
        newProgress[phase] = phaseProgress;
        return newProgress;
      });
      setProgress(overallProgress);
      setEstimatedTime(estimatedTime);
      
      addLog('info', `Phase ${phase + 1}: ${phaseProgress}% (${estimatedTime}s)`);
    };
    
    const handlePhaseComplete = (phase: number, success: boolean) => {
      addLog(success ? 'info' : 'error', `Phase ${phase + 1} ${success ? 'completed' : 'failed'}`);
      
      if (!success && errorHandler) {
        const errorInfo = errorHandler.getErrorHistory().slice(-1)[0];
        if (errorInfo) {
          handleError(`Phase ${phase + 1} failed`, errorInfo.message, errorInfo.type);
        }
      }
    };
    
    const handleLaunchComplete = (success: boolean) => {
      if (success) {
        setStatus('running');
        addLog('info', 'All services started');
        
        // 打开浏览器
          setTimeout(() => {
            // 使用普通方式打开浏览器，兼容非Electron环境
            window.open('http://localhost:3000', '_blank');
          }, 2000);
      } else {
        setStatus('error');
      }
    };
    
    launchManager.onProgressUpdate(handleProgressUpdate);
    launchManager.onPhaseComplete(handlePhaseComplete);
    launchManager.onLaunchComplete(handleLaunchComplete);
    
    return () => {
      launchManager.offProgressUpdate(handleProgressUpdate);
      launchManager.offPhaseComplete(handlePhaseComplete);
      launchManager.offLaunchComplete(handleLaunchComplete);
    };
  }, [launchManager, errorHandler]);



  // 初始化（移除 Node 依赖，仅初始化前端监控）
  useEffect(() => {
    const initManagers = async () => {
      try {
        // 可选：在 Electron 环境下获取一次诊断
        try {
          const bridgeDiag = (window as any).codeinsight?.diagnostics;
          if (bridgeDiag?.run) {
            const diag = await bridgeDiag.run();
            setSystemDiagnostics(diag);
          }
        } catch (_) {}
      } catch (err) {
        console.error('Initialization failed:', err);
        handleError('Initialization failed', 'Unable to start system monitoring');
      }
    };
    initManagers();
    
    // 浏览器环境下的资源轮询（Electron 订阅另有 useEffect）
    let interval: any;
    if (status === 'running') {
      interval = setInterval(updateSystemResources, 2000);
    }
    return () => interval && clearInterval(interval);
  }, [status, updateSystemResources]);

// 新增：Electron 监控桥接订阅（带浏览器回退）
  useEffect(() => {
    const bridge = (window as any).codeinsight?.monitor;
    let unsubscribe: null | (() => void) = null;
    let interval: any = null;
    if (bridge) {
      unsubscribe = bridge.onPush((snap: any) => {
        setSystemResources({
          cpu: snap?.cpu?.pct || 0,
          memory: snap?.mem?.usedPct || 0,
          disk: snap?.disk?.usedPct || 0
        });
        if (snap?.alerts?.length) {
          const msg = 'Resource alert: ' + snap.alerts.map((a: any) => a.message).join('; ');
          addLog('warn', msg);
        }
      });
      try { bridge.start(); } catch (_) {}
      try {
        bridge.getSnapshot?.().then((res: any) => {
          const data = res?.data || res;
          if (data) {
            setSystemResources({
              cpu: data?.cpu?.pct || 0,
              memory: data?.mem?.usedPct || 0,
              disk: data?.disk?.usedPct || 0
            });
          }
        }).catch(() => {});
      } catch (_) {}
    } else {
      // 浏览器预览回退：周期调用内部 SystemMonitor（若不可用则保留默认值）
      interval = setInterval(() => { updateSystemResources(); }, 3000);
    }
    return () => {
      if (unsubscribe) unsubscribe();
      try { bridge?.stop(); } catch (_) {}
      if (interval) clearInterval(interval);
    };
  }, [updateSystemResources, addLog]);
  
  // 错误处理函数已提前声明在文件前部
  
  // 一键修复函数
  const handleAutoFix = useCallback(async () => {
    if (!errorHandler || !errorDetails) return;
    
    addLog('info', 'Starting auto-fix...');
    
    try {
      const result = await errorHandler.autoFix(errorDetails.id);
      
      if (result.success) {
        addLog('info', 'Auto-fix succeeded');
        message.success('Issues have been auto-fixed');
        
        // 重新获取系统诊断
        if (systemMonitor) {
          // 简化诊断信息获取
          // 暂时跳过诊断信息更新，避免类型错误
          import('../utils/logger').then(({ defaultLogger }) => {
            defaultLogger.info('System diagnostics update skipped');
          });
        }
        
        // 清除错误状态
        setError('');
        setErrorDetails(null);
        setSuggestedSolutions([]);
        
        // 重试启动
        setTimeout(() => {
          startServices();
        }, 1000);
      } else {
        addLog('error', `Auto-fix failed: ${result.message}`);
        message.error('Auto-fix failed, please resolve manually');
      }
    } catch (err) {
      addLog('error', `Auto-fix exception: ${err}`);
      message.error('An exception occurred during auto-fix');
    }
  }, [errorHandler, errorDetails, systemMonitor]);
  
  // 重试函数已提前声明在文件前部
  
  // 跳过问题函数
  const handleSkip = useCallback(() => {
    addLog('warn', 'Skipped current issue, continue launch');
    message.warning('Issue skipped; some features may be affected');
    
    // 继续下一阶段
    if (currentPhase < 3) {
      setCurrentPhase(prev => prev + 1);
      setPhaseProgress(prev => {
        const newProgress = [...prev];
        newProgress[currentPhase] = 100;
        return newProgress;
      });
    }
  }, [currentPhase]);
  
  // 暂停/继续函数
  const handlePauseResume = useCallback(() => {
    setIsPaused(!isPaused);
    addLog(isPaused ? 'info' : 'warn', !isPaused ? 'Continue launch' : 'Pause launch');
  }, [isPaused]);
  
  // 生成诊断报告
  const handleGenerateReport = useCallback(() => {
    if (errorHandler) {
      const report = errorHandler.generateDiagnosticReport();
      setShowDiagnosticReport(true);
      
      // 可以保存报告到文件或显示模态框
      import('../utils/logger').then(({ defaultLogger }) => {
        defaultLogger.info('Diagnostic report:', report);
      });
    }
  }, [errorHandler]);
  
  // 获取阶段信息
  const getPhaseInfo = useCallback((phaseIndex: number) => {
    const phases = [
      {
        title: 'Initial Check',
        description: 'Preflight system environment and configuration validation',
        icon: <InfoCircleOutlined />,
        status: 'wait'
      },
      {
        title: 'Dependency Validation',
        description: 'Check Docker environment and dependency status',
        icon: <ExclamationCircleOutlined />,
        status: 'wait'
      },
      {
        title: 'Service Deployment',
        description: 'Start containers and service instances',
        icon: <ClockCircleOutlined />,
        status: 'wait'
      },
      {
        title: 'Readiness Confirmation',
        description: 'Verify service health and access links',
        icon: <CheckCircleOutlined />,
        status: 'wait'
      }
    ];
    
    // 更新状态
    phases[phaseIndex].status = 'process';
    for (let i = 0; i < phaseIndex; i++) {
      phases[i].status = 'finish';
    }
    
    return phases[phaseIndex];
  }, []);

  // 格式化时间
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  // 获取状态颜色
  const getStatusColor = () => {
    switch (status) {
      case 'idle': return 'gray';
      case 'checking': return 'blue';
      case 'starting': return 'orange';
      case 'running': return 'green';
      case 'error': return 'red';
      default: return 'gray';
    }
  };

  // 获取状态文本
  const getStatusText = () => {
    switch (status) {
      case 'idle': return 'Ready to launch';
      case 'checking': return 'Checking environment';
      case 'starting': return 'Starting services';
      case 'running': return 'Running';
      case 'error': return 'Launch failed';
      default: return 'Unknown status';
    }
  };

  return (
    <div className="launch-wizard">
      <div className="wizard-header">
        <h1>CodeInsight Launch Wizard</h1>
        <div className="status-indicator" style={{ backgroundColor: getStatusColor() }}>
          {getStatusText()}
        </div>
      </div>
      
      <div className="progress-section">
        <div className="progress-bar">
          <div 
            className="progress-fill" 
            style={{ width: `${progress}%`, backgroundColor: getStatusColor() }}
          ></div>
        </div>
        <div className="progress-text">{progress}%</div>
      </div>
      
      {status === 'error' && error && (
        <div className="error-section">
          <h3>Error Details</h3>
          <div className="error-message">{error}</div>
          
          <h3>Suggested Solutions</h3>
          <ul className="solutions-list">
            {suggestedSolutions.map((solution, index) => (
              <li key={index}>{solution}</li>
            ))}
          </ul>
        </div>
      )}
      
      <div className="resources-section">
        <h3>System Resource Monitor</h3>
        <div className="resource-item">
          <span className="resource-label">CPU Usage</span>
          <div className="resource-bar">
            <div 
              className="resource-fill cpu" 
              style={{ width: `${systemResources.cpu}%` }}
            ></div>
          </div>
          <span className="resource-value">{systemResources.cpu}%</span>
        </div>
        
        <div className="resource-item">
          <span className="resource-label">Memory Usage</span>
          <div className="resource-bar">
            <div 
              className="resource-fill memory" 
              style={{ width: `${systemResources.memory}%` }}
            ></div>
          </div>
          <span className="resource-value">{systemResources.memory}%</span>
        </div>
        
        <div className="resource-item">
          <span className="resource-label">Disk Usage</span>
          <div className="resource-bar">
            <div 
              className="resource-fill disk" 
              style={{ width: `${systemResources.disk}%` }}
            ></div>
          </div>
          <span className="resource-value">{systemResources.disk}%</span>
        </div>
      </div>
      
      <div className="log-section">
        <div className="log-header">
          <h3>Launch Logs</h3>
          <button 
            className="clear-log-btn"
            onClick={() => setLogs([])}
            disabled={logs.length === 0}
          >
            Clear Logs
          </button>
        </div>
        
        <div className="log-container">
          {logs.length === 0 ? (
            <div className="no-logs">No logs yet</div>
          ) : (
            logs.map(log => (
              <div 
                key={log.id} 
                className={`log-entry ${log.level}`}
              >
                <span className="log-time">[{formatTime(log.timestamp)}]</span>
                <span className={`log-level ${log.level}`}>
                  [{log.level.toUpperCase()}]
                </span>
                <span className="log-message">{log.message}</span>
              </div>
            ))
          )}
        </div>
      </div>
      
      <div className="actions-section">
        <button 
          className="start-btn"
          onClick={startServices}
          disabled={status === 'starting' || status === 'running'}
        >
          {status === 'running' ? 'Services Running' : 'Start Services'}
        </button>
        
        <button 
          className="stop-btn"
          onClick={stopServices}
          disabled={status !== 'running'}
        >
          Stop Services
        </button>
      </div>
      
      {/* Inline styles moved to external CSS at ./styles/LaunchWizard.css */}
    </div>
  );
};

export default LaunchWizard;
