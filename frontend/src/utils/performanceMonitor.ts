import { defaultLogger as logger } from './logger';

// 性能指标类型
export enum PerformanceMetricType {
  NAVIGATION = 'navigation',
  RESOURCE = 'resource',
  PAINT = 'paint',
  FIRST_INPUT = 'first-input',
  LAYOUT_SHIFT = 'layout-shift',
  LONGTASK = 'longtask',
  CUSTOM = 'custom',
}

// 性能指标数据接口
export interface PerformanceMetric {
  type: PerformanceMetricType;
  name: string;
  value: number;
  timestamp: number;
  details?: Record<string, any>;
}

// 性能监控配置接口
export interface PerformanceMonitorConfig {
  sampleRate: number; // 采样率 (0-1)
  reportingInterval: number; // 上报间隔 (ms)
  maxBufferSize: number; // 最大缓冲区大小
  apiEndpoint?: string; // 上报API端点
  enableConsoleLogging: boolean; // 是否启用控制台日志
  enableNavigationTiming: boolean; // 是否启用导航计时
  enableResourceTiming: boolean; // 是否启用资源计时
  enableUserTiming: boolean; // 是否启用用户计时
  enableLongTaskTiming: boolean; // 是否启用长任务计时
  enableLayoutShiftTracking: boolean; // 是否启用布局偏移跟踪
  enableFirstInputTracking: boolean; // 是否启用首次输入跟踪
  enablePaintTiming: boolean; // 是否启用绘制计时
}

// 默认配置
const DEFAULT_CONFIG: PerformanceMonitorConfig = {
  sampleRate: 1.0,
  reportingInterval: 10000,
  maxBufferSize: 100,
  enableConsoleLogging: true,
  enableNavigationTiming: true,
  enableResourceTiming: true,
  enableUserTiming: true,
  enableLongTaskTiming: true,
  enableLayoutShiftTracking: true,
  enableFirstInputTracking: true,
  enablePaintTiming: true,
};

/**
 * 性能监控管理器
 * 负责收集、处理和上报性能指标数据
 */
export class PerformanceMonitor {
  private static instance: PerformanceMonitor;
  private config: PerformanceMonitorConfig;
  private metricsBuffer: PerformanceMetric[] = [];
  private reportingTimer: number | null = null;
  private isInitialized = false;
  private cumulativeLayoutShift = 0;
  private resourceObserver: PerformanceObserver | null = null;
  private paintObserver: PerformanceObserver | null = null;
  private firstInputObserver: PerformanceObserver | null = null;
  private layoutShiftObserver: PerformanceObserver | null = null;
  private longTaskObserver: PerformanceObserver | null = null;
  private userTimingObserver: PerformanceObserver | null = null;
  private sessionId: string;
  private pageLoadStart: number;

  private constructor(config: Partial<PerformanceMonitorConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.sessionId = this.generateSessionId();
    this.pageLoadStart = performance.now();
  }

  /**
   * 获取性能监控实例（单例模式）
   */
  public static getInstance(config?: Partial<PerformanceMonitorConfig>): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor(config);
    } else if (config) {
      PerformanceMonitor.instance.updateConfig(config);
    }
    return PerformanceMonitor.instance;
  }

  /**
   * 更新配置
   */
  public updateConfig(config: Partial<PerformanceMonitorConfig>): void {
    this.config = { ...this.config, ...config };
    if (this.isInitialized) {
      this.stop();
      this.start();
    }
  }

  /**
   * 初始化并开始监控
   */
  public start(): void {
    if (!this.isSupported()) {
      logger.warn('Performance API not fully supported in this browser');
      return;
    }

    if (this.isInitialized) {
      return;
    }

    try {
      // 注册各种性能观察器
      this.registerObservers();

      // 收集导航计时指标
      if (this.config.enableNavigationTiming) {
        this.collectNavigationTiming();
      }

      // 启动定期上报
      this.startReporting();

      // 注册页面卸载事件，确保在页面关闭前上报数据
      window.addEventListener('beforeunload', this.onBeforeUnload);

      this.isInitialized = true;
      logger.info('Performance monitoring started');
    } catch (error) {
      logger.error('Failed to start performance monitoring', error);
    }
  }

  /**
   * 停止监控
   */
  public stop(): void {
    if (!this.isInitialized) {
      return;
    }

    // 停止所有观察器
    this.disconnectObservers();

    // 停止定期上报
    if (this.reportingTimer !== null) {
      window.clearInterval(this.reportingTimer);
      this.reportingTimer = null;
    }

    // 移除页面卸载事件监听
    window.removeEventListener('beforeunload', this.onBeforeUnload);

    // 上报剩余数据
    this.reportMetrics();

    this.isInitialized = false;
    logger.info('Performance monitoring stopped');
  }

  /**
   * 手动记录自定义性能指标
   */
  public recordCustomMetric(name: string, value: number, details?: Record<string, any>): void {
    if (!this.shouldSample()) {
      return;
    }

    const metric: PerformanceMetric = {
      type: PerformanceMetricType.CUSTOM,
      name,
      value,
      timestamp: performance.now(),
      details,
    };

    this.addMetric(metric);
    logger.debug(`Custom metric recorded: ${name} = ${value}`);
  }

  /**
   * 开始计时自定义操作
   */
  public startMeasure(name: string): void {
    if (!this.isSupported() || !this.isInitialized) {
      return;
    }

    try {
      performance.mark(`${name}_start`);
    } catch (error) {
      logger.error(`Failed to start measure: ${name}`, error);
    }
  }

  /**
   * 结束计时自定义操作并记录
   */
  public endMeasure(name: string, details?: Record<string, any>): void {
    if (!this.isSupported() || !this.isInitialized) {
      return;
    }

    try {
      performance.mark(`${name}_end`);
      performance.measure(name, `${name}_start`, `${name}_end`);
      
      const entries = performance.getEntriesByName(name, 'measure');
      if (entries.length > 0) {
        const duration = entries[0].duration;
        this.recordCustomMetric(name, duration, {
          ...details,
          measureType: 'duration',
        });
      }
      
      // 清理标记
      performance.clearMarks(`${name}_start`);
      performance.clearMarks(`${name}_end`);
      performance.clearMeasures(name);
    } catch (error) {
      logger.error(`Failed to end measure: ${name}`, error);
    }
  }

  /**
   * 获取当前收集的所有指标
   */
  public getMetrics(): PerformanceMetric[] {
    return [...this.metricsBuffer];
  }

  /**
   * 清除所有收集的指标
   */
  public clearMetrics(): void {
    this.metricsBuffer = [];
  }

  /**
   * 获取关键性能指标摘要
   */
  public getPerformanceSummary(): Record<string, any> {
    const summary: Record<string, any> = {
      sessionId: this.sessionId,
      timestamp: Date.now(),
      pageLoadTime: this.getMetricValue('page_load'),
      firstContentfulPaint: this.getMetricValue('first-contentful-paint'),
      largestContentfulPaint: this.getMetricValue('largest-contentful-paint'),
      firstInputDelay: this.getMetricValue('first-input-delay'),
      cumulativeLayoutShift: this.cumulativeLayoutShift,
      resourceCount: this.countMetricsByType(PerformanceMetricType.RESOURCE),
      longTaskCount: this.countMetricsByType(PerformanceMetricType.LONGTASK),
    };

    return summary;
  }

  // 私有方法

  /**
   * 检查浏览器是否支持性能API
   */
  private isSupported(): boolean {
    return typeof window !== 'undefined' && 
           typeof performance !== 'undefined' && 
           typeof PerformanceObserver !== 'undefined';
  }

  /**
   * 根据采样率决定是否应该采样
   */
  private shouldSample(): boolean {
    return Math.random() < this.config.sampleRate;
  }

  /**
   * 生成会话ID
   */
  private generateSessionId(): string {
    return `${Date.now()}-${Math.random().toString(36).substring(2, 15)}`;
  }

  /**
   * 注册所有性能观察器
   */
  private registerObservers(): void {
    // 资源计时观察器
    if (this.config.enableResourceTiming) {
      try {
        this.resourceObserver = new PerformanceObserver((entryList) => {
          const entries = entryList.getEntries();
          for (const entry of entries) {
            if (entry.entryType === 'resource') {
              this.processResourceTiming(entry as PerformanceResourceTiming);
            }
          }
        });
        this.resourceObserver.observe({ entryTypes: ['resource'] });
      } catch (e) {
        logger.warn('Resource timing observer not supported', e);
      }
    }

    // 绘制计时观察器
    if (this.config.enablePaintTiming) {
      try {
        this.paintObserver = new PerformanceObserver((entryList) => {
          const entries = entryList.getEntries();
          for (const entry of entries) {
            this.processPaintTiming(entry);
          }
        });
        this.paintObserver.observe({ entryTypes: ['paint', 'largest-contentful-paint'] });
      } catch (e) {
        logger.warn('Paint timing observer not supported', e);
      }
    }

    // 首次输入延迟观察器
    if (this.config.enableFirstInputTracking) {
      try {
        this.firstInputObserver = new PerformanceObserver((entryList) => {
          const entries = entryList.getEntries();
          for (const entry of entries) {
            this.processFirstInput(entry);
          }
          // 首次输入只需要观察一次
          this.firstInputObserver?.disconnect();
        });
        this.firstInputObserver.observe({ type: 'first-input', buffered: true });
      } catch (e) {
        logger.warn('First input observer not supported', e);
      }
    }

    // 布局偏移观察器
    if (this.config.enableLayoutShiftTracking) {
      try {
        this.layoutShiftObserver = new PerformanceObserver((entryList) => {
          const entries = entryList.getEntries();
          for (const entry of entries) {
            if (entry.entryType === 'layout-shift' && !(entry as any).hadRecentInput) {
              const shift = (entry as any).value;
              this.cumulativeLayoutShift += shift;
              this.processLayoutShift(entry, shift);
            }
          }
        });
        this.layoutShiftObserver.observe({ type: 'layout-shift', buffered: true });
      } catch (e) {
        logger.warn('Layout shift observer not supported', e);
      }
    }

    // 长任务观察器
    if (this.config.enableLongTaskTiming) {
      try {
        this.longTaskObserver = new PerformanceObserver((entryList) => {
          const entries = entryList.getEntries();
          for (const entry of entries) {
            this.processLongTask(entry);
          }
        });
        this.longTaskObserver.observe({ entryTypes: ['longtask'] });
      } catch (e) {
        logger.warn('Long task observer not supported', e);
      }
    }

    // 用户计时观察器
    if (this.config.enableUserTiming) {
      try {
        this.userTimingObserver = new PerformanceObserver((entryList) => {
          const entries = entryList.getEntries();
          for (const entry of entries) {
            if (entry.entryType === 'measure') {
              this.processUserTiming(entry);
            }
          }
        });
        this.userTimingObserver.observe({ entryTypes: ['measure'] });
      } catch (e) {
        logger.warn('User timing observer not supported', e);
      }
    }
  }

  /**
   * 断开所有观察器连接
   */
  private disconnectObservers(): void {
    this.resourceObserver?.disconnect();
    this.paintObserver?.disconnect();
    this.firstInputObserver?.disconnect();
    this.layoutShiftObserver?.disconnect();
    this.longTaskObserver?.disconnect();
    this.userTimingObserver?.disconnect();
  }

  /**
   * 收集导航计时指标
   */
  private collectNavigationTiming(): void {
    if (!this.shouldSample()) {
      return;
    }

    try {
      // 等待导航完成
      setTimeout(() => {
        const navEntry = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
        
        if (navEntry) {
          // 页面加载时间 (loadEventEnd - navigationStart)
          this.addMetric({
            type: PerformanceMetricType.NAVIGATION,
            name: 'page_load',
            value: navEntry.loadEventEnd,
            timestamp: performance.now(),
            details: {
              domContentLoaded: navEntry.domContentLoadedEventEnd,
              domInteractive: navEntry.domInteractive,
              loadEvent: navEntry.loadEventEnd,
              unloadEvent: navEntry.unloadEventEnd,
              redirectCount: navEntry.redirectCount,
              type: navEntry.type,
            },
          });

          // DNS解析时间
          this.addMetric({
            type: PerformanceMetricType.NAVIGATION,
            name: 'dns',
            value: navEntry.domainLookupEnd - navEntry.domainLookupStart,
            timestamp: performance.now(),
          });

          // TCP连接时间
          this.addMetric({
            type: PerformanceMetricType.NAVIGATION,
            name: 'tcp',
            value: navEntry.connectEnd - navEntry.connectStart,
            timestamp: performance.now(),
          });

          // 请求响应时间
          this.addMetric({
            type: PerformanceMetricType.NAVIGATION,
            name: 'request',
            value: navEntry.responseEnd - navEntry.requestStart,
            timestamp: performance.now(),
          });

          // DOM处理时间
          this.addMetric({
            type: PerformanceMetricType.NAVIGATION,
            name: 'dom_processing',
            value: navEntry.domComplete - navEntry.domInteractive,
            timestamp: performance.now(),
          });
        }
      }, 0);
    } catch (error) {
      logger.error('Failed to collect navigation timing', error);
    }
  }

  /**
   * 处理资源计时条目
   */
  private processResourceTiming(entry: PerformanceResourceTiming): void {
    if (!this.shouldSample()) {
      return;
    }

    // 过滤掉一些不必要的资源，如分析脚本等
    if (entry.name.includes('analytics') || entry.name.includes('tracking')) {
      return;
    }

    const urlParts = entry.name.split('/');
    const fileName = urlParts[urlParts.length - 1].split('?')[0];
    const fileType = fileName.split('.').pop() || 'unknown';

    this.addMetric({
      type: PerformanceMetricType.RESOURCE,
      name: `resource_${fileType}`,
      value: entry.responseEnd - entry.startTime,
      timestamp: performance.now(),
      details: {
        url: entry.name,
        initiatorType: entry.initiatorType,
        size: entry.transferSize,
        encodedSize: entry.encodedBodySize,
        decodedSize: entry.decodedBodySize,
        dns: entry.domainLookupEnd - entry.domainLookupStart,
        tcp: entry.connectEnd - entry.connectStart,
        request: entry.responseStart - entry.requestStart,
        response: entry.responseEnd - entry.responseStart,
      },
    });
  }

  /**
   * 处理绘制计时条目
   */
  private processPaintTiming(entry: PerformanceEntry): void {
    if (!this.shouldSample()) {
      return;
    }

    this.addMetric({
      type: PerformanceMetricType.PAINT,
      name: entry.name,
      value: entry.startTime,
      timestamp: performance.now(),
    });
  }

  /**
   * 处理首次输入延迟条目
   */
  private processFirstInput(entry: PerformanceEntry): void {
    if (!this.shouldSample()) {
      return;
    }

    // 首次输入延迟 = 处理延迟
    const firstInputDelay = (entry as any).processingStart - entry.startTime;

    this.addMetric({
      type: PerformanceMetricType.FIRST_INPUT,
      name: 'first-input-delay',
      value: firstInputDelay,
      timestamp: performance.now(),
      details: {
        target: (entry as any).target?.tagName || 'unknown',
        startTime: entry.startTime,
        processingStart: (entry as any).processingStart,
        processingEnd: (entry as any).processingEnd,
      },
    });
  }

  /**
   * 处理布局偏移条目
   */
  private processLayoutShift(entry: PerformanceEntry, value: number): void {
    if (!this.shouldSample()) {
      return;
    }

    this.addMetric({
      type: PerformanceMetricType.LAYOUT_SHIFT,
      name: 'layout-shift',
      value: value,
      timestamp: performance.now(),
      details: {
        cumulativeLayoutShift: this.cumulativeLayoutShift,
      },
    });
  }

  /**
   * 处理长任务条目
   */
  private processLongTask(entry: PerformanceEntry): void {
    if (!this.shouldSample()) {
      return;
    }

    this.addMetric({
      type: PerformanceMetricType.LONGTASK,
      name: 'long-task',
      value: entry.duration,
      timestamp: performance.now(),
      details: {
        startTime: entry.startTime,
        attribution: (entry as any).attribution?.[0]?.name || 'unknown',
      },
    });
  }

  /**
   * 处理用户计时条目
   */
  private processUserTiming(entry: PerformanceEntry): void {
    if (!this.shouldSample()) {
      return;
    }

    this.addMetric({
      type: PerformanceMetricType.CUSTOM,
      name: entry.name,
      value: entry.duration,
      timestamp: performance.now(),
      details: {
        startTime: entry.startTime,
        duration: entry.duration,
      },
    });
  }

  /**
   * 添加指标到缓冲区
   */
  private addMetric(metric: PerformanceMetric): void {
    this.metricsBuffer.push(metric);

    // 如果启用了控制台日志，输出指标信息
    if (this.config.enableConsoleLogging) {
      logger.debug(`Performance metric: ${metric.type} - ${metric.name} = ${metric.value.toFixed(2)}`);
    }

    // 如果缓冲区已满，立即上报
    if (this.metricsBuffer.length >= this.config.maxBufferSize) {
      this.reportMetrics();
    }
  }

  /**
   * 启动定期上报
   */
  private startReporting(): void {
    if (this.reportingTimer !== null) {
      window.clearInterval(this.reportingTimer);
    }

    this.reportingTimer = window.setInterval(() => {
      this.reportMetrics();
    }, this.config.reportingInterval);
  }

  /**
   * 上报指标数据
   */
  private reportMetrics(): void {
    if (this.metricsBuffer.length === 0) {
      return;
    }

    const metrics = [...this.metricsBuffer];
    this.metricsBuffer = [];

    // 如果配置了API端点，发送数据到服务器
    if (this.config.apiEndpoint) {
      this.sendToServer(metrics);
    }

    // 记录性能摘要
    if (this.config.enableConsoleLogging) {
      logger.info('Performance summary:', this.getPerformanceSummary());
    }
  }

  /**
   * 发送指标数据到服务器
   */
  private sendToServer(metrics: PerformanceMetric[]): void {
    try {
      const payload = {
        sessionId: this.sessionId,
        timestamp: Date.now(),
        userAgent: navigator.userAgent,
        metrics: metrics,
      };

      fetch(this.config.apiEndpoint!, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
        // 使用keepalive确保在页面卸载时数据仍能发送
        keepalive: true,
      }).catch(error => {
        logger.error('Failed to send performance metrics to server', error);
      });
    } catch (error) {
      logger.error('Error preparing or sending metrics', error);
    }
  }

  /**
   * 页面卸载前处理
   */
  private onBeforeUnload = (): void => {
    // 确保在页面卸载前上报所有数据
    this.reportMetrics();
  };

  /**
   * 获取特定指标的值
   */
  private getMetricValue(name: string): number | undefined {
    const metric = this.metricsBuffer.find(m => m.name === name);
    return metric?.value;
  }

  /**
   * 计算特定类型指标的数量
   */
  private countMetricsByType(type: PerformanceMetricType): number {
    return this.metricsBuffer.filter(m => m.type === type).length;
  }
}

// 创建性能监控实例
export const performanceMonitor = PerformanceMonitor.getInstance();

// 性能监控工具函数
export const performance_utils = {
  /**
   * 开始测量操作耗时
   */
  startMeasure: (name: string): void => {
    performanceMonitor.startMeasure(name);
  },

  /**
   * 结束测量操作耗时
   */
  endMeasure: (name: string, details?: Record<string, any>): void => {
    performanceMonitor.endMeasure(name, details);
  },

  /**
   * 记录自定义性能指标
   */
  recordMetric: (name: string, value: number, details?: Record<string, any>): void => {
    performanceMonitor.recordCustomMetric(name, value, details);
  },

  /**
   * 获取性能摘要
   */
  getSummary: (): Record<string, any> => {
    return performanceMonitor.getPerformanceSummary();
  },

  /**
   * 测量函数执行时间的装饰器
   */
  measurePerformance: (target: any, propertyKey: string, descriptor: PropertyDescriptor) => {
    const originalMethod = descriptor.value;

    descriptor.value = function(...args: any[]) {
      const methodName = `${target.constructor.name}.${propertyKey}`;
      performanceMonitor.startMeasure(methodName);
      
      try {
        const result = originalMethod.apply(this, args);
        
        // 处理Promise返回值
        if (result instanceof Promise) {
          return result.finally(() => {
            performanceMonitor.endMeasure(methodName);
          });
        }
        
        performanceMonitor.endMeasure(methodName);
        return result;
      } catch (error) {
        performanceMonitor.endMeasure(methodName, { error: true });
        throw error;
      }
    };

    return descriptor;
  },
};

// 初始化性能监控
export function initPerformanceMonitoring(config?: Partial<PerformanceMonitorConfig>): void {
  const monitor = PerformanceMonitor.getInstance(config);
  monitor.start();
  logger.info('Performance monitoring initialized');
}