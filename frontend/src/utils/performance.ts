import { defaultLogger as logger } from './logger';

/**
 * 性能指标类型
 */
export enum MetricType {
  // 核心性能指标
  PAGE_LOAD = 'page_load',
  API_RESPONSE = 'api_response',
  RESOURCE_LOAD = 'resource_load',
  
  // 用户体验指标
  FIRST_CONTENTFUL_PAINT = 'first_contentful_paint',
  FIRST_INPUT_DELAY = 'first_input_delay',
  CUMULATIVE_LAYOUT_SHIFT = 'cumulative_layout_shift',
  LARGEST_CONTENTFUL_PAINT = 'largest_contentful_paint',
  
  // 自定义指标
  CUSTOM = 'custom',
}

/**
 * 性能指标数据接口
 */
export interface PerformanceMetric {
  type: MetricType;
  value: number;
  name: string;
  timestamp: number;
  tags?: Record<string, string>;
}

/**
 * 性能监控配置
 */
export interface PerformanceMonitorConfig {
  sampleRate: number; // 采样率 (0-1)
  reportingEndpoint?: string; // 上报端点
  reportingInterval: number; // 上报间隔 (ms)
  bufferSize: number; // 缓冲区大小
  enableConsoleLogging: boolean; // 是否启用控制台日志
}

/**
 * 性能监控服务
 */
export class PerformanceMonitor {
  private static instance: PerformanceMonitor;
  private config: PerformanceMonitorConfig;
  private metricsBuffer: PerformanceMetric[] = [];
  private reportingTimer: number | null = null;
  private isMonitoring = false;
  private observedResources: Set<string> = new Set();
  private perfObserver: PerformanceObserver | null = null;

  /**
   * 获取单例实例
   */
  public static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor();
    }
    return PerformanceMonitor.instance;
  }

  /**
   * 构造函数
   */
  private constructor() {
    this.config = {
      sampleRate: 0.1, // 默认采样率10%
      reportingInterval: 10000, // 默认10秒上报一次
      bufferSize: 100, // 默认缓冲区大小
      enableConsoleLogging: false, // 默认不启用控制台日志
    };
  }

  /**
   * 初始化性能监控
   */
  public init(config: Partial<PerformanceMonitorConfig> = {}): void {
    this.config = { ...this.config, ...config };
    
    if (this.shouldSample()) {
      this.startMonitoring();
      logger.info('Performance monitoring initialized', { sampleRate: this.config.sampleRate });
    } else {
      logger.debug('Performance monitoring skipped due to sampling');
    }
  }

  /**
   * 开始监控
   */
  private startMonitoring(): void {
    if (this.isMonitoring) return;
    
    this.isMonitoring = true;
    
    // 监控页面加载性能
    this.capturePageLoadMetrics();
    
    // 监控Web Vitals
    this.captureWebVitals();
    
    // 监控资源加载性能
    this.observeResourceTiming();
    
    // 监控API请求性能
    this.monitorApiPerformance();
    
    // 启动定时上报
    this.startReporting();
    
    // 页面卸载前发送剩余指标
    window.addEventListener('beforeunload', () => this.flush());
  }

  /**
   * 停止监控
   */
  public stopMonitoring(): void {
    if (!this.isMonitoring) return;
    
    this.isMonitoring = false;
    
    if (this.perfObserver) {
      this.perfObserver.disconnect();
      this.perfObserver = null;
    }
    
    if (this.reportingTimer !== null) {
      window.clearInterval(this.reportingTimer);
      this.reportingTimer = null;
    }
    
    window.removeEventListener('beforeunload', () => this.flush());
    
    logger.info('Performance monitoring stopped');
  }

  /**
   * 是否应该采样
   */
  private shouldSample(): boolean {
    return Math.random() < this.config.sampleRate;
  }

  /**
   * 捕获页面加载性能指标
   */
  private capturePageLoadMetrics(): void {
    window.addEventListener('load', () => {
      setTimeout(() => {
        if (performance && performance.timing) {
          const timing = performance.timing;
          
          // 计算关键性能指标
          const pageLoadTime = timing.loadEventEnd - timing.navigationStart;
          const dnsTime = timing.domainLookupEnd - timing.domainLookupStart;
          const tcpTime = timing.connectEnd - timing.connectStart;
          const ttfb = timing.responseStart - timing.requestStart;
          const domContentLoaded = timing.domContentLoadedEventEnd - timing.navigationStart;
          const domInteractive = timing.domInteractive - timing.navigationStart;
          
          // 记录页面加载性能指标
          this.recordMetric({
            type: MetricType.PAGE_LOAD,
            name: 'page_load_time',
            value: pageLoadTime,
            timestamp: Date.now(),
            tags: {
              page: window.location.pathname,
              dns_time: String(dnsTime),
              tcp_time: String(tcpTime),
              ttfb: String(ttfb),
              dom_content_loaded: String(domContentLoaded),
              dom_interactive: String(domInteractive)
            }
          });
        }
      }, 0);
    });
  }

  /**
   * 捕获Web Vitals指标
   */
  private captureWebVitals(): void {
    // 首次内容绘制 (FCP)
    this.observePerformanceEntry('paint', (entries) => {
      for (const entry of entries) {
        if (entry.name === 'first-contentful-paint') {
          this.recordMetric({
            type: MetricType.FIRST_CONTENTFUL_PAINT,
            name: 'first_contentful_paint',
            value: entry.startTime,
            timestamp: Date.now(),
            tags: { page: window.location.pathname }
          });
        }
      }
    });
    
    // 最大内容绘制 (LCP)
    this.observePerformanceEntry('largest-contentful-paint', (entries) => {
      // 只记录最后一个LCP
      const lastEntry = entries[entries.length - 1];
      this.recordMetric({
        type: MetricType.LARGEST_CONTENTFUL_PAINT,
        name: 'largest_contentful_paint',
        value: lastEntry.startTime,
        timestamp: Date.now(),
        tags: { page: window.location.pathname }
      });
    });
    
    // 累积布局偏移 (CLS)
    let clsValue = 0;
    let clsEntries: PerformanceEntry[] = [];
    
    this.observePerformanceEntry('layout-shift', (entries) => {
      for (const entry of entries) {
        // 只计算没有用户输入的布局偏移
        if (!(entry as any).hadRecentInput) {
          clsValue += (entry as any).value;
          clsEntries.push(entry);
        }
      }
      
      // 定期记录CLS
      if (clsEntries.length > 0) {
        this.recordMetric({
          type: MetricType.CUMULATIVE_LAYOUT_SHIFT,
          name: 'cumulative_layout_shift',
          value: clsValue,
          timestamp: Date.now(),
          tags: { page: window.location.pathname }
        });
      }
    });
    
    // 首次输入延迟 (FID)
    this.observePerformanceEntry('first-input', (entries) => {
      for (const entry of entries) {
        const entryAny = entry as any;
        const processingStart = typeof entryAny?.processingStart === 'number'
          ? entryAny.processingStart
          : entry.startTime;
        const delay = processingStart - entry.startTime;
        this.recordMetric({
          type: MetricType.FIRST_INPUT_DELAY,
          name: 'first_input_delay',
          value: delay,
          timestamp: Date.now(),
          tags: { page: window.location.pathname }
        });
      }
    });
  }

  /**
   * 观察性能条目
   */
  private observePerformanceEntry(entryType: string, callback: (entries: PerformanceEntry[]) => void): void {
    if (!window.PerformanceObserver) return;
    
    try {
      const observer = new PerformanceObserver((list) => {
        callback(list.getEntries());
      });
      
      observer.observe({ type: entryType, buffered: true });
    } catch (e) {
      logger.error(`Failed to observe ${entryType} entries`, { error: String(e) });
    }
  }

  /**
   * 观察资源加载性能
   */
  private observeResourceTiming(): void {
    if (!window.PerformanceObserver) return;
    
    try {
      this.perfObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        
        for (const entry of entries) {
          if (entry.entryType === 'resource' && !this.observedResources.has(entry.name)) {
            this.observedResources.add(entry.name);
            
            const resourceEntry = entry as PerformanceResourceTiming;
            const resourceType = this.getResourceType(resourceEntry.name);
            
            this.recordMetric({
              type: MetricType.RESOURCE_LOAD,
              name: `resource_${resourceType}`,
              value: resourceEntry.duration,
              timestamp: Date.now(),
              tags: {
                url: resourceEntry.name,
                type: resourceType,
                size: resourceEntry.transferSize ? String(resourceEntry.transferSize) : 'unknown',
                protocol: resourceEntry.nextHopProtocol || 'unknown'
              }
            });
          }
        }
      });
      
      this.perfObserver.observe({ entryTypes: ['resource'] });
    } catch (e) {
      logger.error('Failed to observe resource timing', { error: String(e) });
    }
  }

  /**
   * 获取资源类型
   */
  private getResourceType(url: string): string {
    const extension = url.split('.').pop()?.toLowerCase() || '';
    
    if (/jpe?g|png|gif|svg|webp|ico/.test(extension)) return 'image';
    if (/js/.test(extension)) return 'script';
    if (/css/.test(extension)) return 'stylesheet';
    if (/woff2?|ttf|otf|eot/.test(extension)) return 'font';
    if (/json/.test(extension)) return 'json';
    if (/xml/.test(extension)) return 'xml';
    
    // 检查URL路径
    if (url.includes('/api/')) return 'api';
    
    return 'other';
  }

  /**
   * 监控API请求性能
   */
  private monitorApiPerformance(): void {
    if (!window.XMLHttpRequest || !window.fetch) return;
    
    // 拦截XMLHttpRequest
    const originalXhrOpen = XMLHttpRequest.prototype.open;
    const originalXhrSend = XMLHttpRequest.prototype.send;
    
    XMLHttpRequest.prototype.open = function(method: string, url: string) {
      (this as any)._perfMonitorData = {
        method,
        url,
        startTime: 0
      };
      return originalXhrOpen.apply(this, arguments as any);
    };
    
    XMLHttpRequest.prototype.send = function() {
      const xhr = this;
      const perfData = (xhr as any)._perfMonitorData;
      
      if (perfData && perfData.url.includes('/api/')) {
        perfData.startTime = performance.now();
        
        xhr.addEventListener('loadend', function() {
          const endTime = performance.now();
          const duration = endTime - perfData.startTime;
          
          PerformanceMonitor.getInstance().recordApiPerformance(
            perfData.url,
            perfData.method,
            duration,
            xhr.status
          );
        });
      }
      
      return originalXhrSend.apply(this, arguments as any);
    };
    
    // 拦截Fetch API
    const originalFetch: typeof window.fetch = window.fetch;
    const patchedFetch: typeof window.fetch = function(input: RequestInfo | URL, init?: RequestInit) {
      const url = typeof input === 'string' ? input : (input instanceof Request ? input.url : String(input));
      const method = init?.method || 'GET';
      
      if (url.includes('/api/')) {
        const startTime = performance.now();
        const promise = originalFetch.apply(window, [input as any, init]);
        return promise.then((response) => {
          const endTime = performance.now();
          const duration = endTime - startTime;
          
          PerformanceMonitor.getInstance().recordApiPerformance(
            url,
            method,
            duration,
            response.status
          );
          
          return response;
        }).catch((error) => {
          const endTime = performance.now();
          const duration = endTime - startTime;
          
          PerformanceMonitor.getInstance().recordApiPerformance(
            url,
            method,
            duration,
            0,
            String(error)
          );
          
          throw error;
        });
      }
      
      return originalFetch.apply(window, [input as any, init] as any);
    };
    window.fetch = patchedFetch;
  }

  /**
   * 记录API性能
   */
  public recordApiPerformance(
    url: string,
    method: string,
    duration: number,
    statusCode: number,
    errorMessage?: string
  ): void {
    // 提取API路径，移除查询参数和域名
    const apiPath = new URL(url, window.location.origin).pathname;
    
    this.recordMetric({
      type: MetricType.API_RESPONSE,
      name: 'api_response_time',
      value: duration,
      timestamp: Date.now(),
      tags: {
        path: apiPath,
        method,
        status: String(statusCode),
        error: errorMessage || '',
        success: String(statusCode >= 200 && statusCode < 300)
      }
    });
  }

  /**
   * 记录自定义性能指标
   */
  public recordCustomMetric(
    name: string,
    value: number,
    tags: Record<string, string> = {}
  ): void {
    this.recordMetric({
      type: MetricType.CUSTOM,
      name,
      value,
      timestamp: Date.now(),
      tags: { ...tags, page: window.location.pathname }
    });
  }

  /**
   * 记录性能指标
   */
  private recordMetric(metric: PerformanceMetric): void {
    if (!this.isMonitoring) return;
    
    // 添加到缓冲区
    this.metricsBuffer.push(metric);
    
    // 记录到控制台
    if (this.config.enableConsoleLogging) {
      logger.debug('Performance metric recorded', metric);
    }
    
    // 如果缓冲区已满，立即上报
    if (this.metricsBuffer.length >= this.config.bufferSize) {
      this.flush();
    }
  }

  /**
   * 开始定时上报
   */
  private startReporting(): void {
    if (this.reportingTimer !== null) {
      window.clearInterval(this.reportingTimer);
    }
    
    this.reportingTimer = window.setInterval(() => {
      this.flush();
    }, this.config.reportingInterval);
  }

  /**
   * 立即上报所有指标
   */
  public flush(): void {
    if (this.metricsBuffer.length === 0) return;
    
    const metrics = [...this.metricsBuffer];
    this.metricsBuffer = [];
    
    if (this.config.reportingEndpoint) {
      this.sendMetricsToServer(metrics);
    }
    
    logger.info(`Flushed ${metrics.length} performance metrics`);
  }

  /**
   * 发送指标到服务器
   */
  private sendMetricsToServer(metrics: PerformanceMetric[]): void {
    if (!this.config.reportingEndpoint) return;
    
    const payload = {
      metrics,
      timestamp: Date.now(),
      userAgent: navigator.userAgent,
      url: window.location.href
    };
    
    fetch(this.config.reportingEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload),
      // 使用keepalive确保页面卸载时数据也能发送
      keepalive: true
    }).catch((error) => {
      logger.error('Failed to send performance metrics', { error: String(error) });
    });
  }

  /**
   * 测量函数执行时间的装饰器工厂
   */
  public static measure(operationName: string, tags: Record<string, string> = {}) {
    return function(
      target: any,
      propertyKey: string,
      descriptor: PropertyDescriptor
    ) {
      const originalMethod = descriptor.value;
      
      descriptor.value = function(...args: any[]) {
        const startTime = performance.now();
        
        try {
          const result = originalMethod.apply(this, args);
          
          // 处理Promise返回值
          if (result instanceof Promise) {
            return result.finally(() => {
              const endTime = performance.now();
              PerformanceMonitor.getInstance().recordCustomMetric(
                operationName,
                endTime - startTime,
                tags
              );
            });
          }
          
          // 处理同步返回值
          const endTime = performance.now();
          PerformanceMonitor.getInstance().recordCustomMetric(
            operationName,
            endTime - startTime,
            tags
          );
          
          return result;
        } catch (error) {
          const endTime = performance.now();
          PerformanceMonitor.getInstance().recordCustomMetric(
            operationName,
            endTime - startTime,
            { ...tags, error: 'true', errorMessage: String(error) }
          );
          
          throw error;
        }
      };
      
      return descriptor;
    };
  }
}

// 导出单例实例
export const performanceMonitor = PerformanceMonitor.getInstance();

/**
 * 测量函数执行时间的工具函数
 */
export async function measureExecutionTime<T>(
  fn: () => Promise<T> | T,
  operationName: string,
  tags: Record<string, string> = {}
): Promise<T> {
  const startTime = performance.now();
  
  try {
    const result = await fn();
    const endTime = performance.now();
    
    performanceMonitor.recordCustomMetric(
      operationName,
      endTime - startTime,
      tags
    );
    
    return result;
  } catch (error) {
    const endTime = performance.now();
    
    performanceMonitor.recordCustomMetric(
      operationName,
      endTime - startTime,
      { ...tags, error: 'true', errorMessage: String(error) }
    );
    
    throw error;
  }
}

// 初始化性能监控
// 注意：应在应用启动时调用 performanceMonitor.init()