import axios, { AxiosError, AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { errorHandlerManager, ErrorType, ErrorSeverity } from './errorHandling';

// 重试配置接口
interface RetryConfig {
  maxRetries: number;
  retryDelay: number;
  retryStatusCodes: number[];
}

// 默认重试配置
const defaultRetryConfig: RetryConfig = {
  maxRetries: 3,
  retryDelay: 1000,
  retryStatusCodes: [408, 429, 500, 502, 503, 504]
};

// 超时配置
const TIMEOUT_MS = 30000;

/**
 * 创建带错误处理和重试机制的API客户端
 */
export function createApiClient(baseURL: string, config?: {
  timeout?: number;
  retryConfig?: Partial<RetryConfig>;
  authToken?: string;
}): AxiosInstance {
  // 创建axios实例
  const instance = axios.create({
    baseURL,
    timeout: config?.timeout || TIMEOUT_MS,
    headers: {
      'Content-Type': 'application/json',
      ...(config?.authToken ? { 'Authorization': `Bearer ${config.authToken}` } : {})
    }
  });

  // 合并重试配置
  const retryConfig: RetryConfig = {
    ...defaultRetryConfig,
    ...config?.retryConfig
  };

  // 请求拦截器
  instance.interceptors.request.use(
    (config) => {
      // 添加请求ID用于跟踪
      config.headers = config.headers || {};
      config.headers['X-Request-ID'] = `req-${Date.now()}-${Math.random().toString(36).substring(2, 10)}`;
      
      // 添加重试计数
      config.retryCount = config.retryCount || 0;
      
      return config;
    },
    (error) => {
      return Promise.reject(error);
    }
  );

  // 响应拦截器
  instance.interceptors.response.use(
    (response) => {
      return response;
    },
    async (error: AxiosError) => {
      const config = error.config as AxiosRequestConfig & { retryCount?: number };
      
      // 如果没有配置或已达到最大重试次数，则拒绝
      if (!config || !config.retryCount || config.retryCount >= retryConfig.maxRetries) {
        // 记录错误
        logApiError(error);
        return Promise.reject(error);
      }

      // 检查是否应该重试
      const status = error.response?.status;
      if (status && retryConfig.retryStatusCodes.includes(status)) {
        // 增加重试计数
        config.retryCount += 1;
        
        // 计算延迟时间（指数退避）
        const delay = retryConfig.retryDelay * Math.pow(2, config.retryCount - 1);
        
        // 等待后重试
        await new Promise(resolve => setTimeout(resolve, delay));
        
        // 重试请求
        return instance(config);
      }

      // 不符合重试条件，记录错误并拒绝
      logApiError(error);
      return Promise.reject(error);
    }
  );

  return instance;
}

/**
 * 记录API错误
 */
function logApiError(error: AxiosError): void {
  const response = error.response;
  const request = error.request;
  const config = error.config;
  
  let errorMessage = 'API请求失败';
  let errorDetails: any = { message: error.message };
  let errorSeverity = ErrorSeverity.MEDIUM;
  
  // 根据错误类型构建详细信息
  if (response) {
    // 服务器响应了错误状态码
    errorMessage = `API响应错误: ${response.status}`;
    errorDetails = {
      status: response.status,
      statusText: response.statusText,
      data: response.data,
      headers: response.headers,
      url: config?.url,
      method: config?.method
    };
    
    // 根据状态码设置严重性
    if (response.status >= 500) {
      errorSeverity = ErrorSeverity.HIGH;
    } else if (response.status === 401 || response.status === 403) {
      errorSeverity = ErrorSeverity.HIGH;
    } else {
      errorSeverity = ErrorSeverity.MEDIUM;
    }
  } else if (request) {
    // 请求已发送但没有收到响应
    errorMessage = '网络请求无响应';
    errorDetails = {
      message: error.message,
      request: request,
      url: config?.url,
      method: config?.method
    };
    errorSeverity = ErrorSeverity.HIGH;
  } else {
    // 请求设置时发生错误
    errorMessage = '请求配置错误';
    errorDetails = {
      message: error.message,
      config: config
    };
    errorSeverity = ErrorSeverity.MEDIUM;
  }
  
  // 添加到错误管理器
  errorHandlerManager.addError({
    id: `API_ERROR_${Date.now()}`,
    type: ErrorType.NETWORK_DOWNLOAD,
    severity: errorSeverity,
    message: errorMessage,
    timestamp: new Date(),
    details: errorDetails,
    autoFixAvailable: false,
    userGuidance: {
      title: 'API请求失败',
      description: '与服务器通信时发生错误',
      steps: [
        '检查网络连接',
        '确认服务器是否在线',
        '重试操作',
        '如果问题持续，请联系技术支持'
      ],
      estimatedTime: '2-5分钟',
      difficulty: 'medium',
      requiredActions: ['检查网络', '重试操作'],
      precautionMeasures: ['确保网络稳定', '定期检查服务状态']
    }
  });
}

/**
 * 创建熔断器
 * 用于防止持续调用已知失败的服务
 */
export class CircuitBreaker {
  private failures: number = 0;
  private lastFailureTime: number = 0;
  private state: 'CLOSED' | 'OPEN' | 'HALF_OPEN' = 'CLOSED';
  
  constructor(
    private readonly failureThreshold: number = 5,
    private readonly resetTimeout: number = 30000,
    private readonly halfOpenMaxCalls: number = 3
  ) {}
  
  public async execute<T>(fn: () => Promise<T>): Promise<T> {
    // 检查熔断器状态
    if (this.state === 'OPEN') {
      // 检查是否应该进入半开状态
      if (Date.now() - this.lastFailureTime > this.resetTimeout) {
        this.state = 'HALF_OPEN';
      } else {
        throw new Error('服务暂时不可用，熔断器已打开');
      }
    }
    
    try {
      // 执行函数
      const result = await fn();
      
      // 成功执行，重置熔断器
      this.reset();
      
      return result;
    } catch (error) {
      // 记录失败
      this.recordFailure();
      
      // 重新抛出错误
      throw error;
    }
  }
  
  private reset(): void {
    this.failures = 0;
    this.state = 'CLOSED';
  }
  
  private recordFailure(): void {
    this.failures++;
    this.lastFailureTime = Date.now();
    
    // 检查是否应该打开熔断器
    if (this.state === 'CLOSED' && this.failures >= this.failureThreshold) {
      this.state = 'OPEN';
    } else if (this.state === 'HALF_OPEN') {
      this.state = 'OPEN';
    }
  }
  
  public getState(): string {
    return this.state;
  }
}