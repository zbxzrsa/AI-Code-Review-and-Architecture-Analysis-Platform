/**
 * 重试工具类
 * 提供通用的重试机制，用于处理可能失败的操作
 */

// 重试配置接口
export interface RetryOptions {
  maxAttempts: number;      // 最大尝试次数
  initialDelay: number;     // 初始延迟时间(毫秒)
  maxDelay?: number;        // 最大延迟时间(毫秒)
  backoffFactor: number;    // 退避因子(每次重试延迟时间的增长倍数)
  retryCondition?: (error: any) => boolean;  // 重试条件函数
  onRetry?: (attempt: number, error: any) => void;  // 重试回调函数
}

// 默认重试配置
const DEFAULT_RETRY_OPTIONS: RetryOptions = {
  maxAttempts: 3,
  initialDelay: 1000,
  maxDelay: 10000,
  backoffFactor: 2,
  retryCondition: () => true,
  onRetry: undefined
};

/**
 * 带重试的异步函数执行器
 * @param fn 要执行的异步函数
 * @param options 重试配置
 * @returns Promise<T> 函数执行结果
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  options: Partial<RetryOptions> = {}
): Promise<T> {
  // 合并配置
  const config: RetryOptions = {
    ...DEFAULT_RETRY_OPTIONS,
    ...options
  };
  
  let attempt = 0;
  let lastError: any;
  let delay = config.initialDelay;

  while (attempt < config.maxAttempts) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      attempt++;
      
      // 检查是否应该重试
      if (attempt >= config.maxAttempts || (config.retryCondition && !config.retryCondition(error))) {
        break;
      }
      
      // 计算下一次重试的延迟时间
      delay = Math.min(delay * config.backoffFactor, config.maxDelay || Number.MAX_SAFE_INTEGER);
      
      // 调用重试回调
      if (config.onRetry) {
        config.onRetry(attempt, error);
      }
      
      // 等待后重试
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  
  // 所有重试都失败，抛出最后一个错误
  throw lastError;
}

/**
 * 创建带重试功能的函数
 * @param fn 原始函数
 * @param options 重试配置
 * @returns 带重试功能的函数
 */
export function createRetryableFunction<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  options: Partial<RetryOptions> = {}
): T {
  return (async (...args: Parameters<T>): Promise<ReturnType<T>> => {
    return withRetry(() => fn(...args), options);
  }) as T;
}

/**
 * 重试装饰器(用于类方法)
 * @param options 重试配置
 */
export function Retryable(options: Partial<RetryOptions> = {}) {
  return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;
    
    descriptor.value = async function (...args: any[]) {
      return withRetry(() => originalMethod.apply(this, args), options);
    };
    
    return descriptor;
  };
}

/**
 * 限流器类
 * 用于限制并发请求数量
 */
export class RateLimiter {
  private queue: Array<() => void> = [];
  private runningCount = 0;
  
  constructor(private maxConcurrent: number) {}
  
  /**
   * 执行受限流控制的异步函数
   * @param fn 要执行的异步函数
   * @returns Promise<T> 函数执行结果
   */
  async execute<T>(fn: () => Promise<T>): Promise<T> {
    // 如果当前运行的请求数小于最大并发数，直接执行
    if (this.runningCount < this.maxConcurrent) {
      return this.runTask(fn);
    }
    
    // 否则加入队列等待执行
    return new Promise<T>((resolve, reject) => {
      this.queue.push(() => {
        this.runTask(fn).then(resolve).catch(reject);
      });
    });
  }
  
  private async runTask<T>(fn: () => Promise<T>): Promise<T> {
    this.runningCount++;
    
    try {
      return await fn();
    } finally {
      this.runningCount--;
      this.processQueue();
    }
  }
  
  private processQueue(): void {
    if (this.queue.length > 0 && this.runningCount < this.maxConcurrent) {
      const nextTask = this.queue.shift();
      if (nextTask) {
        nextTask();
      }
    }
  }
}

/**
 * 超时控制函数
 * @param promise 原始Promise
 * @param timeoutMs 超时时间(毫秒)
 * @param errorMessage 超时错误消息
 * @returns Promise<T> 带超时控制的Promise
 */
export function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  errorMessage = '操作超时'
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      reject(new Error(errorMessage));
    }, timeoutMs);
    
    promise
      .then(result => {
        clearTimeout(timeoutId);
        resolve(result);
      })
      .catch(error => {
        clearTimeout(timeoutId);
        reject(error);
      });
  });
}