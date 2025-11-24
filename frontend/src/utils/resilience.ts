import { defaultLogger } from './logger';
import { AppError, ErrorType, ErrorSeverity, errorHandler } from './errorHandler';

/**
 * 熔断器状态枚举
 */
export enum CircuitState {
  CLOSED = 'CLOSED',    // 正常状态，允许请求通过
  OPEN = 'OPEN',        // 熔断状态，拒绝所有请求
  HALF_OPEN = 'HALF_OPEN'  // 半开状态，允许部分请求通过以测试服务是否恢复
}

/**
 * 熔断器配置接口
 */
export interface CircuitBreakerConfig {
  failureThreshold: number;     // 触发熔断的失败次数阈值
  recoveryTimeout: number;      // 熔断恢复超时时间（毫秒）
  halfOpenMaxCalls: number;     // 半开状态下允许的最大请求数
}

/**
 * 重试配置接口
 */
export interface RetryConfig {
  maxRetries: number;           // 最大重试次数
  initialBackoff: number;       // 初始退避时间（毫秒）
  maxBackoff: number;           // 最大退避时间（毫秒）
  backoffMultiplier: number;    // 退避时间乘数
  jitter: boolean;              // 是否添加随机抖动
}

/**
 * 熔断器类
 * 用于防止对失败服务的持续请求，避免级联故障
 */
export class CircuitBreaker {
  private name: string;
  private config: CircuitBreakerConfig;
  private state: CircuitState = CircuitState.CLOSED;
  private failureCount: number = 0;
  private lastFailureTime: number = 0;
  private halfOpenCalls: number = 0;

  constructor(name: string, config?: Partial<CircuitBreakerConfig>) {
    this.name = name;
    this.config = {
      failureThreshold: config?.failureThreshold || 5,
      recoveryTimeout: config?.recoveryTimeout || 30000,
      halfOpenMaxCalls: config?.halfOpenMaxCalls || 3
    };
  }

  /**
   * 检查是否允许请求通过
   */
  public allowRequest(): boolean {
    const currentTime = Date.now();

    if (this.state === CircuitState.OPEN) {
      // 检查是否达到恢复超时时间
      if (currentTime - this.lastFailureTime >= this.config.recoveryTimeout) {
        defaultLogger.info(`熔断器 ${this.name} 进入半开状态`);
        this.state = CircuitState.HALF_OPEN;
        this.halfOpenCalls = 0;
        return true;
      }
      return false;
    }

    if (this.state === CircuitState.HALF_OPEN) {
      // 在半开状态下限制请求数量
      if (this.halfOpenCalls < this.config.halfOpenMaxCalls) {
        this.halfOpenCalls += 1;
        return true;
      }
      return false;
    }

    // 闭合状态，允许所有请求
    return true;
  }

  /**
   * 记录成功请求
   */
  public recordSuccess(): void {
    if (this.state === CircuitState.HALF_OPEN) {
      defaultLogger.info(`熔断器 ${this.name} 恢复正常，关闭熔断`);
      this.state = CircuitState.CLOSED;
      this.failureCount = 0;
      this.halfOpenCalls = 0;
    }
  }

  /**
   * 记录失败请求
   */
  public recordFailure(): void {
    const currentTime = Date.now();
    this.lastFailureTime = currentTime;

    if (this.state === CircuitState.HALF_OPEN) {
      defaultLogger.warn(`熔断器 ${this.name} 在半开状态下检测到失败，重新打开熔断`);
      this.state = CircuitState.OPEN;
      return;
    }

    this.failureCount += 1;

    if (this.state === CircuitState.CLOSED && this.failureCount >= this.config.failureThreshold) {
      defaultLogger.warn(`熔断器 ${this.name} 失败次数达到阈值 ${this.config.failureThreshold}，打开熔断`);
      this.state = CircuitState.OPEN;
    }
  }

  /**
   * 获取当前熔断器状态
   */
  public getState(): CircuitState {
    return this.state;
  }

  /**
   * 重置熔断器状态
   */
  public reset(): void {
    this.state = CircuitState.CLOSED;
    this.failureCount = 0;
    this.lastFailureTime = 0;
    this.halfOpenCalls = 0;
  }
}

/**
 * 熔断器注册表
 * 用于管理和复用熔断器实例
 */
class CircuitBreakerRegistry {
  private static instance: CircuitBreakerRegistry;
  private breakers: Map<string, CircuitBreaker> = new Map();

  private constructor() {}

  public static getInstance(): CircuitBreakerRegistry {
    if (!CircuitBreakerRegistry.instance) {
      CircuitBreakerRegistry.instance = new CircuitBreakerRegistry();
    }
    return CircuitBreakerRegistry.instance;
  }

  /**
   * 获取或创建熔断器
   */
  public getBreaker(name: string, config?: Partial<CircuitBreakerConfig>): CircuitBreaker {
    if (!this.breakers.has(name)) {
      this.breakers.set(name, new CircuitBreaker(name, config));
    }
    return this.breakers.get(name)!;
  }

  /**
   * 重置所有熔断器
   */
  public resetAll(): void {
    this.breakers.forEach(breaker => breaker.reset());
  }
}

// 导出熔断器注册表单例
export const circuitBreakerRegistry = CircuitBreakerRegistry.getInstance();

/**
 * 重试工具类
 * 提供自动重试功能，支持指数退避策略
 */
export class RetryUtil {
  /**
   * 使用重试策略执行异步函数
   * @param fn 要执行的异步函数
   * @param config 重试配置
   * @param breakerName 熔断器名称（可选）
   * @returns 函数执行结果的Promise
   */
  public static async withRetry<T>(
    fn: () => Promise<T>,
    config?: Partial<RetryConfig>,
    breakerName?: string
  ): Promise<T> {
    const retryConfig: RetryConfig = {
      maxRetries: config?.maxRetries || 3,
      initialBackoff: config?.initialBackoff || 1000,
      maxBackoff: config?.maxBackoff || 30000,
      backoffMultiplier: config?.backoffMultiplier || 2,
      jitter: config?.jitter !== undefined ? config.jitter : true
    };

    let breaker: CircuitBreaker | undefined;
    if (breakerName) {
      breaker = circuitBreakerRegistry.getBreaker(breakerName);
    }

    let retryCount = 0;
    let lastError: any;

    while (retryCount <= retryConfig.maxRetries) {
      try {
        // 检查熔断器状态
        if (breaker && !breaker.allowRequest()) {
          const error = new AppError(
            `服务 ${breakerName} 暂时不可用`,
            ErrorType.NETWORK,
            ErrorSeverity.ERROR,
            { breakerName },
            '服务暂时不可用，请稍后重试'
          );
          errorHandler.handleError(error);
          throw error;
        }

        // 执行函数
        const result = await fn();

        // 记录成功
        if (breaker) {
          breaker.recordSuccess();
        }

        return result;
      } catch (error) {
        lastError = error;
        retryCount += 1;

        // 记录失败
        if (breaker) {
          breaker.recordFailure();
        }

        // 达到最大重试次数，抛出异常
        if (retryCount > retryConfig.maxRetries) {
          defaultLogger.error(`重试次数达到上限 ${retryConfig.maxRetries}，放弃重试`, {
            function: fn.name || 'anonymous',
            error,
            retryCount
          });
          throw error;
        }

        // 计算退避时间
        let backoff = Math.min(
          retryConfig.initialBackoff * Math.pow(retryConfig.backoffMultiplier, retryCount - 1),
          retryConfig.maxBackoff
        );

        // 添加抖动以避免惊群效应
        if (retryConfig.jitter) {
          backoff = backoff * (0.5 + Math.random());
        }

        defaultLogger.warn(`操作失败，将在 ${(backoff / 1000).toFixed(2)} 秒后重试 (${retryCount}/${retryConfig.maxRetries})`, {
          function: fn.name || 'anonymous',
          error,
          retryCount,
          backoff
        });

        // 等待退避时间
        await new Promise(resolve => setTimeout(resolve, backoff));
      }
    }

    // 不应该到达这里，但为了类型安全添加
    throw lastError;
  }

  /**
   * 创建带有重试功能的API调用包装器
   * @param apiClient API客户端对象
   * @param config 重试配置
   * @returns 包装后的API客户端
   */
  public static createRetryableApiClient<T extends object>(
    apiClient: T,
    config?: Partial<RetryConfig>
  ): T {
    const retryableClient = {} as T;

    // 遍历原始客户端的所有方法
    for (const key of Object.getOwnPropertyNames(Object.getPrototypeOf(apiClient))) {
      const prop = (apiClient as any)[key];

      // 只包装函数
      if (typeof prop === 'function' && key !== 'constructor') {
        (retryableClient as any)[key] = async (...args: any[]) => {
          return RetryUtil.withRetry(
            () => prop.apply(apiClient, args),
            config,
            `api.${key}`
          );
        };
      } else {
        (retryableClient as any)[key] = prop;
      }
    }

    return retryableClient;
  }
}

/**
 * 超时控制工具类
 */
export class TimeoutUtil {
  /**
   * 使用超时控制执行异步函数
   * @param fn 要执行的异步函数
   * @param timeoutMs 超时时间（毫秒）
   * @returns 函数执行结果的Promise
   */
  public static async withTimeout<T>(fn: () => Promise<T>, timeoutMs: number): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      // 创建超时计时器
      const timeoutId = setTimeout(() => {
        const error = new AppError(
          `操作超时 (${timeoutMs}毫秒)`,
          ErrorType.NETWORK,
          ErrorSeverity.ERROR,
          { timeoutMs },
          '操作超时，请稍后重试'
        );
        reject(error);
      }, timeoutMs);

      // 执行函数
      fn()
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
}

/**
 * 限流工具类
 * 用于控制并发请求数量
 */
export class RateLimiter {
  private maxConcurrent: number;
  private currentConcurrent: number = 0;
  private queue: Array<() => void> = [];

  constructor(maxConcurrent: number = 5) {
    this.maxConcurrent = maxConcurrent;
  }

  /**
   * 使用限流控制执行异步函数
   * @param fn 要执行的异步函数
   * @returns 函数执行结果的Promise
   */
  public async execute<T>(fn: () => Promise<T>): Promise<T> {
    // 等待获取执行权
    await this.acquireSlot();

    try {
      // 执行函数
      return await fn();
    } finally {
      // 释放执行权
      this.releaseSlot();
    }
  }

  /**
   * 获取执行槽位
   */
  private async acquireSlot(): Promise<void> {
    if (this.currentConcurrent < this.maxConcurrent) {
      this.currentConcurrent += 1;
      return;
    }

    // 如果已达到最大并发数，则等待
    return new Promise<void>(resolve => {
      this.queue.push(resolve);
    });
  }

  /**
   * 释放执行槽位
   */
  private releaseSlot(): void {
    if (this.queue.length > 0) {
      // 如果队列中有等待的任务，则唤醒一个
      const next = this.queue.shift();
      if (next) {
        next();
      }
    } else {
      // 否则减少当前并发数
      this.currentConcurrent -= 1;
    }
  }
}