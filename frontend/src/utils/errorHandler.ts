import { defaultLogger } from './logger';

/**
 * 错误类型枚举
 */
export enum ErrorType {
  // 网络错误
  NETWORK = 'NETWORK',
  // API错误
  API = 'API',
  // 用户输入错误
  INPUT = 'INPUT',
  // 权限错误
  PERMISSION = 'PERMISSION',
  // 资源错误
  RESOURCE = 'RESOURCE',
  // 未知错误
  UNKNOWN = 'UNKNOWN'
}

/**
 * 错误严重程度枚举
 */
export enum ErrorSeverity {
  // 致命错误 - 阻止应用继续运行
  FATAL = 'FATAL',
  // 错误 - 功能无法完成但应用可继续
  ERROR = 'ERROR',
  // 警告 - 可能影响用户体验但功能仍可用
  WARNING = 'WARNING',
  // 信息 - 仅供记录，不影响功能
  INFO = 'INFO'
}

/**
 * 应用错误类
 */
export class AppError extends Error {
  type: ErrorType;
  severity: ErrorSeverity;
  timestamp: Date;
  details: any;
  handled: boolean;
  recoverable: boolean;
  userMessage: string;

  constructor(
    message: string,
    type: ErrorType = ErrorType.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    details: any = {},
    userMessage: string = 'An error occurred. Please try again later.'
  ) {
    super(message);
    this.name = 'AppError';
    this.type = type;
    this.severity = severity;
    this.timestamp = new Date();
    this.details = details;
    this.handled = false;
    this.recoverable = severity !== ErrorSeverity.FATAL;
    this.userMessage = userMessage;
  }
}

/**
 * 错误处理服务
 */
export class ErrorHandler {
  private static instance: ErrorHandler;
  private errorListeners: Array<(error: AppError) => void> = [];
  private retryStrategies: Map<string, (error: AppError, retryCount: number) => Promise<any>> = new Map();
  private maxRetries: number = 3;

  private constructor() {
    // 设置全局未捕获异常处理
    this.setupGlobalHandlers();
  }

  /**
   * 获取单例实例
   */
  public static getInstance(): ErrorHandler {
    if (!ErrorHandler.instance) {
      ErrorHandler.instance = new ErrorHandler();
    }
    return ErrorHandler.instance;
  }

  /**
   * 设置全局错误处理器
   */
  private setupGlobalHandlers(): void {
    // 处理未捕获的Promise异常
    window.addEventListener('unhandledrejection', (event) => {
      const error = this.normalizeError(event.reason);
      this.handleError(error);
      // 防止错误冒泡
      event.preventDefault();
    });

    // 处理未捕获的JS异常
    window.addEventListener('error', (event) => {
      // 忽略资源加载错误，这些会由资源错误处理器处理
      if (event.error) {
        const error = this.normalizeError(event.error);
        this.handleError(error);
        // 防止错误冒泡
        event.preventDefault();
      }
    });

    // 处理资源加载错误
    document.addEventListener('error', (event: Event) => {
      const target = event.target as HTMLElement;
      // 检查是否为资源加载错误
      if (target && (target instanceof HTMLImageElement || 
                     target instanceof HTMLScriptElement || 
                     target instanceof HTMLLinkElement)) {
        let resourceUrl = '';
        if (target instanceof HTMLImageElement || target instanceof HTMLScriptElement) {
          resourceUrl = (target as any).src || '';
        } else if (target instanceof HTMLLinkElement) {
          resourceUrl = target.href || '';
        }
        
        const error = new AppError(
          `资源加载失败: ${resourceUrl}`,
          ErrorType.RESOURCE,
          ErrorSeverity.WARNING,
          { resourceUrl, tagName: target.tagName },
          '部分资源加载失败，可能影响显示效果'
        );
        
        this.handleError(error);
      }
    }, true); // 使用捕获阶段
  }

  /**
   * 将任何错误转换为AppError
   */
  private normalizeError(error: any): AppError {
    if (error instanceof AppError) {
      return error;
    }

    // 处理网络错误
    if (error instanceof TypeError && error.message.includes('Network')) {
      return new AppError(
        error.message,
        ErrorType.NETWORK,
        ErrorSeverity.ERROR,
        { originalError: error },
        'Network error. Please check your connection.'
      );
    }

    // 处理API错误
    if (error && error.status && error.statusText) {
      return new AppError(
        `API error: ${error.status} ${error.statusText}`,
        ErrorType.API,
        ErrorSeverity.ERROR,
        { originalError: error, status: error.status },
        this.getApiErrorUserMessage(error.status)
      );
    }

    // 处理一般错误
    return new AppError(
      error?.message || 'Unknown error',
      ErrorType.UNKNOWN,
      ErrorSeverity.ERROR,
      { originalError: error },
      'An error occurred. Please try again later.'
    );
  }

  /**
   * 根据API状态码获取用户友好的错误消息
   */
  private getApiErrorUserMessage(status: number): string {
    switch (status) {
      case 400:
        return 'Invalid request parameters. Please check your input.';
      case 401:
        return 'Your session has expired. Please log in again.';
      case 403:
        return 'You do not have permission to perform this action.';
      case 404:
        return 'Requested resource was not found.';
      case 500:
        return 'Internal server error. Please try again later.';
      case 503:
        return 'Service temporarily unavailable. Please try again later.';
      default:
        return 'Service request failed. Please try again later.';
    }
  }

  /**
   * 处理错误
   */
  public handleError(error: AppError): void {
    // 标记错误已处理
    error.handled = true;

    // 记录错误
    this.logError(error);

    // 通知所有监听器
    this.notifyListeners(error);

    // 尝试恢复
    if (error.recoverable) {
      this.attemptRecovery(error);
    }
  }

  /**
   * 记录错误
   */
  private logError(error: AppError): void {
    const logData = {
      message: error.message,
      type: error.type,
      severity: error.severity,
      timestamp: error.timestamp,
      details: error.details,
      stack: error.stack
    };

    switch (error.severity) {
      case ErrorSeverity.FATAL:
      case ErrorSeverity.ERROR:
        defaultLogger.error('Application error', logData);
        break;
      case ErrorSeverity.WARNING:
        defaultLogger.warn('Application warning', logData);
        break;
      case ErrorSeverity.INFO:
        defaultLogger.info('Application info', logData);
        break;
    }
  }

  /**
   * 通知所有错误监听器
   */
  private notifyListeners(error: AppError): void {
    this.errorListeners.forEach(listener => {
      try {
        listener(error);
      } catch (listenerError) {
        defaultLogger.error('Error listener execution failed', { 
          listenerError, 
          originalError: error 
        });
      }
    });
  }

  /**
   * 尝试恢复错误
   */
  private async attemptRecovery(error: AppError): Promise<void> {
    // 查找对应错误类型的重试策略
    const retryStrategy = this.retryStrategies.get(error.type);
    
    if (retryStrategy) {
      let retryCount = 0;
      let success = false;

      while (retryCount < this.maxRetries && !success) {
        try {
          await retryStrategy(error, retryCount);
          success = true;
          defaultLogger.info('Error recovery succeeded', { 
            errorType: error.type, 
            retryCount 
          });
        } catch (retryError) {
          retryCount++;
          defaultLogger.warn('Error recovery attempt failed', { 
            errorType: error.type, 
            retryCount, 
            retryError 
          });
          
          // 指数退避策略
          const delay = Math.pow(2, retryCount) * 1000 + Math.random() * 1000;
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }
  }

  /**
   * 添加错误监听器
   */
  public addErrorListener(listener: (error: AppError) => void): () => void {
    this.errorListeners.push(listener);
    
    // 返回移除监听器的函数
    return () => {
      this.errorListeners = this.errorListeners.filter(l => l !== listener);
    };
  }

  /**
   * 注册重试策略
   */
  public registerRetryStrategy(
    errorType: ErrorType,
    strategy: (error: AppError, retryCount: number) => Promise<any>
  ): void {
    this.retryStrategies.set(errorType, strategy);
  }

  /**
   * 设置最大重试次数
   */
  public setMaxRetries(maxRetries: number): void {
    this.maxRetries = maxRetries;
  }

  /**
   * 包装异步函数以处理错误
   */
  public async wrapAsync<T>(
    fn: () => Promise<T>,
    errorType: ErrorType = ErrorType.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    userMessage: string = 'An error occurred. Please try again later.'
  ): Promise<T> {
    try {
      return await fn();
    } catch (error) {
      const appError = new AppError(
        error instanceof Error ? error.message : String(error),
        errorType,
        severity,
        { originalError: error },
        userMessage
      );
      
      this.handleError(appError);
      throw appError;
    }
  }
}

// 导出单例实例
export const errorHandler = ErrorHandler.getInstance();

/**
 * 错误边界组件的Props接口
 */
export interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode | ((error: AppError, reset: () => void) => React.ReactNode);
  onError?: (error: AppError) => void;
}

/**
 * 错误边界组件的State接口
 */
export interface ErrorBoundaryState {
  error: AppError | null;
}
