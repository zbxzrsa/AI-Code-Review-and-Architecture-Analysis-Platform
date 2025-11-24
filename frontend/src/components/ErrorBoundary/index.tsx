import React, { Component } from 'react';
import { AppError, ErrorBoundaryProps, ErrorBoundaryState, errorHandler, ErrorType, ErrorSeverity } from '../../utils/errorHandler';
import { defaultLogger } from '../../utils/logger';

/**
 * 错误边界组件
 * 捕获子组件树中的 JavaScript 错误，记录错误并显示备用 UI
 */
class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    // 将错误转换为 AppError
    const appError = error instanceof AppError 
      ? error 
      : new AppError(
          error.message,
          ErrorType.UNKNOWN,
          ErrorSeverity.ERROR,
          { originalError: error, stack: error.stack },
          '组件渲染过程中发生错误'
        );
    
    return { error: appError };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    // 记录错误信息
    const appError = this.state.error || 
      new AppError(
        error.message,
        ErrorType.UNKNOWN,
        ErrorSeverity.ERROR,
        { originalError: error, componentStack: errorInfo.componentStack },
        '组件渲染过程中发生错误'
      );
    
    // 记录到错误处理系统
    errorHandler.handleError(appError);
    
    // 调用自定义错误处理函数
    if (this.props.onError) {
      this.props.onError(appError);
    }
  }

  resetErrorBoundary = (): void => {
    this.setState({ error: null });
  };

  render(): React.ReactNode {
    if (this.state.error) {
      // 如果提供了自定义的错误UI，则使用它
      if (this.props.fallback) {
        if (typeof this.props.fallback === 'function') {
          return this.props.fallback(this.state.error, this.resetErrorBoundary);
        }
        return this.props.fallback;
      }

      // 默认错误UI
      return (
        <div className="error-boundary-fallback">
          <h2>出现了一些问题</h2>
          <p>{this.state.error.userMessage}</p>
          <button onClick={this.resetErrorBoundary}>
            重试
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;

/**
 * 错误边界HOC
 * 用于包装组件，提供错误边界功能
 */
export function withErrorBoundary<P>(
  Component: React.ComponentType<P>,
  errorBoundaryProps: Omit<ErrorBoundaryProps, 'children'> = {}
): React.ComponentType<P> {
  const displayName = Component.displayName || Component.name || 'Component';
  
  const WrappedComponent = (props: P): JSX.Element => (
    <ErrorBoundary {...errorBoundaryProps}>
      {(Component as any)({ ...(props as any) })}
    </ErrorBoundary>
  );
  
  WrappedComponent.displayName = `withErrorBoundary(${displayName})`;
  
  return WrappedComponent;
}

/**
 * 使用错误边界的Hook
 * 用于函数组件中手动触发错误处理
 */
export function useErrorHandler(): (error: Error | unknown) => void {
  return (error: Error | unknown): void => {
    const appError = error instanceof AppError 
      ? error 
      : new AppError(
          error instanceof Error ? error.message : String(error),
          ErrorType.UNKNOWN,
          ErrorSeverity.ERROR,
          { originalError: error },
          '操作过程中发生错误'
        );
    
    errorHandler.handleError(appError);
    throw appError; // 抛出错误，触发最近的错误边界
  };
}
