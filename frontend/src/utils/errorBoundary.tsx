import React, { Component, ErrorInfo, ReactNode } from 'react';
import { ErrorType, ErrorSeverity, errorHandlerManager } from './errorHandling';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode | ((error: Error, resetError: () => void) => ReactNode);
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  componentName?: string;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

/**
 * 全局错误边界组件
 * 用于捕获React组件树中的JavaScript错误，记录错误并显示备用UI
 */
class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null
    };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      error
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // 记录错误到错误处理系统
    errorHandlerManager.addError({
      id: `REACT_ERROR_${Date.now()}`,
      type: ErrorType.SERVICE_RUNTIME,
      severity: ErrorSeverity.HIGH,
      message: `React组件错误: ${this.props.componentName || 'Unknown'}`,
      timestamp: new Date(),
      details: { 
        error: error.message, 
        stack: error.stack,
        componentStack: errorInfo.componentStack,
        componentName: this.props.componentName
      },
      autoFixAvailable: false,
      userGuidance: {
        title: 'UI组件错误',
        description: '界面组件发生了错误，可能导致部分功能不可用',
        steps: [
          '刷新页面',
          '清除浏览器缓存',
          '如果问题持续，请联系技术支持'
        ],
        estimatedTime: '1-3分钟',
        difficulty: 'easy',
        requiredActions: ['刷新页面'],
        precautionMeasures: ['定期更新应用']
      }
    });

    // 调用自定义错误处理函数
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
  }

  resetError = (): void => {
    this.setState({
      hasError: false,
      error: null
    });
  };

  render(): ReactNode {
    const { hasError, error } = this.state;
    const { children, fallback } = this.props;

    if (hasError && error) {
      if (typeof fallback === 'function') {
        return fallback(error, this.resetError);
      }
      
      if (fallback) {
        return fallback;
      }

      // 默认错误UI
      return (
        <div className="error-boundary-fallback">
          <h2>组件发生错误</h2>
          <p>抱歉，界面组件发生了错误。</p>
          <details>
            <summary>查看错误详情</summary>
            <p>{error.message}</p>
            <pre>{error.stack}</pre>
          </details>
          <button onClick={this.resetError}>
            尝试恢复
          </button>
        </div>
      );
    }

    return children;
  }
}

export default ErrorBoundary;