import React from 'react';

interface ErrorBoundaryProps {
  fallback?: (error: Error, info: React.ErrorInfo) => React.ReactNode;
  children?: React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
  info?: React.ErrorInfo;
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    // Log error to console or report system
    // Can connect to backend error collection API
    // eslint-disable-next-line no-console
    console.error('[ErrorBoundary]', error, info);
    this.setState({ error, info });
  }

  render(): React.ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback && this.state.error && this.state.info) {
        return this.props.fallback(this.state.error, this.state.info);
      }
      return (
        <div style={{ padding: 24 }}>
          <h2>Something went wrong</h2>
          <p>Sorry, the page failed to render. Try refreshing or go back.</p>
          <button onClick={() => window.location.reload()}>刷新页面</button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;