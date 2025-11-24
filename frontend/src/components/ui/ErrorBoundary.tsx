import React from 'react';
import { Button } from 'antd';

interface State {
  hasError: boolean;
  error?: Error | null;
}

class ErrorBoundary extends React.Component<React.PropsWithChildren<{}>, State> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: any) {
    // TODO: send to remote logging
    // console.error(error, info);
  }

  reset = () => this.setState({ hasError: false, error: null });

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 32 }} role="alert">
          <h2>出现错误</h2>
          <div style={{ marginBottom: 16 }}>{this.state.error?.message}</div>
          <div style={{ display: 'flex', gap: 8 }}>
            <Button onClick={this.reset}>重试</Button>
            <Button onClick={() => navigator.clipboard?.writeText(String(this.state.error))}>复制诊断信息</Button>
          </div>
        </div>
      );
    }
    return this.props.children as React.ReactElement;
  }
}

export default ErrorBoundary;
