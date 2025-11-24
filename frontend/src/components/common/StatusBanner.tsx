import React from 'react';

export type UIStatus = 'idle' | 'loading' | 'error' | 'empty' | 'success';

export interface StatusBannerProps {
  status: UIStatus;
  title?: string;
  message?: string;
  onRetry?: () => void;
}

export function StatusBanner({ status, title, message, onRetry }: StatusBannerProps) {
  const baseStyle: React.CSSProperties = {
    padding: '12px 16px',
    borderRadius: 8,
    margin: '8px 0',
    display: 'flex',
    alignItems: 'center',
    gap: 8
  };

  if (status === 'loading') {
    return (
      <div style={{ ...baseStyle, background: '#f5f5f5', color: '#555' }}>
        <span className="spinner" aria-label="loading" />
        <div>
          <div>{title || '正在加载数据…'}</div>
          <div style={{ fontSize: 12, opacity: 0.8 }}>{message || '请稍候，我们正在准备内容'}</div>
        </div>
      </div>
    );
  }

  if (status === 'error') {
    return (
      <div style={{ ...baseStyle, background: '#fff1f0', color: '#a8071a' }}>
        <strong>{title || '加载失败'}</strong>
        <span>{message || '请检查网络或稍后重试'}</span>
        {onRetry && (
          <button onClick={onRetry} style={{ marginLeft: 'auto' }}>重试</button>
        )}
      </div>
    );
  }

  if (status === 'empty') {
    return (
      <div style={{ ...baseStyle, background: '#fafafa', color: '#666' }}>
        <strong>{title || '暂无数据'}</strong>
        <span>{message || '尝试调整筛选条件或探索推荐内容'}</span>
      </div>
    );
  }

  if (status === 'success') {
    return (
      <div style={{ ...baseStyle, background: '#f6ffed', color: '#135200' }}>
        <strong>{title || '操作成功'}</strong>
        <span>{message || '已完成，感谢你的耐心'}</span>
      </div>
    );
  }

  return null;
}

export default StatusBanner;